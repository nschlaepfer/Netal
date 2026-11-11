#!/usr/bin/env python3
# SPDX-License-Identifier: MIT AND Apache-2.0
"""Parity and GPU latency comparison between Netal's native Metal NR runtime and MLX-DLSS.

Compares `netal-nr-fast-cli` against `mlxdlss` on:
1. Direct `[1, H, W, 16] -> [1, H, W, 4]` transformer tensor evaluation (`eval-tensor` vs `mlxdlss run`),
   plus optional native FP8 (`mlxdlss-native-e4m3-v1`) bit-exactness verification against `.dlssmodel`.
2. End-to-end `render-image` (`standard` and `cinematic` detail-split upsampling profiles), verifying
   RGB MAE < 0.005, RMSE, PSNR (dB), and max absolute difference.
3. Optional multi-resolution GPU latency sweep (`960x544`, `1280x704`, `1920x1088`) on Apple Silicon.

Example:
  python3 scripts/compare-nr-native-vs-mlx.py \\
    --mlx-bin /Users/example/bin/mlxdlss \\
    --model /Users/example/models/NeuralRendering.dlssmodel \\
    --fp8-model /Users/example/models/dlssnr-weights-native-fp8.safetensors \\
    --bench-resolutions
"""

from __future__ import annotations

import argparse
import array
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import zlib

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = REPO_ROOT / "scripts" / "build-netal-nr-fast.py"


def _write_rgb8_png(path: Path, width: int, height: int, rgb_bytes: bytes) -> None:
    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    raw = bytearray()
    stride = width * 3
    for y in range(height):
        raw.append(0)
        raw.extend(rgb_bytes[y * stride : (y + 1) * stride])
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"sRGB", b"\x00")
        + chunk(b"IDAT", zlib.compress(bytes(raw), 6))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def _read_png_rgb_float(path: Path) -> tuple[int, int, list[float]]:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"invalid PNG signature in {path}")
    pos = 8
    width = height = bit_depth = color_type = 0
    idat = bytearray()
    while pos < len(data):
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        tag = data[pos + 4 : pos + 8]
        body = data[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if tag == b"IHDR":
            width, height, bit_depth, color_type, _, _, _ = struct.unpack(">IIBBBBB", body)
        elif tag == b"IDAT":
            idat.extend(body)
        elif tag == b"IEND":
            break
    if bit_depth not in (8, 16) or color_type not in (2, 6):
        raise ValueError(f"unsupported PNG format bit_depth={bit_depth} color_type={color_type}")
    channels = 3 if color_type == 2 else 4
    bytes_per_sample = bit_depth // 8
    bpp = channels * bytes_per_sample
    stride = width * bpp
    raw = zlib.decompress(bytes(idat))
    prev = bytearray(stride)
    out = [0.0] * (width * height * 3)
    src = 0
    for y in range(height):
        flt = raw[src]
        src += 1
        cur = bytearray(raw[src : src + stride])
        src += stride
        if flt == 1:
            for i in range(bpp, stride):
                cur[i] = (cur[i] + cur[i - bpp]) & 0xFF
        elif flt == 2:
            for i in range(stride):
                cur[i] = (cur[i] + prev[i]) & 0xFF
        elif flt == 3:
            for i in range(stride):
                left = cur[i - bpp] if i >= bpp else 0
                cur[i] = (cur[i] + ((left + prev[i]) >> 1)) & 0xFF
        elif flt == 4:
            for i in range(stride):
                a = cur[i - bpp] if i >= bpp else 0
                b = prev[i]
                c = prev[i - bpp] if i >= bpp else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                cur[i] = (cur[i] + pr) & 0xFF
        prev = cur
        row_base = y * width * 3
        if bit_depth == 8:
            for x in range(width):
                out[row_base + x * 3 + 0] = cur[x * bpp + 0] / 255.0
                out[row_base + x * 3 + 1] = cur[x * bpp + 1] / 255.0
                out[row_base + x * 3 + 2] = cur[x * bpp + 2] / 255.0
        else:
            for x in range(width):
                base = x * bpp
                r = (cur[base] << 8) | cur[base + 1]
                g = (cur[base + 2] << 8) | cur[base + 3]
                b = (cur[base + 4] << 8) | cur[base + 5]
                out[row_base + x * 3 + 0] = r / 65535.0
                out[row_base + x * 3 + 1] = g / 65535.0
                out[row_base + x * 3 + 2] = b / 65535.0
    return width, height, out


def _metrics(a: list[float] | array.array[float], b: list[float] | array.array[float]) -> dict[str, float]:
    n = len(a)
    sum_abs = 0.0
    sum_sq = 0.0
    max_abs = 0.0
    for x, y in zip(a, b):
        d = abs(x - y)
        sum_abs += d
        sum_sq += d * d
        if d > max_abs:
            max_abs = d
    mae = sum_abs / n
    rmse = math.sqrt(sum_sq / n)
    psnr = 99.0 if rmse <= 1e-12 else (20.0 * math.log10(1.0 / rmse))
    return {"mae": mae, "rmse": rmse, "psnr_db": psnr, "max_abs_diff": max_abs}


def _make_procedural_features(width: int, height: int) -> bytes:
    arr = array.array("f", [0.0] * (width * height * 16))
    for y in range(height):
        fy = y / max(1, height - 1)
        for x in range(width):
            fx = x / max(1, width - 1)
            base = (y * width + x) * 16
            r = min(1.0, max(0.0, 0.15 + 0.55 * fx + 0.15 * math.sin(fx * 18.0)))
            g = min(1.0, max(0.0, 0.10 + 0.60 * fy + 0.15 * math.cos(fy * 14.0)))
            b = min(1.0, max(0.0, 0.25 + 0.35 * (1.0 - fx) + (0.20 if ((x // 12 + y // 12) % 2 == 0) else 0.0)))
            arr[base + 0] = math.sin((x * 13 + y * 7) * 0.37) * 0.8
            arr[base + 1] = math.cos((x * 11 + y * 17) * 0.29) * 0.8
            arr[base + 2] = math.sin((x * 5 + y * 23) * 0.41) * 0.8
            arr[base + 3] = 1.0
            arr[base + 4] = r
            arr[base + 5] = g
            arr[base + 6] = b
            arr[base + 7] = r
            arr[base + 8] = g
            arr[base + 9] = b
            arr[base + 10] = 0.0
            arr[base + 11] = 0.0
            arr[base + 12] = 0.0
            arr[base + 13] = 1.0
            arr[base + 14] = 1.0
            arr[base + 15] = 0.0
    if sys.byteorder != "little":
        arr.byteswap()
    return arr.tobytes()


def _make_procedural_png(path: Path, width: int, height: int) -> None:
    rgb = bytearray(width * height * 3)
    for y in range(height):
        fy = y / max(1, height - 1)
        for x in range(width):
            fx = x / max(1, width - 1)
            checker = 0.18 if (((x // 16) + (y // 16)) % 2 == 0) else -0.10
            ring = 0.12 * math.sin(math.hypot(fx - 0.5, fy - 0.5) * 42.0)
            r = min(1.0, max(0.0, 0.28 + 0.45 * fx + checker + ring))
            g = min(1.0, max(0.0, 0.24 + 0.48 * fy + 0.5 * checker - ring))
            b = min(1.0, max(0.0, 0.32 + 0.35 * (1.0 - fx * 0.6 - fy * 0.4) + ring))
            idx = (y * width + x) * 3
            rgb[idx + 0] = int(round(r * 255.0))
            rgb[idx + 1] = int(round(g * 255.0))
            rgb[idx + 2] = int(round(b * 255.0))
    _write_rgb8_png(path, width, height, bytes(rgb))


def _load_f32(path: Path) -> array.array[float]:
    a = array.array("f")
    a.frombytes(path.read_bytes())
    if sys.byteorder != "little":
        a.byteswap()
    return a


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-bin", type=Path, required=True, help="Path to compiled mlxdlss CLI binary")
    parser.add_argument("--model", type=Path, required=True, help="Path to .dlssmodel directory or .safetensors")
    parser.add_argument("--fp8-model", type=Path, default=None, help="Optional native FP8 .safetensors to verify bit-exactness")
    parser.add_argument("--netal-bin", type=Path, default=None, help="Optional pre-built netal-nr-fast-cli binary")
    parser.add_argument("--mae-limit", type=float, default=0.005, help="Maximum allowed rendered RGB MAE vs MLX (default: 0.005)")
    parser.add_argument("--bench-resolutions", action="store_true", help="Run multi-resolution GPU latency sweep")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional path to write full JSON report")
    args = parser.parse_args()

    mlx_bin = args.mlx_bin.resolve()
    model_path = args.model.resolve()
    fp8_path = args.fp8_model.resolve() if args.fp8_model else None

    if not mlx_bin.is_file():
        raise SystemExit(f"mlxdlss binary not found: {mlx_bin}")
    if not model_path.exists():
        raise SystemExit(f"model path not found: {model_path}")
    if fp8_path is not None and not fp8_path.is_file():
        raise SystemExit(f"fp8 model path not found: {fp8_path}")

    with tempfile.TemporaryDirectory(prefix="netal-vs-mlx-") as tmp:
        work = Path(tmp)
        if args.netal_bin is not None:
            netal_bin = args.netal_bin.resolve()
        else:
            build_dir = work / "netal_build"
            subprocess.check_call([sys.executable, str(BUILD_SCRIPT), "--output", str(build_dir)])
            netal_bin = build_dir / "netal-nr-fast-cli"

        report: dict[str, object] = {
            "schema": "netal.nr-fast.mlx-parity.v1",
            "mae_limit": args.mae_limit,
        }

        # 1. Raw [1, 256, 384, 16] -> [1, 256, 384, 4] tensor parity
        th, tw = 256, 384
        feat_path = work / "features_256x384x16.bin"
        feat_path.write_bytes(_make_procedural_features(tw, th))

        mlx_tensor_out = work / "mlx_head.bin"
        mlx_runs_ms: list[float] = []
        for i in range(3):
            cur_out = work / f"mlx_head_{i}.bin"
            mlx_res = json.loads(
                subprocess.check_output(
                    [
                        str(mlx_bin),
                        "run",
                        str(model_path),
                        "--input",
                        str(feat_path),
                        "--height",
                        str(th),
                        "--width",
                        str(tw),
                        "--channels",
                        "16",
                        "--execution",
                        "metal-fused",
                        "--precision",
                        "float16",
                        "--output",
                        str(cur_out),
                    ],
                    text=True,
                )
            )
            mlx_runs_ms.append(mlx_res["executionNanoseconds"] / 1e6)
            if i == 2:
                cur_out.replace(mlx_tensor_out)
            else:
                cur_out.unlink()

        netal_tensor_out = work / "netal_head.bin"
        netal_eval = json.loads(
            subprocess.check_output(
                [
                    str(netal_bin),
                    "eval-tensor",
                    "--model",
                    str(model_path),
                    "--input",
                    str(feat_path),
                    "--height",
                    str(th),
                    "--width",
                    str(tw),
                    "--output",
                    str(netal_tensor_out),
                    "--warmup",
                    "2",
                    "--repeat",
                    "6",
                ],
                text=True,
            )
        )

        mlx_head = _load_f32(mlx_tensor_out)
        netal_head = _load_f32(netal_tensor_out)
        rgb_delta_mlx = [mlx_head[i] * 0.25 for i in range(len(mlx_head)) if (i & 3) < 3]
        rgb_delta_netal = [netal_head[i] * 0.25 for i in range(len(netal_head)) if (i & 3) < 3]
        conf_mlx = [1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, mlx_head[i])))) for i in range(3, len(mlx_head), 4)]
        conf_netal = [1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, netal_head[i])))) for i in range(3, len(netal_head), 4)]

        tensor_section: dict[str, object] = {
            "shape": [1, th, tw, 16],
            "mlx_warm_ms": min(mlx_runs_ms),
            "netal_gpu_ms_p50": netal_eval["gpu_ms_p50"],
            "netal_gpu_ms_p95": netal_eval["gpu_ms_p95"],
            "speedup_vs_mlx": min(mlx_runs_ms) / max(1e-6, netal_eval["gpu_ms_p50"]),
            "scaled_rgb_residual": _metrics(rgb_delta_mlx, rgb_delta_netal),
            "temporal_confidence_sigmoid": _metrics(conf_mlx, conf_netal),
            "raw_4ch_head": _metrics(mlx_head, netal_head),
        }

        if fp8_path is not None:
            fp8_out = work / "netal_fp8_head.bin"
            fp8_eval = json.loads(
                subprocess.check_output(
                    [
                        str(netal_bin),
                        "eval-tensor",
                        "--model",
                        str(fp8_path),
                        "--input",
                        str(feat_path),
                        "--height",
                        str(th),
                        "--width",
                        str(tw),
                        "--output",
                        str(fp8_out),
                        "--warmup",
                        "2",
                        "--repeat",
                        "6",
                    ],
                    text=True,
                )
            )
            fp8_head = _load_f32(fp8_out)
            tensor_section["fp8_vs_dlssmodel"] = {
                **_metrics(netal_head, fp8_head),
                "fp8_gpu_ms_p50": fp8_eval["gpu_ms_p50"],
                "bit_exact": max(abs(a - b) for a, b in zip(netal_head, fp8_head)) == 0.0,
            }

        report["tensor_core_parity"] = tensor_section

        # 2. End-to-end render-image parity (standard + cinematic detail-split upsample)
        in_png = work / "scene_384x256.png"
        _make_procedural_png(in_png, 384, 256)
        render_profiles: list[dict[str, object]] = []
        all_passed = True

        profile_configs = [
            ("standard", []),
            ("cinematic", ["--profile", "cinematic", "--detail-strength", "2.5", "--colour-strength", "0.8", "--intensity", "0.9"]),
        ]
        for label, extra_flags in profile_configs:
            mlx_png = work / f"mlx_{label}.png"
            netal_png = work / f"netal_{label}.png"
            mlx_r = json.loads(
                subprocess.check_output(
                    [
                        str(mlx_bin),
                        "render-image",
                        str(in_png),
                        str(model_path),
                        "--output",
                        str(mlx_png),
                        "--execution",
                        "metal-fused",
                        "--precision",
                        "float16",
                    ]
                    + extra_flags,
                    text=True,
                )
            )
            netal_r = json.loads(
                subprocess.check_output(
                    [
                        str(netal_bin),
                        "render-image",
                        "--model",
                        str(fp8_path if fp8_path is not None else model_path),
                        "--input",
                        str(in_png),
                        "--output",
                        str(netal_png),
                        "--warmup",
                        "2",
                        "--repeat",
                        "5",
                    ]
                    + extra_flags,
                    text=True,
                )
            )
            mw, mh, m_px = _read_png_rgb_float(mlx_png)
            nw, nh, n_px = _read_png_rgb_float(netal_png)
            if (mw, mh) != (nw, nh):
                raise RuntimeError(f"output size mismatch: MLX={(mw, mh)} vs Netal={(nw, nh)}")
            m = _metrics(m_px, n_px)
            passed = m["mae"] < args.mae_limit
            if not passed:
                all_passed = False
            render_profiles.append(
                {
                    "profile": label,
                    "output_size": [nw, nh],
                    "passed_mae_gate": passed,
                    **m,
                    "mlx_execution_ms": mlx_r["executionNanoseconds"] / 1e6,
                    "netal_gpu_ms_p50": netal_r["gpu_ms_p50"],
                    "netal_gpu_ms_p95": netal_r["gpu_ms_p95"],
                }
            )

        if tensor_section["scaled_rgb_residual"]["mae"] >= args.mae_limit:  # type: ignore[index]
            all_passed = False
        if fp8_path is not None and not tensor_section["fp8_vs_dlssmodel"]["bit_exact"]:  # type: ignore[index]
            all_passed = False

        report["render_image_parity"] = render_profiles

        # 3. Optional multi-resolution M4 Max latency sweep
        if args.bench_resolutions:
            bench_rows: list[dict[str, object]] = []
            active_real = fp8_path if fp8_path is not None else model_path
            for topo_label, model_flags in [
                ("small31_synthetic", ["--topology", "small"]),
                ("ht71_real", ["--model", str(active_real)]),
            ]:
                for net_sz, out_sz in [
                    ("960x544", "1920x1080"),
                    ("1280x704", "1920x1080"),
                    ("1920x1088", "1920x1088"),
                ]:
                    b = json.loads(
                        subprocess.check_output(
                            [
                                str(netal_bin),
                                "bench",
                                *model_flags,
                                "--net-size",
                                net_sz,
                                "--out-size",
                                out_sz,
                                "--temporal",
                                "--warmup",
                                "2",
                                "--repeat",
                                "6",
                            ],
                            text=True,
                        )
                    )
                    bench_rows.append(
                        {
                            "variant": topo_label,
                            "device": b["device"],
                            "network_size": b["network_size"],
                            "output_size": b["output_size"],
                            "gpu_ms_min": b["gpu_ms_min"],
                            "gpu_ms_p50": b["gpu_ms_p50"],
                            "gpu_ms_p95": b["gpu_ms_p95"],
                            "fps_p50": b["fps_p50"],
                            "cpu_encode_ms_p50": b["cpu_encode_ms_p50"],
                            "dispatch_count": b["dispatch_count"],
                            "weight_buffer_mb": b["weight_buffer_mb"],
                            "scratch_buffer_mb": b["scratch_buffer_mb"],
                        }
                    )
            report["m4_max_benchmarks"] = bench_rows

        report["status"] = "pass" if all_passed else "fail"
        formatted = json.dumps(report, indent=2)
        if args.json_out is not None:
            args.json_out.write_text(formatted + "\n", encoding="utf-8")
        print(formatted)
        return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
