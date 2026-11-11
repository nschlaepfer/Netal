#!/usr/bin/env python3
# SPDX-License-Identifier: MIT AND Apache-2.0
"""Model-free CI verification suite for Netal's native Metal DLSS 5 NR runtime (src/nr_fast).

Exercises:
1. Atomic build script (`scripts/build-netal-nr-fast.py`), output-collision refusal, and SHA-256 receipts.
2. Safetensors & `.dlssmodel` parser fault injection (truncated headers, malformed JSON, OOB/overlapping
   byte intervals, unsupported dtypes, manifest SHA-256 mismatch).
3. Bit-exact `F8_E4M3` (`mlxdlss-native-e4m3-v1`) vs `F16` safetensors round-trip on a full 31-block graph.
4. `Small31` (31-block) and `HT71` (71-block) synthetic GPU execution + contract verification
   (NaN-poison overwrite, finite outputs, input immutability, 4-frame temporal evolution, reset equivalence).
5. End-to-end `render-image` PNG upsampling, sRGB transfer, and detail-split profile differentiation.
"""

from __future__ import annotations

import array
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = REPO_ROOT / "scripts" / "build-netal-nr-fast.py"


def _build_e4m3_decode_lut() -> list[int]:
    """Returns 256 uint16 IEEE-754 binary16 bit patterns matching `mlxdlss-native-e4m3-v1`."""
    lut: list[int] = []
    for b in range(256):
        low_bits = (b & 0x7F) << 7
        mag_f = struct.unpack("<e", struct.pack("<H", low_bits))[0] * 256.0
        val_f = -mag_f if (b & 0x80) else mag_f
        f16_bits = struct.unpack("<H", struct.pack("<e", val_f))[0]
        lut.append(f16_bits)
    return lut


E4M3_LUT = _build_e4m3_decode_lut()


def _pack_safetensors(tensors: dict[str, tuple[str, list[int], bytes]], metadata: dict[str, str] | None = None) -> bytes:
    header: dict[str, object] = {}
    if metadata is not None:
        header["__metadata__"] = metadata
    payload = bytearray()
    offset = 0
    for name in sorted(tensors.keys()):
        dtype, shape, raw = tensors[name]
        end = offset + len(raw)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, end],
        }
        payload.extend(raw)
        offset = end
    hdr_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (8 - (len(hdr_bytes) % 8)) % 8
    hdr_bytes += b" " * pad
    return struct.pack("<Q", len(hdr_bytes)) + hdr_bytes + bytes(payload)


def _generate_small31_weight_pair() -> tuple[bytes, bytes]:
    """Builds matching F16 and F8_E4M3 (`mlxdlss-native-e4m3-v1`) safetensors for Small31 (31 blocks)."""
    f16_specs: list[tuple[str, list[int], bool]] = []  # (name, shape, is_fp8_quantizable)

    def add_window_core(idx: int, ch: int) -> None:
        h = ch // 32
        l0 = f"block{idx}.layer0."
        if ch == 32:
            f16_specs.append((l0 + "weight1", [32, 128], True))
            f16_specs.append((l0 + "weight2", [128, 32], True))
        else:
            f16_specs.append((l0 + "ffn_expand_weight", [h, 4, h, 32, 32], True))
            f16_specs.append((l0 + "ffn_branch_projection_weight", [h, 4, 32, 32], True))
            f16_specs.append((l0 + "ffn_output_projection_weight", [ch, ch], True))
        f16_specs.append((l0 + "ffn_cos_skip", [ch], False))
        f16_specs.append((l0 + "qkv_weight", [ch, ch * 3], True))
        f16_specs.append((l0 + "attn_scale", [h], False))
        f16_specs.append((l0 + "attn_bias", [h, 64, 64], False))
        f16_specs.append((l0 + "projection_weight", [ch, ch], True))
        f16_specs.append((l0 + "attn_cos_skip", [ch], False))

    def add_split_block(idx: int, downsample: bool) -> None:
        b = f"block{idx}"
        f16_specs.append((f"{b}.layer0.first_projection_weight", [512, 512], True))
        f16_specs.append((f"{b}.layer0.group_expand_weight", [8, 64, 256], True))
        f16_specs.append((f"{b}.layer0.group_project_weight", [8, 256, 64], True))
        f16_specs.append((f"{b}.layer1.weight3", [512, 512], True))
        f16_specs.append((f"{b}.layer1.ffn_cos_skip", [512], False))
        f16_specs.append((f"{b}.layer2.qkv_weight", [512, 1536], True))
        f16_specs.append((f"{b}.layer2.attn_scale", [16], False))
        f16_specs.append((f"{b}.layer2.attn_bias", [16, 64, 64], False))
        f16_specs.append((f"{b}.layer3.projection_weight", [512, 512], True))
        f16_specs.append((f"{b}.layer3.attn_cos_skip", [512], False))
        if downsample:
            f16_specs.append((f"{b}.layer4.weight", [512, 1024], True))

    # 0: Pre (32ch)
    f16_specs.append(("block0.layer0.input_adapter_weight", [16, 32], True))
    add_window_core(0, 32)

    # 1..8: Encoder stages [32, 64, 128, 256], depth=2 each (Window then Downsample)
    for b0, ch in [(1, 32), (3, 64), (5, 128), (7, 256)]:
        add_window_core(b0, ch)
        f16_specs.append((f"block{b0 + 1}.layer0.weight0", [ch, ch * 2], True))
        add_window_core(b0 + 1, ch)

    # 9..10: Split encoder (512ch)
    add_split_block(9, downsample=False)
    add_split_block(10, downsample=True)

    # 11..18: Global attention (8 blocks, 1024ch)
    for b in range(11, 19):
        f16_specs.append((f"block{b}.layer0.weight", [1024, 4096], True))
        f16_specs.append((f"block{b}.layer1.weight", [4096, 1024], True))
        f16_specs.append((f"block{b}.layer1.ffn_cos_skip", [1024], False))
        f16_specs.append((f"block{b}.layer2.qkv_weight", [1024, 3072], True))
        f16_specs.append((f"block{b}.layer2.attn_scale", [32], False))
        f16_specs.append((f"block{b}.layer4.projection_weight", [1024, 1024], True))
        f16_specs.append((f"block{b}.layer4.attn_cos_skip", [1024], False))

    # 19: Bridge (1024 -> 512)
    f16_specs.append(("block19.layer0.conv_weight", [1024, 512], True))
    f16_specs.append(("block19.layer0.inp_upsample_sin", [512], False))

    # 20..21: Split decoder (512ch)
    add_split_block(20, downsample=False)
    add_split_block(21, downsample=False)

    # 22..29: Decoder stages [256, 128, 64, 32], depth=2 each (Upsample then Window)
    for b0, ch in [(22, 256), (24, 128), (26, 64), (28, 32)]:
        f16_specs.append((f"block{b0}.layer0.weight0", [ch * 2, ch], True))
        f16_specs.append((f"block{b0}.layer0.sin", [ch], False))
        add_window_core(b0, ch)
        add_window_core(b0 + 1, ch)

    # 30: Post (32ch -> 4ch head)
    f16_specs.append(("block30.layer0.inp_merge_sin", [32], False))
    f16_specs.append(("block30.layer0.inp_merge_cos", [32], False))
    f16_specs.append(("block30.layer0.out_gain", [16, 4], True))
    f16_specs.append(("block30.layer0.out_conv_weight", [16, 4], True))
    f16_specs.append(("block30.layer0.blend_scale", [1], False))
    add_window_core(30, 32)

    # Small safe E4M3 codes (indices with small magnitudes around +/- 0.008 .. 0.03)
    safe_codes = [0x10, 0x12, 0x14, 0x16, 0x90, 0x92, 0x94, 0x96, 0x00]
    tensors_f16: dict[str, tuple[str, list[int], bytes]] = {}
    tensors_fp8: dict[str, tuple[str, list[int], bytes]] = {}

    rng = 0x12345678
    for name, shape, quant in f16_specs:
        numel = 1
        for d in shape:
            numel *= d
        if name.endswith("blend_scale"):
            bits = struct.unpack("<H", struct.pack("<e", 0.75))[0]
            raw16 = struct.pack("<H", bits)
            tensors_f16[name] = ("F16", shape, raw16)
            tensors_fp8[name] = ("F16", shape, raw16)
        elif not quant:
            val = (
                0.85
                if ("cos" in name)
                else (0.15 if ("sin" in name) else (1.0 if "attn_scale" in name else 0.0))
            )
            bits = struct.unpack("<H", struct.pack("<e", val))[0]
            raw16 = struct.pack("<H", bits) * numel
            tensors_f16[name] = ("F16", shape, raw16)
            tensors_fp8[name] = ("F16", shape, raw16)
        else:
            u8_buf = bytearray(numel)
            u16_arr = array.array("H")
            for i in range(numel):
                rng = (rng * 1664525 + 1013904223) & 0xFFFFFFFF
                code = safe_codes[(rng >> 16) % len(safe_codes)]
                u8_buf[i] = code
                u16_arr.append(E4M3_LUT[code])
            if sys.byteorder != "little":
                u16_arr.byteswap()
            tensors_f16[name] = ("F16", shape, u16_arr.tobytes())
            tensors_fp8[name] = ("F8_E4M3", shape, bytes(u8_buf))

    st_f16 = _pack_safetensors(tensors_f16)
    st_fp8 = _pack_safetensors(tensors_fp8, metadata={"format": "mlxdlss-native-e4m3-v1"})
    return st_f16, st_fp8


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
        + chunk(b"IDAT", zlib.compress(bytes(raw), 6))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def _read_png_rgb_float(path: Path) -> tuple[int, int, list[float]]:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("invalid PNG header")
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
    if bit_depth != 8 or color_type not in (2, 6):
        raise ValueError(f"unsupported PNG format bit_depth={bit_depth} color_type={color_type}")
    bpp = 3 if color_type == 2 else 4
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
        if flt == 1:  # Sub
            for i in range(bpp, stride):
                cur[i] = (cur[i] + cur[i - bpp]) & 0xFF
        elif flt == 2:  # Up
            for i in range(stride):
                cur[i] = (cur[i] + prev[i]) & 0xFF
        elif flt == 3:  # Average
            for i in range(stride):
                left = cur[i - bpp] if i >= bpp else 0
                cur[i] = (cur[i] + ((left + prev[i]) >> 1)) & 0xFF
        elif flt == 4:  # Paeth
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
        for x in range(width):
            out[row_base + x * 3 + 0] = cur[x * bpp + 0] / 255.0
            out[row_base + x * 3 + 1] = cur[x * bpp + 1] / 255.0
            out[row_base + x * 3 + 2] = cur[x * bpp + 2] / 255.0
    return width, height, out


class NetalNrFastTestSuite(unittest.TestCase):
    _tmpdir: tempfile.TemporaryDirectory[str]
    _build_dir: Path
    _cli: Path
    _has_apple7_gpu: bool = True

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.TemporaryDirectory(prefix="netal-nr-fast-ci-")
        cls._build_dir = Path(cls._tmpdir.name) / "build"
        proc = subprocess.run(
            [sys.executable, str(BUILD_SCRIPT), "--output", str(cls._build_dir)],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"build-netal-nr-fast.py failed:\n{proc.stdout}\n{proc.stderr}")
        cls._cli = cls._build_dir / "netal-nr-fast-cli"

        # Detect whether the host exposes an Apple7+ GPU supporting simdgroup_matrix
        # (true on all physical M1-M4 Macs; false on paravirtualized headless CI VMs).
        cap_mm = Path(cls._tmpdir.name) / "gpu_cap.mm"
        cap_bin = Path(cls._tmpdir.name) / "gpu_cap"
        cap_mm.write_text(
            "#import <Metal/Metal.h>\n"
            "int main() { @autoreleasepool {\n"
            "  id<MTLDevice> d = MTLCreateSystemDefaultDevice();\n"
            "  return (d && [d supportsFamily:MTLGPUFamilyApple7]) ? 0 : 1;\n"
            "} }\n",
            encoding="utf-8",
        )
        subprocess.check_call(
            ["xcrun", "--sdk", "macosx", "clang++", "-fobjc-arc", "-framework", "Metal", str(cap_mm), "-o", str(cap_bin)]
        )
        cls._has_apple7_gpu = subprocess.run([str(cap_bin)], check=False).returncode == 0

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()

    def test_01_build_receipt_and_atomic_refusal(self) -> None:
        receipt_path = self._build_dir / "build-receipt.json"
        self.assertTrue(receipt_path.is_file())
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        self.assertEqual(receipt["schema"], "netal.nr.fast-build.v1")
        expected_names = {"nr_fast_kernels.metallib", "libnetal_nr_fast.a", "netal-nr-fast-cli"}
        found_names = set()
        for entry in receipt["artifacts"]:
            art = self._build_dir / entry["name"]
            self.assertTrue(art.is_file(), f"missing build artifact {entry['name']}")
            data = art.read_bytes()
            self.assertEqual(len(data), entry["bytes"])
            self.assertEqual(hashlib.sha256(data).hexdigest(), entry["sha256"])
            found_names.add(entry["name"])
        self.assertEqual(found_names, expected_names)

        # Existing output directory must be refused cleanly.
        collide = subprocess.run(
            [sys.executable, str(BUILD_SCRIPT), "--output", str(self._build_dir)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(collide.returncode, 0)

    def test_02_safetensors_and_bundle_fault_injection(self) -> None:
        bad_dir = Path(self._tmpdir.name) / "faults"
        bad_dir.mkdir(exist_ok=True)

        cases: dict[str, Path] = {}

        # 1. Truncated < 8 bytes
        p_trunc = bad_dir / "truncated.safetensors"
        p_trunc.write_bytes(b"\x04\x00\x00")
        cases["truncated"] = p_trunc

        # 2. Declared header larger than file
        p_hdr_oob = bad_dir / "hdr_oob.safetensors"
        p_hdr_oob.write_bytes(struct.pack("<Q", 4096) + b"{}")
        cases["hdr_oob"] = p_hdr_oob

        # 3. Malformed JSON header
        p_bad_json = bad_dir / "bad_json.safetensors"
        bad_j = b"{not_valid_json  "
        p_bad_json.write_bytes(struct.pack("<Q", len(bad_j)) + bad_j)
        cases["bad_json"] = p_bad_json

        # 4. Out-of-bounds tensor data_offsets
        p_oob = bad_dir / "offset_oob.safetensors"
        oob_j = json.dumps(
            {"block0.layer0.cosine_scale": {"dtype": "F16", "shape": [32], "data_offsets": [0, 128]}}
        ).encode("utf-8")
        p_oob.write_bytes(struct.pack("<Q", len(oob_j)) + oob_j + b"\x00" * 64)
        cases["offset_oob"] = p_oob

        # 5. Overlapping tensor slices
        p_overlap = bad_dir / "overlap.safetensors"
        ov_j = json.dumps(
            {
                "t0": {"dtype": "F16", "shape": [32], "data_offsets": [0, 64]},
                "t1": {"dtype": "F16", "shape": [32], "data_offsets": [32, 96]},
            }
        ).encode("utf-8")
        p_overlap.write_bytes(struct.pack("<Q", len(ov_j)) + ov_j + b"\x00" * 96)
        cases["overlap"] = p_overlap

        # 6. Unsupported dtype
        p_dtype = bad_dir / "bad_dtype.safetensors"
        dt_j = json.dumps(
            {"block0.layer0.cosine_scale": {"dtype": "I64", "shape": [8], "data_offsets": [0, 64]}}
        ).encode("utf-8")
        p_dtype.write_bytes(struct.pack("<Q", len(dt_j)) + dt_j + b"\x00" * 64)
        cases["bad_dtype"] = p_dtype

        for label, path in cases.items():
            proc = subprocess.run(
                [str(self._cli), "validate-model", "--model", str(path)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(proc.returncode, 0, f"expected {label} to fail validation")

        # 7. Bundle with mismatched SHA-256 in manifest.json
        bundle = bad_dir / "corrupt.dlssmodel"
        bundle.mkdir(exist_ok=True)
        valid_st, _ = _generate_small31_weight_pair()
        (bundle / "weights.safetensors").write_bytes(valid_st)
        manifest = {
            "schema": "mlxdlss.neural-rendering-transformer.v1",
            "weights": {
                "file": "weights.safetensors",
                "sha256": "0" * 64,
                "byteCount": len(valid_st),
            },
        }
        (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        proc_b = subprocess.run(
            [str(self._cli), "validate-model", "--model", str(bundle)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(proc_b.returncode, 0, "expected SHA-256 mismatch bundle to fail")

    def test_03_fp8_e4m3_safetensors_bit_exact_roundtrip(self) -> None:
        rt_dir = Path(self._tmpdir.name) / "fp8_roundtrip"
        rt_dir.mkdir(exist_ok=True)
        st_f16, st_fp8 = _generate_small31_weight_pair()
        p_f16 = rt_dir / "small_f16.safetensors"
        p_fp8 = rt_dir / "small_fp8.safetensors"
        p_f16.write_bytes(st_f16)
        p_fp8.write_bytes(st_fp8)

        val = json.loads(
            subprocess.check_output(
                [str(self._cli), "validate-model", "--model", str(p_fp8), "--check-e4m3-lut"],
                text=True,
            )
        )
        self.assertEqual(val["status"], "valid")
        self.assertEqual(val["fp8_e4m3_count"], 153)
        self.assertEqual(val["total_blocks"], 31)
        if not self._has_apple7_gpu:
            self.skipTest("Apple7+ GPU required for simdgroup_matrix execution")

        h, w = 128, 256
        feat = array.array("f", (((i % 19) - 9.0) * 0.05 for i in range(h * w * 16)))
        if sys.byteorder != "little":
            feat.byteswap()
        p_in = rt_dir / "feat_in.bin"
        p_in.write_bytes(feat.tobytes())
        p_out16 = rt_dir / "out16.bin"
        p_out8 = rt_dir / "out8.bin"

        subprocess.check_call(
            [
                str(self._cli),
                "eval-tensor",
                "--model",
                str(p_f16),
                "--input",
                str(p_in),
                "--height",
                str(h),
                "--width",
                str(w),
                "--output",
                str(p_out16),
            ],
            stdout=subprocess.DEVNULL,
        )
        subprocess.check_call(
            [
                str(self._cli),
                "eval-tensor",
                "--model",
                str(p_fp8),
                "--input",
                str(p_in),
                "--height",
                str(h),
                "--width",
                str(w),
                "--output",
                str(p_out8),
            ],
            stdout=subprocess.DEVNULL,
        )

        a16 = array.array("f")
        a16.frombytes(p_out16.read_bytes())
        a8 = array.array("f")
        a8.frombytes(p_out8.read_bytes())
        if sys.byteorder != "little":
            a16.byteswap()
            a8.byteswap()
        self.assertEqual(len(a16), h * w * 4)
        self.assertTrue(all(math.isfinite(x) for x in a16))
        max_diff = max(abs(x - y) for x, y in zip(a16, a8))
        self.assertEqual(max_diff, 0.0)

    def test_04_synthetic_small31_and_ht71_gpu_contracts(self) -> None:
        if not self._has_apple7_gpu:
            self.skipTest("Apple7+ GPU required for simdgroup_matrix execution")
        for topo, out_sz, expected_blocks in [("small", "256x128", 31), ("ht", "384x192", 71)]:
            res = json.loads(
                subprocess.check_output(
                    [
                        str(self._cli),
                        "bench",
                        "--topology",
                        topo,
                        "--net-size",
                        "256x128",
                        "--out-size",
                        out_sz,
                        "--temporal",
                        "--verify-contracts",
                        "--warmup",
                        "1",
                        "--repeat",
                        "2",
                    ],
                    text=True,
                )
            )
            self.assertEqual(res["block_count"], expected_blocks)
            self.assertTrue(res["contracts_verified"], f"contract verification failed for {topo}")
            self.assertGreater(res["temporal_l1_delta"], 0.0)
            self.assertEqual(res["reset_max_err"], 0.0)
            self.assertGreaterEqual(res["gpu_ms_p50"], 0.0)

    def test_05_render_image_profiles_and_detail_upsample(self) -> None:
        if not self._has_apple7_gpu:
            self.skipTest("Apple7+ GPU required for simdgroup_matrix execution")
        img_dir = Path(self._tmpdir.name) / "render"
        img_dir.mkdir(exist_ok=True)
        in_w, in_h = 256, 128
        rgb = bytearray(in_w * in_h * 3)
        for y in range(in_h):
            for x in range(in_w):
                idx = (y * in_w + x) * 3
                edge = 180 if ((x // 16 + y // 16) % 2 == 0) else 45
                rgb[idx + 0] = min(255, int(edge + (x / in_w) * 60))
                rgb[idx + 1] = min(255, int(edge * 0.8 + (y / in_h) * 70))
                rgb[idx + 2] = min(255, int(40 + ((x ^ y) & 63) * 2))
        in_png = img_dir / "in.png"
        _write_rgb8_png(in_png, in_w, in_h, bytes(rgb))

        out_std = img_dir / "out_std.png"
        out_cin = img_dir / "out_cin.png"
        rep_std = json.loads(
            subprocess.check_output(
                [
                    str(self._cli),
                    "render-image",
                    "--synthetic",
                    "small",
                    "--input",
                    str(in_png),
                    "--output",
                    str(out_std),
                    "--net-size",
                    "256x128",
                    "--out-size",
                    "384x192",
                    "--profile",
                    "standard",
                ],
                text=True,
            )
        )
        rep_cin = json.loads(
            subprocess.check_output(
                [
                    str(self._cli),
                    "render-image",
                    "--synthetic",
                    "small",
                    "--input",
                    str(in_png),
                    "--output",
                    str(out_cin),
                    "--net-size",
                    "256x128",
                    "--out-size",
                    "384x192",
                    "--profile",
                    "cinematic",
                    "--detail-strength",
                    "2.4",
                    "--colour-strength",
                    "0.8",
                    "--intensity",
                    "0.95",
                    "--srgb",
                ],
                text=True,
            )
        )
        self.assertEqual(rep_std["output_size"], [384, 192])
        self.assertEqual(rep_cin["output_size"], [384, 192])

        w1, h1, px_std = _read_png_rgb_float(out_std)
        w2, h2, px_cin = _read_png_rgb_float(out_cin)
        self.assertEqual((w1, h1), (384, 192))
        self.assertEqual((w2, h2), (384, 192))
        self.assertTrue(all(0.0 <= v <= 1.0 and math.isfinite(v) for v in px_std))
        self.assertTrue(all(0.0 <= v <= 1.0 and math.isfinite(v) for v in px_cin))
        diff = sum(abs(a - b) for a, b in zip(px_std, px_cin)) / len(px_std)
        self.assertGreater(diff, 1e-4)

    def test_06_motion_scale_and_cached_delta_composition(self) -> None:
        if not self._has_apple7_gpu:
            self.skipTest("Apple7+ GPU required for simdgroup_matrix execution")
        probe_dir = Path(self._tmpdir.name) / "api_probe"
        probe_dir.mkdir(exist_ok=True)
        probe_mm = probe_dir / "probe.mm"
        probe_bin = probe_dir / "probe"
        metallib = self._build_dir / "nr_fast_kernels.metallib"
        static_lib = self._build_dir / "libnetal_nr_fast.a"

        probe_mm.write_text(
            r'''
#import "nr_fast_runtime.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

using namespace netal_nr_fast;

static id<MTLTexture> MakeTex(id<MTLDevice> dev, uint32_t w, uint32_t h, MTLPixelFormat pf, MTLTextureUsage usage) {
  MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:pf width:w height:h mipmapped:NO];
  td.storageMode = MTLStorageModeShared;
  td.usage = usage;
  return [dev newTextureWithDescriptor:td];
}

int main(int argc, char** argv) {
  @autoreleasepool {
    if (argc < 2) return 2;
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) return 3;
    id<MTLCommandQueue> q_main = [dev newCommandQueue];
    id<MTLCommandQueue> q_worker = [dev newCommandQueue];

    SyntheticOptions sopt{};
    sopt.metallib_path = argv[1];
    std::string err;
    auto ctx = Context::PrepareSynthetic(dev, TopologyKind::Small31, sopt, &err);
    if (!ctx) {
      std::fprintf(stderr, "PrepareSynthetic failed: %s\n", err.c_str());
      return 1;
    }

    const uint32_t net_w = 256, net_h = 128;
    const uint32_t out_w = 384, out_h = 192;
    id<MTLTexture> col0 = MakeTex(dev, out_w, out_h, MTLPixelFormatRGBA32Float, MTLTextureUsageShaderRead);
    id<MTLTexture> col1 = MakeTex(dev, out_w, out_h, MTLPixelFormatRGBA32Float, MTLTextureUsageShaderRead);
    id<MTLTexture> mv_uv = MakeTex(dev, out_w, out_h, MTLPixelFormatRG32Float, MTLTextureUsageShaderRead);
    id<MTLTexture> mv_px = MakeTex(dev, out_w, out_h, MTLPixelFormatRG16Float, MTLTextureUsageShaderRead);
    id<MTLTexture> out_ref = MakeTex(dev, out_w, out_h, MTLPixelFormatRGBA32Float, MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite);
    id<MTLTexture> out_test = MakeTex(dev, out_w, out_h, MTLPixelFormatRGBA32Float, MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite);

    std::vector<float> c0(out_w * out_h * 4u), c1(out_w * out_h * 4u);
    std::vector<float> m_uv(out_w * out_h * 2u);
    std::vector<uint16_t> m_px(out_w * out_h * 2u);
    auto f32_to_f16 = [](float v) -> uint16_t {
      _Float16 h = static_cast<_Float16>(v);
      uint16_t b = 0;
      std::memcpy(&b, &h, 2);
      return b;
    };
    // -6/384 = -0.015625 and 3/192 = 0.015625 are exact binary fractions in FP16 and FP32.
    const float dx_px = -6.0f, exact_dy_px = 3.0f;
    for (uint32_t y = 0; y < out_h; ++y) {
      for (uint32_t x = 0; x < out_w; ++x) {
        size_t i4 = (size_t(y) * out_w + x) * 4u;
        size_t i2 = (size_t(y) * out_w + x) * 2u;
        float fx = float(x) / float(out_w);
        float fy = float(y) / float(out_h);
        c0[i4 + 0] = 0.45f + 0.35f * std::sin(fx * 21.0f + fy * 9.0f);
        c0[i4 + 1] = 0.50f + 0.30f * std::cos(fx * 13.0f - fy * 17.0f);
        c0[i4 + 2] = 0.40f + 0.35f * std::sin((fx + fy) * 19.0f);
        c0[i4 + 3] = 1.0f;

        c1[i4 + 0] = 0.45f + 0.35f * std::sin((fx - 0.02f) * 21.0f + (fy + 0.015f) * 9.0f);
        c1[i4 + 1] = 0.50f + 0.30f * std::cos((fx - 0.02f) * 13.0f - (fy + 0.015f) * 17.0f);
        c1[i4 + 2] = 0.40f + 0.35f * std::sin((fx + fy - 0.005f) * 19.0f);
        c1[i4 + 3] = 1.0f;

        m_uv[i2 + 0] = dx_px / float(out_w);
        m_uv[i2 + 1] = exact_dy_px / float(out_h);
        m_px[i2 + 0] = f32_to_f16(dx_px);
        m_px[i2 + 1] = f32_to_f16(exact_dy_px);
      }
    }
    [col0 replaceRegion:MTLRegionMake2D(0, 0, out_w, out_h) mipmapLevel:0 withBytes:c0.data() bytesPerRow:out_w * 16u];
    [col1 replaceRegion:MTLRegionMake2D(0, 0, out_w, out_h) mipmapLevel:0 withBytes:c1.data() bytesPerRow:out_w * 16u];
    [mv_uv replaceRegion:MTLRegionMake2D(0, 0, out_w, out_h) mipmapLevel:0 withBytes:m_uv.data() bytesPerRow:out_w * 8u];
    [mv_px replaceRegion:MTLRegionMake2D(0, 0, out_w, out_h) mipmapLevel:0 withBytes:m_px.data() bytesPerRow:out_w * 4u];

    // 1. Verify motion_scale bit-exactness against pre-normalized UV motion.
    Report rep{};
    FrameInputs fi{};
    fi.params.output_width = out_w;
    fi.params.output_height = out_h;
    fi.params.network_width = net_w;
    fi.params.network_height = net_h;
    fi.params.geometry_mode = GeometryMode::ScaledNetwork;

    ctx->reset_history();
    fi.color_texture = col0;
    fi.motion_texture = nil;
    fi.output_texture = out_ref;
    fi.params.frame_index = 0;
    fi.params.reset = true;
    if (!ctx->run_sync(fi, q_main, rep, &err)) return 10;

    fi.color_texture = col1;
    fi.motion_texture = mv_uv;
    fi.params.frame_index = 1;
    fi.params.reset = false;
    fi.params.motion_scale_x = 1.0f;
    fi.params.motion_scale_y = 1.0f;
    if (!ctx->run_sync(fi, q_main, rep, &err)) return 11;

    ctx->reset_history();
    fi.color_texture = col0;
    fi.motion_texture = nil;
    fi.output_texture = out_test;
    fi.params.frame_index = 0;
    fi.params.reset = true;
    if (!ctx->run_sync(fi, q_main, rep, &err)) return 12;

    fi.color_texture = col1;
    fi.motion_texture = mv_px;
    fi.params.frame_index = 1;
    fi.params.reset = false;
    fi.params.motion_scale_x = 1.0f / float(out_w);
    fi.params.motion_scale_y = 1.0f / float(out_h);
    if (!ctx->run_sync(fi, q_main, rep, &err)) return 13;

    std::vector<float> r_uv(out_w * out_h * 4u), r_px(out_w * out_h * 4u);
    [out_ref getBytes:r_uv.data() bytesPerRow:out_w * 16u fromRegion:MTLRegionMake2D(0, 0, out_w, out_h) mipmapLevel:0];
    [out_test getBytes:r_px.data() bytesPerRow:out_w * 16u fromRegion:MTLRegionMake2D(0, 0, out_w, out_h) mipmapLevel:0];
    double max_mv_diff = 0.0;
    for (size_t i = 0; i < r_uv.size(); ++i) {
      max_mv_diff = std::max(max_mv_diff, std::abs(double(r_uv[i]) - double(r_px[i])));
    }

    // 2. Verify export_delta_buffer + encode_compose_delta bit-exactness (both fast and blur paths),
    //    plus worker-only export (output_texture == nil) and concurrent worker + display compose.
    const size_t delta_bytes = size_t(net_w) * net_h * 16u;
    id<MTLBuffer> delta_slot0 = [dev newBufferWithLength:delta_bytes options:MTLResourceStorageModePrivate];
    id<MTLBuffer> delta_slot1 = [dev newBufferWithLength:delta_bytes options:MTLResourceStorageModePrivate];
    MTLTextureDescriptor* td_baked = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                                                        width:net_w
                                                                                       height:net_h
                                                                                    mipmapped:NO];
    td_baked.storageMode = MTLStorageModePrivate;
    td_baked.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    id<MTLTexture> baked_slot0 = [dev newTextureWithDescriptor:td_baked];

    double max_compose_fast_diff = 0.0;
    double max_compose_blur_diff = 0.0;
    double max_baked_fast_diff = 0.0;
    double max_baked_blur_diff = 0.0;
    double live_reuse_l1 = 0.0;

    for (int mode = 0; mode < 2; ++mode) {
      ctx->reset_history();
      FrameInputs wfi{};
      wfi.color_texture = col0;
      wfi.output_texture = out_ref;
      wfi.export_delta_buffer = delta_slot0;
      wfi.export_baked_delta_texture = baked_slot0;
      wfi.params.output_width = out_w;
      wfi.params.output_height = out_h;
      wfi.params.network_width = net_w;
      wfi.params.network_height = net_h;
      wfi.params.geometry_mode = GeometryMode::ScaledNetwork;
      wfi.params.detail_strength = (mode == 0) ? 1.0f : 2.2f;
      wfi.params.colour_strength = (mode == 0) ? 1.0f : 0.7f;
      wfi.params.intensity = 0.9f;
      wfi.params.srgb_transfer = (mode == 1);
      wfi.params.reset = true;
      if (!ctx->run_sync(wfi, q_worker, rep, &err)) return 20 + mode;

      // Poison out_test with NaNs, then compose from exported delta_slot0!
      std::vector<float> nan_buf(out_w * out_h * 4u, std::numeric_limits<float>::quiet_NaN());
      [out_test replaceRegion:MTLRegionMake2D(0, 0, out_w, out_h) mipmapLevel:0 withBytes:nan_buf.data() bytesPerRow:out_w * 16u];

      id<MTLCommandBuffer> ccmd = [q_main commandBuffer];
      if (!ctx->encode_compose_delta(col0, delta_slot0, net_w, net_h, wfi.params, out_test, ccmd, &err)) {
        std::fprintf(stderr, "encode_compose_delta failed: %s\n", err.c_str());
        return 22 + mode;
      }
      [ccmd commit];
      [ccmd waitUntilCompleted];

      std::vector<float> b_ref(out_w * out_h * 4u), b_cmp(out_w * out_h * 4u), b_baked(out_w * out_h * 4u);
      [out_ref getBytes:b_ref.data() bytesPerRow:out_w * 16u fromRegion:MTLRegionMake2D(0, 0, out_w, out_h) mipmapLevel:0];
      [out_test getBytes:b_cmp.data() bytesPerRow:out_w * 16u fromRegion:MTLRegionMake2D(0, 0, out_w, out_h) mipmapLevel:0];
      double md = 0.0;
      for (size_t i = 0; i < b_ref.size(); ++i) {
        if (!std::isfinite(b_cmp[i])) return 25;
        md = std::max(md, std::abs(double(b_ref[i]) - double(b_cmp[i])));
      }
      if (mode == 0) max_compose_fast_diff = md;
      else max_compose_blur_diff = md;

      // Also verify single-pass hardware-bilinear baked delta composition!
      [out_test replaceRegion:MTLRegionMake2D(0, 0, out_w, out_h) mipmapLevel:0 withBytes:nan_buf.data() bytesPerRow:out_w * 16u];
      id<MTLCommandBuffer> bcmd = [q_main commandBuffer];
      if (!ctx->encode_compose_baked_delta(col0, baked_slot0, wfi.params, out_test, bcmd, &err)) {
        std::fprintf(stderr, "encode_compose_baked_delta failed: %s\n", err.c_str());
        return 26 + mode;
      }
      [bcmd commit];
      [bcmd waitUntilCompleted];
      [out_test getBytes:b_baked.data() bytesPerRow:out_w * 16u fromRegion:MTLRegionMake2D(0, 0, out_w, out_h) mipmapLevel:0];
      double mbd = 0.0;
      for (size_t i = 0; i < b_ref.size(); ++i) {
        if (!std::isfinite(b_baked[i])) return 28;
        mbd = std::max(mbd, std::abs(double(b_ref[i]) - double(b_baked[i])));
      }
      if (mode == 0) max_baked_fast_diff = mbd;
      else max_baked_blur_diff = mbd;
    }

    // 3. Concurrent async worker (output_texture == nil, writing delta_slot1) while display queue
    //    simultaneously composes delta_slot0 onto live frame col1 (detail != colour blur path)!
    FrameParams blur_params{};
    blur_params.output_width = out_w;
    blur_params.output_height = out_h;
    blur_params.network_width = net_w;
    blur_params.network_height = net_h;
    blur_params.geometry_mode = GeometryMode::ScaledNetwork;
    blur_params.detail_strength = 2.2f;
    blur_params.colour_strength = 0.7f;
    blur_params.intensity = 0.9f;
    blur_params.srgb_transfer = true;
    blur_params.frame_index = 1;
    blur_params.reset = false;
    blur_params.motion_scale_x = 1.0f / float(out_w);
    blur_params.motion_scale_y = 1.0f / float(out_h);

    FrameInputs async_worker_in{};
    async_worker_in.color_texture = col1;
    async_worker_in.motion_texture = mv_px;
    async_worker_in.output_texture = nil;  // pure worker delta export!
    async_worker_in.export_delta_buffer = delta_slot1;
    async_worker_in.params = blur_params;

    id<MTLCommandBuffer> cb_worker = [q_worker commandBuffer];
    id<MTLCommandBuffer> cb_display = [q_main commandBuffer];
    FrameTicket worker_ticket{};
    Report worker_rep{};
    if (!ctx->encode(async_worker_in, cb_worker, worker_ticket, worker_rep, &err)) {
      std::fprintf(stderr, "worker encode failed: %s\n", err.c_str());
      return 30;
    }
    if (!ctx->encode_compose_delta(col1, delta_slot0, net_w, net_h, blur_params, out_test, cb_display, &err)) {
      std::fprintf(stderr, "concurrent display compose failed: %s\n", err.c_str());
      return 31;
    }
    [cb_worker commit];
    [cb_display commit];
    [cb_display waitUntilCompleted];
    [cb_worker waitUntilCompleted];
    if (cb_display.status != MTLCommandBufferStatusCompleted || cb_worker.status != MTLCommandBufferStatusCompleted) {
      return 32;
    }
    double compose_gpu_ms = (cb_display.GPUEndTime > cb_display.GPUStartTime && cb_display.GPUStartTime > 0.0)
                                ? (cb_display.GPUEndTime - cb_display.GPUStartTime) * 1000.0
                                : 0.0;

    std::vector<float> live_out(out_w * out_h * 4u);
    [out_test getBytes:live_out.data() bytesPerRow:out_w * 16u fromRegion:MTLRegionMake2D(0, 0, out_w, out_h) mipmapLevel:0];
    for (size_t px = 0; px < size_t(out_w) * out_h; ++px) {
      for (size_t c = 0; c < 3; ++c) {
        float v = live_out[px * 4u + c];
        if (!std::isfinite(v) || v < 0.0f || v > 1.0f) return 33;
        live_reuse_l1 += std::abs(double(v) - double(c1[px * 4u + c]));
      }
    }
    live_reuse_l1 /= double(size_t(out_w) * out_h * 3u);

    std::printf(
        "{\"max_mv_diff\":%.9g,\"max_compose_fast_diff\":%.9g,\"max_compose_blur_diff\":%.9g,"
        "\"max_baked_fast_diff\":%.9g,\"max_baked_blur_diff\":%.9g,"
        "\"live_reuse_l1\":%.9g,\"compose_gpu_ms\":%.4f}\n",
        max_mv_diff, max_compose_fast_diff, max_compose_blur_diff,
        max_baked_fast_diff, max_baked_blur_diff, live_reuse_l1, compose_gpu_ms);
    return 0;
  }
}
''',
            encoding="utf-8",
        )

        subprocess.check_call(
            [
                "xcrun",
                "--sdk",
                "macosx",
                "clang++",
                "-std=c++20",
                "-arch",
                "arm64",
                "-fobjc-arc",
                "-O2",
                "-Wall",
                "-Wextra",
                "-Werror",
                f"-I{REPO_ROOT / 'src/nr_fast'}",
                str(probe_mm),
                str(static_lib),
                "-framework",
                "Foundation",
                "-framework",
                "Metal",
                "-o",
                str(probe_bin),
            ]
        )
        res = json.loads(subprocess.check_output([str(probe_bin), str(metallib)], text=True))
        self.assertEqual(res["max_mv_diff"], 0.0)
        self.assertEqual(res["max_compose_fast_diff"], 0.0)
        self.assertEqual(res["max_compose_blur_diff"], 0.0)
        self.assertLess(res["max_baked_fast_diff"], 0.005)
        self.assertLess(res["max_baked_blur_diff"], 0.02)
        self.assertGreater(res["live_reuse_l1"], 1e-4)
        self.assertLess(res["compose_gpu_ms"], 25.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
