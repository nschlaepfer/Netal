#!/usr/bin/env python3
"""Assemble the immutable exact-SR runtime subset for the RE9 bridge."""

from __future__ import annotations

import hashlib
import json
import shutil
import struct
import subprocess
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
WORKSPACE = REPOSITORY.parent
AUTHORITY = WORKSPACE / "NeuralStudio-sr-research"
CUMETAL = WORKSPACE / "CuMetal-sr-wmma"
OUTPUT = REPOSITORY / "work" / "re9-sr-runtime-bundle"

AUTHORITY_COMMIT = "adc05ed"
CUMETAL_COMMIT = "14a6777"
WEIGHTS_SHA256 = "ddaa59cd82deed53bad9d65b82b454f1439fc4875e4220b3ffafafb6cd0296a1"
IMPLEMENTATION_ROOT = "04cbe24e1bbf6326fbd02401af6f1f18e2c56060efcece7954048472af2908db"
DEPTH_INVERTED_INPUT_SHA256 = "be6c8a7b70925e4bc7b624a8002795150758ca9fa9e2d34dba9ff7541db75124"
DEPTH_INVERTED_OUTPUT_SHA256 = "bb14335d8977dff5ddbc7b1418df5af96e54b3883b89780f48be021da3671d40"
LIVE_TEXTURE_CONVERTER_SHA256 = "1a60b8a9da8162feb87dca08cd8252526183e7314fb733ad7533862e241bf65c"

PIPELINES = (
    "dltss_pwin_enc0_layer",
    "dltss_pwin_enc1_layer",
    "dltss_pwin_enc2_layer",
    "dltss_pwin_enc3_layer",
    "dltss_pwin_enc4_layer",
    "dltss_pwin_dec5_layer",
    "dltss_pwin_dec4_layer",
    "dltss_pwin_dec3_layer",
    "dltss_pwin_dec2_layer",
    "dltss_pwin_dec1_layer",
    "dltss_pwin_dec0_layer",
    "hiluma_input_poison_a",
    "hiluma_input_poison_b",
    "hiluma_output_exact_math_safe",
    "cuda_copy_exposure_kernel",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(repository: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "--short", "HEAD"], text=True
    ).strip()


def canonical_root(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def copy(source: Path, relative: Path, rows: list[dict[str, object]]) -> None:
    if not source.is_file():
        raise RuntimeError(f"missing runtime artifact: {source}")
    destination = OUTPUT / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    rows.append(
        {
            "path": relative.as_posix(),
            "bytes": destination.stat().st_size,
            "sha256": sha256(destination),
        }
    )


def write_sanitized(
    source: Path,
    relative: Path,
    pointer_offsets: list[int],
    expected_size: int,
    rows: list[dict[str, object]],
) -> None:
    data = bytearray(source.read_bytes())
    if len(data) != expected_size:
        raise RuntimeError(f"ABI template size drift: {source}")
    for offset in pointer_offsets:
        if offset < 0 or offset + 8 > len(data):
            raise RuntimeError(f"ABI pointer offset outside template: {source}:{offset}")
        data[offset : offset + 8] = b"\0" * 8
    destination = OUTPUT / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    rows.append(
        {
            "path": relative.as_posix(),
            "bytes": len(data),
            "sha256": sha256(destination),
            "sanitized_pointer_offsets": pointer_offsets,
        }
    )


def replay_pointer_offsets(config: Path) -> list[int]:
    data = config.read_bytes()
    if len(data) != 296 or data[:7] != b"NSPWIN1":
        raise RuntimeError(f"PWin replay config drift: {config}")
    count = struct.unpack_from("<I", data, 164)[0]
    if count > 8:
        raise RuntimeError(f"PWin relocation count invalid: {config}")
    return [struct.unpack_from("<I", data, 168 + index * 16)[0] for index in range(count)]


def main() -> int:
    if git_commit(AUTHORITY) != AUTHORITY_COMMIT:
        raise RuntimeError("exact-SR authority commit drift")
    if git_commit(CUMETAL) != CUMETAL_COMMIT:
        raise RuntimeError("CuMetal implementation commit drift")
    if OUTPUT.exists():
        raise RuntimeError(f"refusing to overwrite existing bundle: {OUTPUT}")

    enc0 = AUTHORITY / "work" / "dlss-sr-pwin-enc0-replay-v2"
    family = AUTHORITY / "work" / "dlss-sr-pwin-family-live"
    math = AUTHORITY / "work" / "dlss-sr-sm120-math-luts-v1"
    boundary = AUTHORITY / "work" / "dlss-sr-boundary-live"
    libraries = CUMETAL / "build"
    depth_inverted = REPOSITORY / "work" / "depth-inverted-boundary"
    live_converter = REPOSITORY / "work" / "live-texture-converter"
    rows: list[dict[str, object]] = []

    copy(enc0 / "allocation-9-weights.bin", Path("model/weights.bin"), rows)
    if rows[-1]["sha256"] != WEIGHTS_SHA256:
        raise RuntimeError("exact model weight identity mismatch")
    write_sanitized(
        enc0 / "arguments-template.bin",
        Path("pwin/enc0-arguments-template.bin"),
        [8, 24, 32, 48, 64],
        176,
        rows,
    )
    for position in range(2, 12):
        source = family / f"pwin-{position:02d}"
        target = Path("pwin") / f"pwin-{position:02d}"
        copy(source / "replay-config.bin", target / "replay-config.bin", rows)
        write_sanitized(
            source / "arguments-template.bin",
            target / "arguments-template.bin",
            replay_pointer_offsets(source / "replay-config.bin"),
            176,
            rows,
        )

    input_fixture = boundary / "hiluma-input-fixture"
    output_fixture = boundary / "hiluma-output-fixture"
    write_sanitized(
        input_fixture / "arguments-0001.bin",
        Path("abi/hiluma-input-reset-template.bin"),
        [168, 176, 184, 192, 200, 208, 216, 224, 240],
        248,
        rows,
    )
    write_sanitized(
        output_fixture / "arguments-0013.bin",
        Path("abi/hiluma-output-reset-template.bin"),
        [200, 224, 232, 240, 248, 256, 264, 272, 280, 288, 296, 304, 312],
        328,
        rows,
    )
    write_sanitized(
        output_fixture / "arguments-0014.bin",
        Path("abi/exposure-reset-template.bin"),
        [0, 8],
        16,
        rows,
    )

    copy(
        boundary / "hiluma-output-fixture" / "allocation-3-constants.bin",
        Path("model/constants.bin"),
        rows,
    )
    copy(math / "ex2-f16x2-lut.bin", Path("math/ex2-f16x2-lut.bin"), rows)
    copy(math / "rcp-f16-to-f16-lut.bin", Path("math/rcp-f16-to-f16-lut.bin"), rows)
    copy(
        boundary / "architecture-roms" / "rsq" / "correction-rom.bin",
        Path("math/rsq-correction-rom.bin"),
        rows,
    )
    copy(
        boundary / "architecture-roms" / "rcp" / "correction-rom.bin",
        Path("math/rcp-correction-rom.bin"),
        rows,
    )

    implementation_material = bytearray()
    for name in PIPELINES:
        source = libraries / f"{name}.metallib"
        copy(source, Path("metallib") / source.name, rows)
        implementation_material.extend(name.encode())
        implementation_material.append(0)
        implementation_material.extend(sha256(source).encode())
        implementation_material.extend(b"\n")
    actual_implementation_root = hashlib.sha256(implementation_material).hexdigest()
    if actual_implementation_root != IMPLEMENTATION_ROOT:
        raise RuntimeError("Metal implementation bundle identity mismatch")
    depth_rows_before = len(rows)
    copy(
        depth_inverted / "hiluma_input_depthinv.metallib",
        Path("metallib/hiluma_input_depthinv.metallib"),
        rows,
    )
    copy(
        depth_inverted / "hiluma_output_depthinv.metallib",
        Path("metallib/hiluma_output_depthinv.metallib"),
        rows,
    )
    depth_rows = rows[depth_rows_before:]
    if [row["sha256"] for row in depth_rows] != [
        DEPTH_INVERTED_INPUT_SHA256,
        DEPTH_INVERTED_OUTPUT_SHA256,
    ]:
        raise RuntimeError("depth-inverted Metal boundary identity mismatch")
    depth_inverted_root = canonical_root(depth_rows)
    copy(
        live_converter / "dlss_sr_live_texture_convert.metallib",
        Path("metallib/dlss_sr_live_texture_convert.metallib"),
        rows,
    )
    if rows[-1]["sha256"] != LIVE_TEXTURE_CONVERTER_SHA256:
        raise RuntimeError("live texture converter identity mismatch")

    forbidden = [row["path"] for row in rows if "expected" in str(row["path"]).lower()]
    if forbidden:
        raise RuntimeError(f"oracle comparison artifacts entered runtime bundle: {forbidden}")

    identity = {
        "schema": "neuralstudio.re9.exact_sr_immutable_runtime_bundle.v1",
        "authority_commit": AUTHORITY_COMMIT,
        "cumetal_commit": CUMETAL_COMMIT,
        "model": "DLSS_SR_Feature1_Quality_K",
        "weights_sha256": WEIGHTS_SHA256,
        "implementation_root_sha256": IMPLEMENTATION_ROOT,
        "depth_inverted_boundary_root_sha256": depth_inverted_root,
        "live_texture_converter_sha256": LIVE_TEXTURE_CONVERTER_SHA256,
        "files": sorted(rows, key=lambda row: str(row["path"])),
        "provenance": {
            "included": [
                "immutable_model_weight",
                "compile_time_shape_layout_constant",
                "sanitized_reset_abi_template",
                "architecture_math_contract",
                "verified_metal_implementation",
                "rtx_mapped_depth_inverted_boundary",
                "live_metal_texture_format_converter",
            ],
            "excluded": [
                "RTX_expected_output",
                "captured_intermediate",
                "captured_frame_input",
                "captured_history_state",
                "comparison_fixture",
            ],
        },
        "claims": {
            "bundle_identity_proven": True,
            "live_texture_contract_proven": False,
            "in_game_model_execution_proven": False,
        },
    }
    identity["bundle_root_sha256"] = canonical_root(identity)
    manifest = OUTPUT / "manifest.json"
    manifest.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"result": "pass", "output": str(OUTPUT), **identity["claims"],
                      "files": len(rows), "bytes": sum(int(row["bytes"]) for row in rows),
                      "bundle_root_sha256": identity["bundle_root_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
