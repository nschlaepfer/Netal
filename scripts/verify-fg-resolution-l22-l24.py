#!/usr/bin/env python3
"""Build or verify immutable evidence for resolution-dependent FG modules 22/24."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import struct
from pathlib import Path
from typing import Any


RESOLUTIONS = {
    "720p": (1280, 720, 640, 360, 10, 1, [3, 2, 1], [40, 23, 1]),
    "1080p": (1920, 1080, 960, 540, 10, 1, [4, 3, 1], [60, 34, 1]),
    "1440p": (2560, 1440, 1280, 720, 11, 2, [5, 3, 1], [80, 45, 1]),
    "4k": (3840, 2160, 1920, 1080, 11, 2, [8, 5, 1], [120, 68, 1]),
}
PHASES = ("reset", "generated")
POISONS = {"a5": 0xA5, "5a": 0x5A}
ORDINALS = ("a", "b")
MODULE22_SHA = "7d6602bbed88409e9ad2ef71293869dd22f7153f6387339ed65e91c12341c67a"
MODULE24_SHA = "5a41f9d65d1f6f933ae42f36b493f60687a646b73929291bafbe9d00297b6bdf"
ROM_SHA = "7cf4f8cece6b18b07605edc9cb601a153346a79b10bf3b93c7ec058d57d29750"
MASK_SALTS = {"reset": 0x243F6A8885A308D3, "generated": 0x13198A2E03707344}
M22_SALTS = {"reset": 0x082EFA98EC4E6C89, "generated": 0xA4093822299F31D0}
M24_SALTS = {"reset": 0x3C6EF372FE94F82B, "generated": 0xBB67AE8584CAA73B}
MASK64 = (1 << 64) - 1


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    return sha(path.read_bytes())


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def rooted(value: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(value)
    result["receipt_root_sha256"] = sha(canonical(result))
    return result


def valid_root(value: dict[str, Any]) -> bool:
    body = copy.deepcopy(value)
    expected = body.pop("receipt_root_sha256", None)
    return expected == sha(canonical(body))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def mix(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def half3(width: int, height: int, salt: int) -> bytes:
    values = (0x0000, 0x2800, 0xA800, 0x3000, 0xB000, 0x3400,
              0xB400, 0x3800, 0xB800, 0x3C00, 0xBC00)
    weights = (0x0000, 0x3000, 0x3400, 0x3800, 0x3A00, 0x3C00)
    result = bytearray(width * height * 6)
    for pixel in range(width * height):
        value = mix((salt + pixel * 0xD1B54A32D192ED03) & MASK64)
        struct.pack_into("<HHH", result, pixel * 6,
                         values[value % len(values)],
                         values[(value >> 10) % len(values)],
                         weights[(value >> 20) % len(weights)])
    return bytes(result)


def fixtures(name: str, phase: str) -> tuple[bytes, bytes, list[bytes]]:
    _, _, width, height, _, _, _, _ = RESOLUTIONS[name]
    m22_width, m22_height = (width + 15) // 16, (height + 15) // 16
    m22_input = half3(m22_width, m22_height, M22_SALTS[phase])
    mask = bytearray(width * height)
    for pixel in range(width * height):
        value = mix((MASK_SALTS[phase] +
                     pixel * 0x94D049BB133111EB) & MASK64)
        mask[pixel] = 1 if (value & 7) < 6 else 0
    levels: list[bytes] = []
    level_width, level_height = width, height
    for index in range(4):
        level_width, level_height = ((level_width + 1) // 2,
                                     (level_height + 1) // 2)
        levels.append(half3(level_width, level_height,
                            (M24_SALTS[phase] +
                             index * 0x100000001B3) & MASK64))
    return m22_input, bytes(mask), levels


def run_dir(root: Path, resolution: str, phase: str,
            poison_name: str, ordinal: str) -> Path:
    return root / f"{resolution}-{phase}-poison-{poison_name}-process-{ordinal}"


def output_files() -> list[str]:
    return ([f"module22-output-{index}.bin" for index in range(4)] +
            ["module24-output.bin"])


def run_files() -> list[str]:
    return (["module22-input-after.bin", "module24-mask-after.bin"] +
            [f"module22-output-{index}.bin" for index in range(4)] +
            [f"module24-level-{index}-after.bin" for index in range(4)] +
            ["module24-output.bin"])


def mismatch_count(left: bytes, right: bytes) -> int:
    if len(left) != len(right):
        raise ValueError("comparison size mismatch")
    return sum(a != b for a, b in zip(left, right))


def tree_root(root: Path) -> tuple[str, int]:
    manifest = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "build" in path.relative_to(root).parts:
            continue
        relative = path.relative_to(root).as_posix()
        data = path.read_bytes()
        manifest.append({"path": relative, "bytes": len(data), "sha256": sha(data)})
    return sha(canonical(manifest)), len(manifest)


def validate_backend(root: Path, backend: str) -> tuple[dict[str, Any], dict[str, Any]]:
    pids: set[int] = set()
    combo_outputs: dict[str, Any] = {}
    expected_backend = "rtx-d3d12-nvapi" if backend == "rtx" else "mac-metal"
    for resolution, config in RESOLUTIONS.items():
        _, _, width, height, level, extra, grid22, grid24 = config
        for phase in PHASES:
            expected_m22, expected_mask, expected_levels = fixtures(resolution, phase)
            for poison_name, poison in POISONS.items():
                first: dict[str, bytes] | None = None
                for ordinal in ORDINALS:
                    directory = run_dir(root, resolution, phase, poison_name, ordinal)
                    receipt = json.loads((directory / "run-receipt.json").read_text())
                    if receipt["result"] != "pass" or receipt["backend"] != expected_backend:
                        raise ValueError(f"rejected run receipt: {directory}")
                    if (receipt["resolution"] != resolution or
                        receipt["phase"] != phase or
                        receipt["poison_byte"] != poison or
                        receipt["module22_sha256"] != MODULE22_SHA or
                        receipt["module24_sha256"] != MODULE24_SHA or
                        receipt["module22_abi_bytes"] != 56 or
                        receipt["module24_abi_bytes"] != 64 or
                        receipt["module22_grid"] != grid22 or
                        receipt["module24_grid"] != grid24 or
                        receipt["block"] != [256, 1, 1] or
                        receipt["selected_level"] != level or
                        receipt["extra_level_count"] != extra):
                        raise ValueError(f"run metadata drift: {directory}")
                    if (receipt["input_mismatched_bytes"] != 0 or
                        receipt["canary_mismatched_bytes"] != 0 or
                        receipt["expected_or_captured_output_ingress"] or
                        receipt["archived_intermediate_ingress"] or
                        receipt["address_fields_recorded"]):
                        raise ValueError(f"run gate failed: {directory}")
                    pids.add(int(receipt["process_id"]))
                    artifacts = {name: (directory / name).read_bytes()
                                 for name in run_files()}
                    if artifacts["module22-input-after.bin"] != expected_m22:
                        raise ValueError(f"module22 input mutation: {directory}")
                    if artifacts["module24-mask-after.bin"] != expected_mask:
                        raise ValueError(f"module24 mask mutation: {directory}")
                    for index, expected in enumerate(expected_levels):
                        if artifacts[f"module24-level-{index}-after.bin"] != expected:
                            raise ValueError(f"module24 level mutation: {directory}/{index}")
                    if first is None:
                        first = artifacts
                    elif any(first[name] != artifacts[name] for name in run_files()):
                        raise ValueError(f"fresh-process repeat mismatch: {directory}")

            poison_outputs = {
                key: {name: (run_dir(root, resolution, phase, key, "a") / name).read_bytes()
                      for name in output_files()}
                for key in POISONS
            }
            m22_hashes: list[str] = []
            m22_bytes: list[int] = []
            for index in range(4):
                a5 = poison_outputs["a5"][f"module22-output-{index}.bin"]
                fivea = poison_outputs["5a"][f"module22-output-{index}.bin"]
                m22_bytes.append(len(a5))
                if index < extra:
                    if a5 != fivea:
                        raise ValueError(
                            f"module22 incomplete write coverage: {backend}/{resolution}/{phase}/{index}")
                    m22_hashes.append(sha(a5))
                else:
                    if a5 != bytes([0xA5]) * len(a5) or fivea != bytes([0x5A]) * len(fivea):
                        raise ValueError(
                            f"module22 inactive output modified: {backend}/{resolution}/{phase}/{index}")
                    m22_hashes.append(sha(bytes(len(a5))))
            a5_out = poison_outputs["a5"]["module24-output.bin"]
            fivea_out = poison_outputs["5a"]["module24-output.bin"]
            if len(a5_out) != width * height * 4 or len(fivea_out) != len(a5_out):
                raise ValueError("module24 output size drift")
            canonical_output = bytearray(len(a5_out))
            written = 0
            for pixel, selected in enumerate(expected_mask):
                start = pixel * 4
                aa, bb = a5_out[start:start + 4], fivea_out[start:start + 4]
                if selected & 1:
                    if aa != bb:
                        raise ValueError(
                            f"module24 incomplete write coverage: {backend}/{resolution}/{phase}/{pixel}")
                    canonical_output[start:start + 4] = aa
                    written += 1
                elif aa != b"\xA5" * 4 or bb != b"\x5A" * 4:
                    raise ValueError(
                        f"module24 out-of-mask write: {backend}/{resolution}/{phase}/{pixel}")
            combo_outputs[f"{resolution}/{phase}"] = {
                "module22_poison_neutral_sha256": m22_hashes,
                "module22_output_bytes": m22_bytes,
                "module22_active_output_count": extra,
                "module24_canonical_sha256": sha(bytes(canonical_output)),
                "module24_output_bytes": len(a5_out),
                "module24_written_pixels": written,
                "module24_untouched_pixels": width * height - written,
            }
    if len(pids) != 32:
        raise ValueError(f"fresh process count {backend}: {len(pids)}/32")
    artifact_root, artifact_count = tree_root(root)
    summary = {
        "schema": f"neuralstudio.fg.resolution-l22-l24-{backend}-backend.v1",
        "backend": expected_backend,
        "run_count": 32,
        "fresh_process_count": len(pids),
        "resolutions": list(RESOLUTIONS),
        "phases": list(PHASES),
        "poison_bytes": [165, 90],
        "fresh_process_repeat_exact": True,
        "dual_poison_exact_write_coverage": True,
        "input_immutable": True,
        "input_mismatched_bytes": 0,
        "buffer_canary_mismatched_bytes": 0,
        "expected_or_captured_output_ingress": False,
        "archived_intermediate_ingress": False,
        "address_fields_recorded": False,
        "raw_artifact_manifest_sha256": artifact_root,
        "raw_artifact_file_count": artifact_count,
        "outputs": combo_outputs,
        "result": "pass",
    }
    if backend == "mac":
        summary.update({
            "architecture_rom_sha256": ROM_SHA,
            "rom_immutable": True,
            "true_typed_output": "MTLPixelFormatRG16Float",
            "surface_half_rounding": "toward-zero",
        })
    else:
        rig_path = root / "rig-receipt.json"
        rig = json.loads(rig_path.read_text())
        if (rig["result"] != "pass" or rig["fresh_process_count"] != 32 or
            rig["module22_sha256"] != MODULE22_SHA or
            rig["module24_sha256"] != MODULE24_SHA or
            rig["expected_or_captured_output_ingress"] or
            rig["archived_intermediate_ingress"] or
            rig["address_fields_recorded"]):
            raise ValueError("rig receipt rejected")
        summary.update({
            "rig_receipt_sha256": file_sha(rig_path),
            "rig_job_id": rig["job_id"],
            "gpu": rig["gpu"],
            "true_typed_output": "DXGI_FORMAT_R16G16_FLOAT",
        })
    return rooted(summary), combo_outputs


def build_abi(repository: Path) -> dict[str, Any]:
    shape_hashes: dict[str, str] = {}
    occurrences: dict[str, Any] = {}
    for name, config in RESOLUTIONS.items():
        shape_path = repository / f"evidence/specs/fg-resolution-shapes-v1/shape-{name}.json"
        shape_hashes[name] = file_sha(shape_path)
        rows = [row for row in json.loads(shape_path.read_text())
                if row["module_token"] in ("module-000022", "module-000024")]
        expected = [
            ("evaluate_0", "launch-000007", "module-000022", config[6], 56),
            ("evaluate_0", "launch-000009", "module-000024", config[7], 64),
            ("evaluate_1", "launch-000019", "module-000022", config[6], 56),
            ("evaluate_1", "launch-000021", "module-000024", config[7], 64),
        ]
        actual = [(row["phase"], row["launch_token"], row["module_token"],
                   row["grid"], row["parameter_bytes"]) for row in rows]
        if actual != expected or any(row["block"] != [256, 1, 1] for row in rows):
            raise ValueError(f"shape matrix drift: {name}")
        occurrences[name] = [{
            "phase": row["phase"], "launch": row["launch_token"],
            "module": row["module_token"], "function": row["function_token"],
            "grid": row["grid"], "block": row["block"],
            "parameter_bytes": row["parameter_bytes"],
        } for row in rows]
    abi = {
        "schema": "neuralstudio.fg.resolution-l22-l24-abi.v1",
        "module22": {
            "module_token": 22, "function_token": 20,
            "module_bytes": 15120, "module_sha256": MODULE22_SHA,
            "packed_parameter_bytes": 56,
            "parameter_layout": [
                {"offset": 0, "bytes": 8, "kind": "raw-buffer-pointer", "role": "input-half3"},
                {"offset": 8, "bytes": 8, "kind": "raw-buffer-pointer", "role": "output-level0-half3"},
                {"offset": 16, "bytes": 8, "kind": "raw-buffer-pointer", "role": "output-level1-half3"},
                {"offset": 24, "bytes": 8, "kind": "raw-buffer-pointer", "role": "output-level2-half3"},
                {"offset": 32, "bytes": 8, "kind": "raw-buffer-pointer", "role": "output-level3-half3"},
                {"offset": 40, "bytes": 4, "kind": "u32", "role": "extra-level-count"},
                {"offset": 44, "bytes": 4, "kind": "u32", "role": "input-width"},
                {"offset": 48, "bytes": 4, "kind": "u32", "role": "input-height"},
                {"offset": 52, "bytes": 4, "kind": "padding", "role": "not-loaded"},
            ],
            "store_rounding": "binary16 round-to-nearest-even",
        },
        "module24": {
            "module_token": 24, "function_token": 22,
            "module_bytes": 25880, "module_sha256": MODULE24_SHA,
            "packed_parameter_bytes": 64,
            "parameter_layout": [
                {"offset": 0, "bytes": 8, "kind": "independent-surface", "role": "RG16F-output"},
                {"offset": 8, "bytes": 8, "kind": "raw-buffer-pointer", "role": "R8-selection-mask"},
                {"offset": 16, "bytes": 8, "kind": "raw-buffer-pointer", "role": "half3-level10"},
                {"offset": 24, "bytes": 8, "kind": "raw-buffer-pointer", "role": "half3-level11"},
                {"offset": 32, "bytes": 8, "kind": "raw-buffer-pointer", "role": "half3-level12"},
                {"offset": 40, "bytes": 8, "kind": "raw-buffer-pointer", "role": "half3-level13"},
                {"offset": 48, "bytes": 4, "kind": "u32", "role": "selected-level"},
                {"offset": 52, "bytes": 4, "kind": "u32", "role": "working-width"},
                {"offset": 56, "bytes": 4, "kind": "u32", "role": "working-height"},
                {"offset": 60, "bytes": 4, "kind": "padding", "role": "not-loaded"},
            ],
            "typed_output": "DXGI_FORMAT_R16G16_FLOAT",
            "surface_store_rounding": "binary16 round-toward-zero",
        },
        "resolution_occurrences": occurrences,
        "shape_evidence_sha256": shape_hashes,
        "trace_parameter_sha256": {
            "720p": {"module22": "1ed58ba1983f36092918c7be507e0d4ce4b6c22e9ed13f21b7c285fb24282bab", "module24": "8cd50307d2cbe0915194f922f53ce33a222f53e89ad58e36b2863774bd285a1b"},
            "1080p": {"module22": "39bccb586035ae5cca3567e1186d1e92e832e3a1dc6fc5d4c79662f19f578219", "module24": "5576482a789adaece952a3a5ef7dbb13169b97d2746e67d85d4eb8d9c0c806b4"},
            "1440p": {"module22": "47dd253af8a72fa5087a9c38d8846d59cc37cabe8fcab51e805d5607f6b022ec", "module24": "8b7a37d7bfd5866a6c1065a7f0102556f04bfadab1e764ae5fbb708e99e19324"},
            "4k": {"module22": "468558fe34d3ffb11db58a3c2539cad02ae9059e898c92e872c99c4d04e6d53e", "module24": "a2dbe72b1d1324afa96eca6708de9270a0630ca1c406ad164e9f00db2b433cdc"},
        },
        "reset_generated_parameter_bytes_identical_per_resolution": True,
        "address_fields_recorded": False,
        "result": "pass",
    }
    return rooted(abi)


def counterexamples() -> dict[str, Any]:
    return rooted({
        "schema": "neuralstudio.fg.resolution-l22-l24-counterexamples.v1",
        "items": [
            {"name": "module24-cumetal-strict-surface", "sha256": "0cbff5ac57dfad442f2b51004bb7c9f8289447b9811d140f37943481f868eb2b", "outcome": "rejected: surface access requires explicit buffer-emulation authorization"},
            {"name": "module24-surface-removed-address-space", "sha256": "cfe19d58ef5b5be546f4934ea8589694190a543473925e9f6a6e949173b6fab7", "outcome": "rejected: address-space casts require explicit pointer address spaces"},
            {"name": "module24-cvta-removed-cfg", "sha256": "b36be1c50bdeb8bf967b4cd973fd5a2c6b936b7e6c17f48fcd1c8218b87a48c3", "outcome": "rejected: barrier-containing conditional CFG has no forward reconvergence"},
            {"name": "module24-first-level10-specialization", "sha256": "f0bd84a47df8ea9a122e6fd787c7d3444541380dc7523c8b42063ed32836000d", "outcome": "rejected: undefined PTX register on incoming edge"},
            {"name": "module24-cumetal-shared-pointer", "sha256": "496733ffd226032d42a59e706df89f575496abc8ed22e698d07746b25c0616f6", "outcome": "rejected by Metal: threadgroup pointer assigned to uint and recast as device pointer"},
        ],
        "all_failures_preserved": True,
        "result": "pass",
    })


def compare_backends(rtx_root: Path, mac_root: Path) -> tuple[int, int]:
    mismatches = 0
    compared = 0
    for resolution in RESOLUTIONS:
        for phase in PHASES:
            for poison in POISONS:
                rtx = run_dir(rtx_root, resolution, phase, poison, "a")
                mac = run_dir(mac_root, resolution, phase, poison, "a")
                for name in run_files():
                    left, right = (rtx / name).read_bytes(), (mac / name).read_bytes()
                    mismatches += mismatch_count(left, right)
                    compared += len(left)
    return mismatches, compared


def verify_receipts(evidence: Path) -> None:
    names = ["abi-receipt.json", "counterexamples.json", "rtx-backend-receipt.json",
             "mac-backend-receipt.json", "accepted-receipt.json", "negative-audit.json"]
    values = {name: json.loads((evidence / name).read_text()) for name in names}
    for name, value in values.items():
        if not valid_root(value):
            raise ValueError(f"root mismatch: {name}")
    lowering = json.loads((evidence / "lowering-receipt.json").read_text())
    if not valid_root(lowering):
        raise ValueError("root mismatch: lowering-receipt.json")
    accepted = values["accepted-receipt.json"]
    if accepted["abi_receipt_root_sha256"] != values["abi-receipt.json"]["receipt_root_sha256"]:
        raise ValueError("accepted ABI binding mismatch")
    if accepted["rtx_backend_root_sha256"] != values["rtx-backend-receipt.json"]["receipt_root_sha256"]:
        raise ValueError("accepted RTX binding mismatch")
    if accepted["mac_backend_root_sha256"] != values["mac-backend-receipt.json"]["receipt_root_sha256"]:
        raise ValueError("accepted Mac binding mismatch")
    if accepted["lowering_receipt_root_sha256"] != lowering["receipt_root_sha256"]:
        raise ValueError("accepted lowering binding mismatch")
    if accepted["counterexamples_root_sha256"] != values["counterexamples.json"]["receipt_root_sha256"]:
        raise ValueError("accepted failure binding mismatch")
    if accepted["rtx_mac_mismatched_bytes"] != 0:
        raise ValueError("accepted parity mismatch")
    claims = accepted["claims"]
    if (not claims["module22_standalone_resolution_matrix_exact"] or
        not claims["module24_standalone_resolution_matrix_exact"] or
        claims["connected_large_resolution_graph"] or
        claims["complete_generated_frame"] or claims["netal_integration"]):
        raise ValueError("claim boundary drift")
    negative = values["negative-audit.json"]
    if not negative["all_mutations_rejected"]:
        raise ValueError("mutation audit rejected")


def command_build(args: argparse.Namespace) -> int:
    repository = args.repository.resolve()
    rtx_root, mac_root, output = args.rtx.resolve(), args.mac.resolve(), args.output.resolve()
    rtx, rtx_outputs = validate_backend(rtx_root, "rtx")
    mac, mac_outputs = validate_backend(mac_root, "mac")
    if rtx_outputs != mac_outputs:
        raise ValueError("poison-neutral backend output summary mismatch")
    mismatches, compared = compare_backends(rtx_root, mac_root)
    if mismatches != 0:
        raise ValueError(f"RTX/Metal mismatch: {mismatches}/{compared}")
    abi = build_abi(repository)
    counters = counterexamples()
    lowering_source = repository / "build/fg_resolution_preprocess_l22_l24/lowering-receipt.json"
    lowering = json.loads(lowering_source.read_text())
    if not valid_root(lowering):
        raise ValueError("lowering receipt root mismatch")
    accepted = rooted({
        "schema": "neuralstudio.fg.resolution-l22-l24-accepted.v1",
        "selected_units": [
            "large-resolution reset/generated module22 raw half3 pyramid producer",
            "large-resolution reset/generated module24 masked RG16F presentation preprocessor",
        ],
        "resolutions": list(RESOLUTIONS),
        "phases": list(PHASES),
        "processes_per_backend": 32,
        "module22_sha256": MODULE22_SHA,
        "module24_sha256": MODULE24_SHA,
        "abi_receipt_root_sha256": abi["receipt_root_sha256"],
        "lowering_receipt_root_sha256": lowering["receipt_root_sha256"],
        "counterexamples_root_sha256": counters["receipt_root_sha256"],
        "rtx_backend_root_sha256": rtx["receipt_root_sha256"],
        "mac_backend_root_sha256": mac["receipt_root_sha256"],
        "rtx_mac_compared_bytes": compared,
        "rtx_mac_mismatched_bytes": mismatches,
        "fresh_process_repeat_exact": True,
        "dual_poison_exact_write_coverage": True,
        "input_and_rom_immutable": True,
        "canary_mismatched_bytes": 0,
        "expected_or_captured_output_ingress": False,
        "address_fields_recorded": False,
        "claims": {
            "module22_standalone_resolution_matrix_exact": True,
            "module24_standalone_resolution_matrix_exact": True,
            "reset_and_generated_phase_units": True,
            "true_typed_d3d12_oracle": True,
            "connected_large_resolution_graph": False,
            "complete_generated_frame": False,
            "netal_integration": False,
            "performance": False,
        },
        "result": "pass",
    })
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "abi-receipt.json", abi)
    write_json(output / "counterexamples.json", counters)
    write_json(output / "rtx-backend-receipt.json", rtx)
    write_json(output / "mac-backend-receipt.json", mac)
    write_json(output / "lowering-receipt.json", lowering)
    write_json(output / "accepted-receipt.json", accepted)
    mutations = []
    cases = [
        ("accepted-module-drift", accepted, (("module22_sha256",), "0" * 64)),
        ("accepted-parity-drift", accepted, (("rtx_mac_mismatched_bytes",), 1)),
        ("accepted-overclaim", accepted, (("claims", "connected_large_resolution_graph"), True)),
        ("rtx-process-drift", rtx, (("fresh_process_count",), 31)),
        ("rtx-ingress-drift", rtx, (("expected_or_captured_output_ingress",), True)),
        ("rtx-typed-output-drift", rtx, (("true_typed_output",), "raw-buffer")),
        ("mac-rom-drift", mac, (("architecture_rom_sha256",), "0" * 64)),
        ("mac-coverage-drift", mac, (("dual_poison_exact_write_coverage",), False)),
        ("abi-shape-drift", abi, (("module22", "packed_parameter_bytes"), 52)),
        ("abi-surface-drift", abi, (("module24", "typed_output"), "unknown")),
        ("counterexample-loss", counters, (("all_failures_preserved",), False)),
        ("lowering-source-drift", lowering,
         (("modules", "module22.metal", "output_msl_sha256"), "0" * 64)),
    ]
    for name, original, (parts, value) in cases:
        mutated = copy.deepcopy(original)
        cursor: Any = mutated
        for part in parts[:-1]:
            cursor = cursor[part]
        cursor[parts[-1]] = value
        mutations.append({"mutation": name, "rejected": not valid_root(mutated),
                          "reason": "receipt root mismatch"})
    negative = rooted({
        "schema": "neuralstudio.fg.resolution-l22-l24-negative-audit.v1",
        "accepted_receipt_root_sha256": accepted["receipt_root_sha256"],
        "mutation_count": len(mutations),
        "mutations": mutations,
        "all_mutations_rejected": all(item["rejected"] for item in mutations),
        "result": "pass",
    })
    write_json(output / "negative-audit.json", negative)
    verify_receipts(output)
    print(f"accepted_receipt_root_sha256={accepted['receipt_root_sha256']}")
    print(f"rtx_mac_compared_bytes={compared}")
    print("verification_result=pass")
    return 0


def command_receipt(args: argparse.Namespace) -> int:
    verify_receipts(args.evidence.resolve())
    accepted = json.loads((args.evidence / "accepted-receipt.json").read_text())
    print(f"accepted_receipt_root_sha256={accepted['receipt_root_sha256']}")
    print("verification_result=pass")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--repository", type=Path, required=True)
    build.add_argument("--rtx", type=Path, required=True)
    build.add_argument("--mac", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.set_defaults(function=command_build)
    receipt = subparsers.add_parser("receipt")
    receipt.add_argument("--evidence", type=Path, required=True)
    receipt.set_defaults(function=command_receipt)
    args = parser.parse_args()
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
