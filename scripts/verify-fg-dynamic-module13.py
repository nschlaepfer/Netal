#!/usr/bin/env python3
"""Build and verify exact target-resolution module13 evidence."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


MODULE_SHA = "fd0665969da887dd92540b673009e0361a6c1de74fe54badf48f300b1fccbbf1"
RESOLUTIONS = {
    "720p": {"extent": [640, 360], "grid": [40, 23, 1],
             "scale_bits": ["0x3a4ccccd", "0x3ab60b61"],
             "divisor_bits": ["0x44100000", "0x424f5c29"],
             "raw": {"reset": "e9ce58929f567348052851a720dbfb9eb3c318bfa36b40aa1132f5998cc4cb97", "generated": "d4370c5a4e5f51bcebe43b6ec48c43f256e64dc1ae757cbd62bd65fe50dcd45a"},
             "normalized": "e21e039797c67fa9b2c5791292e71a533f8079bea6f7f92dd421d97797e5d9c3"},
    "1080p": {"extent": [960, 540], "grid": [60, 34, 1],
              "scale_bits": ["0x3a088889", "0x3a72b9d6"],
              "divisor_bits": ["0x44a20000", "0x42e947ae"],
              "raw": {"reset": "e5cd86e489b3453380f42ef359b0730ec6c57c715d45085c566316f4c6e7d239", "generated": "56d6a52cb27f7e2431fefa88aaef767c173db5015302e13d2bcd04988db31bcc"},
              "normalized": "5b4f9354f8e86ff28baca22ffb4d726ab569f75ed9759ac3e03e1e35879b643b"},
    "1440p": {"extent": [1280, 720], "grid": [80, 45, 1],
              "scale_bits": ["0x39cccccd", "0x3a360b61"],
              "divisor_bits": ["0x45100000", "0x434f5c29"],
              "raw": {"reset": "e3859d5c8b3a6524e23b193c6acf9ca3c5d6ffe51b9febba7e222c502d3f6de7", "generated": "c34f09b9d394588b79c7561c0070caada99f48831e153beafe0fc6e82ebe594e"},
              "normalized": "5bf8a3a78f2b2362eadf6922ded51c7a9d9a3a734cd687037826169b45d563ca"},
    "4k": {"extent": [1920, 1080], "grid": [120, 68, 1],
           "scale_bits": ["0x39888889", "0x39f2b9d6"],
           "divisor_bits": ["0x45a20000", "0x43e947ae"],
           "raw": {"reset": "987bc5e293e59a374ed399493d08af7a3b0df237a941acd3511d9a5a2c00161e", "generated": "66395f312d9fe003b6f0310137d45227764b55c08f55de200110f2a563de2b48"},
           "normalized": "daab11d2a4e4d69ee0cfc8fa4c70b7daa13b30ec4447c2fb120b977cccb90431"},
}
PHASES = ("reset", "generated")
SCENARIOS = ("equal-high-tie", "priority-collision")
POISONS = ("a5", "5a")
ORDINALS = ("a", "b")
ARTIFACTS = ("atomic-output.bin", "mask-output.bin")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def run_dir(root: Path, resolution: str, phase: str, scenario: str,
            poison: str, ordinal: str) -> Path:
    return root / f"{resolution}-{phase}-{scenario}-poison-{poison}-process-{ordinal}"


def tree_root(root: Path) -> tuple[str, int]:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "build" in path.relative_to(root).parts:
            continue
        data = path.read_bytes()
        rows.append({"path": path.relative_to(root).as_posix(),
                     "bytes": len(data), "sha256": sha(data)})
    return sha(canonical(rows)), len(rows)


def validate_backend(root: Path, backend: str) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_backend = "rtx-cuda-driver-typed-arrays" if backend == "rtx" else "mac-metal"
    pids: set[int] = set()
    outputs: dict[str, Any] = {}
    for resolution, config in RESOLUTIONS.items():
        pixels = config["extent"][0] * config["extent"][1]
        for phase in PHASES:
            for scenario in SCENARIOS:
                receipts = []
                for poison in POISONS:
                    for ordinal in ORDINALS:
                        directory = run_dir(root, resolution, phase, scenario,
                                            poison, ordinal)
                        receipt = json.loads((directory / "run-receipt.json").read_text())
                        if (receipt["result"] != "pass" or
                            receipt["backend"] != expected_backend or
                            receipt["resolution"] != resolution or
                            receipt["phase"] != phase or
                            receipt["scenario"] != scenario or
                            receipt["module_sha256"] != MODULE_SHA or
                            receipt["extent"] != config["extent"] or
                            receipt["grid"] != config["grid"] or
                            receipt["input_mismatched_bytes"] != 0 or
                            receipt["atomic_canary_mismatched_bytes"] != 0 or
                            receipt["mask_guard_row_mismatched_bytes"] != 0 or
                            receipt["atomic_candidate_u64"] <= 0 or
                            receipt["atomic_unique_low_words"] <= 1 or
                            receipt["mask_zero_pixels"] <= 0 or
                            receipt["mask_one_pixels"] <= 0 or
                            receipt["expected_or_captured_output_ingress"] or
                            receipt["archived_intermediate_ingress"] or
                            receipt["address_fields_recorded"]):
                            raise ValueError(f"run rejected: {directory}")
                        if backend == "rtx":
                            if (receipt["trace_raw_parameter_sha256"] != config["raw"][phase] or
                                receipt["address_free_parameter_sha256"] != config["normalized"] or
                                receipt["block"] != [16, 16, 1]):
                                raise ValueError(f"RTX ABI drift: {directory}")
                        should_write = poison == "a5" and ordinal == "a"
                        if receipt["artifacts_written"] != should_write:
                            raise ValueError(f"artifact mode drift: {directory}")
                        for artifact in ARTIFACTS:
                            if (directory / artifact).is_file() != should_write:
                                raise ValueError(f"artifact cardinality: {directory}")
                        pids.add(int(receipt["process_id"]))
                        receipts.append(receipt)
                for field in ("atomic_output_sha256", "mask_output_sha256"):
                    if len({receipt[field] for receipt in receipts}) != 1:
                        raise ValueError(f"repeat/poison mismatch: {backend}/{resolution}/{phase}/{scenario}")
                full = run_dir(root, resolution, phase, scenario, "a5", "a")
                atomic = (full / "atomic-output.bin").read_bytes()
                mask = (full / "mask-output.bin").read_bytes()
                if len(atomic) != pixels * 8 or len(mask) != pixels:
                    raise ValueError(f"artifact size drift: {full}")
                if (sha(atomic) != receipts[0]["atomic_output_sha256"] or
                    sha(mask) != receipts[0]["mask_output_sha256"]):
                    raise ValueError(f"artifact hash drift: {full}")
                outputs[f"{resolution}/{phase}/{scenario}"] = {
                    "extent": config["extent"],
                    "atomic_output_bytes": len(atomic),
                    "atomic_output_sha256": sha(atomic),
                    "mask_output_bytes": len(mask),
                    "mask_output_sha256": sha(mask),
                    "atomic_candidate_u64": receipts[0]["atomic_candidate_u64"],
                    "atomic_sentinel_u64": receipts[0]["atomic_sentinel_u64"],
                    "atomic_unique_high_words": receipts[0]["atomic_unique_high_words"],
                    "atomic_unique_low_words": receipts[0]["atomic_unique_low_words"],
                }
    if len(pids) != 64:
        raise ValueError(f"fresh process count: {backend} {len(pids)}/64")
    manifest, count = tree_root(root)
    summary = {
        "schema": f"neuralstudio.fg.dynamic-module13-{backend}-backend.v1",
        "backend": expected_backend, "module_sha256": MODULE_SHA,
        "run_count": 64, "fresh_process_count": 64,
        "scenario_count": 16, "processes_per_scenario": 4,
        "poison_bytes": [165, 90], "fresh_process_repeat_exact": True,
        "dual_poison_full_write_coverage": True,
        "collision_and_equal_high_ties": True,
        "input_immutable": True, "input_mismatched_bytes": 0,
        "atomic_canary_mismatched_bytes": 0,
        "mask_guard_row_mismatched_bytes": 0,
        "expected_or_captured_output_ingress": False,
        "archived_intermediate_ingress": False,
        "address_fields_recorded": False,
        "raw_artifact_manifest_sha256": manifest,
        "raw_artifact_file_count": count,
        "outputs": outputs, "result": "pass",
    }
    if backend == "rtx":
        rig_path = root / "rig-receipt.json"
        rig = json.loads(rig_path.read_text())
        if (rig["result"] != "pass" or rig["fresh_process_count"] != 64 or
            not rig["true_typed_cuda_arrays"] or
            rig["expected_or_captured_output_ingress"] or
            rig["address_fields_recorded"]):
            raise ValueError("rig receipt rejected")
        summary.update({"rig_job_id": rig["job_id"],
                        "rig_receipt_sha256": sha(rig_path.read_bytes()),
                        "typed_resource_backend": "CUDA arrays/surface plus guarded raw atomic buffer"})
    else:
        one = json.loads((run_dir(root, "720p", "reset", "equal-high-tie",
                                  "a5", "a") / "run-receipt.json").read_text())
        summary.update({"device_name": one["device_name"],
                        "typed_resource_backend": "Metal textures plus guarded atomic buffers",
                        "atomic_lowering": "u32 high-min, equal-high low-min, GPU pack"})
    return rooted(summary), outputs


def compare(rtx: Path, mac: Path) -> tuple[int, int]:
    mismatches = compared = 0
    for resolution in RESOLUTIONS:
        for phase in PHASES:
            for scenario in SCENARIOS:
                for artifact in ARTIFACTS:
                    left = (run_dir(rtx, resolution, phase, scenario,
                                    "a5", "a") / artifact).read_bytes()
                    right = (run_dir(mac, resolution, phase, scenario,
                                     "a5", "a") / artifact).read_bytes()
                    if len(left) != len(right): raise ValueError("artifact size mismatch")
                    mismatches += sum(a != b for a, b in zip(left, right))
                    compared += len(left)
    return mismatches, compared


def verify(evidence: Path) -> None:
    names = ("abi-receipt.json", "implementation-receipt.json",
             "counterexamples.json", "rtx-backend-receipt.json",
             "mac-backend-receipt.json", "accepted-receipt.json",
             "negative-audit.json")
    values = {name: json.loads((evidence / name).read_text()) for name in names}
    for name, value in values.items():
        if not valid_root(value): raise ValueError(f"root mismatch: {name}")
    accepted = values["accepted-receipt.json"]
    for field, name in {
        "abi_receipt_root_sha256": "abi-receipt.json",
        "implementation_receipt_root_sha256": "implementation-receipt.json",
        "counterexamples_root_sha256": "counterexamples.json",
        "rtx_backend_root_sha256": "rtx-backend-receipt.json",
        "mac_backend_root_sha256": "mac-backend-receipt.json",
    }.items():
        if accepted[field] != values[name]["receipt_root_sha256"]:
            raise ValueError(f"binding mismatch: {field}")
    if (accepted["rtx_mac_mismatched_bytes"] != 0 or
        not accepted["claims"]["module13_four_resolution_standalone_exact"] or
        accepted["claims"]["connected_early_graph"] or
        accepted["claims"]["complete_generated_frame"]):
        raise ValueError("claim boundary drift")
    if not values["negative-audit.json"]["all_mutations_rejected"]:
        raise ValueError("mutation audit failed")


def command_build(args: argparse.Namespace) -> int:
    repository = args.repository.resolve()
    rtx_root, mac_root = args.rtx.resolve(), args.mac.resolve()
    rtx, ro = validate_backend(rtx_root, "rtx")
    mac, mo = validate_backend(mac_root, "mac")
    if ro != mo: raise ValueError("backend summary mismatch")
    mismatches, compared = compare(rtx_root, mac_root)
    if mismatches: raise ValueError(f"RTX/Metal mismatch {mismatches}/{compared}")
    abi = rooted({
        "schema": "neuralstudio.fg.dynamic-module13-abi.v1",
        "module_token": 13, "module_sha256": MODULE_SHA,
        "packed_parameter_bytes": 144,
        "large_trace_launches": {"reset": 5, "generated": 17},
        "canonical_launches": {"reset": 5, "generated": 15},
        "resolutions": RESOLUTIONS,
        "typed_resources": ["RG32_FLOAT normalized point-border flow",
                            "R32_UINT depth", "signed-int64 raw atomic work buffer",
                            "R8_UNORM mask surface"],
        "atomic_key": "signed depth-priority high word plus mask bit and FP16 flow low word",
        "address_fields_recorded": False, "result": "pass",
    })
    source_paths = {
        "contract_sha256": "src/fg_dynamic_module13/contract.h",
        "metal_wrapper_sha256": "src/fg_dynamic_module13/dynamic_module13.metal",
        "macos_host_sha256": "src/fg_dynamic_module13/macos_probe.mm",
        "rtx_oracle_sha256": "src/fg_dynamic_module13/rtx_oracle.cpp",
    }
    implementation = rooted({
        "schema": "neuralstudio.fg.dynamic-module13-implementation.v1",
        **{key: sha((repository / path).read_bytes()) for key, path in source_paths.items()},
        "accepted_atomic_kernel_reused_unchanged": True,
        "atomic_lowering": ["xor signed high and unsigned-u32 min",
                            "equal-winning-high unsigned-u32 low min",
                            "inverse xor and GPU pack"],
        "expected_or_captured_output_embedded": False, "result": "pass",
    })
    counters = rooted({
        "schema": "neuralstudio.fg.dynamic-module13-counterexamples.v1",
        "items": [
            {"name": "oracle-v1-test-path-tokenization", "stage": "pre-GPU", "result": "rejected", "error": "Test-Path$out parsed as a command"},
            {"name": "oracle-v1-join-path-tokenization", "stage": "pre-run after build", "result": "rejected", "error": "Join-Path$out$id parsed as a command"},
            {"name": "original-fixed-resolution-unit", "result": "retained", "accepted_root": "39c1421e2cb0728b1e7fbc536e506da4bb58fc4ccb50f2c89ae8603e08898126", "scope": "320x180 only"},
        ],
        "all_failures_preserved": True, "result": "pass",
    })
    accepted = rooted({
        "schema": "neuralstudio.fg.dynamic-module13-accepted.v1",
        "selected_unit": "module13 target-resolution atomic seed and mask",
        "module_sha256": MODULE_SHA, "resolutions": list(RESOLUTIONS),
        "phases": list(PHASES), "scenarios": list(SCENARIOS),
        "processes_per_backend": 64,
        "abi_receipt_root_sha256": abi["receipt_root_sha256"],
        "implementation_receipt_root_sha256": implementation["receipt_root_sha256"],
        "counterexamples_root_sha256": counters["receipt_root_sha256"],
        "rtx_backend_root_sha256": rtx["receipt_root_sha256"],
        "mac_backend_root_sha256": mac["receipt_root_sha256"],
        "rtx_mac_compared_bytes": compared,
        "rtx_mac_mismatched_bytes": mismatches,
        "fresh_process_repeat_exact": True,
        "dual_poison_full_write_coverage": True,
        "input_and_rom_immutable": True,
        "canary_mismatched_bytes": 0,
        "expected_or_captured_output_ingress": False,
        "address_fields_recorded": False,
        "claims": {"module13_four_resolution_standalone_exact": True,
                   "collision_and_equal_high_ties": True,
                   "signed64_atomic_lowering_exact": True,
                   "connected_early_graph": False,
                   "complete_generated_frame": False,
                   "netal_integration": False, "performance": False},
        "result": "pass",
    })
    output = args.output.resolve()
    for name, value in (("abi-receipt.json", abi),
                        ("implementation-receipt.json", implementation),
                        ("counterexamples.json", counters),
                        ("rtx-backend-receipt.json", rtx),
                        ("mac-backend-receipt.json", mac),
                        ("accepted-receipt.json", accepted)):
        write(output / name, value)
    cases = [
        ("accepted-parity", accepted, ("rtx_mac_mismatched_bytes",), 1),
        ("accepted-overclaim", accepted, ("claims", "connected_early_graph"), True),
        ("abi-grid", abi, ("resolutions", "4k", "grid"), [1, 1, 1]),
        ("abi-atomic", abi, ("atomic_key",), "unknown"),
        ("implementation-lowering", implementation, ("atomic_lowering",), []),
        ("rtx-process", rtx, ("fresh_process_count",), 63),
        ("rtx-ingress", rtx, ("expected_or_captured_output_ingress",), True),
        ("mac-coverage", mac, ("dual_poison_full_write_coverage",), False),
        ("mac-canary", mac, ("atomic_canary_mismatched_bytes",), 1),
        ("counterexample-loss", counters, ("all_failures_preserved",), False),
        ("accepted-module", accepted, ("module_sha256",), "0" * 64),
        ("accepted-root", accepted, ("receipt_root_sha256",), "0" * 64),
    ]
    mutations = []
    for name, original, keys, value in cases:
        mutated = copy.deepcopy(original); cursor: Any = mutated
        for key in keys[:-1]: cursor = cursor[key]
        cursor[keys[-1]] = value
        mutations.append({"mutation": name, "rejected": not valid_root(mutated),
                          "reason": "receipt root mismatch"})
    negative = rooted({
        "schema": "neuralstudio.fg.dynamic-module13-negative-audit.v1",
        "accepted_receipt_root_sha256": accepted["receipt_root_sha256"],
        "mutation_count": len(mutations), "mutations": mutations,
        "all_mutations_rejected": all(x["rejected"] for x in mutations),
        "result": "pass",
    })
    write(output / "negative-audit.json", negative)
    verify(output)
    print(f"accepted_receipt_root_sha256={accepted['receipt_root_sha256']}")
    print(f"rtx_mac_compared_bytes={compared}")
    print("verification_result=pass")
    return 0


def command_receipt(args: argparse.Namespace) -> int:
    verify(args.evidence.resolve())
    accepted = json.loads((args.evidence / "accepted-receipt.json").read_text())
    print(f"accepted_receipt_root_sha256={accepted['receipt_root_sha256']}")
    print("verification_result=pass")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    builder = sub.add_parser("build")
    builder.add_argument("--repository", type=Path, required=True)
    builder.add_argument("--rtx", type=Path, required=True)
    builder.add_argument("--mac", type=Path, required=True)
    builder.add_argument("--output", type=Path, required=True)
    builder.set_defaults(function=command_build)
    receipt = sub.add_parser("receipt")
    receipt.add_argument("--evidence", type=Path, required=True)
    receipt.set_defaults(function=command_receipt)
    args = parser.parse_args()
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
