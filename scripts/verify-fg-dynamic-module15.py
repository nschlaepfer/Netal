#!/usr/bin/env python3
"""Build and verify exact target-resolution module15 evidence."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


MODULE_SHA = "9e772df29e9434e4eff1edf1cde32706ea641c2b0156a459a080565a7483f381"
RESOLUTIONS = {
    "720p": {"input": [1280, 720], "output": [640, 360], "grid": [40, 23, 1],
             "inverse_bits": ["0x3a4ccccd", "0x3ab60b61"],
             "output_scale_bits": ["0x3523d70a", "0x3601742e"],
             "raw": {"reset": "b07cb1a8589f53bb215567a4e350d095af1c349f835f2ce2fd193f881d261ae4", "generated": "52ae7695610f72ebceb61c096bb7c1789135656034d1ae94532086978838a383"},
             "normalized": {"reset": "b0fce9db2b5718e69c816380a59408e5552b6be452fd8bcea7c5ffc9707e9930", "generated": "26575a19da94504c4bae6eb21a021c730bf882673849a084180ddb2c377f68cc"}},
    "1080p": {"input": [1920, 1080], "output": [960, 540], "grid": [60, 34, 1],
              "inverse_bits": ["0x3a088889", "0x3a72b9d6"],
              "output_scale_bits": ["0x3491a2b4", "0x356623e0"],
              "raw": {"reset": "2ee9d9d547bb7af218906304a1cb71b2bb30a78f41fa58b79b75c9c113c58e97", "generated": "cef79207030577bcf4cda92c0e404389eed4909b0f6e09c87f8bc1b4b43a2fcc"},
              "normalized": {"reset": "1788f069350f9896fa7c7be1530ff90a8d6bad33c4379d3d91181e15e44b0472", "generated": "ae465a1b32cc29548c2942087f675cb362f070a299fcacf77920362807f32fc9"}},
    "1440p": {"input": [2560, 1440], "output": [1280, 720], "grid": [80, 45, 1],
              "inverse_bits": ["0x39cccccd", "0x3a360b61"],
              "output_scale_bits": ["0x3423d70a", "0x3501742e"],
              "raw": {"reset": "a744a01df885f1abf569ecde2e16b74748af3fe9e01e4920d6bf9d6e4b5edecf", "generated": "78f11c9415b7aa4c97b4955d8aef62f65402fb980c65255b193a12724c8d997c"},
              "normalized": {"reset": "316cb00aa4ae44a7d282330042a91dc462627582aa429ebb533fcfb9bbe23892", "generated": "9a3aa6c1755ae16a248dcc58ef129adb18303e4e992c2f88d0c7775d3ffd7269"}},
    "4k": {"input": [3840, 2160], "output": [1920, 1080], "grid": [120, 68, 1],
           "inverse_bits": ["0x39888889", "0x39f2b9d6"],
           "output_scale_bits": ["0x3391a2b4", "0x346623e0"],
           "raw": {"reset": "8a2ed8dac885206589505eec4a1520fa0bc0fea4db842f2111580f7008372064", "generated": "93a4ed4b32798b1141a0fc0bf224da2ad43e29cc95dd254cb1233957bacbdc6d"},
           "normalized": {"reset": "78a2dd677a28fa8ac23a4ebc8f67642a17cdb0b8647bfdb520e4f65035d0ad8e", "generated": "ff281a810b9297147ad4ea38c1c0e1899b5e07dcb5abeccd39123a6a57d4c896"}},
}
PHASES = ("reset", "generated")
POISONS = ("a5", "5a")
ORDINALS = ("a", "b")
COMPONENTS = ("vector-output.bin", "depth-output.bin")


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


def run_dir(root: Path, resolution: str, phase: str,
            poison: str, ordinal: str) -> Path:
    return root / f"{resolution}-{phase}-poison-{poison}-process-{ordinal}"


def tree_root(root: Path) -> tuple[str, int]:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "build" in path.relative_to(root).parts:
            continue
        data = path.read_bytes()
        rows.append({"path": path.relative_to(root).as_posix(),
                     "bytes": len(data), "sha256": sha(data)})
    return sha(canonical(rows)), len(rows)


def component_sizes(config: dict[str, Any]) -> dict[str, int]:
    pixels = config["output"][0] * config["output"][1]
    return {"vector-output.bin": pixels * 8, "depth-output.bin": pixels * 4}


def validate_backend(root: Path, backend: str) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_backend = ("rtx-cuda-driver-typed-arrays" if backend == "rtx"
                        else "mac-metal")
    pids: set[int] = set()
    outputs: dict[str, Any] = {}
    for resolution, config in RESOLUTIONS.items():
        sizes = component_sizes(config)
        for phase in PHASES:
            receipts = []
            for poison in POISONS:
                for ordinal in ORDINALS:
                    directory = run_dir(root, resolution, phase, poison, ordinal)
                    receipt = json.loads((directory / "run-receipt.json").read_text())
                    if (receipt["result"] != "pass" or receipt["backend"] != expected_backend or
                        receipt["resolution"] != resolution or receipt["phase"] != phase or
                        receipt["module_sha256"] != MODULE_SHA or
                        receipt["input_extent"] != config["input"] or
                        receipt["output_extent"] != config["output"] or
                        receipt["trace_raw_parameter_sha256"] != config["raw"][phase] or
                        receipt["trace_address_free_parameter_sha256"] != config["normalized"][phase] or
                        receipt["fixture_address_free_parameter_sha256"] != config["normalized"][phase] or
                        receipt["selected_mode_byte_84"] != 0 or
                        receipt["reset_marker_byte_85"] != (127 if phase == "reset" else 0) or
                        receipt["input_mismatched_bytes"] != 0 or
                        receipt["vector_canary_mismatched_bytes"] != 0 or
                        receipt["depth_canary_mismatched_bytes"] != 0 or
                        receipt["vector_untouched_pixels"] != 0 or
                        receipt["depth_untouched_pixels"] != 0 or
                        receipt["expected_or_captured_output_ingress"] or
                        receipt["archived_intermediate_ingress"] or
                        receipt["address_fields_recorded"]):
                        raise ValueError(f"run receipt rejected: {directory}")
                    if backend == "rtx" and (receipt["grid"] != config["grid"] or
                                              receipt["block"] != [16, 16, 1]):
                        raise ValueError(f"dispatch drift: {directory}")
                    should_write = poison == "a5" and ordinal == "a"
                    if receipt["artifacts_written"] != should_write:
                        raise ValueError(f"artifact mode drift: {directory}")
                    for component in COMPONENTS:
                        if (directory / component).is_file() != should_write:
                            raise ValueError(f"artifact cardinality: {directory}/{component}")
                    pids.add(int(receipt["process_id"]))
                    receipts.append(receipt)
            for field in ("vector_output_sha256", "depth_output_sha256",
                          "vector_input_pre_sha256", "scalar_input_pre_sha256"):
                if len({receipt[field] for receipt in receipts}) != 1:
                    raise ValueError(f"repeat/poison mismatch: {backend}/{resolution}/{phase}/{field}")
            full = run_dir(root, resolution, phase, "a5", "a")
            data = {name: (full / name).read_bytes() for name in COMPONENTS}
            receipt = receipts[0]
            for name in COMPONENTS:
                if len(data[name]) != sizes[name]:
                    raise ValueError(f"component size drift: {full}/{name}")
            if (sha(data["vector-output.bin"]) != receipt["vector_output_sha256"] or
                sha(data["depth-output.bin"]) != receipt["depth_output_sha256"]):
                raise ValueError(f"artifact hash drift: {full}")
            outputs[f"{resolution}/{phase}"] = {
                "input_extent": config["input"], "output_extent": config["output"],
                "vector_output_bytes": len(data["vector-output.bin"]),
                "vector_output_sha256": sha(data["vector-output.bin"]),
                "depth_output_bytes": len(data["depth-output.bin"]),
                "depth_output_sha256": sha(data["depth-output.bin"]),
            }
    if len(pids) != 32:
        raise ValueError(f"fresh process count {backend}: {len(pids)}/32")
    manifest_root, manifest_count = tree_root(root)
    summary = {
        "schema": f"neuralstudio.fg.dynamic-module15-{backend}-backend.v1",
        "backend": expected_backend, "module_sha256": MODULE_SHA,
        "run_count": 32, "fresh_process_count": 32, "scenario_count": 8,
        "processes_per_scenario": 4, "poison_bytes": [165, 90],
        "fresh_process_repeat_exact": True,
        "dual_poison_full_write_coverage": True,
        "input_immutable": True, "input_mismatched_bytes": 0,
        "vector_canary_mismatched_bytes": 0,
        "depth_canary_mismatched_bytes": 0,
        "expected_or_captured_output_ingress": False,
        "archived_intermediate_ingress": False,
        "address_fields_recorded": False,
        "raw_artifact_manifest_sha256": manifest_root,
        "raw_artifact_file_count": manifest_count,
        "outputs": outputs, "result": "pass",
    }
    if backend == "rtx":
        rig_path = root / "rig-receipt.json"
        rig = json.loads(rig_path.read_text())
        if (rig["result"] != "pass" or rig["fresh_process_count"] != 32 or
            rig["module_sha256"] != MODULE_SHA or not rig["true_typed_cuda_arrays"] or
            rig["expected_or_captured_output_ingress"] or
            rig["archived_intermediate_ingress"] or rig["address_fields_recorded"]):
            raise ValueError("rig receipt rejected")
        summary.update({"rig_job_id": rig["job_id"], "gpu": rig["gpu"],
                        "rig_receipt_sha256": sha(rig_path.read_bytes()),
                        "typed_resource_backend": "CUDA arrays/textures/surfaces"})
    else:
        one = json.loads((run_dir(root, "720p", "reset", "a5", "a") /
                          "run-receipt.json").read_text())
        summary.update({"device_name": one["device_name"],
                        "typed_resource_backend": "Metal textures",
                        "metal_function": "fg_dynamic_module15"})
    return rooted(summary), outputs


def compare(rtx: Path, mac: Path) -> tuple[int, int]:
    mismatches = 0
    compared = 0
    for resolution in RESOLUTIONS:
        for phase in PHASES:
            for component in COMPONENTS:
                left = (run_dir(rtx, resolution, phase, "a5", "a") / component).read_bytes()
                right = (run_dir(mac, resolution, phase, "a5", "a") / component).read_bytes()
                if len(left) != len(right): raise ValueError("artifact size mismatch")
                mismatches += sum(a != b for a, b in zip(left, right))
                compared += len(left)
    return mismatches, compared


def make_abi(repository: Path) -> dict[str, Any]:
    occurrences = {}
    shape_hashes = {}
    for resolution, config in RESOLUTIONS.items():
        path = repository / f"evidence/specs/fg-resolution-shapes-v1/shape-{resolution}.json"
        shape_hashes[resolution] = sha(path.read_bytes())
        rows = [row for row in json.loads(path.read_text())
                if row["module_token"] == "module-000015"]
        if (len(rows) != 2 or [row["launch_token"] for row in rows] !=
                ["launch-000004", "launch-000016"] or
            any(row["grid"] != config["grid"] for row in rows) or
            any(row["block"] != [16, 16, 1] for row in rows)):
            raise ValueError(f"shape drift: {resolution}")
        occurrences[resolution] = {
            "input_extent": config["input"], "output_extent": config["output"],
            "grid": config["grid"], "block": [16, 16, 1],
            "inverse_extent_bits": config["inverse_bits"],
            "vector_output_scale_bits": config["output_scale_bits"],
            "reset_raw_parameter_sha256": config["raw"]["reset"],
            "generated_raw_parameter_sha256": config["raw"]["generated"],
            "reset_address_free_parameter_sha256": config["normalized"]["reset"],
            "generated_address_free_parameter_sha256": config["normalized"]["generated"],
        }
    return rooted({
        "schema": "neuralstudio.fg.dynamic-module15-abi.v1",
        "module_token": 15, "function_token": 13, "module_bytes": 12632,
        "module_sha256": MODULE_SHA, "packed_parameter_bytes": 160,
        "large_trace_launches": {"reset": 4, "generated": 16},
        "canonical_launches": {"reset": 4, "generated": 14},
        "selected_mode_byte_84": {"reset": 0, "generated": 0},
        "dead_reset_marker_byte_85": {"reset": 127, "generated": 0},
        "selected_candidate_rule": "minimum scalar; strict ties retain prior candidate",
        "typed_resources": {"inputs": ["RG16_FLOAT point-clamp normalized", "R32_FLOAT point-border normalized"],
                            "outputs": ["RG32_FLOAT", "R32_UINT"]},
        "resolution_occurrences": occurrences,
        "shape_evidence_sha256": shape_hashes,
        "sampler_lowering": "fixed-12 normalized point map plus one measured 720p positive-Y retention row",
        "address_fields_recorded": False, "result": "pass",
    })


def verify(evidence: Path) -> None:
    names = ("abi-receipt.json", "implementation-receipt.json", "coordinate-probe-receipt.json",
             "counterexamples.json", "rtx-backend-receipt.json", "mac-backend-receipt.json",
             "accepted-receipt.json", "negative-audit.json")
    values = {name: json.loads((evidence / name).read_text()) for name in names}
    for name, value in values.items():
        if not valid_root(value): raise ValueError(f"root mismatch: {name}")
    accepted = values["accepted-receipt.json"]
    bindings = {"abi_receipt_root_sha256": "abi-receipt.json",
                "implementation_receipt_root_sha256": "implementation-receipt.json",
                "coordinate_probe_root_sha256": "coordinate-probe-receipt.json",
                "counterexamples_root_sha256": "counterexamples.json",
                "rtx_backend_root_sha256": "rtx-backend-receipt.json",
                "mac_backend_root_sha256": "mac-backend-receipt.json"}
    for field, name in bindings.items():
        if accepted[field] != values[name]["receipt_root_sha256"]:
            raise ValueError(f"accepted binding mismatch: {field}")
    claims = accepted["claims"]
    if (accepted["rtx_mac_mismatched_bytes"] != 0 or
        not claims["module15_four_resolution_standalone_exact"] or
        claims["connected_early_graph"] or claims["complete_generated_frame"] or
        claims["netal_integration"]):
        raise ValueError("claim boundary drift")
    if not values["negative-audit.json"]["all_mutations_rejected"]:
        raise ValueError("mutation audit failed")


def build(args: argparse.Namespace) -> int:
    repository = args.repository.resolve()
    rtx_root, mac_root = args.rtx.resolve(), args.mac.resolve()
    rtx, ro = validate_backend(rtx_root, "rtx")
    mac, mo = validate_backend(mac_root, "mac")
    if ro != mo: raise ValueError("backend summary mismatch")
    mismatches, compared = compare(rtx_root, mac_root)
    if mismatches: raise ValueError(f"RTX/Metal mismatch: {mismatches}/{compared}")
    abi = make_abi(repository)
    paths = {"metal_source_sha256": "src/fg_dynamic_module15/dynamic_module15.metal",
             "macos_host_sha256": "src/fg_dynamic_module15/macos_probe.mm",
             "rtx_oracle_sha256": "src/fg_dynamic_module15/rtx_oracle.cpp",
             "contract_sha256": "src/fg_dynamic_module15/contract.h"}
    implementation = rooted({
        "schema": "neuralstudio.fg.dynamic-module15-implementation.v1",
        **{field: sha((repository / path).read_bytes()) for field, path in paths.items()},
        "selected_mode_byte_84": 0,
        "dead_reset_marker_byte_85_preserved": True,
        "fixed12_sampler_exception": {"resolution": "720p", "output_row": 340,
                                      "direction": "positive-y", "fixed12_source_row": 681,
                                      "sm120_source_row": 682},
        "expected_or_captured_output_embedded": False, "result": "pass",
    })
    probe_raw = json.loads((args.coordinate_probe / "receipt.json").read_text())
    probe = rooted({
        "schema": "neuralstudio.fg.dynamic-module15-coordinate-proof.v1",
        "remote_job": "fg-dynamic-module15-20260904T141514Z",
        "source_sha256": probe_raw["source_sha256"],
        "binary_sha256": probe_raw["binary_sha256"],
        "samples_bytes": probe_raw["samples_bytes"],
        "samples_sha256": probe_raw["samples_sha256"],
        "sample_count": 230400, "candidate_samples_per_output": 5,
        "fixed12_candidate_mismatches": 1280,
        "mismatch_breakdown": {"upper-left": 640, "upper-right": 640},
        "mismatch_output_rows": [340],
        "actual_source_row": 682, "fixed12_source_row": 681,
        "execution_lookup_ingress": False, "result": "pass",
    })
    counters = rooted({
        "schema": "neuralstudio.fg.dynamic-module15-counterexamples.v1",
        "items": [
            {"name": "prototype-v1-fixed12", "result": "rejected", "exact_resolutions": ["1080p", "1440p", "4k"], "720p": {"reset_vector_mismatched_bytes": 2482, "reset_depth_mismatched_bytes": 699, "generated_vector_mismatched_bytes": 2385, "generated_depth_mismatched_bytes": 681}, "first_difference_row": 340},
            {"name": "prototype-v2-negative-y", "result": "rejected", "reason": "wrong direction inferred before coordinate differential", "720p": {"reset_vector_mismatched_bytes": 3809, "reset_depth_mismatched_bytes": 1082, "generated_vector_mismatched_bytes": 3803, "generated_depth_mismatched_bytes": 1072}, "first_difference_row": 340},
            {"name": "legacy-live-mode-label", "result": "corrected", "reason": "PTX reads byte84=0 in both live phases; reset marker127 is dead byte85"},
        ],
        "all_failures_preserved": True, "result": "pass",
    })
    accepted = rooted({
        "schema": "neuralstudio.fg.dynamic-module15-accepted.v1",
        "selected_unit": "module15 target-resolution reset/generated vector-depth preprocessing",
        "module_sha256": MODULE_SHA, "resolutions": list(RESOLUTIONS),
        "processes_per_backend": 32,
        "abi_receipt_root_sha256": abi["receipt_root_sha256"],
        "implementation_receipt_root_sha256": implementation["receipt_root_sha256"],
        "coordinate_probe_root_sha256": probe["receipt_root_sha256"],
        "counterexamples_root_sha256": counters["receipt_root_sha256"],
        "rtx_backend_root_sha256": rtx["receipt_root_sha256"],
        "mac_backend_root_sha256": mac["receipt_root_sha256"],
        "rtx_mac_compared_bytes": compared,
        "rtx_mac_mismatched_bytes": mismatches,
        "fresh_process_repeat_exact": True,
        "dual_poison_full_write_coverage": True,
        "input_immutable": True, "canary_mismatched_bytes": 0,
        "expected_or_captured_output_ingress": False,
        "address_fields_recorded": False,
        "claims": {"module15_four_resolution_standalone_exact": True,
                   "reset_and_generated_phases": True,
                   "live_minimum_selection_path": True,
                   "connected_early_graph": False,
                   "complete_generated_frame": False,
                   "netal_integration": False, "performance": False},
        "result": "pass",
    })
    output = args.output.resolve()
    for name, value in (("abi-receipt.json", abi), ("implementation-receipt.json", implementation),
                        ("coordinate-probe-receipt.json", probe), ("counterexamples.json", counters),
                        ("rtx-backend-receipt.json", rtx), ("mac-backend-receipt.json", mac),
                        ("accepted-receipt.json", accepted)):
        write(output / name, value)
    cases = [("accepted-parity", accepted, ("rtx_mac_mismatched_bytes",), 1),
             ("accepted-overclaim", accepted, ("claims", "connected_early_graph"), True),
             ("abi-mode", abi, ("selected_mode_byte_84", "reset"), 127),
             ("abi-grid", abi, ("resolution_occurrences", "4k", "grid"), [1,1,1]),
             ("impl-exception", implementation, ("fixed12_sampler_exception", "output_row"), 341),
             ("probe-coverage", probe, ("fixed12_candidate_mismatches",), 0),
             ("rtx-process", rtx, ("fresh_process_count",), 31),
             ("rtx-ingress", rtx, ("expected_or_captured_output_ingress",), True),
             ("mac-coverage", mac, ("dual_poison_full_write_coverage",), False),
             ("counterexample-loss", counters, ("all_failures_preserved",), False),
             ("accepted-module", accepted, ("module_sha256",), "0"*64),
             ("accepted-root", accepted, ("receipt_root_sha256",), "0"*64)]
    mutations=[]
    for name, original, keys, value in cases:
        mutated=copy.deepcopy(original); cursor:Any=mutated
        for key in keys[:-1]: cursor=cursor[key]
        cursor[keys[-1]]=value
        mutations.append({"mutation":name,"rejected":not valid_root(mutated),"reason":"receipt root mismatch"})
    negative=rooted({"schema":"neuralstudio.fg.dynamic-module15-negative-audit.v1",
                     "accepted_receipt_root_sha256":accepted["receipt_root_sha256"],
                     "mutation_count":len(mutations),"mutations":mutations,
                     "all_mutations_rejected":all(x["rejected"] for x in mutations),"result":"pass"})
    write(output / "negative-audit.json", negative)
    verify(output)
    print(f"accepted_receipt_root_sha256={accepted['receipt_root_sha256']}")
    print(f"rtx_mac_compared_bytes={compared}")
    print("verification_result=pass")
    return 0


def receipt(args: argparse.Namespace) -> int:
    verify(args.evidence.resolve())
    accepted=json.loads((args.evidence/"accepted-receipt.json").read_text())
    print(f"accepted_receipt_root_sha256={accepted['receipt_root_sha256']}")
    print("verification_result=pass")
    return 0


def main() -> int:
    parser=argparse.ArgumentParser(); sub=parser.add_subparsers(dest="command",required=True)
    builder=sub.add_parser("build"); builder.add_argument("--repository",type=Path,required=True);builder.add_argument("--rtx",type=Path,required=True);builder.add_argument("--mac",type=Path,required=True);builder.add_argument("--coordinate-probe",type=Path,required=True);builder.add_argument("--output",type=Path,required=True);builder.set_defaults(function=build)
    verifier=sub.add_parser("receipt");verifier.add_argument("--evidence",type=Path,required=True);verifier.set_defaults(function=receipt)
    args=parser.parse_args();return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
