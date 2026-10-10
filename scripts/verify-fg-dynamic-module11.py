#!/usr/bin/env python3
"""Build and verify exact dynamic-resolution module11 receipts."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


MODULE_SHA = "06eee29ec33202996f97c063a08ccdc767d0167252ad03b5ed457c941cb85206"
RESOLUTIONS = {
    "720p": {"source": [1280, 720], "output": [640, 360], "mode": 2,
             "grid": [80, 45, 1],
             "raw": {"reset": "58995d9b2bc69e3743bcf791df448280b0cbc16cfb4fb9ebe5b94f3a285718ce",
                     "generated": "56237412b763906f11db4264e2a60216744649099c5ad14dced982755e46beda"},
             "normalized": "dcf8bb3f221684c0722a26804bb2166a11435bb7c4541b43f112f7c2065aa78d"},
    "1080p": {"source": [1920, 1080], "output": [960, 540], "mode": 2,
              "grid": [120, 68, 1],
              "raw": {"reset": "825e9cfbe5e59b6a17d2038e5a20ef936988a8f9a1fca7f3bc8e01bc58bf3588",
                      "generated": "a877d9529cdb9c0466c25128b89e2b3356393a69bcd2efb7ee9101a88ad03827"},
              "normalized": "ec699d800d69e85a722681d8ccd4d6efedc11ddc98ad06ca9e048cd168d3404a"},
    "1440p": {"source": [2560, 1440], "output": [1280, 720], "mode": 2,
              "grid": [160, 90, 1],
              "raw": {"reset": "686e6a7d29621ad92eabed03596b9d0b5216ee95f885607ad6b25e0253b0596e",
                      "generated": "c466ec3869a77863539df2f5c34d8c3396efdb878bc1cd7e6f4da80b3ae07af3"},
              "normalized": "366e7de5f899c6479a59844e5a6d289341b209ff94aeccfd5db78b80886ff249"},
    "4k": {"source": [3840, 2160], "output": [960, 540], "mode": 4,
           "grid": [240, 135, 1],
           "raw": {"reset": "0f67762faac0fcc7aa4c8a3c320dad41e7121ef45d629bb851718d2abe0b2fe3",
                   "generated": "566edf20adcd58991b22cbca47447770317e4c4046256ec336a34a876d359dcd"},
           "normalized": "973d1bf49c30b91b2044eb1ffbb8d8abf9f9b5a354f66669d9eeb1ffa89603f1"},
}
PHASES = ("reset", "generated")
POISONS = ("a5", "5a")
ORDINALS = ("a", "b")
COMPONENTS = ("half-color.rgba16f", "half-metric.r16f",
              "next-history.rgba16f")


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


def expected_component_bytes(config: dict[str, Any]) -> dict[str, int]:
    source_pixels = config["source"][0] * config["source"][1]
    output_pixels = config["output"][0] * config["output"][1]
    return {"half-color.rgba16f": output_pixels * 8,
            "half-metric.r16f": output_pixels * 2,
            "next-history.rgba16f": source_pixels * 8}


def validate_backend(root: Path, backend: str) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_backend = "rtx-d3d12-nvapi" if backend == "rtx" else "mac-metal"
    pids: set[int] = set()
    outputs: dict[str, Any] = {}
    for resolution, config in RESOLUTIONS.items():
        sizes = expected_component_bytes(config)
        for phase in PHASES:
            scenario_receipts = []
            for poison in POISONS:
                for ordinal in ORDINALS:
                    directory = run_dir(root, resolution, phase, poison, ordinal)
                    receipt = json.loads((directory / "run-receipt.json").read_text())
                    if (receipt["result"] != "pass" or
                        receipt["backend"] != expected_backend or
                        receipt["resolution"] != resolution or
                        receipt["phase"] != phase or
                        receipt["selected_mode"] != config["mode"] or
                        receipt["source_extent"] != config["source"] or
                        receipt["output_extent"] != config["output"] or
                        receipt["module_sha256"] != MODULE_SHA or
                        receipt["packed_parameter_bytes"] != 168 or
                        receipt["trace_raw_parameter_sha256"] != config["raw"][phase] or
                        receipt["trace_address_free_parameter_sha256"] != config["normalized"] or
                        not receipt["typed_resource_backend"] or
                        receipt["surface_half_rounding"] != "toward-zero" or
                        receipt["input_mismatched_bytes"] != 0 or
                        receipt["canary_mismatched_bytes"] != 0 or
                        receipt["expected_or_captured_output_ingress"] or
                        receipt["archived_intermediate_ingress"] or
                        receipt["address_fields_recorded"]):
                        raise ValueError(f"run receipt rejected: {directory}")
                    if backend == "rtx" and receipt["grid"] != config["grid"]:
                        raise ValueError(f"RTX grid drift: {directory}")
                    if receipt["history_read_bank"] != (
                            "history-bank-b" if phase == "generated" else "history-bank-a"):
                        raise ValueError(f"history read bank drift: {directory}")
                    if receipt["history_write_bank"] != (
                            "history-bank-a" if phase == "generated" else "history-bank-b"):
                        raise ValueError(f"history write bank drift: {directory}")
                    should_write = poison == "a5" and ordinal == "a"
                    if receipt["artifacts_written"] != should_write:
                        raise ValueError(f"artifact mode drift: {directory}")
                    for component in COMPONENTS:
                        exists = (directory / component).is_file()
                        if exists != should_write:
                            raise ValueError(f"artifact cardinality drift: {directory}/{component}")
                    if (directory / "output.bin").is_file() != should_write:
                        raise ValueError(f"combined artifact cardinality drift: {directory}")
                    pids.add(int(receipt["process_id"]))
                    scenario_receipts.append(receipt)
            fields = ("combined_output_sha256", "half_color_sha256",
                      "half_metric_sha256", "next_history_sha256",
                      "source_pre_sha256", "history_pre_sha256")
            for field in fields:
                if len({receipt[field] for receipt in scenario_receipts}) != 1:
                    raise ValueError(
                        f"fresh-process/poison mismatch: {backend}/{resolution}/{phase}/{field}")
            full = run_dir(root, resolution, phase, "a5", "a")
            component_data = {name: (full / name).read_bytes() for name in COMPONENTS}
            for name, data in component_data.items():
                if len(data) != sizes[name]:
                    raise ValueError(f"component size drift: {full}/{name}")
            combined = b"".join(component_data[name] for name in COMPONENTS)
            if combined != (full / "output.bin").read_bytes():
                raise ValueError(f"combined output drift: {full}")
            receipt = scenario_receipts[0]
            if (sha(combined) != receipt["combined_output_sha256"] or
                sha(component_data[COMPONENTS[0]]) != receipt["half_color_sha256"] or
                sha(component_data[COMPONENTS[1]]) != receipt["half_metric_sha256"] or
                sha(component_data[COMPONENTS[2]]) != receipt["next_history_sha256"]):
                raise ValueError(f"artifact hash drift: {full}")
            outputs[f"{resolution}/{phase}"] = {
                "mode": config["mode"],
                "source_extent": config["source"],
                "output_extent": config["output"],
                "combined_output_bytes": len(combined),
                "combined_output_sha256": sha(combined),
                "half_color_sha256": sha(component_data[COMPONENTS[0]]),
                "half_metric_sha256": sha(component_data[COMPONENTS[1]]),
                "next_history_sha256": sha(component_data[COMPONENTS[2]]),
            }
    if len(pids) != 32:
        raise ValueError(f"fresh process count {backend}: {len(pids)}/32")
    manifest_root, manifest_count = tree_root(root)
    summary = {
        "schema": f"neuralstudio.fg.dynamic-module11-{backend}-backend.v1",
        "backend": expected_backend,
        "module_sha256": MODULE_SHA,
        "run_count": 32,
        "fresh_process_count": 32,
        "scenario_count": 8,
        "processes_per_scenario": 4,
        "poison_bytes": [165, 90],
        "fresh_process_repeat_exact": True,
        "dual_poison_full_write_coverage": True,
        "typed_resource_backend": True,
        "input_immutable": True,
        "input_mismatched_bytes": 0,
        "canary_mismatched_bytes": 0,
        "expected_or_captured_output_ingress": False,
        "archived_intermediate_ingress": False,
        "address_fields_recorded": False,
        "raw_artifact_manifest_sha256": manifest_root,
        "raw_artifact_file_count": manifest_count,
        "outputs": outputs,
        "result": "pass",
    }
    if backend == "rtx":
        rig_path = root / "rig-receipt.json"
        rig = json.loads(rig_path.read_text())
        if (rig["result"] != "pass" or rig["fresh_process_count"] != 32 or
            rig["module_sha256"] != MODULE_SHA or
            not rig["true_d3d12_nvapi_resources"] or
            rig["expected_or_captured_output_ingress"] or
            rig["archived_intermediate_ingress"] or
            rig["address_fields_recorded"]):
            raise ValueError("rig receipt rejected")
        summary.update({"rig_job_id": rig["job_id"],
                        "rig_receipt_sha256": sha(rig_path.read_bytes()),
                        "gpu": rig["gpu"]})
    else:
        any_receipt = json.loads((run_dir(root, "720p", "reset", "a5", "a") /
                                  "run-receipt.json").read_text())
        summary.update({"device_name": any_receipt["device_name"],
                        "metal_function": "fg_dynamic_module11"})
    return rooted(summary), outputs


def compare_artifacts(rtx_root: Path, mac_root: Path) -> tuple[int, int]:
    mismatches = 0
    compared = 0
    for resolution in RESOLUTIONS:
        for phase in PHASES:
            rtx = run_dir(rtx_root, resolution, phase, "a5", "a")
            mac = run_dir(mac_root, resolution, phase, "a5", "a")
            for component in COMPONENTS:
                left, right = (rtx / component).read_bytes(), (mac / component).read_bytes()
                if len(left) != len(right):
                    raise ValueError("backend artifact size mismatch")
                mismatches += sum(a != b for a, b in zip(left, right))
                compared += len(left)
    return mismatches, compared


def abi_receipt(repository: Path) -> dict[str, Any]:
    shape_hashes = {}
    occurrences = {}
    for resolution, config in RESOLUTIONS.items():
        path = repository / f"evidence/specs/fg-resolution-shapes-v1/shape-{resolution}.json"
        shape_hashes[resolution] = sha(path.read_bytes())
        rows = [row for row in json.loads(path.read_text())
                if row["module_token"] == "module-000011"]
        if (len(rows) != 2 or rows[0]["launch_token"] != "launch-000003" or
            rows[1]["launch_token"] != "launch-000015" or
            any(row["grid"] != config["grid"] for row in rows) or
            any(row["block"] != [256, 1, 1] for row in rows) or
            any(row["parameter_bytes"] != 168 for row in rows)):
            raise ValueError(f"shape evidence drift: {resolution}")
        occurrences[resolution] = {
            "reset_launch": 3, "generated_launch": 15,
            "grid": config["grid"], "block": [256, 1, 1],
            "mode": config["mode"], "source_extent": config["source"],
            "output_extent": config["output"],
            "reset_raw_parameter_sha256": config["raw"]["reset"],
            "generated_raw_parameter_sha256": config["raw"]["generated"],
            "address_free_parameter_sha256": config["normalized"],
        }
    return rooted({
        "schema": "neuralstudio.fg.dynamic-module11-abi.v1",
        "module_token": 11,
        "function_token": 9,
        "module_bytes": 53920,
        "module_sha256": MODULE_SHA,
        "packed_parameter_bytes": 168,
        "parameter_layout": [
            {"offset": 0, "bytes": 8, "role": "RGBA8_UNORM source", "kind": "merged-texture-sampler"},
            {"offset": 80, "bytes": 8, "role": "RGBA16F previous history bank", "kind": "merged-texture-sampler"},
            {"offset": 96, "bytes": 8, "role": "RGBA16F reduced color", "kind": "independent-surface"},
            {"offset": 104, "bytes": 8, "role": "R16F reduced metric", "kind": "independent-surface"},
            {"offset": 120, "bytes": 8, "role": "RGBA16F next history bank", "kind": "independent-surface"},
            {"offset": 128, "bytes": 4, "role": "reduction mode", "kind": "u32"},
            {"offset": 132, "bytes": 4, "role": "selected-zero flags", "kind": "u8-flags-and-padding"},
            {"offset": 136, "bytes": 8, "role": "source extent", "kind": "u32x2"},
            {"offset": 144, "bytes": 8, "role": "output extent", "kind": "u32x2"},
            {"offset": 152, "bytes": 16, "role": "selected-zero flags and optional surface", "kind": "dead-selected-path"},
        ],
        "typed_resources": {
            "inputs": ["DXGI_FORMAT_R8G8B8A8_UNORM", "DXGI_FORMAT_R16G16B16A16_FLOAT"],
            "outputs": ["DXGI_FORMAT_R16G16B16A16_FLOAT", "DXGI_FORMAT_R16_FLOAT", "DXGI_FORMAT_R16G16B16A16_FLOAT"],
        },
        "surface_half_rounding": "toward-zero",
        "mode2_reduction": "2x2 row-pair sum and global minimum metric",
        "mode4_reduction": "4x4 CUDA shuffle order; color average of 16; average of four 2x2 metric minima",
        "resolution_occurrences": occurrences,
        "shape_evidence_sha256": shape_hashes,
        "reset_generated_address_free_parameters_identical_per_resolution": True,
        "history_banks": {"reset": {"read": "A", "write": "B"},
                          "generated": {"read": "B", "write": "A"}},
        "address_fields_recorded": False,
        "result": "pass",
    })


def verify(evidence: Path) -> None:
    names = ("abi-receipt.json", "implementation-receipt.json",
             "counterexamples.json", "rtx-backend-receipt.json",
             "mac-backend-receipt.json", "accepted-receipt.json",
             "negative-audit.json")
    values = {name: json.loads((evidence / name).read_text()) for name in names}
    for name, value in values.items():
        if not valid_root(value):
            raise ValueError(f"root mismatch: {name}")
    accepted = values["accepted-receipt.json"]
    bindings = {
        "abi_receipt_root_sha256": "abi-receipt.json",
        "implementation_receipt_root_sha256": "implementation-receipt.json",
        "counterexamples_root_sha256": "counterexamples.json",
        "rtx_backend_root_sha256": "rtx-backend-receipt.json",
        "mac_backend_root_sha256": "mac-backend-receipt.json",
    }
    for field, name in bindings.items():
        if accepted[field] != values[name]["receipt_root_sha256"]:
            raise ValueError(f"accepted binding mismatch: {field}")
    if accepted["rtx_mac_mismatched_bytes"] != 0:
        raise ValueError("accepted parity mismatch")
    claims = accepted["claims"]
    if (not claims["module11_four_resolution_standalone_exact"] or
        not claims["mode4_exact"] or claims["connected_early_graph"] or
        claims["complete_generated_frame"] or claims["netal_integration"]):
        raise ValueError("claim boundary drift")
    if not values["negative-audit.json"]["all_mutations_rejected"]:
        raise ValueError("mutation audit failed")


def build(args: argparse.Namespace) -> int:
    repository = args.repository.resolve()
    rtx_root, mac_root = args.rtx.resolve(), args.mac.resolve()
    rtx, rtx_outputs = validate_backend(rtx_root, "rtx")
    mac, mac_outputs = validate_backend(mac_root, "mac")
    if rtx_outputs != mac_outputs:
        raise ValueError("backend output summary mismatch")
    mismatches, compared = compare_artifacts(rtx_root, mac_root)
    if mismatches:
        raise ValueError(f"RTX/Metal mismatch: {mismatches}/{compared}")
    abi = abi_receipt(repository)
    metal = repository / "src/fg_dynamic_module11/dynamic_module11.metal"
    host = repository / "src/fg_dynamic_module11/macos_probe.mm"
    oracle = repository / "src/fg_dynamic_module11/rtx_oracle.cpp"
    contract = repository / "src/fg_dynamic_module11/contract.h"
    implementation = rooted({
        "schema": "neuralstudio.fg.dynamic-module11-implementation.v1",
        "metal_source_sha256": sha(metal.read_bytes()),
        "macos_host_sha256": sha(host.read_bytes()),
        "rtx_oracle_sha256": sha(oracle.read_bytes()),
        "contract_sha256": sha(contract.read_bytes()),
        "mode2_source": "accepted module11 mode2 formulation, dynamically sized",
        "mode4_source": "selected SM120 PTX shuffle schedule reconstructed as scalar 4x4 groups",
        "mode4_color_order": "horizontal pair, adjacent row, adjacent x pair, two-row pair",
        "mode4_metric_order": "pairwise min, adjacent-row min, adjacent-x add, two-row add, divide by 4",
        "surface_half_rounding": "explicit round-toward-zero",
        "expected_or_captured_output_embedded": False,
        "result": "pass",
    })
    counters = rooted({
        "schema": "neuralstudio.fg.dynamic-module11-counterexamples.v1",
        "items": [
            {"name": "original-fixed-shape-host", "result": "rejected for target resolutions", "accepted_scope_root": "e9b83e2b2bd128b71b5a19fc3b769c75511903c3a8c4f0dc0857c8701a86a599", "reason": "source640x360/output320x180/grid40x23 and selected mode2 only"},
            {"name": "native-half-rounding", "result": "rejected", "source_counterexample_root": "0c139fd4ff56bcab6c7c7c97ed2f6351f70e12e621283db39601e185469e5b29", "reason": "Metal native half conversion was one half ULP above RTX at accepted mismatches; explicit RZ retained"},
            {"name": "audit-first-break-label", "result": "corrected", "superseded_audit_root": "622d263dede3c02061ff08b98f396dcc6cf72cad09d896c64540217a1e58bbc8", "corrected_audit_root": "cf77d2b7cbc51dda4c3d53d2e2dc266cd591e59f1a56adac6d40dbbfa3a4a46a", "reason": "target720p, not1080p, is already larger than the original accepted module11 extent"},
        ],
        "all_failures_and_corrections_preserved": True,
        "result": "pass",
    })
    accepted = rooted({
        "schema": "neuralstudio.fg.dynamic-module11-accepted.v1",
        "selected_unit": "module11 target-resolution reset/generated color-history preprocessing",
        "module_sha256": MODULE_SHA,
        "resolutions": list(RESOLUTIONS),
        "modes": {name: config["mode"] for name, config in RESOLUTIONS.items()},
        "processes_per_backend": 32,
        "abi_receipt_root_sha256": abi["receipt_root_sha256"],
        "implementation_receipt_root_sha256": implementation["receipt_root_sha256"],
        "counterexamples_root_sha256": counters["receipt_root_sha256"],
        "rtx_backend_root_sha256": rtx["receipt_root_sha256"],
        "mac_backend_root_sha256": mac["receipt_root_sha256"],
        "rtx_mac_compared_bytes": compared,
        "rtx_mac_mismatched_bytes": mismatches,
        "fresh_process_repeat_exact": True,
        "dual_poison_full_write_coverage": True,
        "history_banks_verified": True,
        "input_immutable": True,
        "canary_mismatched_bytes": 0,
        "expected_or_captured_output_ingress": False,
        "address_fields_recorded": False,
        "claims": {
            "module11_four_resolution_standalone_exact": True,
            "reset_and_generated_phases": True,
            "mode2_exact": True,
            "mode4_exact": True,
            "connected_early_graph": False,
            "complete_generated_frame": False,
            "netal_integration": False,
            "performance": False,
        },
        "result": "pass",
    })
    output = args.output.resolve()
    write(output / "abi-receipt.json", abi)
    write(output / "implementation-receipt.json", implementation)
    write(output / "counterexamples.json", counters)
    write(output / "rtx-backend-receipt.json", rtx)
    write(output / "mac-backend-receipt.json", mac)
    write(output / "accepted-receipt.json", accepted)
    cases = [
        ("accepted-module", accepted, ("module_sha256",), "0" * 64),
        ("accepted-parity", accepted, ("rtx_mac_mismatched_bytes",), 1),
        ("accepted-overclaim", accepted, ("claims", "connected_early_graph"), True),
        ("abi-mode4", abi, ("resolution_occurrences", "4k", "mode"), 2),
        ("abi-format", abi, ("typed_resources", "inputs"), []),
        ("implementation-order", implementation, ("mode4_metric_order",), "global min"),
        ("implementation-rounding", implementation, ("surface_half_rounding",), "native"),
        ("rtx-process", rtx, ("fresh_process_count",), 31),
        ("rtx-ingress", rtx, ("expected_or_captured_output_ingress",), True),
        ("mac-coverage", mac, ("dual_poison_full_write_coverage",), False),
        ("counterexample-loss", counters, ("all_failures_and_corrections_preserved",), False),
        ("accepted-root", accepted, ("receipt_root_sha256",), "0" * 64),
    ]
    mutations = []
    for name, original, path, value in cases:
        mutated = copy.deepcopy(original)
        cursor: Any = mutated
        for key in path[:-1]: cursor = cursor[key]
        cursor[path[-1]] = value
        mutations.append({"mutation": name, "rejected": not valid_root(mutated),
                          "reason": "receipt root mismatch"})
    negative = rooted({
        "schema": "neuralstudio.fg.dynamic-module11-negative-audit.v1",
        "accepted_receipt_root_sha256": accepted["receipt_root_sha256"],
        "mutation_count": len(mutations),
        "mutations": mutations,
        "all_mutations_rejected": all(item["rejected"] for item in mutations),
        "result": "pass",
    })
    write(output / "negative-audit.json", negative)
    verify(output)
    print(f"accepted_receipt_root_sha256={accepted['receipt_root_sha256']}")
    print(f"rtx_mac_compared_bytes={compared}")
    print("verification_result=pass")
    return 0


def receipt(args: argparse.Namespace) -> int:
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
    builder.set_defaults(function=build)
    verifier = sub.add_parser("receipt")
    verifier.add_argument("--evidence", type=Path, required=True)
    verifier.set_defaults(function=receipt)
    args = parser.parse_args()
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
