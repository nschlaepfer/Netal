#!/usr/bin/env python3
"""Verify selected FG launch 84 on its RTX and Metal backends."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Callable


RUN_SCHEMA = "neuralstudio.fg.postprocess-l84-run.v1"
BACKEND_SCHEMA = "neuralstudio.fg.postprocess-l84-backend.v1"
FINAL_SCHEMA = "neuralstudio.fg.postprocess-l84-verification.v1"
MODULE_SHA = "3d8c6aeba3a82fdeb99672243a3d392776bd1742576ab44e8ef42af105c2fd76"
RAW_PARAMETER_SHA = "31cfc2016f76ad79fe92717d4bb3b38ceb6c8b3f14968f6b5f1e2f66503e548a"
RCP_SHA = "7cf4f8cece6b18b07605edc9cb601a153346a79b10bf3b93c7ec058d57d29750"
SIGMOID_SHA = "394394a5258bad437495d68076be75d8413fa0ed5b752400e3947c332437b850"
GENERATOR = "splitmix64-fg-postprocess-l84-independent-textures-motion-v1"
DESCRIPTOR_ROOT = "d3c72b68dcec1d1bb307239c7a782a4c49e06c92b121e8e74a1f0a63ba509733"
CENSUS_ROOT = "324e579b133caf212ad9db44908a73ae6baa90ffbb4978696b51ae99cb114c26"
RUNS = (
    ("open-poison-a5", "open", 0xA5),
    ("open-poison-5a", "open", 0x5A),
    ("islands-poison-a5", "islands", 0xA5),
    ("islands-poison-5a", "islands", 0x5A),
)
ARTIFACTS = (
    ("output-color.bin", 640 * 360 * 4, "output_color_sha256"),
    ("output-mask.bin", 640 * 360, "output_mask_sha256"),
    ("pyramid-0.bin", 320 * 180 * 8, None),
    ("pyramid-1.bin", 160 * 90 * 8, None),
    ("pyramid-2.bin", 80 * 45 * 8, None),
    ("pyramid-3.bin", 40 * 23 * 8, None),
)


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise VerificationError(f"unable to load {path}: {error}") from error
    require(isinstance(value, dict), f"{path}: document must be an object")
    return value


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    try:
        return sha_bytes(path.read_bytes())
    except OSError as error:
        raise VerificationError(f"unable to hash {path}: {error}") from error


def canonical(value: Any) -> str:
    return sha_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=True).encode()
    )


def rooted(value: dict[str, Any], field: str) -> bool:
    copied = copy.deepcopy(value)
    actual = copied.pop(field, None)
    return isinstance(actual, str) and actual == canonical(copied)


def reject_address_material(value: Any, path: str = "root") -> None:
    forbidden = ("gpu_virtual_address", "resource_identity", "descriptor_ptr",
                 "host_pointer", "device_pointer")
    if isinstance(value, dict):
        for key, child in value.items():
            require(not any(token in key.lower() for token in forbidden),
                    f"{path}.{key}: volatile address material")
            reject_address_material(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_address_material(child, f"{path}[{index}]")


def run_artifacts(root: Path, run_id: str) -> tuple[list[bytes], dict[str, str]]:
    payloads: list[bytes] = []
    hashes: dict[str, str] = {}
    for name, size, _ in ARTIFACTS:
        path = root / run_id / name
        try:
            data = path.read_bytes()
        except OSError as error:
            raise VerificationError(f"unable to read {path}: {error}") from error
        require(len(data) == size, f"{path}: size mismatch")
        payloads.append(data)
        hashes[name] = sha_bytes(data)
    return payloads, hashes


def validate_run(root: Path, run_id: str, fixture: str, poison: int,
                 backend: str) -> tuple[dict[str, Any], list[bytes], dict[str, str]]:
    receipt = load(root / run_id / "run-receipt.json")
    reject_address_material(receipt, f"{run_id}.receipt")
    require(receipt.get("schema") == RUN_SCHEMA, f"{run_id}: schema")
    expected_backend = "rtx-d3d12-nvapi" if backend == "rtx" else "mac-metal"
    require(receipt.get("backend") == expected_backend, f"{run_id}: backend")
    require(receipt.get("run_id") == run_id, f"{run_id}: id")
    require(receipt.get("fixture") == fixture, f"{run_id}: fixture")
    require(receipt.get("poison_byte") == poison, f"{run_id}: poison")
    require(isinstance(receipt.get("process_id"), int) and receipt["process_id"] > 0,
            f"{run_id}: process id")
    require(receipt.get("selected_launch_token") == 84, f"{run_id}: launch")
    require(receipt.get("module_token") == 17, f"{run_id}: module token")
    require(receipt.get("module_bytes") == 33936, f"{run_id}: module bytes")
    require(receipt.get("module_sha256") == MODULE_SHA, f"{run_id}: module hash")
    require(receipt.get("raw_live_parameter_sha256") == RAW_PARAMETER_SHA,
            f"{run_id}: parameter hash")
    require(receipt.get("packed_parameter_bytes") == 152, f"{run_id}: params")
    require(receipt.get("grid") == [40, 23, 1], f"{run_id}: grid")
    require(receipt.get("block") == [256, 1, 1], f"{run_id}: block")
    require(receipt.get("input_generator") == GENERATOR, f"{run_id}: generator")
    require(receipt.get("input_mismatched_bytes") == 0,
            f"{run_id}: input mutation")
    require(receipt.get("buffer_canary_mismatched_bytes") == 0,
            f"{run_id}: canary mutation")
    require(receipt.get("expected_output_fed_to_execution") is False,
            f"{run_id}: expected bytes ingress")
    require(receipt.get("archived_intermediates_consumed") is False,
            f"{run_id}: archived ingress")
    require(receipt.get("address_fields_recorded") is False,
            f"{run_id}: address field declaration")
    require(receipt.get("result") == "pass", f"{run_id}: result")
    if backend == "rtx":
        require(receipt.get("resource_backend") ==
                "live-format-d3d12-descriptors-through-nvapi",
                f"{run_id}: D3D12/NvAPI resource backend")
    else:
        require(receipt.get("sm120_rcp_rom_sha256") == RCP_SHA,
                f"{run_id}: RCP ROM")
        require(receipt.get("sm120_half_logit_sigmoid_lut_sha256") == SIGMOID_SHA,
                f"{run_id}: sigmoid LUT")
    payloads, hashes = run_artifacts(root, run_id)
    require(receipt.get("output_root_sha256") == sha_bytes(b"".join(payloads)),
            f"{run_id}: output root")
    require(receipt.get("output_bytes") == sum(len(value) for value in payloads),
            f"{run_id}: output byte count")
    require(receipt.get("output_color_sha256") == hashes["output-color.bin"],
            f"{run_id}: color hash")
    require(receipt.get("output_mask_sha256") == hashes["output-mask.bin"],
            f"{run_id}: mask hash")
    require(receipt.get("pyramid_sha256") ==
            [hashes[f"pyramid-{index}.bin"] for index in range(4)],
            f"{run_id}: pyramid hashes")
    mask = payloads[1]
    require(all(value in (0, 255) for value in mask), f"{run_id}: mask domain")
    if backend == "mac":
        native_path = root / run_id / "native-metal-output-color.bin"
        require(native_path.stat().st_size == 640 * 360 * 4,
                f"{run_id}: native negative size")
        require(receipt.get("native_metal_output_color_sha256") ==
                sha_file(native_path), f"{run_id}: native negative hash")
    return receipt, payloads, hashes


def build_backend(root: Path, backend: str) -> dict[str, Any]:
    require(backend in ("rtx", "mac"), "invalid backend")
    runs: list[dict[str, Any]] = []
    payloads: dict[str, list[bytes]] = {}
    receipts: dict[str, dict[str, Any]] = {}
    for run_id, fixture, poison in RUNS:
        receipt, output, hashes = validate_run(root, run_id, fixture, poison,
                                                backend)
        payloads[run_id] = output
        receipts[run_id] = receipt
        runs.append({
            "run_id": run_id,
            "fixture": fixture,
            "poison_byte": poison,
            "process_id": receipt["process_id"],
            "input_root_sha256": receipt["input_root_sha256"],
            "output_root_sha256": receipt["output_root_sha256"],
            "artifacts": hashes,
        })
    require(len({item["process_id"] for item in runs}) == 4,
            "fresh-process count")
    for fixture in ("open", "islands"):
        pair = [item for item in runs if item["fixture"] == fixture]
        require(len(pair) == 2 and pair[0]["input_root_sha256"] ==
                pair[1]["input_root_sha256"], f"{fixture}: input repeat")
        first, second = payloads[pair[0]["run_id"]], payloads[pair[1]["run_id"]]
        require(first == second, f"{fixture}: two-poison output invariance")
    negative_counts: dict[str, int] = {}
    if backend == "mac":
        for fixture in ("open", "islands"):
            run_id = f"{fixture}-poison-a5"
            native = (root / run_id / "native-metal-output-color.bin").read_bytes()
            accepted = payloads[run_id][0]
            count = sum(left != right for left, right in zip(native, accepted))
            require(count > 0, f"{fixture}: native UNORM negative missing")
            negative_counts[fixture] = count
    document: dict[str, Any] = {
        "schema": BACKEND_SCHEMA,
        "backend": "rtx-d3d12-nvapi" if backend == "rtx" else "mac-metal",
        "selected_unit": "evaluate-1/launch-000084/main_kernel/module-000017",
        "module_sha256": MODULE_SHA,
        "raw_live_parameter_sha256": RAW_PARAMETER_SHA,
        "input_generator": GENERATOR,
        "runs": runs,
        "fresh_process_count": 4,
        "fixture_count": 2,
        "poison_bytes": [90, 165],
        "two_poison_output_invariance": True,
        "full_write_coverage_from_two_poison_invariance": True,
        "input_immutability": True,
        "buffer_canary_integrity": True,
        "native_metal_unorm_negative_mismatched_bytes": negative_counts,
        "expected_output_fed_to_execution": False,
        "archived_intermediates_consumed": False,
        "address_fields_recorded": False,
        "result": "pass",
    }
    document["receipt_root_sha256"] = canonical(document)
    return document


def build_final(rtx_root: Path, mac_root: Path) -> dict[str, Any]:
    rtx = build_backend(rtx_root, "rtx")
    mac = build_backend(mac_root, "mac")
    comparisons: list[dict[str, Any]] = []
    for run_id, fixture, poison in RUNS:
        rtx_run = next(item for item in rtx["runs"] if item["run_id"] == run_id)
        mac_run = next(item for item in mac["runs"] if item["run_id"] == run_id)
        require(rtx_run["input_root_sha256"] == mac_run["input_root_sha256"],
                f"{run_id}: backend input drift")
        artifact_rows = []
        for name, size, _ in ARTIFACTS:
            rtx_bytes = (rtx_root / run_id / name).read_bytes()
            mac_bytes = (mac_root / run_id / name).read_bytes()
            mismatches = sum(left != right
                             for left, right in zip(rtx_bytes, mac_bytes))
            require(len(rtx_bytes) == len(mac_bytes) == size,
                    f"{run_id}/{name}: size")
            require(mismatches == 0, f"{run_id}/{name}: parity")
            artifact_rows.append({
                "artifact": name,
                "bytes": size,
                "sha256": sha_bytes(rtx_bytes),
                "mismatched_bytes": 0,
            })
        comparisons.append({
            "run_id": run_id,
            "fixture": fixture,
            "poison_byte": poison,
            "input_root_sha256": rtx_run["input_root_sha256"],
            "artifacts": artifact_rows,
            "mismatched_bytes": 0,
        })
    document: dict[str, Any] = {
        "schema": FINAL_SCHEMA,
        "selected_launch_token": 84,
        "module_token": 17,
        "module_sha256": MODULE_SHA,
        "raw_live_parameter_sha256": RAW_PARAMETER_SHA,
        "selected_module_census_root_sha256": CENSUS_ROOT,
        "resource_descriptor_provenance_root_sha256": DESCRIPTOR_ROOT,
        "input_generator": GENERATOR,
        "supported_domain": {
            "width": 640,
            "height": 360,
            "padded_motion_height": 368,
            "divisor": 1,
            "input_formats": ["RGBA16_FLOAT", "RGBA16_FLOAT", "R8_UNORM",
                              "half8-motion"],
            "output_formats": ["RGBA8_UNORM", "R8_UNORM", "half4-pyramid"],
        },
        "comparisons": comparisons,
        "rtx_backend_root_sha256": rtx["receipt_root_sha256"],
        "mac_backend_root_sha256": mac["receipt_root_sha256"],
        "sm120_rcp_rom_sha256": RCP_SHA,
        "sm120_half_logit_sigmoid_lut_sha256": SIGMOID_SHA,
        "sust_p_unorm8_rule": {
            "fixed_fraction_bits": 12,
            "fixed_conversion": "floor(clamp(value,0,1)*4096)",
            "unorm_conversion": "min(255,(fixed12*255+2047)>>12)",
            "native_metal_counterexample_mismatched_bytes":
                mac["native_metal_unorm_negative_mismatched_bytes"],
        },
        "fresh_process_count_per_backend": 4,
        "two_poison_output_invariance": True,
        "full_write_coverage_from_two_poison_invariance": True,
        "all_output_artifacts_rtx_metal_exact": True,
        "input_immutability": True,
        "buffer_canary_integrity": True,
        "expected_output_fed_to_execution": False,
        "archived_intermediates_consumed": False,
        "address_fields_recorded": False,
        "claims": {
            "launch84_dependency_closed_unit_exact": True,
            "launch85_or_later_inferred": False,
            "complete_tail_exact": False,
            "complete_frame_exact": False,
            "netal_integrated": False,
            "performance_advantage_proven": False,
        },
        "result": "pass",
    }
    document["receipt_root_sha256"] = canonical(document)
    return document


def validate_final_document(document: dict[str, Any]) -> None:
    reject_address_material(document)
    require(document.get("schema") == FINAL_SCHEMA, "final schema")
    require(rooted(document, "receipt_root_sha256"), "final root")
    require(document.get("module_sha256") == MODULE_SHA, "final module")
    require(document.get("all_output_artifacts_rtx_metal_exact") is True,
            "final parity")
    require(document.get("two_poison_output_invariance") is True,
            "final poison")
    require(document.get("full_write_coverage_from_two_poison_invariance") is True,
            "final coverage")
    require(document.get("expected_output_fed_to_execution") is False,
            "final expected ingress")
    require(document.get("archived_intermediates_consumed") is False,
            "final archived ingress")
    require(document.get("address_fields_recorded") is False,
            "final address declaration")
    require(document.get("result") == "pass", "final result")
    comparisons = document.get("comparisons")
    require(isinstance(comparisons, list) and len(comparisons) == 4,
            "final comparison count")
    require(all(item.get("mismatched_bytes") == 0 for item in comparisons),
            "final comparison mismatch")


def build_audit(accepted: dict[str, Any]) -> dict[str, Any]:
    validate_final_document(accepted)
    cases: list[tuple[str, Callable[[dict[str, Any]], None]]] = []
    cases.append(("output_mismatch", lambda value:
                  value["comparisons"][0].__setitem__("mismatched_bytes", 1)))
    cases.append(("module_drift", lambda value:
                  value.__setitem__("module_sha256", "0" * 64)))
    cases.append(("poison_gate", lambda value:
                  value.__setitem__("two_poison_output_invariance", False)))
    cases.append(("coverage_gate", lambda value:
                  value.__setitem__("full_write_coverage_from_two_poison_invariance", False)))
    cases.append(("expected_ingress", lambda value:
                  value.__setitem__("expected_output_fed_to_execution", True)))
    cases.append(("archived_ingress", lambda value:
                  value.__setitem__("archived_intermediates_consumed", True)))
    cases.append(("address_leak", lambda value:
                  value.__setitem__("gpu_virtual_address", "0x1234")))
    cases.append(("result_drift", lambda value:
                  value.__setitem__("result", "fail")))
    outcomes = []
    for name, mutation in cases:
        mutated = copy.deepcopy(accepted)
        mutation(mutated)
        rejected = False
        try:
            validate_final_document(mutated)
        except VerificationError:
            rejected = True
        require(rejected, f"mutation was not rejected: {name}")
        outcomes.append({"mutation": name, "rejected": True})
    document: dict[str, Any] = {
        "schema": "neuralstudio.fg.postprocess-l84-negative-audit.v1",
        "accepted_receipt_root_sha256": accepted["receipt_root_sha256"],
        "mutations": outcomes,
        "mutation_count": len(outcomes),
        "all_mutations_rejected": True,
        "address_fields_recorded": False,
        "result": "pass",
    }
    document["receipt_root_sha256"] = canonical(document)
    return document


def write_exclusive(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x") as stream:
            json.dump(document, stream, indent=2, sort_keys=True)
            stream.write("\n")
    except FileExistsError as error:
        raise VerificationError(f"refusing to overwrite {path}") from error


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    backend_parser = subparsers.add_parser("backend")
    backend_parser.add_argument("--backend", choices=("rtx", "mac"), required=True)
    backend_parser.add_argument("--root", type=Path, required=True)
    backend_parser.add_argument("--output", type=Path, required=True)
    final_parser = subparsers.add_parser("final")
    final_parser.add_argument("--rtx-root", type=Path, required=True)
    final_parser.add_argument("--mac-root", type=Path, required=True)
    final_parser.add_argument("--output", type=Path, required=True)
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--accepted", type=Path, required=True)
    audit_parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        if arguments.command == "backend":
            document = build_backend(arguments.root, arguments.backend)
        elif arguments.command == "final":
            document = build_final(arguments.rtx_root, arguments.mac_root)
        else:
            document = build_audit(load(arguments.accepted))
        write_exclusive(arguments.output, document)
        print(f"receipt_root_sha256={document['receipt_root_sha256']}")
        print("result=pass")
        return 0
    except VerificationError as error:
        print(f"verification_error={error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
