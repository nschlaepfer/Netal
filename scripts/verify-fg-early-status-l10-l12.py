#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Callable


RUN_SCHEMA = "neuralstudio.fg.early-status-l10-l12-run.v1"
BACKEND_SCHEMA = "neuralstudio.fg.early-status-l10-l12-backend.v1"
FINAL_SCHEMA = "neuralstudio.fg.early-status-l10-l12-verification.v1"
MODULE_SHA = "c8b8c082139005c1bf059f156827c4e331f9a6579ded1b2c9646198b1cc2abf7"
GENERATOR = "no-tensor-input-status-u32-v1"
MODES = {
    "reset": {
        "launch": 10,
        "value": 1,
        "raw": "7d61d31336110db4d7ca50f8ed86e5a512bf414641011c84616c87d6dcb02331",
        "address_free": "b342c83d7060c5852f741e8e009d375103df2e1f7c76fbc512e575cd86d7b03d",
        "output": "67abdd721024f0ff4e0b3f4c2fc13bc5bad42d0b7851d456d88d203d15aaa450",
    },
    "generated": {
        "launch": 12,
        "value": 0,
        "raw": "45ea95d049547e4afa55733c880dd3f2681c9fb161b7d47202c736de0ee8aa3d",
        "address_free": "9d34149fbd1fe777eb238799054c8cbfbce372255f219f8740838def9bfd02db",
        "output": "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119",
    },
}
RUNS = tuple(
    (f"status-{mode}-poison-{poison}-process-{ordinal}", mode,
     int(poison, 16))
    for mode in ("reset", "generated")
    for poison in ("a5", "5a")
    for ordinal in ("a", "b")
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
    require(isinstance(value, dict), f"{path}: expected object")
    return value


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: Any) -> str:
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True).encode())


def rooted(value: dict[str, Any]) -> bool:
    copied = copy.deepcopy(value)
    actual = copied.pop("receipt_root_sha256", None)
    return isinstance(actual, str) and actual == canonical(copied)


def reroot(value: dict[str, Any]) -> None:
    value.pop("receipt_root_sha256", None)
    value["receipt_root_sha256"] = canonical(value)


def reject_addresses(value: Any, path: str = "root") -> None:
    forbidden = ("gpu_virtual_address", "device_pointer", "host_pointer",
                 "resource_identity", "descriptor_ptr")
    if isinstance(value, dict):
        for key, child in value.items():
            require(not any(item in key.lower() for item in forbidden),
                    f"{path}.{key}: volatile address")
            reject_addresses(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_addresses(child, f"{path}[{index}]")


def validate_run(root: Path, run_id: str, mode: str, poison: int,
                 backend: str) -> tuple[dict[str, Any], bytes]:
    receipt = load(root / run_id / "run-receipt.json")
    reject_addresses(receipt, run_id)
    identity = MODES[mode]
    require(receipt.get("schema") == RUN_SCHEMA, f"{run_id}: schema")
    require(receipt.get("backend") ==
            ("rtx-cuda-driver" if backend == "rtx" else "mac-metal"),
            f"{run_id}: backend")
    require(receipt.get("run_id") == run_id, f"{run_id}: id")
    require(receipt.get("mode") == mode, f"{run_id}: mode")
    require(receipt.get("canonical_launch") == identity["launch"],
            f"{run_id}: launch")
    require(receipt.get("poison_byte") == poison, f"{run_id}: poison")
    require(isinstance(receipt.get("process_id"), int), f"{run_id}: process")
    require((receipt.get("module_token") or receipt.get("source_module_token")) == 27,
            f"{run_id}: module token")
    require((receipt.get("module_bytes") or receipt.get("source_module_bytes")) == 4024,
            f"{run_id}: module bytes")
    require((receipt.get("module_sha256") or receipt.get("source_module_sha256")) == MODULE_SHA,
            f"{run_id}: module hash")
    require(receipt.get("trace_raw_parameter_sha256") == identity["raw"],
            f"{run_id}: traced raw params")
    require(receipt.get("live_address_free_parameter_sha256") ==
            identity["address_free"], f"{run_id}: live params")
    require(receipt.get("address_free_parameter_sha256") == identity["address_free"],
            f"{run_id}: address-free params")
    require(receipt.get("packed_parameter_bytes") == 16, f"{run_id}: params")
    require((receipt.get("grid") or receipt.get("source_grid")) == [1, 1, 1],
            f"{run_id}: grid")
    require((receipt.get("block") or receipt.get("source_block")) == [64, 1, 1],
            f"{run_id}: block")
    require(receipt.get("tensor_input_count") == 0, f"{run_id}: inputs")
    require(receipt.get("input_generator") == GENERATOR, f"{run_id}: generator")
    require(receipt.get("element_count") == 1, f"{run_id}: count")
    require(receipt.get("requested_value") == identity["value"] and
            receipt.get("actual_value") == identity["value"],
            f"{run_id}: value")
    require(receipt.get("output_bytes") == 4, f"{run_id}: output bytes")
    require(receipt.get("canary_mismatched_bytes") == 0, f"{run_id}: canary")
    require(receipt.get("untouched_payload_bytes") == 0,
            f"{run_id}: write coverage")
    require(receipt.get("expected_output_fed_to_execution") is False,
            f"{run_id}: expected ingress")
    require(receipt.get("archived_intermediates_consumed") is False,
            f"{run_id}: archive ingress")
    require(receipt.get("address_fields_recorded") is False,
            f"{run_id}: addresses")
    require(receipt.get("result") == "pass", f"{run_id}: result")
    output = (root / run_id / "output-u32.bin").read_bytes()
    require(len(output) == 4 and sha(output) == identity["output"] and
            receipt.get("output_sha256") == identity["output"],
            f"{run_id}: output identity")
    return receipt, output


def build_backend(root: Path, backend: str) -> dict[str, Any]:
    rows, outputs = [], {}
    for run_id, mode, poison in RUNS:
        receipt, output = validate_run(root, run_id, mode, poison, backend)
        outputs[run_id] = output
        rows.append({"run_id": run_id, "mode": mode, "poison_byte": poison,
                     "process_id": receipt["process_id"],
                     "output_sha256": receipt["output_sha256"]})
    require(len({row["process_id"] for row in rows}) == 8,
            "fresh process gate")
    mode_rows = {}
    for mode in ("reset", "generated"):
        selected = [row for row in rows if row["mode"] == mode]
        first = outputs[selected[0]["run_id"]]
        require(len(selected) == 4 and
                all(outputs[row["run_id"]] == first for row in selected[1:]),
                f"{mode}: repeat/poison invariance")
        mode_rows[mode] = {
            "canonical_launch": MODES[mode]["launch"],
            "value": MODES[mode]["value"],
            "trace_raw_parameter_sha256": MODES[mode]["raw"],
            "live_address_free_parameter_sha256":
                MODES[mode]["address_free"],
            "address_free_parameter_sha256": MODES[mode]["address_free"],
            "output_sha256": MODES[mode]["output"],
        }
    document: dict[str, Any] = {
        "schema": BACKEND_SCHEMA,
        "backend": "rtx-cuda-driver" if backend == "rtx" else "mac-metal",
        "selected_units": ["launch-000010", "launch-000012"],
        "module_token": 27,
        "module_sha256": MODULE_SHA,
        "input_generator": GENERATOR,
        "tensor_input_count": 0,
        "runs": rows,
        "modes": mode_rows,
        "fresh_process_count": 8,
        "two_process_repeat_per_poison": True,
        "two_poison_output_invariance": True,
        "full_payload_write_coverage": True,
        "canary_integrity": True,
        "expected_output_fed_to_execution": False,
        "archived_intermediates_consumed": False,
        "address_fields_recorded": False,
        "result": "pass",
    }
    document["receipt_root_sha256"] = canonical(document)
    return document


def build_final(rtx_root: Path, mac_root: Path) -> dict[str, Any]:
    rtx, mac = build_backend(rtx_root, "rtx"), build_backend(mac_root, "mac")
    comparisons = []
    for run_id, mode, poison in RUNS:
        left = (rtx_root / run_id / "output-u32.bin").read_bytes()
        right = (mac_root / run_id / "output-u32.bin").read_bytes()
        mismatch = sum(a != b for a, b in zip(left, right))
        require(len(left) == len(right) == 4 and mismatch == 0,
                f"{run_id}: parity")
        comparisons.append({"run_id": run_id, "mode": mode,
                            "poison_byte": poison, "bytes": 4,
                            "sha256": sha(left), "mismatched_bytes": 0})
    document: dict[str, Any] = {
        "schema": FINAL_SCHEMA,
        "canonical_launches": [10, 12],
        "module_token": 27,
        "module_sha256": MODULE_SHA,
        "modes": rtx["modes"],
        "comparisons": comparisons,
        "rtx_backend_root_sha256": rtx["receipt_root_sha256"],
        "mac_backend_root_sha256": mac["receipt_root_sha256"],
        "early_inventory_root_sha256":
            "30ff3228e9e910a1c07fb5011a6e904fddbda620d64611039381304de4cfa6ed",
        "optional_absence_proof_root_sha256":
            "945e01721f39931ae11a7d34b4780e17dcde7c982cff26c6205685928a61d7b6",
        "fresh_process_count_per_backend": 8,
        "two_process_repeat_per_poison": True,
        "two_poison_output_invariance": True,
        "full_payload_write_coverage": True,
        "canary_integrity": True,
        "tensor_input_count": 0,
        "expected_output_fed_to_execution": False,
        "archived_intermediates_consumed": False,
        "address_fields_recorded": False,
        "claims": {
            "launch10_reset_status_exact": True,
            "launch12_generated_status_exact": True,
            "optional_profile_completeness_controls": True,
            "arithmetic_model_path_affected_when_absent": False,
            "generated_frame_affected_when_absent": False,
            "other_early_launches_inferred": False,
            "complete_frame_metal_parity_proven": False,
        },
        "result": "pass",
    }
    document["receipt_root_sha256"] = canonical(document)
    return document


def validate_final(value: dict[str, Any]) -> None:
    reject_addresses(value)
    require(value.get("schema") == FINAL_SCHEMA, "final schema")
    require(rooted(value), "final root")
    require(value.get("canonical_launches") == [10, 12],
            "canonical launches")
    require(value.get("module_token") == 27 and
            value.get("module_sha256") == MODULE_SHA, "module identity")
    require(value.get("early_inventory_root_sha256") ==
            "30ff3228e9e910a1c07fb5011a6e904fddbda620d64611039381304de4cfa6ed",
            "early inventory provenance")
    require(value.get("optional_absence_proof_root_sha256") ==
            "945e01721f39931ae11a7d34b4780e17dcde7c982cff26c6205685928a61d7b6",
            "optional absence provenance")
    require(value.get("modes") == {
        mode: {
            "canonical_launch": identity["launch"],
            "value": identity["value"],
            "trace_raw_parameter_sha256": identity["raw"],
            "live_address_free_parameter_sha256": identity["address_free"],
            "address_free_parameter_sha256": identity["address_free"],
            "output_sha256": identity["output"],
        }
        for mode, identity in MODES.items()
    }, "mode identities")
    require(value.get("fresh_process_count_per_backend") == 8,
            "fresh process count")
    require(value.get("two_process_repeat_per_poison") is True,
            "process repeat")
    require(value.get("full_payload_write_coverage") is True, "coverage")
    require(value.get("canary_integrity") is True, "canaries")
    require(value.get("tensor_input_count") == 0, "tensor input count")
    require(value.get("two_poison_output_invariance") is True, "poison")
    require(value.get("expected_output_fed_to_execution") is False,
            "expected ingress")
    require(value.get("archived_intermediates_consumed") is False,
            "archive ingress")
    require(value.get("address_fields_recorded") is False, "addresses")
    require(value.get("result") == "pass", "result")
    expected_comparisons = [
        {"run_id": run_id, "mode": mode, "poison_byte": poison,
         "bytes": 4, "sha256": MODES[mode]["output"],
         "mismatched_bytes": 0}
        for run_id, mode, poison in RUNS
    ]
    require(value.get("comparisons") == expected_comparisons, "comparisons")
    require(value.get("claims") == {
        "launch10_reset_status_exact": True,
        "launch12_generated_status_exact": True,
        "optional_profile_completeness_controls": True,
        "arithmetic_model_path_affected_when_absent": False,
        "generated_frame_affected_when_absent": False,
        "other_early_launches_inferred": False,
        "complete_frame_metal_parity_proven": False,
    }, "claim boundary")


def build_audit(value: dict[str, Any]) -> dict[str, Any]:
    validate_final(value)
    mutations: list[tuple[str, Callable[[dict[str, Any]], None]]] = [
        ("output_mismatch", lambda x: x["comparisons"][0].__setitem__("mismatched_bytes", 1)),
        ("module_drift", lambda x: x.__setitem__("module_sha256", "0" * 64)),
        ("coverage_gate", lambda x: x.__setitem__("full_payload_write_coverage", False)),
        ("poison_gate", lambda x: x.__setitem__("two_poison_output_invariance", False)),
        ("expected_ingress", lambda x: x.__setitem__("expected_output_fed_to_execution", True)),
        ("archive_ingress", lambda x: x.__setitem__("archived_intermediates_consumed", True)),
        ("address_leak", lambda x: x.__setitem__("device_pointer", "0x1")),
        ("result_drift", lambda x: x.__setitem__("result", "fail")),
    ]
    outcomes = []
    for name, mutation in mutations:
        changed = copy.deepcopy(value)
        mutation(changed)
        # Recompute the integrity root so every negative exercises the semantic
        # validator rather than failing only because the document was edited.
        reroot(changed)
        try:
            validate_final(changed)
        except VerificationError:
            outcomes.append({"mutation": name, "rejected": True})
            continue
        raise VerificationError(f"mutation accepted: {name}")
    document: dict[str, Any] = {
        "schema": "neuralstudio.fg.early-status-l10-l12-negative-audit.v1",
        "accepted_receipt_root_sha256": value["receipt_root_sha256"],
        "mutations": outcomes,
        "mutation_count": len(outcomes),
        "all_mutations_rejected": True,
        "address_fields_recorded": False,
        "result": "pass",
    }
    document["receipt_root_sha256"] = canonical(document)
    return document


def write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
    except FileExistsError as error:
        raise VerificationError(f"refusing to overwrite {path}") from error


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("backend")
    p.add_argument("--backend", choices=("rtx", "mac"), required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("final")
    p.add_argument("--rtx-root", type=Path, required=True)
    p.add_argument("--mac-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("audit")
    p.add_argument("--accepted", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "backend":
            value = build_backend(args.root, args.backend)
        elif args.command == "final":
            value = build_final(args.rtx_root, args.mac_root)
        else:
            value = build_audit(load(args.accepted))
        write_exclusive(args.output, value)
        print(f"receipt_root_sha256={value['receipt_root_sha256']}")
        print("result=pass")
        return 0
    except (VerificationError, OSError) as error:
        print(f"verification_error={error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
