#!/usr/bin/env python3
"""Verify the live selected FG L20-L87 cross-backend receipts."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_FINAL = "ac3552c65cffdad8e1cf7b4261c859fe42e3ced0c67f59052bf249f94b954da3"
EXPECTED_KEYS = [f"launch-{index:06d}" for index in range(20, 88)]
EXPECTED_ROOTS = {
    "selected_module_census_root_sha256":
        "324e579b133caf212ad9db44908a73ae6baa90ffbb4978696b51ae99cb114c26",
    "allocation_provenance_root_sha256":
        "8a687084ddbc9c0ebeefd4a12dabb98162691bb8dbf62a5505e531ef64c9c75c",
    "descriptor_provenance_root_sha256":
        "d3c72b68dcec1d1bb307239c7a782a4c49e06c92b121e8e74a1f0a63ba509733",
}


class VerificationError(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def validate_payloads(receipts: list[dict[str, Any]], backend: str) -> dict[str, Any]:
    require(len(receipts) == 4, f"{backend}: expected four run receipts")
    expected_backend = "mac-metal" if backend == "mac" else \
        "rtx-cuda-external-memory-d3d12-nvapi"
    for receipt in receipts:
        require(receipt.get("schema") ==
                "neuralstudio.fg.connected-l20-l87-run.v1",
                f"{backend}: schema drift")
        require(receipt.get("backend") == expected_backend,
                f"{backend}: backend drift")
        require(receipt.get("launch_range") == [20, 87] and
                receipt.get("launch_count") == 68,
                f"{backend}: launch range/count drift")
        require(list(receipt.get("checkpoint_sha256", {}).keys()) ==
                EXPECTED_KEYS, f"{backend}: checkpoint key drift")
        require(receipt.get("external_flow_scenario") == "small-flow",
                f"{backend}: unsupported flow scenario")
        require(receipt.get("l84_color_edge", "").startswith(
                    "l55-surface-a-b-live"),
                f"{backend}: L84 live surface edge missing")
        for key, expected in EXPECTED_ROOTS.items():
            require(receipt.get(key) == expected,
                    f"{backend}: provenance root drift: {key}")
        for key in ("input_model_mismatched_bytes",
                    "canary_mismatched_bytes",
                    "readonly_tail_mismatched_bytes"):
            require(receipt.get(key) == 0,
                    f"{backend}: integrity failure: {key}")
        require(receipt.get("live_device_handoff_count") == 67,
                f"{backend}: handoff count drift")
        require(receipt.get("host_intermediate_ingress") is False and
                receipt.get("expected_or_archived_tensor_ingress") is False,
                f"{backend}: forbidden execution ingress")
        require(receipt.get("address_fields_recorded") is False,
                f"{backend}: address-bearing receipt")
        require(receipt.get("final_output_bytes") == 921600,
                f"{backend}: final byte count drift")
        require(receipt.get("final_output_sha256") ==
                receipt["checkpoint_sha256"]["launch-000087"],
                f"{backend}: final/checkpoint disagreement")
        require(receipt.get("result") == "pass",
                f"{backend}: failed run receipt")
    require(len({receipt["process_id"] for receipt in receipts}) == 4,
            f"{backend}: fresh-process gate failed")
    poisons = sorted(receipt["poison_byte"] for receipt in receipts)
    require(poisons == [90, 90, 165, 165],
            f"{backend}: two-poison population drift")
    checkpoint_variants = {
        canonical_sha256(receipt["checkpoint_sha256"])
        for receipt in receipts
    }
    require(len(checkpoint_variants) == 1,
            f"{backend}: checkpoint poison/process variance")
    finals = {receipt["final_output_sha256"] for receipt in receipts}
    require(finals == {EXPECTED_FINAL},
            f"{backend}: unexpected final frame identity")
    return {
        "backend": backend,
        "fresh_process_count": 4,
        "poison_bytes": [90, 165],
        "checkpoint_sha256": receipts[0]["checkpoint_sha256"],
        "checkpoint_root_sha256":
            canonical_sha256(receipts[0]["checkpoint_sha256"]),
        "final_output_sha256": EXPECTED_FINAL,
        "final_full_write_coverage_from_two_poison_invariance": True,
        "input_model_immutable": True,
        "canaries_clean": True,
        "readonly_tail_immutable": True,
        "address_fields_recorded": False,
    }


def load_backend(root: Path, backend: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    receipt_paths = sorted(root.glob("*/run-receipt.json"))
    receipts = [json.loads(path.read_text()) for path in receipt_paths]
    summary = validate_payloads(receipts, backend)
    require(len(receipt_paths) == 4, f"{backend}: receipt discovery drift")
    for path, receipt in zip(receipt_paths, receipts, strict=True):
        output = path.parent / "output.bin"
        require(output.is_file(), f"{backend}: output artifact missing")
        data = output.read_bytes()
        require(len(data) == receipt["final_output_bytes"],
                f"{backend}: output file size drift")
        require(sha256_bytes(data) == receipt["final_output_sha256"],
                f"{backend}: output file hash drift")
    backend_receipt = root / ("rig-receipt.json" if backend == "rtx"
                              else "backend-receipt.json")
    require(backend_receipt.is_file(), f"{backend}: aggregate receipt missing")
    summary["aggregate_receipt_sha256"] = sha256_bytes(
        backend_receipt.read_bytes())
    return receipts, summary


def validate_cross(rtx: dict[str, Any], mac: dict[str, Any]) -> None:
    require(rtx["checkpoint_sha256"] == mac["checkpoint_sha256"],
            "RTX/Metal checkpoint mismatch")
    require(rtx["final_output_sha256"] == mac["final_output_sha256"],
            "RTX/Metal final mismatch")


def mutation_audit(rtx_receipts: list[dict[str, Any]],
                   mac_receipts: list[dict[str, Any]]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []

    def reject(name: str, mutate: Any, cross: bool = False) -> None:
        left = copy.deepcopy(rtx_receipts)
        right = copy.deepcopy(mac_receipts)
        mutate(left, right)
        rejected = False
        try:
            l_summary = validate_payloads(left, "rtx")
            r_summary = validate_payloads(right, "mac")
            if cross:
                validate_cross(l_summary, r_summary)
        except VerificationError:
            rejected = True
        require(rejected, f"mutation was accepted: {name}")
        cases.append({"name": name, "rejected": True})

    reject("checkpoint-l54", lambda left, right:
           left[0]["checkpoint_sha256"].__setitem__("launch-000054", "0"*64))
    reject("final-output", lambda left, right:
           left[0].__setitem__("final_output_sha256", "0"*64))
    reject("host-ingress", lambda left, right:
           left[0].__setitem__("host_intermediate_ingress", True))
    reject("canary", lambda left, right:
           left[0].__setitem__("canary_mismatched_bytes", 1))
    reject("readonly-tail", lambda left, right:
           left[0].__setitem__("readonly_tail_mismatched_bytes", 1))
    reject("pixel-flow-unsupported", lambda left, right:
           left[0].__setitem__("external_flow_scenario", "pixel-flow"))
    reject("wrong-l84-edge", lambda left, right:
           left[0].__setitem__("l84_color_edge", "original-source-textures"))
    reject("duplicate-process", lambda left, right:
           left[1].__setitem__("process_id", left[0]["process_id"]))
    reject("poison-collapse", lambda left, right:
           [receipt.__setitem__("poison_byte", 90) for receipt in left])
    reject("launch-count", lambda left, right:
           left[0].__setitem__("launch_count", 67))
    reject("cross-backend-l84", lambda left, right:
           [receipt["checkpoint_sha256"].__setitem__(
               "launch-000084", "1"*64) for receipt in right], cross=True)
    result = {
        "schema": "neuralstudio.fg.connected-l20-l87-negative-audit.v1",
        "mutation_count": len(cases),
        "rejected_count": len(cases),
        "failed_to_reject": 0,
        "cases": cases,
        "result": "pass",
    }
    result["receipt_root_sha256"] = canonical_sha256(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtx-root", type=Path, required=True)
    parser.add_argument("--mac-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mutation-output", type=Path, required=True)
    args = parser.parse_args()
    rtx_receipts, rtx = load_backend(args.rtx_root, "rtx")
    mac_receipts, mac = load_backend(args.mac_root, "mac")
    validate_cross(rtx, mac)
    result = {
        "schema": "neuralstudio.fg.connected-l20-l87-verification.v1",
        "scope": "selected launches L20-L87 on independent small-flow resources",
        "claims": {
            "complete_selected_launch_chain": True,
            "rtx_metal_byte_exact_all_68_checkpoints": True,
            "generated_frame_exact_within_l20_l87_scope": True,
            "preprocess_l0_l19_connected": False,
            "production_game_resources": False,
            "netal_integration": False,
            "temporal_modes": False,
            "performance": False,
        },
        "rtx": rtx,
        "mac": mac,
        "cross_backend_checkpoint_root_sha256": rtx["checkpoint_root_sha256"],
        "final_output_bytes": 921600,
        "final_output_sha256": EXPECTED_FINAL,
        "live_device_handoff_count": 67,
        "host_intermediate_ingress": False,
        "expected_or_archived_tensor_ingress": False,
        "address_fields_recorded": False,
        "result": "pass",
    }
    result["receipt_root_sha256"] = canonical_sha256(result)
    negative = mutation_audit(rtx_receipts, mac_receipts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.mutation_output.write_text(
        json.dumps(negative, indent=2, sort_keys=True) + "\n")
    print(f"receipt_root_sha256={result['receipt_root_sha256']}")
    print(f"negative_root_sha256={negative['receipt_root_sha256']}")
    print("result=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
