#!/usr/bin/env python3
"""Verify the live, same-process selected FG chain from L3 through L87."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

RUN_SCHEMA = "neuralstudio.fg.connected-l3-l87-run.v1"
ALLOCATION37_SHA256 = (
    "1e76efeec100162f35091e71f4c779d15e7e4bcc8ba00f2f7ed49f5727b885af"
)
L20_SHA256 = (
    "84d91cb542569f657c59530f1e99efad3444cb50e482772077f7f842919da32a"
)
FINAL_SHA256 = (
    "c9ca926497653ec25cbab6cb56737ea271a54a12cd1bad96e92d4d31d4abe98c"
)
EXPECTED_KEYS = [f"launch-{launch:06d}" for launch in range(3, 88)]


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"object required: {path}")
    return value


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return sha256_bytes(encoded)


def validate_combined(
    combined: dict[str, Any], early: dict[str, Any], tail: dict[str, Any],
    backend: str, run_name: str,
) -> dict[str, str]:
    require(combined.get("schema") == RUN_SCHEMA, f"{run_name}: schema")
    require(combined.get("backend") == backend, f"{run_name}: backend")
    require(combined.get("run_id") == run_name, f"{run_name}: run id")
    require(combined.get("process_id") == early.get("process_id") ==
            tail.get("process_id"), f"{run_name}: same process")
    require(combined.get("poison_byte") in (0x5A, 0xA5),
            f"{run_name}: poison")
    require(combined.get("launch_range") == [3, 87] and
            combined.get("launch_count") == 85, f"{run_name}: coverage")
    checkpoints = combined.get("checkpoint_sha256")
    require(isinstance(checkpoints, dict) and list(checkpoints) == EXPECTED_KEYS,
            f"{run_name}: checkpoint sequence")
    require(checkpoints == {**early["checkpoint_sha256"],
                            **tail["checkpoint_sha256"]},
            f"{run_name}: checkpoint provenance")
    require(checkpoints["launch-000020"] == L20_SHA256,
            f"{run_name}: live L20 root")
    require(combined.get("allocation37_sha256") == ALLOCATION37_SHA256,
            f"{run_name}: allocation37 root")
    require(combined.get("live_device_handoff_l19_l20") is True and
            early.get("l20_live_handoff") is True,
            f"{run_name}: live handoff")
    require(early.get("allocation29_capture_bytes_consumed") is False and
            early.get("expected_or_archived_intermediate_ingress") is False,
            f"{run_name}: early ingress")
    require(tail.get("host_intermediate_ingress") is False and
            tail.get("expected_or_archived_tensor_ingress") is False and
            combined.get("host_intermediate_ingress") is False and
            combined.get("expected_or_archived_tensor_ingress") is False,
            f"{run_name}: tail ingress")
    require(early.get("input_mismatched_bytes") == 0 and
            early.get("canary_mismatched_bytes") == 0 and
            early.get("allocation37_poison_halfwords") == 0,
            f"{run_name}: early integrity")
    require(tail.get("input_model_mismatched_bytes") == 0 and
            tail.get("canary_mismatched_bytes") == 0 and
            tail.get("readonly_tail_mismatched_bytes") == 0,
            f"{run_name}: tail integrity")
    require(combined.get("address_fields_recorded") is False and
            early.get("address_fields_recorded") is False and
            tail.get("address_fields_recorded") is False,
            f"{run_name}: address-free")
    require(combined.get("final_output_bytes") == 921600 and
            combined.get("final_output_sha256") == FINAL_SHA256 ==
            tail.get("final_output_sha256"), f"{run_name}: final output")
    require(combined.get("result") == early.get("result") ==
            tail.get("result") == "pass", f"{run_name}: result")
    return checkpoints


def validate_backend(root: Path, backend: str) -> dict[str, Any]:
    runs = sorted(path for path in root.glob("connected-l3-l87-*")
                  if path.is_dir())
    require(len(runs) == 4, f"{backend}: four fresh run directories")
    maps: list[dict[str, str]] = []
    pids: list[int] = []
    poisons: list[int] = []
    for run in runs:
        early = load(run / "run-receipt.json")
        tail = load(run / "l20-l87" / "run-receipt.json")
        combined = load(run / "combined-receipt.json")
        maps.append(validate_combined(combined, early, tail, backend,
                                      run.name))
        pids.append(combined["process_id"])
        poisons.append(combined["poison_byte"])
        allocation = (run / "allocation37.fp16").read_bytes()
        output = (run / "l20-l87" / "output.bin").read_bytes()
        require(len(allocation) == 1474560 and
                sha256_bytes(allocation) == ALLOCATION37_SHA256,
                f"{run.name}: allocation37 artifact")
        require(len(output) == 921600 and sha256_bytes(output) == FINAL_SHA256,
                f"{run.name}: final artifact")
    require(len(set(pids)) == 4, f"{backend}: fresh process ids")
    require(sorted(poisons) == [0x5A, 0x5A, 0xA5, 0xA5],
            f"{backend}: two poisons/two processes")
    require(all(value == maps[0] for value in maps[1:]),
            f"{backend}: checkpoint poison invariance")
    return {
        "backend": backend,
        "fresh_process_count": 4,
        "poison_bytes": [0x5A, 0xA5],
        "checkpoint_sha256": maps[0],
        "checkpoint_root_sha256": canonical_sha256(maps[0]),
        "allocation37_sha256": ALLOCATION37_SHA256,
        "final_output_sha256": FINAL_SHA256,
        "result": "pass",
    }


def verify(mac_root: Path, rtx_root: Path) -> dict[str, Any]:
    mac = validate_backend(mac_root, "mac-metal")
    rtx = validate_backend(rtx_root,
                           "rtx-cuda-external-memory-d3d12-nvapi")
    require(mac["checkpoint_sha256"] == rtx["checkpoint_sha256"],
            "RTX/Metal checkpoint mismatch")
    first_mac = sorted(mac_root.glob("connected-l3-l87-*"))[0]
    first_rtx = sorted(rtx_root.glob("connected-l3-l87-*"))[0]
    require((first_mac / "l20-l87" / "output.bin").read_bytes() ==
            (first_rtx / "l20-l87" / "output.bin").read_bytes(),
            "RTX/Metal final bytes mismatch")
    return {
        "schema": "neuralstudio.fg.connected-l3-l87-verification.v1",
        "scope": "deterministic 640x360 first-generated-frame selected chain",
        "launch_range": [3, 87],
        "launch_count": 85,
        "allocation37_sha256": ALLOCATION37_SHA256,
        "l20_sha256": L20_SHA256,
        "final_output_bytes": 921600,
        "final_output_sha256": FINAL_SHA256,
        "cross_backend_checkpoint_root_sha256":
            mac["checkpoint_root_sha256"],
        "rtx": {key: value for key, value in rtx.items()
                if key != "checkpoint_sha256"},
        "mac": {key: value for key, value in mac.items()
                if key != "checkpoint_sha256"},
        "fresh_processes_per_backend": 4,
        "two_poison_checkpoint_invariance": True,
        "full_write_coverage": True,
        "same_process_all_85_launches": True,
        "live_device_handoff_l19_l20": True,
        "input_model_immutability": True,
        "canary_integrity": True,
        "expected_or_archived_tensor_ingress": False,
        "address_fields_recorded": False,
        "claims": {
            "deterministic_fixture_full_selected_chain_exact": True,
            "production_game_resources": False,
            "temporal_multiframe": False,
            "subrect_hudless_distortion_modes": False,
            "netal_integration": False,
            "performance": False,
        },
        "result": "pass",
    }


def negative_audit(mac_root: Path, rtx_root: Path) -> dict[str, Any]:
    run = sorted(mac_root.glob("connected-l3-l87-*"))[0]
    early = load(run / "run-receipt.json")
    tail = load(run / "l20-l87" / "run-receipt.json")
    original = load(run / "combined-receipt.json")
    mutations = {
        "checkpoint": lambda value: value["checkpoint_sha256"].__setitem__(
            "launch-000052", "0" * 64),
        "handoff": lambda value: value.__setitem__(
            "live_device_handoff_l19_l20", False),
        "same-process": lambda value: value.__setitem__(
            "process_id", value["process_id"] + 1),
        "ingress": lambda value: value.__setitem__(
            "expected_or_archived_tensor_ingress", True),
        "final-output": lambda value: value.__setitem__(
            "final_output_sha256", "f" * 64),
        "coverage": lambda value: value.__setitem__("launch_count", 84),
    }
    rejected: list[dict[str, str]] = []
    for name, mutate in mutations.items():
        value = copy.deepcopy(original)
        mutate(value)
        try:
            validate_combined(value, early, tail, "mac-metal", run.name)
        except VerificationError as error:
            rejected.append({"mutation": name, "observed_error": str(error)})
        else:
            raise VerificationError(f"mutation accepted: {name}")
    baseline = verify(mac_root, rtx_root)
    return {
        "schema": "neuralstudio.fg.connected-l3-l87-negative-audit.v1",
        "baseline_verification_sha256": canonical_sha256(baseline),
        "mutations": rejected,
        "mutation_count": len(rejected),
        "all_mutations_rejected": len(rejected) == len(mutations),
        "result": "pass",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("verify", "negative-audit"))
    parser.add_argument("--mac", type=Path, required=True)
    parser.add_argument("--rtx", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = (verify(args.mac, args.rtx) if args.mode == "verify" else
              negative_audit(args.mac, args.rtx))
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        if args.output.exists():
            raise VerificationError(f"preserve existing output: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationError as error:
        raise SystemExit(f"verification_error={error}")
