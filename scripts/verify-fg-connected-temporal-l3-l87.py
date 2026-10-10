#!/usr/bin/env python3
"""Verify the live reset + two-generated-frame selected FG chain."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import struct
from pathlib import Path
from typing import Any

FRAME_OUTPUTS = [
    "c9ca926497653ec25cbab6cb56737ea271a54a12cd1bad96e92d4d31d4abe98c",
    "c644b6563f1c7f7ff30118d92383b814254ccfad944f86c75e42c9f4204be3fa",
]
ALLOCATION37 = [
    "1e76efeec100162f35091e71f4c779d15e7e4bcc8ba00f2f7ed49f5727b885af",
    "73f84d6309511414a1b6cd122a4e47f5bde6c73c8723c3fc05f59d9ec6095fae",
]
ALLOCATION29 = [
    "26d1174d4e2b8524335721dcd6b385e7ed2a24ab8bb46d47325c1d405d61cd2d",
    "1d6725923a102f5c20a8df5b01df250fd60f4ab6de3bf55ef30525995f30dc82",
]
STATUS_ZERO = "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119"


class VerificationError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise VerificationError(message)


def load(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text())
    require(isinstance(result, dict), f"object required: {path}")
    return result


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical(value: Any) -> str:
    return sha(json.dumps(value, sort_keys=True,
                          separators=(",", ":")).encode())


def expected_keys() -> list[str]:
    result = [f"reset-launch-{launch:06d}" for launch in range(3, 12)]
    for frame in (1, 2):
        result += [f"generated-{frame}-launch-{launch:06d}"
                   for launch in range(12, 88)]
    return result


EXPECTED_KEYS = expected_keys()


def validate_temporal_receipt(
    receipt: dict[str, Any], early: dict[str, Any], tail1: dict[str, Any],
    tail2: dict[str, Any], backend: str, run_name: str,
) -> dict[str, str]:
    require(receipt.get("schema") ==
            "neuralstudio.fg.connected-temporal-l3-l87-run.v1",
            f"{run_name}: schema")
    require(receipt.get("backend") == backend, f"{run_name}: backend")
    require(receipt.get("run_id") == run_name, f"{run_name}: run id")
    require(receipt.get("process_id") == early.get("process_id") ==
            tail1.get("process_id") == tail2.get("process_id"),
            f"{run_name}: same process")
    require(receipt.get("phase_launch_counts") == [9, 76, 76] and
            receipt.get("launch_count") == 161, f"{run_name}: launch count")
    checkpoints = receipt.get("checkpoint_sha256")
    require(isinstance(checkpoints, dict) and set(checkpoints) ==
            set(EXPECTED_KEYS) and len(checkpoints) == 161,
            f"{run_name}: checkpoint coverage")
    expected = dict(early["checkpoint_sha256"])
    expected.update({f"generated-1-{key}": value
                     for key, value in tail1["checkpoint_sha256"].items()})
    expected.update({f"generated-2-{key}": value
                     for key, value in tail2["checkpoint_sha256"].items()})
    require(checkpoints == expected, f"{run_name}: checkpoint provenance")
    require(checkpoints["generated-1-launch-000012"] == STATUS_ZERO and
            checkpoints["generated-2-launch-000012"] == STATUS_ZERO,
            f"{run_name}: optional status profile")
    require(receipt.get("allocation37_sha256") == ALLOCATION37 and
            receipt.get("allocation29_sha256") == ALLOCATION29,
            f"{run_name}: persistent state roots")
    require(receipt.get("final_output_sha256") == FRAME_OUTPUTS and
            receipt.get("final_output_bytes") == [921600, 921600],
            f"{run_name}: final frames")
    require(early.get("allocation29_runtime_source") ==
            "generated-half-0.5-then-live-L52" and
            early.get("allocation29_capture_bytes_consumed") is False,
            f"{run_name}: allocation29 provenance")
    require(early.get("allocation37_poison_halfwords") == 0 and
            early.get("allocation29_poison_halfwords") == 0 and
            early.get("input_mismatched_bytes") == 0 and
            early.get("canary_mismatched_bytes") == 0,
            f"{run_name}: early integrity")
    for index, tail in enumerate((tail1, tail2), 1):
        require(tail.get("input_model_mismatched_bytes") == 0 and
                tail.get("canary_mismatched_bytes") == 0 and
                tail.get("readonly_tail_mismatched_bytes") == 0,
                f"{run_name}: generated-{index} integrity")
        require(tail.get("host_intermediate_ingress") is False and
                tail.get("expected_or_archived_tensor_ingress") is False,
                f"{run_name}: generated-{index} ingress")
    require(receipt.get("live_device_handoff_l19_l20") is True and
            receipt.get("host_intermediate_ingress") is False and
            receipt.get("expected_or_archived_tensor_ingress") is False and
            receipt.get("address_fields_recorded") is False,
            f"{run_name}: handoff/receipt policy")
    require(receipt.get("result") == early.get("result") ==
            tail1.get("result") == tail2.get("result") == "pass",
            f"{run_name}: result")
    return checkpoints


def validate_backend(root: Path, backend: str) -> dict[str, Any]:
    runs = sorted(path for path in root.glob("connected-temporal-l3-l87-*")
                  if path.is_dir())
    require(len(runs) == 4, f"{backend}: run count")
    maps: list[dict[str, str]] = []
    pids: list[int] = []
    poisons: list[int] = []
    for run in runs:
        early = load(run / "run-receipt.json")
        tail1 = load(run / "generated-1/l20-l87/run-receipt.json")
        tail2 = load(run / "generated-2/l20-l87/run-receipt.json")
        receipt = load(run / "temporal-receipt.json")
        maps.append(validate_temporal_receipt(receipt, early, tail1, tail2,
                                              backend, run.name))
        pids.append(receipt["process_id"])
        poisons.append(receipt["poison_byte"])
        for frame, expected_output in enumerate(FRAME_OUTPUTS, 1):
            output = (run / f"generated-{frame}/l20-l87/output.bin")
            if output.exists():
                value = output.read_bytes()
                require(len(value) == 921600 and sha(value) == expected_output,
                        f"{run.name}: generated-{frame} output artifact")
        for frame, expected_state in enumerate(ALLOCATION29, 1):
            state = run / f"generated-{frame}/allocation29-post-l52.r16f"
            if state.exists():
                value = state.read_bytes()
                require(len(value) == 115200 and sha(value) == expected_state,
                        f"{run.name}: allocation29 frame {frame}")
    require(len(set(pids)) == 4, f"{backend}: fresh process ids")
    require(sorted(poisons) == [90, 90, 165, 165],
            f"{backend}: poison schedule")
    require(all(item == maps[0] for item in maps[1:]),
            f"{backend}: two-poison invariance")
    return {"backend": backend, "checkpoint_sha256": maps[0],
            "checkpoint_root_sha256": canonical(maps[0]),
            "fresh_process_count": 4, "poison_bytes": [90, 165],
            "result": "pass"}


def verify(mac_root: Path, rtx_root: Path) -> dict[str, Any]:
    mac = validate_backend(mac_root, "mac-metal")
    rtx = validate_backend(rtx_root,
                           "rtx-cuda-external-memory-d3d12-nvapi")
    require(mac["checkpoint_sha256"] == rtx["checkpoint_sha256"],
            "cross-backend checkpoint mismatch")
    mac_run = sorted(mac_root.glob("connected-temporal-l3-l87-*"))[0]
    rtx_run = sorted(rtx_root.glob("connected-temporal-l3-l87-*"))[0]
    for frame in (1, 2):
        mac_output = (mac_run / f"generated-{frame}/l20-l87/output.bin").read_bytes()
        rtx_output = (rtx_run / f"generated-{frame}/l20-l87/output.bin").read_bytes()
        require(mac_output == rtx_output,
                f"generated-{frame} final byte mismatch")
    mac_l22 = (mac_run / "generated-2/l20-l87/l22-debug.bin").read_bytes()
    rtx_l22 = (rtx_run / "generated-2/l20-l87/l22-debug.bin").read_bytes()
    require(mac_l22 == rtx_l22, "L22 debug artifact mismatch")
    values = struct.unpack(f"<{len(mac_l22)//2}H", mac_l22)
    require(values[17215] == 0x0000 and values[30084] == 0x8000,
            "HMMA signed-zero semantic drift")
    result = {
        "schema": "neuralstudio.fg.connected-temporal-l3-l87-verification.v1",
        "scope": "independent deterministic 640x360 reset plus two generated frames",
        "phase_launch_counts": [9, 76, 76], "launch_count": 161,
        "cross_backend_checkpoint_root_sha256":
            mac["checkpoint_root_sha256"],
        "allocation37_sha256": ALLOCATION37,
        "allocation29_sha256": ALLOCATION29,
        "final_output_sha256": FRAME_OUTPUTS,
        "fresh_processes_per_backend": 4,
        "two_poison_checkpoint_invariance": True,
        "same_process_all_161_launches": True,
        "live_allocation29_l52_to_next_l19": True,
        "hmma_rounded_negative_zero_canonicalized": True,
        "input_model_immutability": True, "canary_integrity": True,
        "full_write_coverage": True,
        "expected_or_archived_tensor_ingress": False,
        "address_fields_recorded": False,
        "claims": {"deterministic_three_frame_temporal_parity": True,
                   "production_game_resources": False,
                   "long_sequence_temporal_parity": False,
                   "netal_integration": False, "performance": False},
        "result": "pass",
    }
    result["verification_root_sha256"] = canonical(result)
    return result


def negative_audit(mac_root: Path, rtx_root: Path) -> dict[str, Any]:
    run = sorted(mac_root.glob("connected-temporal-l3-l87-*"))[0]
    early = load(run / "run-receipt.json")
    tail1 = load(run / "generated-1/l20-l87/run-receipt.json")
    tail2 = load(run / "generated-2/l20-l87/run-receipt.json")
    original = load(run / "temporal-receipt.json")
    mutations = {
        "checkpoint": lambda x: x["checkpoint_sha256"].__setitem__(
            "generated-2-launch-000022", "0" * 64),
        "allocation29": lambda x: x["allocation29_sha256"].__setitem__(
            0, "1" * 64),
        "same-process": lambda x: x.__setitem__("process_id",
                                                 x["process_id"] + 1),
        "ingress": lambda x: x.__setitem__(
            "expected_or_archived_tensor_ingress", True),
        "launch-count": lambda x: x.__setitem__("launch_count", 160),
        "frame-output": lambda x: x["final_output_sha256"].__setitem__(
            1, "f" * 64),
    }
    rejected = []
    for name, mutate in mutations.items():
        candidate = copy.deepcopy(original)
        mutate(candidate)
        try:
            validate_temporal_receipt(candidate, early, tail1, tail2,
                                      "mac-metal", run.name)
        except VerificationError as error:
            rejected.append({"mutation": name,
                             "observed_error": str(error)})
        else:
            raise VerificationError(f"mutation accepted: {name}")
    baseline = verify(mac_root, rtx_root)
    return {"schema":
            "neuralstudio.fg.connected-temporal-l3-l87-negative-audit.v1",
            "baseline_verification_root_sha256":
                baseline["verification_root_sha256"],
            "mutations": rejected, "mutation_count": len(rejected),
            "all_mutations_rejected": len(rejected) == len(mutations),
            "result": "pass"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("verify", "negative-audit"))
    parser.add_argument("--mac", required=True, type=Path)
    parser.add_argument("--rtx", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = (verify(args.mac, args.rtx) if args.mode == "verify" else
              negative_audit(args.mac, args.rtx))
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        if args.output.exists():
            raise VerificationError(f"preserve existing output: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationError as error:
        raise SystemExit(f"verification_error={error}")
