#!/usr/bin/env python3
"""Verify live allocation29 texture state at create and reset boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from pathlib import Path
from typing import Any


RUNTIME_SHA = "5d5cbf14d2727d47f93fd10bf77bd91708ae122482a6f86fd564971641ebd47b"
CAPTURE_SOURCE_SHA = "a3b610f9e57c4ca862f103c381d1a6e0c9e1d773ff5318283316c194e4f4c110"
HOST_SOURCE_SHA = "55d9203f90513b7a39636fcaf4a8cd2bbdfc2a037f50c7f9beab643c69678c9e"
HOST_SOURCE_COMMIT = "fb64cbe3f94e8ffa331587de81211095bd19398d"
ORCHESTRATOR_SHA = "ffbd9fdc9e57a54c3f7d52c4778cdc46144ec1bb32468adc708bce428e812462"
CREATE_SHA = "d8b443032200e143b1c49820b6d78c32d519553a3f63a38f48e8bc31da084f0a"
RESET_SHA = "69a13b293b84ff925c02204c56925db2cfae04b65630303dbbb4f664797d6f5a"
BYTES = 320 * 180 * 2


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    return sha(path.read_bytes())


def canonical(value: Any) -> str:
    return sha(json.dumps(value, sort_keys=True,
                          separators=(",", ":")).encode())


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    require(isinstance(value, dict), f"not an object: {path}")
    return value


def validate_manifest(manifest: dict[str, Any], capture_point: str,
                      expected_sha: str) -> None:
    require(manifest.get("schema") ==
            "neuralstudio.dlss_fg.default_resource_readback.v2",
            "texture manifest schema drift")
    require(manifest.get("capture_point") == capture_point,
            "texture manifest capture point drift")
    require(manifest.get("source_state_strategy") ==
            "texture_original_to_copy_source_and_restore",
            "texture state-restore strategy drift")
    require(manifest.get("explicit_source_transition") is True,
            "texture transition gate drift")
    require(manifest.get("direct_default_map") is False and
            manifest.get("bytes_fed_back_to_execution") is False,
            "texture capture isolation drift")
    require(manifest.get("captured_resources") == 1 and
            manifest.get("captured_logical_bytes") == BYTES,
            "texture manifest totals drift")
    entries = manifest.get("entries")
    require(isinstance(entries, list) and len(entries) == 1,
            "texture manifest entry count drift")
    entry = entries[0]
    require(entry.get("source_token") == 29 and
            entry.get("descriptor_dimension") == 3 and
            entry.get("descriptor_width") == 320,
            "allocation29 descriptor identity drift")
    require(entry.get("source_initial_state") == 8 and
            entry.get("resource_kind") == "texture2d",
            "allocation29 source-state identity drift")
    require(entry.get("row_bytes") == 640 and
            entry.get("row_count") == 180 and
            entry.get("readback_row_pitch") == 768,
            "allocation29 row-footprint drift")
    require(entry.get("logical_bytes") == BYTES and
            entry.get("sha256") == expected_sha,
            "allocation29 manifest payload drift")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-receipt", type=Path, required=True)
    parser.add_argument("--post-create-manifest", type=Path, required=True)
    parser.add_argument("--post-reset-manifest", type=Path, required=True)
    parser.add_argument("--post-create", type=Path, required=True)
    parser.add_argument("--post-reset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parent.parent
    try:
        capture = load(args.capture_receipt)
        create_manifest = load(args.post_create_manifest)
        reset_manifest = load(args.post_reset_manifest)
        create = args.post_create.read_bytes()
        reset = args.post_reset.read_bytes()
        require(capture.get("schema") ==
                "neuralstudio.fg.allocation29-texture-capture.v1" and
                capture.get("result") == "pass",
                "allocation29 capture receipt identity drift")
        require(capture.get("runtime_sha256") == RUNTIME_SHA,
                "allocation29 runtime identity drift")
        require(capture.get("host_source_sha256") == HOST_SOURCE_SHA and
                capture.get("allocation_capture_source_sha256") ==
                    CAPTURE_SOURCE_SHA,
                "allocation29 capture source identity drift")
        require(capture.get("expected_frame_bytes_fed_to_capture") is False and
                capture.get("captured_bytes_fed_back_to_execution") is False and
                capture.get("address_fields_recorded") is False,
                "allocation29 capture isolation drift")
        runs = capture.get("runs")
        require(isinstance(runs, list) and len(runs) == 4,
                "allocation29 fresh-run count drift")
        by_point: dict[str, list[dict[str, Any]]] = {}
        for run in runs:
            by_point.setdefault(str(run.get("capture_point")), []).append(run)
            require(run.get("dump_bytes") == BYTES,
                    "allocation29 run byte count drift")
        require(set(by_point) == {"post_create", "post_evaluate_0"},
                "allocation29 capture-point matrix drift")
        for point, expected in (("post_create", CREATE_SHA),
                                ("post_evaluate_0", RESET_SHA)):
            selected = by_point[point]
            require(len(selected) == 2 and
                    {item.get("poison_byte") for item in selected} == {0xA5, 0x5A},
                    f"allocation29 {point} poison matrix drift")
            require({item.get("dump_sha256") for item in selected} == {expected},
                    f"allocation29 {point} repeat mismatch")
            require(len({item.get("process_receipt_sha256") for item in selected}) == 2,
                    f"allocation29 {point} fresh-process receipts collapsed")
        validate_manifest(create_manifest,
                          "post_create_fence_pre_evaluate", CREATE_SHA)
        validate_manifest(reset_manifest,
                          "post_evaluate_0_fence_pre_evaluate_1", RESET_SHA)
        require(len(create) == len(reset) == BYTES,
                "allocation29 dump byte count drift")
        require(sha(create) == CREATE_SHA and sha(reset) == RESET_SHA,
                "allocation29 dump hash drift")
        require(create == bytes(BYTES),
                "allocation29 post-create state is not all zero")
        reset_half = struct.unpack("<%dH" % (BYTES // 2), reset)
        require(set(reset_half) == {0x3800},
                "allocation29 post-reset state is not uniform half 0.5")
        require(file_sha(repository /
                         "src/fg_host/runtime_allocation_capture.cpp") ==
                CAPTURE_SOURCE_SHA,
                "local allocation texture capture source drift")
        require(file_sha(repository /
                         "scripts/run-fg-allocation29-texture-capture.ps1") ==
                ORCHESTRATOR_SHA,
                "local allocation29 orchestration source drift")
        result = {
            "schema": "neuralstudio.fg.allocation29-texture-verification.v1",
            "runtime_sha256": RUNTIME_SHA,
            "host_source": {
                "commit": HOST_SOURCE_COMMIT,
                "sha256": HOST_SOURCE_SHA,
            },
            "capture_source_sha256": CAPTURE_SOURCE_SHA,
            "orchestrator_sha256": ORCHESTRATOR_SHA,
            "rig_capture_receipt_sha256": file_sha(args.capture_receipt),
            "fresh_process_count": 4,
            "capture_points": {
                "post_create": {
                    "bytes": BYTES,
                    "sha256": CREATE_SHA,
                    "value_census": {"half_zero": BYTES // 2},
                },
                "post_reset": {
                    "bytes": BYTES,
                    "sha256": RESET_SHA,
                    "value_census": {"half_0.5": BYTES // 2},
                },
            },
            "state_transition": "uniform half zero to uniform half 0.5",
            "selected_l19_initial_input_provenance": True,
            "captured_bytes_are_oracle_only": True,
            "captured_bytes_fed_back_to_execution": False,
            "claims": {
                "first_generated_frame_allocation29_state": True,
                "later_temporal_frames": False,
                "l19_numerical_parity": False,
                "connected_early_path": False,
            },
            "address_fields_recorded": False,
            "result": "pass",
        }
        result["receipt_root_sha256"] = canonical(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, args.output)
        print(f"receipt_root_sha256={result['receipt_root_sha256']}")
        print("result=pass")
        return 0
    except (OSError, ValueError, json.JSONDecodeError, VerificationError) as error:
        print(f"verification_error={error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
