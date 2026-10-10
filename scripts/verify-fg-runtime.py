#!/usr/bin/env python3
"""Verify the stateful Metal FG runtime against the accepted long16 roots."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Callable


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def load(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise VerificationError(f"receipt read failed: {path}: {error}") from error


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded)


def expected_outputs(reference: Path) -> list[str]:
    receipt = load(reference)
    outputs = receipt.get("final_output_sha256")
    require(receipt.get("result") == "pass", "reference result")
    require(isinstance(outputs, list) and len(outputs) == 15,
            "reference output roots")
    require(all(isinstance(value, str) and len(value) == 64 for value in outputs),
            "reference output hash shape")
    return outputs


def expected_checkpoint_root(reference: Path) -> str:
    root = load(reference).get("checkpoint_root_sha256")
    require(isinstance(root, str) and len(root) == 64,
            "reference checkpoint root")
    return root


def validate_run(run: Path, outputs: list[str], checkpoint_root: str,
                 ex2_mode: str, auxiliary_mode: str,
                 override: dict[str, Any] | None = None) -> dict[str, Any]:
    receipt = override or load(run / "run-receipt.json")
    require(receipt.get("schema") == "neuralstudio.fg.runtime-run.v1",
            f"{run.name}: schema")
    require(receipt.get("backend") == "mac-metal-runtime-prototype",
            f"{run.name}: backend")
    require(receipt.get("ex2_mode") == ex2_mode, f"{run.name}: EX2 mode")
    require(receipt.get("auxiliary_mode") == auxiliary_mode,
            f"{run.name}: auxiliary mode")
    require(receipt.get("run_id") == run.name, f"{run.name}: run identity")
    expected_poison = 90 if "-poison-5a-" in run.name else 165
    require(receipt.get("poison_byte") == expected_poison,
            f"{run.name}: poison identity")
    require(receipt.get("rendered_frame_count") == 16 and
            receipt.get("reset_launch_count") == 9 and
            receipt.get("generated_launch_count") == 1140,
            f"{run.name}: launch coverage")
    require(receipt.get("output_sha256") == outputs, f"{run.name}: output roots")
    checkpoints = receipt.get("checkpoint_sha256")
    expected_keys = {
        *(f"reset-launch-{launch:06d}" for launch in range(3, 12)),
        *(f"generated-{frame}-launch-{launch:06d}"
          for frame in range(1, 16) for launch in range(12, 88)),
    }
    require(isinstance(checkpoints, dict) and len(checkpoints) == 1149 and
            set(checkpoints) == expected_keys and
            canonical(checkpoints) == checkpoint_root,
            f"{run.name}: checkpoint map")
    allocations = receipt.get("resource_allocations_per_generated_frame")
    require(isinstance(allocations, list) and len(allocations) == 15 and
            isinstance(allocations[0], int) and allocations[0] > 0 and
            allocations[1:] == [0] * 14,
            f"{run.name}: per-frame Metal allocations")
    require(receipt.get("metal_resource_allocations_after_warmup") == 0 and
            receipt.get("no_resource_allocation_after_warmup") is True and
            receipt.get("persistent_tail_resources") is True and
            receipt.get("model_and_architecture_buffers_retained") is True,
            f"{run.name}: persistence")
    expected_auxiliary_profile = ("live-early-resource-aliases"
                                  if auxiliary_mode == "live"
                                  else "accepted-internal-small-flow")
    require(receipt.get("caller_owned_current_previous_rgba8") is True and
            receipt.get("caller_owned_motion_rg16f_depth_r32f") is True and
            receipt.get("caller_owned_generated_target_rgba8") is True and
            receipt.get("l53_l55_auxiliary_profile") ==
                expected_auxiliary_profile and
            receipt.get("l53_l55_auxiliary_caller_owned") is False and
            receipt.get("l52_l53_l55_live_early_resource_aliases") is
                (auxiliary_mode == "live"),
            f"{run.name}: resource provenance")
    require(receipt.get("caller_owned_input_mismatches") == 0 and
            receipt.get("runtime_input_mismatched_bytes") == 0 and
            receipt.get("runtime_canary_mismatched_bytes") == 0,
            f"{run.name}: immutability/canaries")
    require(receipt.get("rejected_output_unchanged") is True and
            receipt.get("reset_replay_exact") is True and
            receipt.get("non_audit_fast_path_exact") is True,
            f"{run.name}: lifecycle/fallback")
    require(receipt.get("tail_probe_file_io") is False and
            receipt.get("expected_or_archived_intermediate_ingress") is False and
            receipt.get("cpu_allocations_counted") is False and
            receipt.get("result") == "pass", f"{run.name}: policy")
    allowed = {"run-receipt.json", "stdout.log", "stderr.log"}
    allowed.update(f"generated-{frame}.rgba8" for frame in range(1, 16))
    actual = {str(path.relative_to(run)) for path in run.rglob("*") if path.is_file()}
    require(actual == allowed, f"{run.name}: unexpected runtime artifact")
    for frame, expected in enumerate(outputs, 1):
        path = run / f"generated-{frame}.rgba8"
        data = path.read_bytes()
        require(len(data) == 921600 and sha256(data) == expected,
                f"{run.name}: generated frame {frame}")
    return receipt


def validate_process_set(receipts: list[dict[str, Any]]) -> None:
    require(len(receipts) == 4, "four fresh runtime receipts")
    require(len({receipt["process_id"] for receipt in receipts}) == 4,
            "fresh process IDs")
    require(sorted(receipt["poison_byte"] for receipt in receipts) ==
            [90, 90, 165, 165], "two-poison coverage")


def validate_backend(backend: dict[str, Any], outputs: list[str],
                     checkpoint_root: str, ex2_mode: str,
                     auxiliary_mode: str) -> None:
    require(backend.get("schema") == "neuralstudio.fg.runtime-backend.v1" and
            backend.get("fresh_process_count") == 4 and
            backend.get("poison_bytes") == [90, 165] and
            backend.get("ex2_mode") == ex2_mode and
            backend.get("auxiliary_mode") == auxiliary_mode and
            backend.get("selected_launch_count") == 1149 and
            backend.get("checkpoint_root_sha256") == checkpoint_root and
            backend.get("output_sha256") == outputs and
            backend.get("persistent_early_resources") is True and
            backend.get("persistent_tail_resources") is True and
            backend.get("no_resource_allocation_after_warmup") is True and
            backend.get("tail_probe_file_io") is False and
            backend.get("l53_l55_auxiliary_caller_owned") is False and
            backend.get("l52_l53_l55_live_early_resource_aliases") is
                (auxiliary_mode == "live") and
            backend.get("expected_or_archived_intermediate_ingress") is False and
            backend.get("result") == "pass", "backend aggregate")


def validate_provenance(attempt: Path, provenance_path: Path,
                        override: dict[str, Any] | None = None) -> None:
    provenance = override or load(provenance_path)
    repository = Path(__file__).resolve().parent.parent
    require(provenance.get("schema") == "neuralstudio.fg.runtime-provenance.v1" and
            provenance.get("expected_or_archived_intermediate_ingress") is False and
            provenance.get("address_fields_recorded") is False and
            provenance.get("result") == "pass", "source provenance policy")
    source_pins = provenance.get("source_sha256")
    require(isinstance(source_pins, dict) and len(source_pins) >= 9,
            "source pin coverage")
    for relative, expected in source_pins.items():
        path = repository / relative
        require(path.is_file() and sha256(path.read_bytes()) == expected,
                f"source pin: {relative}")
    require(sha256((attempt / "build/runtime_tail.mm").read_bytes()) ==
            provenance.get("generated_runtime_tail_sha256"),
            "generated runtime tail pin")
    require(sha256((attempt / "fg-runtime-offline").read_bytes()) ==
            provenance.get("offline_host_binary_sha256"),
            "offline host binary pin")


def verify(attempt: Path, reference: Path, ex2_mode: str = "reference",
           auxiliary_mode: str = "internal",
           provenance: Path | None = None) -> dict[str, Any]:
    outputs = expected_outputs(reference)
    checkpoint_root = expected_checkpoint_root(reference)
    runs = sorted(path for path in attempt.glob("fg-runtime-poison-*-process-*")
                  if path.is_dir())
    require(len(runs) == 4, "four fresh runtime processes")
    receipts = [validate_run(run, outputs, checkpoint_root, ex2_mode,
                             auxiliary_mode)
                for run in runs]
    validate_process_set(receipts)
    require(len({canonical(receipt["output_sha256"]) for receipt in receipts}) == 1,
            "process/poison output invariance")
    backend = load(attempt / "backend-receipt.json")
    validate_backend(backend, outputs, checkpoint_root, ex2_mode,
                     auxiliary_mode)
    if provenance is not None:
        validate_provenance(attempt, provenance)
    result: dict[str, Any] = {
        "schema": "neuralstudio.fg.runtime-verification.v1",
        "scope": "stateful 640x360 Metal runtime over accepted internal auxiliary profile",
        "rendered_frame_count": 16,
        "selected_launch_count": 1149,
        "ex2_mode": ex2_mode,
        "auxiliary_mode": auxiliary_mode,
        "checkpoint_root_sha256": checkpoint_root,
        "fresh_process_count": 4,
        "poison_bytes": [90, 165],
        "output_sha256": outputs,
        "caller_owned_primary_inputs_and_output": True,
        "caller_owned_inputs_immutable": True,
        "reset_replay_exact": True,
        "non_audit_fast_path_exact": True,
        "rejected_output_unchanged": True,
        "persistent_model_architecture_and_scratch_resources": True,
        "metal_resource_allocations_per_generated_frame":
            backend["metal_resource_allocations_per_generated_frame"],
        "metal_resource_allocations_after_warmup": 0,
        "tail_probe_file_io": False,
        "source_build_pins_verified": provenance is not None,
        "expected_or_archived_intermediate_ingress": False,
        "limits": {
            "l53_l55_auxiliary_profile": (
                "live-early-resource-aliases" if auxiliary_mode == "live"
                else "accepted-internal-small-flow"),
            "l53_l55_auxiliary_caller_owned": False,
            "cpu_allocations_measured": False,
            "netal_integration": False,
            "performance": False,
        },
        "result": "pass",
    }
    result["verification_root_sha256"] = canonical(result)
    return result


def negative_audit(attempt: Path, reference: Path,
                   ex2_mode: str = "reference",
                   auxiliary_mode: str = "internal",
                   provenance: Path | None = None) -> dict[str, Any]:
    outputs = expected_outputs(reference)
    checkpoint_root = expected_checkpoint_root(reference)
    run = sorted(path for path in attempt.glob("fg-runtime-poison-*-process-*")
                 if path.is_dir())[0]
    original = load(run / "run-receipt.json")
    mutations: dict[str, Callable[[dict[str, Any]], None]] = {
        "output": lambda value: value["output_sha256"].__setitem__(7, "0" * 64),
        "checkpoint": lambda value: value["checkpoint_sha256"].__setitem__(
            "generated-9-launch-000052", "f" * 64),
        "allocation": lambda value: value[
            "resource_allocations_per_generated_frame"].__setitem__(8, 1),
        "ingress": lambda value: value.__setitem__(
            "expected_or_archived_intermediate_ingress", True),
        "canary": lambda value: value.__setitem__(
            "runtime_canary_mismatched_bytes", 1),
        "fallback": lambda value: value.__setitem__(
            "rejected_output_unchanged", False),
        "reset": lambda value: value.__setitem__("reset_replay_exact", False),
        "fast_path": lambda value: value.__setitem__(
            "non_audit_fast_path_exact", False),
        "auxiliary_claim": lambda value: value.__setitem__(
            "l53_l55_auxiliary_caller_owned", True),
        "poison": lambda value: value.__setitem__("poison_byte", 17),
        "ex2_mode": lambda value: value.__setitem__(
            "ex2_mode", "compact" if ex2_mode == "reference" else "reference"),
        "auxiliary_mode": lambda value: value.__setitem__(
            "auxiliary_mode", "live" if auxiliary_mode == "internal" else "internal"),
    }
    rejected: list[dict[str, str]] = []
    for name, mutate in mutations.items():
        changed = copy.deepcopy(original)
        mutate(changed)
        try:
            validate_run(run, outputs, checkpoint_root, ex2_mode,
                         auxiliary_mode, changed)
        except VerificationError as error:
            rejected.append({"mutation": name, "observed_error": str(error)})
        else:
            raise VerificationError(f"mutation accepted: {name}")
    receipts = [load(path / "run-receipt.json") for path in sorted(
        item for item in attempt.glob("fg-runtime-poison-*-process-*")
        if item.is_dir())]
    duplicate_process = copy.deepcopy(receipts)
    duplicate_process[1]["process_id"] = duplicate_process[0]["process_id"]
    try:
        validate_process_set(duplicate_process)
    except VerificationError as error:
        rejected.append({"mutation": "fresh_process",
                         "observed_error": str(error)})
    else:
        raise VerificationError("mutation accepted: fresh_process")
    backend = load(attempt / "backend-receipt.json")
    changed_backend = copy.deepcopy(backend)
    changed_backend["checkpoint_root_sha256"] = "0" * 64
    try:
        validate_backend(changed_backend, outputs, checkpoint_root, ex2_mode,
                         auxiliary_mode)
    except VerificationError as error:
        rejected.append({"mutation": "aggregate",
                         "observed_error": str(error)})
    else:
        raise VerificationError("mutation accepted: aggregate")
    if provenance is not None:
        changed_provenance = copy.deepcopy(load(provenance))
        first_source = sorted(changed_provenance["source_sha256"])[0]
        changed_provenance["source_sha256"][first_source] = "f" * 64
        try:
            validate_provenance(attempt, provenance, changed_provenance)
        except VerificationError as error:
            rejected.append({"mutation": "source_pin",
                             "observed_error": str(error)})
        else:
            raise VerificationError("mutation accepted: source_pin")
    baseline = verify(attempt, reference, ex2_mode, auxiliary_mode, provenance)
    return {
        "schema": "neuralstudio.fg.runtime-negative-audit.v1",
        "baseline_verification_root_sha256":
            baseline["verification_root_sha256"],
        "mutations": rejected,
        "all_mutations_rejected": True,
        "result": "pass",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("verify", "negative-audit"))
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--ex2-mode", choices=("reference", "compact"),
                        default="reference")
    parser.add_argument("--auxiliary-mode", choices=("internal", "live"),
                        default="internal")
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = (verify(arguments.attempt, arguments.reference,
                     arguments.ex2_mode,
                     arguments.auxiliary_mode,
                     arguments.provenance)
              if arguments.mode == "verify"
              else negative_audit(arguments.attempt, arguments.reference,
                                  arguments.ex2_mode,
                                  arguments.auxiliary_mode,
                                  arguments.provenance))
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        if arguments.output.exists():
            raise VerificationError(f"preserve existing output: {arguments.output}")
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationError as error:
        raise SystemExit(f"verification_error={error}")
