#!/usr/bin/env python3
"""Prove that address-free compositor receipts reject semantic mutations."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile
from typing import Any, Callable

ROOT_KEY = "receipt_root_sha256"


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def reroot(value: dict[str, Any]) -> None:
    value.pop(ROOT_KEY, None)
    value[ROOT_KEY] = hashlib.sha256(canonical(value)).hexdigest()


def mutate_primary_delta(accepted: dict[str, Any], _: dict[str, Any]) -> None:
    accepted["primary_rtx_mac_mismatched_bytes"] = 1


def mutate_whole_model(accepted: dict[str, Any], _: dict[str, Any]) -> None:
    accepted["claims"]["complete_frame_generation_model"] = True


def mutate_backend_output(accepted: dict[str, Any], _: dict[str, Any]) -> None:
    accepted["rtx_backend_receipt"]["primary_output_sha256"] = "0" * 64
    reroot(accepted["rtx_backend_receipt"])


def mutate_float_output(accepted: dict[str, Any], _: dict[str, Any]) -> None:
    accepted["diagnostic_float_output_sha256"] = "1" * 64


def mutate_ingress(accepted: dict[str, Any], _: dict[str, Any]) -> None:
    accepted["claims"]["expected_or_archived_tensor_ingress"] = True


def mutate_formula(accepted: dict[str, Any], _: dict[str, Any]) -> None:
    accepted["sust_unorm_characterization"]["selected_integer_formula"] = "native-metal"


def mutate_counterexample(accepted: dict[str, Any], _: dict[str, Any]) -> None:
    accepted["sust_unorm_characterization"]["native_metal_unorm_mismatched_bytes"] = 0


def mutate_remote_job(accepted: dict[str, Any], _: dict[str, Any]) -> None:
    accepted["remote_job_retained"] = "D:\\CodexLab\\wrong"


def mutate_abi_module(_: dict[str, Any], abi: dict[str, Any]) -> None:
    abi["module_sha256"] = "2" * 64


def mutate_abi_binding(_: dict[str, Any], abi: dict[str, Any]) -> None:
    abi["bindings"][0]["meaning"] = "unknown texture"


CASES: list[tuple[str, str, Callable[[dict[str, Any], dict[str, Any]], None]]] = [
    ("primary-delta", "accepted primary parity drift", mutate_primary_delta),
    ("whole-model-overclaim", "whole-model overclaim", mutate_whole_model),
    ("backend-output", "backend primary output drift", mutate_backend_output),
    ("float-output", "accepted float binding drift", mutate_float_output),
    ("expected-ingress", "accepted tensor ingress drift", mutate_ingress),
    ("conversion-formula", "SUST.P conversion formula drift", mutate_formula),
    ("counterexample", "SUST.P counterexample drift", mutate_counterexample),
    ("remote-job", "remote job identity drift", mutate_remote_job),
    ("abi-module", "ABI module drift", mutate_abi_module),
    ("abi-binding", "ABI mask binding drift", mutate_abi_binding),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accepted", type=pathlib.Path, required=True)
    parser.add_argument("--abi", type=pathlib.Path, required=True)
    parser.add_argument("--verifier", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    original_accepted = json.loads(args.accepted.read_text())
    original_abi = json.loads(args.abi.read_text())
    results = []
    with tempfile.TemporaryDirectory(prefix="fg-compositor-l87-mutations-") as temp:
        root = pathlib.Path(temp)
        for name, expected, mutation in CASES:
            accepted = copy.deepcopy(original_accepted)
            abi = copy.deepcopy(original_abi)
            mutation(accepted, abi)
            reroot(abi)
            accepted["abi_receipt_root_sha256"] = abi[ROOT_KEY]
            abi_path = root / f"{name}-abi.json"
            abi_path.write_bytes(json.dumps(abi, indent=2, sort_keys=True).encode() + b"\n")
            accepted["abi_receipt_sha256"] = hashlib.sha256(abi_path.read_bytes()).hexdigest()
            reroot(accepted)
            accepted_path = root / f"{name}-accepted.json"
            accepted_path.write_bytes(
                json.dumps(accepted, indent=2, sort_keys=True).encode() + b"\n")
            process = subprocess.run(
                [sys.executable, str(args.verifier), "receipt", "--accepted",
                 str(accepted_path), "--abi", str(abi_path)],
                text=True, capture_output=True, check=False)
            observed = process.stderr.strip()
            passed = process.returncode != 0 and expected in observed
            if not passed:
                print(f"mutation {name} was not rejected as expected: {observed}",
                      file=sys.stderr)
                return 1
            results.append({
                "name": name,
                "expected_error_fragment": expected,
                "observed_error": observed,
                "rejected": True,
            })
    output = {
        "schema": "neuralstudio.fg.compositor-l87-negative-audit.v1",
        "mutation_count": len(results),
        "all_mutations_rejected": True,
        "mutations": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(json.dumps(output, indent=2, sort_keys=True).encode() + b"\n")
    print(f"mutation_count={len(results)}")
    print("result=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
