#!/usr/bin/env python3
"""Exercise semantic rejection gates for FG L55 receipts."""

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


def output_delta(a, _): a["rtx_mac_mismatched_bytes"]["surface_a"] = 1
def backend_output(a, _):
    a["rtx_backend_receipt"]["output_hashes"]["surface_a"] = "0" * 64
    reroot(a["rtx_backend_receipt"])
def backend_ingress(a, _):
    a["rtx_backend_receipt"]["expected_or_archived_tensor_ingress"] = True
    reroot(a["rtx_backend_receipt"])
def accepted_ingress(a, _): a["claims"]["expected_or_archived_tensor_ingress"] = True
def optional_branch(a, _): a["claims"]["optional_ui_branch"] = True
def multi_mode(a, _): a["claims"]["multi_sample_modes"] = True
def full_frame(a, _): a["claims"]["full_generated_frame"] = True
def netal(a, _): a["claims"]["netal_integration"] = True
def point_formula(_, b): b["selected_numerics"]["normalized_point_coordinate"] = "native-metal"
def abi_module(_, b): b["module_sha256"] = "1" * 64
def abi_binding(_, b): b["bindings"][0]["meaning"] = "unknown"
def native_counterexample(a, _): a["failure_counterexamples"]["native_metal_sampler_mismatched_bytes"] = 0
def fixed12_counterexample(a, _): a["failure_counterexamples"]["fixed12_point_sampler_mismatched_bytes"] = 1
def remote_job(a, _): a["remote_job_retained"] = "D:\\CodexLab\\wrong"


CASES: list[tuple[str, str, Callable[[dict[str, Any], dict[str, Any]], None]]] = [
    ("output-delta", "accepted output parity drift", output_delta),
    ("backend-output", "backend output binding drift", backend_output),
    ("backend-ingress", "backend receipt gate drift", backend_ingress),
    ("accepted-ingress", "accepted claim boundary drift", accepted_ingress),
    ("optional-branch", "accepted claim boundary drift", optional_branch),
    ("multi-mode", "accepted claim boundary drift", multi_mode),
    ("full-frame", "accepted claim boundary drift", full_frame),
    ("netal", "accepted claim boundary drift", netal),
    ("point-formula", "point-sampler primitive drift", point_formula),
    ("abi-module", "ABI identity/control drift", abi_module),
    ("abi-binding", "ABI resource binding drift", abi_binding),
    ("native-counterexample", "failure counterexample drift", native_counterexample),
    ("fixed12-counterexample", "failure counterexample drift", fixed12_counterexample),
    ("remote-job", "remote job identity drift", remote_job),
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
    with tempfile.TemporaryDirectory(prefix="fg-mid-l55-mutations-") as temporary:
        root = pathlib.Path(temporary)
        for name, expected, mutation in CASES:
            accepted = copy.deepcopy(original_accepted)
            abi = copy.deepcopy(original_abi)
            mutation(accepted, abi)
            reroot(abi)
            abi_path = root / f"{name}-abi.json"
            abi_path.write_bytes(json.dumps(abi, indent=2, sort_keys=True).encode() + b"\n")
            accepted["abi_receipt_root_sha256"] = abi[ROOT_KEY]
            accepted["abi_receipt_sha256"] = hashlib.sha256(abi_path.read_bytes()).hexdigest()
            reroot(accepted)
            accepted_path = root / f"{name}-accepted.json"
            accepted_path.write_bytes(
                json.dumps(accepted, indent=2, sort_keys=True).encode() + b"\n")
            process = subprocess.run(
                [sys.executable, str(args.verifier), "receipt", "--accepted",
                 str(accepted_path), "--abi", str(abi_path)],
                capture_output=True, text=True, check=False)
            observed = process.stderr.strip()
            if process.returncode == 0 or expected not in observed:
                print(f"mutation {name} rejection drift: {observed}", file=sys.stderr)
                return 1
            results.append({"name": name, "expected_error_fragment": expected,
                            "observed_error": observed, "rejected": True})
    output = {
        "schema": "neuralstudio.fg.mid-l55-negative-audit.v1",
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
