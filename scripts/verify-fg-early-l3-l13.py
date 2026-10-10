#!/usr/bin/env python3
"""Verify exact standalone L3/L13 module11 parity and negative gates."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED = {
    "reset-history":
        "733b4af34b536e0373a79d3d9fafc62d160ce515b293de841318be89386716c7",
    "generated-history":
        "d6c619e1052928c38c2087d10978ce43aeaae0772caa363de16123bf3a8bf791",
}
EXPECTED_COMPONENTS = {
    "reset-history": {
        "half_color": "95167dfcfbe0f3500f06631d85088c1ff53dd68abf862ce3d9f8aaf8d5b3d0ce",
        "half_metric": "8aec1d742dec79b7454413ce0d54d98de6a17f149c526bc044b9fe4464947fd2",
        "next_history": "2a5af5667469d1388408be12a2ef7474f1d9637b519baef1ac490b3f4b84fb91",
    },
    "generated-history": {
        "half_color": "95167dfcfbe0f3500f06631d85088c1ff53dd68abf862ce3d9f8aaf8d5b3d0ce",
        "half_metric": "eed0b8054f4a0a21b3949f9cdf697b8aec51fe3d57953f5af10939da173a4531",
        "next_history": "2a5af5667469d1388408be12a2ef7474f1d9637b519baef1ac490b3f4b84fb91",
    },
}
PHASES = ("l3-evaluate-0", "l13-evaluate-1")
FIXTURES = ("reset-history", "generated-history")
RAW = {
    "l3-evaluate-0":
        "f9d25459eeb37610a50065102b61cd7f458fb1598ad754f3da241293aa15e435",
    "l13-evaluate-1":
        "9eb930556ec08fd8db9ec361c917b8462707a6dc091b24227512a6a503115ffa",
}
ADDRESS_FREE = {
    "l3-evaluate-0":
        "631199cad3cff1b7e60a775d029de73644fcd82c085620aec984d6cdf6e292e5",
    "l13-evaluate-1":
        "dfcd9a6cda167f7ef89c43dcb56997e35f58c6ec348730396e940d29a6560625",
}
BANKS = {
    "l3-evaluate-0": ("history-bank-a", "history-bank-b", 3),
    "l13-evaluate-1": ("history-bank-b", "history-bank-a", 13),
}


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: Any) -> str:
    return digest(json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode())


def validate_receipts(receipts: list[dict[str, Any]], backend: str) -> dict[str, Any]:
    require(len(receipts) == 16, f"{backend}: receipt count")
    expected_backend = "rtx-d3d12-nvapi" if backend == "rtx" else "mac-metal"
    for receipt in receipts:
        phase = receipt.get("phase")
        fixture = receipt.get("fixture")
        require(receipt.get("schema") ==
                "neuralstudio.fg.early-l3-l13-run.v1", f"{backend}: schema")
        require(receipt.get("backend") == expected_backend, f"{backend}: name")
        require(phase in PHASES and fixture in FIXTURES,
                f"{backend}: phase/fixture")
        read_bank, write_bank, launch = BANKS[phase]
        require(receipt.get("selected_launch") == launch,
                f"{backend}: launch")
        require(receipt.get("module_token") == 11 and
                receipt.get("module_sha256") ==
                "06eee29ec33202996f97c063a08ccdc767d0167252ad03b5ed457c941cb85206",
                f"{backend}: module")
        require(receipt.get("raw_parameter_sha256") == RAW[phase],
                f"{backend}: raw ABI")
        require(receipt.get("address_free_parameter_sha256") ==
                ADDRESS_FREE[phase], f"{backend}: canonical ABI")
        require(receipt.get("history_read_bank") == read_bank and
                receipt.get("history_write_bank") == write_bank,
                f"{backend}: bank roles")
        require(receipt.get("combined_output_bytes") == 2419200,
                f"{backend}: output bytes")
        require(receipt.get("combined_output_sha256") == EXPECTED[fixture],
                f"{backend}: output identity")
        require(receipt.get("output_sha256") == EXPECTED_COMPONENTS[fixture],
                f"{backend}: component output identity")
        require(receipt.get("input_mismatched_bytes") == 0 and
                receipt.get("canary_mismatched_bytes") == 0,
                f"{backend}: immutable/canary")
        require(receipt.get("poison_pattern_halfwords") == 0,
                f"{backend}: incomplete write")
        require(receipt.get("typed_resource_backend") is True,
                f"{backend}: typed-resource gate")
        require(receipt.get("expected_or_archived_tensor_ingress") is False,
                f"{backend}: forbidden ingress")
        require(receipt.get("address_fields_recorded") is False and
                receipt.get("result") == "pass", f"{backend}: receipt result")
    require(len({r["process_id"] for r in receipts}) == 16,
            f"{backend}: fresh processes")
    for phase in PHASES:
        for fixture in FIXTURES:
            subset = [r for r in receipts
                      if r["phase"] == phase and r["fixture"] == fixture]
            require(sorted(r["poison_byte"] for r in subset) == [90,90,165,165],
                    f"{backend}: poison population")
            require(len({r["combined_output_sha256"] for r in subset}) == 1,
                    f"{backend}: repeat invariance")
    for fixture in FIXTURES:
        require(len({r["combined_output_sha256"] for r in receipts
                     if r["fixture"] == fixture}) == 1,
                f"{backend}: phase alias mismatch")
    return {
        "backend": backend,
        "fresh_process_count": 16,
        "fixture_output_sha256": EXPECTED,
        "fixture_component_sha256": EXPECTED_COMPONENTS,
        "all_phase_poison_process_invariant": True,
        "full_write_coverage": True,
        "input_immutable": True,
        "canaries_clean": True,
        "typed_resource_backend": True,
        "address_fields_recorded": False,
    }


def load_backend(root: Path, backend: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paths = sorted(root.glob("*/run-receipt.json"))
    receipts = [json.loads(path.read_text()) for path in paths]
    summary = validate_receipts(receipts, backend)
    for path, receipt in zip(paths, receipts, strict=True):
        output = path.parent / "output.bin"
        require(output.is_file(), f"{backend}: output missing")
        data = output.read_bytes()
        require(len(data) == 2419200 and
                digest(data) == receipt["combined_output_sha256"],
                f"{backend}: output artifact drift")
    aggregate = root / "backend-receipt.json"
    require(aggregate.is_file(), f"{backend}: backend receipt missing")
    summary["aggregate_receipt_sha256"] = digest(aggregate.read_bytes())
    return receipts, summary


def negative_audit(rtx: list[dict[str, Any]], mac: list[dict[str, Any]]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []

    def reject(name: str, mutation: Any) -> None:
        left, right = copy.deepcopy(rtx), copy.deepcopy(mac)
        mutation(left, right)
        rejected = False
        try:
            validate_receipts(left, "rtx")
            validate_receipts(right, "mac")
            require({r["combined_output_sha256"] for r in left} ==
                    {r["combined_output_sha256"] for r in right},
                    "cross-backend identity")
        except VerificationError:
            rejected = True
        require(rejected, f"mutation accepted: {name}")
        cases.append({"name": name, "rejected": True})

    reject("output", lambda left,right:
           left[0].__setitem__("combined_output_sha256", "0"*64))
    reject("component-output", lambda left,right:
           left[0]["output_sha256"].__setitem__("half_metric", "0"*64))
    reject("phase", lambda left,right:
           left[0].__setitem__("phase", "l4"))
    reject("history-bank", lambda left,right:
           left[0].__setitem__("history_write_bank", "wrong-bank"))
    reject("raw-abi", lambda left,right:
           left[0].__setitem__("raw_parameter_sha256", "0"*64))
    reject("module", lambda left,right: left[0].__setitem__("module_token", 10))
    reject("canary", lambda left,right:
           left[0].__setitem__("canary_mismatched_bytes", 1))
    reject("untouched-output", lambda left,right:
           left[0].__setitem__("poison_pattern_halfwords", 1))
    reject("forbidden-ingress", lambda left,right:
           left[0].__setitem__("expected_or_archived_tensor_ingress", True))
    reject("untyped-backend", lambda left,right:
           left[0].__setitem__("typed_resource_backend", False))
    reject("duplicate-process", lambda left,right:
           left[1].__setitem__("process_id", left[0]["process_id"]))
    reject("poison-collapse", lambda left,right:
           [item.__setitem__("poison_byte", 90) for item in left])
    reject("cross-backend", lambda left,right:
           [item.__setitem__("combined_output_sha256", "1"*64)
            for item in right if item["fixture"] == "reset-history"])
    result = {
        "schema": "neuralstudio.fg.early-l3-l13-negative-audit.v1",
        "mutation_count": len(cases),
        "rejected_count": len(cases),
        "failed_to_reject": 0,
        "cases": cases,
        "result": "pass",
    }
    result["receipt_root_sha256"] = canonical(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtx-root", type=Path, required=True)
    parser.add_argument("--mac-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--negative-output", type=Path, required=True)
    args = parser.parse_args()
    rtx_receipts, rtx = load_backend(args.rtx_root, "rtx")
    mac_receipts, mac = load_backend(args.mac_root, "mac")
    require(rtx["fixture_output_sha256"] == mac["fixture_output_sha256"],
            "RTX/Metal output mismatch")
    cuarray_path = args.rtx_root / "cuarray-transport.json"
    require(cuarray_path.is_file(), "CUarray transport diagnostic missing")
    cuarray = json.loads(cuarray_path.read_text())
    require(cuarray.get("authoritative_backend") ==
            "d3d12-nvapi-typed-resources" and
            cuarray.get("result") == "diagnostic-only",
            "CUarray diagnostic authority drift")
    result = {
        "schema": "neuralstudio.fg.early-l3-l13-verification.v1",
        "scope": "standalone L3/L13 module11 selected mode2 only",
        "claims": {
            "rtx_metal_exact": True,
            "both_phase_aliases": True,
            "reset_and_generated_history_fixtures": True,
            "connected_temporal_history": False,
            "full_frame": False,
        },
        "rtx": rtx,
        "mac": mac,
        "fixture_output_sha256": EXPECTED,
        "cuarray_transport": cuarray,
        "cuarray_transport_receipt_sha256": digest(cuarray_path.read_bytes()),
        "expected_or_archived_tensor_ingress": False,
        "address_fields_recorded": False,
        "result": "pass",
    }
    result["receipt_root_sha256"] = canonical(result)
    negative = negative_audit(rtx_receipts, mac_receipts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    args.negative_output.write_text(
        json.dumps(negative, indent=2, sort_keys=True)+"\n")
    print(f"receipt_root_sha256={result['receipt_root_sha256']}")
    print(f"negative_root_sha256={negative['receipt_root_sha256']}")
    print("result=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
