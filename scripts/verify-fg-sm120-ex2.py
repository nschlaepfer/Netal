#!/usr/bin/env python3
"""Verify the exact finite-nonpositive SM120 scalar EX2 primitive."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


MANTISSAS = 1 << 23
FINITE_NEGATIVE_PATTERNS = 255 * MANTISSAS
UNIT_TABLE_BYTES = 67_108_868
FRACTION_TABLE_BYTES = 33_554_436
UNIT_TABLE_SHA256 = "40088ff59b58c4018114b4a2dd1fb5b124488b91e7cad79bdf3bf143bf267b75"
FRACTION_TABLE_SHA256 = "f7cfb2fd9df97ae01bc747b866e4928a79311e8443700f938859f43772f73d86"
IDENTITY_SHA256 = "35081520af28773487a37dfd2f8ece07e2fbfe83544ccf1568b8a0569af6c0ee"

RAW_FILE_SHA256 = {
    "fraction": "bd0e7d4c329a23828cd36682a2e91532f20928e8d980c19457d36bbce95bab8c",
    "unit": "e48b312648b1a1fc3f41bff8003350a454b7bbf1751c022789e07d0655754ac9",
    "scale": "f7e0d8ef4c902d55daea36a4d12e3507a3a2d01de766b2b79631483c867aa48e",
    "mac": "cfacf85351708e944231994dbac7f333cf145ed13d9ab8a4039cc12b8c65dbf8",
}

SOURCE_SHA256 = {
    "rtx_fraction": ("src/fg_math_ex2/rtx_fraction_probe.cu", "ccfcdca812f30e26ca2b1462e5b6dbb111a141163ebb766c80a11b3d2478f4d8"),
    "rtx_unit": ("src/fg_math_ex2/rtx_unit_quantization_probe.cu", "870a0d91fd82beefa2aad93324367b5b08c05b15779a69c9c59042ddec2f0895"),
    "rtx_scale": ("src/fg_math_ex2/rtx_scale_boundary_probe.cu", "60c19a1db51d8806353e87fbdb51d1b07c63dd41cf5a7584d4b8c13809522374"),
    "metal_header": ("src/fg_math_ex2/sm120_ex2.metalh", "94dbe259e1665c71d31e4f05d6c428554f7aad21be7de5fa50cf8d94e4c005d9"),
    "metal_probe": ("src/fg_math_ex2/sm120_ex2_probe.metal", "30df41fd7683529aead97ac81a1852a03e11bfb0d9c8f3a70051f6a03535815b"),
    "mac_host": ("src/fg_math_ex2/macos_probe.mm", "6d4adf7e624004b3a638c6ebd48b3f243f7f6d27cf1e16f357255fd1c7147ae0"),
    "mac_build": ("scripts/build-macos-fg-sm120-ex2.sh", "9c214aa8905f65ac0f934f90d9be2df0efa832425aed6f08a070beab2947bb82"),
}


class VerificationError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def canonical_sha(value: Any) -> str:
    return sha_bytes(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode())


def load(path: Path, expected_sha256: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"input missing: {path}")
    require(sha_file(path) == expected_sha256, f"input hash drift: {path}")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def validate_result(value: dict[str, Any]) -> None:
    require(value.get("finite_negative_input_patterns") == FINITE_NEGATIVE_PATTERNS, "finite pattern count drift")
    require(value.get("negative_exponent_range") == [0, 254], "exponent range drift")
    require(value.get("tiny_non_one_results") == 0, "tiny branch mismatch")
    require(value.get("unit_floor_mismatches") == 0, "unit floor mapping mismatch")
    require(value.get("scale_threshold_mismatches") == 0, "scale branch mismatch")
    require(value.get("high_nonzero_results") == 0, "high FTZ branch mismatch")
    require(value.get("mac_random_mismatched_bytes") == 0, "Metal random comparison mismatch")
    require(value.get("mac_fresh_process_count") == 4, "Metal process count drift")
    require(value.get("mac_poison_bytes") == [90, 165], "Metal poison matrix drift")
    require(value.get("tables_immutable") is True, "math tables mutated")
    require(value.get("expected_output_bound_to_gpu") is False, "expected output entered GPU execution")
    require(value.get("table_bytes") == UNIT_TABLE_BYTES + FRACTION_TABLE_BYTES, "table extent drift")
    claims = value.get("claims", {})
    require(claims.get("finite_nonpositive_sm120_ex2_exact") is True, "exact EX2 claim missing")
    require(claims.get("positive_input_domain") is False, "positive domain overclaim")
    require(claims.get("optimized_for_production") is False, "optimization overclaim")
    require(claims.get("frame_or_model_output_lookup") is False, "model lookup overclaim")
    require(value.get("address_fields_recorded") is False, "address field enabled")
    require(value.get("result") == "pass", "verification rejected")


def negative_audit(result: dict[str, Any]) -> dict[str, Any]:
    cases = []

    def reject(name: str, mutate: Any) -> None:
        value = copy.deepcopy(result)
        mutate(value)
        rejected = False
        try:
            validate_result(value)
        except VerificationError:
            rejected = True
        require(rejected, f"EX2 mutation accepted: {name}")
        cases.append({"case": name, "rejected": True})

    reject("finite-pattern-count", lambda value: value.__setitem__("finite_negative_input_patterns", FINITE_NEGATIVE_PATTERNS - 1))
    reject("tiny", lambda value: value.__setitem__("tiny_non_one_results", 1))
    reject("unit-floor", lambda value: value.__setitem__("unit_floor_mismatches", 1))
    reject("scale-threshold", lambda value: value.__setitem__("scale_threshold_mismatches", 1))
    reject("high-ftz", lambda value: value.__setitem__("high_nonzero_results", 1))
    reject("mac-output", lambda value: value.__setitem__("mac_random_mismatched_bytes", 1))
    reject("process-count", lambda value: value.__setitem__("mac_fresh_process_count", 3))
    reject("poison", lambda value: value.__setitem__("mac_poison_bytes", [165]))
    reject("table-mutation", lambda value: value.__setitem__("tables_immutable", False))
    reject("expected-ingress", lambda value: value.__setitem__("expected_output_bound_to_gpu", True))
    reject("positive-overclaim", lambda value: value["claims"].__setitem__("positive_input_domain", True))
    reject("optimized-overclaim", lambda value: value["claims"].__setitem__("optimized_for_production", True))
    reject("model-lookup", lambda value: value["claims"].__setitem__("frame_or_model_output_lookup", True))
    audit = {
        "schema": "neuralstudio.fg.sm120-ex2-negative-audit.v1",
        "case_count": len(cases),
        "rejected_count": len(cases),
        "cases": cases,
        "result": "pass",
    }
    audit["receipt_root_sha256"] = canonical_sha(audit)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--fraction-rig", type=Path, required=True)
    parser.add_argument("--unit-rig", type=Path, required=True)
    parser.add_argument("--scale-rig", type=Path, required=True)
    parser.add_argument("--mac-backend", type=Path, required=True)
    parser.add_argument("--unit-table", type=Path, required=True)
    parser.add_argument("--fraction-table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--negative-output", type=Path, required=True)
    arguments = parser.parse_args()
    repository = arguments.repository.resolve()

    fraction = load(arguments.fraction_rig.resolve(), RAW_FILE_SHA256["fraction"])
    unit = load(arguments.unit_rig.resolve(), RAW_FILE_SHA256["unit"])
    scale = load(arguments.scale_rig.resolve(), RAW_FILE_SHA256["scale"])
    mac = load(arguments.mac_backend.resolve(), RAW_FILE_SHA256["mac"])
    for value in (fraction, unit, scale, mac):
        require(value.get("result") == "pass", "backend receipt rejected")

    for _, (relative, expected) in SOURCE_SHA256.items():
        require(sha_file(repository / relative) == expected, f"source drift: {relative}")
    require(fraction.get("source_sha256") == SOURCE_SHA256["rtx_fraction"][1], "fraction source mismatch")
    require(unit.get("source_sha256") == SOURCE_SHA256["rtx_unit"][1], "unit source mismatch")
    require(scale.get("source_sha256") == SOURCE_SHA256["rtx_scale"][1], "scale source mismatch")

    require(fraction.get("repeat_equal") is True, "fraction repeat gate failed")
    require(fraction.get("fraction", {}).get("records") == (1 << 23) + 1, "fraction records drift")
    require(fraction.get("fraction", {}).get("sha256") == FRACTION_TABLE_SHA256, "fraction table drift")
    require(fraction.get("normalization_identity", {}).get("records") == 1 << 20, "identity record drift")
    require(fraction.get("normalization_identity", {}).get("sha256") == IDENTITY_SHA256, "identity hash drift")
    require(fraction.get("normalization_identity", {}).get("mismatches") == 369_557, "normalization counterexample drift")

    require(unit.get("repeat_equal") is True, "unit repeat gate failed")
    require(unit.get("unit_table", {}).get("records") == (1 << 24) + 1, "unit table records drift")
    require(unit.get("unit_table", {}).get("bytes") == UNIT_TABLE_BYTES, "unit table bytes drift")
    require(unit.get("unit_table", {}).get("sha256") == UNIT_TABLE_SHA256, "unit table hash drift")
    unit_audit = unit.get("audit", {})
    require(unit_audit.get("records") == 27 * MANTISSAS, "unit audit count drift")
    require(unit_audit.get("first_exponent") == 100 and unit_audit.get("last_exponent") == 126, "unit exponent range drift")
    require(unit_audit.get("floor_mismatches") == 0, "unit floor rule mismatch")
    require(unit_audit.get("nearest_mismatches") == 50_328_533, "nearest counterexample drift")
    require(unit_audit.get("ceil_mismatches") == 96_465_877, "ceil counterexample drift")

    require(scale.get("repeat_equal") is True, "scale repeat gate failed")
    tiny = scale.get("tiny_audit", {})
    scale_audit = scale.get("scale_audit", {})
    high = scale.get("high_audit", {})
    require(tiny.get("records") == 104 * MANTISSAS and tiny.get("non_one_results") == 0, "tiny exhaustive audit drift")
    require(scale_audit.get("records") == 7 * MANTISSAS, "scale exhaustive count drift")
    require(scale_audit.get("raw_mismatches") == 262_143 and scale_audit.get("threshold_mismatches") == 0, "scale threshold audit drift")
    require(high.get("records") == 121 * MANTISSAS and high.get("nonzero_results") == 0, "high exhaustive audit drift")

    unit_table = arguments.unit_table.resolve()
    fraction_table = arguments.fraction_table.resolve()
    require(unit_table.stat().st_size == UNIT_TABLE_BYTES and sha_file(unit_table) == UNIT_TABLE_SHA256, "local unit table drift")
    require(fraction_table.stat().st_size == FRACTION_TABLE_BYTES and sha_file(fraction_table) == FRACTION_TABLE_SHA256, "local fraction table drift")

    require(mac.get("fresh_process_count") == 4 and mac.get("poison_bytes") == [90, 165], "Mac process matrix drift")
    require(mac.get("input_records") == 1 << 20, "Mac input count drift")
    require(mac.get("total_output_mismatched_bytes") == 0, "Mac output mismatch")
    require(mac.get("total_input_mismatched_bytes") == 0 and mac.get("total_canary_mismatched_bytes") == 0, "Mac integrity mismatch")
    require(mac.get("tables_immutable") is True and mac.get("expected_output_bound_to_gpu") is False, "Mac provenance gate failed")

    result = {
        "schema": "neuralstudio.fg.sm120-ex2-verification.v1",
        "instruction": "ex2.approx.ftz.f32",
        "input_domain": "all finite nonpositive FP32 values",
        "negative_exponent_range": [0, 254],
        "finite_negative_input_patterns": FINITE_NEGATIVE_PATTERNS,
        "tiny_records": tiny["records"],
        "tiny_non_one_results": 0,
        "unit_audit_records": unit_audit["records"],
        "unit_floor_mismatches": 0,
        "unit_nearest_counterexamples": unit_audit["nearest_mismatches"],
        "unit_ceil_counterexamples": unit_audit["ceil_mismatches"],
        "scale_audit_records": scale_audit["records"],
        "raw_scale_counterexamples": scale_audit["raw_mismatches"],
        "scale_threshold_mismatches": 0,
        "high_audit_records": high["records"],
        "high_nonzero_results": 0,
        "unit_table": {"bytes": UNIT_TABLE_BYTES, "sha256": UNIT_TABLE_SHA256},
        "fraction_table": {"bytes": FRACTION_TABLE_BYTES, "sha256": FRACTION_TABLE_SHA256},
        "table_bytes": UNIT_TABLE_BYTES + FRACTION_TABLE_BYTES,
        "mac_random_records": mac["input_records"],
        "mac_random_mismatched_bytes": mac["total_output_mismatched_bytes"],
        "mac_output_sha256": mac["output_sha256"],
        "mac_fresh_process_count": 4,
        "mac_poison_bytes": [90, 165],
        "tables_immutable": True,
        "expected_output_bound_to_gpu": False,
        "address_fields_recorded": False,
        "source_sha256": {name: value[1] for name, value in SOURCE_SHA256.items()},
        "claims": {
            "finite_nonpositive_sm120_ex2_exact": True,
            "positive_input_domain": False,
            "optimized_for_production": False,
            "frame_or_model_output_lookup": False,
        },
        "remaining_limits": [
            "The exact reference uses 100,663,304 bytes of direct instruction tables.",
            "Compression and cache behavior are optimization work and may not change outputs.",
            "Positive finite inputs, NaNs, and positive infinity are outside this FG consumer contract.",
        ],
        "result": "pass",
    }
    validate_result(result)
    result["receipt_root_sha256"] = canonical_sha(result)
    audit = negative_audit(result)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    arguments.negative_output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(f"receipt_root_sha256={result['receipt_root_sha256']}")
    print(f"negative_root_sha256={audit['receipt_root_sha256']}")
    print("result=pass")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationError as error:
        raise SystemExit(f"verification_error={error}")
