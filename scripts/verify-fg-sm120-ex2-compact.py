#!/usr/bin/env python3
"""Verify the lossless two-bit SM120 scalar EX2 correction form."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


PARENT_SHA256 = "2262a5c97139ea69b59b45309c693486f60fab2950e86a8cf82f1562b339cae9"
BACKEND_SHA256 = "d75bfbf0dc7c4f5e281ea726ac6cb40105f59db117d544dc834400e9da2a9fd2"
PARENT_ROOT = "b691eedc53067f3c24c22e729f366fd19cf1803cebe96b2a3a8afb5e52c4bfef"
BACKEND_ROOT = "8ef4553203dba4a61f4ce740a97af0f1d02dfbd4247917adb3a55f8209d1fe07"
UNIT_CORRECTION_BYTES = 4_194_305
FRACTION_CORRECTION_BYTES = 2_097_153
UNIT_CORRECTION_SHA256 = "60c2ae1a0869ce6f10ba093a33def68ecdbf4ba0d789cd186289586b63b9f234"
FRACTION_CORRECTION_SHA256 = "76bacb1adf8e53accdd6a40a64940c073f001e091232e9119b155f2ab25f1a13"
SOURCE_SHA256 = {
    "fast_probe": ("src/fg_math_ex2/fast_table_probe.metal", "0725078fce53efecda8001bfb8c980d429fe0a75157415ff148a268b6caf215e"),
    "fast_host": ("src/fg_math_ex2/macos_fast_table_probe.mm", "2d0f6caa33df728c8074a88cbc1f2a4b508e0d96ef1f10507fd4f367f4987577"),
    "fast_build": ("scripts/build-macos-fg-sm120-ex2-fast-tables.sh", "88643a40b7dd16184622ec49f91987dd24125f9364023c4f5496855c9fa6dde2"),
    "packer": ("src/fg_math_ex2/pack_fast_table_corrections.cc", "6f58c0b386d29b642feefb285a4c79c7345af0200d4906e94c54a525d465271f"),
    "compact_header": ("src/fg_math_ex2/sm120_ex2_compact.metalh", "fe9f18d8f513ac07d910eba33d27ac52f1cd7f2662fdf8292bc65ae2f1cb3e99"),
    "compact_probe": ("src/fg_math_ex2/sm120_ex2_compact_probe.metal", "0a5276753863c992c85fc4bd3a10339fd626ae015379a1ca5111f6a3492d4ef5"),
    "compact_host": ("src/fg_math_ex2/macos_compact_probe.mm", "f68a69b5019665e316b155a28ad4407f437516ff3bc378ff6664ba08628243a6"),
    "compact_build": ("scripts/build-macos-fg-sm120-ex2-compact.sh", "cffe528b704d6dd1837e95e170e5522b45f8ef4cecf158eb42037d0b92063b5f"),
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
    require(value.get("parent_exact_root_sha256") == PARENT_ROOT, "parent root drift")
    require(value.get("backend_root_sha256") == BACKEND_ROOT, "backend root drift")
    require(value.get("original_table_bytes") == 100_663_304, "reference extent drift")
    require(value.get("compact_correction_bytes") == 6_291_458, "compact extent drift")
    require(value.get("saved_bytes") == 94_371_846, "saved extent drift")
    require(value.get("finite_negative_input_patterns") == 2_139_095_040,
            "finite domain drift")
    require(value.get("unit_correction") == {
        "bytes": UNIT_CORRECTION_BYTES, "sha256": UNIT_CORRECTION_SHA256},
        "unit correction identity drift")
    require(value.get("fraction_correction") == {
        "bytes": FRACTION_CORRECTION_BYTES,
        "sha256": FRACTION_CORRECTION_SHA256},
        "fraction correction identity drift")
    require(value.get("unit_reconstruction_mismatches") == 0, "unit reconstruction mismatch")
    require(value.get("fraction_reconstruction_mismatches") == 0, "fraction reconstruction mismatch")
    require(value.get("mac_random_mismatched_bytes") == 0, "Metal output mismatch")
    require(value.get("mac_fresh_process_count") == 4, "Metal process count drift")
    require(value.get("mac_poison_bytes") == [90, 165], "poison matrix drift")
    require(value.get("fast_table_process_count") == 2, "fast table repeat drift")
    require(value.get("correction_generation_count") == 2, "correction repeat drift")
    require(value.get("correction_repeat_equal") is True, "correction repeat mismatch")
    require(value.get("corrections_immutable") is True, "correction mutation")
    require(value.get("expected_output_bound_to_gpu") is False, "expected output ingress")
    claims = value.get("claims", {})
    require(claims.get("finite_nonpositive_sm120_ex2_exact") is True, "exactness claim missing")
    require(claims.get("frame_or_model_output_lookup") is False, "model lookup overclaim")
    require(claims.get("production_performance_measured") is False, "performance overclaim")
    require(claims.get("complete_fg_optimized") is False, "complete FG overclaim")
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
        require(rejected, f"compact EX2 mutation accepted: {name}")
        cases.append({"case": name, "rejected": True})

    reject("parent-root", lambda value: value.__setitem__("parent_exact_root_sha256", "0" * 64))
    reject("compact-extent", lambda value: value.__setitem__("compact_correction_bytes", 6_291_459))
    reject("unit-correction-hash", lambda value: value["unit_correction"].__setitem__("sha256", "0" * 64))
    reject("fraction-correction-hash", lambda value: value["fraction_correction"].__setitem__("sha256", "0" * 64))
    reject("unit-reconstruction", lambda value: value.__setitem__("unit_reconstruction_mismatches", 1))
    reject("fraction-reconstruction", lambda value: value.__setitem__("fraction_reconstruction_mismatches", 1))
    reject("mac-output", lambda value: value.__setitem__("mac_random_mismatched_bytes", 1))
    reject("process-count", lambda value: value.__setitem__("mac_fresh_process_count", 3))
    reject("poison", lambda value: value.__setitem__("mac_poison_bytes", [165]))
    reject("generation-repeat", lambda value: value.__setitem__("correction_repeat_equal", False))
    reject("correction-mutation", lambda value: value.__setitem__("corrections_immutable", False))
    reject("expected-ingress", lambda value: value.__setitem__("expected_output_bound_to_gpu", True))
    reject("performance-overclaim", lambda value: value["claims"].__setitem__("production_performance_measured", True))
    reject("complete-overclaim", lambda value: value["claims"].__setitem__("complete_fg_optimized", True))
    audit = {
        "schema": "neuralstudio.fg.sm120-ex2-compact-negative-audit.v1",
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
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--unit-corrections", type=Path, required=True)
    parser.add_argument("--fraction-corrections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--negative-output", type=Path, required=True)
    arguments = parser.parse_args()
    repository = arguments.repository.resolve()
    parent = load(arguments.parent.resolve(), PARENT_SHA256)
    backend = load(arguments.backend.resolve(), BACKEND_SHA256)

    require(parent.get("receipt_root_sha256") == PARENT_ROOT, "parent receipt root drift")
    require(parent.get("claims", {}).get("finite_nonpositive_sm120_ex2_exact") is True,
            "parent exactness missing")
    require(parent.get("finite_negative_input_patterns") == 2_139_095_040,
            "parent exhaustive domain drift")
    require(parent.get("unit_floor_mismatches") == 0 and
            parent.get("scale_threshold_mismatches") == 0,
            "parent branch mapping drift")
    for _, (relative, expected) in SOURCE_SHA256.items():
        require(sha_file(repository / relative) == expected, f"source drift: {relative}")

    require(backend.get("receipt_root_sha256") == BACKEND_ROOT, "backend receipt root drift")
    require(backend.get("result") == "pass", "backend failed")
    require(backend.get("fresh_fast_table_process_count") == 2, "fast repeat count drift")
    require(backend.get("fresh_correction_generation_count") == 2, "correction repeat count drift")
    require(backend.get("correction_repeat_equal") is True, "correction repeat failed")
    require(backend.get("fresh_compact_process_count") == 4, "compact process count drift")
    require(backend.get("poison_bytes") == [90, 165], "backend poison drift")
    require(backend.get("total_output_mismatched_bytes") == 0, "backend output mismatch")
    require(backend.get("total_input_mismatched_bytes") == 0 and
            backend.get("total_canary_mismatched_bytes") == 0,
            "backend integrity mismatch")
    require(backend.get("corrections_immutable") is True and
            backend.get("expected_output_bound_to_gpu") is False,
            "backend provenance failure")

    unit = backend.get("unit", {})
    fraction = backend.get("fraction", {})
    require(unit.get("records") == 16_777_217 and
            unit.get("packed_bytes") == UNIT_CORRECTION_BYTES and
            unit.get("packed_sha256") == UNIT_CORRECTION_SHA256,
            "unit correction identity drift")
    require(unit.get("delta_min") == -1 and unit.get("delta_max") == 2 and
            unit.get("reconstruction_mismatches") == 0,
            "unit correction domain drift")
    require(fraction.get("records") == 8_388_609 and
            fraction.get("packed_bytes") == FRACTION_CORRECTION_BYTES and
            fraction.get("packed_sha256") == FRACTION_CORRECTION_SHA256,
            "fraction correction identity drift")
    require(fraction.get("delta_min") == 0 and fraction.get("delta_max") == 2 and
            fraction.get("reconstruction_mismatches") == 0,
            "fraction correction domain drift")

    unit_path = arguments.unit_corrections.resolve()
    fraction_path = arguments.fraction_corrections.resolve()
    require(unit_path.stat().st_size == UNIT_CORRECTION_BYTES and
            sha_file(unit_path) == UNIT_CORRECTION_SHA256,
            "local unit correction drift")
    require(fraction_path.stat().st_size == FRACTION_CORRECTION_BYTES and
            sha_file(fraction_path) == FRACTION_CORRECTION_SHA256,
            "local fraction correction drift")

    result = {
        "schema": "neuralstudio.fg.sm120-ex2-compact-verification.v1",
        "parent_exact_root_sha256": PARENT_ROOT,
        "backend_root_sha256": BACKEND_ROOT,
        "input_domain": "all finite nonpositive FP32 values",
        "finite_negative_input_patterns": parent["finite_negative_input_patterns"],
        "original_table_bytes": backend["original_table_bytes"],
        "compact_correction_bytes": backend["compact_correction_bytes"],
        "saved_bytes": backend["saved_bytes"],
        "compression_ratio": backend["compression_ratio"],
        "unit_correction": {"bytes": UNIT_CORRECTION_BYTES, "sha256": UNIT_CORRECTION_SHA256},
        "fraction_correction": {"bytes": FRACTION_CORRECTION_BYTES, "sha256": FRACTION_CORRECTION_SHA256},
        "unit_delta_histogram": unit["delta_histogram"],
        "fraction_delta_histogram": fraction["delta_histogram"],
        "unit_reconstruction_mismatches": unit["reconstruction_mismatches"],
        "fraction_reconstruction_mismatches": fraction["reconstruction_mismatches"],
        "fast_table_process_count": backend["fresh_fast_table_process_count"],
        "correction_generation_count": backend["fresh_correction_generation_count"],
        "correction_repeat_equal": backend["correction_repeat_equal"],
        "mac_random_records": backend["input_records"],
        "mac_random_mismatched_bytes": backend["total_output_mismatched_bytes"],
        "mac_output_sha256": backend["output_sha256"],
        "mac_fresh_process_count": backend["fresh_compact_process_count"],
        "mac_poison_bytes": backend["poison_bytes"],
        "corrections_immutable": backend["corrections_immutable"],
        "expected_output_bound_to_gpu": False,
        "address_fields_recorded": False,
        "source_sha256": {name: value[1] for name, value in SOURCE_SHA256.items()},
        "claims": {
            "finite_nonpositive_sm120_ex2_exact": True,
            "frame_or_model_output_lookup": False,
            "production_performance_measured": False,
            "complete_fg_optimized": False,
        },
        "remaining_limits": [
            "This proves the compact scalar EX2 primitive, not complete FG latency.",
            "The compact form must replace the reference tables inside a connected FG run before promotion.",
            "Positive finite inputs, NaNs, and positive infinity remain outside the FG consumer contract.",
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
