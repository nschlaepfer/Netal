#!/usr/bin/env python3
"""Verify compact EX2 promotion through the accepted 16-frame FG chain."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


PARENT_SHA256 = "ddea2f457e243031d59d65f0f61da1ac49266bbb419849064abd4d5fea4867d5"
COMPACT_EX2_SHA256 = "25250cd89126554704dfc9bd3e0009a82213cb86c22c0aa8b44fe00ed9881d17"
PROMOTION_SHA256 = "59c75c57863a5b96341f6b97646a5aa2f345e40d8ecdc965d970202212c20e9f"
PARENT_ROOT = "7d911b7783a047807571bf2c685c9e23470fcff75798d59c9d400d469312df72"
CHECKPOINT_ROOT = "86b63bfdf21772fe343304d065ec91ebcab3362d52ccbb1d5a14eb6c483bb296"
COMPACT_EX2_ROOT = "f1b0a84083b2292594680034b0241ae627e32379b60e54a99f597fffd482ec28"
PROMOTION_ROOT = "90a00206b73a42952ce6598a3161eec4f8345a365b731ba4e2ce3b0b113831e7"
UNIT_SHA256 = "60c2ae1a0869ce6f10ba093a33def68ecdbf4ba0d789cd186289586b63b9f234"
FRACTION_SHA256 = "76bacb1adf8e53accdd6a40a64940c073f001e091232e9119b155f2ab25f1a13"
WRAPPER_SHA256 = "4349021d5c2747962a59a00d87c39a96956fe0f7f48c8830a524533bd980a9fd"


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
    require(value.get("parent_verification_root_sha256") == PARENT_ROOT,
            "parent root drift")
    require(value.get("compact_ex2_root_sha256") == COMPACT_EX2_ROOT,
            "compact EX2 root drift")
    require(value.get("promotion_root_sha256") == PROMOTION_ROOT,
            "promotion root drift")
    require(value.get("checkpoint_root_sha256") == CHECKPOINT_ROOT,
            "checkpoint root drift")
    require(value.get("fresh_process_count") == 4, "process count drift")
    require(value.get("poison_bytes") == [90, 165], "poison matrix drift")
    require(value.get("rendered_frame_count") == 16, "frame count drift")
    require(value.get("launch_count") == 1149 and
            value.get("checkpoint_count") == 1149, "launch coverage drift")
    require(value.get("reference_checkpoint_exact") is True,
            "reference comparison failed")
    require(value.get("two_poison_invariant") is True,
            "two-poison invariant failed")
    require(value.get("total_input_mismatched_bytes") == 0,
            "input mutation")
    require(value.get("total_canary_mismatched_bytes") == 0,
            "canary mutation")
    require(value.get("expected_or_archived_tensor_ingress") is False,
            "expected tensor ingress")
    require(value.get("unit_correction") == {
        "bytes": 4_194_305, "sha256": UNIT_SHA256},
        "unit correction drift")
    require(value.get("fraction_correction") == {
        "bytes": 2_097_153, "sha256": FRACTION_SHA256},
        "fraction correction drift")
    claims = value.get("claims", {})
    require(claims.get("connected_long16_compact_ex2_exact") is True,
            "connected exactness claim missing")
    require(claims.get("persistent_caller_owned_runtime") is False,
            "persistent runtime overclaim")
    require(claims.get("production_game_capture") is False,
            "game capture overclaim")
    require(claims.get("netal_integration") is False,
            "Netal overclaim")
    require(claims.get("complete_fg_performance") is False,
            "performance overclaim")
    require(value.get("address_fields_recorded") is False,
            "address field enabled")
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
        require(rejected, f"compact promotion mutation accepted: {name}")
        cases.append({"case": name, "rejected": True})

    reject("parent-root", lambda value: value.__setitem__("parent_verification_root_sha256", "0" * 64))
    reject("compact-root", lambda value: value.__setitem__("compact_ex2_root_sha256", "0" * 64))
    reject("checkpoint", lambda value: value.__setitem__("checkpoint_root_sha256", "0" * 64))
    reject("process", lambda value: value.__setitem__("fresh_process_count", 3))
    reject("poison", lambda value: value.__setitem__("poison_bytes", [165]))
    reject("launch-count", lambda value: value.__setitem__("launch_count", 1148))
    reject("reference", lambda value: value.__setitem__("reference_checkpoint_exact", False))
    reject("two-poison", lambda value: value.__setitem__("two_poison_invariant", False))
    reject("input", lambda value: value.__setitem__("total_input_mismatched_bytes", 1))
    reject("canary", lambda value: value.__setitem__("total_canary_mismatched_bytes", 1))
    reject("expected-ingress", lambda value: value.__setitem__("expected_or_archived_tensor_ingress", True))
    reject("unit-correction", lambda value: value["unit_correction"].__setitem__("sha256", "0" * 64))
    reject("persistent-overclaim", lambda value: value["claims"].__setitem__("persistent_caller_owned_runtime", True))
    reject("netal-overclaim", lambda value: value["claims"].__setitem__("netal_integration", True))
    reject("performance-overclaim", lambda value: value["claims"].__setitem__("complete_fg_performance", True))
    audit = {
        "schema": "neuralstudio.fg.production-contract-compact-ex2-negative-audit.v1",
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
    parser.add_argument("--compact-ex2", type=Path, required=True)
    parser.add_argument("--promotion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--negative-output", type=Path, required=True)
    arguments = parser.parse_args()
    repository = arguments.repository.resolve()
    parent = load(arguments.parent.resolve(), PARENT_SHA256)
    compact = load(arguments.compact_ex2.resolve(), COMPACT_EX2_SHA256)
    promotion = load(arguments.promotion.resolve(), PROMOTION_SHA256)
    require(sha_file(repository / "scripts/build-macos-fg-production-contract-l3-l87-compact-ex2.sh") == WRAPPER_SHA256,
            "promotion wrapper source drift")
    require(parent.get("verification_root_sha256") == PARENT_ROOT,
            "parent verification root drift")
    require(parent.get("checkpoint_root_sha256") == CHECKPOINT_ROOT and
            parent.get("claims", {}).get("production_contract_long16_exact") is True,
            "parent exactness drift")
    require(compact.get("receipt_root_sha256") == COMPACT_EX2_ROOT and
            compact.get("claims", {}).get("finite_nonpositive_sm120_ex2_exact") is True,
            "compact primitive exactness drift")
    require(promotion.get("receipt_root_sha256") == PROMOTION_ROOT,
            "promotion receipt root drift")
    require(promotion.get("checkpoint_root_sha256") == parent["checkpoint_root_sha256"],
            "promotion checkpoint mismatch")
    require(promotion.get("invariant_output_sha256") == parent["final_output_sha256"],
            "promotion final outputs mismatch")
    require(promotion.get("allocation37_sha256") == parent["allocation37_sha256"] and
            promotion.get("allocation29_sha256") == parent["allocation29_sha256"],
            "promotion temporal state mismatch")
    require(promotion.get("unit_correction") == compact["unit_correction"] and
            promotion.get("fraction_correction") == compact["fraction_correction"],
            "promotion correction identity mismatch")

    result = {
        "schema": "neuralstudio.fg.production-contract-compact-ex2-verification.v1",
        "parent_verification_root_sha256": PARENT_ROOT,
        "compact_ex2_root_sha256": COMPACT_EX2_ROOT,
        "promotion_root_sha256": PROMOTION_ROOT,
        "checkpoint_root_sha256": CHECKPOINT_ROOT,
        "fresh_process_count": promotion["fresh_process_count"],
        "poison_bytes": promotion["poison_bytes"],
        "rendered_frame_count": promotion["rendered_frame_count"],
        "launch_count": promotion["launch_count"],
        "checkpoint_count": promotion["checkpoint_count"],
        "reference_checkpoint_exact": promotion["reference_checkpoint_exact"],
        "two_poison_invariant": promotion["two_poison_invariant"],
        "final_output_sha256": promotion["invariant_output_sha256"],
        "unit_correction": promotion["unit_correction"],
        "fraction_correction": promotion["fraction_correction"],
        "total_input_mismatched_bytes": promotion["total_input_mismatched_bytes"],
        "total_canary_mismatched_bytes": promotion["total_canary_mismatched_bytes"],
        "expected_or_archived_tensor_ingress": False,
        "address_fields_recorded": False,
        "claims": {
            "connected_long16_compact_ex2_exact": True,
            "persistent_caller_owned_runtime": False,
            "production_game_capture": False,
            "netal_integration": False,
            "complete_fg_performance": False,
        },
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
