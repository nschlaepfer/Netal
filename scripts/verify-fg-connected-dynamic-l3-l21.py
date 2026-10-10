#!/usr/bin/env python3
"""Fail-closed verification for live connected dynamic L3-L21."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


JOB_ID = "fg-connected-dynamic-l3-l21-20260904T163030Z"
RUN_SCHEMA = "neuralstudio.fg.connected-dynamic-l3-l21-run.v1"
GROUP_SCHEMA = "neuralstudio.fg.connected-dynamic-l20-l21-run.v1"
WEIGHT_SHA = "44595b02307a6b934a124abe476cfd327440e2acba7f5ed69f73195ca743e3d9"
L20_MODULE_SHA = "7f9c1f110e8ec9347aa968bc9019e5ffc8839057987a2e5676a0f64637c8df2d"
L21_MODULE_SHA = "a29a3d7041cbe7e44707e74dddf7aff61871a2397d1c097ef913bd753a149977"
EARLY_ROOT = "e47b515113402aec1cb8c2a6b4466a257e093ff9908bb5e3a9838fce411b493a"
EARLY_COUNTER_ROOT = "9dccf83feaf59043b54bf5e5bf1479df18ba2511fbc0d538fdd8bdf052d1d9b6"
GENERATOR_ROOT = "3963039331e816c0ee4b5730fc13ee2ce3f2433cbef51e1a85bf9ae246fedb2d"
CHECKPOINTS = (
    "reset-module11", "reset-module15", "reset-module13", "reset-module21",
    "reset-module22", "reset-module23", "reset-module24", "reset-module25",
    "reset-module14", "reset-module27", "reset-module4",
    "generated-module27", "generated-module11", "generated-module15",
    "generated-module13", "generated-module21", "generated-module22",
    "generated-module23", "generated-module24", "generated-module25",
    "generated-module5", "generated-module28", "generated-module29",
)
RESOLUTIONS = {
    "720p": {"tensor": [320, 192], "l20": 3_932_160, "pool": [160, 96], "l21": 983_040},
    "1080p": {"tensor": [480, 288], "l20": 8_847_360, "pool": [240, 144], "l21": 2_211_840},
    "1440p": {"tensor": [640, 384], "l20": 15_728_640, "pool": [320, 192], "l21": 3_932_160},
    "4k": {"tensor": [480, 288], "l20": 8_847_360, "pool": [240, 144], "l21": 2_211_840},
}
ADDRESS_PATTERN = re.compile(r"(?i)0x[0-9a-f]{8,16}")


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    return sha(path.read_bytes())


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def rooted(value: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(value)
    result["receipt_root_sha256"] = sha(canonical(result))
    return result


def verify_root(value: dict[str, Any], label: str) -> None:
    body = copy.deepcopy(value)
    observed = body.pop("receipt_root_sha256", None)
    require(observed == sha(canonical(body)), f"{label}: receipt root drift")
    address_free(value, label)


def address_free(value: Any, label: str) -> None:
    def visit(item: Any, path: str) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                lowered = str(key).lower()
                if lowered != "address_fields_recorded":
                    require(not any(marker in lowered for marker in (
                        "pointer", "gpu_va", "virtual_address",
                        "device_address", "gpu_address")),
                        f"{label}: address field at {path}.{key}")
                visit(child, f"{path}.{key}")
        elif isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")
        elif isinstance(item, str):
            require(ADDRESS_PATTERN.search(item) is None,
                    f"{label}: address-like value at {path}")
    visit(value, "$")


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    require(isinstance(value, dict), f"not an object: {path}")
    return value


def write(path: Path, value: dict[str, Any]) -> None:
    address_free(value, path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def source_identity(repository: Path) -> dict[str, str]:
    paths = (
        "src/fg_connected_dynamic_l3_l21/mac_tail.inc",
        "src/fg_connected_dynamic_l3_l21/rtx_tail.inc",
        "scripts/generate-fg-connected-dynamic-l3-l21.py",
        "scripts/build-macos-fg-connected-dynamic-l3-l21.sh",
        "scripts/build-fg-connected-dynamic-l3-l21-rtx.ps1",
        "scripts/verify-fg-connected-dynamic-l3-l21.py",
        "scripts/test-fg-connected-dynamic-l3-l21-mutations.py",
        "src/fg_connected_dynamic_early/macos_connected.mm",
        "src/fg_connected_dynamic_early/rtx_connected.cpp",
        "src/fg_connected_dynamic_early/contract.h",
        "src/fg_conv/conv_l20.metal",
        "src/fg_pooling/pooling.metal",
    )
    return {path: file_sha(repository / path) for path in paths}


def run_id(resolution: str, poison: str, ordinal: str) -> str:
    return f"{resolution}-poison-{poison}-process-{ordinal}"


def validate_generator(value: dict[str, Any], label: str) -> None:
    verify_root(value, label)
    require(value.get("schema") ==
            "neuralstudio.fg.connected-dynamic-l3-l21-generator.v1" and
            value.get("receipt_root_sha256") == GENERATOR_ROOT and
            value.get("schema_replacements") == 2 and
            value.get("launch_count_replacements") == 2 and
            value.get("pass_gate_replacements") == 2 and
            value.get("live_tail_injections") == 2 and
            value.get("expected_or_archived_intermediate_embedded") is False and
            value.get("result") == "pass", f"{label}: generator drift")


def validate_rig(rig: dict[str, Any], sources: dict[str, str],
                 rig_generator_path: Path) -> None:
    require(rig.get("schema") ==
            "neuralstudio.fg.connected-dynamic-l3-l21-rig.v1" and
            rig.get("job_id") == JOB_ID and
            rig.get("attempt_id") == "oracle-v1" and
            rig.get("smoke") is False, "rig identity drift")
    require(rig.get("generator_sha256") ==
            sources["scripts/generate-fg-connected-dynamic-l3-l21.py"] and
            rig.get("generator_receipt_sha256") == file_sha(rig_generator_path) and
            rig.get("accepted_early_source_sha256") ==
            sources["src/fg_connected_dynamic_early/rtx_connected.cpp"] and
            rig.get("tail_source_sha256") ==
            sources["src/fg_connected_dynamic_l3_l21/rtx_tail.inc"] and
            rig.get("harness_sha256") ==
            sources["scripts/build-fg-connected-dynamic-l3-l21-rtx.ps1"],
            "rig source identity drift")
    require(rig.get("weight_sha256") == WEIGHT_SHA and
            rig.get("l20_module_sha256") == L20_MODULE_SHA and
            rig.get("l21_module_sha256") == L21_MODULE_SHA,
            "rig model identity drift")
    require(rig.get("run_count") == 16 and
            rig.get("fresh_process_count") == 16 and
            rig.get("selected_launches_per_process") == 23 and
            rig.get("live_allocation37_device_handoff") is True and
            rig.get("exit_code") == 0 and
            isinstance(rig.get("started_utc"), str) and
            isinstance(rig.get("completed_utc"), str),
            "rig execution drift")
    require(rig.get("expected_or_archived_intermediate_ingress") is False and
            rig.get("address_fields_recorded") is False and
            rig.get("result") == "pass", "rig forbidden ingress")
    address_free(rig, "rig")


def validate_backend(kind: str, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    backend = "rtx-d3d12-nvapi" if kind == "rtx" else "mac-metal"
    process_ids: set[int] = set()
    summaries: dict[str, Any] = {}
    receipt_hashes: dict[str, Any] = {}
    for resolution, spec in RESOLUTIONS.items():
        rows = []
        for poison_name, poison in (("a5", 0xA5), ("5a", 0x5A)):
            for ordinal in ("a", "b"):
                identity = run_id(resolution, poison_name, ordinal)
                run_root = root / identity
                main_path = run_root / "run-receipt.json"
                group_path = run_root / "l20-l21-receipt.json"
                main = load(main_path)
                group = load(group_path)
                require(main.get("schema") == RUN_SCHEMA and
                        main.get("backend") == backend and
                        main.get("run_id") == identity and
                        main.get("resolution") == resolution and
                        main.get("poison_byte") == poison,
                        f"{identity}: main identity drift")
                require(main.get("selected_launch_count") == 23 and
                        tuple(main.get("checkpoint_sha256", {}).keys()) ==
                        CHECKPOINTS, f"{identity}: checkpoint topology drift")
                require(group.get("schema") == GROUP_SCHEMA and
                        group.get("backend") == backend and
                        group.get("resolution") == resolution and
                        group.get("weight_sha256") == WEIGHT_SHA,
                        f"{identity}: group identity drift")
                require(group.get("allocation37_pre_sha256") ==
                        main.get("allocation37_sha256") ==
                        group.get("allocation37_post_sha256"),
                        f"{identity}: live allocation37 handoff drift")
                require(group.get("l20_bytes") == spec["l20"] and
                        group.get("l21_bytes") == spec["l21"] and
                        main["checkpoint_sha256"]["generated-module28"] ==
                        group.get("l20_sha256") and
                        main["checkpoint_sha256"]["generated-module29"] ==
                        group.get("l21_sha256"),
                        f"{identity}: L20/L21 checkpoint drift")
                require(group.get("immutable_mismatched_bytes") == 0 and
                        group.get("canary_mismatched_bytes") == 0 and
                        group.get("live_allocation37_device_handoff") is True and
                        group.get("expected_or_archived_intermediate_ingress") is False and
                        group.get("address_fields_recorded") is False and
                        group.get("result") == "pass",
                        f"{identity}: group integrity drift")
                require(main.get("input_mismatched_bytes") == 0 and
                        main.get("canary_mismatched_bytes") == 0 and
                        main.get("expected_or_captured_output_ingress") is False and
                        main.get("archived_intermediate_ingress") is False and
                        main.get("result") == "pass",
                        f"{identity}: main integrity drift")
                full = poison_name == "a5" and ordinal == "a"
                require(main.get("artifacts_written") is full,
                        f"{identity}: artifact policy drift")
                require(isinstance(main.get("process_id"), int),
                        f"{identity}: process identity missing")
                address_free(main, identity)
                address_free(group, identity + "/group")
                process_ids.add(main["process_id"])
                receipt_hashes[identity] = {
                    "main": file_sha(main_path), "group": file_sha(group_path)}
                rows.append((main, group))
                if full:
                    for filename, bytes_key, hash_key in (
                        ("l20-intermediate.fp16", "l20_bytes", "l20_sha256"),
                        ("l21-output.fp16", "l21_bytes", "l21_sha256"),
                    ):
                        payload = (run_root / filename).read_bytes()
                        require(len(payload) == group[bytes_key] and
                                sha(payload) == group[hash_key],
                                f"{identity}: {filename} artifact drift")
        require(all(len({main["checkpoint_sha256"][name]
                         for main, group in rows}) == 1
                    for name in CHECKPOINTS),
                f"{resolution}: checkpoint poison/repeat drift")
        require(len({group["l20_sha256"] for main, group in rows}) == 1 and
                len({group["l21_sha256"] for main, group in rows}) == 1,
                f"{resolution}: group poison/repeat drift")
        first_main, first_group = rows[0]
        summaries[resolution] = {
            "checkpoint_sha256": first_main["checkpoint_sha256"],
            "allocation37_sha256": first_group["allocation37_pre_sha256"],
            "l20_bytes": first_group["l20_bytes"],
            "l20_sha256": first_group["l20_sha256"],
            "l21_bytes": first_group["l21_bytes"],
            "l21_sha256": first_group["l21_sha256"],
        }
    require(len(process_ids) == 16, f"{kind.upper()} process count drift")
    result = rooted({
        "schema": f"neuralstudio.fg.connected-dynamic-l3-l21-{kind}-backend.v1",
        "backend": backend,
        "fresh_process_count": 16,
        "selected_launches_per_process": 23,
        "resolutions": summaries,
        "run_receipt_sha256": receipt_hashes,
        "checkpoint_repeat_and_dual_poison_exact": True,
        "full_write_coverage_from_dual_poison": True,
        "allocation37_immutable_through_group": True,
        "live_allocation37_device_handoff": True,
        "canaries_intact": True,
        "expected_or_archived_intermediate_ingress": False,
        "address_fields_recorded": False,
        "result": "pass",
    })
    return result, summaries


def compare_outputs(rtx: Path, mac: Path) -> tuple[int, int, list[dict[str, Any]]]:
    compared = 0
    mismatched = 0
    rows = []
    for resolution in RESOLUTIONS:
        identity = run_id(resolution, "a5", "a")
        for filename in ("l20-intermediate.fp16", "l21-output.fp16"):
            left = (rtx / identity / filename).read_bytes()
            right = (mac / identity / filename).read_bytes()
            require(len(left) == len(right), f"{resolution}/{filename}: size drift")
            differences = sum(a != b for a, b in zip(left, right))
            compared += len(left)
            mismatched += differences
            rows.append({"resolution": resolution, "artifact": filename,
                         "bytes": len(left), "sha256": sha(left),
                         "mismatched_bytes": differences})
    return compared, mismatched, rows


def build(args: argparse.Namespace) -> None:
    repository = args.repository.resolve()
    sources = source_identity(repository)
    rtx_generator_path = args.rtx / "generator-receipt.json"
    mac_generator_path = args.mac / "generated/generator-receipt.json"
    rtx_generator = load(rtx_generator_path)
    mac_generator = load(mac_generator_path)
    validate_generator(rtx_generator, "RTX generator")
    validate_generator(mac_generator, "Mac generator")
    require(rtx_generator == mac_generator, "cross-platform generator drift")
    rig_path = args.rtx / "rig-receipt.json"
    validate_rig(load(rig_path), sources, rtx_generator_path)
    rtx, rtx_summary = validate_backend("rtx", args.rtx.resolve())
    mac, mac_summary = validate_backend("mac", args.mac.resolve())
    require(rtx_summary == mac_summary, "RTX/Metal group disagreement")
    compared, mismatched, comparisons = compare_outputs(
        args.rtx.resolve(), args.mac.resolve())
    require(compared == 46_694_400 and mismatched == 0,
            "RTX/Metal full artifact mismatch")
    topology = rooted({
        "schema": "neuralstudio.fg.connected-dynamic-l3-l21-topology.v1",
        "accepted_early_receipt_root_sha256": EARLY_ROOT,
        "generator_receipt_root_sha256": GENERATOR_ROOT,
        "checkpoint_order": list(CHECKPOINTS),
        "resolutions": RESOLUTIONS,
        "l20": {"module": 28, "module_sha256": L20_MODULE_SHA,
                 "function": "k_conv_fp16_nhwc", "weight_token": 63,
                 "weight_sha256": WEIGHT_SHA},
        "l21": {"module": 29, "module_sha256": L21_MODULE_SHA,
                 "function": "k_pooling", "channels": 32},
        "allocation37_handoff": "same live GPU buffer; no CPU refill",
        "four_k_mixed_early_extents_preserved": True,
        "result": "pass",
    })
    implementation = rooted({
        "schema": "neuralstudio.fg.connected-dynamic-l3-l21-implementation.v1",
        "source_sha256": sources,
        "rtx_generated_source_sha256":
            rtx_generator["generated"]["rtx"]["generated_sha256"],
        "mac_generated_source_sha256":
            mac_generator["generated"]["mac"]["generated_sha256"],
        "live_tail_injection_count": 2,
        "expected_output_embedded": False,
        "captured_intermediate_embedded": False,
        "result": "pass",
    })
    failures = rooted({
        "schema": "neuralstudio.fg.connected-dynamic-l3-l21-counterexamples.v1",
        "items": [
            {"name": "archived allocation37 cutpoint", "result": "rejected",
             "reason": "both tails consume the same device buffer produced by module5"},
            {"name": "RTX oracle-smoke-v1", "result": "generation-rejected",
             "reason": "the cross-platform generator required the accepted Mac source to be staged too"},
            {"name": "checkpoint-derived allocation37 pre-hash", "result": "review-rejected",
             "reason": "the checkpoint is a three-output join; the final tail hashes allocation37 directly"},
        ],
        "inherited_early_counterexamples_root_sha256":
            EARLY_COUNTER_ROOT,
        "all_failures_preserved": True,
        "result": "pass",
    })
    accepted = rooted({
        "schema": "neuralstudio.fg.connected-dynamic-l3-l21-accepted.v1",
        "selected_range": "large-resolution launches 3 through 25",
        "resolutions": list(RESOLUTIONS),
        "processes_per_backend": 16,
        "selected_launches_per_process": 23,
        "rtx_mac_checkpoint_comparisons": 368,
        "rtx_mac_checkpoint_mismatches": 0,
        "l20_l21_full_artifact_compared_bytes": compared,
        "l20_l21_full_artifact_mismatched_bytes": mismatched,
        "topology_receipt_root_sha256": topology["receipt_root_sha256"],
        "implementation_receipt_root_sha256": implementation["receipt_root_sha256"],
        "counterexamples_root_sha256": failures["receipt_root_sha256"],
        "rtx_backend_root_sha256": rtx["receipt_root_sha256"],
        "mac_backend_root_sha256": mac["receipt_root_sha256"],
        "rtx_rig_receipt_sha256": file_sha(rig_path),
        "comparisons": comparisons,
        "outputs": {name: {key: value for key, value in summary.items()
                            if key != "checkpoint_sha256"}
                    for name, summary in rtx_summary.items()},
        "claims": {
            "connected_dynamic_L3_L21_exact": True,
            "L20_checkpoint_exact": True,
            "L21_checkpoint_exact": True,
            "allocation37_live_handoff_exact": True,
            "allocation37_immutable_through_group": True,
            "fresh_process_dual_poison_coverage": True,
            "connected_L22_and_later": False,
            "complete_generated_frame": False,
            "Netal_integration": False,
        },
        "scope": "connected dynamic early path plus first convolution and pooling only",
        "result": "pass",
    })
    for name, value in (
        ("topology-receipt.json", topology),
        ("implementation-receipt.json", implementation),
        ("counterexamples.json", failures),
        ("rtx-backend-receipt.json", rtx),
        ("mac-backend-receipt.json", mac),
        ("accepted-receipt.json", accepted),
    ):
        write(args.output / name, value)
    print(f"accepted_receipt_root_sha256={accepted['receipt_root_sha256']}")
    print(f"l20_l21_full_artifact_compared_bytes={compared}")
    print("verification_result=pass")


def verify_evidence(repository: Path, evidence: Path,
                    require_negative: bool) -> None:
    names = ("topology-receipt.json", "implementation-receipt.json",
             "counterexamples.json", "rtx-backend-receipt.json",
             "mac-backend-receipt.json", "accepted-receipt.json")
    values = {name: load(evidence / name) for name in names}
    for name, value in values.items():
        verify_root(value, name)
    topology = values["topology-receipt.json"]
    implementation = values["implementation-receipt.json"]
    failures = values["counterexamples.json"]
    rtx = values["rtx-backend-receipt.json"]
    mac = values["mac-backend-receipt.json"]
    accepted = values["accepted-receipt.json"]
    require(topology.get("accepted_early_receipt_root_sha256") == EARLY_ROOT and
            topology.get("generator_receipt_root_sha256") == GENERATOR_ROOT and
            topology.get("checkpoint_order") == list(CHECKPOINTS) and
            topology.get("resolutions") == RESOLUTIONS and
            topology.get("four_k_mixed_early_extents_preserved") is True,
            "topology drift")
    require(topology.get("l20", {}).get("module_sha256") == L20_MODULE_SHA and
            topology.get("l21", {}).get("module_sha256") == L21_MODULE_SHA and
            topology.get("allocation37_handoff") ==
            "same live GPU buffer; no CPU refill", "group topology drift")
    require(implementation.get("source_sha256") == source_identity(repository) and
            implementation.get("live_tail_injection_count") == 2 and
            implementation.get("expected_output_embedded") is False and
            implementation.get("captured_intermediate_embedded") is False,
            "implementation drift")
    require(failures.get("all_failures_preserved") is True and
            len(failures.get("items", [])) == 3 and
            failures.get("inherited_early_counterexamples_root_sha256") ==
            EARLY_COUNTER_ROOT,
            "failure ledger drift")
    require(rtx.get("fresh_process_count") == 16,
            "RTX process count drift")
    require(mac.get("fresh_process_count") == 16,
            "Mac process count drift")
    for label, backend in (("RTX", rtx), ("Mac", mac)):
        require(backend.get("selected_launches_per_process") == 23 and
                backend.get("checkpoint_repeat_and_dual_poison_exact") is True and
                backend.get("full_write_coverage_from_dual_poison") is True and
                backend.get("allocation37_immutable_through_group") is True and
                backend.get("live_allocation37_device_handoff") is True and
                backend.get("canaries_intact") is True and
                backend.get("expected_or_archived_intermediate_ingress") is False and
                backend.get("result") == "pass", f"{label} backend drift")
    require(rtx.get("resolutions") == mac.get("resolutions"),
            "RTX/Metal group disagreement")
    require(accepted.get("topology_receipt_root_sha256") ==
            topology["receipt_root_sha256"] and
            accepted.get("implementation_receipt_root_sha256") ==
            implementation["receipt_root_sha256"] and
            accepted.get("counterexamples_root_sha256") ==
            failures["receipt_root_sha256"] and
            accepted.get("rtx_backend_root_sha256") ==
            rtx["receipt_root_sha256"] and
            accepted.get("mac_backend_root_sha256") ==
            mac["receipt_root_sha256"], "receipt link drift")
    require(accepted.get("rtx_mac_checkpoint_comparisons") == 368 and
            accepted.get("rtx_mac_checkpoint_mismatches") == 0 and
            accepted.get("l20_l21_full_artifact_compared_bytes") == 46_694_400 and
            accepted.get("l20_l21_full_artifact_mismatched_bytes") == 0 and
            len(accepted.get("comparisons", [])) == 8,
            "accepted comparison drift")
    claims = accepted.get("claims", {})
    require(claims.get("connected_dynamic_L3_L21_exact") is True and
            claims.get("L20_checkpoint_exact") is True and
            claims.get("L21_checkpoint_exact") is True and
            claims.get("allocation37_live_handoff_exact") is True and
            claims.get("allocation37_immutable_through_group") is True and
            claims.get("fresh_process_dual_poison_coverage") is True and
            claims.get("connected_L22_and_later") is False and
            claims.get("complete_generated_frame") is False and
            claims.get("Netal_integration") is False,
            "scope drift")
    if require_negative:
        negative = load(evidence / "negative-audit.json")
        verify_root(negative, "negative-audit.json")
        require(negative.get("mutation_count", 0) >= 14 and
                negative.get("all_mutations_rejected") is True and
                negative.get("verifier_sha256") == file_sha(
                    repository / "scripts/verify-fg-connected-dynamic-l3-l21.py"),
                "negative audit drift")


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build")
    build_parser.add_argument("--repository", type=Path, required=True)
    build_parser.add_argument("--rtx", type=Path, required=True)
    build_parser.add_argument("--mac", type=Path, required=True)
    build_parser.add_argument("--output", type=Path, required=True)
    receipt_parser = commands.add_parser("receipt")
    receipt_parser.add_argument("--repository", type=Path, required=True)
    receipt_parser.add_argument("--evidence", type=Path, required=True)
    receipt_parser.add_argument("--skip-negative", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "build":
            build(args)
        else:
            verify_evidence(args.repository.resolve(), args.evidence.resolve(),
                            not args.skip_negative)
            print("verification_result=pass")
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError,
            VerificationError) as error:
        print(f"verification_error={error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
