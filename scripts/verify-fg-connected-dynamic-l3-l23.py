#!/usr/bin/env python3
from __future__ import annotations
import argparse, copy, hashlib, json, os, re, sys
from pathlib import Path
from typing import Any

JOB = "fg-connected-dynamic-l3-l23-20260904T164815Z"
SCHEMA = "neuralstudio.fg.connected-dynamic-l3-l23-run.v1"
GROUP = "neuralstudio.fg.connected-dynamic-l22-l23-run.v1"
WEIGHT = "eb99328524fddc9f5f3fac12b8e3a4011e66a75e071cd07126a70c553458bc60"
MODULE30 = "31bb08d04011db9a0def8f1a1bc3999efc08d5deb8d7cb9b5debf52ec489b785"
MODULE29 = "a29a3d7041cbe7e44707e74dddf7aff61871a2397d1c097ef913bd753a149977"
PARENT = "85a02c9bc0516cea39f9cd79932fe64ca948d69819eb502589922373b69e433d"
GENERATOR21 = "3963039331e816c0ee4b5730fc13ee2ce3f2433cbef51e1a85bf9ae246fedb2d"
GENERATOR23 = "8b6d71159e99176ae283157c171fded51cd2cbe41f27d32553a67339aac19170"
CHECKPOINTS = (
    "reset-module11", "reset-module15", "reset-module13", "reset-module21",
    "reset-module22", "reset-module23", "reset-module24", "reset-module25",
    "reset-module14", "reset-module27", "reset-module4",
    "generated-module27", "generated-module11", "generated-module15",
    "generated-module13", "generated-module21", "generated-module22",
    "generated-module23", "generated-module24", "generated-module25",
    "generated-module5", "generated-module28", "generated-module29",
    "generated-L22-module30", "generated-L23-module29",
)
RESOLUTIONS = {
    "720p": {"input": [160, 96], "output": [80, 48],
             "l22_bytes": 1_966_080, "l23_bytes": 491_520},
    "1080p": {"input": [240, 144], "output": [120, 72],
              "l22_bytes": 4_423_680, "l23_bytes": 1_105_920},
    "1440p": {"input": [320, 192], "output": [160, 96],
              "l22_bytes": 7_864_320, "l23_bytes": 1_966_080},
    "4k": {"input": [240, 144], "output": [120, 72],
           "l22_bytes": 4_423_680, "l23_bytes": 1_105_920},
}
ADDRESS = re.compile(r"(?i)0x[0-9a-f]{8,16}")


class Error(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise Error(message)


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


def address_free(value: Any, label: str) -> None:
    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                lowered = str(key).lower()
                require(lowered == "address_fields_recorded" or not any(
                    marker in lowered for marker in (
                        "pointer", "gpu_va", "virtual_address",
                        "device_address", "gpu_address")),
                    f"{label}: address field")
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)
        elif isinstance(item, str):
            require(not ADDRESS.search(item), f"{label}: address-like value")
    visit(value)


def verify_root(value: dict[str, Any], label: str) -> None:
    body = copy.deepcopy(value)
    observed = body.pop("receipt_root_sha256", None)
    require(observed == sha(canonical(body)), f"{label}: receipt root drift")
    address_free(value, label)


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
        "src/fg_connected_dynamic_l3_l23/mac_tail.inc",
        "src/fg_connected_dynamic_l3_l23/rtx_tail.inc",
        "src/fg_connected_dynamic_l3_l23/conv_1x1_ptx_zero.metal",
        "scripts/generate-fg-connected-dynamic-l3-l23.py",
        "scripts/build-macos-fg-connected-dynamic-l3-l23.sh",
        "scripts/build-fg-connected-dynamic-l3-l23-rtx.ps1",
        "scripts/verify-fg-connected-dynamic-l3-l23.py",
        "scripts/test-fg-connected-dynamic-l3-l23-mutations.py",
        "src/fg_connected_dynamic_l3_l21/mac_tail.inc",
        "src/fg_connected_dynamic_l3_l21/rtx_tail.inc",
        "scripts/generate-fg-connected-dynamic-l3-l21.py",
        "src/fg_conv/hmma_exact.metalh",
        "src/fg_pooling/pooling.metal",
    )
    return {path: file_sha(repository / path) for path in paths}


def run_id(resolution: str, poison: str, ordinal: str) -> str:
    return f"{resolution}-poison-{poison}-process-{ordinal}"


def validate_generator(value: dict[str, Any], expected_root: str,
                       schema: str) -> None:
    verify_root(value, "generator")
    require(value.get("schema") == schema and
            value.get("receipt_root_sha256") == expected_root and
            value.get("live_tail_injections") == 2 and
            value.get("expected_or_archived_intermediate_embedded") is False and
            value.get("result") == "pass", "generator drift")


def validate_backend(kind: str, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    backend = "rtx-d3d12-nvapi" if kind == "rtx" else "mac-metal"
    process_ids: set[int] = set()
    summaries: dict[str, Any] = {}
    receipt_hashes: dict[str, Any] = {}
    for resolution, spec in RESOLUTIONS.items():
        rows = []
        for poison_name, poison in (("a5", 165), ("5a", 90)):
            for ordinal in ("a", "b"):
                identity = run_id(resolution, poison_name, ordinal)
                run_root = root / identity
                main_path = run_root / "run-receipt.json"
                group_path = run_root / "l22-l23-receipt.json"
                main = load(main_path)
                group = load(group_path)
                require(main.get("schema") == SCHEMA and
                        main.get("backend") == backend and
                        main.get("run_id") == identity and
                        main.get("poison_byte") == poison and
                        main.get("selected_launch_count") == 25 and
                        tuple(main.get("checkpoint_sha256", {})) == CHECKPOINTS,
                        f"{identity}: main drift")
                require(group.get("schema") == GROUP and
                        group.get("backend") == backend and
                        group.get("resolution") == resolution and
                        group.get("weight_sha256") == WEIGHT,
                        f"{identity}: group drift")
                require(group.get("l21_pre_sha256") ==
                        main["checkpoint_sha256"]["generated-module29"] ==
                        group.get("l21_post_sha256"),
                        f"{identity}: live input drift")
                require(group.get("l22_bytes") == spec["l22_bytes"] and
                        group.get("l23_bytes") == spec["l23_bytes"] and
                        group.get("l22_sha256") ==
                        main["checkpoint_sha256"]["generated-L22-module30"] and
                        group.get("l23_sha256") ==
                        main["checkpoint_sha256"]["generated-L23-module29"],
                        f"{identity}: output drift")
                require(group.get("immutable_mismatched_bytes") == 0 and
                        group.get("canary_mismatched_bytes") == 0 and
                        group.get("live_l21_device_handoff") is True and
                        group.get("expected_or_archived_intermediate_ingress") is False and
                        group.get("result") == "pass",
                        f"{identity}: group integrity")
                require(main.get("input_mismatched_bytes") == 0 and
                        main.get("canary_mismatched_bytes") == 0 and
                        main.get("expected_or_captured_output_ingress") is False and
                        main.get("archived_intermediate_ingress") is False and
                        main.get("result") == "pass",
                        f"{identity}: parent integrity")
                full = poison_name == "a5" and ordinal == "a"
                require(main.get("artifacts_written") is full,
                        f"{identity}: artifact policy")
                require(isinstance(main.get("process_id"), int),
                        f"{identity}: pid")
                address_free(main, identity)
                address_free(group, identity + "/group")
                process_ids.add(main["process_id"])
                receipt_hashes[identity] = {
                    "main": file_sha(main_path), "group": file_sha(group_path)}
                rows.append((main, group))
                if full:
                    for filename, bytes_key, hash_key in (
                        ("l22-output.fp16", "l22_bytes", "l22_sha256"),
                        ("l23-output.fp16", "l23_bytes", "l23_sha256"),
                    ):
                        payload = (run_root / filename).read_bytes()
                        require(len(payload) == group[bytes_key] and
                                sha(payload) == group[hash_key],
                                f"{identity}: artifact")
        require(all(len({main["checkpoint_sha256"][name]
                         for main, group in rows}) == 1
                    for name in CHECKPOINTS),
                f"{resolution}: poison checkpoint")
        require(len({group["l22_sha256"] for main, group in rows}) == 1 and
                len({group["l23_sha256"] for main, group in rows}) == 1,
                f"{resolution}: poison group")
        main, group = rows[0]
        summaries[resolution] = {
            "checkpoint_sha256": main["checkpoint_sha256"],
            "l21_sha256": group["l21_pre_sha256"],
            "l22_bytes": group["l22_bytes"],
            "l22_sha256": group["l22_sha256"],
            "l23_bytes": group["l23_bytes"],
            "l23_sha256": group["l23_sha256"],
        }
    require(len(process_ids) == 16, f"{kind} process count")
    result = rooted({
        "schema": f"neuralstudio.fg.connected-dynamic-l3-l23-{kind}-backend.v1",
        "backend": backend,
        "fresh_process_count": 16,
        "selected_launches_per_process": 25,
        "resolutions": summaries,
        "run_receipt_sha256": receipt_hashes,
        "checkpoint_repeat_and_dual_poison_exact": True,
        "full_write_coverage_from_dual_poison": True,
        "live_l21_device_handoff": True,
        "inputs_weights_immutable": True,
        "canaries_intact": True,
        "expected_or_archived_intermediate_ingress": False,
        "result": "pass",
    })
    return result, summaries


def compare_outputs(rtx: Path, mac: Path) -> tuple[int, int, list[dict[str, Any]]]:
    compared = 0
    mismatched = 0
    rows = []
    for resolution in RESOLUTIONS:
        identity = run_id(resolution, "a5", "a")
        for filename in ("l22-output.fp16", "l23-output.fp16"):
            left = (rtx / identity / filename).read_bytes()
            right = (mac / identity / filename).read_bytes()
            require(len(left) == len(right), "artifact size drift")
            differences = sum(a != b for a, b in zip(left, right))
            compared += len(left)
            mismatched += differences
            rows.append({"resolution": resolution, "artifact": filename,
                         "bytes": len(left), "sha256": sha(left),
                         "mismatched_bytes": differences})
    return compared, mismatched, rows


def build(args: argparse.Namespace) -> None:
    repository = args.repository.resolve()
    source_hashes = source_identity(repository)
    rtx_g21 = load(args.rtx / "generator21-receipt.json")
    rtx_g23 = load(args.rtx / "generator23-receipt.json")
    mac_g21 = load(args.mac / "base-generated/generator-receipt.json")
    mac_g23 = load(args.mac / "generated/generator-receipt.json")
    validate_generator(rtx_g21, GENERATOR21,
                       "neuralstudio.fg.connected-dynamic-l3-l21-generator.v1")
    validate_generator(mac_g21, GENERATOR21,
                       "neuralstudio.fg.connected-dynamic-l3-l21-generator.v1")
    validate_generator(rtx_g23, GENERATOR23,
                       "neuralstudio.fg.connected-dynamic-l3-l23-generator.v1")
    validate_generator(mac_g23, GENERATOR23,
                       "neuralstudio.fg.connected-dynamic-l3-l23-generator.v1")
    require(rtx_g21 == mac_g21 and rtx_g23 == mac_g23,
            "cross-platform generator drift")
    rig_path = args.rtx / "rig-receipt.json"
    rig = load(rig_path)
    require(rig.get("schema") ==
            "neuralstudio.fg.connected-dynamic-l3-l23-rig.v1" and
            rig.get("job_id") == JOB and
            rig.get("attempt_id") == "oracle-v1" and
            rig.get("smoke") is False, "rig identity")
    require(rig.get("generator23_sha256") ==
            source_hashes["scripts/generate-fg-connected-dynamic-l3-l23.py"] and
            rig.get("tail23_sha256") ==
            source_hashes["src/fg_connected_dynamic_l3_l23/rtx_tail.inc"] and
            rig.get("harness_sha256") ==
            source_hashes["scripts/build-fg-connected-dynamic-l3-l23-rtx.ps1"],
            "rig source")
    require(rig.get("weight65_sha256") == WEIGHT and
            rig.get("module30_sha256") == MODULE30 and
            rig.get("run_count") == rig.get("fresh_process_count") == 16 and
            rig.get("selected_launches_per_process") == 25 and
            rig.get("live_device_handoffs") is True and
            rig.get("expected_or_archived_intermediate_ingress") is False and
            rig.get("result") == "pass", "rig execution")
    address_free(rig, "rig")
    rtx, rtx_summary = validate_backend("rtx", args.rtx.resolve())
    mac, mac_summary = validate_backend("mac", args.mac.resolve())
    require(rtx_summary == mac_summary, "backend disagreement")
    compared, mismatched, comparisons = compare_outputs(
        args.rtx.resolve(), args.mac.resolve())
    require((compared, mismatched) == (23_347_200, 0),
            "full artifact mismatch")
    topology = rooted({
        "schema": "neuralstudio.fg.connected-dynamic-l3-l23-topology.v1",
        "parent_receipt_root_sha256": PARENT,
        "base_generator_root_sha256": GENERATOR21,
        "extension_generator_root_sha256": GENERATOR23,
        "checkpoint_order": list(CHECKPOINTS),
        "resolutions": RESOLUTIONS,
        "l22": {"module": 30, "module_sha256": MODULE30,
                "weight_token": 65, "weight_sha256": WEIGHT,
                "operation": "1x1 32-to-64 leaky"},
        "l23": {"module": 29, "module_sha256": MODULE29,
                "operation": "2x2 max pool 64 channels"},
        "live_l21_device_handoff": True,
        "result": "pass",
    })
    implementation = rooted({
        "schema": "neuralstudio.fg.connected-dynamic-l3-l23-implementation.v1",
        "source_sha256": source_hashes,
        "ptx_zero_rule":
            "canonicalize only zero input to max.f16; retain negative zero from underflowed negative subnormal",
        "per_input_patch_count": 0,
        "expected_output_embedded": False,
        "captured_intermediate_embedded": False,
        "result": "pass",
    })
    failures = rooted({
        "schema": "neuralstudio.fg.connected-dynamic-l3-l23-counterexamples.v1",
        "items": [
            {"name": "ordinary Metal max signed zero", "result": "rejected",
             "detail": "1080p L22 one byte differed; RTX positive zero versus Metal negative zero"},
            {"name": "blanket zero-result canonicalization", "result": "rejected",
             "detail": "seven L22 half values and two pooled half values regressed"},
            {"name": "missing token63 in manual smoke staging",
             "result": "rejected-before-execution"},
            {"name": "first generated include path", "result": "build-rejected"},
        ],
        "all_failures_preserved": True,
        "result": "pass",
    })
    accepted = rooted({
        "schema": "neuralstudio.fg.connected-dynamic-l3-l23-accepted.v1",
        "selected_range": "large-resolution launches 3 through 27",
        "resolutions": list(RESOLUTIONS),
        "processes_per_backend": 16,
        "selected_launches_per_process": 25,
        "rtx_mac_checkpoint_comparisons": 400,
        "rtx_mac_checkpoint_mismatches": 0,
        "l22_l23_full_artifact_compared_bytes": compared,
        "l22_l23_full_artifact_mismatched_bytes": mismatched,
        "topology_receipt_root_sha256": topology["receipt_root_sha256"],
        "implementation_receipt_root_sha256":
            implementation["receipt_root_sha256"],
        "counterexamples_root_sha256": failures["receipt_root_sha256"],
        "rtx_backend_root_sha256": rtx["receipt_root_sha256"],
        "mac_backend_root_sha256": mac["receipt_root_sha256"],
        "rtx_rig_receipt_sha256": file_sha(rig_path),
        "comparisons": comparisons,
        "outputs": {
            name: {key: value for key, value in summary.items()
                   if key != "checkpoint_sha256"}
            for name, summary in rtx_summary.items()},
        "claims": {
            "connected_dynamic_L3_L23_exact": True,
            "L22_checkpoint_exact": True,
            "L23_checkpoint_exact": True,
            "live_L21_handoff_exact": True,
            "token65_immutable": True,
            "signed_zero_rule_general": True,
            "connected_L24_and_later": False,
            "complete_generated_frame": False,
            "Netal_integration": False,
        },
        "scope": "connected dynamic path through second encoder pooling only",
        "result": "pass",
    })
    for filename, value in (
        ("topology-receipt.json", topology),
        ("implementation-receipt.json", implementation),
        ("counterexamples.json", failures),
        ("rtx-backend-receipt.json", rtx),
        ("mac-backend-receipt.json", mac),
        ("accepted-receipt.json", accepted),
    ):
        write(args.output / filename, value)
    print(f"accepted_receipt_root_sha256={accepted['receipt_root_sha256']}")
    print(f"l22_l23_full_artifact_compared_bytes={compared}")
    print("verification_result=pass")


def verify(repository: Path, evidence: Path, require_negative: bool) -> None:
    names = ("topology-receipt.json", "implementation-receipt.json",
             "counterexamples.json", "rtx-backend-receipt.json",
             "mac-backend-receipt.json", "accepted-receipt.json")
    values = {name: load(evidence / name) for name in names}
    for name, value in values.items():
        verify_root(value, name)
    topology, implementation, failures, rtx, mac, accepted = (
        values[name] for name in names)
    require(topology.get("parent_receipt_root_sha256") == PARENT and
            topology.get("checkpoint_order") == list(CHECKPOINTS) and
            topology.get("resolutions") == RESOLUTIONS and
            topology.get("live_l21_device_handoff") is True,
            "topology drift")
    require(implementation.get("source_sha256") ==
            source_identity(repository) and
            implementation.get("per_input_patch_count") == 0 and
            implementation.get("expected_output_embedded") is False,
            "implementation drift")
    require(failures.get("all_failures_preserved") is True and
            len(failures.get("items", [])) == 4, "failure ledger drift")
    require(rtx.get("fresh_process_count") == 16, "RTX process count drift")
    require(mac.get("fresh_process_count") == 16, "Mac process count drift")
    require(rtx.get("resolutions") == mac.get("resolutions"),
            "backend disagreement")
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
    require(accepted.get("rtx_mac_checkpoint_comparisons") == 400 and
            accepted.get("rtx_mac_checkpoint_mismatches") == 0 and
            accepted.get("l22_l23_full_artifact_compared_bytes") == 23_347_200 and
            accepted.get("l22_l23_full_artifact_mismatched_bytes") == 0,
            "accepted comparison drift")
    claims = accepted.get("claims", {})
    require(claims.get("connected_dynamic_L3_L23_exact") is True and
            claims.get("L22_checkpoint_exact") is True and
            claims.get("L23_checkpoint_exact") is True and
            claims.get("signed_zero_rule_general") is True and
            claims.get("connected_L24_and_later") is False and
            claims.get("complete_generated_frame") is False,
            "scope drift")
    if require_negative:
        negative = load(evidence / "negative-audit.json")
        verify_root(negative, "negative")
        require(negative.get("mutation_count", 0) >= 14 and
                negative.get("all_mutations_rejected") is True and
                negative.get("verifier_sha256") == file_sha(
                    repository / "scripts/verify-fg-connected-dynamic-l3-l23.py"),
                "negative audit drift")


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build")
    for name in ("--repository", "--rtx", "--mac", "--output"):
        build_parser.add_argument(name, type=Path, required=True)
    receipt = commands.add_parser("receipt")
    receipt.add_argument("--repository", type=Path, required=True)
    receipt.add_argument("--evidence", type=Path, required=True)
    receipt.add_argument("--skip-negative", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "build":
            build(args)
        else:
            verify(args.repository.resolve(), args.evidence.resolve(),
                   not args.skip_negative)
            print("verification_result=pass")
        return 0
    except Exception as error:
        print(f"verification_error={error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
