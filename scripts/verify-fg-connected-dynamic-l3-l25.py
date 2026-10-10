#!/usr/bin/env python3
from __future__ import annotations
import argparse, copy, hashlib, json, os, re, sys
from pathlib import Path
from typing import Any

JOB = "fg-connected-dynamic-l3-l25-20260904T170735Z"
SCHEMA = "neuralstudio.fg.connected-dynamic-l3-l25-run.v1"
GROUP = "neuralstudio.fg.connected-dynamic-l24-l25-run.v1"
WEIGHT = "4191924ff335d5018987f2526f933cb4cdbb9145411e3ce34f63c8206321be55"
MODULE31 = "e925f51165408fae4657d50247b10304802d46979b863712aeff844a217ade63"
MODULE29 = "a29a3d7041cbe7e44707e74dddf7aff61871a2397d1c097ef913bd753a149977"
MODEL_CAPTURE = "2a1179d4a25be030f772cd90ca7d017ed6eac9291917f7d43ea9b0cd5ac6a294"
MODEL_PACKAGE = "da9b6fd040f449931bd094849f6d71ea884a0073f5b6f0e319f6816a0947f9e3"
MODEL_L24_RTX = "6b19660f41e2fbce2cca37435802b6cfb6e58ff188264645b91a266b535d5868"
MODEL_L24_MAC = "1f5a17a5be615da1b4a3711b16a5e450607cd320ddd10a21fcc9d43b1412519c"
PARENT = "d1ed89007f5e50eee5fc49cb60ad6bea318c859e3b15d6dfd23fa3122e8e390a"
GENERATOR21 = "3963039331e816c0ee4b5730fc13ee2ce3f2433cbef51e1a85bf9ae246fedb2d"
GENERATOR23 = "8b6d71159e99176ae283157c171fded51cd2cbe41f27d32553a67339aac19170"
GENERATOR25 = "461603b31ada269b53cd0e91e9d81ae325a9eea1598136ea3e72da8a1dc36ced"
CHECKPOINTS = (
    "reset-module11", "reset-module15", "reset-module13", "reset-module21",
    "reset-module22", "reset-module23", "reset-module24", "reset-module25",
    "reset-module14", "reset-module27", "reset-module4",
    "generated-module27", "generated-module11", "generated-module15",
    "generated-module13", "generated-module21", "generated-module22",
    "generated-module23", "generated-module24", "generated-module25",
    "generated-module5", "generated-module28", "generated-module29",
    "generated-L22-module30", "generated-L23-module29",
    "generated-L24-module31", "generated-L25-module29",
)
RESOLUTIONS = {
    "720p": {"input": [80, 48], "output": [40, 24],
             "l24_bytes": 983_040, "l25_bytes": 245_760},
    "1080p": {"input": [120, 72], "output": [60, 36],
              "l24_bytes": 2_211_840, "l25_bytes": 552_960},
    "1440p": {"input": [160, 96], "output": [80, 48],
              "l24_bytes": 3_932_160, "l25_bytes": 983_040},
    "4k": {"input": [120, 72], "output": [60, 36],
           "l24_bytes": 2_211_840, "l25_bytes": 552_960},
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
        "src/fg_connected_dynamic_l3_l25/mac_tail.inc",
        "src/fg_connected_dynamic_l3_l25/rtx_tail.inc",
        "src/fg_connected_dynamic_l3_l25/conv_3x3_ptx_zero.metal",
        "src/fg_connected_dynamic_l3_l23/mac_tail.inc",
        "src/fg_connected_dynamic_l3_l23/rtx_tail.inc",
        "src/fg_connected_dynamic_l3_l23/conv_1x1_ptx_zero.metal",
        "scripts/generate-fg-connected-dynamic-l3-l23.py",
        "scripts/generate-fg-connected-dynamic-l3-l25.py",
        "scripts/build-macos-fg-connected-dynamic-l3-l25.sh",
        "scripts/build-fg-connected-dynamic-l3-l25-rtx.ps1",
        "scripts/collect-fg-connected-dynamic-l3-l25-rtx.ps1",
        "scripts/verify-fg-connected-dynamic-l3-l25.py",
        "scripts/test-fg-connected-dynamic-l3-l25-mutations.py",
        "src/fg_connected_dynamic_l3_l21/mac_tail.inc",
        "src/fg_connected_dynamic_l3_l21/rtx_tail.inc",
        "scripts/generate-fg-connected-dynamic-l3-l21.py",
        "src/fg_conv/hmma_exact.metalh",
        "src/fg_pooling/pooling.metal",
    )
    return {path: file_sha(repository / path) for path in paths}


def model_state_identity(repository: Path) -> dict[str, Any]:
    package = repository / "evidence/specs/fg-model-state-package-v4"
    require(file_sha(package / "package-manifest.json") == MODEL_PACKAGE,
            "model-state package drift")
    require(file_sha(package / "capture-manifest.json") == MODEL_CAPTURE,
            "model-state capture drift")
    roots = {}
    for backend, expected_root in (("rtx", MODEL_L24_RTX),
                                   ("mac", MODEL_L24_MAC)):
        value = load(package / f"backend-receipts/l24-{backend}.json")
        verify_root(value, f"model-state L24 {backend}")
        require(value.get("receipt_root_sha256") == expected_root and
                value.get("weight_source_token") == 67 and
                value.get("weight_sha256") == WEIGHT and
                value.get("module_sha256") == MODULE31 and
                value.get("weight_capture_manifest_sha256") == MODEL_CAPTURE and
                value.get("input_weight_immutability") is True and
                value.get("expected_output_fed_to_execution") is False and
                value.get("result") == "pass", "model-state consumer drift")
        roots[backend] = expected_root
    return {
        "package_manifest_sha256": MODEL_PACKAGE,
        "capture_manifest_sha256": MODEL_CAPTURE,
        "l24_consumer_receipt_roots": roots,
        "capture_point": "post-evaluate-0",
    }


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


def validate_collection(root: Path, repository: Path) -> dict[str, Any]:
    receipt = load(root / "collection-receipt.json")
    require(receipt.get("schema") ==
            "neuralstudio.fg.connected-dynamic-l3-l25-collection.v1" and
            receipt.get("source_attempt_id") == "oracle-v1" and
            receipt.get("collection_id") == "retrieval-v2" and
            receipt.get("collector_sha256") == file_sha(
                repository /
                "scripts/collect-fg-connected-dynamic-l3-l25-rtx.ps1") and
            receipt.get("source_rig_receipt_sha256") == file_sha(
                root / "rig-receipt.json") and
            receipt.get("run_receipt_count") == 16 and
            receipt.get("group_receipt_count") == 16 and
            receipt.get("full_artifact_resolution_count") == 4 and
            receipt.get("absolute_addresses_recorded") is False and
            receipt.get("expected_or_archived_intermediate_included") is False and
            receipt.get("result") == "pass", "collection identity")
    rows = receipt.get("files")
    require(isinstance(rows, list) and len(rows) == 44,
            "collection file count")
    paths: set[str] = set()
    for row in rows:
        require(isinstance(row, dict) and isinstance(row.get("path"), str) and
                isinstance(row.get("bytes"), int) and
                isinstance(row.get("sha256"), str), "collection row")
        relative = row["path"]
        require(relative not in paths and not relative.startswith("/") and
                ".." not in Path(relative).parts, "collection path")
        paths.add(relative)
        path = root / relative
        require(path.is_file() and path.stat().st_size == row["bytes"] and
                file_sha(path) == row["sha256"],
                f"collection hash drift: {relative}")
    observed = {
        str(path.relative_to(root)).replace(os.sep, "/")
        for path in root.rglob("*") if path.is_file() and
        path.name != "collection-receipt.json"
    }
    require(observed == paths, "collection unmanifested file")
    address_free(receipt, "collection")
    return receipt


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
                group_path = run_root / "l24-l25-receipt.json"
                main = load(main_path)
                group = load(group_path)
                require(main.get("schema") == SCHEMA and
                        main.get("backend") == backend and
                        main.get("run_id") == identity and
                        main.get("poison_byte") == poison and
                        main.get("selected_launch_count") == 27 and
                        tuple(main.get("checkpoint_sha256", {})) == CHECKPOINTS,
                        f"{identity}: main drift")
                require(group.get("schema") == GROUP and
                        group.get("backend") == backend and
                        group.get("resolution") == resolution and
                        group.get("weight_sha256") == WEIGHT,
                        f"{identity}: group drift")
                require(group.get("l23_pre_sha256") ==
                        main["checkpoint_sha256"]["generated-L23-module29"] ==
                        group.get("l23_post_sha256"),
                        f"{identity}: live input drift")
                require(group.get("l24_bytes") == spec["l24_bytes"] and
                        group.get("l25_bytes") == spec["l25_bytes"] and
                        group.get("l24_sha256") ==
                        main["checkpoint_sha256"]["generated-L24-module31"] and
                        group.get("l25_sha256") ==
                        main["checkpoint_sha256"]["generated-L25-module29"],
                        f"{identity}: output drift")
                require(group.get("immutable_mismatched_bytes") == 0 and
                        group.get("canary_mismatched_bytes") == 0 and
                        group.get("live_l23_device_handoff") is True and
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
                        ("l24-output.fp16", "l24_bytes", "l24_sha256"),
                        ("l25-output.fp16", "l25_bytes", "l25_sha256"),
                    ):
                        payload = (run_root / filename).read_bytes()
                        require(len(payload) == group[bytes_key] and
                                sha(payload) == group[hash_key],
                                f"{identity}: artifact")
        require(all(len({main["checkpoint_sha256"][name]
                         for main, group in rows}) == 1
                    for name in CHECKPOINTS),
                f"{resolution}: poison checkpoint")
        require(len({group["l24_sha256"] for main, group in rows}) == 1 and
                len({group["l25_sha256"] for main, group in rows}) == 1,
                f"{resolution}: poison group")
        main, group = rows[0]
        summaries[resolution] = {
            "checkpoint_sha256": main["checkpoint_sha256"],
            "l23_sha256": group["l23_pre_sha256"],
            "l24_bytes": group["l24_bytes"],
            "l24_sha256": group["l24_sha256"],
            "l25_bytes": group["l25_bytes"],
            "l25_sha256": group["l25_sha256"],
        }
    require(len(process_ids) == 16, f"{kind} process count")
    result = rooted({
        "schema": f"neuralstudio.fg.connected-dynamic-l3-l25-{kind}-backend.v1",
        "backend": backend,
        "fresh_process_count": 16,
        "selected_launches_per_process": 27,
        "resolutions": summaries,
        "run_receipt_sha256": receipt_hashes,
        "checkpoint_repeat_and_dual_poison_exact": True,
        "full_write_coverage_from_dual_poison": True,
        "live_l23_device_handoff": True,
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
        for filename in ("l24-output.fp16", "l25-output.fp16"):
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
    model_state = model_state_identity(repository)
    validate_collection(args.rtx.resolve(), repository)
    rtx_g21 = load(args.rtx / "generator21-receipt.json")
    rtx_g23 = load(args.rtx / "generator23-receipt.json")
    rtx_g25 = load(args.rtx / "generator25-receipt.json")
    mac_g21 = load(args.mac / "base21-generated/generator-receipt.json")
    mac_g23 = load(args.mac / "base23-generated/generator-receipt.json")
    mac_g25 = load(args.mac / "generated/generator-receipt.json")
    validate_generator(rtx_g21, GENERATOR21,
                       "neuralstudio.fg.connected-dynamic-l3-l21-generator.v1")
    validate_generator(mac_g21, GENERATOR21,
                       "neuralstudio.fg.connected-dynamic-l3-l21-generator.v1")
    validate_generator(rtx_g23, GENERATOR23,
                       "neuralstudio.fg.connected-dynamic-l3-l23-generator.v1")
    validate_generator(mac_g23, GENERATOR23,
                       "neuralstudio.fg.connected-dynamic-l3-l23-generator.v1")
    validate_generator(rtx_g25, GENERATOR25,
                       "neuralstudio.fg.connected-dynamic-l3-l25-generator.v1")
    validate_generator(mac_g25, GENERATOR25,
                       "neuralstudio.fg.connected-dynamic-l3-l25-generator.v1")
    require(rtx_g21 == mac_g21 and rtx_g23 == mac_g23 and
            rtx_g25 == mac_g25,
            "cross-platform generator drift")
    rig_path = args.rtx / "rig-receipt.json"
    rig = load(rig_path)
    require(rig.get("schema") ==
            "neuralstudio.fg.connected-dynamic-l3-l25-rig.v1" and
            rig.get("job_id") == JOB and
            rig.get("attempt_id") == "oracle-v1" and
            rig.get("smoke") is False, "rig identity")
    require(rig.get("generator21_sha256") ==
            source_hashes["scripts/generate-fg-connected-dynamic-l3-l21.py"] and
            rig.get("generator23_sha256") ==
            source_hashes["scripts/generate-fg-connected-dynamic-l3-l23.py"] and
            rig.get("generator25_sha256") ==
            source_hashes["scripts/generate-fg-connected-dynamic-l3-l25.py"] and
            rig.get("tail25_sha256") ==
            source_hashes["src/fg_connected_dynamic_l3_l25/rtx_tail.inc"] and
            rig.get("harness_sha256") ==
            source_hashes["scripts/build-fg-connected-dynamic-l3-l25-rtx.ps1"],
            "rig source")
    require(rig.get("weight67_sha256") == WEIGHT and
            rig.get("module31_sha256") == MODULE31 and
            rig.get("run_count") == rig.get("fresh_process_count") == 16 and
            rig.get("selected_launches_per_process") == 27 and
            rig.get("live_device_handoffs") is True and
            rig.get("expected_or_archived_intermediate_ingress") is False and
            rig.get("result") == "pass", "rig execution")
    address_free(rig, "rig")
    rtx, rtx_summary = validate_backend("rtx", args.rtx.resolve())
    mac, mac_summary = validate_backend("mac", args.mac.resolve())
    require(rtx_summary == mac_summary, "backend disagreement")
    compared, mismatched, comparisons = compare_outputs(
        args.rtx.resolve(), args.mac.resolve())
    require((compared, mismatched) == (11_673_600, 0),
            "full artifact mismatch")
    topology = rooted({
        "schema": "neuralstudio.fg.connected-dynamic-l3-l25-topology.v1",
        "parent_receipt_root_sha256": PARENT,
        "base_generator_root_sha256": GENERATOR21,
        "l23_generator_root_sha256": GENERATOR23,
        "l25_generator_root_sha256": GENERATOR25,
        "checkpoint_order": list(CHECKPOINTS),
        "resolutions": RESOLUTIONS,
        "model_state": model_state,
        "l24": {"module": 31, "module_sha256": MODULE31,
                "weight_token": 67, "weight_sha256": WEIGHT,
                "operation": "3x3 64-to-128 leaky",
                "packed_parameter_bytes": 48,
                "parameter_fields": [
                    "weight_u64", "input_u64", "width_u32", "height_u32",
                    "padded_width_u32", "padded_height_u32", "output_u64",
                    "output_width_u32", "output_height_u32"],
                "block": [32, 1, 4],
                "grid_expression": ["ceil(width/16)", "height", "1"]},
        "l25": {"module": 29, "module_sha256": MODULE29,
                "operation": "2x2 max pool 128 channels",
                "packed_parameter_bytes": 36,
                "parameter_fields": [
                    "input_u64", "output_u64", "input_width_u32",
                    "input_height_u32", "output_width_u32",
                    "output_height_u32", "channels_u32"],
                "block": [4, 4, 16],
                "grid_expression": ["ceil(output_width/4)",
                                    "ceil(output_height/4)", "1"]},
        "live_l23_device_handoff": True,
        "result": "pass",
    })
    implementation = rooted({
        "schema": "neuralstudio.fg.connected-dynamic-l3-l25-implementation.v1",
        "source_sha256": source_hashes,
        "hmma_signed_zero_rule":
            "canonicalize exact negative-zero accumulator after every selected HMMA stage; preserve final leaky multiply selection",
        "per_input_patch_count": 0,
        "expected_output_embedded": False,
        "captured_intermediate_embedded": False,
        "result": "pass",
    })
    failures = rooted({
        "schema": "neuralstudio.fg.connected-dynamic-l3-l25-counterexamples.v1",
        "items": [
            {"name": "literal wildcard remote staging",
             "result": "rejected-before-execution",
             "detail": "PowerShell Copy-Item -LiteralPath did not expand the wildcard and copied zero inherited files"},
            {"name": "quoted PowerShell file path over SSH",
             "result": "rejected-before-execution",
             "detail": "nested quotes produced an unsupported path; forward-slash path invocation was used without changing the job"},
        ],
        "all_failures_preserved": True,
        "result": "pass",
    })
    accepted = rooted({
        "schema": "neuralstudio.fg.connected-dynamic-l3-l25-accepted.v1",
        "selected_range": "connected dynamic stages L3 through L25",
        "resolutions": list(RESOLUTIONS),
        "processes_per_backend": 16,
        "selected_launches_per_process": 27,
        "rtx_mac_checkpoint_comparisons": 432,
        "rtx_mac_checkpoint_mismatches": 0,
        "l24_l25_full_artifact_compared_bytes": compared,
        "l24_l25_full_artifact_mismatched_bytes": mismatched,
        "topology_receipt_root_sha256": topology["receipt_root_sha256"],
        "implementation_receipt_root_sha256":
            implementation["receipt_root_sha256"],
        "counterexamples_root_sha256": failures["receipt_root_sha256"],
        "rtx_backend_root_sha256": rtx["receipt_root_sha256"],
        "mac_backend_root_sha256": mac["receipt_root_sha256"],
        "rtx_rig_receipt_sha256": file_sha(rig_path),
        "rtx_collection_receipt_sha256": file_sha(
            args.rtx / "collection-receipt.json"),
        "comparisons": comparisons,
        "outputs": {
            name: {key: value for key, value in summary.items()
                   if key != "checkpoint_sha256"}
            for name, summary in rtx_summary.items()},
        "claims": {
            "connected_dynamic_L3_L25_exact": True,
            "L24_checkpoint_exact": True,
            "L25_checkpoint_exact": True,
            "live_L23_handoff_exact": True,
            "token67_immutable": True,
            "hmma_signed_zero_rule_exact_for_module31": True,
            "connected_L26_and_later": False,
            "complete_generated_frame": False,
            "Netal_integration": False,
        },
        "scope": "connected dynamic path through third encoder pooling only",
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
    print(f"l24_l25_full_artifact_compared_bytes={compared}")
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
            topology.get("model_state") == model_state_identity(repository) and
            topology.get("l24", {}).get("packed_parameter_bytes") == 48 and
            topology.get("l24", {}).get("block") == [32, 1, 4] and
            topology.get("l25", {}).get("packed_parameter_bytes") == 36 and
            topology.get("l25", {}).get("block") == [4, 4, 16] and
            topology.get("live_l23_device_handoff") is True,
            "topology drift")
    require(implementation.get("source_sha256") ==
            source_identity(repository) and
            implementation.get("per_input_patch_count") == 0 and
            implementation.get("expected_output_embedded") is False,
            "implementation drift")
    require(failures.get("all_failures_preserved") is True and
            len(failures.get("items", [])) == 2, "failure ledger drift")
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
    require(accepted.get("rtx_mac_checkpoint_comparisons") == 432 and
            accepted.get("rtx_mac_checkpoint_mismatches") == 0 and
            accepted.get("l24_l25_full_artifact_compared_bytes") == 11_673_600 and
            accepted.get("l24_l25_full_artifact_mismatched_bytes") == 0,
            "accepted comparison drift")
    claims = accepted.get("claims", {})
    require(claims.get("connected_dynamic_L3_L25_exact") is True and
            claims.get("L24_checkpoint_exact") is True and
            claims.get("L25_checkpoint_exact") is True and
            claims.get("hmma_signed_zero_rule_exact_for_module31") is True and
            claims.get("connected_L26_and_later") is False and
            claims.get("complete_generated_frame") is False,
            "scope drift")
    if require_negative:
        negative = load(evidence / "negative-audit.json")
        verify_root(negative, "negative")
        require(negative.get("mutation_count", 0) >= 14 and
                negative.get("all_mutations_rejected") is True and
                negative.get("verifier_sha256") == file_sha(
                    repository / "scripts/verify-fg-connected-dynamic-l3-l25.py"),
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
