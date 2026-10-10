#!/usr/bin/env python3
"""Build and verify the complete selected-convolution FG state package v4."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


REPOSITORY = Path(__file__).resolve().parent.parent
V3_BUILDER = REPOSITORY / "scripts/build-fg-model-state-package-v3.py"
V3_PACKAGE = REPOSITORY / "evidence/specs/fg-model-state-package-v3"
SCHEMA = "neuralstudio.fg.model-state-package.v4"
AUDIT_SCHEMA = "neuralstudio.fg.model-state-package-negative-audit.v4"
PACKAGE_FILE = "selected-convolution-state.bin"
AGGREGATE_PATH = REPOSITORY / "evidence/specs/fg-custom-conv-25-aggregate-v1/aggregate.json"
AGGREGATE_ROOT_SHA256 = "6309c8c32c7fcb17434f429f16220b2385cfe6e8f34dfd329d5abede3641e1c0"
TOTAL_CAPTURE_BUFFERS = 67
TOTAL_CAPTURE_BYTES = 4_868_576


def load_v3() -> Any:
    spec = importlib.util.spec_from_file_location(
        "fg_model_state_package_v3_dependency", V3_BUILDER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load v3 model-state package builder")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


v3 = load_v3()


@dataclass(frozen=True)
class BottomConsumer:
    consumer_id: str
    launch: int
    function_name: str
    module_token: int
    module_sha256: str
    weight_token: int
    weight_bytes: int
    weight_sha256: str
    bias_token: int
    bias_bytes: int
    bias_sha256: str
    spec_directory: str
    schema_stem: str
    exact_root_sha256: str


BOTTOM_CONSUMERS = (
    BottomConsumer(
        "custom-block0-bot0", 68, "custom_block0_conv_bot0_hf_kernel", 58,
        "3acb96881f20a50e575e2b2959f0aa433e03e3da20fbf624ae6bb183d77a3e48",
        150, 110592, "6b59f16f1be43617ebf0c94ff29cf7c8b7a2f2c7b7bfb41df8f9ede585e70684",
        152, 192, "c3e301df2e15c48e0c2fb35fc636b944653c6f91071a101d71b256b740224d29",
        "fg-custom-bot0-v1", "custom-bot0",
        "591d4db70fdbf249268d67df78f8ff2075d0a54bcd51c0512441a5c752ebddbd",
    ),
    BottomConsumer(
        "custom-block0-bot1", 69, "custom_block0_conv_bot1_hf_kernel", 59,
        "4d35017769f6338b6f8ba3ac3079756c32410eeacca7f817e48ad5d7dc59eb3c",
        154, 4608, "4d477655f6ec78e0e6a0d3c41e0848e537091c0c311fb04d986ad8891823e4e8",
        156, 16, "0a9a85630034bb3dc5442cd52381424c1e7b9c676014b764477341005cb2f998",
        "fg-custom-block0-bot1-v1", "custom-block0-bot1",
        "37f79782c90ada12af75a80ffc5eff8aeca6d27614de1193fb0fedcb3a4836ae",
    ),
    BottomConsumer(
        "custom-block1-bot0", 82, "custom_block1_conv_bot0_hf_kernel", 65,
        "0f80c161e89ca4354f932ea6c3f7a0d3206e0a4fff7530d1da0fe4b162102faa",
        198, 27648, "4ab572a084310d9f79b7878aefd73a356900ea8428320b861c6de28c114977d6",
        200, 96, "3ae35c7933b00a52114c7a75f0184bd480c5c814d2204271bd46da8f88772913",
        "fg-custom-block1-bot0-v1", "custom-block1-bot0",
        "f5a92553b3d6640a7a2f440ec1b7e5b45dd1528677bb96b6da12a3bb9c19a1f9",
    ),
    BottomConsumer(
        "custom-block1-bot1", 83, "custom_block1_conv_bot1_hf_kernel", 66,
        "6f82303a09e3d7f5da446047e0b9667d8da04cfd96f6b48d38c460a95c55953a",
        202, 2304, "1514d5a5c51a81d8436de6e8e538275895dafc2e072512dbdcbe640871ec48f0",
        204, 16, "89f7b17382bc6f08bbbc9d416da1c4c5b72abaf0c4faf870966a2585f912b493",
        "fg-custom-block1-bot1-v1", "custom-block1-bot1",
        "b1af4def5512e99ef05b4a5f1c43ba10928a232949543b0e0c37d759be1e9aac",
    ),
)


class PackageV4Error(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PackageV4Error(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rooted(value: dict[str, Any], field: str = "receipt_root_sha256") -> str:
    body = copy.deepcopy(value)
    body.pop(field, None)
    return canonical_sha256(body)


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(),
            f"missing or unsafe JSON: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as error:
        raise PackageV4Error(f"invalid JSON: {path}: {error}") from error
    require(isinstance(value, dict), f"JSON is not an object: {path}")
    return value


def source_record(path: Path) -> dict[str, str]:
    return {
        "relative_path": str(path.relative_to(REPOSITORY)),
        "file_sha256": file_sha256(path),
    }


def validate_source_record(record: Any, path: Path, label: str) -> None:
    require(isinstance(record, dict) and
            record.get("relative_path") == str(path.relative_to(REPOSITORY)) and
            record.get("file_sha256") == file_sha256(path),
            f"{label}: source file drift")


def require_address_free(value: Any, label: str) -> None:
    def visit(node: Any, location: str) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                if key not in ("address_fields_recorded", "addresses_recorded"):
                    lowered = key.lower()
                    require(not any(part in lowered for part in (
                        "pointer", "virtual_address", "device_address", "gpu_address")),
                        f"{label}: address-bearing key at {location}.{key}")
                visit(child, f"{location}.{key}")
        elif isinstance(node, list):
            for index, child in enumerate(node):
                visit(child, f"{location}[{index}]")
        elif isinstance(node, str):
            require(re.search(r"0x[0-9a-fA-F]{12,16}(?:\b|$)", node) is None,
                    f"{label}: address-like text at {location}")
    visit(value, label)


def write_exclusive(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    write_exclusive(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def validate_aggregate(inventory_by_launch: dict[int, dict[str, Any]]) -> dict[str, Any]:
    aggregate = load_json(AGGREGATE_PATH)
    require(aggregate.get("schema") == "neuralstudio.fg.custom-conv-25-aggregate.v1" and
            aggregate.get("receipt_root_sha256") == AGGREGATE_ROOT_SHA256 and
            rooted(aggregate) == AGGREGATE_ROOT_SHA256,
            "custom-convolution aggregate identity drift")
    require(aggregate.get("launch_count") == 25 and
            aggregate.get("every_launch_executed_independently") is True and
            aggregate.get("every_launch_rtx_mac_exact") is True and
            aggregate.get("fresh_process_and_two_poison_gates_in_source_receipts") is True and
            aggregate.get("connected_execution_claimed") is False and
            aggregate.get("full_frame_claimed") is False and
            aggregate.get("address_fields_recorded") is False and
            aggregate.get("result") == "pass",
            "custom-convolution aggregate gate drift")
    launches = aggregate.get("launches")
    require(isinstance(launches, list) and len(launches) == 25,
            "custom-convolution aggregate launch count")
    for entry in launches:
        inventory = inventory_by_launch.get(entry.get("launch"), {})
        require(entry.get("module_token") == inventory.get("module_token") and
                entry.get("module_sha256") == inventory.get("module_sha256") and
                entry.get("function_name") == inventory.get("function_name") and
                entry.get("weight_token") == inventory.get("weight_token") and
                entry.get("weight_sha256") == inventory.get("weight_sha256") and
                entry.get("bias_token") == inventory.get("bias_token") and
                entry.get("bias_sha256") == inventory.get("bias_sha256"),
                f"aggregate/inventory drift at launch {entry.get('launch')}")
    require_address_free(aggregate, "custom-convolution aggregate")
    return aggregate


def validate_backend(value: dict[str, Any], consumer: BottomConsumer,
                     backend: str, expected_root: str) -> None:
    require(value.get("schema") ==
            f"neuralstudio.fg.{consumer.schema_stem}-backend.v1" and
            value.get("backend") == backend and
            value.get("receipt_root_sha256") == expected_root and
            rooted(value) == expected_root,
            f"{consumer.consumer_id}/{backend}: backend identity")
    require(value.get("selected_unit") ==
            f"evaluate-1/launch-{consumer.launch:06d}/{consumer.function_name}/module-{consumer.module_token}" and
            value.get("module_sha256") == consumer.module_sha256 and
            value.get("weight_sha256") == consumer.weight_sha256 and
            value.get("bias_sha256") == consumer.bias_sha256,
            f"{consumer.consumer_id}/{backend}: state identity")
    require(value.get("fresh_process_count") == 4 and
            value.get("two_process_two_poison_invariant") is True and
            value.get("full_write_coverage_from_two_poison_invariance") is True and
            value.get("input_model_immutability") is True and
            value.get("canary_integrity") is True and
            value.get("output_mutation_rejected") is True and
            value.get("expected_output_fed_to_execution") is False and
            value.get("model_manifest_sha256") == v3.v2.v1.CAPTURE_MANIFEST_SHA256 and
            value.get("address_fields_recorded") is False and
            value.get("result") == "pass",
            f"{consumer.consumer_id}/{backend}: execution gates")
    require_address_free(value, f"{consumer.consumer_id}/{backend}")


def validate_bottom_evidence(
    inventory_by_launch: dict[int, dict[str, Any]], aggregate: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    aggregate_by_launch = {entry["launch"]: entry for entry in aggregate["launches"]}
    consumers: list[dict[str, Any]] = []
    receipt_bytes: dict[str, bytes] = {}
    for consumer in BOTTOM_CONSUMERS:
        inventory = inventory_by_launch.get(consumer.launch, {})
        require(inventory.get("launch_token") == f"launch-{consumer.launch:06d}" and
                inventory.get("function_name") == consumer.function_name and
                inventory.get("module_token") == consumer.module_token and
                inventory.get("module_sha256") == consumer.module_sha256 and
                inventory.get("weight_token") == consumer.weight_token and
                inventory.get("weight_bytes") == consumer.weight_bytes and
                inventory.get("weight_sha256") == consumer.weight_sha256 and
                inventory.get("bias_token") == consumer.bias_token and
                inventory.get("bias_bytes") == consumer.bias_bytes and
                inventory.get("bias_sha256") == consumer.bias_sha256,
                f"{consumer.consumer_id}: inventory binding")
        aggregate_entry = aggregate_by_launch.get(consumer.launch, {})
        require(aggregate_entry.get("verification_root_sha256") ==
                consumer.exact_root_sha256,
                f"{consumer.consumer_id}: aggregate exact-root binding")
        root = REPOSITORY / "evidence/specs" / consumer.spec_directory
        final_path = root / "verification.json"
        artifact_path = root / "artifact-manifest.json"
        rtx_path = root / "rtx-backend.json"
        mac_path = root / "mac-backend.json"
        final = load_json(final_path)
        require(final.get("schema") ==
                f"neuralstudio.fg.{consumer.schema_stem}-verification.v1" and
                final.get("receipt_root_sha256") == consumer.exact_root_sha256 and
                rooted(final) == consumer.exact_root_sha256 and
                final.get("selected_unit") ==
                f"evaluate-1/launch-{consumer.launch:06d}/{consumer.function_name}/module-{consumer.module_token}" and
                final.get("module_sha256") == consumer.module_sha256 and
                final.get("weight_sha256") == consumer.weight_sha256 and
                final.get("bias_sha256") == consumer.bias_sha256 and
                final.get("rtx_mac_differing_bytes") == 0 and
                final.get("rtx_mac_exact") is True and
                final.get("address_fields_recorded") is False and
                final.get("result") == "pass",
                f"{consumer.consumer_id}: exact receipt")
        rtx = load_json(rtx_path)
        mac = load_json(mac_path)
        validate_backend(rtx, consumer, "rtx-cuda-driver",
                         final["rtx_backend_root_sha256"])
        validate_backend(mac, consumer, "mac-metal",
                         final["mac_backend_root_sha256"])
        require(rtx.get("output_sha256") == mac.get("output_sha256") ==
                final.get("output_sha256"),
                f"{consumer.consumer_id}: RTX/Mac output identity")
        packaged: dict[str, tuple[str, str]] = {}
        for label, receipt in (("rtx", rtx), ("mac", mac)):
            relative = f"backend-receipts/{consumer.consumer_id}-{label}.json"
            data = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
            receipt_bytes[relative] = data
            packaged[label] = (relative, bytes_sha256(data))
        consumers.append({
            "consumer_id": consumer.consumer_id,
            "family": "custom-bottom-convolution",
            "trace_launch_token": f"launch-{consumer.launch:06d}",
            "selected_unit":
                f"evaluate-1/launch-{consumer.launch:06d}/{consumer.function_name}/module-{consumer.module_token}",
            "function_name": consumer.function_name,
            "module_token": consumer.module_token,
            "module_sha256": consumer.module_sha256,
            "source_tokens": [consumer.weight_token, consumer.bias_token],
            "state_roles": ["weight", "bias"],
            "exact_receipt": source_record(final_path),
            "exact_receipt_root_sha256": consumer.exact_root_sha256,
            "aggregate_receipt_root_sha256": AGGREGATE_ROOT_SHA256,
            "artifact_manifest": source_record(artifact_path),
            "rtx_backend_receipt": packaged["rtx"][0],
            "rtx_backend_receipt_file_sha256": packaged["rtx"][1],
            "rtx_backend_receipt_root_sha256": final["rtx_backend_root_sha256"],
            "mac_backend_receipt": packaged["mac"][0],
            "mac_backend_receipt_file_sha256": packaged["mac"][1],
            "mac_backend_receipt_root_sha256": final["mac_backend_root_sha256"],
            "fresh_process_count_per_backend": 4,
            "poison_bytes": [90, 165],
            "expected_output_bytes_packaged": False,
            "expected_output_ingress_rejected_by_bound_verifier": True,
        })
    return consumers, receipt_bytes


def predecessor_receipts(manifest: dict[str, Any]) -> dict[str, bytes]:
    receipts: dict[str, bytes] = {}
    for consumer in manifest["proven_consumers"]:
        for label in ("rtx", "mac"):
            relative = consumer[f"{label}_backend_receipt"]
            data = (V3_PACKAGE / relative).read_bytes()
            require(bytes_sha256(data) ==
                    consumer[f"{label}_backend_receipt_file_sha256"],
                    f"{consumer['consumer_id']}/{label}: v3 receipt hash")
            receipts[relative] = data
    return receipts


def append_bottom_state(
    prefix: bytes, prefix_mappings: list[dict[str, Any]], buffers: dict[int, bytes]
) -> tuple[bytes, list[dict[str, Any]]]:
    package = bytearray(prefix)
    mappings = copy.deepcopy(prefix_mappings)
    for consumer in BOTTOM_CONSUMERS:
        for role, token, logical_bytes, source_sha in (
            ("weight", consumer.weight_token, consumer.weight_bytes,
             consumer.weight_sha256),
            ("bias", consumer.bias_token, consumer.bias_bytes,
             consumer.bias_sha256),
        ):
            data = buffers[token]
            require(len(data) == logical_bytes and bytes_sha256(data) == source_sha,
                    f"token {token}: captured state bytes")
            offset = len(package)
            package.extend(data)
            mappings.append({
                "consumer_id": consumer.consumer_id,
                "family": "custom-bottom-convolution",
                "trace_launch_token": f"launch-{consumer.launch:06d}",
                "state_role": role,
                "source_token": token,
                "logical_bytes": logical_bytes,
                "source_sha256": source_sha,
                "package_offset": offset,
                "package_end_exclusive": len(package),
            })
    return bytes(package), mappings


def expected_claims() -> dict[str, Any]:
    return {
        "generic_convolution_launches_proven": 17,
        "generic_state_buffers_proven": 17,
        "custom_convolution_launches_proven": 25,
        "custom_state_buffers_proven": 50,
        "bottom_branch_launches_proven": 4,
        "bottom_branch_state_buffers_proven": 8,
        "all_custom_convolution_launches_proven": True,
        "all_67_captured_buffers_direct_consumer_proven": True,
        "complete_selected_convolution_parameter_state_recovery": True,
        "complete_selected_model_path_recovery": False,
        "whole_model_weights_recovered": False,
        "connected_tail_proven_by_this_package": False,
        "full_generated_frame_parity": False,
        "netal_integration": False,
        "performance_claim": False,
    }


def validate_manifest_semantics(
    manifest: dict[str, Any], package: bytes, capture: dict[str, Any],
    inventory: dict[str, Any], aggregate: dict[str, Any]
) -> None:
    require(manifest.get("schema") == SCHEMA and
            manifest.get("address_fields_recorded") is False,
            "v4 package identity")
    require_address_free(manifest, "v4 package manifest")
    require(manifest.get("package_format") == {
        "file": PACKAGE_FILE,
        "encoding": "opaque captured bytes",
        "layout": "byte-identical predecessor-v3 prefix; then L68, L69, L82, L83 weight before bias; no padding",
        "padding_bytes": 0,
    }, "v4 package format")
    v3_manifest = v3.verify_package(V3_PACKAGE)
    source = manifest.get("source_evidence", {})
    validate_source_record(source.get("predecessor_v3_manifest"),
                           V3_PACKAGE / "package-manifest.json",
                           "predecessor v3 manifest")
    require(source.get("predecessor_v3_root_sha256") ==
            v3_manifest["package_root_sha256"] and
            source.get("predecessor_v3_payload_sha256") ==
            v3_manifest["included_state"]["package_sha256"],
            "predecessor v3 identity")
    validate_source_record(source.get("custom_convolution_aggregate"),
                           AGGREGATE_PATH, "custom aggregate source")
    require(source.get("custom_convolution_aggregate_root_sha256") ==
            AGGREGATE_ROOT_SHA256 and
            source.get("capture_manifest_sha256") ==
            v3.v2.v1.CAPTURE_MANIFEST_SHA256 and
            source.get("custom_convolution_inventory_root_sha256") ==
            v3.v2.INVENTORY_ROOT_SHA256 and
            source.get("capture_buffers_repeat_exact") is True and
            source.get("capture_bytes_fed_back_to_execution") is False,
            "v4 source evidence")
    require(bytes_sha256((V3_PACKAGE / v3.PACKAGE_FILE).read_bytes()) ==
            v3_manifest["included_state"]["package_sha256"],
            "v3 payload identity")
    prefix = (V3_PACKAGE / v3.PACKAGE_FILE).read_bytes()
    require(package[:len(prefix)] == prefix, "v3 payload prefix drift")

    included = manifest.get("included_state", {})
    mappings = included.get("mappings")
    require(included.get("classification") ==
            "complete_selected_convolution_parameter_state" and
            included.get("buffer_count") == TOTAL_CAPTURE_BUFFERS and
            included.get("logical_bytes") == TOTAL_CAPTURE_BYTES == len(package) and
            included.get("package_sha256") == bytes_sha256(package) and
            isinstance(mappings, list) and len(mappings) == TOTAL_CAPTURE_BUFFERS,
            "v4 included-state totals")
    require(mappings[:len(v3_manifest["included_state"]["mappings"])] ==
            v3_manifest["included_state"]["mappings"],
            "v3 mappings prefix drift")
    expected_new = []
    for consumer in BOTTOM_CONSUMERS:
        expected_new.extend([(consumer.weight_token, "weight"),
                             (consumer.bias_token, "bias")])
    new_mappings = mappings[len(v3_manifest["included_state"]["mappings"]):]
    require([(item.get("source_token"), item.get("state_role"))
             for item in new_mappings] == expected_new,
            "bottom-state append order")
    capture_entries = {item["source_token"]: item for item in capture["entries"]}
    tokens: list[int] = []
    offset = 0
    for mapping in mappings:
        token = mapping.get("source_token")
        logical_bytes = mapping.get("logical_bytes")
        end = mapping.get("package_end_exclusive")
        capture_entry = capture_entries.get(token, {})
        require(isinstance(token, int) and token not in tokens and
                mapping.get("package_offset") == offset and
                isinstance(logical_bytes, int) and logical_bytes > 0 and
                end == offset + logical_bytes and
                capture_entry.get("logical_bytes") == logical_bytes and
                capture_entry.get("sha256") == mapping.get("source_sha256") and
                bytes_sha256(package[offset:end]) == mapping.get("source_sha256"),
                f"mapping/capture/payload drift at token {token}")
        tokens.append(token)
        offset = end
    require(offset == len(package) and sorted(tokens) ==
            sorted(v3.v2.v1.EXPECTED_TOKENS),
            "complete capture partition drift")
    require(manifest.get("unverified_state") == {
        "classification": "none_all_captured_buffers_direct_consumer_proven",
        "buffer_count": 0,
        "logical_bytes": 0,
        "buffers": [],
    }, "v4 unverified-state partition")
    consumers = manifest.get("proven_consumers")
    require(isinstance(consumers, list) and len(consumers) == 42 and
            consumers[:len(v3_manifest["proven_consumers"])] ==
            v3_manifest["proven_consumers"] and
            [item.get("consumer_id") for item in consumers[-4:]] ==
            [item.consumer_id for item in BOTTOM_CONSUMERS],
            "v4 consumer closure")
    for record, consumer in zip(consumers[-4:], BOTTOM_CONSUMERS):
        require(record.get("exact_receipt_root_sha256") ==
                consumer.exact_root_sha256 and
                record.get("aggregate_receipt_root_sha256") ==
                AGGREGATE_ROOT_SHA256 and
                record.get("source_tokens") ==
                [consumer.weight_token, consumer.bias_token] and
                record.get("module_sha256") == consumer.module_sha256,
                f"{consumer.consumer_id}: manifest evidence binding")
    custom_launches = sorted(int(item["trace_launch_token"].split("-")[-1])
                             for item in consumers if item.get("family") !=
                             "generic-convolution")
    require(custom_launches == sorted(entry["launch"] for entry in
                                      aggregate["launches"]),
            "consumer/aggregate launch closure")
    require(manifest.get("claims") == expected_claims(), "v4 claim boundary")
    require(manifest.get("execution_provenance") == {
        "proven_consumer_count": 42,
        "generic_consumer_count": 17,
        "custom_consumer_count": 25,
        "directly_proven_state_buffer_count": 67,
        "per_consumer_backend_receipts_packaged": True,
        "backends": ["rtx-cuda-driver", "mac-metal"],
        "fresh_process_count_per_consumer_per_backend": 4,
        "poison_bytes": [90, 165],
        "two_process_two_poison_invariance_required": True,
        "input_and_model_immutability_required": True,
        "canary_integrity_required": True,
        "expected_output_fed_to_execution": False,
    }, "v4 execution provenance")
    require(manifest.get("execution_ingress") == {
        "contains_expected_output_bytes": False,
        "contains_oracle_output_bytes": False,
        "contains_activation_or_intermediate_buffers": False,
        "contains_unverified_state_buffers": False,
        "permitted_role": "complete selected-convolution parameter state only",
    }, "v4 execution ingress")
    require(inventory.get("custom_convolution_launch_count") == 25 and
            aggregate.get("receipt_root_sha256") == AGGREGATE_ROOT_SHA256,
            "v4 inventory/aggregate closure")


def validate_new_consumers_and_receipts(
    package_dir: Path, manifest: dict[str, Any],
    inventory_by_launch: dict[int, dict[str, Any]], aggregate: dict[str, Any]
) -> None:
    records = {item["consumer_id"]: item for item in manifest["proven_consumers"]}
    aggregate_by_launch = {item["launch"]: item for item in aggregate["launches"]}
    for consumer in BOTTOM_CONSUMERS:
        record = records.get(consumer.consumer_id, {})
        require(record.get("exact_receipt_root_sha256") ==
                consumer.exact_root_sha256 and
                record.get("aggregate_receipt_root_sha256") ==
                AGGREGATE_ROOT_SHA256 and
                record.get("source_tokens") ==
                [consumer.weight_token, consumer.bias_token],
                f"{consumer.consumer_id}: packaged consumer identity")
        final_path = (REPOSITORY / record["exact_receipt"]["relative_path"])
        validate_source_record(record["exact_receipt"], final_path,
                               f"{consumer.consumer_id}: exact receipt")
        require(aggregate_by_launch[consumer.launch]["verification_root_sha256"] ==
                consumer.exact_root_sha256 and
                inventory_by_launch[consumer.launch]["weight_token"] ==
                consumer.weight_token,
                f"{consumer.consumer_id}: inventory/aggregate binding")
        final = load_json(final_path)
        for label, backend in (("rtx", "rtx-cuda-driver"),
                               ("mac", "mac-metal")):
            receipt_path = package_dir / record[f"{label}_backend_receipt"]
            require(file_sha256(receipt_path) ==
                    record[f"{label}_backend_receipt_file_sha256"],
                    f"{consumer.consumer_id}/{label}: packaged receipt hash")
            receipt = load_json(receipt_path)
            validate_backend(receipt, consumer, backend,
                             record[f"{label}_backend_receipt_root_sha256"])
            require(receipt["receipt_root_sha256"] ==
                    final[f"{label}_backend_root_sha256"],
                    f"{consumer.consumer_id}/{label}: final binding")


def negative_audit(
    base: dict[str, Any], package: bytes, capture: dict[str, Any],
    inventory: dict[str, Any], aggregate: dict[str, Any]
) -> dict[str, Any]:
    cases: list[dict[str, str]] = []

    def validate(mutated_manifest: dict[str, Any] | None = None,
                 mutated_package: bytes | None = None,
                 mutated_capture: dict[str, Any] | None = None,
                 mutated_inventory: dict[str, Any] | None = None,
                 mutated_aggregate: dict[str, Any] | None = None) -> None:
        validate_manifest_semantics(
            mutated_manifest or base, mutated_package or package,
            mutated_capture or capture, mutated_inventory or inventory,
            mutated_aggregate or aggregate)

    def reject(name: str, action: Callable[[], None]) -> None:
        try:
            action()
        except (PackageV4Error, v3.PackageV3Error, v3.v2.PackageV2Error,
                v3.v2.v1.PackageError, KeyError, TypeError, IndexError) as error:
            cases.append({"name": name, "result": "rejected",
                          "error": str(error)})
            return
        raise PackageV4Error(f"mutation was not rejected: {name}")

    corrupt = bytearray(package); corrupt[-1] ^= 1
    reject("payload_byte_flip", lambda: validate(mutated_package=bytes(corrupt)))
    reject("payload_truncation", lambda: validate(mutated_package=package[:-1]))
    corrupt = bytearray(package); corrupt[len((V3_PACKAGE / v3.PACKAGE_FILE).read_bytes()) - 1] ^= 1
    reject("predecessor_prefix_flip", lambda: validate(mutated_package=bytes(corrupt)))
    mutated = copy.deepcopy(base); mutated["included_state"]["mappings"][-1]["package_offset"] -= 1
    reject("mapping_offset_drift", lambda: validate(mutated_manifest=mutated))
    mutated = copy.deepcopy(base); mutated["included_state"]["mappings"][-1]["source_sha256"] = "0" * 64
    reject("mapping_hash_drift", lambda: validate(mutated_manifest=mutated))
    mutated = copy.deepcopy(base); mutated["included_state"]["buffer_count"] = 66
    reject("buffer_count_drift", lambda: validate(mutated_manifest=mutated))
    mutated = copy.deepcopy(base); mutated["unverified_state"]["buffer_count"] = 1
    reject("unverified_state_reintroduced", lambda: validate(mutated_manifest=mutated))
    mutated = copy.deepcopy(base); mutated["proven_consumers"] = mutated["proven_consumers"][:-1]
    reject("consumer_omission", lambda: validate(mutated_manifest=mutated))
    mutated = copy.deepcopy(base); mutated["proven_consumers"][-1]["exact_receipt_root_sha256"] = "1" * 64
    reject("bottom_root_drift", lambda: validate(mutated_manifest=mutated))
    mutated = copy.deepcopy(base); mutated["source_evidence"]["custom_convolution_aggregate_root_sha256"] = "2" * 64
    reject("aggregate_root_drift", lambda: validate(mutated_manifest=mutated))
    mutated = copy.deepcopy(base); mutated["claims"]["complete_selected_convolution_parameter_state_recovery"] = False
    reject("state_recovery_demotion", lambda: validate(mutated_manifest=mutated))
    mutated = copy.deepcopy(base); mutated["claims"]["whole_model_weights_recovered"] = True
    reject("whole_model_overclaim", lambda: validate(mutated_manifest=mutated))
    mutated = copy.deepcopy(base); mutated["claims"]["connected_tail_proven_by_this_package"] = True
    reject("connected_tail_overclaim", lambda: validate(mutated_manifest=mutated))
    mutated = copy.deepcopy(base); mutated["claims"]["full_generated_frame_parity"] = True
    reject("full_frame_overclaim", lambda: validate(mutated_manifest=mutated))
    mutated = copy.deepcopy(base); mutated["execution_ingress"]["contains_expected_output_bytes"] = True
    reject("expected_output_ingress", lambda: validate(mutated_manifest=mutated))
    mutated = copy.deepcopy(base); mutated["execution_ingress"]["contains_activation_or_intermediate_buffers"] = True
    reject("activation_ingress", lambda: validate(mutated_manifest=mutated))
    mutated = copy.deepcopy(aggregate); mutated.pop("receipt_root_sha256"); mutated["launch_count"] = 24; mutated["receipt_root_sha256"] = canonical_sha256(mutated)
    reject("aggregate_launch_omission", lambda: validate(mutated_aggregate=mutated))
    mutated = copy.deepcopy(base); mutated["volatile_pointer"] = "0x123456789abc"
    reject("address_injection", lambda: validate(mutated_manifest=mutated))
    require(len(cases) == 18 and all(case["result"] == "rejected" for case in cases),
            "v4 negative audit incomplete")
    audit = {
        "schema": AUDIT_SCHEMA,
        "mutation_count": len(cases),
        "rejected_count": len(cases),
        "failed_to_reject": [],
        "cases": cases,
    }
    audit["receipt_root_sha256"] = canonical_sha256(audit)
    return audit


def build_package(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    require(not output_dir.is_symlink(), "unsafe v4 output directory")
    generated = [output_dir / PACKAGE_FILE, output_dir / "capture-manifest.json",
                 output_dir / "custom-conv-inventory.json",
                 output_dir / "custom-conv-25-aggregate.json",
                 output_dir / "negative-audit.json",
                 output_dir / "package-manifest.json",
                 output_dir / "backend-receipts"]
    require(not any(path.exists() for path in generated),
            "refusing to overwrite immutable v4 artifacts")

    v3_manifest = v3.verify_package(V3_PACKAGE)
    capture, capture_entries, buffers = v3.v2.v1.load_capture_pair(args.source_job)
    inventory_path = REPOSITORY / "evidence/specs/fg-custom-conv-inventory-v1/inventory.json"
    inventory, inventory_by_launch = v3.v2.validate_inventory(inventory_path)
    aggregate = validate_aggregate(inventory_by_launch)
    bottom_consumers, bottom_receipts = validate_bottom_evidence(
        inventory_by_launch, aggregate)
    consumers = copy.deepcopy(v3_manifest["proven_consumers"]) + bottom_consumers
    receipts = {**predecessor_receipts(v3_manifest), **bottom_receipts}
    prefix = (V3_PACKAGE / v3.PACKAGE_FILE).read_bytes()
    package, mappings = append_bottom_state(
        prefix, v3_manifest["included_state"]["mappings"], buffers)
    included_tokens = {item["source_token"] for item in mappings}
    require(included_tokens == set(capture_entries),
            "v4 does not cover the complete captured token set")
    base = {
        "schema": SCHEMA,
        "package_format": {
            "file": PACKAGE_FILE,
            "encoding": "opaque captured bytes",
            "layout": "byte-identical predecessor-v3 prefix; then L68, L69, L82, L83 weight before bias; no padding",
            "padding_bytes": 0,
        },
        "source_evidence": {
            "predecessor_v3_manifest": source_record(V3_PACKAGE / "package-manifest.json"),
            "predecessor_v3_root_sha256": v3_manifest["package_root_sha256"],
            "predecessor_v3_payload_sha256":
                v3_manifest["included_state"]["package_sha256"],
            "runtime_sha256": v3.v2.v1.RUNTIME_SHA256,
            "capture_point": v3.v2.v1.CAPTURE_POINT,
            "capture_manifest_file": "capture-manifest.json",
            "capture_manifest_sha256": v3.v2.v1.CAPTURE_MANIFEST_SHA256,
            "capture_fresh_process_count": 2,
            "capture_poison_bytes": [90, 165],
            "capture_buffers_repeat_exact": True,
            "capture_bytes_fed_back_to_execution": False,
            "custom_convolution_inventory": source_record(inventory_path),
            "custom_convolution_inventory_file": "custom-conv-inventory.json",
            "custom_convolution_inventory_root_sha256":
                v3.v2.INVENTORY_ROOT_SHA256,
            "custom_convolution_aggregate": source_record(AGGREGATE_PATH),
            "custom_convolution_aggregate_file":
                "custom-conv-25-aggregate.json",
            "custom_convolution_aggregate_root_sha256": AGGREGATE_ROOT_SHA256,
            "builder_source": source_record(Path(__file__).resolve()),
        },
        "proven_consumers": consumers,
        "included_state": {
            "classification": "complete_selected_convolution_parameter_state",
            "buffer_count": len(mappings),
            "logical_bytes": len(package),
            "package_sha256": bytes_sha256(package),
            "mappings": mappings,
        },
        "unverified_state": {
            "classification": "none_all_captured_buffers_direct_consumer_proven",
            "buffer_count": 0,
            "logical_bytes": 0,
            "buffers": [],
        },
        "execution_provenance": {
            "proven_consumer_count": 42,
            "generic_consumer_count": 17,
            "custom_consumer_count": 25,
            "directly_proven_state_buffer_count": 67,
            "per_consumer_backend_receipts_packaged": True,
            "backends": ["rtx-cuda-driver", "mac-metal"],
            "fresh_process_count_per_consumer_per_backend": 4,
            "poison_bytes": [90, 165],
            "two_process_two_poison_invariance_required": True,
            "input_and_model_immutability_required": True,
            "canary_integrity_required": True,
            "expected_output_fed_to_execution": False,
        },
        "execution_ingress": {
            "contains_expected_output_bytes": False,
            "contains_oracle_output_bytes": False,
            "contains_activation_or_intermediate_buffers": False,
            "contains_unverified_state_buffers": False,
            "permitted_role": "complete selected-convolution parameter state only",
        },
        "claims": expected_claims(),
        "address_fields_recorded": False,
    }
    validate_manifest_semantics(base, package, capture, inventory, aggregate)
    audit = negative_audit(base, package, capture, inventory, aggregate)
    audit_bytes = (json.dumps(audit, indent=2, sort_keys=True) + "\n").encode()
    base["mutation_testing"] = {
        "file": "negative-audit.json",
        "file_sha256": bytes_sha256(audit_bytes),
        "receipt_root_sha256": audit["receipt_root_sha256"],
        "mutation_count": audit["mutation_count"],
        "rejected_count": audit["rejected_count"],
    }
    base["package_root_sha256"] = canonical_sha256(base)
    require_address_free(base, "final v4 package")
    try:
        write_exclusive(output_dir / PACKAGE_FILE, package)
        capture_source = (args.source_job / "eval0-a" /
                          "runtime-default-buffers-post-evaluate-0.manifest.json")
        write_exclusive(output_dir / "capture-manifest.json",
                        capture_source.read_bytes())
        write_exclusive(output_dir / "custom-conv-inventory.json",
                        inventory_path.read_bytes())
        write_exclusive(output_dir / "custom-conv-25-aggregate.json",
                        AGGREGATE_PATH.read_bytes())
        for relative, data in receipts.items():
            write_exclusive(output_dir / relative, data)
        write_exclusive(output_dir / "negative-audit.json", audit_bytes)
        write_json_exclusive(output_dir / "package-manifest.json", base)
    except Exception:
        for target in generated[:-1]:
            if target.exists() and target.is_file():
                target.unlink()
        receipt_dir = output_dir / "backend-receipts"
        if receipt_dir.exists() and receipt_dir.is_dir():
            shutil.rmtree(receipt_dir)
        raise
    return base


def verify_package(package_dir: Path) -> dict[str, Any]:
    manifest = load_json(package_dir / "package-manifest.json")
    require(manifest.get("package_root_sha256") ==
            rooted(manifest, "package_root_sha256"), "v4 package root invalid")
    payload_path = package_dir / PACKAGE_FILE
    require(payload_path.is_file() and not payload_path.is_symlink(),
            "v4 payload missing or unsafe")
    package = payload_path.read_bytes()
    capture_path = package_dir / "capture-manifest.json"
    require(file_sha256(capture_path) == v3.v2.v1.CAPTURE_MANIFEST_SHA256,
            "v4 packaged capture manifest hash")
    capture = load_json(capture_path)
    inventory_path = package_dir / "custom-conv-inventory.json"
    inventory, inventory_by_launch = v3.v2.validate_inventory(inventory_path)
    require(file_sha256(inventory_path) ==
            manifest["source_evidence"]["custom_convolution_inventory"]["file_sha256"],
            "v4 packaged inventory hash")
    aggregate_path = package_dir / "custom-conv-25-aggregate.json"
    require(file_sha256(aggregate_path) == file_sha256(AGGREGATE_PATH),
            "v4 packaged aggregate hash")
    aggregate = load_json(aggregate_path)
    require(aggregate == validate_aggregate(inventory_by_launch),
            "v4 packaged aggregate content")
    validate_manifest_semantics(manifest, package, capture, inventory, aggregate)
    validate_new_consumers_and_receipts(package_dir, manifest,
                                        inventory_by_launch, aggregate)
    for consumer in manifest["proven_consumers"][:-4]:
        for label in ("rtx", "mac"):
            path = package_dir / consumer[f"{label}_backend_receipt"]
            require(file_sha256(path) ==
                    consumer[f"{label}_backend_receipt_file_sha256"] and
                    path.read_bytes() ==
                    (V3_PACKAGE / consumer[f"{label}_backend_receipt"]).read_bytes(),
                    f"{consumer['consumer_id']}/{label}: v3 receipt prefix drift")
    audit_meta = manifest.get("mutation_testing", {})
    audit_path = package_dir / audit_meta.get("file", "")
    require(audit_path.is_file() and file_sha256(audit_path) ==
            audit_meta.get("file_sha256"), "v4 negative audit file hash")
    audit = load_json(audit_path)
    require(audit.get("schema") == AUDIT_SCHEMA and
            rooted(audit) == audit.get("receipt_root_sha256") ==
            audit_meta.get("receipt_root_sha256") and
            audit.get("mutation_count") == audit.get("rejected_count") ==
            audit_meta.get("mutation_count") == 18 and
            audit.get("failed_to_reject") == [],
            "v4 negative audit invalid")
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument(
        "--source-job", type=Path,
        default=REPOSITORY / "build/fg_rig/fg-default-readback-20260904T095705Z")
    build.add_argument("--output-dir", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--package-dir", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        manifest = (build_package(args) if args.command == "build" else
                    verify_package(args.package_dir.resolve()))
        print(f"package_root_sha256={manifest['package_root_sha256']}")
        print(f"package_sha256={manifest['included_state']['package_sha256']}")
        print(f"proven_buffers={manifest['included_state']['buffer_count']}")
        print(f"unverified_buffers={manifest['unverified_state']['buffer_count']}")
        print("result=pass")
        return 0
    except (OSError, KeyError, TypeError, PackageV4Error, v3.PackageV3Error,
            v3.v2.PackageV2Error, v3.v2.v1.PackageError) as error:
        print(f"package_v4_error={error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
