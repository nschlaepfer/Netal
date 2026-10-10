#!/usr/bin/env python3
"""Build and verify FG state package v3 with the exact block1 convolution state."""

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
V2_BUILDER = REPOSITORY / "scripts/build-fg-model-state-package-v2.py"
V2_PACKAGE = REPOSITORY / "evidence/specs/fg-model-state-package-v2"
SCHEMA = "neuralstudio.fg.model-state-package.v3"
AUDIT_SCHEMA = "neuralstudio.fg.model-state-package-negative-audit.v3"
PACKAGE_FILE = "selected-convolution-state.bin"
BLOCK1_CONV2_FAMILY_ROOT_SHA256 = "c0349c1c07ab903e656ee19d35b8edd4510fbacc96b129c004f4644a00cbfd23"


def load_v2() -> Any:
    spec = importlib.util.spec_from_file_location("fg_model_state_package_v2_dependency", V2_BUILDER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load v2 model-state package builder")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


v2 = load_v2()


@dataclass(frozen=True)
class Block1Consumer:
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
    residual_enabled: bool | None
    schema_stem: str
    spec_directory: str


BLOCK1_CONSUMERS = (
    Block1Consumer(
        "custom-block1-conv0", 72, "custom_block1_conv0_kernel", 62,
        "5a33f08403b49bbabd4dfb04f476daaac8f9fa0964befc752dc4e7fd6d2471d6",
        158, 5184, "255701f510034a1ec801207b0542c2d49919d131c35d1337a60ad16fdc9004e6",
        160, 32, "8ad9d15f85e8cea4e0a5b67792f39e170e75c7db010f727c7894e2c23be5d4c1",
        None, "custom-block1-conv0", "fg-custom-block1-conv0-v1",
    ),
    Block1Consumer(
        "custom-block1-conv1", 73, "custom_block1_conv1_kernel", 63,
        "28801f3c1c1ae5f4a98c852234a48b8f258ede6eec9e6c881f2cd18cdb804ef6",
        162, 9216, "33dfd20f3a6901ed847959663a234434ddc64d1a3a2b4d171bde63a7e21d4af9",
        164, 64, "f9a1b4cf11f4f4163d976260d8acf61a3a58e0db6ee7357bb502cb82b81f7889",
        None, "custom-block1-conv1", "fg-custom-block1-conv1-v1",
    ),
    Block1Consumer("custom-block1-conv2-l74", 74, "custom_block1_conv2_kernel", 64, "ae48122fca1d4494c10e3e1b27b53282e1998346dc6d6a1c61c2658bcbd0e474", 166, 18432, "bf39c57db6f20c38951cc91584a98543feb8f396c1529667d7541ce6d4732162", 168, 64, "87d1d98747c14b03644c43ef21aae52ad19a3058d0f5ce673d954bc3c9d3ebeb", False, "custom-block1-conv2", "fg-custom-block1-conv2-family-v1"),
    Block1Consumer("custom-block1-conv2-l75", 75, "custom_block1_conv2_kernel", 64, "ae48122fca1d4494c10e3e1b27b53282e1998346dc6d6a1c61c2658bcbd0e474", 170, 18432, "1ee5ffe7c824fec11aade50a433dd5c7d18e3f13452c84154c920f9b42e81653", 172, 64, "4cf2f999a3b60e58553c0fa0e32c134a2de378ee22d6d4d0a0140cb2e640b882", True, "custom-block1-conv2", "fg-custom-block1-conv2-family-v1"),
    Block1Consumer("custom-block1-conv2-l76", 76, "custom_block1_conv2_kernel", 64, "ae48122fca1d4494c10e3e1b27b53282e1998346dc6d6a1c61c2658bcbd0e474", 174, 18432, "a11cfbf3ae8b114b6e98f39fc59d11df69d9fc52741255130019cb7cd22f93b4", 176, 64, "0898c1586b71e2ea5bbd4d9cbaad3e15f416743544446cc8d2157a4f23f628fa", False, "custom-block1-conv2", "fg-custom-block1-conv2-family-v1"),
    Block1Consumer("custom-block1-conv2-l77", 77, "custom_block1_conv2_kernel", 64, "ae48122fca1d4494c10e3e1b27b53282e1998346dc6d6a1c61c2658bcbd0e474", 178, 18432, "21f1e5dcf02d33ad5eec044251148b5f851d264f86529a63633edb8a8ae2fe75", 180, 64, "ff53999e00052d0f16117a358f65efddb1d36814cc8820edc3b0f56fca630a07", True, "custom-block1-conv2", "fg-custom-block1-conv2-family-v1"),
    Block1Consumer("custom-block1-conv2-l78", 78, "custom_block1_conv2_kernel", 64, "ae48122fca1d4494c10e3e1b27b53282e1998346dc6d6a1c61c2658bcbd0e474", 182, 18432, "87b22102aab453ff63e8f2245a9ee9d9adce55b1b489d43a6246ae7b6b9d23fd", 184, 64, "2d134dee471fda6e55f345531803da054d8edf9c6a0385a637843f5e39d0619f", False, "custom-block1-conv2", "fg-custom-block1-conv2-family-v1"),
    Block1Consumer("custom-block1-conv2-l79", 79, "custom_block1_conv2_kernel", 64, "ae48122fca1d4494c10e3e1b27b53282e1998346dc6d6a1c61c2658bcbd0e474", 186, 18432, "f07e7bd536bbc5bd4dbb7e4d87c9575073d4fb8b3c119a11dae7b1cfbdce8170", 188, 64, "07d8006bd5c7857575653d74815dc300ca71048a72f105165f0a2534b7fec2b0", True, "custom-block1-conv2", "fg-custom-block1-conv2-family-v1"),
    Block1Consumer("custom-block1-conv2-l80", 80, "custom_block1_conv2_kernel", 64, "ae48122fca1d4494c10e3e1b27b53282e1998346dc6d6a1c61c2658bcbd0e474", 190, 18432, "d819d65c7f2582134c098b99b53105dd64c630554a072ef579a2d4bc0b51c713", 192, 64, "715c6b649120b4d1449a0b8654d19101ab660ca3f490d8b827778c817cf9c30b", False, "custom-block1-conv2", "fg-custom-block1-conv2-family-v1"),
    Block1Consumer("custom-block1-conv2-l81", 81, "custom_block1_conv2_kernel", 64, "ae48122fca1d4494c10e3e1b27b53282e1998346dc6d6a1c61c2658bcbd0e474", 194, 18432, "ab3a82644cdaca91b322abe6f835a63ef073005fceacf1cb8a1e66096c3bdac3", 196, 64, "a301af6f017ef2324d205769e43c4327fbc9e9aa20dce506b2663acc0adca06e", True, "custom-block1-conv2", "fg-custom-block1-conv2-family-v1"),
)


class PackageV3Error(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PackageV3Error(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def rooted(value: dict[str, Any], field: str = "receipt_root_sha256") -> str:
    body = copy.deepcopy(value)
    body.pop(field, None)
    return canonical_sha256(body)


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing or unsafe JSON: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as error:
        raise PackageV3Error(f"invalid JSON: {path}: {error}") from error
    require(isinstance(value, dict), f"JSON is not an object: {path}")
    return value


def source_record(path: Path) -> dict[str, str]:
    return {"relative_path": str(path.relative_to(REPOSITORY)), "file_sha256": file_sha256(path)}


def validate_source_record(record: Any, path: Path, label: str) -> None:
    require(
        isinstance(record, dict)
        and record.get("relative_path") == str(path.relative_to(REPOSITORY))
        and record.get("file_sha256") == file_sha256(path),
        f"{label}: source file drift",
    )


def require_address_free(value: Any, label: str) -> None:
    def visit(node: Any, location: str) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                if key not in ("address_fields_recorded", "addresses_recorded"):
                    require(
                        not any(part in key.lower() for part in ("pointer", "virtual_address", "device_address", "gpu_address")),
                        f"{label}: address-bearing key at {location}.{key}",
                    )
                visit(child, f"{location}.{key}")
        elif isinstance(node, list):
            for index, child in enumerate(node):
                visit(child, f"{location}[{index}]")
        elif isinstance(node, str):
            require(re.search(r"0x[0-9a-fA-F]{12,16}(?:\b|$)", node) is None, f"{label}: address-like text at {location}")
    visit(value, label)


def write_exclusive(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)


def write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    write_exclusive(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def block1_paths(consumer: Block1Consumer) -> dict[str, Path]:
    root = REPOSITORY / "evidence/specs" / consumer.spec_directory
    if consumer.residual_enabled is None:
        final = root / "verification.json"
        rtx = root / "rtx-backend.json"
        mac = root / "mac-backend.json"
    else:
        final = root / "layers" / f"l{consumer.launch}-verification.json"
        rtx = Path()
        mac = Path()
    return {"root": root, "final": final, "rtx": rtx, "mac": mac, "artifacts": root / "artifact-manifest.json"}


def validate_block1_backend(
    value: dict[str, Any], consumer: Block1Consumer, backend: str, expected_root: str
) -> None:
    require(
        value.get("schema") == f"neuralstudio.fg.{consumer.schema_stem}-backend.v1"
        and value.get("backend") == backend
        and value.get("receipt_root_sha256") == expected_root
        and rooted(value) == expected_root,
        f"{consumer.consumer_id}/{backend}: backend identity",
    )
    if consumer.residual_enabled is None:
        identity_ok = value.get("selected_unit") == (
            f"evaluate-1/launch-{consumer.launch:06d}/{consumer.function_name}/module-{consumer.module_token}"
        )
    else:
        identity_ok = (
            value.get("layer_id") == f"l{consumer.launch}"
            and value.get("trace_launch_token") == f"launch-{consumer.launch:06d}"
            and value.get("residual_enabled") is consumer.residual_enabled
        )
    require(
        identity_ok
        and value.get("module_sha256") == consumer.module_sha256
        and value.get("weight_sha256") == consumer.weight_sha256
        and value.get("bias_sha256") == consumer.bias_sha256,
        f"{consumer.consumer_id}/{backend}: state identity",
    )
    require(
        value.get("fresh_process_count") == 4
        and value.get("two_process_two_poison_invariant") is True
        and value.get("full_write_coverage_from_two_poison_invariance") is True
        and value.get("input_model_immutability") is True
        and value.get("canary_integrity") is True
        and value.get("output_mutation_rejected") is True
        and value.get("address_fields_recorded") is False
        and value.get("result") == "pass",
        f"{consumer.consumer_id}/{backend}: execution gates",
    )
    if consumer.residual_enabled is None:
        require(
            value.get("model_manifest_sha256") == v2.v1.CAPTURE_MANIFEST_SHA256
            and value.get("expected_output_fed_to_execution") is False,
            f"{consumer.consumer_id}/{backend}: ingress/model provenance",
        )
    require_address_free(value, f"{consumer.consumer_id}/{backend}")


def validate_block1_evidence(
    inventory_by_launch: dict[int, dict[str, Any]], rtx_conv2: Path, mac_conv2: Path
) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    family_root = REPOSITORY / "evidence/specs/fg-custom-block1-conv2-family-v1"
    family_path = family_root / "family-verification.json"
    family = load_json(family_path)
    require(
        family.get("schema") == "neuralstudio.fg.custom-block1-conv2-family.v1"
        and family.get("receipt_root_sha256") == BLOCK1_CONV2_FAMILY_ROOT_SHA256
        and rooted(family) == BLOCK1_CONV2_FAMILY_ROOT_SHA256
        and family.get("layer_count") == 8
        and family.get("every_launch_executed_independently") is True
        and family.get("every_launch_rtx_mac_exact") is True
        and family.get("residual_off_and_on_validated") is True
        and family.get("result") == "pass",
        "block1 conv2 family receipt drift",
    )
    family_members = {item["layer_id"]: item for item in family["layers"]}
    records: list[dict[str, Any]] = []
    receipt_bytes: dict[str, bytes] = {}
    for consumer in BLOCK1_CONSUMERS:
        inventory = inventory_by_launch.get(consumer.launch, {})
        require(
            inventory.get("launch_token") == f"launch-{consumer.launch:06d}"
            and inventory.get("function_name") == consumer.function_name
            and inventory.get("module_token") == consumer.module_token
            and inventory.get("module_sha256") == consumer.module_sha256
            and inventory.get("weight_token") == consumer.weight_token
            and inventory.get("weight_bytes") == consumer.weight_bytes
            and inventory.get("weight_sha256") == consumer.weight_sha256
            and inventory.get("bias_token") == consumer.bias_token
            and inventory.get("bias_bytes") == consumer.bias_bytes
            and inventory.get("bias_sha256") == consumer.bias_sha256,
            f"{consumer.consumer_id}: inventory binding",
        )
        paths = block1_paths(consumer)
        final = load_json(paths["final"])
        require(
            final.get("schema") == f"neuralstudio.fg.{consumer.schema_stem}-verification.v1"
            and rooted(final) == final.get("receipt_root_sha256")
            and final.get("module_sha256") == consumer.module_sha256
            and final.get("weight_sha256") == consumer.weight_sha256
            and final.get("bias_sha256") == consumer.bias_sha256
            and final.get("rtx_mac_differing_bytes") == 0
            and final.get("rtx_mac_exact") is True
            and final.get("result") == "pass",
            f"{consumer.consumer_id}: final exact receipt",
        )
        if consumer.residual_enabled is None:
            require(
                final.get("selected_unit")
                == f"evaluate-1/launch-{consumer.launch:06d}/{consumer.function_name}/module-{consumer.module_token}",
                f"{consumer.consumer_id}: selected unit",
            )
            rtx_path = paths["rtx"]
            mac_path = paths["mac"]
        else:
            member = family_members.get(f"l{consumer.launch}", {})
            require(
                final.get("layer_id") == f"l{consumer.launch}"
                and final.get("residual_enabled") is consumer.residual_enabled
                and final.get("receipt_root_sha256") == member.get("receipt_root_sha256"),
                f"{consumer.consumer_id}: family member binding",
            )
            rtx_path = rtx_conv2 / f"l{consumer.launch}" / "backend-receipt.json"
            mac_path = mac_conv2 / f"l{consumer.launch}" / "backend-receipt.json"
        rtx = load_json(rtx_path)
        mac = load_json(mac_path)
        validate_block1_backend(rtx, consumer, "rtx-cuda-driver", final["rtx_backend_root_sha256"])
        validate_block1_backend(mac, consumer, "mac-metal", final["mac_backend_root_sha256"])
        require(rtx.get("output_sha256") == mac.get("output_sha256") == final.get("output_sha256"), f"{consumer.consumer_id}: RTX/Mac output identity")
        packaged_paths: dict[str, str] = {}
        packaged_hashes: dict[str, str] = {}
        for label, value in (("rtx", rtx), ("mac", mac)):
            relative = f"backend-receipts/{consumer.consumer_id}-{label}.json"
            data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
            receipt_bytes[relative] = data
            packaged_paths[label] = relative
            packaged_hashes[label] = bytes_sha256(data)
        record = {
            "consumer_id": consumer.consumer_id,
            "family": "custom-block1-convolution",
            "trace_launch_token": f"launch-{consumer.launch:06d}",
            "selected_unit": f"evaluate-1/launch-{consumer.launch:06d}/{consumer.function_name}/module-{consumer.module_token}",
            "function_name": consumer.function_name,
            "module_token": consumer.module_token,
            "module_sha256": consumer.module_sha256,
            "source_tokens": [consumer.weight_token, consumer.bias_token],
            "state_roles": ["weight", "bias"],
            "exact_receipt": source_record(paths["final"]),
            "exact_receipt_root_sha256": final["receipt_root_sha256"],
            "artifact_manifest": source_record(paths["artifacts"]),
            "rtx_backend_receipt": packaged_paths["rtx"],
            "rtx_backend_receipt_file_sha256": packaged_hashes["rtx"],
            "rtx_backend_receipt_root_sha256": final["rtx_backend_root_sha256"],
            "mac_backend_receipt": packaged_paths["mac"],
            "mac_backend_receipt_file_sha256": packaged_hashes["mac"],
            "mac_backend_receipt_root_sha256": final["mac_backend_root_sha256"],
            "fresh_process_count_per_backend": 4,
            "poison_bytes": [90, 165],
            "expected_output_bytes_packaged": False,
            "expected_output_ingress_rejected_by_bound_verifier": True,
        }
        if consumer.residual_enabled is not None:
            record.update(
                {
                    "residual_enabled": consumer.residual_enabled,
                    "family_receipt": source_record(family_path),
                    "family_receipt_root_sha256": BLOCK1_CONV2_FAMILY_ROOT_SHA256,
                }
            )
        records.append(record)
    return records, receipt_bytes


def predecessor_receipts(v2_manifest: dict[str, Any]) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    for consumer in v2_manifest["proven_consumers"]:
        for label in ("rtx", "mac"):
            relative = consumer[f"{label}_backend_receipt"]
            data = (V2_PACKAGE / relative).read_bytes()
            require(bytes_sha256(data) == consumer[f"{label}_backend_receipt_file_sha256"], f"{consumer['consumer_id']}/{label}: v2 receipt hash")
            result[relative] = data
    return result


def append_block1_state(
    prefix: bytes, prefix_mappings: list[dict[str, Any]], buffers: dict[int, bytes]
) -> tuple[bytes, list[dict[str, Any]]]:
    package = bytearray(prefix)
    mappings = copy.deepcopy(prefix_mappings)
    for consumer in BLOCK1_CONSUMERS:
        for role, token, logical_bytes, source_sha256 in (
            ("weight", consumer.weight_token, consumer.weight_bytes, consumer.weight_sha256),
            ("bias", consumer.bias_token, consumer.bias_bytes, consumer.bias_sha256),
        ):
            data = buffers[token]
            require(len(data) == logical_bytes and bytes_sha256(data) == source_sha256, f"token {token}: captured state bytes")
            offset = len(package)
            package.extend(data)
            mappings.append(
                {
                    "consumer_id": consumer.consumer_id,
                    "family": "custom-block1-convolution",
                    "trace_launch_token": f"launch-{consumer.launch:06d}",
                    "state_role": role,
                    "source_token": token,
                    "logical_bytes": logical_bytes,
                    "source_sha256": source_sha256,
                    "package_offset": offset,
                    "package_end_exclusive": len(package),
                }
            )
    return bytes(package), mappings


def validate_manifest_semantics(
    manifest: dict[str, Any], package: bytes, capture: dict[str, Any], inventory: dict[str, Any]
) -> None:
    require(manifest.get("schema") == SCHEMA and manifest.get("address_fields_recorded") is False, "v3 package identity")
    require_address_free(manifest, "v3 package manifest")
    require(
        manifest.get("package_format")
        == {
            "file": PACKAGE_FILE,
            "encoding": "opaque captured bytes",
            "layout": "predecessor-v2 prefix; then launch-ascending block1 weight before bias; no padding",
            "padding_bytes": 0,
        },
        "v3 package format",
    )
    v2_manifest_path = V2_PACKAGE / "package-manifest.json"
    v2_payload_path = V2_PACKAGE / v2.PACKAGE_FILE
    v2_manifest = v2.verify_package(V2_PACKAGE)
    inventory_path = REPOSITORY / "evidence/specs/fg-custom-conv-inventory-v1/inventory.json"
    family_path = REPOSITORY / "evidence/specs/fg-custom-block1-conv2-family-v1/family-verification.json"
    artifact_path = REPOSITORY / "evidence/specs/fg-custom-block1-conv2-family-v1/artifact-manifest.json"
    source = manifest.get("source_evidence", {})
    validate_source_record(source.get("predecessor_v2_manifest"), v2_manifest_path, "predecessor v2")
    validate_source_record(source.get("custom_convolution_inventory"), inventory_path, "custom inventory")
    validate_source_record(source.get("block1_conv2_family"), family_path, "block1 conv2 family")
    validate_source_record(source.get("block1_conv2_artifact_manifest"), artifact_path, "block1 conv2 artifacts")
    validate_source_record(source.get("builder_source"), Path(__file__).resolve(), "v3 builder")
    require(
        source.get("predecessor_v2_root_sha256") == v2_manifest["package_root_sha256"]
        and source.get("predecessor_v2_payload_sha256") == v2_manifest["included_state"]["package_sha256"]
        and source.get("runtime_sha256") == v2.v1.RUNTIME_SHA256
        and source.get("capture_point") == v2.v1.CAPTURE_POINT
        and source.get("capture_manifest_sha256") == v2.v1.CAPTURE_MANIFEST_SHA256
        and source.get("capture_fresh_process_count") == 2
        and source.get("capture_poison_bytes") == [90, 165]
        and source.get("capture_buffers_repeat_exact") is True
        and source.get("capture_bytes_fed_back_to_execution") is False
        and source.get("custom_convolution_inventory_root_sha256") == v2.INVENTORY_ROOT_SHA256
        and source.get("block1_conv2_family_root_sha256") == BLOCK1_CONV2_FAMILY_ROOT_SHA256,
        "v3 source identity",
    )
    require(package[: len(v2_payload_path.read_bytes())] == v2_payload_path.read_bytes(), "v2 payload prefix not preserved")
    capture_entries = v2.v1.validate_capture_manifest(capture)
    require(rooted(inventory) == v2.INVENTORY_ROOT_SHA256, "v3 inventory root")
    family = load_json(family_path)
    require(rooted(family) == BLOCK1_CONV2_FAMILY_ROOT_SHA256, "v3 block1 family root")

    consumers = manifest.get("proven_consumers")
    mappings = manifest.get("included_state", {}).get("mappings")
    remaining = manifest.get("unverified_state", {}).get("buffers")
    predecessor_count = len(v2_manifest["proven_consumers"])
    predecessor_mapping_count = len(v2_manifest["included_state"]["mappings"])
    require(isinstance(consumers, list) and consumers[:predecessor_count] == v2_manifest["proven_consumers"], "v2 consumer prefix drift")
    require(isinstance(mappings, list) and mappings[:predecessor_mapping_count] == v2_manifest["included_state"]["mappings"], "v2 mapping prefix drift")
    require([item.get("consumer_id") for item in consumers[predecessor_count:]] == [item.consumer_id for item in BLOCK1_CONSUMERS], "block1 consumer order")

    expected_new_state: list[tuple[Block1Consumer, str, int, int, str]] = []
    for consumer in BLOCK1_CONSUMERS:
        expected_new_state.extend(
            [
                (consumer, "weight", consumer.weight_token, consumer.weight_bytes, consumer.weight_sha256),
                (consumer, "bias", consumer.bias_token, consumer.bias_bytes, consumer.bias_sha256),
            ]
        )
    new_mappings = mappings[predecessor_mapping_count:]
    require(len(new_mappings) == len(expected_new_state), "block1 state mapping count")
    offset = len(v2_payload_path.read_bytes())
    for mapping, expected in zip(new_mappings, expected_new_state, strict=True):
        consumer, role, token, logical_bytes, source_sha256 = expected
        require(
            mapping.get("consumer_id") == consumer.consumer_id
            and mapping.get("family") == "custom-block1-convolution"
            and mapping.get("trace_launch_token") == f"launch-{consumer.launch:06d}"
            and mapping.get("state_role") == role
            and mapping.get("source_token") == token
            and mapping.get("logical_bytes") == logical_bytes
            and mapping.get("source_sha256") == source_sha256
            and mapping.get("package_offset") == offset
            and mapping.get("package_end_exclusive") == offset + logical_bytes,
            f"token {token}: block1 mapping",
        )
        require(bytes_sha256(package[offset : offset + logical_bytes]) == source_sha256, f"token {token}: package slice")
        offset += logical_bytes
    included = manifest["included_state"]
    require(
        included.get("buffer_count") == len(mappings)
        and included.get("logical_bytes") == len(package) == offset
        and included.get("package_sha256") == bytes_sha256(package),
        "v3 included state summary",
    )
    included_tokens = [item["source_token"] for item in mappings]
    expected_remaining = [token for token in v2.v1.EXPECTED_TOKENS if token not in set(included_tokens)]
    require(isinstance(remaining, list) and [item.get("source_token") for item in remaining] == expected_remaining, "v3 derived unverified partition")
    for item in remaining:
        source_entry = capture_entries[item["source_token"]]
        require(
            item.get("logical_bytes") == source_entry["logical_bytes"]
            and item.get("source_sha256") == source_entry["sha256"]
            and item.get("package_offset") is None
            and item.get("status") == "unverified_bottom_branch_candidate_excluded",
            f"token {item['source_token']}: bottom branch boundary",
        )
    unverified = manifest["unverified_state"]
    remaining_bytes = v2.v1.EXPECTED_CAPTURE_BYTES - len(package)
    require(
        unverified.get("buffer_count") == len(expected_remaining)
        and unverified.get("logical_bytes") == remaining_bytes
        and sum(item["logical_bytes"] for item in remaining) == remaining_bytes,
        "v3 unverified state summary",
    )
    claims = manifest.get("claims", {})
    require(
        claims.get("generic_convolution_launches_proven") == 17
        and claims.get("generic_state_buffers_proven") == 17
        and claims.get("custom_convolution_launches_proven") == 21
        and claims.get("custom_state_buffers_proven") == 42
        and claims.get("block1_convolution_launches_proven") == 10
        and claims.get("block1_state_buffers_proven") == 20
        and claims.get("bottom_branch_launches_proven") == 0
        and claims.get("bottom_branch_state_proven") is False
        and claims.get("all_custom_convolution_launches_proven") is False
        and claims.get("all_runtime_state_buffers_classified") is False
        and claims.get("whole_model_weights_recovered") is False
        and claims.get("connected_custom_group_proven") is False
        and claims.get("full_generated_frame_parity") is False
        and claims.get("netal_integration") is False
        and claims.get("performance_claim") is False,
        "v3 claim boundary",
    )
    require(
        manifest.get("execution_ingress")
        == {
            "contains_expected_output_bytes": False,
            "contains_oracle_output_bytes": False,
            "contains_activation_or_intermediate_buffers": False,
            "contains_unverified_state_buffers": False,
            "permitted_role": "proven_convolution_parameter_state_only",
        },
        "v3 ingress boundary",
    )
    provenance = manifest.get("execution_provenance", {})
    require(
        provenance.get("proven_consumer_count") == len(consumers)
        and provenance.get("generic_consumer_count") == 17
        and provenance.get("custom_consumer_count") == 21
        and provenance.get("per_consumer_backend_receipts_packaged") is True
        and provenance.get("backends") == ["rtx-cuda-driver", "mac-metal"]
        and provenance.get("fresh_process_count_per_consumer_per_backend") == 4
        and provenance.get("poison_bytes") == [90, 165]
        and provenance.get("two_process_two_poison_invariance_required") is True
        and provenance.get("input_and_model_immutability_required") is True
        and provenance.get("canary_integrity_required") is True
        and provenance.get("expected_output_fed_to_execution") is False,
        "v3 execution provenance",
    )


def validate_new_consumers_and_receipts(
    package_dir: Path, manifest: dict[str, Any], inventory_by_launch: dict[int, dict[str, Any]]
) -> None:
    v2_manifest = v2.verify_package(V2_PACKAGE)
    predecessor_count = len(v2_manifest["proven_consumers"])
    v2.validate_consumers_and_receipts(
        package_dir,
        {"proven_consumers": manifest["proven_consumers"][:predecessor_count]},
        inventory_by_launch,
    )
    records = manifest["proven_consumers"][predecessor_count:]
    family_root = REPOSITORY / "evidence/specs/fg-custom-block1-conv2-family-v1"
    for record, consumer in zip(records, BLOCK1_CONSUMERS, strict=True):
        paths = block1_paths(consumer)
        final = load_json(paths["final"])
        require(
            record.get("consumer_id") == consumer.consumer_id
            and record.get("family") == "custom-block1-convolution"
            and record.get("trace_launch_token") == f"launch-{consumer.launch:06d}"
            and record.get("function_name") == consumer.function_name
            and record.get("module_token") == consumer.module_token
            and record.get("module_sha256") == consumer.module_sha256
            and record.get("source_tokens") == [consumer.weight_token, consumer.bias_token]
            and record.get("state_roles") == ["weight", "bias"]
            and record.get("exact_receipt_root_sha256") == final.get("receipt_root_sha256")
            and record.get("rtx_backend_receipt_root_sha256") == final.get("rtx_backend_root_sha256")
            and record.get("mac_backend_receipt_root_sha256") == final.get("mac_backend_root_sha256")
            and record.get("fresh_process_count_per_backend") == 4
            and record.get("poison_bytes") == [90, 165]
            and record.get("expected_output_bytes_packaged") is False
            and record.get("expected_output_ingress_rejected_by_bound_verifier") is True,
            f"{consumer.consumer_id}: consumer binding",
        )
        validate_source_record(record.get("exact_receipt"), paths["final"], f"{consumer.consumer_id}: exact receipt")
        validate_source_record(record.get("artifact_manifest"), paths["artifacts"], f"{consumer.consumer_id}: artifacts")
        if consumer.residual_enabled is not None:
            require(
                record.get("residual_enabled") is consumer.residual_enabled
                and record.get("family_receipt_root_sha256") == BLOCK1_CONV2_FAMILY_ROOT_SHA256,
                f"{consumer.consumer_id}: family/residual binding",
            )
            validate_source_record(record.get("family_receipt"), family_root / "family-verification.json", f"{consumer.consumer_id}: family receipt")
        backend_values: dict[str, dict[str, Any]] = {}
        for label, backend in (("rtx", "rtx-cuda-driver"), ("mac", "mac-metal")):
            receipt_path = package_dir / record[f"{label}_backend_receipt"]
            require(file_sha256(receipt_path) == record[f"{label}_backend_receipt_file_sha256"], f"{consumer.consumer_id}/{label}: packaged receipt hash")
            value = load_json(receipt_path)
            validate_block1_backend(value, consumer, backend, record[f"{label}_backend_receipt_root_sha256"])
            backend_values[label] = value
        require(
            backend_values["rtx"].get("output_sha256")
            == backend_values["mac"].get("output_sha256")
            == final.get("output_sha256"),
            f"{consumer.consumer_id}: backend/final output mismatch",
        )
        inventory = inventory_by_launch[consumer.launch]
        require(record.get("source_tokens") == [inventory["weight_token"], inventory["bias_token"]], f"{consumer.consumer_id}: inventory tokens")


def negative_audit(
    base: dict[str, Any], package: bytes, capture: dict[str, Any], inventory: dict[str, Any]
) -> dict[str, Any]:
    cases: list[dict[str, str]] = []

    def reject(name: str, operation: Callable[[], None]) -> None:
        try:
            operation()
        except (PackageV3Error, v2.PackageV2Error, v2.v1.PackageError) as error:
            cases.append({"name": name, "result": "rejected", "reason": str(error)})
        else:
            raise PackageV3Error(f"negative audit failed to reject: {name}")

    def validate(mutated_manifest: dict[str, Any] | None = None, mutated_package: bytes | None = None, mutated_capture: dict[str, Any] | None = None, mutated_inventory: dict[str, Any] | None = None) -> None:
        validate_manifest_semantics(
            mutated_manifest if mutated_manifest is not None else base,
            mutated_package if mutated_package is not None else package,
            mutated_capture if mutated_capture is not None else capture,
            mutated_inventory if mutated_inventory is not None else inventory,
        )

    changed = bytearray(package); changed[len(changed) - 97] ^= 1
    reject("block1_payload_byte_flip", lambda: validate(mutated_package=bytes(changed)))
    reject("package_truncation", lambda: validate(mutated_package=package[:-1]))
    prefix_changed = bytearray(package); prefix_changed[17] ^= 1
    reject("v2_prefix_mutation", lambda: validate(mutated_package=bytes(prefix_changed)))
    first_new = len(base["included_state"]["mappings"]) - 20
    mutated = copy.deepcopy(base); mutated["included_state"]["mappings"][first_new]["package_offset"] += 1
    reject("block1_offset_gap", lambda: validate(mutated_manifest=mutated))
    mutated = copy.deepcopy(base); mutated["included_state"]["mappings"][first_new + 1]["state_role"] = "weight"
    reject("block1_bias_role_swap", lambda: validate(mutated_manifest=mutated))
    mutated = copy.deepcopy(base); del mutated["included_state"]["mappings"][first_new + 1]
    reject("block1_bias_omission", lambda: validate(mutated_manifest=mutated))
    mutated = copy.deepcopy(base); mutated["unverified_state"]["buffers"][0]["status"] = "verified"
    reject("bottom_state_promotion", lambda: validate(mutated_manifest=mutated))
    mutated = copy.deepcopy(base); mutated["claims"]["bottom_branch_state_proven"] = True
    reject("bottom_branch_overclaim", lambda: validate(mutated_manifest=mutated))
    mutated = copy.deepcopy(base); mutated["claims"]["whole_model_weights_recovered"] = True
    reject("whole_model_overclaim", lambda: validate(mutated_manifest=mutated))
    mutated = copy.deepcopy(base); mutated["claims"]["connected_custom_group_proven"] = True
    reject("connected_custom_group_overclaim", lambda: validate(mutated_manifest=mutated))
    mutated = copy.deepcopy(base); mutated["execution_ingress"]["contains_expected_output_bytes"] = True
    reject("expected_output_ingress", lambda: validate(mutated_manifest=mutated))
    mutated = copy.deepcopy(capture); mutated["capture_point"] = "post_create_fence_pre_evaluate"
    reject("wrong_capture_point", lambda: validate(mutated_capture=mutated))
    mutated = copy.deepcopy(inventory); mutated["launches"][15]["weight_token"] = 160
    reject("block1_inventory_token_mutation", lambda: validate(mutated_inventory=mutated))
    mutated = copy.deepcopy(base); mutated["proven_consumers"] = mutated["proven_consumers"][:-1]
    reject("block1_consumer_omission", lambda: validate(mutated_manifest=mutated))
    mutated = copy.deepcopy(base); mutated["volatile_pointer"] = "0x123456789abc"
    reject("address_injection", lambda: validate(mutated_manifest=mutated))
    require(len(cases) == 15 and all(case["result"] == "rejected" for case in cases), "v3 negative audit incomplete")
    result = {
        "schema": AUDIT_SCHEMA,
        "mutation_count": len(cases),
        "rejected_count": len(cases),
        "failed_to_reject": [],
        "cases": cases,
    }
    result["receipt_root_sha256"] = canonical_sha256(result)
    return result


def build_package(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    require(not output_dir.is_symlink(), "unsafe v3 output directory")
    generated = [output_dir / PACKAGE_FILE, output_dir / "capture-manifest.json", output_dir / "custom-conv-inventory.json", output_dir / "negative-audit.json", output_dir / "package-manifest.json", output_dir / "backend-receipts"]
    require(not any(path.exists() for path in generated), "refusing to overwrite immutable v3 artifacts")

    v2_manifest = v2.verify_package(V2_PACKAGE)
    capture, capture_entries, buffers = v2.v1.load_capture_pair(args.source_job)
    inventory_path = REPOSITORY / "evidence/specs/fg-custom-conv-inventory-v1/inventory.json"
    inventory, inventory_by_launch = v2.validate_inventory(inventory_path)
    block1_consumers, block1_receipts = validate_block1_evidence(inventory_by_launch, args.rtx_conv2, args.mac_conv2)
    consumers = copy.deepcopy(v2_manifest["proven_consumers"]) + block1_consumers
    receipts = {**predecessor_receipts(v2_manifest), **block1_receipts}
    v2_payload = (V2_PACKAGE / v2.PACKAGE_FILE).read_bytes()
    package, mappings = append_block1_state(v2_payload, v2_manifest["included_state"]["mappings"], buffers)
    included_tokens = {mapping["source_token"] for mapping in mappings}
    remaining = [
        {
            "source_token": token,
            "logical_bytes": entry["logical_bytes"],
            "source_sha256": entry["sha256"],
            "package_offset": None,
            "status": "unverified_bottom_branch_candidate_excluded",
        }
        for token, entry in capture_entries.items()
        if token not in included_tokens
    ]
    family_path = REPOSITORY / "evidence/specs/fg-custom-block1-conv2-family-v1/family-verification.json"
    artifact_path = REPOSITORY / "evidence/specs/fg-custom-block1-conv2-family-v1/artifact-manifest.json"
    base = {
        "schema": SCHEMA,
        "package_format": {
            "file": PACKAGE_FILE,
            "encoding": "opaque captured bytes",
            "layout": "predecessor-v2 prefix; then launch-ascending block1 weight before bias; no padding",
            "padding_bytes": 0,
        },
        "source_evidence": {
            "predecessor_v2_manifest": source_record(V2_PACKAGE / "package-manifest.json"),
            "predecessor_v2_root_sha256": v2_manifest["package_root_sha256"],
            "predecessor_v2_payload_sha256": v2_manifest["included_state"]["package_sha256"],
            "runtime_sha256": v2.v1.RUNTIME_SHA256,
            "capture_point": v2.v1.CAPTURE_POINT,
            "capture_manifest_file": "capture-manifest.json",
            "capture_manifest_sha256": v2.v1.CAPTURE_MANIFEST_SHA256,
            "capture_fresh_process_count": 2,
            "capture_poison_bytes": [90, 165],
            "capture_buffers_repeat_exact": True,
            "capture_bytes_fed_back_to_execution": False,
            "custom_convolution_inventory": source_record(inventory_path),
            "custom_convolution_inventory_file": "custom-conv-inventory.json",
            "custom_convolution_inventory_root_sha256": v2.INVENTORY_ROOT_SHA256,
            "block1_conv2_family": source_record(family_path),
            "block1_conv2_family_root_sha256": BLOCK1_CONV2_FAMILY_ROOT_SHA256,
            "block1_conv2_artifact_manifest": source_record(artifact_path),
            "builder_source": source_record(Path(__file__).resolve()),
        },
        "proven_consumers": consumers,
        "included_state": {
            "classification": "proven_generic_block0_and_block1_convolution_parameter_state",
            "buffer_count": len(mappings),
            "logical_bytes": len(package),
            "package_sha256": bytes_sha256(package),
            "mappings": mappings,
        },
        "unverified_state": {
            "classification": "bottom_branch_candidate_state_not_yet_selected_module_consumption_proven",
            "buffer_count": len(remaining),
            "logical_bytes": sum(item["logical_bytes"] for item in remaining),
            "buffers": remaining,
        },
        "execution_provenance": {
            "proven_consumer_count": len(consumers),
            "generic_consumer_count": 17,
            "custom_consumer_count": len(consumers) - 17,
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
            "permitted_role": "proven_convolution_parameter_state_only",
        },
        "claims": {
            "generic_convolution_launches_proven": 17,
            "generic_state_buffers_proven": 17,
            "custom_convolution_launches_proven": len(consumers) - 17,
            "custom_state_buffers_proven": len(mappings) - 17,
            "block1_convolution_launches_proven": len(BLOCK1_CONSUMERS),
            "block1_state_buffers_proven": 2 * len(BLOCK1_CONSUMERS),
            "bottom_branch_launches_proven": 0,
            "bottom_branch_state_proven": False,
            "all_custom_convolution_launches_proven": False,
            "all_runtime_state_buffers_classified": False,
            "whole_model_weights_recovered": False,
            "connected_custom_group_proven": False,
            "full_generated_frame_parity": False,
            "netal_integration": False,
            "performance_claim": False,
        },
        "address_fields_recorded": False,
    }
    validate_manifest_semantics(base, package, capture, inventory)
    audit = negative_audit(base, package, capture, inventory)
    audit_bytes = (json.dumps(audit, indent=2, sort_keys=True) + "\n").encode()
    base["mutation_testing"] = {
        "file": "negative-audit.json",
        "file_sha256": bytes_sha256(audit_bytes),
        "receipt_root_sha256": audit["receipt_root_sha256"],
        "mutation_count": audit["mutation_count"],
        "rejected_count": audit["rejected_count"],
    }
    base["package_root_sha256"] = canonical_sha256(base)
    require_address_free(base, "final v3 package")
    try:
        write_exclusive(output_dir / PACKAGE_FILE, package)
        capture_source = args.source_job / "eval0-a/runtime-default-buffers-post-evaluate-0.manifest.json"
        write_exclusive(output_dir / "capture-manifest.json", capture_source.read_bytes())
        write_exclusive(output_dir / "custom-conv-inventory.json", inventory_path.read_bytes())
        for relative, data in receipts.items():
            write_exclusive(output_dir / relative, data)
        write_exclusive(output_dir / "negative-audit.json", audit_bytes)
        write_json_exclusive(output_dir / "package-manifest.json", base)
    except Exception:
        for target in (output_dir / PACKAGE_FILE, output_dir / "capture-manifest.json", output_dir / "custom-conv-inventory.json", output_dir / "negative-audit.json", output_dir / "package-manifest.json"):
            if target.exists() and target.is_file():
                target.unlink()
        receipt_dir = output_dir / "backend-receipts"
        if receipt_dir.exists() and receipt_dir.is_dir():
            shutil.rmtree(receipt_dir)
        raise
    return base


def verify_package(package_dir: Path) -> dict[str, Any]:
    manifest = load_json(package_dir / "package-manifest.json")
    require(manifest.get("package_root_sha256") == rooted(manifest, "package_root_sha256"), "v3 package root invalid")
    payload_path = package_dir / PACKAGE_FILE
    require(payload_path.is_file() and not payload_path.is_symlink(), "v3 payload missing or unsafe")
    package = payload_path.read_bytes()
    capture_path = package_dir / "capture-manifest.json"
    require(file_sha256(capture_path) == v2.v1.CAPTURE_MANIFEST_SHA256, "v3 packaged capture manifest hash")
    capture = load_json(capture_path)
    inventory_path = package_dir / "custom-conv-inventory.json"
    require(file_sha256(inventory_path) == manifest["source_evidence"]["custom_convolution_inventory"]["file_sha256"], "v3 packaged inventory hash")
    inventory, inventory_by_launch = v2.validate_inventory(inventory_path)
    validate_manifest_semantics(manifest, package, capture, inventory)
    validate_new_consumers_and_receipts(package_dir, manifest, inventory_by_launch)
    audit_meta = manifest.get("mutation_testing", {})
    audit_path = package_dir / audit_meta.get("file", "")
    require(audit_path.is_file() and file_sha256(audit_path) == audit_meta.get("file_sha256"), "v3 negative audit file hash")
    audit = load_json(audit_path)
    require(
        audit.get("schema") == AUDIT_SCHEMA
        and rooted(audit) == audit.get("receipt_root_sha256") == audit_meta.get("receipt_root_sha256")
        and audit.get("mutation_count") == audit.get("rejected_count") == audit_meta.get("mutation_count") == 15
        and audit.get("failed_to_reject") == [],
        "v3 negative audit invalid",
    )
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--source-job", type=Path, default=REPOSITORY / "build/fg_rig/fg-default-readback-20260904T095705Z")
    build.add_argument("--rtx-conv2", type=Path, default=REPOSITORY / "build/fg_conv/fg-custom-block1-conv2-family-20260904T111149Z/retrieved")
    build.add_argument("--mac-conv2", type=Path, default=REPOSITORY / "build/fg_conv/macos-custom-block1-conv2-root-v2")
    build.add_argument("--output-dir", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--package-dir", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        manifest = build_package(args) if args.command == "build" else verify_package(args.package_dir.resolve())
        print(f"package_root_sha256={manifest['package_root_sha256']}")
        print(f"package_sha256={manifest['included_state']['package_sha256']}")
        print(f"proven_buffers={manifest['included_state']['buffer_count']}")
        print(f"unverified_buffers={manifest['unverified_state']['buffer_count']}")
        print("result=pass")
        return 0
    except (OSError, KeyError, TypeError, PackageV3Error, v2.PackageV2Error, v2.v1.PackageError) as error:
        print(f"package_v3_error={error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
