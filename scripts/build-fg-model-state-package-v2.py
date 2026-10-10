#!/usr/bin/env python3
"""Build and verify FG state package v2: generic plus three custom convolutions."""

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
V1_BUILDER = REPOSITORY / "scripts/build-fg-model-state-package.py"
V1_PACKAGE = REPOSITORY / "evidence/specs/fg-model-state-package-v1"
V2_PACKAGE = REPOSITORY / "evidence/specs/fg-model-state-package-v2"
SCHEMA = "neuralstudio.fg.model-state-package.v2"
AUDIT_SCHEMA = "neuralstudio.fg.model-state-package-negative-audit.v2"
PACKAGE_FILE = "selected-generic-plus-custom-conv-state.bin"
INVENTORY_ROOT_SHA256 = "93d869870d34327b1718ec37f418640936ac1d5829923218a61fc21344028f45"
CONV2_FAMILY_ROOT_SHA256 = "fefe047d76f8b0104773d3242ef5c5ab816f31e114c965e725c645e9316d0e67"


def load_v1() -> Any:
    spec = importlib.util.spec_from_file_location("fg_model_state_package_v1_dependency", V1_BUILDER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load v1 model-state package builder")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


v1 = load_v1()


@dataclass(frozen=True)
class CustomConsumer:
    consumer_id: str
    launch: int
    function_name: str
    module_token: int
    module_sha256: str
    spec_directory: str
    schema_stem: str
    selected_unit: str
    weight_token: int
    weight_bytes: int
    weight_sha256: str
    bias_token: int
    bias_bytes: int
    bias_sha256: str


CUSTOM_CONSUMERS = (
    CustomConsumer(
        "custom-conv-pre", 57, "custom_block0_convPre_kernel", 54,
        "a371abef8e41995064c25e171d0f2f9119a24ad4b671041be16c7b639a7c68a2",
        "fg-custom-conv-pre-v1", "custom-conv-pre",
        "evaluate-1/launch-000057/custom_block0_convPre_kernel/module-54",
        106, 9216, "14058e1781a3e362cd9f0106cf5c3fcd985ede67f1391ff3dfc73bceeceeae38",
        108, 64, "349bbe2fcd8fc31407bfe238f4889871f0d64dd593b4146b59eca837fcc9a77f",
    ),
    CustomConsumer(
        "custom-conv0-c8", 58, "custom_block0_conv0_c8_kernel", 55,
        "8b0501ef1e90ee5c0621138406982cb06822de06de930184ce40f7f4d6e1709e",
        "fg-custom-conv0-c8-v1", "custom-conv0-c8",
        "evaluate-1/launch-000058/custom_block0_conv0_c8_kernel/module-55",
        110, 18432, "be4a2cdb5d93c3b3ee3ff5ee8663d27017488664ca8682313740ff99aec4b829",
        112, 64, "097bd45ed222a90e9311fd89bbb8bca8db92760ec932380d3c44e29eb3870b69",
    ),
    CustomConsumer(
        "custom-conv1", 59, "custom_block0_conv1_kernel", 56,
        "c36bedc4fe275c9891dbf35fc79f3377e71f91e7150a2d6b2ae7eb7f0d0548f7",
        "fg-custom-conv1-v1", "custom-conv1",
        "evaluate-1/launch-000059/custom_block0_conv1_kernel/module-56",
        114, 36864, "c314eefc0c076af12dc246cbffbcf32fe78b50640b451567aa85832eff367092",
        116, 128, "9718b129b9bf59ee1a683c6001880cfb910a5e95e9b2338e4fcfc742d7f83be7",
    ),
)


@dataclass(frozen=True)
class Conv2Consumer:
    consumer_id: str
    launch: int
    weight_token: int
    bias_token: int
    weight_sha256: str
    bias_sha256: str
    residual_enabled: bool


CONV2_CONSUMERS = (
    Conv2Consumer("custom-conv2-l60", 60, 118, 120, "819fa5692e109843574dd0608312014a32abc7e7499784066142a4d82c486985", "ba63d29aecb3781452154a32c8e771471f49ff4cd1d7f56d06a1cd89081cd461", False),
    Conv2Consumer("custom-conv2-l61", 61, 122, 124, "5f4ac25a208ceb920b34fae48b7cee7107cd3e8da565e0a689a2cab93fe7ce93", "a7548519b77593645b6d5c46777597c0eba48255cd9f079f77d166da4c406bb5", True),
    Conv2Consumer("custom-conv2-l62", 62, 126, 128, "42a3ff7432377294d3c1fbdbe3c9876d13b989afb3952dd8ce964c531eae47a3", "f0dcd605c4a628cd940630048aeca237e2f563ec5219f265d93d382abef569ff", False),
    Conv2Consumer("custom-conv2-l63", 63, 130, 132, "672b9d32f4510335489894e12379bdc53a82ba27474c4484f6a58e1a18657af3", "591d8e9cb8e6c383b40636a814e714e676857deecfea6331792d84131edbf618", True),
    Conv2Consumer("custom-conv2-l64", 64, 134, 136, "54356c51efe3b2a4aa798275b73e1f16857d1976eaa99bfffe32c1d6f9905eae", "8f2b5996e111aa7d5d9711f0c58cea678de5974e3a733ae6a1a8bf1e0ec5840c", False),
    Conv2Consumer("custom-conv2-l65", 65, 138, 140, "7323eede30ef99e54156437d5f0430c266c1bf076018ad942afd48e7a62eca47", "7b48edd449f5442b4af26a7ac5d9e610aa0f087422347d7acfe6fa2d46b573d8", True),
    Conv2Consumer("custom-conv2-l66", 66, 142, 144, "971c7195720f720d7ce59564d9ee85259adfd0f66b2f2c426ca7f43c9290877c", "755ff7524a6855c81f50da913721a63f8fe39aa9f8106e577f14a0575a24633e", False),
    Conv2Consumer("custom-conv2-l67", 67, 146, 148, "145a29061230186864b92a934c6dcafa00ddede91381002ce7086e42c13b9490", "89e683bf9d2d108add777b4fb521d9d72940c66270cf96d1735943bf423bb93d", True),
)

CONV2_MODULE_TOKEN = 57
CONV2_MODULE_SHA256 = "d2b6ef4d2533614364bffab475f98a32f3c2cacbe02e0fa5825588dc5dfa3b4d"
CONV2_WEIGHT_BYTES = 73_728
CONV2_BIAS_BYTES = 128


class PackageV2Error(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PackageV2Error(message)


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
        raise PackageV2Error(f"invalid JSON: {path}: {error}") from error
    require(isinstance(value, dict), f"JSON is not an object: {path}")
    return value


def source_record(path: Path) -> dict[str, str]:
    return {"relative_path": str(path.relative_to(REPOSITORY)), "file_sha256": file_sha256(path)}


def require_address_free(value: Any, label: str) -> None:
    def visit(node: Any, location: str) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                if key not in ("address_fields_recorded", "addresses_recorded"):
                    lowered = key.lower()
                    require(
                        not any(part in lowered for part in ("pointer", "virtual_address", "device_address", "gpu_address")),
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


def validate_inventory(path: Path) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    inventory = load_json(path)
    require(
        inventory.get("schema") == "neuralstudio.fg.custom-conv-inventory.v1"
        and inventory.get("receipt_root_sha256") == INVENTORY_ROOT_SHA256
        and rooted(inventory) == INVENTORY_ROOT_SHA256
        and inventory.get("custom_convolution_launch_count") == 25
        and inventory.get("model_weight_bias_pair_count") == 25
        and inventory.get("all_model_pairs_post_evaluate_0") is True
        and inventory.get("all_modules_rehashed") is True
        and inventory.get("all_parameter_blocks_repeat_canonical") is True
        and inventory.get("addresses_recorded") is False
        and inventory.get("parity_claimed") is False
        and inventory.get("source", {}).get("default_manifest_sha256") == v1.CAPTURE_MANIFEST_SHA256,
        "custom convolution inventory drift",
    )
    launches = inventory.get("launches")
    require(isinstance(launches, list) and len(launches) == 25, "custom inventory launches incomplete")
    by_launch: dict[int, dict[str, Any]] = {}
    for item in launches:
        launch = item.get("launch")
        require(isinstance(launch, int) and launch not in by_launch, "invalid custom inventory launch")
        by_launch[launch] = item
    require_address_free(inventory, "custom convolution inventory")
    return inventory, by_launch


def custom_paths(consumer: CustomConsumer) -> dict[str, Path]:
    root = REPOSITORY / "evidence/specs" / consumer.spec_directory
    return {
        "verification": root / "verification.json",
        "rtx": root / "rtx-backend.json",
        "mac": root / "mac-backend.json",
        "artifacts": root / "artifact-manifest.json",
    }


def validate_custom_backend(
    value: dict[str, Any], consumer: CustomConsumer, backend: str, expected_root: str
) -> None:
    require(
        value.get("schema") == f"neuralstudio.fg.{consumer.schema_stem}-backend.v1"
        and value.get("backend") == backend
        and value.get("receipt_root_sha256") == expected_root
        and rooted(value) == expected_root,
        f"{consumer.consumer_id}/{backend}: backend identity",
    )
    require(
        value.get("selected_unit") == consumer.selected_unit
        and value.get("module_sha256") == consumer.module_sha256
        and value.get("weight_sha256") == consumer.weight_sha256
        and value.get("bias_sha256") == consumer.bias_sha256
        and value.get("model_manifest_sha256") == v1.CAPTURE_MANIFEST_SHA256,
        f"{consumer.consumer_id}/{backend}: state identity",
    )
    require(
        value.get("fresh_process_count") == 4
        and value.get("two_process_two_poison_invariant") is True
        and value.get("full_write_coverage_from_two_poison_invariance") is True
        and value.get("input_model_immutability") is True
        and value.get("canary_integrity") is True
        and value.get("expected_output_fed_to_execution") is False
        and value.get("output_mutation_rejected") is True
        and value.get("address_fields_recorded") is False
        and value.get("result") == "pass",
        f"{consumer.consumer_id}/{backend}: execution gates",
    )
    require_address_free(value, f"{consumer.consumer_id}/{backend}")


def validate_custom_evidence(
    inventory_by_launch: dict[int, dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    consumers: list[dict[str, Any]] = []
    packaged_receipts: dict[str, bytes] = {}
    for consumer in CUSTOM_CONSUMERS:
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
        paths = custom_paths(consumer)
        final = load_json(paths["verification"])
        require(
            final.get("schema") == f"neuralstudio.fg.{consumer.schema_stem}-verification.v1"
            and rooted(final) == final.get("receipt_root_sha256")
            and final.get("selected_unit") == consumer.selected_unit
            and final.get("module_sha256") == consumer.module_sha256
            and final.get("weight_sha256") == consumer.weight_sha256
            and final.get("bias_sha256") == consumer.bias_sha256
            and final.get("rtx_mac_differing_bytes") == 0
            and final.get("rtx_mac_exact") is True
            and final.get("result") == "pass",
            f"{consumer.consumer_id}: final exact receipt",
        )
        rtx = load_json(paths["rtx"])
        mac = load_json(paths["mac"])
        validate_custom_backend(rtx, consumer, "rtx-cuda-driver", final["rtx_backend_root_sha256"])
        validate_custom_backend(mac, consumer, "mac-metal", final["mac_backend_root_sha256"])
        require(rtx.get("output_sha256") == mac.get("output_sha256") == final.get("output_sha256"), f"{consumer.consumer_id}: RTX/Mac output identity")
        receipt_paths: dict[str, str] = {}
        receipt_hashes: dict[str, str] = {}
        for label, value in (("rtx", rtx), ("mac", mac)):
            relative = f"backend-receipts/{consumer.consumer_id}-{label}.json"
            data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
            packaged_receipts[relative] = data
            receipt_paths[label] = relative
            receipt_hashes[label] = bytes_sha256(data)
        consumers.append(
            {
                "consumer_id": consumer.consumer_id,
                "family": "custom-convolution",
                "trace_launch_token": f"launch-{consumer.launch:06d}",
                "selected_unit": consumer.selected_unit,
                "function_name": consumer.function_name,
                "module_token": consumer.module_token,
                "module_sha256": consumer.module_sha256,
                "source_tokens": [consumer.weight_token, consumer.bias_token],
                "state_roles": ["weight", "bias"],
                "exact_receipt": source_record(paths["verification"]),
                "exact_receipt_root_sha256": final["receipt_root_sha256"],
                "artifact_manifest": source_record(paths["artifacts"]),
                "rtx_backend_receipt": receipt_paths["rtx"],
                "rtx_backend_receipt_file_sha256": receipt_hashes["rtx"],
                "rtx_backend_receipt_root_sha256": final["rtx_backend_root_sha256"],
                "mac_backend_receipt": receipt_paths["mac"],
                "mac_backend_receipt_file_sha256": receipt_hashes["mac"],
                "mac_backend_receipt_root_sha256": final["mac_backend_root_sha256"],
                "fresh_process_count_per_backend": 4,
                "poison_bytes": [90, 165],
                "expected_output_bytes_packaged": False,
            }
        )
    return consumers, packaged_receipts


def validate_conv2_backend(
    value: dict[str, Any], consumer: Conv2Consumer, backend: str, expected_root: str
) -> None:
    layer_id = f"l{consumer.launch}"
    require(
        value.get("schema") == "neuralstudio.fg.custom-conv2-backend.v1"
        and value.get("backend") == backend
        and value.get("receipt_root_sha256") == expected_root
        and rooted(value) == expected_root,
        f"{consumer.consumer_id}/{backend}: backend identity",
    )
    require(
        value.get("layer_id") == layer_id
        and value.get("trace_launch_token") == f"launch-{consumer.launch:06d}"
        and value.get("module_sha256") == CONV2_MODULE_SHA256
        and value.get("weight_sha256") == consumer.weight_sha256
        and value.get("bias_sha256") == consumer.bias_sha256
        and value.get("residual_enabled") is consumer.residual_enabled,
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
    require_address_free(value, f"{consumer.consumer_id}/{backend}")


def validate_conv2_evidence(
    inventory_by_launch: dict[int, dict[str, Any]], rtx_root: Path, mac_root: Path
) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    family_root = REPOSITORY / "evidence/specs/fg-custom-conv2-family-v1"
    family_path = family_root / "family-verification.json"
    family = load_json(family_path)
    require(
        family.get("schema") == "neuralstudio.fg.custom-conv2-family.v1"
        and family.get("receipt_root_sha256") == CONV2_FAMILY_ROOT_SHA256
        and rooted(family) == CONV2_FAMILY_ROOT_SHA256
        and family.get("layer_count") == len(CONV2_CONSUMERS)
        and family.get("every_launch_executed_independently") is True
        and family.get("every_launch_rtx_mac_exact") is True
        and family.get("residual_off_and_on_validated") is True
        and family.get("result") == "pass",
        "custom conv2 family receipt drift",
    )
    family_members = {item["layer_id"]: item for item in family["layers"]}
    artifact_path = family_root / "artifact-manifest.json"
    artifact = load_json(artifact_path)
    require(
        artifact.get("schema") == "neuralstudio.fg.custom-conv2-artifacts.v1"
        and artifact.get("address_fields_recorded") is False,
        "custom conv2 artifact manifest drift",
    )
    consumers: list[dict[str, Any]] = []
    receipt_bytes: dict[str, bytes] = {}
    for consumer in CONV2_CONSUMERS:
        layer_id = f"l{consumer.launch}"
        inventory = inventory_by_launch.get(consumer.launch, {})
        require(
            inventory.get("launch_token") == f"launch-{consumer.launch:06d}"
            and inventory.get("function_name") == "custom_block0_conv2_kernel"
            and inventory.get("module_token") == CONV2_MODULE_TOKEN
            and inventory.get("module_sha256") == CONV2_MODULE_SHA256
            and inventory.get("weight_token") == consumer.weight_token
            and inventory.get("weight_bytes") == CONV2_WEIGHT_BYTES
            and inventory.get("weight_sha256") == consumer.weight_sha256
            and inventory.get("bias_token") == consumer.bias_token
            and inventory.get("bias_bytes") == CONV2_BIAS_BYTES
            and inventory.get("bias_sha256") == consumer.bias_sha256,
            f"{consumer.consumer_id}: inventory binding",
        )
        final_path = family_root / "layers" / f"{layer_id}-verification.json"
        final = load_json(final_path)
        member = family_members.get(layer_id, {})
        require(
            final.get("schema") == "neuralstudio.fg.custom-conv2-verification.v1"
            and rooted(final) == final.get("receipt_root_sha256")
            and final.get("receipt_root_sha256") == member.get("receipt_root_sha256")
            and final.get("layer_id") == layer_id
            and final.get("module_sha256") == CONV2_MODULE_SHA256
            and final.get("weight_sha256") == consumer.weight_sha256
            and final.get("bias_sha256") == consumer.bias_sha256
            and final.get("residual_enabled") is consumer.residual_enabled
            and final.get("rtx_mac_differing_bytes") == 0
            and final.get("rtx_mac_exact") is True
            and final.get("result") == "pass",
            f"{consumer.consumer_id}: final exact receipt",
        )
        rtx_path = rtx_root / layer_id / "backend-receipt.json"
        mac_path = mac_root / layer_id / "backend-receipt.json"
        rtx = load_json(rtx_path)
        mac = load_json(mac_path)
        validate_conv2_backend(rtx, consumer, "rtx-cuda-driver", final["rtx_backend_root_sha256"])
        validate_conv2_backend(mac, consumer, "mac-metal", final["mac_backend_root_sha256"])
        require(rtx.get("output_sha256") == mac.get("output_sha256") == final.get("output_sha256"), f"{consumer.consumer_id}: RTX/Mac output identity")
        packaged: dict[str, str] = {}
        hashes: dict[str, str] = {}
        for label, value in (("rtx", rtx), ("mac", mac)):
            relative = f"backend-receipts/{consumer.consumer_id}-{label}.json"
            data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
            receipt_bytes[relative] = data
            packaged[label] = relative
            hashes[label] = bytes_sha256(data)
        consumers.append(
            {
                "consumer_id": consumer.consumer_id,
                "family": "custom-convolution-conv2",
                "trace_launch_token": f"launch-{consumer.launch:06d}",
                "selected_unit": f"evaluate-1/launch-{consumer.launch:06d}/custom_block0_conv2_kernel/module-57",
                "function_name": "custom_block0_conv2_kernel",
                "module_token": CONV2_MODULE_TOKEN,
                "module_sha256": CONV2_MODULE_SHA256,
                "source_tokens": [consumer.weight_token, consumer.bias_token],
                "state_roles": ["weight", "bias"],
                "residual_enabled": consumer.residual_enabled,
                "exact_receipt": source_record(final_path),
                "exact_receipt_root_sha256": final["receipt_root_sha256"],
                "family_receipt": source_record(family_path),
                "family_receipt_root_sha256": CONV2_FAMILY_ROOT_SHA256,
                "artifact_manifest": source_record(artifact_path),
                "rtx_backend_receipt": packaged["rtx"],
                "rtx_backend_receipt_file_sha256": hashes["rtx"],
                "rtx_backend_receipt_root_sha256": final["rtx_backend_root_sha256"],
                "mac_backend_receipt": packaged["mac"],
                "mac_backend_receipt_file_sha256": hashes["mac"],
                "mac_backend_receipt_root_sha256": final["mac_backend_root_sha256"],
                "fresh_process_count_per_backend": 4,
                "poison_bytes": [90, 165],
                "expected_output_bytes_packaged": False,
                "expected_output_ingress_rejected_by_bound_verifier": True,
            }
        )
    return consumers, receipt_bytes


def generic_consumers_and_receipts(v1_manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    consumers: list[dict[str, Any]] = []
    receipt_bytes: dict[str, bytes] = {}
    for layer, mapping in zip(v1.LAYERS, v1_manifest["included_state"]["mappings"], strict=True):
        record = {
            "consumer_id": layer.layer_id,
            "family": "generic-convolution",
            "trace_launch_token": f"launch-{layer.launch:06d}",
            "selected_unit": f"evaluate-1/launch-{layer.launch:06d}/generic-convolution",
            "function_name": "k_conv_fp16_nhwc",
            "module_token": layer.module_token,
            "module_sha256": layer.module_sha256,
            "source_tokens": [layer.source_token],
            "state_roles": ["opaque-combined-parameter-state"],
            "exact_receipt": {
                "relative_path": mapping["exact_receipt_relative_path"],
                "file_sha256": mapping["exact_receipt_file_sha256"],
            },
            "exact_receipt_root_sha256": mapping["exact_receipt_root_sha256"],
            "rtx_backend_receipt": f"backend-receipts/{layer.layer_id}-rtx.json",
            "rtx_backend_receipt_file_sha256": mapping["rtx_backend_receipt_file_sha256"],
            "rtx_backend_receipt_root_sha256": mapping["rtx_backend_receipt_root_sha256"],
            "mac_backend_receipt": f"backend-receipts/{layer.layer_id}-mac.json",
            "mac_backend_receipt_file_sha256": mapping["mac_backend_receipt_file_sha256"],
            "mac_backend_receipt_root_sha256": mapping["mac_backend_receipt_root_sha256"],
            "fresh_process_count_per_backend": 4,
            "poison_bytes": [90, 165],
            "expected_output_bytes_packaged": False,
        }
        for label in ("rtx", "mac"):
            source = V1_PACKAGE / mapping[f"{label}_backend_receipt"]
            data = source.read_bytes()
            require(bytes_sha256(data) == mapping[f"{label}_backend_receipt_file_sha256"], f"{layer.layer_id}/{label}: v1 backend receipt hash")
            receipt_bytes[record[f"{label}_backend_receipt"]] = data
        consumers.append(record)
    return consumers, receipt_bytes


def build_mappings(
    buffers: dict[int, bytes], consumers: list[dict[str, Any]]
) -> tuple[bytes, list[dict[str, Any]]]:
    package = bytearray()
    mappings: list[dict[str, Any]] = []
    custom_by_id = {consumer.consumer_id: consumer for consumer in CUSTOM_CONSUMERS}
    conv2_by_id = {consumer.consumer_id: consumer for consumer in CONV2_CONSUMERS}
    generic_by_id = {layer.layer_id: layer for layer in v1.LAYERS}
    for consumer in consumers:
        consumer_id = consumer["consumer_id"]
        if consumer["family"] == "generic-convolution":
            layer = generic_by_id[consumer_id]
            state = ((layer.source_token, layer.logical_bytes, layer.source_sha256, "opaque-combined-parameter-state"),)
        elif consumer["family"] == "custom-convolution":
            custom = custom_by_id[consumer_id]
            state = (
                (custom.weight_token, custom.weight_bytes, custom.weight_sha256, "weight"),
                (custom.bias_token, custom.bias_bytes, custom.bias_sha256, "bias"),
            )
        else:
            custom = conv2_by_id[consumer_id]
            state = (
                (custom.weight_token, CONV2_WEIGHT_BYTES, custom.weight_sha256, "weight"),
                (custom.bias_token, CONV2_BIAS_BYTES, custom.bias_sha256, "bias"),
            )
        for token, logical_bytes, source_sha256, role in state:
            data = buffers[token]
            require(len(data) == logical_bytes and bytes_sha256(data) == source_sha256, f"token {token}: source bytes")
            offset = len(package)
            package.extend(data)
            mappings.append(
                {
                    "consumer_id": consumer_id,
                    "family": consumer["family"],
                    "trace_launch_token": consumer["trace_launch_token"],
                    "state_role": role,
                    "source_token": token,
                    "logical_bytes": logical_bytes,
                    "source_sha256": source_sha256,
                    "package_offset": offset,
                    "package_end_exclusive": len(package),
                }
            )
    return bytes(package), mappings


def validate_source_record(record: Any, path: Path, label: str) -> None:
    require(
        isinstance(record, dict)
        and record.get("relative_path") == str(path.relative_to(REPOSITORY))
        and record.get("file_sha256") == file_sha256(path),
        f"{label}: source file drift",
    )


def validate_manifest_semantics(
    manifest: dict[str, Any], package: bytes, capture: dict[str, Any], inventory: dict[str, Any]
) -> None:
    require(manifest.get("schema") == SCHEMA and manifest.get("address_fields_recorded") is False, "v2 package identity")
    require_address_free(manifest, "v2 package manifest")
    require(
        manifest.get("package_format")
        == {
            "file": PACKAGE_FILE,
            "encoding": "opaque captured bytes",
            "layout": "launch-ascending; custom weight before bias; no padding",
            "padding_bytes": 0,
        },
        "v2 package format",
    )
    source = manifest.get("source_evidence", {})
    v1_manifest_path = V1_PACKAGE / "package-manifest.json"
    v1_payload_path = V1_PACKAGE / v1.PACKAGE_FILE
    inventory_path = REPOSITORY / "evidence/specs/fg-custom-conv-inventory-v1/inventory.json"
    conv2_family_path = REPOSITORY / "evidence/specs/fg-custom-conv2-family-v1/family-verification.json"
    conv2_artifact_path = REPOSITORY / "evidence/specs/fg-custom-conv2-family-v1/artifact-manifest.json"
    validate_source_record(source.get("predecessor_v1_manifest"), v1_manifest_path, "predecessor v1")
    validate_source_record(source.get("custom_convolution_inventory"), inventory_path, "custom inventory")
    validate_source_record(source.get("custom_conv2_family"), conv2_family_path, "custom conv2 family")
    validate_source_record(source.get("custom_conv2_artifact_manifest"), conv2_artifact_path, "custom conv2 artifacts")
    validate_source_record(source.get("builder_source"), Path(__file__).resolve(), "v2 builder")
    require(
        source.get("predecessor_v1_root_sha256") == v1.verify_package(V1_PACKAGE)["package_root_sha256"]
        and source.get("predecessor_v1_payload_sha256") == file_sha256(v1_payload_path)
        and source.get("runtime_sha256") == v1.RUNTIME_SHA256
        and source.get("capture_point") == v1.CAPTURE_POINT
        and source.get("capture_manifest_sha256") == v1.CAPTURE_MANIFEST_SHA256
        and source.get("capture_fresh_process_count") == 2
        and source.get("capture_poison_bytes") == [90, 165]
        and source.get("capture_buffers_repeat_exact") is True
        and source.get("capture_bytes_fed_back_to_execution") is False
        and source.get("custom_convolution_inventory_root_sha256") == INVENTORY_ROOT_SHA256,
        "v2 source evidence identity",
    )
    conv2_family = load_json(conv2_family_path)
    require(
        source.get("custom_conv2_family_root_sha256") == CONV2_FAMILY_ROOT_SHA256
        and conv2_family.get("receipt_root_sha256") == CONV2_FAMILY_ROOT_SHA256
        and rooted(conv2_family) == CONV2_FAMILY_ROOT_SHA256,
        "custom conv2 family source binding",
    )
    require(package[: v1.EXPECTED_INCLUDED_BYTES] == v1_payload_path.read_bytes(), "v1 payload prefix not preserved")
    capture_entries = v1.validate_capture_manifest(capture)
    require(rooted(inventory) == INVENTORY_ROOT_SHA256, "packaged inventory root")

    consumers = manifest.get("proven_consumers")
    mappings = manifest.get("included_state", {}).get("mappings")
    remaining = manifest.get("unverified_state", {}).get("buffers")
    require(isinstance(consumers, list) and len(consumers) == len(v1.LAYERS) + len(CUSTOM_CONSUMERS) + len(CONV2_CONSUMERS), "proven consumer count")
    require(isinstance(mappings, list), "included mappings missing")
    require(isinstance(remaining, list), "unverified mappings missing")
    expected_consumer_ids = [layer.layer_id for layer in v1.LAYERS] + [item.consumer_id for item in CUSTOM_CONSUMERS] + [item.consumer_id for item in CONV2_CONSUMERS]
    require([item.get("consumer_id") for item in consumers] == expected_consumer_ids, "consumer order/inventory")

    expected_state: list[tuple[str, str, str, int, int, str]] = []
    for layer in v1.LAYERS:
        expected_state.append((layer.layer_id, "generic-convolution", "opaque-combined-parameter-state", layer.source_token, layer.logical_bytes, layer.source_sha256))
    for custom in CUSTOM_CONSUMERS:
        expected_state.extend(
            [
                (custom.consumer_id, "custom-convolution", "weight", custom.weight_token, custom.weight_bytes, custom.weight_sha256),
                (custom.consumer_id, "custom-convolution", "bias", custom.bias_token, custom.bias_bytes, custom.bias_sha256),
            ]
        )
    for custom in CONV2_CONSUMERS:
        expected_state.extend(
            [
                (custom.consumer_id, "custom-convolution-conv2", "weight", custom.weight_token, CONV2_WEIGHT_BYTES, custom.weight_sha256),
                (custom.consumer_id, "custom-convolution-conv2", "bias", custom.bias_token, CONV2_BIAS_BYTES, custom.bias_sha256),
            ]
        )
    require(len(mappings) == len(expected_state), "included mapping count")
    offset = 0
    for mapping, expected in zip(mappings, expected_state, strict=True):
        consumer_id, family, role, token, logical_bytes, source_sha256 = expected
        require(
            mapping.get("consumer_id") == consumer_id
            and mapping.get("family") == family
            and mapping.get("state_role") == role
            and mapping.get("source_token") == token
            and mapping.get("logical_bytes") == logical_bytes
            and mapping.get("source_sha256") == source_sha256
            and mapping.get("package_offset") == offset
            and mapping.get("package_end_exclusive") == offset + logical_bytes,
            f"token {token}: included mapping",
        )
        require(bytes_sha256(package[offset : offset + logical_bytes]) == source_sha256, f"token {token}: package slice")
        offset += logical_bytes
    included = manifest["included_state"]
    require(
        included.get("buffer_count") == len(expected_state)
        and included.get("logical_bytes") == len(package) == offset
        and included.get("package_sha256") == bytes_sha256(package),
        "included state summary",
    )
    included_tokens = [item[3] for item in expected_state]
    expected_remaining_tokens = [token for token in v1.EXPECTED_TOKENS if token not in set(included_tokens)]
    require(len(remaining) == len(expected_remaining_tokens), "derived unverified count")
    require([item.get("source_token") for item in remaining] == expected_remaining_tokens, "unverified token partition")
    for item in remaining:
        source_entry = capture_entries[item["source_token"]]
        require(
            item.get("logical_bytes") == source_entry["logical_bytes"]
            and item.get("source_sha256") == source_entry["sha256"]
            and item.get("package_offset") is None
            and item.get("status") == "unverified_candidate_excluded",
            f"token {item['source_token']}: unverified boundary",
        )
    unverified = manifest["unverified_state"]
    expected_unverified_bytes = v1.EXPECTED_CAPTURE_BYTES - len(package)
    require(
        unverified.get("buffer_count") == len(expected_remaining_tokens)
        and unverified.get("logical_bytes") == expected_unverified_bytes
        and sum(item["logical_bytes"] for item in remaining) == expected_unverified_bytes,
        "unverified state summary",
    )
    claims = manifest.get("claims", {})
    require(
        claims.get("generic_convolution_launches_proven") == 17
        and claims.get("generic_state_buffers_proven") == 17
        and claims.get("custom_convolution_launches_proven") == 11
        and claims.get("custom_state_buffers_proven") == 22
        and claims.get("custom_conv2_launches_proven") == 8
        and claims.get("custom_conv2_state_buffers_proven") == 16
        and claims.get("all_custom_convolution_launches_proven") is False
        and claims.get("all_runtime_state_buffers_classified") is False
        and claims.get("whole_model_weights_recovered") is False
        and claims.get("connected_multi_kernel_group_proven_by_this_package") is False
        and claims.get("full_generated_frame_parity") is False
        and claims.get("netal_integration") is False
        and claims.get("performance_claim") is False,
        "v2 claim boundary",
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
        "v2 ingress boundary",
    )
    provenance = manifest.get("execution_provenance", {})
    require(
        provenance.get("proven_consumer_count") == len(expected_consumer_ids)
        and provenance.get("generic_consumer_count") == len(v1.LAYERS)
        and provenance.get("custom_consumer_count") == len(CUSTOM_CONSUMERS) + len(CONV2_CONSUMERS)
        and provenance.get("per_consumer_backend_receipts_packaged") is True
        and provenance.get("backends") == ["rtx-cuda-driver", "mac-metal"]
        and provenance.get("fresh_process_count_per_consumer_per_backend") == 4
        and provenance.get("poison_bytes") == [90, 165]
        and provenance.get("two_process_two_poison_invariance_required") is True
        and provenance.get("input_and_model_immutability_required") is True
        and provenance.get("canary_integrity_required") is True
        and provenance.get("expected_output_fed_to_execution") is False,
        "v2 execution provenance summary",
    )


def validate_consumers_and_receipts(
    package_dir: Path, manifest: dict[str, Any], inventory_by_launch: dict[int, dict[str, Any]]
) -> None:
    v1_manifest = v1.verify_package(V1_PACKAGE)
    generic_by_id = {layer.layer_id: (layer, mapping) for layer, mapping in zip(v1.LAYERS, v1_manifest["included_state"]["mappings"], strict=True)}
    custom_by_id = {item.consumer_id: item for item in CUSTOM_CONSUMERS}
    conv2_by_id = {item.consumer_id: item for item in CONV2_CONSUMERS}
    for record in manifest["proven_consumers"]:
        consumer_id = record["consumer_id"]
        final: dict[str, Any]
        if record["family"] == "generic-convolution":
            layer, predecessor = generic_by_id[consumer_id]
            final_path = REPOSITORY / predecessor["exact_receipt_relative_path"]
            final = load_json(final_path)
            require(
                record.get("trace_launch_token") == f"launch-{layer.launch:06d}"
                and record.get("module_token") == layer.module_token
                and record.get("module_sha256") == layer.module_sha256
                and record.get("source_tokens") == [layer.source_token]
                and record.get("state_roles") == ["opaque-combined-parameter-state"]
                and record.get("exact_receipt")
                == {
                    "relative_path": predecessor["exact_receipt_relative_path"],
                    "file_sha256": predecessor["exact_receipt_file_sha256"],
                }
                and record.get("exact_receipt_root_sha256") == predecessor["exact_receipt_root_sha256"]
                and record.get("rtx_backend_receipt_root_sha256") == predecessor["rtx_backend_receipt_root_sha256"]
                and record.get("mac_backend_receipt_root_sha256") == predecessor["mac_backend_receipt_root_sha256"],
                f"{consumer_id}: generic predecessor binding",
            )
        elif record["family"] == "custom-convolution":
            consumer = custom_by_id[consumer_id]
            paths = custom_paths(consumer)
            final = load_json(paths["verification"])
            require(
                record.get("trace_launch_token") == f"launch-{consumer.launch:06d}"
                and record.get("selected_unit") == consumer.selected_unit
                and record.get("function_name") == consumer.function_name
                and record.get("module_token") == consumer.module_token
                and record.get("module_sha256") == consumer.module_sha256
                and record.get("source_tokens") == [consumer.weight_token, consumer.bias_token]
                and record.get("state_roles") == ["weight", "bias"]
                and record.get("exact_receipt_root_sha256") == final.get("receipt_root_sha256")
                and record.get("rtx_backend_receipt_root_sha256") == final.get("rtx_backend_root_sha256")
                and record.get("mac_backend_receipt_root_sha256") == final.get("mac_backend_root_sha256"),
                f"{consumer_id}: custom receipt binding",
            )
        else:
            consumer = conv2_by_id[consumer_id]
            family_root = REPOSITORY / "evidence/specs/fg-custom-conv2-family-v1"
            final_path = family_root / "layers" / f"l{consumer.launch}-verification.json"
            final = load_json(final_path)
            require(
                record.get("trace_launch_token") == f"launch-{consumer.launch:06d}"
                and record.get("function_name") == "custom_block0_conv2_kernel"
                and record.get("module_token") == CONV2_MODULE_TOKEN
                and record.get("module_sha256") == CONV2_MODULE_SHA256
                and record.get("source_tokens") == [consumer.weight_token, consumer.bias_token]
                and record.get("state_roles") == ["weight", "bias"]
                and record.get("residual_enabled") is consumer.residual_enabled
                and record.get("exact_receipt_root_sha256") == final.get("receipt_root_sha256")
                and record.get("rtx_backend_receipt_root_sha256") == final.get("rtx_backend_root_sha256")
                and record.get("mac_backend_receipt_root_sha256") == final.get("mac_backend_root_sha256")
                and record.get("family_receipt_root_sha256") == CONV2_FAMILY_ROOT_SHA256,
                f"{consumer_id}: conv2 receipt binding",
            )
        require(
            record.get("fresh_process_count_per_backend") == 4
            and record.get("poison_bytes") == [90, 165]
            and record.get("expected_output_bytes_packaged") is False,
            f"{consumer_id}: packaged execution gates",
        )
        backend_values: dict[str, dict[str, Any]] = {}
        for label, backend in (("rtx", "rtx-cuda-driver"), ("mac", "mac-metal")):
            path = package_dir / record[f"{label}_backend_receipt"]
            require(file_sha256(path) == record[f"{label}_backend_receipt_file_sha256"], f"{consumer_id}/{label}: packaged receipt hash")
            value = load_json(path)
            if record["family"] == "generic-convolution":
                layer, mapping = generic_by_id[consumer_id]
                v1.validate_backend_receipt(value, layer, backend, mapping[f"{label}_backend_receipt_root_sha256"])
            elif record["family"] == "custom-convolution":
                consumer = custom_by_id[consumer_id]
                validate_custom_backend(value, consumer, backend, record[f"{label}_backend_receipt_root_sha256"])
            else:
                consumer = conv2_by_id[consumer_id]
                validate_conv2_backend(value, consumer, backend, record[f"{label}_backend_receipt_root_sha256"])
            backend_values[label] = value
        require(
            backend_values["rtx"].get("output_sha256")
            == backend_values["mac"].get("output_sha256")
            == final.get("output_sha256"),
            f"{consumer_id}: backend/final output mismatch",
        )
        if record["family"] == "custom-convolution":
            consumer = custom_by_id[consumer_id]
            paths = custom_paths(consumer)
            final = load_json(paths["verification"])
            validate_source_record(record.get("exact_receipt"), paths["verification"], f"{consumer_id}: exact receipt")
            validate_source_record(record.get("artifact_manifest"), paths["artifacts"], f"{consumer_id}: artifact manifest")
            require(record.get("exact_receipt_root_sha256") == final.get("receipt_root_sha256") and rooted(final) == final.get("receipt_root_sha256"), f"{consumer_id}: exact receipt root")
            inventory = inventory_by_launch[consumer.launch]
            require(record.get("source_tokens") == [inventory["weight_token"], inventory["bias_token"]], f"{consumer_id}: inventory source tokens")
        elif record["family"] == "custom-convolution-conv2":
            consumer = conv2_by_id[consumer_id]
            family_root = REPOSITORY / "evidence/specs/fg-custom-conv2-family-v1"
            final_path = family_root / "layers" / f"l{consumer.launch}-verification.json"
            family_path = family_root / "family-verification.json"
            artifact_path = family_root / "artifact-manifest.json"
            final = load_json(final_path)
            validate_source_record(record.get("exact_receipt"), final_path, f"{consumer_id}: exact receipt")
            validate_source_record(record.get("family_receipt"), family_path, f"{consumer_id}: family receipt")
            validate_source_record(record.get("artifact_manifest"), artifact_path, f"{consumer_id}: artifact manifest")
            require(
                record.get("exact_receipt_root_sha256") == final.get("receipt_root_sha256")
                and rooted(final) == final.get("receipt_root_sha256")
                and record.get("family_receipt_root_sha256") == CONV2_FAMILY_ROOT_SHA256,
                f"{consumer_id}: exact/family receipt roots",
            )
            inventory = inventory_by_launch[consumer.launch]
            require(record.get("source_tokens") == [inventory["weight_token"], inventory["bias_token"]], f"{consumer_id}: inventory source tokens")


def negative_audit(
    base: dict[str, Any], package: bytes, capture: dict[str, Any], inventory: dict[str, Any]
) -> dict[str, Any]:
    cases: list[dict[str, str]] = []

    def reject(name: str, operation: Callable[[], None]) -> None:
        try:
            operation()
        except (PackageV2Error, v1.PackageError) as error:
            cases.append({"name": name, "result": "rejected", "reason": str(error)})
        else:
            raise PackageV2Error(f"negative audit failed to reject: {name}")

    def validate(mutated_manifest: dict[str, Any] | None = None, mutated_package: bytes | None = None, mutated_capture: dict[str, Any] | None = None, mutated_inventory: dict[str, Any] | None = None) -> None:
        validate_manifest_semantics(
            mutated_manifest if mutated_manifest is not None else base,
            mutated_package if mutated_package is not None else package,
            mutated_capture if mutated_capture is not None else capture,
            mutated_inventory if mutated_inventory is not None else inventory,
        )

    changed = bytearray(package); changed[len(changed) // 2] ^= 1
    reject("package_byte_flip", lambda: validate(mutated_package=bytes(changed)))
    reject("package_truncation", lambda: validate(mutated_package=package[:-1]))
    mutated = copy.deepcopy(base); mutated["included_state"]["mappings"][17]["package_offset"] += 1
    reject("custom_weight_offset_gap", lambda: validate(mutated_manifest=mutated))
    mutated = copy.deepcopy(base); mutated["included_state"]["mappings"][18]["state_role"] = "weight"
    reject("custom_bias_role_swap", lambda: validate(mutated_manifest=mutated))
    mutated = copy.deepcopy(base); del mutated["included_state"]["mappings"][18]
    reject("custom_bias_omission", lambda: validate(mutated_manifest=mutated))
    mutated = copy.deepcopy(base); mutated["included_state"]["mappings"][24]["state_role"] = "weight"
    reject("conv2_bias_role_swap", lambda: validate(mutated_manifest=mutated))
    mutated = copy.deepcopy(base); mutated["unverified_state"]["buffers"][0]["status"] = "verified"
    reject("unverified_state_promotion", lambda: validate(mutated_manifest=mutated))
    mutated = copy.deepcopy(base); mutated["claims"]["whole_model_weights_recovered"] = True
    reject("whole_model_overclaim", lambda: validate(mutated_manifest=mutated))
    mutated = copy.deepcopy(base); mutated["claims"]["all_custom_convolution_launches_proven"] = True
    reject("custom_family_overclaim", lambda: validate(mutated_manifest=mutated))
    mutated = copy.deepcopy(base); mutated["execution_ingress"]["contains_expected_output_bytes"] = True
    reject("expected_output_ingress", lambda: validate(mutated_manifest=mutated))
    mutated = copy.deepcopy(capture); mutated["capture_point"] = "post_create_fence_pre_evaluate"
    reject("wrong_capture_point", lambda: validate(mutated_capture=mutated))
    mutated = copy.deepcopy(capture); mutated["entries"][17]["logical_bytes"] += 2
    reject("custom_source_size_mutation", lambda: validate(mutated_capture=mutated))
    mutated = copy.deepcopy(inventory); mutated["launches"][0]["bias_token"] = 110
    reject("custom_inventory_token_mutation", lambda: validate(mutated_inventory=mutated))
    mutated = copy.deepcopy(base); mutated["volatile_pointer"] = "0x123456789abc"
    reject("address_injection", lambda: validate(mutated_manifest=mutated))
    require(len(cases) == 14 and all(case["result"] == "rejected" for case in cases), "v2 negative audit incomplete")
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
    require(not output_dir.is_symlink(), "unsafe v2 output directory")
    generated = [output_dir / PACKAGE_FILE, output_dir / "capture-manifest.json", output_dir / "custom-conv-inventory.json", output_dir / "negative-audit.json", output_dir / "package-manifest.json", output_dir / "backend-receipts"]
    require(not any(path.exists() for path in generated), "refusing to overwrite immutable v2 artifacts")

    v1_manifest = v1.verify_package(V1_PACKAGE)
    capture, capture_entries, buffers = v1.load_capture_pair(args.source_job)
    inventory_path = REPOSITORY / "evidence/specs/fg-custom-conv-inventory-v1/inventory.json"
    inventory, inventory_by_launch = validate_inventory(inventory_path)
    generic_consumers, generic_receipts = generic_consumers_and_receipts(v1_manifest)
    custom_consumers, custom_receipts = validate_custom_evidence(inventory_by_launch)
    conv2_consumers, conv2_receipts = validate_conv2_evidence(
        inventory_by_launch, args.rtx_conv2, args.mac_conv2
    )
    consumers = generic_consumers + custom_consumers + conv2_consumers
    receipt_bytes = {**generic_receipts, **custom_receipts, **conv2_receipts}
    package, mappings = build_mappings(buffers, consumers)
    require(package[: v1.EXPECTED_INCLUDED_BYTES] == (V1_PACKAGE / v1.PACKAGE_FILE).read_bytes(), "v1 package prefix mismatch")

    included_tokens = {mapping["source_token"] for mapping in mappings}
    remaining = [
        {
            "source_token": token,
            "logical_bytes": entry["logical_bytes"],
            "source_sha256": entry["sha256"],
            "package_offset": None,
            "status": "unverified_candidate_excluded",
        }
        for token, entry in capture_entries.items()
        if token not in included_tokens
    ]
    base = {
        "schema": SCHEMA,
        "package_format": {
            "file": PACKAGE_FILE,
            "encoding": "opaque captured bytes",
            "layout": "launch-ascending; custom weight before bias; no padding",
            "padding_bytes": 0,
        },
        "source_evidence": {
            "predecessor_v1_manifest": source_record(V1_PACKAGE / "package-manifest.json"),
            "predecessor_v1_root_sha256": v1_manifest["package_root_sha256"],
            "predecessor_v1_payload_sha256": v1_manifest["included_state"]["package_sha256"],
            "runtime_sha256": v1.RUNTIME_SHA256,
            "capture_point": v1.CAPTURE_POINT,
            "capture_manifest_file": "capture-manifest.json",
            "capture_manifest_sha256": v1.CAPTURE_MANIFEST_SHA256,
            "capture_fresh_process_count": 2,
            "capture_poison_bytes": [90, 165],
            "capture_buffers_repeat_exact": True,
            "capture_bytes_fed_back_to_execution": False,
            "custom_convolution_inventory": source_record(inventory_path),
            "custom_convolution_inventory_file": "custom-conv-inventory.json",
            "custom_convolution_inventory_root_sha256": INVENTORY_ROOT_SHA256,
            "custom_conv2_family": source_record(REPOSITORY / "evidence/specs/fg-custom-conv2-family-v1/family-verification.json"),
            "custom_conv2_family_root_sha256": CONV2_FAMILY_ROOT_SHA256,
            "custom_conv2_artifact_manifest": source_record(REPOSITORY / "evidence/specs/fg-custom-conv2-family-v1/artifact-manifest.json"),
            "builder_source": source_record(Path(__file__).resolve()),
        },
        "proven_consumers": consumers,
        "included_state": {
            "classification": "proven_generic_and_three_custom_convolution_parameter_state",
            "buffer_count": len(mappings),
            "logical_bytes": len(package),
            "package_sha256": bytes_sha256(package),
            "mappings": mappings,
        },
        "unverified_state": {
            "classification": "candidate_runtime_state_not_yet_selected_module_consumption_proven",
            "buffer_count": len(remaining),
            "logical_bytes": sum(item["logical_bytes"] for item in remaining),
            "buffers": remaining,
        },
        "execution_provenance": {
            "proven_consumer_count": len(consumers),
            "generic_consumer_count": len(generic_consumers),
            "custom_consumer_count": len(custom_consumers) + len(conv2_consumers),
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
            "generic_convolution_launches_proven": len(generic_consumers),
            "generic_state_buffers_proven": len(v1.LAYERS),
            "custom_convolution_launches_proven": len(custom_consumers) + len(conv2_consumers),
            "custom_state_buffers_proven": sum(len(item["source_tokens"]) for item in custom_consumers + conv2_consumers),
            "custom_conv2_launches_proven": len(conv2_consumers),
            "custom_conv2_state_buffers_proven": sum(len(item["source_tokens"]) for item in conv2_consumers),
            "all_custom_convolution_launches_proven": False,
            "all_runtime_state_buffers_classified": False,
            "whole_model_weights_recovered": False,
            "connected_multi_kernel_group_proven_by_this_package": False,
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
    require_address_free(base, "final v2 package")
    try:
        write_exclusive(output_dir / PACKAGE_FILE, package)
        capture_source = args.source_job / "eval0-a/runtime-default-buffers-post-evaluate-0.manifest.json"
        write_exclusive(output_dir / "capture-manifest.json", capture_source.read_bytes())
        write_exclusive(output_dir / "custom-conv-inventory.json", inventory_path.read_bytes())
        for relative, data in receipt_bytes.items():
            write_exclusive(output_dir / relative, data)
        write_exclusive(output_dir / "negative-audit.json", audit_bytes)
        write_json_exclusive(output_dir / "package-manifest.json", base)
    except Exception:
        for target in (output_dir / PACKAGE_FILE, output_dir / "capture-manifest.json", output_dir / "custom-conv-inventory.json", output_dir / "negative-audit.json", output_dir / "package-manifest.json"):
            if target.exists() and target.is_file():
                target.unlink()
        receipts = output_dir / "backend-receipts"
        if receipts.exists() and receipts.is_dir():
            shutil.rmtree(receipts)
        raise
    return base


def verify_package(package_dir: Path) -> dict[str, Any]:
    manifest = load_json(package_dir / "package-manifest.json")
    require(manifest.get("package_root_sha256") == rooted(manifest, "package_root_sha256"), "v2 package root invalid")
    payload = package_dir / PACKAGE_FILE
    require(payload.is_file() and not payload.is_symlink(), "v2 package payload missing or unsafe")
    package = payload.read_bytes()
    capture_path = package_dir / "capture-manifest.json"
    require(file_sha256(capture_path) == v1.CAPTURE_MANIFEST_SHA256, "v2 packaged capture manifest hash")
    capture = load_json(capture_path)
    inventory_path = package_dir / "custom-conv-inventory.json"
    require(file_sha256(inventory_path) == manifest["source_evidence"]["custom_convolution_inventory"]["file_sha256"], "v2 packaged inventory file hash")
    inventory, inventory_by_launch = validate_inventory(inventory_path)
    validate_manifest_semantics(manifest, package, capture, inventory)
    validate_consumers_and_receipts(package_dir, manifest, inventory_by_launch)
    audit_meta = manifest.get("mutation_testing", {})
    audit_path = package_dir / audit_meta.get("file", "")
    require(audit_path.is_file() and file_sha256(audit_path) == audit_meta.get("file_sha256"), "v2 negative audit file hash")
    audit = load_json(audit_path)
    require(
        audit.get("schema") == AUDIT_SCHEMA
        and rooted(audit) == audit.get("receipt_root_sha256") == audit_meta.get("receipt_root_sha256")
        and audit.get("mutation_count") == audit.get("rejected_count") == audit_meta.get("mutation_count") == 14
        and audit.get("failed_to_reject") == [],
        "v2 negative audit invalid",
    )
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--source-job", type=Path, default=REPOSITORY / "build/fg_rig/fg-default-readback-20260904T095705Z")
    build.add_argument("--rtx-conv2", type=Path, default=REPOSITORY / "build/fg_conv/fg-custom-conv2-family-20260904T105849Z/retrieved")
    build.add_argument("--mac-conv2", type=Path, default=REPOSITORY / "build/fg_conv/macos-custom-conv2-root-v3")
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
    except (OSError, KeyError, TypeError, PackageV2Error, v1.PackageError) as error:
        print(f"package_v2_error={error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
