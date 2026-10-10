#!/usr/bin/env python3
"""Build and verify the narrowly proven FG generic-convolution state package."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


REPOSITORY = Path(__file__).resolve().parent.parent
SCHEMA = "neuralstudio.fg.model-state-package.v1"
AUDIT_SCHEMA = "neuralstudio.fg.model-state-package-negative-audit.v1"
CAPTURE_SCHEMA = "neuralstudio.dlss_fg.default_buffer_readback.v1"
CAPTURE_POINT = "post_evaluate_0_fence_pre_evaluate_1"
CAPTURE_MANIFEST_SHA256 = (
    "2a1179d4a25be030f772cd90ca7d017ed6eac9291917f7d43ea9b0cd5ac6a294"
)
RUNTIME_SHA256 = (
    "5d5cbf14d2727d47f93fd10bf77bd91708ae122482a6f86fd564971641ebd47b"
)
RUNTIME_EXTRACTION_ROOT_SHA256 = (
    "6ba5d3d6340a77b1984c08144517aa2b6616744e81a9d0335807dcce50c43b85"
)
GENERIC_CONV_ROOT_SHA256 = (
    "b01e03d08e309049ef89ed489b4051213a455b952182068492fe9ec5443a87fd"
)
ONE_BY_ONE_ROOT_SHA256 = (
    "100f98d331cf879e734d5303e4f1f0f4693da0f96d19e47f8793c93c88d90d71"
)
THREE_BY_THREE_ROOT_SHA256 = (
    "972a6888b346a87423949b6cfa33c6dc91e3f93c303ebfc9e420a355c14550b3"
)
EXPECTED_TOKENS = list(range(63, 96, 2)) + list(range(106, 205, 2))
EXPECTED_CAPTURE_BYTES = 4_868_576
EXPECTED_INCLUDED_BYTES = 3_905_024
EXPECTED_EXCLUDED_BYTES = 963_552
PACKAGE_FILE = "selected-generic-conv-state.bin"


@dataclass(frozen=True)
class Layer:
    layer_id: str
    launch: int
    kind: str
    module_token: int
    module_sha256: str
    source_token: int
    logical_bytes: int
    source_sha256: str


LAYERS = (
    Layer("l20", 20, "3x3", 28, "7f9c1f110e8ec9347aa968bc9019e5ffc8839057987a2e5676a0f64637c8df2d", 63, 56320, "44595b02307a6b934a124abe476cfd327440e2acba7f5ed69f73195ca743e3d9"),
    Layer("l22", 22, "1x1", 30, "31bb08d04011db9a0def8f1a1bc3999efc08d5deb8d7cb9b5debf52ec489b785", 65, 10240, "eb99328524fddc9f5f3fac12b8e3a4011e66a75e071cd07126a70c553458bc60"),
    Layer("l24", 24, "3x3", 31, "e925f51165408fae4657d50247b10304802d46979b863712aeff844a217ade63", 67, 299008, "4191924ff335d5018987f2526f933cb4cdbb9145411e3ce34f63c8206321be55"),
    Layer("l26", 26, "3x3", 32, "c74961f6f71f4e7bac5b216cc704b10a0fdd6f0987352fdd315f371467ca7187", 69, 593920, "c0c5648f9cb839e41d0e786c8034aa0bcc931b6ac84c58cfbde0502dc8a53404"),
    Layer("l28", 28, "3x3", 33, "6f2ffbe7faacde8119719d92dacde40f21ddd562878381b81654dad7a66775dd", 71, 1187840, "fe2c3057f2e894219601d3d7200eabfa97d990e08ccb23b43d52dfcba2d2fb93"),
    Layer("l30", 30, "1x1", 34, "2f490857c6f7aa22f2435057706ec10b28b6b8d3081bd967ebe1d61550ac6419", 73, 540672, "e23316a86707c1d3d377aa0711df47d05976ad1a30f6428524d030dd17145de9"),
    Layer("l31", 31, "1x1", 35, "331853771cae4aa041038d8e9c74dcb1f8d0d38b4d5603e2e8d079ad1e44fd04", 75, 532480, "9f20d6589cc9269ad020233e68196d66eba0e8a76497374cf6d445385f94dbc4"),
    Layer("l34", 34, "1x1", 38, "8e7abed27b7eae0dd274b2409a13d6dde98277d9e743f0e36a1042022a16ceda", 77, 270336, "9e5da09b93650c74e89d5262aa8a9a6a3dcbd0d753dc8c9d9aed39731e39b2a0"),
    Layer("l35", 35, "1x1", 39, "939d5d9cf7012fb3ed3ce4e4792ccf5c6befec165ae7f7a0d3704cb1103c3030", 79, 135168, "3969384493583e98dcf357d3046a10a5b836585261f3e584605bf11e23fb8d46"),
    Layer("l38", 38, "1x1", 42, "93f3bb1be638eaf8003ba04cc7977859dfb16ae7821801c3017e45546301ecbc", 81, 69632, "a0c6690faee2838521c6ad0f9445425d639446c0ef1b7fed9b43d21e7a8ec650"),
    Layer("l39", 39, "1x1", 42, "93f3bb1be638eaf8003ba04cc7977859dfb16ae7821801c3017e45546301ecbc", 83, 69632, "3a13970b482f1f78ede1bfe531956c203f08f99077c81bda2d1e4d0f453cf9a0"),
    Layer("l42", 42, "1x1", 43, "53fdf8d5404c93f2fe64582f271192df3b24c2ef08d9564ffeb9d7990485db0e", 85, 69632, "47361a699fa6514332dc560cf2416872256e61d1b36b1ca29df323a53ee2be77"),
    Layer("l43", 43, "1x1", 44, "7113dfaae4945541276b3e662cba366f9a3fd9666db7a338efa867cfd7f46e31", 87, 34816, "327bd9b02da4aaac8b251320f60b4aeccb0df7df84bba44d7f1f6a9fdf62c934"),
    Layer("l46", 46, "1x1", 47, "8a3efc1e9166ed3ee4a54aea535910c0562b6753b2d0daae0402e1ed36ff19dc", 89, 18432, "e24f67110bab401ab7f70a861f3f09d2cd3cd6abf8d1aa158ba3300efb64797e"),
    Layer("l47", 47, "1x1", 48, "11906664ac548c61262a12b0950bf135926cf269de2d7691d0edfbe4a95e63dd", 91, 9216, "325d7dfac5620a89edc1d2315731414cf755c50506ea5758a8c3cf563649577c"),
    Layer("l50", 50, "1x1", 51, "304d0c0e58958fa7eab6a82fd046ff689c51a986aff88a069f6f509c434a5b96", 93, 5120, "bb98eb15be7d255a15a46994589705560e0691e273e286a36a68fd679a1890a9"),
    Layer("l51", 51, "1x1", 52, "27486bcc1db8b04b0e18d72f33b06662da9c327eb15b41a09b3f909a3a52cad2", 95, 2560, "477c68ebace499e2e1ac331f214fae3e9fb4df815a9e7d8142a0f1fb414b340e"),
)


class PackageError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PackageError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing or unsafe JSON: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as error:
        raise PackageError(f"invalid JSON: {path}: {error}") from error
    require(isinstance(value, dict), f"JSON is not an object: {path}")
    return value


def rooted(value: dict[str, Any], field: str) -> str:
    body = copy.deepcopy(value)
    body.pop(field, None)
    return canonical_sha256(body)


def require_address_free(value: Any, label: str) -> None:
    def visit(node: Any, location: str) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                if key != "address_fields_recorded":
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
            require(
                re.search(r"0x[0-9a-fA-F]{12,16}(?:\b|$)", node) is None,
                f"{label}: address-like text at {location}",
            )
    visit(value, label)


def write_exclusive(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)


def write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    write_exclusive(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def validate_rooted_receipt(path: Path, schema: str, expected_root: str) -> dict[str, Any]:
    value = load_json(path)
    require(value.get("schema") == schema, f"schema drift: {path}")
    require(value.get("receipt_root_sha256") == expected_root, f"root identity drift: {path}")
    require(rooted(value, "receipt_root_sha256") == expected_root, f"invalid receipt root: {path}")
    require_address_free(value, str(path))
    return value


def validate_runtime_summary(path: Path) -> dict[str, Any]:
    value = load_json(path)
    require(value.get("schema") == "neuralstudio.fg.runtime-buffer-extraction-summary.v1", "runtime summary schema")
    require(rooted(value, "summary_root_sha256") == value.get("summary_root_sha256"), "runtime summary root")
    repeat = value.get("post_evaluate_0_default_repeat", {})
    require(
        value.get("runtime_buffer_extraction_accepted") is True
        and value.get("model_weights_recovered") is False
        and value.get("exact_selected_module_consumption_proven") is False,
        "runtime extraction frontier drift",
    )
    require(
        value.get("frozen_allocation_provenance", {}).get("runtime_sha256") == RUNTIME_SHA256
        and repeat.get("capture_point") == CAPTURE_POINT
        and repeat.get("fresh_process_count") == 2
        and repeat.get("poison_bytes") == [90, 165]
        and repeat.get("buffer_count_per_run") == 67
        and repeat.get("logical_bytes_per_run") == EXPECTED_CAPTURE_BYTES
        and repeat.get("all_dump_hashes_repeat_exact") is True
        and repeat.get("bytes_fed_back_to_execution") is False,
        "runtime extraction provenance drift",
    )
    require(
        value.get("receipt_reproducibility", {}).get("shared_receipt_root_sha256")
        == RUNTIME_EXTRACTION_ROOT_SHA256,
        "runtime extraction receipt identity drift",
    )
    return value


def validate_capture_manifest(value: dict[str, Any]) -> dict[int, dict[str, Any]]:
    expected_fields = {
        "schema", "capture_point", "captured_buffers", "captured_logical_bytes",
        "source_heap", "direct_default_map", "copy_submission", "source_state_strategy",
        "explicit_source_transition", "bytes_fed_back_to_execution", "entries", "privacy",
    }
    require(set(value) == expected_fields, "capture manifest fields drift")
    require(
        value.get("schema") == CAPTURE_SCHEMA
        and value.get("capture_point") == CAPTURE_POINT
        and value.get("captured_buffers") == 67
        and value.get("captured_logical_bytes") == EXPECTED_CAPTURE_BYTES
        and value.get("source_heap") == "default"
        and value.get("bytes_fed_back_to_execution") is False
        and value.get("privacy") == "private_local_evidence",
        "capture manifest contract drift",
    )
    entries = value.get("entries")
    require(isinstance(entries, list) and len(entries) == 67, "capture entries incomplete")
    result: dict[int, dict[str, Any]] = {}
    for entry in entries:
        require(isinstance(entry, dict), "capture entry is not an object")
        token = entry.get("source_token")
        require(isinstance(token, int) and token not in result, "invalid or duplicate source token")
        require(
            entry.get("dump_file") == f"default-token-{token:06d}.bin"
            and entry.get("descriptor_dimension") == 1
            and entry.get("descriptor_width") == entry.get("logical_bytes")
            and entry.get("source_heap_type") == 1
            and entry.get("source_initial_state") == 0
            and isinstance(entry.get("logical_bytes"), int)
            and entry["logical_bytes"] > 0
            and isinstance(entry.get("sha256"), str)
            and len(entry["sha256"]) == 64,
            f"invalid capture entry for token {token}",
        )
        result[token] = entry
    require(list(result) == EXPECTED_TOKENS, "capture token order/inventory drift")
    require(sum(entry["logical_bytes"] for entry in result.values()) == EXPECTED_CAPTURE_BYTES, "capture byte total drift")
    require_address_free(value, "capture manifest")
    return result


def load_capture_pair(job_root: Path) -> tuple[dict[str, Any], dict[int, dict[str, Any]], dict[int, bytes]]:
    run_a = job_root / "eval0-a"
    run_b = job_root / "eval0-b"
    manifest_name = "runtime-default-buffers-post-evaluate-0.manifest.json"
    manifest_a_path = run_a / manifest_name
    manifest_b_path = run_b / manifest_name
    require(file_sha256(manifest_a_path) == CAPTURE_MANIFEST_SHA256, "capture A manifest hash")
    require(file_sha256(manifest_b_path) == CAPTURE_MANIFEST_SHA256, "capture B manifest hash")
    require(manifest_a_path.read_bytes() == manifest_b_path.read_bytes(), "capture manifests differ")
    manifest = load_json(manifest_a_path)
    entries = validate_capture_manifest(manifest)
    buffers: dict[int, bytes] = {}
    for token, entry in entries.items():
        relative = Path("runtime-default-buffers-post-evaluate-0") / entry["dump_file"]
        path_a = run_a / relative
        path_b = run_b / relative
        require(path_a.is_file() and path_b.is_file(), f"missing source token {token}")
        data_a = path_a.read_bytes()
        data_b = path_b.read_bytes()
        require(len(data_a) == entry["logical_bytes"], f"token {token} size")
        require(bytes_sha256(data_a) == entry["sha256"], f"token {token} capture A hash")
        require(bytes_sha256(data_b) == entry["sha256"], f"token {token} capture B hash")
        require(data_a == data_b, f"token {token} fresh-process/poison mismatch")
        buffers[token] = data_a
    return manifest, entries, buffers


def validate_backend_receipt(value: dict[str, Any], layer: Layer, backend: str, expected_root: str) -> None:
    expected_schema = f"neuralstudio.fg.conv-{layer.kind}-backend.v1"
    require(value.get("schema") == expected_schema, f"{layer.layer_id}/{backend}: schema")
    require(value.get("backend") == backend, f"{layer.layer_id}/{backend}: backend")
    require(value.get("receipt_root_sha256") == expected_root, f"{layer.layer_id}/{backend}: root identity")
    require(rooted(value, "receipt_root_sha256") == expected_root, f"{layer.layer_id}/{backend}: invalid root")
    require(
        value.get("layer_id") == layer.layer_id
        and value.get("trace_launch_token") == f"launch-{layer.launch:06d}"
        and value.get("module_token") == layer.module_token
        and value.get("module_sha256") == layer.module_sha256
        and value.get("weight_source_token") == layer.source_token
        and value.get("weight_sha256") == layer.source_sha256
        and value.get("weight_capture_manifest_sha256") == CAPTURE_MANIFEST_SHA256,
        f"{layer.layer_id}/{backend}: selected-state identity drift",
    )
    require(
        value.get("fresh_process_count") == 4
        and value.get("two_process_two_poison_invariant") is True
        and value.get("full_write_coverage_from_two_poison_invariance") is True
        and value.get("input_weight_immutability") is True
        and value.get("canary_integrity") is True
        and value.get("expected_output_fed_to_execution") is False
        and value.get("output_mutation_rejected") is True
        and value.get("address_fields_recorded") is False
        and value.get("result") == "pass",
        f"{layer.layer_id}/{backend}: execution gates drift",
    )
    require_address_free(value, f"{layer.layer_id}/{backend}")


def layer_receipt_path(layer: Layer) -> Path:
    return (
        REPOSITORY / "evidence" / "specs" / f"fg-conv-{layer.kind}-family-v1"
        / "layers" / f"{layer.layer_id}-verification.json"
    )


def validate_conv_evidence(
    rtx_1x1: Path, mac_1x1: Path, rtx_3x3: Path, mac_3x3: Path
) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    aggregate_path = REPOSITORY / "evidence/specs/fg-conv-family-v1/verification.json"
    aggregate = validate_rooted_receipt(aggregate_path, "neuralstudio.fg.conv-family.v1", GENERIC_CONV_ROOT_SHA256)
    require(
        aggregate.get("selected_convolution_launch_count") == 17
        and aggregate.get("every_selected_convolution_executed_independently") is True
        and aggregate.get("every_selected_convolution_rtx_mac_exact") is True
        and aggregate.get("connected_multi_kernel_group_executed") is False
        and aggregate.get("full_frame_parity_claimed") is False,
        "generic convolution aggregate frontier drift",
    )
    family_specs = (
        ("1x1", "neuralstudio.fg.conv-1x1-family.v1", ONE_BY_ONE_ROOT_SHA256),
        ("3x3", "neuralstudio.fg.conv-3x3-family.v1", THREE_BY_THREE_ROOT_SHA256),
    )
    for kind, schema, root in family_specs:
        validate_rooted_receipt(
            REPOSITORY / f"evidence/specs/fg-conv-{kind}-family-v1/family-verification.json",
            schema,
            root,
        )

    records: list[dict[str, Any]] = []
    receipt_bytes: dict[str, bytes] = {}
    for layer in LAYERS:
        final_path = layer_receipt_path(layer)
        final = load_json(final_path)
        require(final.get("schema") == f"neuralstudio.fg.conv-{layer.kind}-verification.v1", f"{layer.layer_id}: final schema")
        require(rooted(final, "receipt_root_sha256") == final.get("receipt_root_sha256"), f"{layer.layer_id}: final root")
        require(
            final.get("layer_id") == layer.layer_id
            and final.get("trace_launch_token") == f"launch-{layer.launch:06d}"
            and final.get("module_sha256") == layer.module_sha256
            and final.get("weight_sha256") == layer.source_sha256
            and final.get("rtx_mac_differing_bytes") == 0
            and final.get("rtx_mac_exact") is True
            and final.get("result") == "pass",
            f"{layer.layer_id}: final exactness drift",
        )
        require_address_free(final, f"{layer.layer_id}/final")

        rtx_root = rtx_1x1 if layer.kind == "1x1" else rtx_3x3
        mac_root = mac_1x1 if layer.kind == "1x1" else mac_3x3
        rtx_path = rtx_root / layer.layer_id / "backend-receipt.json"
        mac_path = mac_root / layer.layer_id / "backend-receipt.json"
        rtx = load_json(rtx_path)
        mac = load_json(mac_path)
        validate_backend_receipt(rtx, layer, "rtx-cuda-driver", final["rtx_backend_root_sha256"])
        validate_backend_receipt(mac, layer, "mac-metal", final["mac_backend_root_sha256"])
        require(rtx.get("output_sha256") == mac.get("output_sha256") == final.get("output_sha256"), f"{layer.layer_id}: output identity")
        for label, source_path, value in (("rtx", rtx_path, rtx), ("mac", mac_path, mac)):
            package_name = f"backend-receipts/{layer.layer_id}-{label}.json"
            source_data = source_path.read_bytes()
            require(json.loads(source_data.decode("utf-8-sig")) == value, f"{layer.layer_id}/{label}: receipt read drift")
            # Preserve canonical receipt content while normalizing Windows evidence
            # to portable LF text in the committed package.
            data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
            receipt_bytes[package_name] = data
        records.append(
            {
                "layer_id": layer.layer_id,
                "trace_launch_token": f"launch-{layer.launch:06d}",
                "kind": layer.kind,
                "module_token": layer.module_token,
                "module_sha256": layer.module_sha256,
                "source_token": layer.source_token,
                "source_sha256": layer.source_sha256,
                "logical_bytes": layer.logical_bytes,
                "exact_receipt_relative_path": str(final_path.relative_to(REPOSITORY)),
                "exact_receipt_file_sha256": file_sha256(final_path),
                "exact_receipt_root_sha256": final["receipt_root_sha256"],
                "rtx_backend_receipt": "backend-receipts/" + layer.layer_id + "-rtx.json",
                "rtx_backend_receipt_file_sha256": bytes_sha256(receipt_bytes[f"backend-receipts/{layer.layer_id}-rtx.json"]),
                "rtx_backend_receipt_root_sha256": final["rtx_backend_root_sha256"],
                "mac_backend_receipt": "backend-receipts/" + layer.layer_id + "-mac.json",
                "mac_backend_receipt_file_sha256": bytes_sha256(receipt_bytes[f"backend-receipts/{layer.layer_id}-mac.json"]),
                "mac_backend_receipt_root_sha256": final["mac_backend_root_sha256"],
                "fresh_process_count_per_backend": 4,
                "poison_bytes": [90, 165],
                "expected_output_bytes_packaged": False,
            }
        )
    require([record["layer_id"] for record in records] == aggregate["selected_convolution_launches"], "aggregate launch order drift")
    return records, receipt_bytes


def validate_package_semantics(manifest: dict[str, Any], package: bytes, capture: dict[str, Any]) -> None:
    require(manifest.get("schema") == SCHEMA, "package schema")
    require(manifest.get("address_fields_recorded") is False, "package address-free flag")
    require_address_free(manifest, "package manifest")
    require(
        manifest.get("package_format")
        == {
            "file": PACKAGE_FILE,
            "encoding": "opaque captured bytes",
            "layout": "launch-ascending raw token payload concatenation",
            "padding_bytes": 0,
        },
        "package format drift",
    )
    source = manifest.get("source_evidence", {})
    require(
        source.get("runtime_sha256") == RUNTIME_SHA256
        and source.get("capture_point") == CAPTURE_POINT
        and source.get("capture_manifest_file") == "capture-manifest.json"
        and source.get("capture_manifest_sha256") == CAPTURE_MANIFEST_SHA256
        and source.get("capture_fresh_process_count") == 2
        and source.get("capture_poison_bytes") == [90, 165]
        and source.get("capture_buffers_repeat_exact") is True
        and source.get("capture_bytes_fed_back_to_execution") is False
        and source.get("runtime_extraction_receipt_root_sha256") == RUNTIME_EXTRACTION_ROOT_SHA256
        and source.get("generic_convolution_aggregate_root_sha256") == GENERIC_CONV_ROOT_SHA256
        and source.get("one_by_one_family_root_sha256") == ONE_BY_ONE_ROOT_SHA256
        and source.get("three_by_three_family_root_sha256") == THREE_BY_THREE_ROOT_SHA256,
        "source evidence identity drift",
    )
    expected_sources = {
        "runtime_extraction_summary": REPOSITORY / "evidence/specs/fg-runtime-buffer-extraction-v1/real-runtime-buffer-extraction-summary.json",
        "generic_convolution_aggregate": REPOSITORY / "evidence/specs/fg-conv-family-v1/verification.json",
        "one_by_one_family": REPOSITORY / "evidence/specs/fg-conv-1x1-family-v1/family-verification.json",
        "three_by_three_family": REPOSITORY / "evidence/specs/fg-conv-3x3-family-v1/family-verification.json",
        "builder_source": Path(__file__).resolve(),
    }
    for label, path in expected_sources.items():
        record = source.get(label, {})
        require(
            record.get("relative_path") == str(path.relative_to(REPOSITORY))
            and record.get("file_sha256") == file_sha256(path),
            f"{label}: source file drift",
        )
    runtime_summary = load_json(expected_sources["runtime_extraction_summary"])
    require(
        source.get("runtime_extraction_summary_root_sha256") == runtime_summary.get("summary_root_sha256"),
        "runtime summary root binding",
    )
    aggregate = validate_rooted_receipt(
        expected_sources["generic_convolution_aggregate"],
        "neuralstudio.fg.conv-family.v1",
        GENERIC_CONV_ROOT_SHA256,
    )
    aggregate_by_layer = {item["layer_id"]: item for item in aggregate["per_launch_receipts"]}
    included = manifest.get("included_state")
    excluded = manifest.get("unverified_state")
    require(isinstance(included, dict) and isinstance(excluded, dict), "state partitions missing")
    mappings = included.get("mappings")
    remaining = excluded.get("buffers")
    require(isinstance(mappings, list) and len(mappings) == 17, "included state count")
    require(isinstance(remaining, list) and len(remaining) == 50, "unverified state count")
    offset = 0
    for layer, mapping in zip(LAYERS, mappings, strict=True):
        require(
            mapping.get("layer_id") == layer.layer_id
            and mapping.get("trace_launch_token") == f"launch-{layer.launch:06d}"
            and mapping.get("kind") == layer.kind
            and mapping.get("module_token") == layer.module_token
            and mapping.get("module_sha256") == layer.module_sha256
            and mapping.get("source_token") == layer.source_token
            and mapping.get("source_sha256") == layer.source_sha256
            and mapping.get("logical_bytes") == layer.logical_bytes,
            f"{layer.layer_id}: package mapping identity",
        )
        require(
            mapping.get("package_offset") == offset
            and mapping.get("package_end_exclusive") == offset + layer.logical_bytes,
            f"{layer.layer_id}: package offset",
        )
        final_path = layer_receipt_path(layer)
        final = load_json(final_path)
        aggregate_item = aggregate_by_layer.get(layer.layer_id, {})
        require(
            mapping.get("exact_receipt_relative_path") == str(final_path.relative_to(REPOSITORY))
            and mapping.get("exact_receipt_file_sha256") == file_sha256(final_path)
            and mapping.get("exact_receipt_root_sha256") == final.get("receipt_root_sha256")
            and mapping.get("exact_receipt_root_sha256") == aggregate_item.get("receipt_root_sha256")
            and mapping.get("rtx_backend_receipt_root_sha256") == final.get("rtx_backend_root_sha256")
            and mapping.get("mac_backend_receipt_root_sha256") == final.get("mac_backend_root_sha256")
            and mapping.get("rtx_backend_receipt") == f"backend-receipts/{layer.layer_id}-rtx.json"
            and mapping.get("mac_backend_receipt") == f"backend-receipts/{layer.layer_id}-mac.json"
            and mapping.get("fresh_process_count_per_backend") == 4
            and mapping.get("poison_bytes") == [90, 165]
            and mapping.get("expected_output_bytes_packaged") is False,
            f"{layer.layer_id}: exact execution provenance binding",
        )
        require(bytes_sha256(package[offset:offset + layer.logical_bytes]) == layer.source_sha256, f"{layer.layer_id}: package slice")
        offset += layer.logical_bytes
    require(offset == len(package) == EXPECTED_INCLUDED_BYTES, "included package byte total")
    require(included.get("buffer_count") == 17 and included.get("logical_bytes") == EXPECTED_INCLUDED_BYTES, "included summary")
    require(included.get("package_sha256") == bytes_sha256(package), "package hash")
    capture_entries = validate_capture_manifest(capture)
    included_tokens = [mapping["source_token"] for mapping in mappings]
    excluded_tokens = [entry.get("source_token") for entry in remaining]
    require(len(excluded_tokens) == len(set(excluded_tokens)), "duplicate unverified token")
    require(included_tokens + excluded_tokens == EXPECTED_TOKENS, "state partition/order drift")
    for item in remaining:
        token = item["source_token"]
        source = capture_entries[token]
        require(
            item.get("logical_bytes") == source["logical_bytes"]
            and item.get("source_sha256") == source["sha256"]
            and item.get("package_offset") is None
            and item.get("status") == "unverified_candidate_excluded",
            f"token {token}: unverified boundary drift",
        )
    require(excluded.get("buffer_count") == 50 and excluded.get("logical_bytes") == EXPECTED_EXCLUDED_BYTES, "unverified summary")
    require(sum(item["logical_bytes"] for item in remaining) == EXPECTED_EXCLUDED_BYTES, "unverified byte total")
    claims = manifest.get("claims", {})
    require(
        claims.get("generic_convolution_state_buffers_proven") is True
        and claims.get("generic_convolution_launches_proven") == 17
        and claims.get("custom_convolution_state_proven") is False
        and claims.get("all_runtime_state_buffers_classified") is False
        and claims.get("whole_model_weights_recovered") is False
        and claims.get("connected_multi_kernel_group_proven") is False
        and claims.get("full_generated_frame_parity") is False
        and claims.get("netal_integration") is False
        and claims.get("performance_claim") is False,
        "claim boundary drift",
    )
    ingress = manifest.get("execution_ingress", {})
    require(
        ingress.get("contains_expected_output_bytes") is False
        and ingress.get("contains_oracle_output_bytes") is False
        and ingress.get("contains_activation_or_intermediate_buffers") is False
        and ingress.get("contains_unverified_state_buffers") is False
        and ingress.get("permitted_role") == "generic_convolution_parameter_state_only",
        "execution ingress boundary drift",
    )
    require(
        manifest.get("execution_provenance")
        == {
            "per_layer_backend_receipts_packaged": True,
            "backends": ["rtx-cuda-driver", "mac-metal"],
            "fresh_process_count_per_layer_per_backend": 4,
            "poison_bytes": [90, 165],
            "two_process_two_poison_invariance_required": True,
            "input_and_weight_immutability_required": True,
            "canary_integrity_required": True,
            "expected_output_fed_to_execution": False,
        },
        "execution provenance summary drift",
    )


def negative_audit(base: dict[str, Any], package: bytes, capture: dict[str, Any]) -> dict[str, Any]:
    cases: list[dict[str, str]] = []

    def reject(name: str, operation: Callable[[], None]) -> None:
        try:
            operation()
        except PackageError as error:
            cases.append({"name": name, "result": "rejected", "reason": str(error)})
        else:
            raise PackageError(f"negative audit failed to reject: {name}")

    def validate(mutated_manifest: dict[str, Any] | None = None, mutated_package: bytes | None = None, mutated_capture: dict[str, Any] | None = None) -> None:
        validate_package_semantics(mutated_manifest or base, mutated_package if mutated_package is not None else package, mutated_capture or capture)

    changed = bytearray(package); changed[len(changed) // 2] ^= 1
    reject("package_byte_flip", lambda: validate(mutated_package=bytes(changed)))
    reject("package_truncation", lambda: validate(mutated_package=package[:-1]))
    mutated = copy.deepcopy(base); mutated["included_state"]["mappings"][4]["package_offset"] += 1
    reject("package_offset_gap", lambda: validate(mutated_manifest=mutated))
    mutated = copy.deepcopy(base); mutated["included_state"]["mappings"][1]["source_token"] = 63
    reject("duplicate_included_source_token", lambda: validate(mutated_manifest=mutated))
    mutated = copy.deepcopy(base); mutated["unverified_state"]["buffers"][0]["status"] = "verified"
    reject("unverified_state_promotion", lambda: validate(mutated_manifest=mutated))
    mutated = copy.deepcopy(base); mutated["claims"]["whole_model_weights_recovered"] = True
    reject("whole_model_overclaim", lambda: validate(mutated_manifest=mutated))
    mutated = copy.deepcopy(base); mutated["execution_ingress"]["contains_expected_output_bytes"] = True
    reject("expected_output_ingress", lambda: validate(mutated_manifest=mutated))
    mutated = copy.deepcopy(capture); mutated["capture_point"] = "post_create_fence_pre_evaluate"
    reject("wrong_capture_point", lambda: validate(mutated_capture=mutated))
    mutated = copy.deepcopy(capture); mutated["entries"][0]["logical_bytes"] += 2
    reject("capture_size_mutation", lambda: validate(mutated_capture=mutated))
    mutated = copy.deepcopy(capture); mutated["entries"][0]["source_token"] = 65
    reject("capture_token_mutation", lambda: validate(mutated_capture=mutated))
    mutated = copy.deepcopy(base); mutated["volatile_pointer"] = "0x123456789abc"
    reject("address_injection", lambda: validate(mutated_manifest=mutated))
    require(len(cases) == 11 and all(case["result"] == "rejected" for case in cases), "negative audit incomplete")
    result = {
        "schema": AUDIT_SCHEMA,
        "mutation_count": len(cases),
        "rejected_count": len(cases),
        "failed_to_reject": [],
        "cases": cases,
    }
    result["receipt_root_sha256"] = canonical_sha256(result)
    return result


def source_record(path: Path) -> dict[str, Any]:
    return {"relative_path": str(path.relative_to(REPOSITORY)), "file_sha256": file_sha256(path)}


def build_package(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    require(not output_dir.is_symlink(), "unsafe output directory")
    generated_targets = [
        output_dir / PACKAGE_FILE,
        output_dir / "capture-manifest.json",
        output_dir / "negative-audit.json",
        output_dir / "package-manifest.json",
        output_dir / "backend-receipts",
    ]
    require(not any(path.exists() for path in generated_targets), "refusing to overwrite immutable package artifacts")

    runtime_summary_path = REPOSITORY / "evidence/specs/fg-runtime-buffer-extraction-v1/real-runtime-buffer-extraction-summary.json"
    validate_runtime_summary(runtime_summary_path)
    capture, entries, buffers = load_capture_pair(args.source_job)
    records, receipt_bytes = validate_conv_evidence(args.rtx_1x1, args.mac_1x1, args.rtx_3x3, args.mac_3x3)

    package = bytearray()
    mappings: list[dict[str, Any]] = []
    for layer, record in zip(LAYERS, records, strict=True):
        require(entries[layer.source_token]["sha256"] == layer.source_sha256, f"{layer.layer_id}: capture/conv hash mismatch")
        require(entries[layer.source_token]["logical_bytes"] == layer.logical_bytes, f"{layer.layer_id}: capture/conv size mismatch")
        offset = len(package)
        package.extend(buffers[layer.source_token])
        mapping = dict(record)
        mapping["package_offset"] = offset
        mapping["package_end_exclusive"] = len(package)
        mappings.append(mapping)
    require(len(package) == EXPECTED_INCLUDED_BYTES, "selected package size drift")

    included_tokens = {layer.source_token for layer in LAYERS}
    remaining = [
        {
            "source_token": token,
            "logical_bytes": entry["logical_bytes"],
            "source_sha256": entry["sha256"],
            "package_offset": None,
            "status": "unverified_candidate_excluded",
        }
        for token, entry in entries.items()
        if token not in included_tokens
    ]

    runtime_summary = load_json(runtime_summary_path)
    conv_aggregate_path = REPOSITORY / "evidence/specs/fg-conv-family-v1/verification.json"
    one_by_one_path = REPOSITORY / "evidence/specs/fg-conv-1x1-family-v1/family-verification.json"
    three_by_three_path = REPOSITORY / "evidence/specs/fg-conv-3x3-family-v1/family-verification.json"
    base = {
        "schema": SCHEMA,
        "package_format": {
            "file": PACKAGE_FILE,
            "encoding": "opaque captured bytes",
            "layout": "launch-ascending raw token payload concatenation",
            "padding_bytes": 0,
        },
        "source_evidence": {
            "runtime_sha256": RUNTIME_SHA256,
            "capture_point": CAPTURE_POINT,
            "capture_manifest_file": "capture-manifest.json",
            "capture_manifest_sha256": CAPTURE_MANIFEST_SHA256,
            "capture_fresh_process_count": 2,
            "capture_poison_bytes": [90, 165],
            "capture_buffers_repeat_exact": True,
            "capture_bytes_fed_back_to_execution": False,
            "runtime_extraction_summary": source_record(runtime_summary_path),
            "runtime_extraction_summary_root_sha256": runtime_summary["summary_root_sha256"],
            "runtime_extraction_receipt_root_sha256": RUNTIME_EXTRACTION_ROOT_SHA256,
            "generic_convolution_aggregate": source_record(conv_aggregate_path),
            "generic_convolution_aggregate_root_sha256": GENERIC_CONV_ROOT_SHA256,
            "one_by_one_family": source_record(one_by_one_path),
            "one_by_one_family_root_sha256": ONE_BY_ONE_ROOT_SHA256,
            "three_by_three_family": source_record(three_by_three_path),
            "three_by_three_family_root_sha256": THREE_BY_THREE_ROOT_SHA256,
            "builder_source": source_record(Path(__file__).resolve()),
        },
        "included_state": {
            "classification": "proven_generic_convolution_parameter_state",
            "buffer_count": len(mappings),
            "logical_bytes": len(package),
            "package_sha256": bytes_sha256(bytes(package)),
            "mappings": mappings,
        },
        "unverified_state": {
            "classification": "candidate_runtime_state_not_yet_module_consumption_proven",
            "buffer_count": len(remaining),
            "logical_bytes": sum(entry["logical_bytes"] for entry in remaining),
            "buffers": remaining,
        },
        "execution_provenance": {
            "per_layer_backend_receipts_packaged": True,
            "backends": ["rtx-cuda-driver", "mac-metal"],
            "fresh_process_count_per_layer_per_backend": 4,
            "poison_bytes": [90, 165],
            "two_process_two_poison_invariance_required": True,
            "input_and_weight_immutability_required": True,
            "canary_integrity_required": True,
            "expected_output_fed_to_execution": False,
        },
        "execution_ingress": {
            "contains_expected_output_bytes": False,
            "contains_oracle_output_bytes": False,
            "contains_activation_or_intermediate_buffers": False,
            "contains_unverified_state_buffers": False,
            "permitted_role": "generic_convolution_parameter_state_only",
        },
        "claims": {
            "generic_convolution_state_buffers_proven": True,
            "generic_convolution_launches_proven": 17,
            "custom_convolution_state_proven": False,
            "all_runtime_state_buffers_classified": False,
            "whole_model_weights_recovered": False,
            "connected_multi_kernel_group_proven": False,
            "full_generated_frame_parity": False,
            "netal_integration": False,
            "performance_claim": False,
        },
        "address_fields_recorded": False,
    }
    validate_package_semantics(base, bytes(package), capture)
    audit = negative_audit(base, bytes(package), capture)
    audit_bytes = (json.dumps(audit, indent=2, sort_keys=True) + "\n").encode()
    base["mutation_testing"] = {
        "file": "negative-audit.json",
        "file_sha256": bytes_sha256(audit_bytes),
        "receipt_root_sha256": audit["receipt_root_sha256"],
        "mutation_count": audit["mutation_count"],
        "rejected_count": audit["rejected_count"],
    }
    base["package_root_sha256"] = canonical_sha256(base)
    require_address_free(base, "final package")

    try:
        write_exclusive(output_dir / PACKAGE_FILE, bytes(package))
        write_exclusive(output_dir / "capture-manifest.json", (args.source_job / "eval0-a" / "runtime-default-buffers-post-evaluate-0.manifest.json").read_bytes())
        for relative, data in receipt_bytes.items():
            write_exclusive(output_dir / relative, data)
        write_exclusive(output_dir / "negative-audit.json", audit_bytes)
        write_json_exclusive(output_dir / "package-manifest.json", base)
    except Exception:
        # Only clean the artifacts this invocation created. Never remove the output root.
        for target in (output_dir / PACKAGE_FILE, output_dir / "capture-manifest.json", output_dir / "negative-audit.json", output_dir / "package-manifest.json"):
            if target.exists() and target.is_file():
                target.unlink()
        receipts = output_dir / "backend-receipts"
        if receipts.exists() and receipts.is_dir():
            shutil.rmtree(receipts)
        raise
    return base


def verify_package(package_dir: Path) -> dict[str, Any]:
    manifest_path = package_dir / "package-manifest.json"
    manifest = load_json(manifest_path)
    require(manifest.get("package_root_sha256") == rooted(manifest, "package_root_sha256"), "package root invalid")
    package_path = package_dir / PACKAGE_FILE
    require(package_path.is_file() and not package_path.is_symlink(), "package payload missing or unsafe")
    package = package_path.read_bytes()
    capture_path = package_dir / "capture-manifest.json"
    require(file_sha256(capture_path) == CAPTURE_MANIFEST_SHA256, "packaged capture manifest hash")
    capture = load_json(capture_path)
    validate_package_semantics(manifest, package, capture)
    for layer, mapping in zip(LAYERS, manifest["included_state"]["mappings"], strict=True):
        backend_values: dict[str, dict[str, Any]] = {}
        for label, backend in (("rtx", "rtx-cuda-driver"), ("mac", "mac-metal")):
            receipt_path = package_dir / mapping[f"{label}_backend_receipt"]
            require(file_sha256(receipt_path) == mapping[f"{label}_backend_receipt_file_sha256"], f"{layer.layer_id}/{label}: packaged receipt hash")
            receipt = load_json(receipt_path)
            validate_backend_receipt(receipt, layer, backend, mapping[f"{label}_backend_receipt_root_sha256"])
            backend_values[label] = receipt
        final = load_json(layer_receipt_path(layer))
        require(
            backend_values["rtx"].get("output_sha256")
            == backend_values["mac"].get("output_sha256")
            == final.get("output_sha256"),
            f"{layer.layer_id}: packaged RTX/Mac output identity",
        )
    audit_meta = manifest.get("mutation_testing", {})
    audit_path = package_dir / audit_meta.get("file", "")
    require(audit_path.is_file() and file_sha256(audit_path) == audit_meta.get("file_sha256"), "negative audit file hash")
    audit = load_json(audit_path)
    require(
        audit.get("schema") == AUDIT_SCHEMA
        and rooted(audit, "receipt_root_sha256") == audit.get("receipt_root_sha256") == audit_meta.get("receipt_root_sha256")
        and audit.get("mutation_count") == audit.get("rejected_count") == audit_meta.get("mutation_count") == 11
        and audit.get("failed_to_reject") == [],
        "negative audit invalid",
    )
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subcommands = result.add_subparsers(dest="command", required=True)
    build = subcommands.add_parser("build")
    build.add_argument("--source-job", type=Path, default=REPOSITORY / "build/fg_rig/fg-default-readback-20260904T095705Z")
    build.add_argument("--rtx-1x1", type=Path, default=REPOSITORY / "build/fg_conv/fg-conv-1x1-family-20260904T103015Z/retrieved")
    build.add_argument("--mac-1x1", type=Path, default=REPOSITORY / "build/fg_conv/macos-1x1-family-v1")
    build.add_argument("--rtx-3x3", type=Path, default=REPOSITORY / "build/fg_conv/fg-conv-3x3-family-20260904T103622Z/retrieved-v2")
    build.add_argument("--mac-3x3", type=Path, default=REPOSITORY / "build/fg_conv/macos-3x3-family-v3")
    build.add_argument("--output-dir", type=Path, required=True)
    verify = subcommands.add_parser("verify")
    verify.add_argument("--package-dir", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        if args.command == "build":
            manifest = build_package(args)
        else:
            manifest = verify_package(args.package_dir.resolve())
        print(f"package_root_sha256={manifest['package_root_sha256']}")
        print(f"package_sha256={manifest['included_state']['package_sha256']}")
        print("result=pass")
        return 0
    except (OSError, KeyError, TypeError, PackageError) as error:
        print(f"package_error={error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
