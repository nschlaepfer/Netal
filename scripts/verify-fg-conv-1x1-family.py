#!/usr/bin/env python3
"""Per-module and aggregate verification for selected FG 1x1 convolutions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MANIFEST_SHA256 = "2a1179d4a25be030f772cd90ca7d017ed6eac9291917f7d43ea9b0cd5ac6a294"
RUN_SCHEMA = "neuralstudio.fg.conv-1x1-run.v1"
BACKEND_SCHEMA = "neuralstudio.fg.conv-1x1-backend.v1"
FINAL_SCHEMA = "neuralstudio.fg.conv-1x1-verification.v1"
FAMILY_SCHEMA = "neuralstudio.fg.conv-1x1-family.v1"
GENERATOR = "splitmix64-fg-conv-1x1-family-v1"


@dataclass(frozen=True)
class Layer:
    id: str
    launch: int
    module_token: int
    module_bytes: int
    module_sha256: str
    weight_token: int
    weight_bytes: int
    weight_sha256: str
    width: int
    height: int
    input_channels: int
    output_channels: int
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    leaky_relu: bool


def layer(*values: Any) -> Layer:
    return Layer(*values)


LAYERS = {
    item.id: item
    for item in (
        layer("l22", 22, 30, 4952, "31bb08d04011db9a0def8f1a1bc3999efc08d5deb8d7cb9b5debf52ec489b785", 65, 10240, "eb99328524fddc9f5f3fac12b8e3a4011e66a75e071cd07126a70c553458bc60", 80, 48, 32, 64, (5, 24, 1), (32, 2, 2), True),
        layer("l30", 30, 34, 11328, "2f490857c6f7aa22f2435057706ec10b28b6b8d3081bd967ebe1d61550ac6419", 73, 540672, "e23316a86707c1d3d377aa0711df47d05976ad1a30f6428524d030dd17145de9", 5, 3, 256, 512, (1, 1, 32), (32, 4, 1), True),
        layer("l31", 31, 35, 19424, "331853771cae4aa041038d8e9c74dcb1f8d0d38b4d5603e2e8d079ad1e44fd04", 75, 532480, "9f20d6589cc9269ad020233e68196d66eba0e8a76497374cf6d445385f94dbc4", 5, 3, 512, 256, (1, 3, 16), (32, 1, 1), True),
        layer("l34", 34, 38, 11320, "8e7abed27b7eae0dd274b2409a13d6dde98277d9e743f0e36a1042022a16ceda", 77, 270336, "9e5da09b93650c74e89d5262aa8a9a6a3dcbd0d753dc8c9d9aed39731e39b2a0", 10, 6, 256, 256, (1, 2, 16), (32, 4, 1), True),
        layer("l35", 35, 39, 11328, "939d5d9cf7012fb3ed3ce4e4792ccf5c6befec165ae7f7a0d3704cb1103c3030", 79, 135168, "3969384493583e98dcf357d3046a10a5b836585261f3e584605bf11e23fb8d46", 10, 6, 256, 128, (1, 2, 8), (32, 4, 1), True),
        layer("l38", 38, 42, 7368, "93f3bb1be638eaf8003ba04cc7977859dfb16ae7821801c3017e45546301ecbc", 81, 69632, "a0c6690faee2838521c6ad0f9445425d639446c0ef1b7fed9b43d21e7a8ec650", 20, 12, 128, 128, (2, 3, 8), (32, 4, 1), True),
        layer("l39", 39, 42, 7368, "93f3bb1be638eaf8003ba04cc7977859dfb16ae7821801c3017e45546301ecbc", 83, 69632, "3a13970b482f1f78ede1bfe531956c203f08f99077c81bda2d1e4d0f453cf9a0", 20, 12, 128, 128, (2, 3, 8), (32, 4, 1), True),
        layer("l42", 42, 43, 8440, "53fdf8d5404c93f2fe64582f271192df3b24c2ef08d9564ffeb9d7990485db0e", 85, 69632, "47361a699fa6514332dc560cf2416872256e61d1b36b1ca29df323a53ee2be77", 40, 24, 128, 128, (3, 24, 1), (32, 1, 4), True),
        layer("l43", 43, 44, 8496, "7113dfaae4945541276b3e662cba366f9a3fd9666db7a338efa867cfd7f46e31", 87, 34816, "327bd9b02da4aaac8b251320f60b4aeccb0df7df84bba44d7f1f6a9fdf62c934", 40, 24, 128, 64, (3, 24, 2), (32, 1, 1), True),
        layer("l46", 46, 47, 6088, "8a3efc1e9166ed3ee4a54aea535910c0562b6753b2d0daae0402e1ed36ff19dc", 89, 18432, "e24f67110bab401ab7f70a861f3f09d2cd3cd6abf8d1aa158ba3300efb64797e", 80, 48, 64, 64, (5, 24, 1), (32, 2, 2), True),
        layer("l47", 47, 48, 5992, "11906664ac548c61262a12b0950bf135926cf269de2d7691d0edfbe4a95e63dd", 91, 9216, "325d7dfac5620a89edc1d2315731414cf755c50506ea5758a8c3cf563649577c", 80, 48, 64, 32, (5, 12, 1), (32, 4, 1), True),
        layer("l50", 50, 51, 4840, "304d0c0e58958fa7eab6a82fd046ff689c51a986aff88a069f6f509c434a5b96", 93, 5120, "bb98eb15be7d255a15a46994589705560e0691e273e286a36a68fd679a1890a9", 160, 96, 32, 32, (10, 24, 1), (32, 4, 1), True),
        layer("l51", 51, 52, 3912, "27486bcc1db8b04b0e18d72f33b06662da9c327eb15b41a09b3f909a3a52cad2", 95, 2560, "477c68ebace499e2e1ac331f214fae3e9fb4df815a9e7d8142a0f1fb414b340e", 160, 96, 32, 16, (10, 24, 1), (32, 4, 1), False),
    )
}


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    require(isinstance(value, dict), f"not an object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def require_address_free(value: Any, label: str) -> None:
    def visit(node: Any, location: str) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                if key != "address_fields_recorded":
                    require(not any(part in key.lower() for part in ("pointer", "virtual_address", "device_address", "gpu_address")), f"{label}: address key {location}.{key}")
                visit(child, f"{location}.{key}")
        elif isinstance(node, list):
            for index, child in enumerate(node):
                visit(child, f"{location}[{index}]")
        elif isinstance(node, str):
            require(re.search(r"0x[0-9a-fA-F]{12,16}(?:\b|$)", node) is None, f"{label}: address text {location}")
    visit(value, label)


def splitmix64(value: int) -> int:
    mask = (1 << 64) - 1
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return (value ^ (value >> 31)) & mask


def generated_input(config: Layer) -> bytes:
    values = (0x0000, 0x2800, 0xA800, 0x2C00, 0xAC00, 0x3000, 0xB000, 0x3400, 0xB400, 0x3800, 0xB800)
    elements = config.width * config.height * config.input_channels
    result = bytearray(elements * 2)
    seed = 0xA0761D6478BD642F ^ (config.launch * 0xE7037ED1A0B428DB)
    step = 0x8EBC6AF09C88C6E3
    mask = (1 << 64) - 1
    for index in range(elements):
        struct.pack_into("<H", result, index * 2, values[splitmix64((seed + index * step) & mask) % len(values)])
    return bytes(result)


def validate_backend(backend: str, config: Layer, root: Path) -> dict[str, Any]:
    expected_backend = "rtx-cuda-driver" if backend == "rtx" else "mac-metal"
    input_sha = sha256_bytes(generated_input(config))
    output_bytes = config.width * config.height * config.output_channels * 2
    receipts: list[dict[str, Any]] = []
    outputs: list[bytes] = []
    for poison in (0xA5, 0x5A):
        for ordinal in ("a", "b"):
            run_id = f"{config.id}-poison-{poison:02x}-process-{ordinal}"
            run_root = root / run_id
            receipt = load_json(run_root / "run-receipt.json")
            output = (run_root / "output.bin").read_bytes()
            require(receipt.get("schema") == RUN_SCHEMA, f"{run_id}: schema")
            require(receipt.get("backend") == expected_backend, f"{run_id}: backend")
            require(receipt.get("layer_id") == config.id and receipt.get("run_id") == run_id, f"{run_id}: identity")
            require(receipt.get("poison_byte") == poison, f"{run_id}: poison")
            require(isinstance(receipt.get("process_id"), int), f"{run_id}: pid")
            require(receipt.get("module_token") == config.module_token, f"{run_id}: module token")
            require(receipt.get("module_bytes") == config.module_bytes, f"{run_id}: module bytes")
            require(receipt.get("module_sha256") == config.module_sha256, f"{run_id}: module hash")
            require(receipt.get("trace_launch_token") == f"launch-{config.launch:06d}", f"{run_id}: launch")
            require(receipt.get("packed_parameter_bytes") == 40, f"{run_id}: params")
            require(receipt.get("abi_scalars") == [config.width, config.height, config.width, config.height], f"{run_id}: ABI")
            require(receipt.get("input_shape_nhwc") == [1, config.height, config.width, config.input_channels], f"{run_id}: input shape")
            require(receipt.get("output_shape_nhwc") == [1, config.height, config.width, config.output_channels], f"{run_id}: output shape")
            require(receipt.get("leaky_relu") is config.leaky_relu, f"{run_id}: activation")
            require(receipt.get("weight_source_token") == config.weight_token, f"{run_id}: weight token")
            require(receipt.get("weight_capture_manifest_sha256") == MANIFEST_SHA256, f"{run_id}: manifest")
            require(receipt.get("weight_pre_sha256") == config.weight_sha256 == receipt.get("weight_post_sha256"), f"{run_id}: weights")
            require(receipt.get("input_generator") == GENERATOR, f"{run_id}: generator")
            require(receipt.get("input_pre_sha256") == input_sha == receipt.get("input_post_sha256"), f"{run_id}: input")
            require(len(output) == output_bytes and receipt.get("output_bytes") == output_bytes, f"{run_id}: output size")
            require(receipt.get("output_sha256") == sha256_bytes(output), f"{run_id}: output hash")
            require(receipt.get("input_mismatched_bytes") == 0 and receipt.get("canary_mismatched_bytes") == 0, f"{run_id}: mutation/canary")
            require(receipt.get("per_run_write_coverage_conclusive") is False, f"{run_id}: coverage overclaim")
            require(receipt.get("external_weight_file_opened") is True and receipt.get("expected_output_file_opened") is False, f"{run_id}: execution provenance")
            require(receipt.get("address_fields_recorded") is False and receipt.get("result") == "pass", f"{run_id}: result")
            if backend == "rtx":
                require(receipt.get("grid") == list(config.grid) and receipt.get("block") == list(config.block), f"{run_id}: RTX geometry")
                require(receipt.get("function_name") == "k_conv_fp16_nhwc", f"{run_id}: RTX function")
            else:
                require(receipt.get("selected_rtx_grid") == list(config.grid) and receipt.get("selected_rtx_block") == list(config.block), f"{run_id}: selected geometry")
                require(receipt.get("metal_dispatch_threads") == [config.width, config.height, config.output_channels], f"{run_id}: dispatch")
                require(receipt.get("metal_threads_per_threadgroup") == [8, 4, 1] and receipt.get("metal_function") == "fg_conv_1x1", f"{run_id}: Metal function")
            require_address_free(receipt, run_id)
            receipts.append(receipt); outputs.append(output)
    require(len({r["process_id"] for r in receipts}) == 4, f"{config.id}: fresh process gate")
    require(len({sha256_bytes(o) for o in outputs}) == 1, f"{config.id}: two-poison/process invariance")
    mutated = bytearray(outputs[0]); mutated[config.launch % len(mutated)] ^= 1
    require(sha256_bytes(mutated) != receipts[0]["output_sha256"], f"{config.id}: mutation undetected")
    result = {
        "schema": BACKEND_SCHEMA,
        "backend": expected_backend,
        "layer_id": config.id,
        "trace_launch_token": f"launch-{config.launch:06d}",
        "module_token": config.module_token,
        "module_sha256": config.module_sha256,
        "weight_source_token": config.weight_token,
        "weight_sha256": config.weight_sha256,
        "weight_capture_manifest_sha256": MANIFEST_SHA256,
        "input_generator": GENERATOR,
        "input_sha256": input_sha,
        "output_sha256": sha256_bytes(outputs[0]),
        "output_bytes": output_bytes,
        "fresh_process_count": 4,
        "two_process_two_poison_invariant": True,
        "full_write_coverage_from_two_poison_invariance": True,
        "input_weight_immutability": True,
        "canary_integrity": True,
        "expected_output_fed_to_execution": False,
        "output_mutation_rejected": True,
        "address_fields_recorded": False,
        "result": "pass",
    }
    result["receipt_root_sha256"] = canonical_sha256(result)
    return result


def validate_final(rtx_path: Path, mac_path: Path, config: Layer) -> dict[str, Any]:
    rtx = load_json(rtx_path); mac = load_json(mac_path)
    for receipt, backend in ((rtx, "rtx-cuda-driver"), (mac, "mac-metal")):
        require(receipt.get("schema") == BACKEND_SCHEMA and receipt.get("backend") == backend, f"{config.id}/{backend}: identity")
        without_root = dict(receipt); root = without_root.pop("receipt_root_sha256", None)
        require(root == canonical_sha256(without_root), f"{config.id}/{backend}: root")
        require_address_free(receipt, f"{config.id}/{backend}")
    require(rtx["layer_id"] == mac["layer_id"] == config.id, f"{config.id}: layer")
    require(rtx["output_sha256"] == mac["output_sha256"], f"{config.id}: RTX/Mac mismatch")
    result = {
        "schema": FINAL_SCHEMA,
        "layer_id": config.id,
        "trace_launch_token": f"launch-{config.launch:06d}",
        "module_sha256": config.module_sha256,
        "weight_sha256": config.weight_sha256,
        "input_sha256": rtx["input_sha256"],
        "output_sha256": rtx["output_sha256"],
        "output_bytes": rtx["output_bytes"],
        "rtx_backend_root_sha256": rtx["receipt_root_sha256"],
        "mac_backend_root_sha256": mac["receipt_root_sha256"],
        "rtx_mac_differing_bytes": 0,
        "rtx_mac_exact": True,
        "scope": "one selected 1x1 convolution; no family or multi-kernel group claim",
        "address_fields_recorded": False,
        "result": "pass",
    }
    result["receipt_root_sha256"] = canonical_sha256(result)
    return result


def validate_family(receipt_dir: Path, layer_ids: list[str]) -> dict[str, Any]:
    require(len(layer_ids) == len(set(layer_ids)), "duplicate family layer id")
    members = []
    for layer_id in layer_ids:
        config = LAYERS[layer_id]
        receipt = load_json(receipt_dir / f"{layer_id}-verification.json")
        require(receipt.get("schema") == FINAL_SCHEMA and receipt.get("layer_id") == layer_id, f"{layer_id}: final receipt")
        without_root = dict(receipt); root = without_root.pop("receipt_root_sha256", None)
        require(root == canonical_sha256(without_root), f"{layer_id}: final root")
        require(receipt.get("rtx_mac_exact") is True and receipt.get("result") == "pass", f"{layer_id}: parity")
        require(receipt.get("module_sha256") == config.module_sha256, f"{layer_id}: module")
        members.append({"layer_id": layer_id, "receipt_root_sha256": root, "output_sha256": receipt["output_sha256"]})
    result = {
        "schema": FAMILY_SCHEMA,
        "layer_count": len(members),
        "layers": members,
        "every_layer_executed_independently": True,
        "every_layer_rtx_mac_exact": True,
        "family_inference_from_shape_only": False,
        "address_fields_recorded": False,
        "result": "pass",
    }
    result["receipt_root_sha256"] = canonical_sha256(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(); subs = parser.add_subparsers(dest="command", required=True)
    backend = subs.add_parser("backend"); backend.add_argument("--backend", choices=("rtx", "mac"), required=True); backend.add_argument("--layer", choices=tuple(LAYERS), required=True); backend.add_argument("--root", type=Path, required=True); backend.add_argument("--output", type=Path, required=True)
    final = subs.add_parser("final"); final.add_argument("--layer", choices=tuple(LAYERS), required=True); final.add_argument("--rtx", type=Path, required=True); final.add_argument("--mac", type=Path, required=True); final.add_argument("--output", type=Path, required=True)
    family = subs.add_parser("family"); family.add_argument("--receipt-dir", type=Path, required=True); family.add_argument("--layers", nargs="+", choices=tuple(LAYERS), required=True); family.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "backend": result = validate_backend(args.backend, LAYERS[args.layer], args.root)
        elif args.command == "final": result = validate_final(args.rtx, args.mac, LAYERS[args.layer])
        else: result = validate_family(args.receipt_dir, args.layers)
        write_json(args.output, result)
        print(f"receipt_root_sha256={result['receipt_root_sha256']}"); print("result=pass")
        return 0
    except (OSError, json.JSONDecodeError, KeyError, VerificationError) as error:
        print(f"verification_error={error}", file=sys.stderr); return 1


if __name__ == "__main__":
    raise SystemExit(main())
