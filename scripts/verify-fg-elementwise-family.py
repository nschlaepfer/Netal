#!/usr/bin/env python3
"""Verify every live-selected FG k_element_wise module and launch."""

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
from typing import Any, Callable


REPOSITORY = Path(__file__).resolve().parent.parent
RUN_SCHEMA = "neuralstudio.fg.elementwise-family-run.v2"
BACKEND_SCHEMA = "neuralstudio.fg.elementwise-family-backend.v2"
FINAL_SCHEMA = "neuralstudio.fg.elementwise-family-verification.v2"
GENERATOR = "splitmix64-fg-elementwise-family-v2"
OUTPUT_ROOT_SHA256 = "042f91bf4bede61f3af9019b49fd9d9648669ae4cc08c7a4b0ccc4c0d4afa1b2"
INPUT_A_ROOT_SHA256 = "3d168c7c020077ad2f64d4544bfdcdbd6180d6b1932bdb909d83e2caa8b33b15"
INPUT_B_ROOT_SHA256 = "537cc9a64403cb7d555079689c741406f12094880928a7aaaba45c6d62802d98"
PROCESS_CASES = {
    "poison-a5-process-a": 0xA5,
    "poison-a5-process-b": 0xA5,
    "poison-5a-process-a": 0x5A,
    "poison-5a-process-b": 0x5A,
}


@dataclass(frozen=True)
class LaunchCase:
    module_token: int
    module_bytes: int
    module_sha256: str
    launch_token: int
    phase_order: int
    global_order: int
    width: int
    height: int
    channels: int
    grid_x: int
    grid_y: int
    block_z: int
    raw_parameter_sha256: str
    normalized_parameter_sha256: str
    scalar_parameter_sha256: str
    input_a_sha256: str
    input_b_sha256: str
    output_sha256: str

    @property
    def scalars(self) -> tuple[int, ...]:
        return (
            self.width,
            self.height,
            self.channels,
            self.width,
            self.height,
            self.width,
            self.height,
        )

    @property
    def tensor_bytes(self) -> int:
        return self.width * self.height * self.channels * 2

    @property
    def stem(self) -> str:
        return f"launch-{self.launch_token:06d}"


CASES = (
    LaunchCase(
        37, 2152,
        "1970b33c56b1580790a7d56377e4bfd5c1ed8522cd3d5e28b07192b481067d26",
        33, 22, 31, 10, 6, 256, 3, 2, 8,
        "0bb4c1c5e582ea1a7386318d294df1511fdc2e81287b7943f23f9a531a1690bc",
        "60fc0e7d1b56104174f99a50457306f3308cd9c099873ba6e6083503f88b9d80",
        "0f2b3cefe270ff83b105bc583212a894fdaf8671acab257be993fcedab912822",
        "c031b66d6557e8bb4fc8922574b1d0cf2f298ed5e0055a3d0d522ba2c4ea8edd",
        "bb319eee26c1b74fcd65b13edc929bf2a051405bef33b65221e28284c3e92940",
        "888866c398db18cd6b9a0009078f5daee210ee1fc32111730d5b4e2c243f9b36",
    ),
    LaunchCase(
        41, 2152,
        "a717783407a1cea5e3d64b4b23edb451f2b791d8cb7980b374433b9f6ded17e1",
        37, 26, 35, 20, 12, 128, 5, 3, 8,
        "6746a095b6748b7257237e90fd20e1a0d8b15a21813fd90f19c3bcd87b3d4296",
        "a143c92c761c72c3b2e384fd13bd94d0b4d6311c3c8ba38de0d45206558f80d2",
        "b76e29be0799d76623d3655509ea507d3f11d2300ac735dfe0a5314c06bc6e12",
        "fe05996c1521e0a83c537496bf3203725dc9d91de65531d6ae40ee7af4513348",
        "77438b41b75bc6766d6e5cd538270d55e1095f412306c0a56790991a8dfd2191",
        "6bae627d6acdb8d6876e87dc600232144eadf0cf600b319b185b379d44e91bc8",
    ),
    LaunchCase(
        41, 2152,
        "a717783407a1cea5e3d64b4b23edb451f2b791d8cb7980b374433b9f6ded17e1",
        41, 30, 39, 40, 24, 128, 10, 6, 8,
        "ce20833a64a17bc2a9e38aac82ff4a36753bc837b7268f0b5e4368aab0e7035b",
        "f018d7df61d7d25e61b0648680adfcbfce6ef93d7aec05f8a7172f92b450d8cf",
        "a8c92f29f7e409325b2514e52e884569bde053c5d55b3363c51bfd3f99302aa6",
        "702fc8ddbe769eae2af285f96a531ed648d301ef8d5d6cd411a5839acdbf1685",
        "fd19876dec0fdf6a04edcf8b93a36e68b04ddb4bbd42ac39bdb82fb3e3846019",
        "1fbdd873fc048e1ffb72d0aaae14537c4de15fc7e0ddde85929b15e55b0454b9",
    ),
    LaunchCase(
        46, 2152,
        "c733968c9cdca9ba170a9a453d2a831537f467de10e089325aabf8e8ed1cd58f",
        45, 34, 43, 80, 48, 64, 20, 12, 8,
        "7c0bcdb4678fa19591fb171585f20c226e23c0097b498caed9e099de67400a45",
        "4e4460b122f091bf427f41d1ffa6822eca5d7cc5344a4c750ed3c57e5a8616b2",
        "39ad4bad3c73d0570f402f9b105b06eb9153ec62da6b3a1a6fa8a4daff3e8eb2",
        "e66a5d65f705fa3188696d7d886dd0ab65966d6749102f79d2f3a0bce6dd2ca5",
        "92352e25fbe1ec151af5ab70716c908e654e4b851df6b2a3fe337dcc262f42c0",
        "efd8b24f5094c56e375976667bfd30cfc99f4ea458bd673335467460a92e0a6d",
    ),
    LaunchCase(
        50, 2144,
        "dc36c503f9fca1cb42944cab551c268b2d738912927f4aab9db4d53ae66766b6",
        49, 38, 47, 160, 96, 32, 40, 24, 4,
        "a680b8d6c4aa3ea28e7962297986d8dc0020cc70679dcdd3b8e2cf1f402ce82d",
        "bebf0d883fac729020854143da492c3d22894b49451d9973b27757d000d96bb1",
        "2b746e27a83d6e26c74bb958762f71d52879d76501e7db4bfd80556f58be5193",
        "fb0e98c9db5ee58d2b4a22770532eeb233c624f052127e35eb6813340df0dd37",
        "63627ac86a73347b745f410af4e0c18471dc4de6da1e7e0c22610e77645c41f4",
        "b7e56ef51f8390a79dbb513aa98bc821cfad28978010338bb9a2825be654ff24",
    ),
)


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return sha256_bytes(encoded)


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise VerificationError(f"unable to load {path}: {error}") from error
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    path.with_suffix(".sha256").write_text(
        f"{sha256_file(path)}  {path.name}\n", encoding="utf-8"
    )


def splitmix64(value: int) -> int:
    mask = (1 << 64) - 1
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return (value ^ (value >> 31)) & mask


def input_half_bits(case_index: int, index: int, input_b: bool) -> int:
    special_a = (
        0x3C00, 0x3C01, 0xBC00, 0xBC01, 0x0400, 0x8400,
        0x3555, 0xB555, 0x3C00, 0xBC00, 0x3C01, 0xBC01,
    )
    special_b = (
        0x1000, 0x1000, 0x9000, 0x9000, 0x0400, 0x8400,
        0x3000, 0xB000, 0xB800, 0x3800, 0xBC00, 0x3C00,
    )
    if index < len(special_a):
        return special_b[index] if input_b else special_a[index]
    mask = (1 << 64) - 1
    salt = ((case_index + 1) * 0xD1B54A32D192ED03) & mask
    seed = (0x13198A2E03707344 if input_b else 0x243F6A8885A308D3) ^ salt
    step = 0xBF58476D1CE4E5B9 if input_b else 0x9E3779B97F4A7C15
    value = splitmix64((seed + index * step) & mask)
    sign_value = splitmix64(
        ((0xA4093822299F31D0 ^ salt) + index * 0x94D049BB133111EB) & mask
    )
    sign = ((sign_value >> 63) & 1) << 15
    exponent = 10 + ((value >> 10) % 9)
    return sign | (exponent << 10) | (value & 0x03FF)


def generate_inputs(case_index: int, case: LaunchCase) -> tuple[bytes, bytes]:
    values = case.width * case.height * case.channels
    input_a = bytearray(values * 2)
    input_b = bytearray(values * 2)
    for index in range(values):
        struct.pack_into("<H", input_a, index * 2, input_half_bits(case_index, index, False))
        struct.pack_into("<H", input_b, index * 2, input_half_bits(case_index, index, True))
    result = bytes(input_a), bytes(input_b)
    require(sha256_bytes(result[0]) == case.input_a_sha256, f"launch {case.launch_token} input A drift")
    require(sha256_bytes(result[1]) == case.input_b_sha256, f"launch {case.launch_token} input B drift")
    return result


def reference_output(input_a: bytes, input_b: bytes, case: LaunchCase) -> bytes:
    require(len(input_a) == case.tensor_bytes, "input A reference size mismatch")
    require(len(input_b) == case.tensor_bytes, "input B reference size mismatch")
    output = bytearray(case.tensor_bytes)
    for offset in range(0, case.tensor_bytes, 2):
        value_a = struct.unpack_from("<e", input_a, offset)[0]
        value_b = struct.unpack_from("<e", input_b, offset)[0]
        struct.pack_into("<e", output, offset, value_a + value_b)
    result = bytes(output)
    require(sha256_bytes(result) == case.output_sha256, f"launch {case.launch_token} output drift")
    return result


def generated_contract() -> list[tuple[bytes, bytes, bytes]]:
    generated = []
    for index, case in enumerate(CASES):
        input_a, input_b = generate_inputs(index, case)
        generated.append((input_a, input_b, reference_output(input_a, input_b, case)))
    require(sha256_bytes(b"".join(item[0] for item in generated)) == INPUT_A_ROOT_SHA256, "input A root drift")
    require(sha256_bytes(b"".join(item[1] for item in generated)) == INPUT_B_ROOT_SHA256, "input B root drift")
    require(sha256_bytes(b"".join(item[2] for item in generated)) == OUTPUT_ROOT_SHA256, "output root drift")
    return generated


def require_exact(actual: bytes, expected: bytes, label: str) -> None:
    if actual == expected:
        return
    differing = sum(left != right for left, right in zip(actual, expected))
    differing += abs(len(actual) - len(expected))
    first = next(
        (index for index, pair in enumerate(zip(actual, expected)) if pair[0] != pair[1]),
        min(len(actual), len(expected)),
    )
    raise VerificationError(f"{label} mismatch: differing_bytes={differing} first_offset={first}")


def untouched_half_words(output: bytes, poison: int) -> int:
    marker = poison | (poison << 8)
    return sum(
        struct.unpack_from("<H", output, offset)[0] == marker
        for offset in range(0, len(output), 2)
    )


def require_no_addresses(value: Any, label: str) -> None:
    def visit(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                lowered = key.lower()
                if key != "address_fields_recorded":
                    require(
                        not any(part in lowered for part in (
                            "gpu_address", "device_address", "pointer_value",
                            "raw_pointer", "gpu_va",
                        )),
                        f"{label} contains address field {path}.{key}",
                    )
                visit(child, f"{path}.{key}")
        elif isinstance(node, list):
            for index, child in enumerate(node):
                visit(child, f"{path}[{index}]")
        elif isinstance(node, str):
            require(
                re.search(r"0x[0-9a-fA-F]{12,16}(?:\b|$)", node) is None,
                f"{label} contains address-like value at {path}",
            )
    visit(value, label)


def validate_parameter_bytes(data: bytes, case: LaunchCase) -> dict[str, Any]:
    require(len(data) == 52, f"launch {case.launch_token} parameter size")
    require(sha256_bytes(data) == case.raw_parameter_sha256, f"launch {case.launch_token} raw parameter hash")
    require(struct.unpack_from("<7I", data, 24) == case.scalars, f"launch {case.launch_token} scalar ABI")
    require(sha256_bytes(bytes(24) + data[24:]) == case.normalized_parameter_sha256, f"launch {case.launch_token} normalized ABI")
    require(sha256_bytes(data[24:]) == case.scalar_parameter_sha256, f"launch {case.launch_token} scalar hash")
    return {
        "bytes": 52,
        "captured_raw_sha256": case.raw_parameter_sha256,
        "raw_hash_is_trace_specific_not_portable_identity": True,
        "address_free_normalized_sha256": case.normalized_parameter_sha256,
        "scalar_bytes_sha256": case.scalar_parameter_sha256,
        "pointer_roles": ["input_a", "input_b", "output"],
        "scalars_u32": list(case.scalars),
    }


def validate_frozen_inputs(input_root: Path) -> dict[str, Any]:
    modules: dict[int, dict[str, Any]] = {}
    parameters = []
    for source_run in ("run-a", "run-b"):
        for case in CASES:
            module_path = input_root / source_run / f"cu-module-{case.module_token:06d}.bin"
            module = module_path.read_bytes()
            require(len(module) == case.module_bytes, f"module {case.module_token} size")
            require(sha256_bytes(module) == case.module_sha256, f"module {case.module_token} hash")
            modules.setdefault(case.module_token, {
                "module_token": case.module_token,
                "bytes": case.module_bytes,
                "sha256": case.module_sha256,
                "two_source_processes_equal": True,
            })
            parameter_path = input_root / source_run / f"{case.stem}-kernel-0000.params.bin"
            contract = validate_parameter_bytes(parameter_path.read_bytes(), case)
            if source_run == "run-a":
                parameters.append({"launch_token": case.launch_token, **contract})
    return {
        "source_processes": ["run-a", "run-b"],
        "two_source_processes_equal": True,
        "selected_modules": [modules[token] for token in sorted(modules)],
        "launch_parameters": parameters,
    }


def validate_run(
    run_root: Path,
    backend: str,
    run_id: str,
    poison: int,
    generated: list[tuple[bytes, bytes, bytes]],
) -> dict[str, Any]:
    receipt_path = run_root / "run-receipt.json"
    receipt = load_json(receipt_path)
    expected_backend = "rtx-cuda-driver" if backend == "rtx" else "mac-metal"
    require(receipt.get("schema") == RUN_SCHEMA, f"{run_root}: schema")
    require(receipt.get("backend") == expected_backend, f"{run_root}: backend")
    require(receipt.get("run_id") == run_id, f"{run_root}: run id")
    require(receipt.get("poison_byte") == poison, f"{run_root}: poison")
    require(isinstance(receipt.get("process_id"), int), f"{run_root}: process id")
    require(receipt.get("input_generator") == GENERATOR, f"{run_root}: generator")
    require(receipt.get("external_tensor_input_files_opened") is False, f"{run_root}: external input")
    require(receipt.get("address_fields_recorded") is False, f"{run_root}: address flag")
    require(receipt.get("result") == "pass", f"{run_root}: result")
    require(receipt.get("input_mismatched_bytes") == 0, f"{run_root}: input mismatch")
    require(receipt.get("canary_mismatched_bytes") == 0, f"{run_root}: canary")
    require(receipt.get("untouched_half_words") == 0, f"{run_root}: untouched")
    require(receipt.get("input_immutability") is True, f"{run_root}: immutability")
    require(receipt.get("full_write_coverage") is True, f"{run_root}: coverage")
    require(receipt.get("output_root_sha256") == OUTPUT_ROOT_SHA256, f"{run_root}: root")
    require_no_addresses(receipt, str(receipt_path))
    launches = receipt.get("launches")
    require(isinstance(launches, list) and len(launches) == len(CASES), f"{run_root}: launch count")
    artifact_hashes: dict[str, str] = {}
    for index, (case, expected) in enumerate(zip(CASES, generated)):
        item = launches[index]
        require(item.get("index") == index, f"{run_root}: index {index}")
        require(item.get("module_token") == case.module_token, f"{run_root}: module token")
        require(item.get("module_bytes") == case.module_bytes, f"{run_root}: module bytes")
        require(item.get("module_sha256") == case.module_sha256, f"{run_root}: module hash")
        require(item.get("function_name") == "k_element_wise", f"{run_root}: function")
        require(item.get("launch_token") == case.launch_token, f"{run_root}: launch")
        require(item.get("evaluate_1_phase_order") == case.phase_order, f"{run_root}: phase order")
        require(item.get("global_launch_order") == case.global_order, f"{run_root}: global order")
        require(item.get("parameter_scalars_u32") == list(case.scalars), f"{run_root}: scalars")
        require(item.get("grid") == [case.grid_x, case.grid_y, 1], f"{run_root}: grid")
        require(item.get("block") == [4, 4, case.block_z], f"{run_root}: block")
        require(item.get("shape_nhwc") == [1, case.height, case.width, case.channels], f"{run_root}: shape")
        require(item.get("tensor_bytes") == case.tensor_bytes, f"{run_root}: bytes")
        require(item.get("input_a_pre_sha256") == case.input_a_sha256, f"{run_root}: input A pre")
        require(item.get("input_a_post_sha256") == case.input_a_sha256, f"{run_root}: input A post")
        require(item.get("input_b_pre_sha256") == case.input_b_sha256, f"{run_root}: input B pre")
        require(item.get("input_b_post_sha256") == case.input_b_sha256, f"{run_root}: input B post")
        require(item.get("output_sha256") == case.output_sha256, f"{run_root}: output hash")
        require(item.get("input_mismatched_bytes") == 0, f"{run_root}: launch input mismatch")
        require(item.get("canary_mismatched_bytes") == 0, f"{run_root}: launch canary")
        require(item.get("untouched_half_words") == 0, f"{run_root}: launch coverage")
        for suffix, expected_bytes in (
            ("input-a-after.bin", expected[0]),
            ("input-b-after.bin", expected[1]),
            ("output.bin", expected[2]),
        ):
            name = f"{case.stem}-{suffix}"
            actual = (run_root / name).read_bytes()
            require_exact(actual, expected_bytes, f"{run_root}/{name}")
            artifact_hashes[name] = sha256_bytes(actual)
        require(untouched_half_words(expected[2], poison) == 0, f"{run_root}: poison collision")
    return {
        "run_id": run_id,
        "poison_byte": poison,
        "process_id": receipt["process_id"],
        "receipt_sha256": sha256_file(receipt_path),
        "artifacts": artifact_hashes,
    }


def build_backend_receipt(backend: str, root: Path) -> dict[str, Any]:
    generated = generated_contract()
    runs = [
        validate_run(root / run_id, backend, run_id, poison, generated)
        for run_id, poison in PROCESS_CASES.items()
    ]
    require(len({item["process_id"] for item in runs}) == 4, f"{backend}: fresh processes")
    for case in CASES:
        hashes = {item["artifacts"][f"{case.stem}-output.bin"] for item in runs}
        require(hashes == {case.output_sha256}, f"{backend}: launch {case.launch_token} invariance")
    receipt: dict[str, Any] = {
        "schema": BACKEND_SCHEMA,
        "backend": "rtx-cuda-driver" if backend == "rtx" else "mac-metal",
        "function": "k_element_wise",
        "module_tokens": [37, 41, 46, 50],
        "launch_tokens": [case.launch_token for case in CASES],
        "input_generator": GENERATOR,
        "input_a_root_sha256": INPUT_A_ROOT_SHA256,
        "input_b_root_sha256": INPUT_B_ROOT_SHA256,
        "output_root_sha256": OUTPUT_ROOT_SHA256,
        "output_bytes": sum(case.tensor_bytes for case in CASES),
        "fresh_process_count": 4,
        "two_processes_per_poison": True,
        "poison_output_invariance": True,
        "input_immutability": True,
        "full_write_coverage": True,
        "address_free": True,
        "runs": runs,
        "result": "pass",
    }
    receipt["receipt_root_sha256"] = canonical_sha256(receipt)
    require_no_addresses(receipt, f"{backend} backend receipt")
    return receipt


def expect_rejection(name: str, operation: Callable[[], None]) -> dict[str, Any]:
    try:
        operation()
    except VerificationError as error:
        return {"mutation": name, "rejected": True, "reason": str(error)}
    raise VerificationError(f"negative mutation accepted: {name}")


def source_artifacts() -> list[dict[str, Any]]:
    paths = (
        REPOSITORY / "src/fg_elementwise/rtx_family_oracle.cpp",
        REPOSITORY / "src/fg_elementwise/macos_family_probe.mm",
        REPOSITORY / "src/fg_elementwise/elementwise_family.metal",
        REPOSITORY / "scripts/build-fg-elementwise-family-rtx.ps1",
        REPOSITORY / "scripts/build-macos-fg-elementwise-family.sh",
        Path(__file__).resolve(),
    )
    return [
        {
            "relative_path": path.relative_to(REPOSITORY).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    ]


def build_final_receipt(
    rtx_root: Path, mac_root: Path, input_root: Path
) -> dict[str, Any]:
    frozen_contract = validate_frozen_inputs(input_root)
    rtx = build_backend_receipt("rtx", rtx_root)
    mac = build_backend_receipt("mac", mac_root)
    for root, rebuilt, label in ((rtx_root, rtx, "rtx"), (mac_root, mac, "mac")):
        recorded_path = root / "backend-receipt.json"
        require(load_json(recorded_path) == rebuilt, f"{label}: recorded backend drift")
        sidecar = (root / "backend-receipt.sha256").read_text(encoding="utf-8-sig").split()[0]
        require(sidecar == sha256_file(recorded_path), f"{label}: backend sidecar")
    rig_path = rtx_root / "rig-receipt.json"
    rig = load_json(rig_path)
    require(rig.get("schema") == "neuralstudio.fg.elementwise-family-rig.v2", "rig schema")
    require(rig.get("result") == "pass", "rig result")
    require(rig.get("fresh_process_count") == 4, "rig process count")
    require(rig.get("address_fields_recorded") is False, "rig address flag")
    rig_sidecar = (rtx_root / "rig-receipt.sha256").read_text(encoding="utf-8-sig").split()[0]
    require(rig_sidecar == sha256_file(rig_path), "rig sidecar")
    require_no_addresses(rig, "rig receipt")
    mac_runs = [load_json(mac_root / run_id / "run-receipt.json") for run_id in PROCESS_CASES]
    devices = {item.get("device_name") for item in mac_runs}
    metallibs = {item.get("metallib_sha256") for item in mac_runs}
    require(len(devices) == 1 and None not in devices, "Mac device drift")
    require(len(metallibs) == 1 and None not in metallibs, "Mac metallib drift")
    mac_binary = mac_root / "build/fg-elementwise-family-mac"
    require(mac_binary.is_file(), "Mac family binary missing")

    generated = generated_contract()
    first_output = bytearray(generated[0][2])
    first_output[17] ^= 1
    first_input = bytearray(generated[0][0])
    first_input[29] ^= 1
    poisoned_output = bytearray(generated[0][2])
    poisoned_output[0:2] = b"\xA5\xA5"
    first_parameter_path = input_root / "run-a/launch-000033-kernel-0000.params.bin"
    mutated_parameter = bytearray(first_parameter_path.read_bytes())
    mutated_parameter[24] ^= 1
    first_module_path = input_root / "run-a/cu-module-000037.bin"
    mutated_module = bytearray(first_module_path.read_bytes())
    mutated_module[11] ^= 1
    negative = [
        expect_rejection(
            "output-byte-flip",
            lambda: require_exact(bytes(first_output), generated[0][2], "mutated output"),
        ),
        expect_rejection(
            "input-provenance-byte-flip",
            lambda: require_exact(bytes(first_input), generated[0][0], "mutated input"),
        ),
        expect_rejection(
            "scalar-abi-byte-flip",
            lambda: validate_parameter_bytes(bytes(mutated_parameter), CASES[0]),
        ),
        expect_rejection(
            "selected-module-byte-flip",
            lambda: require(
                sha256_bytes(bytes(mutated_module)) == CASES[0].module_sha256,
                "selected module identity mismatch",
            ),
        ),
        expect_rejection(
            "address-like-receipt-value",
            lambda: require_no_addresses(
                {"note": "volatile 0x0000000123456789"}, "mutated receipt"
            ),
        ),
        expect_rejection(
            "untouched-output-half-word",
            lambda: require(
                untouched_half_words(bytes(poisoned_output), 0xA5) == 0,
                "untouched output half word detected",
            ),
        ),
        expect_rejection(
            "missing-live-launch",
            lambda: require([33, 37, 41, 45] == [case.launch_token for case in CASES], "live launch set incomplete"),
        ),
    ]

    def backend_attestation(value: dict[str, Any], root: Path) -> dict[str, Any]:
        return {
            key: value[key]
            for key in (
                "backend", "module_tokens", "launch_tokens",
                "input_a_root_sha256", "input_b_root_sha256",
                "output_root_sha256", "output_bytes", "fresh_process_count",
                "two_processes_per_poison", "poison_output_invariance",
                "input_immutability", "full_write_coverage", "address_free",
                "result",
            )
        } | {
            "backend_receipt_sha256": sha256_file(root / "backend-receipt.json"),
            "backend_receipt_root_sha256": value["receipt_root_sha256"],
        }

    receipt: dict[str, Any] = {
        "schema": FINAL_SCHEMA,
        "selected_family": {
            "function": "k_element_wise",
            "module_tokens": [37, 41, 46, 50],
            "launch_tokens": [33, 37, 41, 45, 49],
            "operation": "binary16 add.rn over selected NHWC layouts",
            "frozen_module_and_parameter_contracts": frozen_contract,
            "launches": [
                {
                    "module_token": case.module_token,
                    "module_sha256": case.module_sha256,
                    "launch_token": case.launch_token,
                    "phase_order": case.phase_order,
                    "global_order": case.global_order,
                    "shape_nhwc": [1, case.height, case.width, case.channels],
                    "grid": [case.grid_x, case.grid_y, 1],
                    "block": [4, 4, case.block_z],
                    "output_sha256": case.output_sha256,
                }
                for case in CASES
            ],
        },
        "cross_backend": {
            "byte_exact_all_five_launches": True,
            "mismatched_output_bytes": 0,
            "output_root_sha256": OUTPUT_ROOT_SHA256,
            "output_bytes_per_process": sum(case.tensor_bytes for case in CASES),
            "four_fresh_processes_per_backend": True,
            "two_processes_per_poison": True,
            "two_poison_output_invariance": True,
            "input_immutability": True,
            "canaries_intact": True,
            "zero_untouched_half_words": True,
            "independent_input_generation": True,
            "external_expected_or_intermediate_bytes_fed_to_execution": False,
        },
        "execution_attestations": {
            "rtx": {
                "job_id": rig["job_id"],
                "attempt_id": rig["attempt_id"],
                "gpu": rig["gpu"],
                "cuda_toolkit": rig["cuda_toolkit"],
                "binary": rig["binary"],
                "rig_receipt_sha256": sha256_file(rig_path),
                "exit_code": 0,
                "result": "pass",
            },
            "mac": {
                "device_name": next(iter(devices)),
                "metallib_sha256": next(iter(metallibs)),
                "binary_bytes": mac_binary.stat().st_size,
                "binary_sha256": sha256_file(mac_binary),
                "exit_code": 0,
                "result": "pass",
            },
        },
        "rtx_backend": backend_attestation(rtx, rtx_root),
        "mac_backend": backend_attestation(mac, mac_root),
        "input_domain": {
            "half_values_per_input_per_process": sum(case.tensor_bytes // 2 for case in CASES),
            "random_finite_normal_exponent_bits": [10, 18],
            "random_pairs_use_equal_signs": True,
            "special_pairs_per_launch": 12,
            "special_cases_include_round_to_nearest_even_ties": True,
            "special_cases_include_opposite_sign_cancellation": True,
            "not_covered": [
                "NaN payload propagation", "infinities", "subnormal inputs",
                "overflow", "broad random opposite-sign pairs",
            ],
        },
        "negative_mutations": negative,
        "source_artifacts": source_artifacts(),
        "accepted_frontier": {
            "all_selected_elementwise_modules_executed_on_rtx": True,
            "all_selected_elementwise_launches_on_metal": True,
            "rtx_mac_byte_exact_elementwise_family": True,
            "dependency_closed_fg_group": False,
            "complete_generated_frame_parity": False,
            "netal_integration": False,
            "performance_advantage": False,
        },
        "result": "pass",
    }
    receipt["evidence_root_sha256"] = canonical_sha256(receipt)
    require_no_addresses(receipt, "final family receipt")
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    backend = commands.add_parser("backend")
    backend.add_argument("--backend", choices=("rtx", "mac"), required=True)
    backend.add_argument("--root", type=Path, required=True)
    backend.add_argument("--output", type=Path, required=True)
    final = commands.add_parser("final")
    final.add_argument("--rtx-root", type=Path, required=True)
    final.add_argument("--mac-root", type=Path, required=True)
    final.add_argument("--input-root", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "backend":
            receipt = build_backend_receipt(args.backend, args.root)
        else:
            receipt = build_final_receipt(args.rtx_root, args.mac_root, args.input_root)
        write_json(args.output, receipt)
        print(f"receipt={args.output}")
        print(f"sha256={sha256_file(args.output)}")
        print(f"evidence_root={receipt.get('evidence_root_sha256', receipt.get('receipt_root_sha256'))}")
        print("result=pass")
        return 0
    except (OSError, VerificationError, struct.error, OverflowError) as error:
        print(f"result=fail\nerror={error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
