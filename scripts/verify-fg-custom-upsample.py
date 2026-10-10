#!/usr/bin/env python3
"""Verify exact RTX/Metal parity for FG custom three-plane upsample.

Both executables generate inputs independently.  This verifier recomputes the
generator, validates all live artifacts, proves two-poison write coverage, and
compares outputs.  It never supplies an expected output to either backend.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import struct
from pathlib import Path
from typing import Any, Callable


MASK64 = (1 << 64) - 1
RUN_SCHEMA = "neuralstudio.fg.custom-upsample-run.v1"
BACKEND_SCHEMA = "neuralstudio.fg.custom-upsample-backend.v1"
RIG_SCHEMA = "neuralstudio.fg.custom-upsample-rig.v1"
CUMETAL_SCHEMA = "neuralstudio.fg.custom-upsample-cumetal-failure.v1"
FINAL_SCHEMA = "neuralstudio.fg.custom-upsample-verification.v1"
NEGATIVE_SCHEMA = "neuralstudio.fg.custom-upsample-negative-audit.v1"
GENERATOR = "splitmix64-fg-custom-upsample-signed-spatial-v2"
MODULE_SHA256 = "1d19bfddd2affc5587ec19829259b070404e44c28a6f0c570a4b7769fd0dbba6"
MODULE_BYTES = 2112
ADDRESS_FREE_PARAMETER_SHA256 = (
    "80189cac9cec275fc90e1a52f3e814475d063b67ed2fd80b05ae98ee1e9654f6"
)
RAW_PARAMETER_SHA256 = (
    "2c38d2f149ec90e2e6cac2a843a824485c16421300fafcce34a1acaaeb8a25e4"
)
CUMETAL_DIAGNOSTIC = (
    "line 65: PTX register '%r114' is undefined on an incoming edge to block "
    "'$L__BB0_2'"
)
INPUT_HEIGHT = 184
INPUT_WIDTH = 320
OUTPUT_HEIGHT = 368
OUTPUT_WIDTH = 640
CHANNELS = (4, 1, 3)
TOTAL_OUTPUT_BYTES = OUTPUT_HEIGHT * OUTPUT_WIDTH * sum(CHANNELS) * 2
CASES = {
    "poison-a5-process-a": 0xA5,
    "poison-a5-process-b": 0xA5,
    "poison-5a-process-a": 0x5A,
    "poison-5a-process-b": 0x5A,
}
BACKEND_NAMES = {"rtx": "rtx-cuda-driver", "mac": "mac-metal"}
ADDRESS_PATTERN = re.compile(r"(?i)0x[0-9a-f]{8,16}")
REPOSITORY = Path(__file__).resolve().parent.parent


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


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def rooted(document: dict[str, Any], key: str) -> bool:
    expected = document.get(key)
    payload = {name: value for name, value in document.items() if name != key}
    return isinstance(expected, str) and expected == canonical_sha256(payload)


def reroot(document: dict[str, Any], key: str) -> None:
    document.pop(key, None)
    document[key] = canonical_sha256(document)


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"unable to load JSON {path}: {error}") from error
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


def write_sidecar(path: Path) -> Path:
    sidecar = path.with_suffix(".sha256")
    sidecar.write_text(f"{sha256_file(path)}  {path.name}\n", encoding="utf-8")
    return sidecar


def require_no_address_fields(value: Any, label: str) -> None:
    def walk(item: Any, location: str) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                lowered = str(key).lower()
                require(
                    not any(
                        marker in lowered
                        for marker in ("pointer", "device_address", "gpu_va")
                    ),
                    f"{label} contains address field at {location}.{key}",
                )
                walk(child, f"{location}.{key}")
        elif isinstance(item, list):
            for index, child in enumerate(item):
                walk(child, f"{location}[{index}]")
        elif isinstance(item, str):
            require(
                ADDRESS_PATTERN.search(item) is None,
                f"{label} contains pointer-like value at {location}",
            )

    walk(value, "$")


def splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def input_half_bits(plane: int, index: int, channels: int) -> int:
    channel = index % channels
    mixed = splitmix64(
        0xD1B54A32D192ED03
        ^ ((plane + 1) * 0x94D049BB133111EB)
        ^ (channel * 0x8EBC6AF09C88C6E3)
        ^ (index * 0x9E3779B97F4A7C15)
    )
    sign = 0x8000 if (mixed >> 63) else 0
    if (mixed & 63) == 0:
        return sign
    exponent = 13 + ((mixed >> 12) & 3)
    return sign | (exponent << 10) | (mixed & 0x03FF)


def generate_input(plane: int) -> bytes:
    channels = CHANNELS[plane]
    values = INPUT_HEIGHT * INPUT_WIDTH * channels
    result = bytearray(values * 2)
    for index in range(values):
        struct.pack_into("<H", result, index * 2, input_half_bits(plane, index, channels))
    return bytes(result)


def expected_plane(plane: int) -> dict[str, int]:
    channels = CHANNELS[plane]
    return {
        "plane": plane,
        "channels": channels,
        "input_height": INPUT_HEIGHT,
        "input_width": INPUT_WIDTH,
        "output_height": OUTPUT_HEIGHT,
        "output_width": OUTPUT_WIDTH,
        "input_bytes": INPUT_HEIGHT * INPUT_WIDTH * channels * 2,
        "output_bytes": OUTPUT_HEIGHT * OUTPUT_WIDTH * channels * 2,
    }


def selected_parameter_abi() -> list[dict[str, Any]]:
    allocations = (
        "allocation-000097",
        "allocation-000101",
        "allocation-000102",
        "allocation-000099",
        "allocation-000100",
        "allocation-000103",
    )
    records: list[dict[str, Any]] = []
    for index in range(6):
        plane = index % 3
        output = index >= 3
        records.append(
            {
                "parameter_offset": index * 24,
                "record_bytes": 24,
                "role": "output" if output else "input",
                "plane": plane,
                "provenance_allocation_token": allocations[index],
                "allocation_offset": 0,
                "lifetime_contained": True,
                "batch": 1,
                "height": OUTPUT_HEIGHT if output else INPUT_HEIGHT,
                "width": OUTPUT_WIDTH if output else INPUT_WIDTH,
                "channels": CHANNELS[plane],
                "active_bytes": (
                    (OUTPUT_HEIGHT if output else INPUT_HEIGHT)
                    * (OUTPUT_WIDTH if output else INPUT_WIDTH)
                    * CHANNELS[plane]
                    * 2
                ),
            }
        )
    return records


def validate_run(
    root: Path, run_id: str, poison: int, backend_key: str
) -> tuple[dict[str, Any], list[bytes]]:
    run_root = root / run_id
    receipt_path = run_root / "run-receipt.json"
    receipt = load_json(receipt_path)
    require_no_address_fields(receipt, f"{backend_key} run receipt")
    require(receipt.get("schema") == RUN_SCHEMA, f"{run_id} schema mismatch")
    require(
        receipt.get("backend") == BACKEND_NAMES[backend_key],
        f"{run_id} backend mismatch",
    )
    require(receipt.get("run_id") == run_id, f"{run_id} identity mismatch")
    require(receipt.get("poison_byte") == poison, f"{run_id} poison mismatch")
    require(
        isinstance(receipt.get("process_id"), int) and receipt["process_id"] > 0,
        f"{run_id} process id missing",
    )
    require(receipt.get("input_generator") == GENERATOR, f"{run_id} generator mismatch")
    require(
        receipt.get("address_free_parameter_sha256")
        == ADDRESS_FREE_PARAMETER_SHA256,
        f"{run_id} address-free parameter identity mismatch",
    )
    require(
        receipt.get("raw_captured_parameter_sha256") == RAW_PARAMETER_SHA256,
        f"{run_id} raw parameter identity mismatch",
    )
    require(receipt.get("packed_parameter_bytes") == 144, f"{run_id} ABI size mismatch")
    require(
        receipt.get("independently_generated_inputs") is True,
        f"{run_id} did not independently generate inputs",
    )
    require(
        receipt.get("external_expected_tensors_consumed") is False,
        f"{run_id} consumed an expected tensor",
    )
    require(
        receipt.get("address_fields_recorded") is False,
        f"{run_id} records an address",
    )
    require(
        receipt.get("single_poison_full_write_claimed") is False,
        f"{run_id} makes a single-poison coverage claim",
    )
    require(receipt.get("input_mismatched_bytes") == 0, f"{run_id} input mutated")
    require(receipt.get("canary_mismatched_bytes") == 0, f"{run_id} canary failed")
    require(receipt.get("input_immutability") is True, f"{run_id} immutability false")
    require(receipt.get("result") == "pass", f"{run_id} receipt did not pass")
    require(
        receipt.get("total_output_bytes") == TOTAL_OUTPUT_BYTES,
        f"{run_id} total output bytes mismatch",
    )

    if backend_key == "rtx":
        require(receipt.get("module_token") == 60, f"{run_id} module token mismatch")
        require(receipt.get("module_bytes") == MODULE_BYTES, f"{run_id} module size mismatch")
        require(receipt.get("module_sha256") == MODULE_SHA256, f"{run_id} module hash mismatch")
        require(
            receipt.get("function_name") == "custom_upsample_hf_kernel",
            f"{run_id} function mismatch",
        )
        require(receipt.get("launch_token") == 70, f"{run_id} launch mismatch")
        require(receipt.get("grid") == [40, 23, 3], f"{run_id} grid mismatch")
        require(receipt.get("block") == [16, 16, 1], f"{run_id} block mismatch")
        require(receipt.get("dynamic_shared_bytes") == 0, f"{run_id} shared bytes mismatch")
    else:
        require(
            receipt.get("selected_module_sha256") == MODULE_SHA256,
            f"{run_id} selected module binding mismatch",
        )
        require(receipt.get("selected_launch_token") == 70, f"{run_id} selected launch mismatch")
        require(receipt.get("metal_dispatch_count") == 3, f"{run_id} dispatch count mismatch")
        require(isinstance(receipt.get("device"), str) and receipt["device"], f"{run_id} device missing")

    plane_receipts = receipt.get("planes")
    require(isinstance(plane_receipts, list) and len(plane_receipts) == 3, f"{run_id} plane count mismatch")
    outputs: list[bytes] = []
    combined = bytearray()
    for plane, plane_receipt in enumerate(plane_receipts):
        require(isinstance(plane_receipt, dict), f"{run_id} plane {plane} is not an object")
        expected = expected_plane(plane)
        for key, value in expected.items():
            require(
                plane_receipt.get(key) == value,
                f"{run_id} plane {plane} {key} mismatch",
            )
        expected_input = generate_input(plane)
        input_path = run_root / f"input-plane-{plane}-after.bin"
        output_path = run_root / f"output-plane-{plane}.bin"
        try:
            actual_input = input_path.read_bytes()
            output = output_path.read_bytes()
        except OSError as error:
            raise VerificationError(f"unable to read {run_id} plane {plane}: {error}") from error
        require(actual_input == expected_input, f"{run_id} plane {plane} input artifact mismatch")
        require(len(output) == expected["output_bytes"], f"{run_id} plane {plane} output size mismatch")
        require(
            plane_receipt.get("input_pre_sha256") == sha256_bytes(expected_input),
            f"{run_id} plane {plane} input pre hash mismatch",
        )
        require(
            plane_receipt.get("input_post_sha256") == sha256_bytes(actual_input),
            f"{run_id} plane {plane} input post hash mismatch",
        )
        require(
            plane_receipt.get("output_sha256") == sha256_bytes(output),
            f"{run_id} plane {plane} output hash mismatch",
        )
        outputs.append(output)
        combined.extend(output)
    require(
        receipt.get("combined_output_sha256") == sha256_bytes(bytes(combined)),
        f"{run_id} combined output hash mismatch",
    )
    return receipt, outputs


def build_backend(root: Path, backend_key: str) -> dict[str, Any]:
    require(backend_key in BACKEND_NAMES, "unknown backend")
    runs: dict[str, dict[str, Any]] = {}
    outputs: dict[str, list[bytes]] = {}
    for run_id, poison in CASES.items():
        receipt, run_outputs = validate_run(root, run_id, poison, backend_key)
        runs[run_id] = receipt
        outputs[run_id] = run_outputs
    process_ids = [runs[run_id]["process_id"] for run_id in CASES]
    require(len(set(process_ids)) == 4, "fresh-process gate failed")

    reference_id = next(iter(CASES))
    reference_outputs = outputs[reference_id]
    for run_id, run_outputs in outputs.items():
        for plane in range(3):
            require(
                run_outputs[plane] == reference_outputs[plane],
                f"{backend_key} output changed in {run_id} plane {plane}",
            )

    a5_outputs = outputs["poison-a5-process-a"]
    p5a_outputs = outputs["poison-5a-process-a"]
    jointly_untouched = 0
    for plane in range(3):
        for offset in range(0, len(a5_outputs[plane]), 2):
            jointly_untouched += (
                a5_outputs[plane][offset : offset + 2] == b"\xa5\xa5"
                and p5a_outputs[plane][offset : offset + 2] == b"\x5a\x5a"
            )
    require(jointly_untouched == 0, "two-poison write coverage failed")

    combined = b"".join(reference_outputs)
    document: dict[str, Any] = {
        "schema": BACKEND_SCHEMA,
        "backend": BACKEND_NAMES[backend_key],
        "run_ids": list(CASES),
        "poison_schedule": [CASES[run_id] for run_id in CASES],
        "fresh_process_count": 4,
        "unique_process_ids": sorted(process_ids),
        "input_generator": GENERATOR,
        "module_sha256": MODULE_SHA256,
        "address_free_parameter_sha256": ADDRESS_FREE_PARAMETER_SHA256,
        "plane_output_sha256": [sha256_bytes(value) for value in reference_outputs],
        "combined_output_sha256": sha256_bytes(combined),
        "total_output_bytes": len(combined),
        "all_process_outputs_identical": True,
        "two_poison_output_invariance": True,
        "jointly_untouched_half_values": jointly_untouched,
        "full_write_coverage": True,
        "input_immutability": True,
        "canary_integrity": True,
        "independently_generated_inputs": True,
        "external_expected_tensors_consumed": False,
        "address_fields_recorded": False,
        "run_receipt_sha256": {
            run_id: sha256_file(root / run_id / "run-receipt.json")
            for run_id in CASES
        },
        "result": "pass",
    }
    document["backend_root_sha256"] = canonical_sha256(document)
    return document


def validate_backend_document(document: dict[str, Any], backend_key: str) -> None:
    require_no_address_fields(document, f"{backend_key} backend receipt")
    require(document.get("schema") == BACKEND_SCHEMA, "backend schema mismatch")
    require(document.get("backend") == BACKEND_NAMES[backend_key], "backend identity mismatch")
    require(rooted(document, "backend_root_sha256"), "backend receipt root mismatch")
    require(document.get("run_ids") == list(CASES), "backend run ids mismatch")
    require(document.get("poison_schedule") == list(CASES.values()), "backend poison schedule mismatch")
    require(document.get("fresh_process_count") == 4, "backend process count mismatch")
    ids = document.get("unique_process_ids")
    require(isinstance(ids, list) and len(ids) == 4 and len(set(ids)) == 4, "backend process ids are not unique")
    require(document.get("input_generator") == GENERATOR, "backend generator mismatch")
    require(document.get("module_sha256") == MODULE_SHA256, "backend module mismatch")
    require(document.get("address_free_parameter_sha256") == ADDRESS_FREE_PARAMETER_SHA256, "backend parameter mismatch")
    hashes = document.get("plane_output_sha256")
    require(isinstance(hashes, list) and len(hashes) == 3, "backend output hashes missing")
    require(all(isinstance(item, str) and len(item) == 64 for item in hashes), "backend output hash invalid")
    require(document.get("total_output_bytes") == TOTAL_OUTPUT_BYTES, "backend output size mismatch")
    for key in (
        "all_process_outputs_identical",
        "two_poison_output_invariance",
        "full_write_coverage",
        "input_immutability",
        "canary_integrity",
        "independently_generated_inputs",
    ):
        require(document.get(key) is True, f"backend gate false: {key}")
    require(document.get("jointly_untouched_half_values") == 0, "backend has untouched half values")
    require(document.get("external_expected_tensors_consumed") is False, "backend consumed expected tensors")
    require(document.get("address_fields_recorded") is False, "backend records addresses")
    require(document.get("result") == "pass", "backend did not pass")


def validate_rig_document(document: dict[str, Any]) -> None:
    require_no_address_fields(document, "rig receipt")
    require(document.get("schema") == RIG_SCHEMA, "rig schema mismatch")
    require(rooted(document, "rig_root_sha256"), "rig receipt root mismatch")
    require(isinstance(document.get("job_id"), str) and document["job_id"], "rig job missing")
    hardware = document.get("hardware")
    require(isinstance(hardware, dict), "rig hardware missing")
    require("RTX 5090" in str(hardware.get("gpu")), "rig GPU mismatch")
    require(hardware.get("compute_capability") == "12.0", "rig compute capability mismatch")
    require(isinstance(hardware.get("driver"), str) and hardware["driver"], "rig driver missing")
    require(isinstance(hardware.get("cuda_toolkit"), str) and hardware["cuda_toolkit"], "rig CUDA missing")
    selected = document.get("selected_module")
    require(isinstance(selected, dict), "rig selected module missing")
    require(selected.get("bytes") == MODULE_BYTES, "rig module bytes mismatch")
    require(selected.get("sha256") == MODULE_SHA256, "rig module hash mismatch")
    require(selected.get("transport") == "cuModuleLoadDataEx-exact-selected-module", "rig transport mismatch")
    require(document.get("fresh_process_count") == 4, "rig process count mismatch")
    require(document.get("process_exit_codes") == [0, 0, 0, 0], "rig process failed")
    require(document.get("compile_exit_code") == 0, "rig compile failed")
    require(document.get("command_exit_code") == 0, "rig command failed")
    require(document.get("external_expected_tensors_consumed") is False, "rig consumed expected tensors")
    require(document.get("address_fields_recorded") is False, "rig records addresses")
    require(document.get("result") == "pass", "rig receipt did not pass")


def validate_cumetal_document(document: dict[str, Any]) -> None:
    require_no_address_fields(document, "CuMetal failure receipt")
    require(document.get("schema") == CUMETAL_SCHEMA, "CuMetal schema mismatch")
    require(rooted(document, "failure_root_sha256"), "CuMetal failure root mismatch")
    require(document.get("module_sha256") == MODULE_SHA256, "CuMetal module mismatch")
    require(document.get("entry") == "custom_upsample_hf_kernel", "CuMetal entry mismatch")
    require(document.get("backend") == "cumetal-ir", "CuMetal backend mismatch")
    require(document.get("strict") is True, "CuMetal strict mode missing")
    require(document.get("exit_code") == 1, "CuMetal exit code mismatch")
    require(document.get("diagnostic") == CUMETAL_DIAGNOSTIC, "CuMetal diagnostic mismatch")
    require(document.get("generated_msl") is False, "CuMetal unexpectedly generated MSL")
    require(document.get("preserved_as_counterexample") is True, "CuMetal counterexample not preserved")


def validate_cross_backend(rtx: dict[str, Any], mac: dict[str, Any]) -> None:
    validate_backend_document(rtx, "rtx")
    validate_backend_document(mac, "mac")
    require(rtx["plane_output_sha256"] == mac["plane_output_sha256"], "RTX/Metal plane outputs differ")
    require(rtx["combined_output_sha256"] == mac["combined_output_sha256"], "RTX/Metal combined output differs")
    require(rtx["total_output_bytes"] == mac["total_output_bytes"], "RTX/Metal output sizes differ")


def expect_rejection(label: str, callback: Callable[[], None]) -> dict[str, Any]:
    try:
        callback()
    except VerificationError as error:
        return {"mutation": label, "rejected": True, "diagnostic": str(error)}
    raise VerificationError(f"negative mutation was accepted: {label}")


def build_negative_audit(
    rtx: dict[str, Any], mac: dict[str, Any], rig: dict[str, Any],
    cumetal: dict[str, Any]
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []

    mutated = copy.deepcopy(rtx)
    mutated["plane_output_sha256"][0] = "0" * 64
    reroot(mutated, "backend_root_sha256")
    entries.append(expect_rejection("rtx_output_hash", lambda: validate_cross_backend(mutated, mac)))

    mutated = copy.deepcopy(mac)
    mutated["combined_output_sha256"] = "f" * 64
    reroot(mutated, "backend_root_sha256")
    entries.append(expect_rejection("mac_combined_output_hash", lambda: validate_cross_backend(rtx, mutated)))

    mutated = copy.deepcopy(rtx)
    mutated["unique_process_ids"][3] = mutated["unique_process_ids"][0]
    reroot(mutated, "backend_root_sha256")
    entries.append(expect_rejection("duplicate_process", lambda: validate_backend_document(mutated, "rtx")))

    mutated = copy.deepcopy(rtx)
    mutated["poison_schedule"][0] = 0
    reroot(mutated, "backend_root_sha256")
    entries.append(expect_rejection("poison_schedule", lambda: validate_backend_document(mutated, "rtx")))

    mutated = copy.deepcopy(rtx)
    mutated["module_sha256"] = "0" * 64
    reroot(mutated, "backend_root_sha256")
    entries.append(expect_rejection("module_identity", lambda: validate_backend_document(mutated, "rtx")))

    mutated = copy.deepcopy(mac)
    mutated["external_expected_tensors_consumed"] = True
    reroot(mutated, "backend_root_sha256")
    entries.append(expect_rejection("expected_tensor_contamination", lambda: validate_backend_document(mutated, "mac")))

    mutated = copy.deepcopy(mac)
    mutated["gpu_va"] = "0x1234567890abcdef"
    reroot(mutated, "backend_root_sha256")
    entries.append(expect_rejection("address_in_receipt", lambda: validate_backend_document(mutated, "mac")))

    mutated = copy.deepcopy(rig)
    mutated["hardware"]["gpu"] = "different GPU"
    reroot(mutated, "rig_root_sha256")
    entries.append(expect_rejection("rig_gpu", lambda: validate_rig_document(mutated)))

    mutated = copy.deepcopy(cumetal)
    mutated["diagnostic"] = "different failure"
    reroot(mutated, "failure_root_sha256")
    entries.append(expect_rejection("cumetal_diagnostic", lambda: validate_cumetal_document(mutated)))

    document: dict[str, Any] = {
        "schema": NEGATIVE_SCHEMA,
        "mutation_count": len(entries),
        "all_mutations_rejected": all(item["rejected"] for item in entries),
        "mutations": entries,
    }
    document["negative_audit_root_sha256"] = canonical_sha256(document)
    return document


def build_final(
    rtx: dict[str, Any], mac: dict[str, Any], rig: dict[str, Any],
    cumetal: dict[str, Any], negative: dict[str, Any],
    implementation_artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    validate_cross_backend(rtx, mac)
    validate_rig_document(rig)
    validate_cumetal_document(cumetal)
    require(rooted(negative, "negative_audit_root_sha256"), "negative audit root mismatch")
    require(negative.get("all_mutations_rejected") is True, "negative audit failed")
    require(negative.get("mutation_count") >= 9, "negative audit incomplete")
    document: dict[str, Any] = {
        "schema": FINAL_SCHEMA,
        "accepted_scope": "evaluate_1 launch 70 custom_upsample_hf_kernel only",
        "selected_module": {
            "token": 60,
            "bytes": MODULE_BYTES,
            "sha256": MODULE_SHA256,
            "function": "custom_upsample_hf_kernel",
            "launch_token": 70,
            "grid": [40, 23, 3],
            "block": [16, 16, 1],
            "dynamic_shared_bytes": 0,
            "packed_parameter_bytes": 144,
            "address_free_parameter_sha256": ADDRESS_FREE_PARAMETER_SHA256,
        },
        "plane_contracts": [expected_plane(plane) for plane in range(3)],
        "selected_parameter_record_layout": [
            "device pointer as unsigned 64-bit field",
            "batch as unsigned 32-bit field",
            "height as unsigned 32-bit field",
            "width as unsigned 32-bit field",
            "channels as unsigned 32-bit field",
        ],
        "selected_parameter_abi": selected_parameter_abi(),
        "selected_parameter_abi_root_sha256": canonical_sha256(
            selected_parameter_abi()
        ),
        "metal_dispatch_count": 3,
        "source_ptx_explicit_half_operation_count": 9,
        "observed_driver_fused_half_operations": 3,
        "separately_rounded_half_multiply_count": 3,
        "approximate_math_operations": 0,
        "descriptor_count": 0,
        "weight_bytes": 0,
        "shared_memory_bytes": 0,
        "input_generator": GENERATOR,
        "input_plane_sha256": [sha256_bytes(generate_input(plane)) for plane in range(3)],
        "numerical_domain": (
            "spatially mixed signed finite FP16 normals or signed zero with "
            "magnitudes in [0.25,4); generated cancellation cases included"
        ),
        "unsupported_numerical_domains": [
            "FP16 subnormal inputs",
            "infinities",
            "NaNs",
            "exhaustive coverage of all possible subnormal intermediates",
            "scales other than exact 2x",
            "shapes other than the three live planes",
        ],
        "rtx_backend_root_sha256": rtx["backend_root_sha256"],
        "mac_backend_root_sha256": mac["backend_root_sha256"],
        "rig_root_sha256": rig["rig_root_sha256"],
        "cumetal_failure_root_sha256": cumetal["failure_root_sha256"],
        "negative_audit_root_sha256": negative["negative_audit_root_sha256"],
        "implementation_artifacts": implementation_artifacts,
        "implementation_artifact_root_sha256": canonical_sha256(implementation_artifacts),
        "plane_output_sha256": rtx["plane_output_sha256"],
        "combined_output_sha256": rtx["combined_output_sha256"],
        "total_output_bytes": TOTAL_OUTPUT_BYTES,
        "fresh_processes_per_backend": 4,
        "two_poison_output_invariance": True,
        "jointly_untouched_half_values": 0,
        "input_immutability": True,
        "canary_integrity": True,
        "exact_rtx_mac_byte_parity": True,
        "independently_generated_inputs": True,
        "external_expected_tensors_consumed": False,
        "archived_intermediates_consumed": False,
        "address_fields_recorded": False,
        "cumetal_strict_failure_preserved": True,
        "handwritten_three_way_metal_dispatch": True,
        "custom_upsample_group_parity_proven": True,
        "surrounding_block_parity_proven": False,
        "complete_generated_frame_parity_proven": False,
        "netal_integration_proven": False,
        "performance_advantage_proven": False,
        "result": "pass",
    }
    require_no_address_fields(document, "final receipt")
    document["verification_root_sha256"] = canonical_sha256(document)
    return document


def collect_implementation_artifacts(
    rtx_backend_path: Path, mac_backend_path: Path, rig_receipt_path: Path
) -> list[dict[str, Any]]:
    paths = {
        "source/custom_upsample.metal": REPOSITORY / "src/fg_custom_upsample/custom_upsample.metal",
        "source/macos_probe.mm": REPOSITORY / "src/fg_custom_upsample/macos_probe.mm",
        "source/rtx_oracle.cpp": REPOSITORY / "src/fg_custom_upsample/rtx_oracle.cpp",
        "source/README.md": REPOSITORY / "src/fg_custom_upsample/README.md",
        "script/build-macos-fg-custom-upsample.sh": REPOSITORY / "scripts/build-macos-fg-custom-upsample.sh",
        "script/build-fg-custom-upsample-rtx.ps1": REPOSITORY / "scripts/build-fg-custom-upsample-rtx.ps1",
        "script/verify-fg-custom-upsample.py": Path(__file__).resolve(),
        "mac/custom_upsample.metallib": mac_backend_path.parent / "build/custom_upsample.metallib",
        "mac/fg-custom-upsample-mac": mac_backend_path.parent / "build/fg-custom-upsample-mac",
        "rtx/fg-custom-upsample-rtx.exe": rtx_backend_path.parent / "build/fg-custom-upsample-rtx.exe",
        "receipt/rtx-backend.json": rtx_backend_path,
        "receipt/mac-backend.json": mac_backend_path,
        "receipt/rig.json": rig_receipt_path,
    }
    result: list[dict[str, Any]] = []
    for name, path in paths.items():
        require(path.is_file(), f"implementation artifact missing: {name}")
        result.append(
            {
                "name": name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    backend_parser = subparsers.add_parser("backend")
    backend_parser.add_argument("--backend", choices=tuple(BACKEND_NAMES), required=True)
    backend_parser.add_argument("--root", type=Path, required=True)
    backend_parser.add_argument("--output", type=Path, required=True)

    final_parser = subparsers.add_parser("final")
    final_parser.add_argument("--rtx-backend", type=Path, required=True)
    final_parser.add_argument("--mac-backend", type=Path, required=True)
    final_parser.add_argument("--rig-receipt", type=Path, required=True)
    final_parser.add_argument("--cumetal-failure", type=Path, required=True)
    final_parser.add_argument("--negative-output", type=Path, required=True)
    final_parser.add_argument("--output", type=Path, required=True)

    rig_parser = subparsers.add_parser("root-rig")
    rig_parser.add_argument("--input", type=Path, required=True)
    rig_parser.add_argument("--output", type=Path, required=True)

    arguments = parser.parse_args()
    try:
        if arguments.command == "root-rig":
            document = load_json(arguments.input)
            require("rig_root_sha256" not in document, "rig draft is already rooted")
            document["rig_root_sha256"] = canonical_sha256(document)
            validate_rig_document(document)
            write_json(arguments.output, document)
            sidecar = write_sidecar(arguments.output)
            print(f"rig_root_sha256={document['rig_root_sha256']}")
            print(f"receipt_file_sha256={sha256_file(arguments.output)}")
            print(f"sidecar={sidecar.name}")
            print("result=pass")
            return 0

        if arguments.command == "backend":
            document = build_backend(arguments.root, arguments.backend)
            validate_backend_document(document, arguments.backend)
            write_json(arguments.output, document)
            sidecar = write_sidecar(arguments.output)
            print(f"backend={arguments.backend}")
            print(f"backend_root_sha256={document['backend_root_sha256']}")
            print(f"receipt_file_sha256={sha256_file(arguments.output)}")
            print(f"sidecar={sidecar.name}")
            print("result=pass")
            return 0

        rtx = load_json(arguments.rtx_backend)
        mac = load_json(arguments.mac_backend)
        rig = load_json(arguments.rig_receipt)
        cumetal = load_json(arguments.cumetal_failure)
        validate_cross_backend(rtx, mac)
        validate_rig_document(rig)
        validate_cumetal_document(cumetal)
        negative = build_negative_audit(rtx, mac, rig, cumetal)
        write_json(arguments.negative_output, negative)
        write_sidecar(arguments.negative_output)
        implementation_artifacts = collect_implementation_artifacts(
            arguments.rtx_backend,
            arguments.mac_backend,
            arguments.rig_receipt,
        )
        final = build_final(
            rtx, mac, rig, cumetal, negative, implementation_artifacts
        )
        write_json(arguments.output, final)
        sidecar = write_sidecar(arguments.output)
        print(f"combined_output_sha256={final['combined_output_sha256']}")
        print(f"verification_root_sha256={final['verification_root_sha256']}")
        print(f"receipt_file_sha256={sha256_file(arguments.output)}")
        print(f"sidecar={sidecar.name}")
        print("result=pass")
        return 0
    except VerificationError as error:
        print(f"result=fail\nerror={error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
