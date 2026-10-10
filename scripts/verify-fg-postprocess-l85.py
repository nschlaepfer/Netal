#!/usr/bin/env python3
"""Fail-closed verifier for live-selected FG launch 85 and its Metal port."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import tarfile
from pathlib import Path
from typing import Any


RUN_SCHEMA = "neuralstudio.fg.postprocess-l85-run.v1"
PROBE_SCHEMA = "neuralstudio.fg.postprocess-l85-div-probe.v1"
BACKEND_SCHEMA = "neuralstudio.fg.postprocess-l85-backend.v1"
FINAL_SCHEMA = "neuralstudio.fg.postprocess-l85-verification.v1"
MODULE_SHA = "07877ed3dcec1977a5ddfd7843314a5044ed9559bb2d222713906afdfc609d1c"
PARAMETER_SHA = "ac5ec30d58d9b7a47fc229acce7e3672af982298ba25e577ec4886bcdd929927"
ROM_SHA = "7cf4f8cece6b18b07605edc9cb601a153346a79b10bf3b93c7ec058d57d29750"
GENERATOR = "splitmix64-fg-postprocess-l85-weighted-half4-v1"
DOMAIN = "signed-normal-finite-rgb-positive-normal-finite-alpha-normal-denominator-and-reciprocal"
WIDTH = 40
HEIGHT = 23
OUTPUT_WIDTH = 20
OUTPUT_HEIGHT = 12
INPUT_BYTES = WIDTH * HEIGHT * 8
OUTPUT_BYTES = OUTPUT_WIDTH * OUTPUT_HEIGHT * 8
LEVEL_BYTES = (OUTPUT_BYTES, 480, 120, 48)
MASK64 = (1 << 64) - 1
CASES = {
    "poison-a5-process-a": 0xA5,
    "poison-a5-process-b": 0xA5,
    "poison-5a-process-a": 0x5A,
    "poison-5a-process-b": 0x5A,
}
ADDRESS_PATTERN = re.compile(r"(?i)0x[0-9a-f]{8,16}")
SPECIAL_RED = (
    (0x3C00, 0x2D54, 0xC0BB, 0xC6CA),
    (0x2AA8, 0x4477, 0x4844, 0x2987),
    (0xB2CA, 0xC1ED, 0x4198, 0xC2EC),
    (0xAFB8, 0xCA0F, 0x4697, 0x3822),
    (0x2E20, 0x3D54, 0x3022, 0x4396),
    (0xB5BA, 0xC3EB, 0xC8BB, 0x2ADB),
    (0xC499, 0x2AEC, 0x4455, 0xABC9),
    (0xC4AA, 0xACEE, 0x4976, 0x35FE),
    (0x2AEC, 0x35BA, 0xB3C9, 0x32FD),
    (0x2F52, 0x4143, 0xAE53, 0xC5FE),
    (0xBDBA, 0xBCAA, 0x4AFD, 0xBE0F),
    (0xC899, 0x42EC, 0x3987, 0x3965),
)
SPECIAL_ALPHA = (
    (0x2F74, 0x2C55, 0x2AA8, 0x39DC),
    (0x32EC, 0x3A20, 0x3F85, 0x3796),
    (0x2D65, 0x2FC9, 0x3ADB, 0x3AA8),
    (0x2822, 0x32CA, 0x3D65, 0x3400),
    (0x3998, 0x3BA7, 0x2D87, 0x2A75),
    (0x2B52, 0x2DDC, 0x43B8, 0x330E),
    (0x3011, 0x28CC, 0x33DA, 0x2CFF),
    (0x3231, 0x2E42, 0x3642, 0x3CBB),
    (0x3686, 0x3CBB, 0x3F63, 0x3877),
    (0x2D65, 0x3D65, 0x41A9, 0x3D98),
    (0x2C55, 0x2921, 0x3576, 0x38CC),
    (0x2FC9, 0x3253, 0x3C99, 0x3CEE),
)


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"unable to load JSON {path}: {error}") from error
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    require_no_addresses(value, path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def require_no_addresses(value: Any, label: str) -> None:
    def visit(item: Any, path: str) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                lowered = str(key).lower()
                if not lowered.startswith("address_free"):
                    require(
                        not any(
                            marker in lowered
                            for marker in ("pointer", "gpu_va", "device_address")
                        ),
                        f"{label} contains address field at {path}.{key}",
                    )
                visit(child, f"{path}.{key}")
        elif isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")
        elif isinstance(item, str):
            require(
                ADDRESS_PATTERN.search(item) is None,
                f"{label} contains address-like value at {path}",
            )

    visit(value, "$")


def splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def regular_rgb(tile: int, lane: int, channel: int) -> int:
    sign_code = splitmix64(
        0xD6E8FEB86659FD93
        ^ (tile * 0x9E3779B97F4A7C15)
        ^ (channel * 0x94D049BB133111EB)
    )
    mixed = splitmix64(sign_code ^ ((lane + 1) * 0xBF58476D1CE4E5B9))
    sign = 0x8000 if (sign_code >> 63) else 0
    exponent = 12 + ((mixed >> 13) & 5)
    return sign | (exponent << 10) | (mixed & 0x03FF)


def regular_alpha(tile: int, lane: int) -> int:
    mixed = splitmix64(
        0xA0761D6478BD642F
        ^ ((tile + 1) * 0xE7037ED1A0B428DB)
        ^ ((lane + 1) * 0x8EBC6AF09C88C6E3)
    )
    exponent = 12 + ((mixed >> 19) & 3)
    return (exponent << 10) | (mixed & 0x03FF)


def generate_input() -> bytes:
    result = bytearray(INPUT_BYTES)
    for y in range(HEIGHT):
        for x in range(WIDTH):
            tile = (y // 2) * OUTPUT_WIDTH + x // 2
            lane = (y & 1) * 2 + (x & 1)
            channels = (
                SPECIAL_RED[tile][lane] if tile < 12 else regular_rgb(tile, lane, 0),
                regular_rgb(tile, lane, 1),
                regular_rgb(tile, lane, 2),
                SPECIAL_ALPHA[tile][lane]
                if tile < 12
                else regular_alpha(tile, lane),
            )
            struct.pack_into("<4H", result, (y * WIDTH + x) * 8, *channels)
    return bytes(result)


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def bits_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def half_value(bits: int) -> float:
    return struct.unpack("<e", struct.pack("<H", bits))[0]


def half_bits(value: float) -> int:
    return struct.unpack("<H", struct.pack("<e", value))[0]


def add_f32(a: float, b: float) -> float:
    return f32(f32(a) + f32(b))


def mul_f32(a: float, b: float) -> float:
    return f32(f32(a) * f32(b))


def div_f32(a: float, b: float) -> float:
    return f32(f32(a) / f32(b))


def rcp_from_rom(denominator: float, rom: bytes) -> tuple[int, int]:
    input_bits = f32_bits(denominator)
    base_bits = f32_bits(div_f32(1.0, denominator))
    mantissa = input_bits & 0x7FFFFF
    code = (rom[mantissa >> 2] >> ((mantissa & 3) * 2)) & 3
    require(code != 3, "RCP ROM contains unsupported code for live denominator")
    if code == 1:
        base_bits -= 1
    elif code == 2:
        base_bits += 1
    return base_bits, code


def reconstructed_reductions(input_bytes: bytes) -> list[tuple[int, int]]:
    pixels = [
        tuple(half_value(bits) for bits in struct.unpack_from("<4H", input_bytes, i))
        for i in range(0, len(input_bytes), 8)
    ]
    result: list[tuple[int, int]] = []
    threshold = bits_f32(0x2B8CBCCC)
    for output_y in range(OUTPUT_HEIGHT):
        for output_x in range(OUTPUT_WIDTH):
            values: list[tuple[float, float, float, float]] = []
            for lane in range(4):
                x = output_x * 2 + (lane & 1)
                y = output_y * 2 + (lane >> 1)
                values.append(
                    pixels[y * WIDTH + x]
                    if x < WIDTH and y < HEIGHT
                    else (0.0, 0.0, 0.0, 0.0)
                )
            numerators: list[float] = []
            for channel in range(3):
                products = [mul_f32(pixel[channel], pixel[3]) for pixel in values]
                numerators.append(
                    add_f32(add_f32(products[0], products[1]),
                            add_f32(products[2], products[3]))
                )
            alpha = add_f32(
                add_f32(values[0][3], values[1][3]),
                add_f32(values[2][3], values[3][3]),
            )
            denominator = max(alpha, threshold)
            for channel in range(4):
                numerator = numerators[channel] if channel < 3 else alpha
                divisor = denominator if channel < 3 else 4.0
                result.append((f32_bits(numerator), f32_bits(divisor)))
    return result


def load_rom(path: Path | None) -> bytes:
    if path is not None:
        data = path.read_bytes()
    else:
        repository = Path(__file__).resolve().parent.parent
        archive = repository / "evidence/checkpoints/2026-08-31-sm120-rcp-rom/external-evidence-sm120-rcp-rom.tar.gz"
        member = "work/g23-rcp-architecture/m3-rcp-rom/correction-rom.bin"
        try:
            with tarfile.open(archive, "r:gz") as bundle:
                extracted = bundle.extractfile(member)
                require(extracted is not None, f"RCP ROM member missing: {member}")
                data = extracted.read()
        except (OSError, tarfile.TarError) as error:
            raise VerificationError(f"unable to load RCP ROM: {error}") from error
    require(len(data) == 2097152, "RCP ROM byte count drift")
    require(sha256_bytes(data) == ROM_SHA, "RCP ROM hash drift")
    return data


def verify_div_records(
    records_bytes: bytes, probe_output: bytes, input_bytes: bytes, rom: bytes
) -> dict[str, int]:
    stride = 7 * 4
    require(len(records_bytes) == 960 * stride, "div record byte count drift")
    require(len(probe_output) == OUTPUT_BYTES, "probe output byte count drift")
    reductions = reconstructed_reductions(input_bytes)
    require(len(reductions) == 960, "reconstructed reduction count drift")
    rcp_product_mismatches = 0
    strict_word_mismatches = 0
    strict_half_mismatches = 0
    unsupported = 0
    for item, expected_reduction in enumerate(reductions):
        record = struct.unpack_from("<7I", records_bytes, item * stride)
        require(record[:2] == expected_reduction, f"div reduction drift at record {item}")
        numerator = bits_f32(record[0])
        denominator = bits_f32(record[1])
        exponent = (record[1] >> 23) & 0xFF
        reciprocal_exponent = (record[3] >> 23) & 0xFF
        unsupported += int(
            bool(record[1] >> 31)
            or exponent in (0, 0xFF)
            or reciprocal_exponent in (0, 0xFF)
        )
        expected_rcp, _ = rcp_from_rom(denominator, rom)
        require(record[3] == expected_rcp, f"RCP ROM replay drift at record {item}")
        expected_product = f32_bits(mul_f32(numerator, bits_f32(expected_rcp)))
        require(record[4] == expected_product, f"RCP product drift at record {item}")
        expected_strict = f32_bits(div_f32(numerator, denominator))
        require(record[5] == expected_strict, f"strict quotient drift at record {item}")
        require(record[6] <= 0xFFFF, f"strict half encoding drift at record {item}")
        require(
            record[6] == half_bits(bits_f32(record[5])),
            f"strict half replay drift at record {item}",
        )
        probe_half = struct.unpack_from("<H", probe_output, item * 2)[0]
        require(
            probe_half == half_bits(bits_f32(record[2])),
            f"approximate half replay drift at record {item}",
        )
        rcp_product_mismatches += record[2] != record[4]
        strict_word_mismatches += record[2] != record[5]
        strict_half_mismatches += probe_half != record[6]
    require(rcp_product_mismatches == 0, "div.approx is not RCP-product exact")
    require(strict_word_mismatches > 0, "precise-division FP32 counterexample vanished")
    require(strict_half_mismatches > 0, "precise-division FP16 counterexample vanished")
    require(unsupported == 0, "live denominator escaped accepted RCP domain")
    return {
        "record_count": 960,
        "div_vs_rcp_product_word_mismatches": rcp_product_mismatches,
        "div_vs_strict_word_mismatches": strict_word_mismatches,
        "div_vs_strict_half_mismatches": strict_half_mismatches,
        "unsupported_denominators": unsupported,
    }


def mismatch_count(left: bytes, right: bytes) -> int:
    require(len(left) == len(right), "mismatch comparison length drift")
    return sum(a != b for a, b in zip(left, right))


def half_mismatch_count(left: bytes, right: bytes) -> int:
    require(len(left) == len(right) and len(left) % 2 == 0, "half comparison drift")
    return sum(
        left[index : index + 2] != right[index : index + 2]
        for index in range(0, len(left), 2)
    )


def validate_run(
    root: Path, run_id: str, backend: str, expected_input: bytes, rom: bytes
) -> dict[str, Any]:
    run_root = root / run_id
    receipt_path = run_root / "run-receipt.json"
    receipt = load_json(receipt_path)
    require(receipt.get("schema") == RUN_SCHEMA, f"{run_id} run schema drift")
    expected_backend = "rtx-cuda-driver" if backend == "rtx" else "mac-metal"
    require(receipt.get("backend") == expected_backend, f"{run_id} backend drift")
    require(receipt.get("run_id") == run_id, f"{run_id} identifier drift")
    require(receipt.get("process_id", 0) > 0, f"{run_id} process id missing")
    require(receipt.get("poison_byte") == CASES[run_id], f"{run_id} poison drift")
    require(receipt.get("input_generator") == GENERATOR, f"{run_id} generator drift")
    require(receipt.get("numerical_domain") == DOMAIN, f"{run_id} domain drift")
    require(receipt.get("address_free_parameter_sha256") == PARAMETER_SHA,
            f"{run_id} parameter identity drift")
    require(receipt.get("abi") == {
        "buffers": ["input_half4", "level0_half4", "level1_half4", "level2_half4", "level3_half4"],
        "level_count": 1, "width": 40, "height": 23, "reserved": 0,
    }, f"{run_id} ABI drift")
    require(receipt.get("input_bytes") == INPUT_BYTES, f"{run_id} input bytes drift")
    require(receipt.get("output_bytes") == OUTPUT_BYTES, f"{run_id} output bytes drift")
    require(receipt.get("input_sha256") == sha256_bytes(expected_input),
            f"{run_id} input receipt hash drift")
    require(receipt.get("input_mismatched_bytes") == 0, f"{run_id} input mutation")
    require(receipt.get("canary_mismatched_bytes") == 0, f"{run_id} canary mutation")
    require(receipt.get("active_untouched_half_words") == 0,
            f"{run_id} active write coverage drift")
    require(receipt.get("inactive_level_mismatched_bytes") == 0,
            f"{run_id} inactive write detected")
    require(receipt.get("external_expected_tensors_consumed") is False,
            f"{run_id} expected tensor consumed")
    require(receipt.get("archived_intermediates_consumed") is False,
            f"{run_id} archived intermediate consumed")
    require(receipt.get("address_fields_recorded") is False,
            f"{run_id} address field flag drift")
    require(receipt.get("result") == "pass", f"{run_id} did not pass")

    input_bytes = (run_root / "input-after.bin").read_bytes()
    output = (run_root / "output.bin").read_bytes()
    require(input_bytes == expected_input, f"{run_id} generated input drift")
    require(len(output) == OUTPUT_BYTES, f"{run_id} output file size drift")
    require(sha256_bytes(output) == receipt.get("output_sha256"),
            f"{run_id} output receipt hash drift")
    poison = bytes([CASES[run_id]])
    require(
        all(output[index : index + 2] != poison * 2 for index in range(0, len(output), 2)),
        f"{run_id} output retains poison half word",
    )
    for level in range(1, 4):
        inactive = (run_root / f"inactive-level{level}-after.bin").read_bytes()
        require(len(inactive) == LEVEL_BYTES[level], f"{run_id} inactive level size drift")
        require(inactive == poison * len(inactive), f"{run_id} inactive level was written")

    result: dict[str, Any] = {
        "run_id": run_id,
        "poison_byte": CASES[run_id],
        "process_id": int(receipt["process_id"]),
        "input_sha256": sha256_bytes(input_bytes),
        "output": output,
        "output_sha256": sha256_bytes(output),
        "run_receipt_sha256": sha256_file(receipt_path),
    }
    if backend == "rtx":
        require(receipt.get("selected_launch_token") == 85, f"{run_id} launch drift")
        require(receipt.get("module_token") == 18, f"{run_id} module token drift")
        require(receipt.get("module_bytes") == 15128, f"{run_id} module bytes drift")
        require(receipt.get("module_sha256") == MODULE_SHA, f"{run_id} module hash drift")
        require(receipt.get("module_transport") == "cuModuleLoadDataEx-exact-selected-module",
                f"{run_id} module transport drift")
        require(receipt.get("packed_parameter_bytes") == 56, f"{run_id} parameter size drift")
        require(receipt.get("grid") == [3, 2, 1], f"{run_id} grid drift")
        require(receipt.get("block") == [256, 1, 1], f"{run_id} block drift")
        probe_receipt_path = run_root / "div-probe-receipt.json"
        probe = load_json(probe_receipt_path)
        require(probe.get("schema") == PROBE_SCHEMA, f"{run_id} probe schema drift")
        require(probe.get("run_id") == run_id, f"{run_id} probe id drift")
        require(probe.get("process_id", 0) > 0, f"{run_id} probe process missing")
        require(probe.get("input_generator") == GENERATOR, f"{run_id} probe generator drift")
        require(probe.get("external_expected_tensors_consumed") is False,
                f"{run_id} probe expected tensor consumed")
        require(probe.get("archived_intermediates_consumed") is False,
                f"{run_id} probe archived tensor consumed")
        require(probe.get("result") == "pass", f"{run_id} probe did not pass")
        probe_input = (run_root / "div-probe-input.bin").read_bytes()
        records = (run_root / "div-records.bin").read_bytes()
        probe_output = (run_root / "probe-output.bin").read_bytes()
        require(probe_input == expected_input, f"{run_id} independent probe input drift")
        require(probe_output == output, f"{run_id} selected module/probe mismatch")
        require(probe.get("input_sha256") == sha256_bytes(probe_input),
                f"{run_id} probe input hash drift")
        require(probe.get("records_sha256") == sha256_bytes(records),
                f"{run_id} probe record hash drift")
        require(probe.get("probe_output_sha256") == sha256_bytes(probe_output),
                f"{run_id} probe output hash drift")
        stats = verify_div_records(records, probe_output, probe_input, rom)
        for key, value in stats.items():
            require(probe.get(key) == value, f"{run_id} probe statistic drift: {key}")
        result.update({
            "probe_process_id": int(probe["process_id"]),
            "probe_receipt_sha256": sha256_file(probe_receipt_path),
            "records_sha256": sha256_bytes(records),
            "div_stats": stats,
        })
    else:
        require(receipt.get("source_launch_token") == 85, f"{run_id} source launch drift")
        require(receipt.get("source_module_token") == 18, f"{run_id} source module drift")
        require(receipt.get("source_module_bytes") == 15128, f"{run_id} source bytes drift")
        require(receipt.get("source_module_sha256") == MODULE_SHA, f"{run_id} source hash drift")
        require(receipt.get("metal_function") == "fg_postprocess_l85_exact",
                f"{run_id} Metal entry drift")
        require(receipt.get("negative_control_function") == "fg_postprocess_l85_strict_division",
                f"{run_id} negative entry drift")
        require(receipt.get("architecture_rom") == {
            "kind": "sm120-mufu-rcp-2bit-correction", "bytes": 2097152,
            "sha256": ROM_SHA, "model_values": 0, "captured_intermediates": 0,
        }, f"{run_id} architecture ROM provenance drift")
        require(receipt.get("architecture_rom_mismatched_bytes") == 0,
                f"{run_id} architecture ROM mutated")
        require(receipt.get("strict_untouched_half_words") == 0,
                f"{run_id} strict output write coverage drift")
        strict = (run_root / "strict-division-output.bin").read_bytes()
        require(len(strict) == OUTPUT_BYTES, f"{run_id} strict output size drift")
        require(receipt.get("strict_division_output_sha256") == sha256_bytes(strict),
                f"{run_id} strict output hash drift")
        result.update({"strict_output": strict, "strict_output_sha256": sha256_bytes(strict)})
    return result


def validate_backend(root: Path, backend: str, rom: bytes) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    expected_input = generate_input()
    runs = [validate_run(root, run_id, backend, expected_input, rom) for run_id in CASES]
    process_ids = {run["process_id"] for run in runs}
    require(len(process_ids) == 4, f"{backend} fresh selected process gate failed")
    if backend == "rtx":
        require(len({run["probe_process_id"] for run in runs}) == 4,
                "RTX fresh div probe process gate failed")
    outputs = {run["output_sha256"] for run in runs}
    require(len(outputs) == 1, f"{backend} output repeat/poison invariance failed")
    if backend == "mac":
        require(len({run["strict_output_sha256"] for run in runs}) == 1,
                "Mac strict control repeat/poison invariance failed")
    normalized_runs = []
    for run in runs:
        item = {
            "run_id": run["run_id"],
            "poison_byte": run["poison_byte"],
            "input_sha256": run["input_sha256"],
            "output_sha256": run["output_sha256"],
            "run_receipt_sha256": run["run_receipt_sha256"],
        }
        if backend == "rtx":
            item.update({
                "div_probe_receipt_sha256": run["probe_receipt_sha256"],
                "div_records_sha256": run["records_sha256"],
            })
        else:
            item["strict_division_output_sha256"] = run["strict_output_sha256"]
        normalized_runs.append(item)
    receipt: dict[str, Any] = {
        "schema": BACKEND_SCHEMA,
        "backend": backend,
        "selected_launch_token": 85,
        "module_token": 18,
        "module_sha256": MODULE_SHA,
        "address_free_parameter_sha256": PARAMETER_SHA,
        "input_generator": GENERATOR,
        "numerical_domain": DOMAIN,
        "input_sha256": sha256_bytes(expected_input),
        "output_sha256": runs[0]["output_sha256"],
        "fresh_process_count": 4,
        "poison_values": [0x5A, 0xA5],
        "repeat_equal": True,
        "poison_invariant": True,
        "full_active_write_coverage": True,
        "inactive_levels_unchanged": True,
        "input_immutable": True,
        "canaries_intact": True,
        "runs": normalized_runs,
        "external_expected_tensors_consumed": False,
        "archived_intermediates_consumed": False,
        "address_fields_recorded": False,
        "result": "pass",
    }
    if backend == "rtx":
        receipt.update({
            "module_transport": "cuModuleLoadDataEx-exact-selected-module",
            "div_approx_characterization": runs[0]["div_stats"],
            "div_probe_fresh_process_count": 4,
            "selected_module_matches_independent_div_probe": True,
        })
    else:
        receipt.update({
            "metal_function": "fg_postprocess_l85_exact",
            "architecture_rom_sha256": ROM_SHA,
            "architecture_rom_model_values": 0,
            "architecture_rom_captured_intermediates": 0,
            "strict_division_output_sha256": runs[0]["strict_output_sha256"],
        })
    receipt["receipt_root_sha256"] = sha256_bytes(canonical_bytes(receipt))
    require_no_addresses(receipt, f"{backend} backend receipt")
    return receipt, runs


def backend_command(args: argparse.Namespace) -> None:
    rom = load_rom(args.rcp_rom)
    receipt, _ = validate_backend(args.root, args.backend, rom)
    write_json(args.output, receipt)
    print(f"backend={args.backend}")
    print(f"receipt_root_sha256={receipt['receipt_root_sha256']}")
    print("result=pass")


def final_command(args: argparse.Namespace) -> None:
    rom = load_rom(args.rcp_rom)
    rtx_receipt, rtx_runs = validate_backend(args.rtx_root, "rtx", rom)
    mac_receipt, mac_runs = validate_backend(args.mac_root, "mac", rom)
    exact_byte_mismatches = mismatch_count(rtx_runs[0]["output"], mac_runs[0]["output"])
    strict_half_mismatches = half_mismatch_count(
        rtx_runs[0]["output"], mac_runs[0]["strict_output"]
    )
    require(exact_byte_mismatches == 0, "RTX/Metal exact output mismatch")
    require(strict_half_mismatches > 0, "ordinary Metal division counterexample vanished")
    receipt: dict[str, Any] = {
        "schema": FINAL_SCHEMA,
        "selected_launch_token": 85,
        "module_token": 18,
        "module_sha256": MODULE_SHA,
        "address_free_parameter_sha256": PARAMETER_SHA,
        "abi": {
            "packed_bytes": 56,
            "buffers": ["input_half4", "level0_half4", "level1_half4", "level2_half4", "level3_half4"],
            "level_count": 1, "width": 40, "height": 23, "reserved": 0,
        },
        "input_generator": GENERATOR,
        "numerical_domain": DOMAIN,
        "input_sha256": rtx_receipt["input_sha256"],
        "output_sha256": rtx_receipt["output_sha256"],
        "output_bytes": OUTPUT_BYTES,
        "rtx_backend_root_sha256": rtx_receipt["receipt_root_sha256"],
        "mac_backend_root_sha256": mac_receipt["receipt_root_sha256"],
        "rtx_fresh_process_count": 4,
        "mac_fresh_process_count": 4,
        "two_poison_values": [0x5A, 0xA5],
        "rtx_mac_mismatched_bytes": exact_byte_mismatches,
        "ordinary_division_mismatched_half_values": strict_half_mismatches,
        "div_approx_characterization": rtx_receipt["div_approx_characterization"],
        "architecture_rom": {
            "sha256": ROM_SHA,
            "bytes": len(rom),
            "model_values": 0,
            "captured_intermediates": 0,
        },
        "gates": {
            "exact_selected_module_execution": True,
            "independent_backend_inputs": True,
            "fresh_process_repeat": True,
            "two_poison_invariance": True,
            "full_active_write_coverage": True,
            "inactive_levels_unchanged": True,
            "input_immutable": True,
            "canaries_intact": True,
            "ordinary_division_rejected": True,
            "no_expected_output_ingress": True,
            "no_archived_intermediate_ingress": True,
            "address_free": True,
        },
        "claims": {
            "launch85_dependency_closed_parity": True,
            "complete_generated_frame_parity": False,
            "netal_integration": False,
            "performance_advantage": False,
        },
        "result": "pass",
    }
    receipt["receipt_root_sha256"] = sha256_bytes(canonical_bytes(receipt))
    write_json(args.output, receipt)
    print(f"output_sha256={receipt['output_sha256']}")
    print(f"ordinary_division_mismatched_half_values={strict_half_mismatches}")
    print(f"receipt_root_sha256={receipt['receipt_root_sha256']}")
    print("result=pass")


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    backend = subparsers.add_parser("backend")
    backend.add_argument("--backend", choices=("rtx", "mac"), required=True)
    backend.add_argument("--root", type=Path, required=True)
    backend.add_argument("--output", type=Path, required=True)
    backend.add_argument("--rcp-rom", type=Path)
    backend.set_defaults(handler=backend_command)
    final = subparsers.add_parser("final")
    final.add_argument("--rtx-root", type=Path, required=True)
    final.add_argument("--mac-root", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    final.add_argument("--rcp-rom", type=Path)
    final.set_defaults(handler=final_command)
    args = parser.parse_args()
    try:
        args.handler(args)
        return 0
    except VerificationError as error:
        print(f"verification_error={error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
