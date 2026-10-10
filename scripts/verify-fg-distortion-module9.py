#!/usr/bin/env python3
"""Build and validate immutable receipts for standalone distortion module 9."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import struct
from pathlib import Path
from typing import Any, Callable

RUN_SCHEMA = "neuralstudio.fg.distortion-module9-run.v1"
BACKEND_SCHEMA = "neuralstudio.fg.distortion-module9-backend.v1"
COORDINATE_SCHEMA = "neuralstudio.fg.distortion-module9-coordinate-map.v1"
FINAL_SCHEMA = "neuralstudio.fg.distortion-module9-verification.v1"
MODULE_SHA256 = "6c507cc85455aeaed386e293c77aa65c861e0093eebd4e2ca220fe57636319ed"
MODULE_BYTES = 6064
RAW_PARAMETERS = [
    "b695e0e6d29050462620ad1f54b593dd57b12a616cb58e2367ffef10a2444fe2",
    "723e2c4a348298753ace7172a6e1452e22faf4518c10952ab7857f7eaf8406c3",
]
FROZEN_ADDRESS_FREE = "f68940013a394407574fc47f1a4399c85e0a4b00203d34c78079577aa2b50187"
FIXTURE_ADDRESS_FREE = "b8066c598cb5dd1b198b9a5284580c0e3ee7e8df0d84e39d4f2c1a4a277edb3c"
GENERATOR = "fg-distortion-module9-independent-nonzero-v1"
INPUT_SHA256 = [
    "4784e143c7dde4b5f19ed2ee41b6bfddefce347c3c8b8c6b2e13c6fefd89551e",
    "911e2f836550408c3c3441e12ecd5a23dff89e28c8b5e66fffb56036d268126c",
    "dbbaa7d20954dcefe52b03d3945b89f00f339a9f2cab2abc834fc22afdced83c",
    "37f6e453c0f5469a3459449f309146775d8cb9cfde5c6a8da86d39b7fb1a8612",
    "9eef819d5229a50212d82966f446237c06463d35fcb566a12b235b138c4c8fb2",
]
OUTPUTS = {
    "out-a.rg16f": "e2cced073a9683ce29b3d2277bb38f04fc05fd231dd03173aed3c3033920276c",
    "out-b.rg16f": "57f6291a3a5e7f7d290c80d6ea9ec2d417cc12275d7011a50799696ec8b08f3f",
    "out-mask.r32u": "928231281ca7a6478e927eec40a8e0a3be0e8db6df775c060199f0fc4c03e9ca",
}
OUTPUT_KEYS = {
    "out-a.rg16f": "out_a",
    "out-b.rg16f": "out_b",
    "out-mask.r32u": "out_mask",
}
OUTPUT_BYTES = 320 * 180 * 4
PROBE_SHA256 = "7a6ad3bbc3c0f2398ce296d6271e6f4c2ac80452b2dc4ce2d8326d78aa0aab92"
PROBE_BYTES = 320 * 180 * 32
RUNS = tuple(
    (f"m9-poison-{poison}-{process}", int(poison, 16))
    for poison in ("a5", "5a")
    for process in ("a", "b")
)


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_root(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return sha256_bytes(encoded)


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise VerificationError(f"cannot load {path}: {error}") from error
    require(isinstance(value, dict), f"{path}: expected object")
    return value


def reroot(value: dict[str, Any]) -> None:
    value.pop("receipt_root_sha256", None)
    value["receipt_root_sha256"] = canonical_root(value)


def rooted(value: dict[str, Any]) -> bool:
    candidate = copy.deepcopy(value)
    recorded = candidate.pop("receipt_root_sha256", None)
    return recorded == canonical_root(candidate)


def require_address_free(value: dict[str, Any]) -> None:
    encoded = json.dumps(value, sort_keys=True).lower()
    for forbidden in (
        "gpu_virtual_address",
        "device_pointer",
        "host_pointer",
        "descriptor_ptr",
        "descriptor_handle_value",
    ):
        require(forbidden not in encoded, f"address field present: {forbidden}")


def validate_run(
    root: Path, run_id: str, poison: int, backend: str
) -> tuple[dict[str, Any], dict[str, bytes]]:
    run_root = root / run_id
    receipt = load_json(run_root / "run-receipt.json")
    require_address_free(receipt)
    require(
        receipt.get("schema") == RUN_SCHEMA
        and receipt.get("backend") == backend
        and receipt.get("run_id") == run_id
        and receipt.get("poison_byte") == poison,
        f"{run_id}: run identity",
    )
    require(isinstance(receipt.get("process_id"), int), f"{run_id}: process id")
    require(
        receipt.get("module_sha256") == MODULE_SHA256
        and receipt.get("module_bytes") == MODULE_BYTES
        and receipt.get("selected_launch_tokens")
        == ["launch-000009", "launch-000020"],
        f"{run_id}: selected module provenance",
    )
    require(
        receipt.get("selected_phase_raw_parameter_sha256") == RAW_PARAMETERS
        and receipt.get("frozen_reset_address_free_parameter_sha256")
        == FROZEN_ADDRESS_FREE
        and receipt.get("fixture_address_free_parameter_sha256")
        == FIXTURE_ADDRESS_FREE
        and receipt.get("packed_parameter_bytes") == 72,
        f"{run_id}: ABI provenance",
    )
    require(
        receipt.get("input_generator") == GENERATOR
        and receipt.get("input_pre_sha256") == INPUT_SHA256
        and receipt.get("input_post_sha256") == INPUT_SHA256
        and receipt.get("input_mismatched_bytes") == 0,
        f"{run_id}: input provenance or mutation",
    )
    require(
        receipt.get("texture_canary_mismatched_bytes") == 0,
        f"{run_id}: canary",
    )
    typed_key = (
        "typed_d3d12_resources"
        if backend == "rtx-d3d12-nvapi"
        else "typed_metal_resources"
    )
    require(receipt.get(typed_key) is True, f"{run_id}: typed resources")
    require(
        receipt.get("complete_write_coverage_checked_by_dual_poison") is True
        and receipt.get("coordinate_probe_consumed_by_execution") is False
        and receipt.get("expected_output_fed_to_execution") is False
        and receipt.get("archived_intermediates_consumed") is False
        and receipt.get("address_fields_recorded") is False
        and receipt.get("result") == "pass",
        f"{run_id}: correctness gates",
    )

    outputs: dict[str, bytes] = {}
    for filename, expected_sha256 in OUTPUTS.items():
        key = OUTPUT_KEYS[filename]
        data = (run_root / filename).read_bytes()
        require(
            len(data) == receipt.get(f"{key}_bytes") == OUTPUT_BYTES,
            f"{run_id}: {filename} size",
        )
        require(
            sha256_bytes(data)
            == receipt.get(f"{key}_sha256")
            == expected_sha256,
            f"{run_id}: {filename} identity",
        )
        require(
            receipt.get(f"{key}_poison_pixels") == 0,
            f"{run_id}: {filename} write coverage",
        )
        outputs[filename] = data
    return receipt, outputs


def build_backend(root: Path, backend: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    outputs: list[dict[str, bytes]] = []
    for run_id, poison in RUNS:
        receipt, current = validate_run(root, run_id, poison, backend)
        outputs.append(current)
        rows.append(
            {
                "run_id": run_id,
                "poison_byte": poison,
                "process_id": receipt["process_id"],
                **{
                    f"{OUTPUT_KEYS[name]}_sha256": receipt[
                        f"{OUTPUT_KEYS[name]}_sha256"
                    ]
                    for name in OUTPUTS
                },
            }
        )
    require(len({row["process_id"] for row in rows}) == 4, "fresh processes")
    require(all(current == outputs[0] for current in outputs), "output invariance")
    mutated = bytearray(outputs[0]["out-b.rg16f"])
    mutated[0] ^= 1
    require(sha256_bytes(mutated) != OUTPUTS["out-b.rg16f"], "output mutation")
    value = {
        "schema": BACKEND_SCHEMA,
        "backend": backend,
        "selected_launch_tokens": ["launch-000009", "launch-000020"],
        "module_token": "module-000009",
        "module_sha256": MODULE_SHA256,
        "fixture_domain": "independent-nonzero-320x180",
        "input_generator": GENERATOR,
        "input_sha256": INPUT_SHA256,
        "runs": rows,
        "fresh_process_count": 4,
        "two_process_repeat_per_poison": True,
        "two_poison_output_invariance": True,
        "input_immutability": True,
        "complete_write_coverage": True,
        "texture_canary_integrity": True,
        "output_mutation_rejected": True,
        "coordinate_probe_consumed_by_execution": False,
        "expected_output_fed_to_execution": False,
        "archived_intermediates_consumed": False,
        "address_fields_recorded": False,
        "result": "pass",
    }
    reroot(value)
    return value


def f32_from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def build_coordinate(
    probe_path: Path, rtx_root: Path, rejected_fixed12_mac_root: Path
) -> dict[str, Any]:
    probe = probe_path.read_bytes()
    require(
        len(probe) == PROBE_BYTES and sha256_bytes(probe) == PROBE_SHA256,
        "coordinate probe identity",
    )
    records = list(struct.iter_unpack("<8I", probe))
    require(len(records) == 320 * 180, "coordinate probe count")
    fixed12_x = fixed12_y = fixed12_both = full_mismatch = 0
    fixed12_positions: set[int] = set()
    for index, record in enumerate(records):
        sampled_x, sampled_y = record[0], record[1]
        coordinate_x = f32_from_bits(record[6])
        coordinate_y = f32_from_bits(record[7])
        require(
            0.0 <= coordinate_x < 1.0 and 0.0 <= coordinate_y < 1.0,
            f"coordinate {index}: unexpected border domain",
        )
        full_x, full_y = int(coordinate_x * 320), int(coordinate_y * 180)
        full_mismatch += full_x != sampled_x or full_y != sampled_y
        fixed_x = (int(coordinate_x * 4096.0) * 320) >> 12
        fixed_y = (int(coordinate_y * 4096.0) * 180) >> 12
        x_mismatch = fixed_x != sampled_x
        y_mismatch = fixed_y != sampled_y
        fixed12_x += x_mismatch
        fixed12_y += y_mismatch
        fixed12_both += x_mismatch and y_mismatch
        if x_mismatch or y_mismatch:
            fixed12_positions.add(index)
    require(full_mismatch == 0, "full-coordinate sampler hypothesis")
    require(
        (len(fixed12_positions), fixed12_x, fixed12_y, fixed12_both)
        == (29400, 15300, 19200, 5100),
        "fixed12 counterexample census",
    )

    rtx = (rtx_root / RUNS[0][0] / "out-b.rg16f").read_bytes()
    rejected = (
        rejected_fixed12_mac_root / RUNS[0][0] / "out-b.rg16f"
    ).read_bytes()
    require(len(rtx) == len(rejected) == OUTPUT_BYTES, "counterexample output size")
    rejected_positions = {
        pixel
        for pixel in range(320 * 180)
        if rtx[pixel * 4 : pixel * 4 + 4]
        != rejected[pixel * 4 : pixel * 4 + 4]
    }
    require(
        rejected_positions == fixed12_positions,
        "probe/output counterexample positions",
    )
    value = {
        "schema": COORDINATE_SCHEMA,
        "probe_sha256": PROBE_SHA256,
        "probe_bytes": PROBE_BYTES,
        "sample_record_bytes": 32,
        "sample_count": 320 * 180,
        "fixture_generator": GENERATOR,
        "sampled_resource_format": "RG16_FLOAT",
        "sampled_resource_extent": [320, 180],
        "full_coordinate_mapping_mismatched_positions": full_mismatch,
        "fixed12_mismatched_positions": len(fixed12_positions),
        "fixed12_x_mismatched_positions": fixed12_x,
        "fixed12_y_mismatched_positions": fixed12_y,
        "fixed12_both_mismatched_positions": fixed12_both,
        "rejected_fixed12_output_b_mismatched_positions": len(
            rejected_positions
        ),
        "probe_and_rejected_output_position_sets_equal": True,
        "selected_mapping": "floor(normalized_coordinate * extent)",
        "mapping_scope": "module9 flow16 lookup on independent nonzero fixture",
        "probe_fed_to_model_execution": False,
        "expected_or_archived_output_consumed_by_execution": False,
        "address_fields_recorded": False,
        "result": "pass",
    }
    reroot(value)
    return value


def validate_coordinate(value: dict[str, Any]) -> None:
    require_address_free(value)
    require(
        rooted(value)
        and value.get("schema") == COORDINATE_SCHEMA
        and value.get("probe_sha256") == PROBE_SHA256
        and value.get("full_coordinate_mapping_mismatched_positions") == 0
        and value.get("fixed12_mismatched_positions") == 29400
        and value.get("probe_and_rejected_output_position_sets_equal") is True
        and value.get("probe_fed_to_model_execution") is False
        and value.get("result") == "pass",
        "coordinate receipt",
    )


def build_final(
    rtx_root: Path, mac_root: Path, coordinate: dict[str, Any]
) -> dict[str, Any]:
    validate_coordinate(coordinate)
    rtx_backend = build_backend(rtx_root, "rtx-d3d12-nvapi")
    mac_backend = build_backend(mac_root, "mac-metal")
    comparisons: list[dict[str, Any]] = []
    for run_id, poison in RUNS:
        row: dict[str, Any] = {"run_id": run_id, "poison_byte": poison}
        for filename, expected_sha256 in OUTPUTS.items():
            rtx = (rtx_root / run_id / filename).read_bytes()
            mac = (mac_root / run_id / filename).read_bytes()
            require(rtx == mac, f"{run_id}: {filename} parity")
            key = OUTPUT_KEYS[filename]
            row[f"{key}_bytes"] = len(rtx)
            row[f"{key}_sha256"] = expected_sha256
            row[f"{key}_mismatched_bytes"] = 0
        comparisons.append(row)
    value = {
        "schema": FINAL_SCHEMA,
        "selected_launch_tokens": ["launch-000009", "launch-000020"],
        "module_token": "module-000009",
        "module_sha256": MODULE_SHA256,
        "abi_bytes": 72,
        "fixture_domain": "independent-nonzero-320x180",
        "comparisons": comparisons,
        "rtx_backend_root_sha256": rtx_backend["receipt_root_sha256"],
        "mac_backend_root_sha256": mac_backend["receipt_root_sha256"],
        "coordinate_map_root_sha256": coordinate["receipt_root_sha256"],
        "fresh_process_count_per_backend": 4,
        "two_process_repeat_per_poison": True,
        "two_poison_output_invariance": True,
        "input_immutability": True,
        "complete_write_coverage": True,
        "texture_canary_integrity": True,
        "coordinate_probe_consumed_by_execution": False,
        "expected_output_fed_to_execution": False,
        "archived_intermediates_consumed": False,
        "address_fields_recorded": False,
        "claims": {
            "standalone_module9_exact_on_fixture_domain": True,
            "both_phase_live_handoffs_proven": False,
            "module10_exact": False,
            "six_launch_distortion_path_exact": False,
            "connected_core_distortion_exact": False,
            "complete_generated_frame_exact": False,
            "netal_integrated": False,
            "performance_advantage_proven": False,
        },
        "result": "pass",
    }
    reroot(value)
    return value


EXPECTED_CLAIMS = {
    "standalone_module9_exact_on_fixture_domain": True,
    "both_phase_live_handoffs_proven": False,
    "module10_exact": False,
    "six_launch_distortion_path_exact": False,
    "connected_core_distortion_exact": False,
    "complete_generated_frame_exact": False,
    "netal_integrated": False,
    "performance_advantage_proven": False,
}


def validate_final(value: dict[str, Any], coordinate_root: str) -> None:
    require_address_free(value)
    require(
        rooted(value)
        and value.get("schema") == FINAL_SCHEMA
        and value.get("module_sha256") == MODULE_SHA256
        and value.get("abi_bytes") == 72
        and value.get("coordinate_map_root_sha256") == coordinate_root,
        "final identity",
    )
    comparisons = value.get("comparisons", [])
    require(len(comparisons) == 4, "comparison count")
    for row in comparisons:
        for filename, expected_sha256 in OUTPUTS.items():
            key = OUTPUT_KEYS[filename]
            require(
                row.get(f"{key}_bytes") == OUTPUT_BYTES
                and row.get(f"{key}_sha256") == expected_sha256
                and row.get(f"{key}_mismatched_bytes") == 0,
                f"{row.get('run_id')}: {key} comparison",
            )
    for gate in (
        "two_process_repeat_per_poison",
        "two_poison_output_invariance",
        "input_immutability",
        "complete_write_coverage",
        "texture_canary_integrity",
    ):
        require(value.get(gate) is True, f"final gate: {gate}")
    require(
        value.get("coordinate_probe_consumed_by_execution") is False
        and value.get("expected_output_fed_to_execution") is False
        and value.get("archived_intermediates_consumed") is False
        and value.get("claims") == EXPECTED_CLAIMS
        and value.get("result") == "pass",
        "final ingress, scope, or result",
    )


def build_audit(value: dict[str, Any], coordinate_root: str) -> dict[str, Any]:
    validate_final(value, coordinate_root)
    mutations: list[tuple[str, Callable[[dict[str, Any]], None]]] = [
        (
            "output_a_mismatch",
            lambda item: item["comparisons"][0].__setitem__(
                "out_a_mismatched_bytes", 1
            ),
        ),
        (
            "output_b_mismatch",
            lambda item: item["comparisons"][0].__setitem__(
                "out_b_mismatched_bytes", 1
            ),
        ),
        (
            "mask_mismatch",
            lambda item: item["comparisons"][0].__setitem__(
                "out_mask_mismatched_bytes", 1
            ),
        ),
        ("module_drift", lambda item: item.__setitem__("module_sha256", "0" * 64)),
        ("abi_drift", lambda item: item.__setitem__("abi_bytes", 64)),
        (
            "coordinate_provenance",
            lambda item: item.__setitem__("coordinate_map_root_sha256", "0" * 64),
        ),
        ("coverage_gate", lambda item: item.__setitem__("complete_write_coverage", False)),
        ("canary_gate", lambda item: item.__setitem__("texture_canary_integrity", False)),
        (
            "probe_ingress",
            lambda item: item.__setitem__("coordinate_probe_consumed_by_execution", True),
        ),
        (
            "expected_ingress",
            lambda item: item.__setitem__("expected_output_fed_to_execution", True),
        ),
        (
            "archive_ingress",
            lambda item: item.__setitem__("archived_intermediates_consumed", True),
        ),
        (
            "scope_expansion",
            lambda item: item["claims"].__setitem__(
                "six_launch_distortion_path_exact", True
            ),
        ),
    ]
    outcomes: list[dict[str, Any]] = []
    for name, mutate in mutations:
        candidate = copy.deepcopy(value)
        mutate(candidate)
        reroot(candidate)
        try:
            validate_final(candidate, coordinate_root)
        except VerificationError:
            outcomes.append({"mutation": name, "rejected": True})
            continue
        raise VerificationError(f"semantic mutation accepted: {name}")
    audit = {
        "schema": "neuralstudio.fg.distortion-module9-negative-audit.v1",
        "accepted_receipt_root_sha256": value["receipt_root_sha256"],
        "coordinate_map_root_sha256": coordinate_root,
        "mutations": outcomes,
        "mutation_count": len(outcomes),
        "all_mutations_rejected": True,
        "address_fields_recorded": False,
        "result": "pass",
    }
    reroot(audit)
    return audit


def write_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    backend = commands.add_parser("backend")
    backend.add_argument("--backend", required=True)
    backend.add_argument("--root", type=Path, required=True)
    backend.add_argument("--output", type=Path, required=True)
    coordinate = commands.add_parser("coordinate")
    coordinate.add_argument("--probe", type=Path, required=True)
    coordinate.add_argument("--rtx-root", type=Path, required=True)
    coordinate.add_argument("--rejected-fixed12-mac-root", type=Path, required=True)
    coordinate.add_argument("--output", type=Path, required=True)
    final = commands.add_parser("final")
    final.add_argument("--rtx-root", type=Path, required=True)
    final.add_argument("--mac-root", type=Path, required=True)
    final.add_argument("--coordinate-receipt", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--accepted", type=Path, required=True)
    audit.add_argument("--coordinate-receipt", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        if arguments.command == "backend":
            value = build_backend(arguments.root, arguments.backend)
        elif arguments.command == "coordinate":
            value = build_coordinate(
                arguments.probe,
                arguments.rtx_root,
                arguments.rejected_fixed12_mac_root,
            )
        elif arguments.command == "final":
            coordinate_value = load_json(arguments.coordinate_receipt)
            value = build_final(
                arguments.rtx_root, arguments.mac_root, coordinate_value
            )
        else:
            coordinate_value = load_json(arguments.coordinate_receipt)
            validate_coordinate(coordinate_value)
            value = build_audit(
                load_json(arguments.accepted),
                coordinate_value["receipt_root_sha256"],
            )
        write_new(arguments.output, value)
        print(f"receipt_root_sha256={value['receipt_root_sha256']}")
        print("result=pass")
        return 0
    except (OSError, VerificationError) as error:
        print(f"verification_error={error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
