#!/usr/bin/env python3
"""Verify standalone module 10 and the connected six-launch distortion path."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import struct
from pathlib import Path
from typing import Any, Callable

M10_SCHEMA = "neuralstudio.fg.distortion-module10-run.v1"
CHAIN_SCHEMA = "neuralstudio.fg.distortion-six-launch-run.v1"
M10_MODULE = "0f7627574281429873003bd700c909d09c895770d8464af0914559d866fe9d3c"
M9_MODULE = "6c507cc85455aeaed386e293c77aa65c861e0093eebd4e2ca220fe57636319ed"
M25_MODULE = "28fe0e153d1ffc80b54995c022f18395c5556f3c041cc9d724bfe7e70dc4e824"
MODULES = [M10_MODULE, M9_MODULE, M25_MODULE]
M10_RAW = [
    "e170b309e44af03949713aa49e319aa27448cc33a002f7a415006d2b3acbfa71",
    "9c50edeea15f0b497785e9cb6a7e8ca9324992df32f12ce3bba553f55199b4a5",
]
M10_ADDRESS_FREE = [
    "a9709a973e4e90362fe8752cfaf629fe4653a2258ffc936a7e58c1bed0b59c16",
    "5e7ccebe87f838586b00a647ff7f463d3626a9c5f13432bcb6cab96e12276f1e",
]
M10_FIXTURE_PARAMETER = "bedcc830e93471dc1685e06e218bad3a5b718f3b02ff720ad51ce555319ca514"
M10_GENERATOR = "fg-distortion-module10-independent-nonzero-v1"
M10_OUTPUT = "912db946ff9812ddeee132e6752f21326b4e11f944cab938e9a4bfc6669a237c"
CHAIN_GENERATOR = "fg-distortion-six-launch-independent-nonzero-v1"
CHAIN_TOKENS = [
    "launch-000005",
    "launch-000009",
    "launch-000010",
    "launch-000016",
    "launch-000020",
    "launch-000021",
]
CHAIN_RAW = [
    "e170b309e44af03949713aa49e319aa27448cc33a002f7a415006d2b3acbfa71",
    "b695e0e6d29050462620ad1f54b593dd57b12a616cb58e2367ffef10a2444fe2",
    "98f06561b037fdfab9742649c3a82e58c39de0038b5df10f638ede5df5aa6b55",
    "9c50edeea15f0b497785e9cb6a7e8ca9324992df32f12ce3bba553f55199b4a5",
    "723e2c4a348298753ace7172a6e1452e22faf4518c10952ab7857f7eaf8406c3",
    "a765a6a6e229ab59eb65faf1393caf9ccff44d949b23f9af11ad0d957f77ce4c",
]
IMMUTABLE_INPUTS = [
    "0c6c3dc8eaa243e9c05d44c8696d41064efcaf10d80f182575c02766c8d775ce",
    "3837bc9094e6544f89198ad715fccb48ab9872df7ab26dfd40713cfddd316cb7",
    "1848edda4ae9b5898f0ec992d7f3ee265094b4922e6a4080fe09a99b42e54878",
    "d24a67f64ee5bb0aedc0addfa081d85c4a0e859a8a619bc696f10d531378725a",
    "eaaa49734088ddd7b26038c5e760812e3f5d4d24fe3d5fe7591548d5d280c53e",
]
INITIAL_STATE = [
    "1615574836d64567e14a4c9d72fc66c7435600ba8d191cb3beee683861a20969",
    "9eef819d5229a50212d82966f446237c06463d35fcb566a12b235b138c4c8fb2",
]
CHECKPOINT_FILES = [
    "reset-m10.rgba16f",
    "reset-m9-a.rg16f",
    "reset-m9-b.rg16f",
    "reset-m9-mask.r32u",
    "reset-m25.r8",
    "generated-m10.rgba16f",
    "generated-m9-a.rg16f",
    "generated-m9-b.rg16f",
    "generated-m9-mask.r32u",
    "generated-m25.r8",
]
CHECKPOINT_BYTES = [
    460800,
    230400,
    230400,
    230400,
    57600,
    460800,
    230400,
    230400,
    230400,
    57600,
]
FULL_HASHES = {
    0: "0c6c3dc8eaa243e9c05d44c8696d41064efcaf10d80f182575c02766c8d775ce",
    1: "e9fc67c488e971aa5383e6cf75414732dcd7760784d7a0d42903981c0bfc60fe",
    2: "a02e7bb98d439e26c6a3094ef61447e0a4d080b201099c3f26333a79d59433cb",
    3: "84b82105e822a9adaf2f1f390a3ad831841f2179c66a47a9956acccb2bd55305",
    5: "3837bc9094e6544f89198ad715fccb48ab9872df7ab26dfd40713cfddd316cb7",
    6: "ebff8e580a879b3bdedebd170017c79735807eca7e650a6b696879ce1b4f65b2",
    7: "e7368ffdcd91a53a95f61d8884206f885ac64791b16ebba19e902743107f18a9",
    8: "250b702b3931ed69228c081257c0f5b1d523c9ef41972351c450ce392ddfc5b5",
}
SCATTER_HASHES = {
    165: {
        4: "a45a757a62dccb4007db7274e2b94199004050878a041dba4da5f4e0a63f139f",
        9: "5458baf7b723d5dad8088cfd61c74ea7f4358a98d4a814b45d129e3ff91edf3d",
    },
    90: {
        4: "214b1a0ee89c409ed53a0dcef936169bfc26f0f4536a1de09dc7e86de5cede93",
        9: "b4391f2c5605c50df929f2dc1dd91c9103c4638254c39d7892a08222204179c1",
    },
}
COVERAGE = [
    "9e1be38c4114040430f18747bf107ca102aa25f0cf9f569a5bfca34cd5a41dd7",
    "8e5277d84b596265902e1437f9b9025dfbf7b76071c3d94b4c130313467dda02",
]
WRITTEN = [57527, 57422]
ROM_SHA256 = "7cf4f8cece6b18b07605edc9cb601a153346a79b10bf3b93c7ec058d57d29750"
PROBE_SHA256 = "11edf20b2d040567f0edf5539cd9f5ae71593307b14dd6f90084dc096320e5d3"
PROBE_BYTES = 2 * 320 * 180 * 48
RUNS_M10 = tuple(
    (f"m10-poison-{poison}-{process}", int(poison, 16))
    for poison in ("a5", "5a")
    for process in ("a", "b")
)
RUNS_CHAIN = tuple(
    (f"distortion-six-poison-{poison}-{process}", int(poison, 16))
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
    return sha256_bytes(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
    )


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


def address_free(value: dict[str, Any]) -> None:
    encoded = json.dumps(value).lower()
    for forbidden in (
        "gpu_virtual_address",
        "device_pointer",
        "host_pointer",
        "descriptor_ptr",
    ):
        require(forbidden not in encoded, f"address field present: {forbidden}")


def validate_m10_run(
    root: Path, run_id: str, poison: int, backend: str
) -> tuple[dict[str, Any], bytes]:
    run_root = root / run_id
    receipt = load_json(run_root / "run-receipt.json")
    address_free(receipt)
    require(
        receipt.get("schema") == M10_SCHEMA
        and receipt.get("backend") == backend
        and receipt.get("run_id") == run_id
        and receipt.get("poison_byte") == poison
        and isinstance(receipt.get("process_id"), int),
        f"{run_id}: identity",
    )
    require(
        receipt.get("module_sha256") == M10_MODULE
        and receipt.get("module_bytes") == 5984
        and receipt.get("selected_launch_tokens")
        == ["launch-000005", "launch-000016"]
        and receipt.get("selected_phase_raw_parameter_sha256") == M10_RAW
        and receipt.get("selected_phase_address_free_parameter_sha256")
        == M10_ADDRESS_FREE
        and receipt.get("packed_parameter_bytes") == 56
        and receipt.get("fixture_address_free_parameter_sha256")
        == M10_FIXTURE_PARAMETER,
        f"{run_id}: module/ABI",
    )
    require(
        receipt.get("input_generator") == M10_GENERATOR
        and receipt.get("input_pre_sha256")
        == receipt.get("input_post_sha256")
        == M10_OUTPUT
        and receipt.get("input_mismatched_bytes") == 0,
        f"{run_id}: input provenance",
    )
    typed = (
        "typed_d3d12_resources"
        if backend == "rtx-d3d12-nvapi"
        else "typed_metal_resources"
    )
    output = (run_root / "output.rgba16f").read_bytes()
    require(
        len(output) == receipt.get("output_bytes") == 460800
        and sha256_bytes(output) == receipt.get("output_sha256") == M10_OUTPUT,
        f"{run_id}: output",
    )
    require(
        receipt.get("texture_canary_mismatched_bytes") == 0
        and receipt.get("output_poison_pixels") == 0
        and receipt.get(typed) is True
        and receipt.get("complete_write_coverage_checked_by_dual_poison") is True
        and receipt.get("expected_output_fed_to_execution") is False
        and receipt.get("archived_intermediates_consumed") is False
        and receipt.get("address_fields_recorded") is False
        and receipt.get("result") == "pass",
        f"{run_id}: gates",
    )
    return receipt, output


def build_m10_backend(root: Path, backend: str) -> dict[str, Any]:
    rows = []
    outputs = []
    for run_id, poison in RUNS_M10:
        receipt, output = validate_m10_run(root, run_id, poison, backend)
        rows.append(
            {
                "run_id": run_id,
                "poison_byte": poison,
                "process_id": receipt["process_id"],
                "output_sha256": receipt["output_sha256"],
            }
        )
        outputs.append(output)
    require(len({row["process_id"] for row in rows}) == 4, "m10 processes")
    require(all(output == outputs[0] for output in outputs), "m10 invariance")
    mutated = bytearray(outputs[0])
    mutated[0] ^= 1
    require(sha256_bytes(mutated) != M10_OUTPUT, "m10 output mutation")
    value = {
        "schema": "neuralstudio.fg.distortion-module10-backend.v1",
        "backend": backend,
        "module_sha256": M10_MODULE,
        "input_generator": M10_GENERATOR,
        "input_sha256": M10_OUTPUT,
        "output_sha256": M10_OUTPUT,
        "runs": rows,
        "fresh_process_count": 4,
        "two_process_repeat_per_poison": True,
        "two_poison_output_invariance": True,
        "input_immutability": True,
        "complete_write_coverage": True,
        "texture_canary_integrity": True,
        "output_mutation_rejected": True,
        "expected_output_fed_to_execution": False,
        "archived_intermediates_consumed": False,
        "address_fields_recorded": False,
        "result": "pass",
    }
    reroot(value)
    return value


def build_m10_final(rtx_root: Path, mac_root: Path) -> dict[str, Any]:
    rtx = build_m10_backend(rtx_root, "rtx-d3d12-nvapi")
    mac = build_m10_backend(mac_root, "mac-metal")
    comparisons = []
    for run_id, poison in RUNS_M10:
        left = (rtx_root / run_id / "output.rgba16f").read_bytes()
        right = (mac_root / run_id / "output.rgba16f").read_bytes()
        require(left == right, f"{run_id}: module10 parity")
        comparisons.append(
            {
                "run_id": run_id,
                "poison_byte": poison,
                "output_bytes": len(left),
                "output_sha256": sha256_bytes(left),
                "mismatched_bytes": 0,
            }
        )
    value = {
        "schema": "neuralstudio.fg.distortion-module10-verification.v1",
        "module_sha256": M10_MODULE,
        "fixture_domain": "selected-live-parameters-nonzero-320x180",
        "comparisons": comparisons,
        "rtx_backend_root_sha256": rtx["receipt_root_sha256"],
        "mac_backend_root_sha256": mac["receipt_root_sha256"],
        "fresh_process_count_per_backend": 4,
        "two_poison_output_invariance": True,
        "complete_write_coverage": True,
        "input_immutability": True,
        "canary_integrity": True,
        "claims": {
            "standalone_module10_exact": True,
            "connected_six_launch_path_exact": False,
            "connected_core_distortion_exact": False,
            "complete_generated_frame_exact": False,
        },
        "result": "pass",
    }
    reroot(value)
    return value


def validate_chain_run(
    root: Path, run_id: str, poison: int, backend: str
) -> tuple[dict[str, Any], list[bytes]]:
    run_root = root / run_id
    receipt = load_json(run_root / "run-receipt.json")
    address_free(receipt)
    require(
        receipt.get("schema") == CHAIN_SCHEMA
        and receipt.get("backend") == backend
        and receipt.get("run_id") == run_id
        and receipt.get("poison_byte") == poison
        and isinstance(receipt.get("process_id"), int),
        f"{run_id}: chain identity",
    )
    require(
        receipt.get("module_sha256") == MODULES
        and receipt.get("launch_sequence") == [10, 9, 25, 10, 9, 25]
        and receipt.get("selected_launch_tokens") == CHAIN_TOKENS
        and receipt.get("selected_raw_parameter_sha256") == CHAIN_RAW,
        f"{run_id}: launch provenance",
    )
    require(
        receipt.get("input_generator") == CHAIN_GENERATOR
        and receipt.get("immutable_input_pre_sha256")
        == receipt.get("immutable_input_post_sha256")
        == IMMUTABLE_INPUTS
        and receipt.get("initial_temporal_state_sha256") == INITIAL_STATE
        and receipt.get("input_mismatched_bytes") == 0
        and receipt.get("handoff_mismatched_bytes") == 0,
        f"{run_id}: inputs/handoffs",
    )
    checkpoints = []
    hashes = []
    for index, filename in enumerate(CHECKPOINT_FILES):
        data = (run_root / filename).read_bytes()
        require(len(data) == CHECKPOINT_BYTES[index], f"{run_id}: {filename} size")
        digest = sha256_bytes(data)
        expected = FULL_HASHES.get(index, SCATTER_HASHES[poison].get(index))
        require(digest == expected, f"{run_id}: {filename} identity")
        checkpoints.append(data)
        hashes.append(digest)
    require(
        hashes == receipt.get("checkpoint_sha256")
        and receipt.get("checkpoint_bytes") == CHECKPOINT_BYTES,
        f"{run_id}: checkpoint receipt",
    )
    for index, filename in enumerate(
        ("reset-m25.coverage", "generated-m25.coverage")
    ):
        coverage = (run_root / filename).read_bytes()
        require(
            len(coverage) == 57600
            and sha256_bytes(coverage) == COVERAGE[index],
            f"{run_id}: {filename}",
        )
    require(
        receipt.get("reset_coverage_sha256") == COVERAGE[0]
        and receipt.get("generated_coverage_sha256") == COVERAGE[1]
        and [
            receipt.get("reset_written_pixels"),
            receipt.get("generated_written_pixels"),
        ]
        == WRITTEN
        and receipt.get("canary_mismatched_bytes") == 0
        and receipt.get("live_device_handoffs") is True
        and receipt.get("host_roundtrip_used_as_launch_input") is False
        and receipt.get("declared_region_coverage_accounted") is True
        and receipt.get("expected_output_fed_to_execution") is False
        and receipt.get("captured_or_archived_intermediate_consumed") is False
        and receipt.get("address_fields_recorded") is False
        and receipt.get("result") == "pass",
        f"{run_id}: chain gates",
    )
    if backend == "mac-metal":
        require(
            receipt.get("architecture_rom_sha256") == ROM_SHA256
            and receipt.get("rom_mismatched_bytes") == 0,
            f"{run_id}: architecture ROM",
        )
    return receipt, checkpoints


def build_chain_backend(root: Path, backend: str) -> dict[str, Any]:
    rows = []
    by_run: dict[str, list[bytes]] = {}
    for run_id, poison in RUNS_CHAIN:
        receipt, checkpoints = validate_chain_run(root, run_id, poison, backend)
        by_run[run_id] = checkpoints
        rows.append(
            {
                "run_id": run_id,
                "poison_byte": poison,
                "process_id": receipt["process_id"],
                "checkpoint_sha256": receipt["checkpoint_sha256"],
            }
        )
    require(len({row["process_id"] for row in rows}) == 4, "chain processes")
    for poison in (165, 90):
        selected = [row for row in rows if row["poison_byte"] == poison]
        require(
            len(selected) == 2
            and by_run[selected[0]["run_id"]] == by_run[selected[1]["run_id"]],
            f"chain repeat poison {poison}",
        )
    first = by_run[RUNS_CHAIN[0][0]]
    last = by_run[RUNS_CHAIN[2][0]]
    for index in FULL_HASHES:
        require(first[index] == last[index], f"full checkpoint poison {index}")
    mutated = bytearray(first[7])
    mutated[0] ^= 1
    require(sha256_bytes(mutated) != FULL_HASHES[7], "chain output mutation")
    value = {
        "schema": "neuralstudio.fg.distortion-six-launch-backend.v1",
        "backend": backend,
        "module_sha256": MODULES,
        "launch_sequence": [10, 9, 25, 10, 9, 25],
        "selected_launch_tokens": CHAIN_TOKENS,
        "input_generator": CHAIN_GENERATOR,
        "runs": rows,
        "fresh_process_count": 4,
        "two_process_repeat_per_poison": True,
        "two_poison_full_checkpoint_invariance": True,
        "exact_partial_write_coverage": True,
        "input_immutability": True,
        "canary_integrity": True,
        "live_device_handoffs": True,
        "host_roundtrip_used_as_launch_input": False,
        "output_mutation_rejected": True,
        "expected_output_fed_to_execution": False,
        "captured_or_archived_intermediate_consumed": False,
        "address_fields_recorded": False,
        "result": "pass",
    }
    reroot(value)
    return value


def f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def point_coordinate(x: float, y: float, bits: int) -> tuple[int, int] | None:
    if not (0.0 <= x < 1.0 and 0.0 <= y < 1.0):
        return None
    scale = 1 << bits
    return ((int(x * scale) * 320) >> bits, (int(y * scale) * 180) >> bits)


def decoded_coordinate(x: int, y: int) -> tuple[int, int] | None:
    return None if x == 0 and y == 0 else (x - 1, y - 1)


def build_coordinate(
    probe_path: Path, rtx_root: Path, rejected_mac_root: Path
) -> dict[str, Any]:
    probe = probe_path.read_bytes()
    require(
        len(probe) == PROBE_BYTES and sha256_bytes(probe) == PROBE_SHA256,
        "connected coordinate probe identity",
    )
    records = list(struct.iter_unpack("<12I", probe))
    fixed12_counts = []
    fixed21_counts = []
    set_equal = []
    for phase_index, phase in enumerate(("reset", "generated")):
        phase_records = records[
            phase_index * 57600 : (phase_index + 1) * 57600
        ]
        fixed12 = set()
        fixed21 = set()
        for index, record in enumerate(phase_records):
            actual = decoded_coordinate(record[2], record[3])
            x, y = f32(record[10]), f32(record[11])
            if actual != point_coordinate(x, y, 12):
                fixed12.add(index)
            if actual != point_coordinate(x, y, 21):
                fixed21.add(index)
        rtx = (rtx_root / RUNS_CHAIN[0][0] / f"{phase}-m9-b.rg16f").read_bytes()
        rejected = (
            rejected_mac_root / RUNS_CHAIN[0][0] / f"{phase}-m9-b.rg16f"
        ).read_bytes()
        output_mismatch = {
            index
            for index in range(57600)
            if rtx[index * 4 : index * 4 + 4]
            != rejected[index * 4 : index * 4 + 4]
        }
        require(not fixed21, f"{phase}: fixed21 mapping")
        require(fixed12 == output_mismatch, f"{phase}: fixed12/output set")
        fixed12_counts.append(len(fixed12))
        fixed21_counts.append(len(fixed21))
        set_equal.append(True)
    require(fixed12_counts == [16340, 11289], "fixed12 census")
    value = {
        "schema": "neuralstudio.fg.distortion-connected-coordinate-map.v1",
        "probe_sha256": PROBE_SHA256,
        "probe_bytes": PROBE_BYTES,
        "sample_record_bytes": 48,
        "sample_count_per_phase": 57600,
        "phases": ["reset", "generated"],
        "sampled_resource_format": "RGBA16_FLOAT",
        "sampled_resource_extent": [320, 180],
        "fixed12_mismatched_positions": fixed12_counts,
        "fixed21_mismatched_positions": fixed21_counts,
        "fixed12_and_rejected_output_position_sets_equal": set_equal,
        "selected_mapping": "21-bit normalized coordinate grid",
        "mapping_scope": "module9 second path-B lookup on connected fixture",
        "probe_fed_to_model_execution": False,
        "expected_or_archived_output_consumed_by_execution": False,
        "address_fields_recorded": False,
        "result": "pass",
    }
    reroot(value)
    return value


def validate_coordinate(value: dict[str, Any]) -> None:
    address_free(value)
    require(
        rooted(value)
        and value.get("probe_sha256") == PROBE_SHA256
        and value.get("fixed12_mismatched_positions") == [16340, 11289]
        and value.get("fixed21_mismatched_positions") == [0, 0]
        and value.get("fixed12_and_rejected_output_position_sets_equal")
        == [True, True]
        and value.get("probe_fed_to_model_execution") is False
        and value.get("result") == "pass",
        "connected coordinate receipt",
    )


CLAIMS = {
    "standalone_module10_exact": True,
    "standalone_module9_exact": True,
    "connected_six_launch_distortion_exact": True,
    "reset_and_generated_phase_handoffs_exact": True,
    "connected_core_distortion_exact": False,
    "complete_generated_frame_exact": False,
    "netal_integrated": False,
    "performance_advantage_proven": False,
}


def build_chain_final(
    rtx_root: Path,
    mac_root: Path,
    coordinate: dict[str, Any],
    module10_root: str,
    module9_root: str,
) -> dict[str, Any]:
    validate_coordinate(coordinate)
    rtx = build_chain_backend(rtx_root, "rtx-d3d12-nvapi")
    mac = build_chain_backend(mac_root, "mac-metal")
    comparisons = []
    for run_id, poison in RUNS_CHAIN:
        row = {"run_id": run_id, "poison_byte": poison, "checkpoints": []}
        for filename in CHECKPOINT_FILES + [
            "reset-m25.coverage",
            "generated-m25.coverage",
        ]:
            left = (rtx_root / run_id / filename).read_bytes()
            right = (mac_root / run_id / filename).read_bytes()
            require(left == right, f"{run_id}: {filename} parity")
            row["checkpoints"].append(
                {
                    "name": filename,
                    "bytes": len(left),
                    "sha256": sha256_bytes(left),
                    "mismatched_bytes": 0,
                }
            )
        comparisons.append(row)
    value = {
        "schema": "neuralstudio.fg.distortion-six-launch-verification.v1",
        "module_sha256": MODULES,
        "launch_sequence": [10, 9, 25, 10, 9, 25],
        "fixture_domain": "independent-nonzero-two-phase-320x180",
        "comparisons": comparisons,
        "rtx_backend_root_sha256": rtx["receipt_root_sha256"],
        "mac_backend_root_sha256": mac["receipt_root_sha256"],
        "coordinate_map_root_sha256": coordinate["receipt_root_sha256"],
        "module10_verification_root_sha256": module10_root,
        "module9_verification_root_sha256": module9_root,
        "fresh_process_count_per_backend": 4,
        "two_process_repeat_per_poison": True,
        "two_poison_full_checkpoint_invariance": True,
        "exact_partial_write_coverage": True,
        "input_immutability": True,
        "canary_integrity": True,
        "live_device_handoffs": True,
        "host_roundtrip_used_as_launch_input": False,
        "expected_output_fed_to_execution": False,
        "captured_or_archived_intermediate_consumed": False,
        "address_fields_recorded": False,
        "claims": CLAIMS,
        "result": "pass",
    }
    reroot(value)
    return value


def validate_final(value: dict[str, Any], coordinate_root: str) -> None:
    address_free(value)
    require(
        rooted(value)
        and value.get("module_sha256") == MODULES
        and value.get("launch_sequence") == [10, 9, 25, 10, 9, 25]
        and value.get("coordinate_map_root_sha256") == coordinate_root
        and len(value.get("comparisons", [])) == 4,
        "connected final identity",
    )
    for row in value["comparisons"]:
        require(
            len(row.get("checkpoints", [])) == 12
            and all(
                item.get("mismatched_bytes") == 0
                for item in row["checkpoints"]
            ),
            f"{row.get('run_id')}: comparisons",
        )
    for gate in (
        "two_process_repeat_per_poison",
        "two_poison_full_checkpoint_invariance",
        "exact_partial_write_coverage",
        "input_immutability",
        "canary_integrity",
        "live_device_handoffs",
    ):
        require(value.get(gate) is True, f"connected final gate: {gate}")
    require(
        value.get("host_roundtrip_used_as_launch_input") is False
        and value.get("expected_output_fed_to_execution") is False
        and value.get("captured_or_archived_intermediate_consumed") is False
        and value.get("claims") == CLAIMS
        and value.get("result") == "pass",
        "connected final scope or ingress",
    )


def build_audit(value: dict[str, Any], coordinate_root: str) -> dict[str, Any]:
    validate_final(value, coordinate_root)
    mutations: list[tuple[str, Callable[[dict[str, Any]], None]]] = [
        (
            "checkpoint_mismatch",
            lambda item: item["comparisons"][0]["checkpoints"][7].__setitem__(
                "mismatched_bytes", 1
            ),
        ),
        ("module_drift", lambda item: item["module_sha256"].__setitem__(1, "0" * 64)),
        ("sequence_drift", lambda item: item["launch_sequence"].__setitem__(0, 9)),
        (
            "coordinate_provenance",
            lambda item: item.__setitem__("coordinate_map_root_sha256", "0" * 64),
        ),
        ("coverage_gate", lambda item: item.__setitem__("exact_partial_write_coverage", False)),
        ("canary_gate", lambda item: item.__setitem__("canary_integrity", False)),
        ("handoff_gate", lambda item: item.__setitem__("live_device_handoffs", False)),
        (
            "host_roundtrip",
            lambda item: item.__setitem__("host_roundtrip_used_as_launch_input", True),
        ),
        (
            "expected_ingress",
            lambda item: item.__setitem__("expected_output_fed_to_execution", True),
        ),
        (
            "captured_ingress",
            lambda item: item.__setitem__(
                "captured_or_archived_intermediate_consumed", True
            ),
        ),
        (
            "core_overclaim",
            lambda item: item["claims"].__setitem__(
                "connected_core_distortion_exact", True
            ),
        ),
        (
            "frame_overclaim",
            lambda item: item["claims"].__setitem__(
                "complete_generated_frame_exact", True
            ),
        ),
    ]
    outcomes = []
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
        "schema": "neuralstudio.fg.distortion-six-launch-negative-audit.v1",
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
    backend10 = commands.add_parser("module10-backend")
    backend10.add_argument("--backend", required=True)
    backend10.add_argument("--root", type=Path, required=True)
    backend10.add_argument("--output", type=Path, required=True)
    final10 = commands.add_parser("module10-final")
    final10.add_argument("--rtx-root", type=Path, required=True)
    final10.add_argument("--mac-root", type=Path, required=True)
    final10.add_argument("--output", type=Path, required=True)
    backend6 = commands.add_parser("connected-backend")
    backend6.add_argument("--backend", required=True)
    backend6.add_argument("--root", type=Path, required=True)
    backend6.add_argument("--output", type=Path, required=True)
    coordinate = commands.add_parser("coordinate")
    coordinate.add_argument("--probe", type=Path, required=True)
    coordinate.add_argument("--rtx-root", type=Path, required=True)
    coordinate.add_argument("--rejected-mac-root", type=Path, required=True)
    coordinate.add_argument("--output", type=Path, required=True)
    final6 = commands.add_parser("connected-final")
    final6.add_argument("--rtx-root", type=Path, required=True)
    final6.add_argument("--mac-root", type=Path, required=True)
    final6.add_argument("--coordinate-receipt", type=Path, required=True)
    final6.add_argument("--module10-root", required=True)
    final6.add_argument("--module9-root", required=True)
    final6.add_argument("--output", type=Path, required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--accepted", type=Path, required=True)
    audit.add_argument("--coordinate-receipt", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        if arguments.command == "module10-backend":
            value = build_m10_backend(arguments.root, arguments.backend)
        elif arguments.command == "module10-final":
            value = build_m10_final(arguments.rtx_root, arguments.mac_root)
        elif arguments.command == "connected-backend":
            value = build_chain_backend(arguments.root, arguments.backend)
        elif arguments.command == "coordinate":
            value = build_coordinate(
                arguments.probe, arguments.rtx_root, arguments.rejected_mac_root
            )
        elif arguments.command == "connected-final":
            value = build_chain_final(
                arguments.rtx_root,
                arguments.mac_root,
                load_json(arguments.coordinate_receipt),
                arguments.module10_root,
                arguments.module9_root,
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
