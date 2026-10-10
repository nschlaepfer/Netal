#!/usr/bin/env python3
"""Verify standalone canonical L7/L17 module23 RTX/Metal parity."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pathlib
import sys
from typing import Any


MODULE_SHA = "d787cf7b861295d1b4c6a80cb374230de793aa69f288a7e724bf59278bc4a62d"
PARAM_SHA = "afc5eb7348c895a196b685c860392dfecb8b0273b6bf7f204612fe0c3ae9e100"
FIXTURE_PARAM_SHA = "84d38136f78b3d6a04164b9b84e43595704a1c41fb6718fd25d7645410e6aa82"
ROM_SHA = "7cf4f8cece6b18b07605edc9cb601a153346a79b10bf3b93c7ec058d57d29750"
DEAD_PROOF_ROOT = "2ac38a2cf904f07c59bb901b71fb42ca1bd987be2f7f9825daadc5c1ee75ba9c"
LOWERING_ROOT = "c1060299b27119df043914ac2d50fbc28b15857eba2f776da8d332f5c1757d7f"
SOURCE_AUDIT_SHA = "d2e4616871961f3a0bd4212889d076ccb52368141e07b45db059bec84917bc0a"
LEVEL_SHA = [
    "843e180ccf823a1aab4dbf7ece2e8469b1f74d8d14547bdc741796e23e63f868",
    "ece7861723f15a28cfa38ad8442aeed7dfccdfcf98a4f069ab3416d2d9a95847",
    "0090663f43596a32ae890a98dc8833c2d7f6b02218d2ca7b3d6991f68133c45d",
    "fce8c15b5a3c836901de2486649cffbcc4d9bf3907ef89977ed55dab8bb19e06",
]
LEVEL_BYTES = [86400, 21600, 5520, 1440]
MASK_SHA = {
    "dense": "b03c161546b8ac65df8c904de01a8e4a85ce5fa4f8b1738a5e49fc99e8788794",
    "sparse": "2a1abaf26369c2a50e373894418ff59dcda461fcb1e5ebc8f200f5c16f76f5de",
}
CANONICAL_SHA = {
    "dense": "58f62230e75e2df2719c3e0c7ddd3b8397e259aaaf403c48715e141a4d45cc79",
    "sparse": "32b5fcd51cc9533c6626c958e1b20dae91df456400307d0bad3524aac4073065",
}
RAW_SHA = {
    ("dense", 165): CANONICAL_SHA["dense"],
    ("dense", 90): CANONICAL_SHA["dense"],
    ("sparse", 165): "99fb611421071dd99796b531352c70d3ddcfdeeb601ea94088a4686cdc98ef9a",
    ("sparse", 90): "da5111508e7e6a4851221e1906bef2187d3925f21edb914a0e2dbafd6dc35a7e",
}
WRITTEN = {"dense": 57600, "sparse": 35974}
UNTOUCHED = {"dense": 0, "sparse": 21626}
OUTPUT_BYTES = 230400
MASK_BYTES = 57600
ROOT_KEY = "receipt_root_sha256"
RUNS = {
    f"l{launch}-{phase}-{fixture}-poison-{poison_name}-process-{ordinal}":
        (phase, launch, fixture, poison)
    for phase, launch in (("reset", 7), ("generated", 17))
    for fixture in ("dense", "sparse")
    for poison_name, poison in (("a5", 165), ("5a", 90))
    for ordinal in ("a", "b")
}


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def load(path: pathlib.Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except Exception as error:
        raise VerificationError(f"unable to load JSON {path}: {error}") from error
    require(isinstance(value, dict), f"JSON root is not object: {path}")
    return value


def canonical(value: Any, newline: bool = True) -> bytes:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return (text + ("\n" if newline else "")).encode()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rooted(value: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(value)
    result.pop(ROOT_KEY, None)
    result[ROOT_KEY] = sha_bytes(canonical(result))
    return result


def verify_root(value: dict[str, Any], label: str,
                newline: bool = True) -> None:
    expected = value.get(ROOT_KEY)
    require(isinstance(expected, str) and len(expected) == 64,
            f"{label}: root missing")
    body = copy.deepcopy(value)
    body.pop(ROOT_KEY, None)
    require(sha_bytes(canonical(body, newline)) == expected,
            f"{label}: root mismatch")


def write_json(path: pathlib.Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")


def verify_file(path: pathlib.Path, digest: str, size: int) -> bytes:
    require(path.is_file(), f"artifact missing: {path}")
    require(path.stat().st_size == size, f"artifact size drift: {path}")
    data = path.read_bytes()
    require(sha_bytes(data) == digest, f"artifact hash drift: {path}")
    return data


def verify_masked_output(raw: bytes, canonical_output: bytes, mask: bytes,
                         poison: int, fixture: str, run_id: str) -> None:
    written = untouched = 0
    for pixel, mask_value in enumerate(mask):
        begin = pixel * 4
        raw_pixel = raw[begin:begin + 4]
        canonical_pixel = canonical_output[begin:begin + 4]
        if mask_value & 1:
            written += 1
            require(raw_pixel == canonical_pixel,
                    f"{run_id}: written pixel canonicalization drift at {pixel}")
        else:
            untouched += 1
            require(raw_pixel == bytes([poison]) * 4,
                    f"{run_id}: untouched pixel write at {pixel}")
            require(canonical_pixel == b"\0\0\0\0",
                    f"{run_id}: untouched canonical pixel not zero at {pixel}")
    require(written == WRITTEN[fixture] and untouched == UNTOUCHED[fixture],
            f"{run_id}: mask coverage count drift")


def verify_backend(backend: str, run_root: pathlib.Path,
                   repository: pathlib.Path,
                   source_audit_path: pathlib.Path | None,
                   lowering_path: pathlib.Path) -> dict[str, Any]:
    require(backend in ("rtx", "mac"), "unknown backend")
    expected_backend = "rtx-cuda-driver" if backend == "rtx" else "mac-metal"
    processes: list[int] = []
    run_receipts: list[dict[str, Any]] = []
    for run_id, (phase, launch, fixture, poison) in RUNS.items():
        directory = run_root / run_id
        receipt_path = directory / "run-receipt.json"
        receipt = load(receipt_path)
        run_receipts.append({"run_id": run_id,
                             "bytes": receipt_path.stat().st_size,
                             "sha256": sha_file(receipt_path)})
        require(receipt.get("schema") == "neuralstudio.fg.early-l7-l17-run.v1" and
                receipt.get("backend") == expected_backend,
                f"{run_id}: schema/backend drift")
        require(receipt.get("run_id") == run_id and
                receipt.get("phase") == phase and
                receipt.get("selected_launch") == launch and
                receipt.get("fixture") == fixture and
                receipt.get("poison_byte") == poison,
                f"{run_id}: identity drift")
        require(receipt.get("module_token") == 23 and
                receipt.get("module_sha256") == MODULE_SHA and
                receipt.get("packed_parameter_bytes") == 72 and
                receipt.get("selected_mode") == 1 and
                receipt.get("dead_offset0_value") == "null",
                f"{run_id}: selected ABI drift")
        block_key = "block" if backend == "rtx" else "selected_block"
        grid_key = "grid" if backend == "rtx" else "selected_grid"
        require(receipt.get(block_key) == [256, 1, 1] and
                receipt.get(grid_key) == [20, 12, 1],
                f"{run_id}: selected geometry drift")
        require(receipt.get("level_pre_sha256") == LEVEL_SHA and
                receipt.get("level_post_sha256") == LEVEL_SHA and
                receipt.get("mask_pre_sha256") == MASK_SHA[fixture] and
                receipt.get("mask_post_sha256") == MASK_SHA[fixture],
                f"{run_id}: input provenance/mutation")
        require(receipt.get("output_bytes") == OUTPUT_BYTES and
                receipt.get("raw_output_sha256") == RAW_SHA[(fixture, poison)] and
                receipt.get("canonical_output_sha256") == CANONICAL_SHA[fixture],
                f"{run_id}: output drift")
        require(receipt.get("written_pixels") == WRITTEN[fixture] and
                receipt.get("untouched_pixels") == UNTOUCHED[fixture],
                f"{run_id}: coverage count drift")
        require(receipt.get("input_mismatched_bytes") == 0 and
                receipt.get("canary_mismatched_bytes") == 0,
                f"{run_id}: input/canary gate")
        require(receipt.get("expected_output_opened") is False and
                receipt.get("archived_intermediate_opened") is False and
                receipt.get("address_fields_recorded") is False and
                receipt.get("result") == "pass",
                f"{run_id}: provenance/result gate")
        if backend == "mac":
            require(receipt.get("architecture_rom_sha256") == ROM_SHA and
                    receipt.get("surface_half_rounding") == "toward-zero",
                    f"{run_id}: Mac architecture semantics drift")
        process = receipt.get("process_id")
        require(isinstance(process, int) and process > 0,
                f"{run_id}: process id missing")
        processes.append(process)
        for index, (digest, size) in enumerate(zip(LEVEL_SHA, LEVEL_BYTES)):
            verify_file(directory / f"level-{index}-after.bin", digest, size)
        mask = verify_file(directory / "mask-after.bin", MASK_SHA[fixture], MASK_BYTES)
        raw = verify_file(directory / "output.bin", RAW_SHA[(fixture, poison)],
                          OUTPUT_BYTES)
        canonical_output = verify_file(
            directory / "canonical-output.bin", CANONICAL_SHA[fixture],
            OUTPUT_BYTES)
        verify_masked_output(raw, canonical_output, mask, poison, fixture, run_id)
    require(len(set(processes)) == 16, "fresh-process gate failed")

    lowering = load(lowering_path)
    verify_root(lowering, "lowering", newline=False)
    require(lowering.get(ROOT_KEY) == LOWERING_ROOT and
            lowering.get("division_sites_rewritten") == 54 and
            lowering.get("folded_bfind_output") == 2 and
            lowering.get("restored_surface_store") ==
            "RG16F float2 to half2 round-toward-zero" and
            lowering.get("expected_output_bytes_embedded") is False,
            "lowering receipt drift")

    source_files: list[dict[str, str]] = []
    build_artifacts: list[dict[str, Any]] = []
    rig_summary = None
    source_audit_summary = None
    if backend == "rtx":
        rig_path = run_root / "rig-receipt.json"
        rig = load(rig_path)
        require(rig.get("schema") == "neuralstudio.fg.early-l7-l17-rig.v1" and
                rig.get("attempt_id") == "oracle-v2" and
                rig.get("fresh_process_count") == 16 and
                rig.get("canonical_outputs") == CANONICAL_SHA and
                rig.get("result") == "pass", "rig receipt drift")
        require(source_audit_path is not None and
                sha_file(source_audit_path) == SOURCE_AUDIT_SHA,
                "source audit missing/hash drift")
        audit = load(source_audit_path)
        require(audit.get("schema") ==
                "neuralstudio.fg.early-l7-l17-source-audit.v1" and
                audit.get("shared_cuda_helper_live_hash_recovered") is True and
                audit.get("result") == "pass", "source audit rejected")
        local = {
            "unit_source": repository / "src/fg_early_l7_l17/rtx_oracle.cpp",
            "contract": repository / "src/fg_early_l7_l17/contract.h",
            "shared_cuda_helper": repository / "src/fg_conv/rtx_oracle.cpp",
            "build_harness": repository / "scripts/build-fg-early-l7-l17-rtx.ps1",
        }
        for name, path in local.items():
            digest = sha_file(path)
            require(audit["files"][name]["sha256"] == digest,
                    f"RTX source audit mismatch: {name}")
            source_files.append({"file": str(path.relative_to(repository)),
                                 "sha256": digest})
        binary = run_root / "build/fg-early-l7-l17-rtx.exe"
        require(audit["files"]["binary"]["sha256"] == sha_file(binary),
                "RTX binary audit mismatch")
        build_artifacts.append({"file": binary.name, "bytes": binary.stat().st_size,
                                "sha256": sha_file(binary)})
        rig_summary = {"job_id": rig["job_id"], "attempt_id": rig["attempt_id"],
                       "gpu": rig["gpu"], "cuda_toolkit": rig["cuda_toolkit"],
                       "file_sha256": sha_file(rig_path)}
        source_audit_summary = {"file_sha256": SOURCE_AUDIT_SHA,
                                "shared_helper_provenance_gap_closed": True}
    else:
        paths = [
            repository / "src/fg_early_l7_l17/macos_probe.mm",
            repository / "src/fg_early_l7_l17/early_l7_l17.metal",
            repository / "src/fg_early_l7_l17/contract.h",
            repository / "src/fg_conv/macos_probe.mm",
            repository / "scripts/build-macos-fg-early-l7-l17.sh",
            repository / "scripts/lower-fg-early-module23-cumetal.py",
        ]
        for path in paths:
            source_files.append({"file": str(path.relative_to(repository)),
                                 "sha256": sha_file(path)})
        for path, expected in (
            (run_root / "build/kernel.metallib", None),
            (run_root / "build/probe", None),
            (run_root / "build/rcp.bin", ROM_SHA),
        ):
            require(path.is_file(), f"Mac artifact missing: {path}")
            digest = sha_file(path)
            if expected is not None:
                require(digest == expected, "RCP ROM drift")
            build_artifacts.append({"file": path.name, "bytes": path.stat().st_size,
                                    "sha256": digest})
        require(sha_file(repository / "src/fg_early_l7_l17/early_l7_l17.metal") ==
                lowering["output_msl_sha256"], "lowered Metal source drift")

    result = {
        "schema": "neuralstudio.fg.early-l7-l17-backend.v1",
        "backend": expected_backend,
        "selected_launches": [7, 17],
        "module_token": 23,
        "module_sha256": MODULE_SHA,
        "packed_parameter_bytes": 72,
        "raw_parameter_sha256": PARAM_SHA,
        "fixture_address_free_parameter_sha256": FIXTURE_PARAM_SHA,
        "selected_grid": [20, 12, 1],
        "selected_block": [256, 1, 1],
        "rejected_block": [16, 16, 1],
        "dead_offset0_proof_root_sha256": DEAD_PROOF_ROOT,
        "lowering_receipt_root_sha256": LOWERING_ROOT,
        "run_count": 16,
        "fresh_process_count": 16,
        "poison_bytes": [165, 90],
        "canonical_output_sha256": CANONICAL_SHA,
        "raw_sparse_output_sha256_by_poison": {
            "165": RAW_SHA[("sparse", 165)],
            "90": RAW_SHA[("sparse", 90)],
        },
        "dense_written_pixels": WRITTEN["dense"],
        "sparse_written_pixels": WRITTEN["sparse"],
        "sparse_untouched_pixels": UNTOUCHED["sparse"],
        "exact_mask_positions": True,
        "dense_full_write_coverage": True,
        "sparse_untouched_regions_preserve_poison": True,
        "input_mismatched_bytes": 0,
        "canary_mismatched_bytes": 0,
        "input_immutable": True,
        "rom_sha256": ROM_SHA if backend == "mac" else None,
        "rom_immutable": True if backend == "mac" else None,
        "surface_half_rounding": "toward-zero",
        "expected_or_archived_ingress": False,
        "address_fields_recorded": False,
        "run_receipts": run_receipts,
        "source_files": source_files,
        "build_artifacts": build_artifacts,
        "rig": rig_summary,
        "source_audit": source_audit_summary,
        "result": "pass",
    }
    return rooted(result)


def validate_backend(value: dict[str, Any], backend: str) -> None:
    verify_root(value, backend)
    expected = "rtx-cuda-driver" if backend == "rtx" else "mac-metal"
    require(value.get("schema") == "neuralstudio.fg.early-l7-l17-backend.v1" and
            value.get("backend") == expected, f"{backend}: identity drift")
    require(value.get("selected_launches") == [7, 17] and
            value.get("module_sha256") == MODULE_SHA and
            value.get("selected_block") == [256, 1, 1] and
            value.get("rejected_block") == [16, 16, 1],
            f"{backend}: selected geometry drift")
    require(value.get("dead_offset0_proof_root_sha256") == DEAD_PROOF_ROOT and
            value.get("lowering_receipt_root_sha256") == LOWERING_ROOT,
            f"{backend}: specialization provenance drift")
    require(value.get("fresh_process_count") == 16 and
            value.get("canonical_output_sha256") == CANONICAL_SHA,
            f"{backend}: process/output drift")
    for field in ("exact_mask_positions", "dense_full_write_coverage",
                  "sparse_untouched_regions_preserve_poison", "input_immutable"):
        require(value.get(field) is True, f"{backend}: {field} false")
    require(value.get("canary_mismatched_bytes") == 0 and
            value.get("input_mismatched_bytes") == 0 and
            value.get("surface_half_rounding") == "toward-zero",
            f"{backend}: canary/input/rounding drift")
    if backend == "mac":
        require(value.get("rom_sha256") == ROM_SHA and
                value.get("rom_immutable") is True, "mac: ROM gate")
    else:
        require(value.get("source_audit", {}).get(
            "shared_helper_provenance_gap_closed") is True,
            "rtx: source audit gap")
    require(value.get("expected_or_archived_ingress") is False and
            value.get("address_fields_recorded") is False and
            value.get("result") == "pass",
            f"{backend}: provenance/result gate")


def make_final(rtx: dict[str, Any], mac: dict[str, Any],
               abi: dict[str, Any], counterexamples: dict[str, Any]) -> dict[str, Any]:
    validate_backend(rtx, "rtx")
    validate_backend(mac, "mac")
    verify_root(abi, "ABI")
    verify_root(counterexamples, "counterexamples")
    require(abi.get("schema") == "neuralstudio.fg.early-l7-l17-abi.v1" and
            abi.get("selected_dispatch", {}).get("block") == [256, 1, 1] and
            abi.get("dead_offset0_proof_root_sha256") == DEAD_PROOF_ROOT,
            "ABI proof drift")
    require(counterexamples.get("all_failures_preserved") is True,
            "failure preservation drift")
    require(rtx["canonical_output_sha256"] == mac["canonical_output_sha256"],
            "RTX/Metal canonical output drift")
    require(rtx["raw_sparse_output_sha256_by_poison"] ==
            mac["raw_sparse_output_sha256_by_poison"],
            "RTX/Metal sparse raw output drift")
    return rooted({
        "schema": "neuralstudio.fg.early-l7-l17-accepted.v1",
        "selected_units": [
            "canonical L7/module-000023 reset multiscale repair",
            "canonical L17/module-000023 generated multiscale repair",
        ],
        "abi_receipt_root_sha256": abi[ROOT_KEY],
        "counterexamples_root_sha256": counterexamples[ROOT_KEY],
        "rtx_backend_root_sha256": rtx[ROOT_KEY],
        "mac_backend_root_sha256": mac[ROOT_KEY],
        "module_sha256": MODULE_SHA,
        "selected_block": [256, 1, 1],
        "dead_offset0_proof_root_sha256": DEAD_PROOF_ROOT,
        "lowering_receipt_root_sha256": LOWERING_ROOT,
        "canonical_output_sha256": CANONICAL_SHA,
        "rtx_mac_dense_mismatched_bytes": 0,
        "rtx_mac_sparse_written_mismatched_bytes": 0,
        "rtx_mac_sparse_untouched_mismatched_pixels": 0,
        "processes_per_backend": 16,
        "dense_full_write_coverage": True,
        "sparse_written_pixels": 35974,
        "sparse_untouched_pixels": 21626,
        "canary_mismatched_bytes": 0,
        "input_and_rom_immutable": True,
        "expected_or_archived_ingress": False,
        "address_fields_recorded": False,
        "claims": {
            "l7_standalone_exact_on_metal": True,
            "l17_standalone_exact_on_metal": True,
            "selected_mode1_exact": True,
            "mode0_supported": False,
            "connected_early_chain": False,
            "generated_frame": False,
            "netal_integration": False,
            "performance": False,
        },
        "result": "pass",
    })


def validate_final(value: dict[str, Any], rtx: dict[str, Any],
                   mac: dict[str, Any], abi: dict[str, Any],
                   counterexamples: dict[str, Any]) -> None:
    verify_root(value, "accepted")
    require(value == make_final(rtx, mac, abi, counterexamples),
            "accepted receipt drift")


def make_negative(accepted: dict[str, Any], rtx: dict[str, Any],
                  mac: dict[str, Any], abi: dict[str, Any],
                  counterexamples: dict[str, Any]) -> dict[str, Any]:
    cases = []
    def add(name: str):
        values = tuple(copy.deepcopy(x) for x in
                       (accepted, rtx, mac, abi, counterexamples))
        cases.append((name, *values)); return values
    a, _, _, _, _ = add("accepted-block-drift"); a["selected_block"] = [16,16,1]
    a, _, _, _, _ = add("accepted-dense-drift"); a["canonical_output_sha256"]["dense"] = "0"*64
    a, _, _, _, _ = add("accepted-overclaim"); a["claims"]["generated_frame"] = True
    _, r, _, _, _ = add("rtx-process-drift"); r["fresh_process_count"] = 15
    _, r, _, _, _ = add("rtx-mask-drift"); r["exact_mask_positions"] = False
    _, r, _, _, _ = add("rtx-source-audit-drift"); r["source_audit"]["shared_helper_provenance_gap_closed"] = False
    _, _, m, _, _ = add("mac-rom-drift"); m["rom_immutable"] = False
    _, _, m, _, _ = add("mac-rounding-drift"); m["surface_half_rounding"] = "nearest-even"
    _, _, m, _, _ = add("mac-canary-drift"); m["canary_mismatched_bytes"] = 1
    _, _, _, b, _ = add("abi-dead-proof-drift"); b["dead_offset0_proof_root_sha256"] = "1"*64
    _, _, _, b, _ = add("abi-geometry-drift"); b["selected_dispatch"]["block"] = [16,16,1]
    _, _, _, _, c = add("counterexample-loss"); c["all_failures_preserved"] = False
    _, r, _, _, _ = add("rtx-ingress-drift"); r["expected_or_archived_ingress"] = True
    a, _, _, _, _ = add("accepted-root-corruption"); a[ROOT_KEY] = "f"*64
    results = []
    for name, ma, mr, mm, mb, mc in cases:
        rejected = False; reason = ""
        try: validate_final(ma, mr, mm, mb, mc)
        except VerificationError as error: rejected = True; reason = str(error)
        require(rejected, f"negative mutation accepted: {name}")
        results.append({"mutation": name, "rejected": True, "reason": reason})
    return rooted({"schema": "neuralstudio.fg.early-l7-l17-negative-audit.v1",
                   "accepted_receipt_root_sha256": accepted[ROOT_KEY],
                   "mutation_count": len(results), "mutations": results,
                   "all_mutations_rejected": True, "result": "pass"})


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    backend = commands.add_parser("backend")
    backend.add_argument("--backend", choices=("rtx", "mac"), required=True)
    backend.add_argument("--run-root", type=pathlib.Path, required=True)
    backend.add_argument("--repository", type=pathlib.Path, required=True)
    backend.add_argument("--source-audit", type=pathlib.Path)
    backend.add_argument("--lowering", type=pathlib.Path, required=True)
    backend.add_argument("--output", type=pathlib.Path, required=True)
    for name in ("final", "receipt", "negative"):
        sub = commands.add_parser(name)
        if name in ("receipt", "negative"):
            sub.add_argument("--accepted", type=pathlib.Path, required=True)
        sub.add_argument("--rtx", type=pathlib.Path, required=True)
        sub.add_argument("--mac", type=pathlib.Path, required=True)
        sub.add_argument("--abi", type=pathlib.Path, required=True)
        sub.add_argument("--counterexamples", type=pathlib.Path, required=True)
        if name in ("final", "negative"): sub.add_argument("--output", type=pathlib.Path, required=True)
    arguments = parser.parse_args()
    try:
        if arguments.command == "backend":
            value = verify_backend(arguments.backend, arguments.run_root,
                                   arguments.repository, arguments.source_audit,
                                   arguments.lowering)
            write_json(arguments.output, value)
        else:
            rtx, mac = load(arguments.rtx), load(arguments.mac)
            abi, counters = load(arguments.abi), load(arguments.counterexamples)
            if arguments.command == "final":
                value = make_final(rtx, mac, abi, counters)
                write_json(arguments.output, value)
            elif arguments.command == "receipt":
                value = load(arguments.accepted)
                validate_final(value, rtx, mac, abi, counters)
            else:
                accepted = load(arguments.accepted)
                validate_final(accepted, rtx, mac, abi, counters)
                value = make_negative(accepted, rtx, mac, abi, counters)
                write_json(arguments.output, value)
        print(f"receipt_root_sha256={value[ROOT_KEY]}")
        print("verification_result=pass")
        return 0
    except VerificationError as error:
        print(f"verification_result=fail\nerror={error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
