#!/usr/bin/env python3
"""Harden and verify standalone parity for canonical FG launches L8/L18."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pathlib
import sys
from typing import Any


MODULE_SHA = "28fe0e153d1ffc80b54995c022f18395c5556f3c041cc9d724bfe7e70dc4e824"
INPUT_SHA = "1487432aa641843223558670fa376563a8625bc786d6fa48769fb58644e88eb4"
COVERAGE_SHA = "dc60073283cd80698c9098debe4c45c64dd9d7ae3c9898d46985abfe075c59e5"
OUTPUT_SHA = {
    165: "ff6464fc227c0482f8fef683c592ee9d68ac763da67cb73c1610e479614867ee",
    90: "ad7cf8bc87c3a566fd5f3bd47422ce335dc6e6b9732154d7474028a9f677d5f5",
}
ROM_SHA = "7cf4f8cece6b18b07605edc9cb601a153346a79b10bf3b93c7ec058d57d29750"
SOURCE_AUDIT_SHA = "5f80a4cdfca5450094829e8f0f815a4af96a1898efe97b593225617b18d0b7be"
FIXTURE_PARAMETER_SHA = "964414862828b4961809f157ff16ea523a36fe1f2b68dfe1a08c13cbf49403ff"
WIDTH = 320
HEIGHT = 180
INPUT_BYTES = 230400
OUTPUT_BYTES = 57600
WRITTEN_PIXELS = 56600
UNTOUCHED_PIXELS = 1000
ROOT_KEY = "receipt_root_sha256"
RUNS = {
    f"l{launch}-{phase}-poison-{poison_name}-process-{ordinal}":
        (phase, launch, poison)
    for phase, launch in (("reset", 8), ("generated", 18))
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
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


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


def verify_root(value: dict[str, Any], label: str) -> None:
    expected = value.get(ROOT_KEY)
    require(isinstance(expected, str) and len(expected) == 64,
            f"{label}: root missing")
    body = copy.deepcopy(value)
    body.pop(ROOT_KEY, None)
    require(sha_bytes(canonical(body)) == expected, f"{label}: root mismatch")


def write_json(path: pathlib.Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")


def verify_file(path: pathlib.Path, expected_sha: str,
                expected_bytes: int) -> dict[str, Any]:
    require(path.is_file(), f"artifact missing: {path}")
    require(path.stat().st_size == expected_bytes, f"artifact size drift: {path}")
    digest = sha_file(path)
    require(digest == expected_sha, f"artifact hash drift: {path}")
    return {"file": path.name, "bytes": expected_bytes, "sha256": digest}


def derive_coverage(output: bytes, poison: int, run_id: str) -> bytes:
    coverage = bytearray(len(output))
    for index, value in enumerate(output):
        if value == 0:
            coverage[index] = 1
        elif value == poison:
            coverage[index] = 0
        else:
            raise VerificationError(
                f"{run_id}: output byte {index} escaped zero/poison domain: {value}")
    return bytes(coverage)


def verify_backend(backend: str, run_root: pathlib.Path,
                   repository: pathlib.Path,
                   source_audit_path: pathlib.Path | None) -> dict[str, Any]:
    require(backend in ("rtx", "mac"), "unknown backend")
    expected_backend = "rtx-cuda-driver" if backend == "rtx" else "mac-metal"
    receipts: dict[str, dict[str, Any]] = {}
    outputs: dict[str, bytes] = {}
    coverages: dict[str, bytes] = {}
    process_ids: list[int] = []
    run_receipts: list[dict[str, Any]] = []
    for run_id, (phase, launch, poison) in RUNS.items():
        directory = run_root / run_id
        receipt_path = directory / "run-receipt.json"
        receipt = load(receipt_path)
        receipts[run_id] = receipt
        run_receipts.append({
            "run_id": run_id,
            "bytes": receipt_path.stat().st_size,
            "sha256": sha_file(receipt_path),
        })
        require(receipt.get("schema") == "neuralstudio.fg.early-l8-l18-run.v1",
                f"{run_id}: schema drift")
        require(receipt.get("backend") == expected_backend,
                f"{run_id}: backend drift")
        require(receipt.get("run_id") == run_id, f"{run_id}: identity drift")
        require(receipt.get("phase") == phase, f"{run_id}: phase drift")
        require(receipt.get("selected_launch") == launch,
                f"{run_id}: launch drift")
        require(receipt.get("poison_byte") == poison, f"{run_id}: poison drift")
        require(receipt.get("module_token") == 25 and
                receipt.get("module_sha256") == MODULE_SHA,
                f"{run_id}: module drift")
        require(receipt.get("packed_parameter_bytes") == 48 and
                receipt.get("selected_mode") == 0,
                f"{run_id}: ABI drift")
        require(receipt.get("input_pre_sha256") == INPUT_SHA and
                receipt.get("input_post_sha256") == INPUT_SHA,
                f"{run_id}: input provenance/mutation")
        require(receipt.get("output_bytes") == OUTPUT_BYTES,
                f"{run_id}: output size drift")
        require(receipt.get("output_sha256") == OUTPUT_SHA[poison],
                f"{run_id}: output hash drift")
        require(receipt.get("coverage_sha256") == COVERAGE_SHA,
                f"{run_id}: coverage hash drift")
        require(receipt.get("written_pixels") == WRITTEN_PIXELS and
                receipt.get("untouched_pixels") == UNTOUCHED_PIXELS,
                f"{run_id}: coverage count drift")
        require(receipt.get("input_mismatched_bytes") == 0,
                f"{run_id}: input/ROM mutation")
        require(receipt.get("canary_mismatched_bytes") == 0,
                f"{run_id}: guard row or ROM guard mutation")
        require(receipt.get("expected_output_opened") is False and
                receipt.get("archived_intermediate_opened") is False,
                f"{run_id}: forbidden ingress")
        require(receipt.get("address_fields_recorded") is False,
                f"{run_id}: address field recorded")
        require(receipt.get("result") == "pass", f"{run_id}: did not pass")
        process_id = receipt.get("process_id")
        require(isinstance(process_id, int) and process_id > 0,
                f"{run_id}: PID missing")
        process_ids.append(process_id)

        input_path = directory / "input-after.bin"
        output_path = directory / "output.bin"
        coverage_path = directory / "coverage.bin"
        verify_file(input_path, INPUT_SHA, INPUT_BYTES)
        verify_file(output_path, OUTPUT_SHA[poison], OUTPUT_BYTES)
        verify_file(coverage_path, COVERAGE_SHA, OUTPUT_BYTES)
        output = output_path.read_bytes()
        coverage = coverage_path.read_bytes()
        derived = derive_coverage(output, poison, run_id)
        require(derived == coverage, f"{run_id}: coverage position drift")
        require(set(output).issubset({0, poison}),
                f"{run_id}: output domain drift")
        require(coverage.count(1) == WRITTEN_PIXELS and
                coverage.count(0) == UNTOUCHED_PIXELS,
                f"{run_id}: coverage bytes are not exact 0/1 positions")
        outputs[run_id] = output
        coverages[run_id] = coverage

    require(len(set(process_ids)) == 8, "fresh-process gate failed")
    require(len(set(coverages.values())) == 1, "coverage positions differ")
    require(len({sha_bytes(value) for value in coverages.values()}) == 1,
            "coverage hash set drift")
    for phase, launch in (("reset", 8), ("generated", 18)):
        phase_runs = [name for name, value in RUNS.items() if value[0] == phase]
        require(len({sha_bytes(coverages[name]) for name in phase_runs}) == 1,
                f"L{launch}: phase coverage drift")
    for poison in (165, 90):
        poison_runs = [name for name, value in RUNS.items() if value[2] == poison]
        require(len({sha_bytes(outputs[name]) for name in poison_runs}) == 1,
                f"poison {poison}: output drift across phase/process")
    a5 = outputs["l8-reset-poison-a5-process-a"]
    five_a = outputs["l8-reset-poison-5a-process-a"]
    differing_positions = sum(left != right for left, right in zip(a5, five_a))
    require(differing_positions == UNTOUCHED_PIXELS,
            "dual-poison differing positions do not equal untouched coverage")
    coverage = coverages["l8-reset-poison-a5-process-a"]
    for index, marker in enumerate(coverage):
        if marker:
            require(a5[index] == 0 and five_a[index] == 0,
                    f"written coverage changed under poison at {index}")
        else:
            require(a5[index] == 165 and five_a[index] == 90,
                    f"untouched coverage did not preserve poison at {index}")

    source_files: list[dict[str, Any]] = []
    build_artifacts: list[dict[str, Any]] = []
    rig_summary = None
    source_audit_summary = None
    if backend == "rtx":
        rig_path = run_root / "rig-receipt.json"
        rig = load(rig_path)
        require(rig.get("schema") == "neuralstudio.fg.early-l8-l18-rig.v1" and
                rig.get("result") == "pass", "rig receipt rejected")
        require(rig.get("fresh_process_count") == 8 and
                rig.get("coverage_sha256") == COVERAGE_SHA and
                rig.get("written_pixels") == WRITTEN_PIXELS,
                "rig summary gate drift")
        require(source_audit_path is not None, "RTX source audit required")
        require(sha_file(source_audit_path) == SOURCE_AUDIT_SHA,
                "source-audit file hash drift")
        audit = load(source_audit_path)
        require(audit.get("schema") ==
                "neuralstudio.fg.early-l8-l18-source-audit.v1" and
                audit.get("result") == "pass", "source audit rejected")
        require(audit.get("shared_cuda_helper_omitted_from_original_rig_receipt")
                is True and audit.get("shared_cuda_helper_live_hash_recovered")
                is True, "source audit did not close helper gap")
        remote = audit["files"]
        local_paths = {
            "unit_source": repository / "src/fg_early_l8_l18/rtx_oracle.cpp",
            "contract": repository / "src/fg_early_l8_l18/contract.h",
            "shared_cuda_helper": repository / "src/fg_conv/rtx_oracle.cpp",
            "build_harness": repository / "scripts/build-fg-early-l8-l18-rtx.ps1",
        }
        for name, path in local_paths.items():
            digest = sha_file(path)
            require(remote[name]["sha256"] == digest,
                    f"remote/local source mismatch: {name}")
            source_files.append({"file": str(path.relative_to(repository)),
                                 "sha256": digest})
        binary = run_root / "build/fg-early-l8-l18-rtx.exe"
        require(remote["binary"]["sha256"] == sha_file(binary),
                "audited binary hash drift")
        build_artifacts.append({
            "file": binary.name,
            "bytes": binary.stat().st_size,
            "sha256": sha_file(binary),
        })
        rig_summary = {
            "job_id": rig["job_id"],
            "attempt_id": rig["attempt_id"],
            "gpu": rig["gpu"],
            "cuda_toolkit": rig["cuda_toolkit"],
            "file_sha256": sha_file(rig_path),
        }
        source_audit_summary = {
            "file_sha256": SOURCE_AUDIT_SHA,
            "captured_utc": audit["captured_utc"],
            "shared_helper_provenance_gap_closed": True,
        }
    else:
        local_paths = [
            repository / "src/fg_early_l8_l18/macos_probe.mm",
            repository / "src/fg_early_l8_l18/early_l8_l18.metal",
            repository / "src/fg_early_l8_l18/contract.h",
            repository / "src/fg_conv/macos_probe.mm",
            repository / "scripts/build-macos-fg-early-l8-l18.sh",
        ]
        for path in local_paths:
            source_files.append({"file": str(path.relative_to(repository)),
                                 "sha256": sha_file(path)})
        artifacts = [
            (run_root / "build/early_l8_l18.metallib", None),
            (run_root / "build/fg-early-l8-l18-mac", None),
            (run_root / "build/sm120-rcp-correction-rom.bin", ROM_SHA),
        ]
        for path, expected in artifacts:
            require(path.is_file(), f"Mac build artifact missing: {path}")
            digest = sha_file(path)
            if expected is not None:
                require(digest == expected, "RCP ROM identity drift")
            build_artifacts.append({"file": path.name,
                                    "bytes": path.stat().st_size,
                                    "sha256": digest})
        source = (repository / "src/fg_early_l8_l18/macos_probe.mm").read_text()
        require("Mismatches(rom, rom_after)" in source,
                "Mac source no longer gates ROM immutability")
        require("CanaryMismatches(rom_buffer, kRomBytes)" in source,
                "Mac source no longer gates ROM canaries")

    result = {
        "schema": "neuralstudio.fg.early-l8-l18-backend.v1",
        "backend": expected_backend,
        "selected_launches": [8, 18],
        "module_token": 25,
        "module_sha256": MODULE_SHA,
        "packed_parameter_bytes": 48,
        "fixture_address_free_parameter_sha256": FIXTURE_PARAMETER_SHA,
        "input_sha256": INPUT_SHA,
        "coverage_sha256": COVERAGE_SHA,
        "output_sha256_by_poison": {"165": OUTPUT_SHA[165], "90": OUTPUT_SHA[90]},
        "run_count": 8,
        "fresh_process_count": 8,
        "poison_bytes": [165, 90],
        "written_zero_pixels": WRITTEN_PIXELS,
        "untouched_poison_pixels": UNTOUCHED_PIXELS,
        "output_domain": [0, "poison_byte"],
        "exact_coverage_positions_across_all_runs": True,
        "l8_l18_standalone_outputs_equal": True,
        "dual_poison_differing_positions": differing_positions,
        "guard_row_bytes": WIDTH,
        "guard_row_mismatched_bytes": 0,
        "input_mismatched_bytes": 0,
        "input_immutable": True,
        "rom_sha256": ROM_SHA if backend == "mac" else None,
        "rom_immutable": True if backend == "mac" else None,
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
    require(value.get("schema") ==
            "neuralstudio.fg.early-l8-l18-backend.v1",
            f"{backend}: schema drift")
    require(value.get("backend") == expected, f"{backend}: identity drift")
    require(value.get("selected_launches") == [8, 18],
            f"{backend}: launch drift")
    require(value.get("module_sha256") == MODULE_SHA,
            f"{backend}: module drift")
    require(value.get("fixture_address_free_parameter_sha256") ==
            FIXTURE_PARAMETER_SHA, f"{backend}: fixture ABI drift")
    require(value.get("fresh_process_count") == 8 and
            value.get("run_count") == 8, f"{backend}: process gate")
    require(value.get("poison_bytes") == [165, 90],
            f"{backend}: poison drift")
    require(value.get("coverage_sha256") == COVERAGE_SHA and
            value.get("exact_coverage_positions_across_all_runs") is True,
            f"{backend}: coverage drift")
    require(value.get("written_zero_pixels") == WRITTEN_PIXELS and
            value.get("untouched_poison_pixels") == UNTOUCHED_PIXELS,
            f"{backend}: coverage count drift")
    require(value.get("output_domain") == [0, "poison_byte"],
            f"{backend}: output domain drift")
    require(value.get("dual_poison_differing_positions") == UNTOUCHED_PIXELS,
            f"{backend}: poison-position drift")
    require(value.get("guard_row_bytes") == WIDTH and
            value.get("guard_row_mismatched_bytes") == 0,
            f"{backend}: guard row drift")
    require(value.get("input_immutable") is True and
            value.get("input_mismatched_bytes") == 0,
            f"{backend}: input mutation")
    if backend == "mac":
        require(value.get("rom_sha256") == ROM_SHA and
                value.get("rom_immutable") is True,
                "mac: ROM provenance/mutation")
    else:
        require(value.get("source_audit", {}).get(
            "shared_helper_provenance_gap_closed") is True,
            "rtx: shared-helper provenance gap")
    require(value.get("expected_or_archived_ingress") is False,
            f"{backend}: forbidden ingress")
    require(value.get("address_fields_recorded") is False,
            f"{backend}: address leak")
    require(value.get("result") == "pass", f"{backend}: did not pass")


def make_final(rtx: dict[str, Any], mac: dict[str, Any],
               abi: dict[str, Any]) -> dict[str, Any]:
    validate_backend(rtx, "rtx")
    validate_backend(mac, "mac")
    verify_root(abi, "ABI")
    require(abi.get("schema") == "neuralstudio.fg.early-l8-l18-abi.v1",
            "ABI schema drift")
    require(abi.get("canonical_launches") == [8, 18] and
            abi.get("module_sha256") == MODULE_SHA and
            abi.get("packed_parameter_bytes") == 48,
            "ABI identity drift")
    require(rtx["input_sha256"] == mac["input_sha256"] == INPUT_SHA,
            "input parity drift")
    require(rtx["coverage_sha256"] == mac["coverage_sha256"] == COVERAGE_SHA,
            "coverage parity drift")
    require(rtx["output_sha256_by_poison"] == mac["output_sha256_by_poison"],
            "output parity drift")
    result = {
        "schema": "neuralstudio.fg.early-l8-l18-accepted.v1",
        "selected_units": [
            "canonical L8/module-000025 reset scatter mask",
            "canonical L18/module-000025 generated scatter mask",
        ],
        "abi_receipt_root_sha256": abi[ROOT_KEY],
        "rtx_backend_root_sha256": rtx[ROOT_KEY],
        "mac_backend_root_sha256": mac[ROOT_KEY],
        "module_sha256": MODULE_SHA,
        "input_sha256": INPUT_SHA,
        "coverage_sha256": COVERAGE_SHA,
        "output_sha256_by_poison": rtx["output_sha256_by_poison"],
        "rtx_mac_output_mismatched_bytes_by_poison": {"165": 0, "90": 0},
        "rtx_mac_coverage_mismatched_positions": 0,
        "processes_per_backend": 8,
        "written_zero_pixels": WRITTEN_PIXELS,
        "untouched_poison_pixels": UNTOUCHED_PIXELS,
        "output_domain_only_zero_or_poison": True,
        "exact_coverage_positions": True,
        "guard_row_mismatched_bytes": 0,
        "input_and_rom_immutable": True,
        "expected_or_archived_ingress": False,
        "address_fields_recorded": False,
        "claims": {
            "l8_standalone_exact_on_metal": True,
            "l18_standalone_exact_on_metal": True,
            "connected_early_chain": False,
            "generated_frame": False,
            "netal_integration": False,
            "performance": False,
        },
        "result": "pass",
    }
    return rooted(result)


def validate_final(value: dict[str, Any], rtx: dict[str, Any],
                   mac: dict[str, Any], abi: dict[str, Any]) -> None:
    verify_root(value, "accepted")
    require(value == make_final(rtx, mac, abi), "accepted receipt drift")


def make_negative(accepted: dict[str, Any], rtx: dict[str, Any],
                  mac: dict[str, Any], abi: dict[str, Any]) -> dict[str, Any]:
    cases: list[tuple[str, dict[str, Any], dict[str, Any],
                      dict[str, Any], dict[str, Any]]] = []

    def add(name: str) -> tuple[dict[str, Any], dict[str, Any],
                                dict[str, Any], dict[str, Any]]:
        values = (copy.deepcopy(accepted), copy.deepcopy(rtx),
                  copy.deepcopy(mac), copy.deepcopy(abi))
        cases.append((name, *values))
        return values

    a, _, _, _ = add("accepted-module-drift"); a["module_sha256"] = "0" * 64
    a, _, _, _ = add("accepted-coverage-drift"); a["coverage_sha256"] = "1" * 64
    a, _, _, _ = add("accepted-output-domain-drift"); a["output_domain_only_zero_or_poison"] = False
    a, _, _, _ = add("accepted-parity-drift"); a["rtx_mac_coverage_mismatched_positions"] = 1
    a, _, _, _ = add("accepted-overclaim"); a["claims"]["generated_frame"] = True
    _, r, _, _ = add("rtx-process-drift"); r["fresh_process_count"] = 7
    _, r, _, _ = add("rtx-coverage-position-drift"); r["exact_coverage_positions_across_all_runs"] = False
    _, r, _, _ = add("rtx-helper-provenance-drift"); r["source_audit"]["shared_helper_provenance_gap_closed"] = False
    _, _, m, _ = add("mac-rom-drift"); m["rom_immutable"] = False
    _, _, m, _ = add("mac-output-domain-drift"); m["output_domain"] = [0, 90, 165]
    _, _, m, _ = add("mac-ingress-drift"); m["expected_or_archived_ingress"] = True
    _, _, _, b = add("abi-size-drift"); b["packed_parameter_bytes"] = 44
    _, _, _, b = add("abi-launch-drift"); b["canonical_launches"] = [8]
    a, _, _, _ = add("accepted-root-corruption"); a[ROOT_KEY] = "f" * 64

    results = []
    for name, mutated_accepted, mutated_rtx, mutated_mac, mutated_abi in cases:
        rejected = False
        reason = ""
        try:
            validate_final(mutated_accepted, mutated_rtx, mutated_mac,
                           mutated_abi)
        except VerificationError as error:
            rejected = True
            reason = str(error)
        require(rejected, f"negative mutation accepted: {name}")
        results.append({"mutation": name, "rejected": True, "reason": reason})
    return rooted({
        "schema": "neuralstudio.fg.early-l8-l18-negative-audit.v1",
        "accepted_receipt_root_sha256": accepted[ROOT_KEY],
        "mutation_count": len(results),
        "mutations": results,
        "all_mutations_rejected": True,
        "result": "pass",
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    backend = commands.add_parser("backend")
    backend.add_argument("--backend", choices=("rtx", "mac"), required=True)
    backend.add_argument("--run-root", type=pathlib.Path, required=True)
    backend.add_argument("--repository", type=pathlib.Path, required=True)
    backend.add_argument("--source-audit", type=pathlib.Path)
    backend.add_argument("--output", type=pathlib.Path, required=True)
    final = commands.add_parser("final")
    final.add_argument("--rtx", type=pathlib.Path, required=True)
    final.add_argument("--mac", type=pathlib.Path, required=True)
    final.add_argument("--abi", type=pathlib.Path, required=True)
    final.add_argument("--output", type=pathlib.Path, required=True)
    receipt = commands.add_parser("receipt")
    receipt.add_argument("--accepted", type=pathlib.Path, required=True)
    receipt.add_argument("--rtx", type=pathlib.Path, required=True)
    receipt.add_argument("--mac", type=pathlib.Path, required=True)
    receipt.add_argument("--abi", type=pathlib.Path, required=True)
    negative = commands.add_parser("negative")
    negative.add_argument("--accepted", type=pathlib.Path, required=True)
    negative.add_argument("--rtx", type=pathlib.Path, required=True)
    negative.add_argument("--mac", type=pathlib.Path, required=True)
    negative.add_argument("--abi", type=pathlib.Path, required=True)
    negative.add_argument("--output", type=pathlib.Path, required=True)
    arguments = parser.parse_args()
    try:
        if arguments.command == "backend":
            value = verify_backend(arguments.backend, arguments.run_root,
                                   arguments.repository, arguments.source_audit)
            write_json(arguments.output, value)
        elif arguments.command == "final":
            value = make_final(load(arguments.rtx), load(arguments.mac),
                               load(arguments.abi))
            write_json(arguments.output, value)
        elif arguments.command == "receipt":
            value = load(arguments.accepted)
            validate_final(value, load(arguments.rtx), load(arguments.mac),
                           load(arguments.abi))
        else:
            accepted = load(arguments.accepted)
            rtx = load(arguments.rtx)
            mac = load(arguments.mac)
            abi = load(arguments.abi)
            validate_final(accepted, rtx, mac, abi)
            value = make_negative(accepted, rtx, mac, abi)
            write_json(arguments.output, value)
        print(f"receipt_root_sha256={value[ROOT_KEY]}")
        print("verification_result=pass")
        return 0
    except VerificationError as error:
        print(f"verification_result=fail\nerror={error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
