#!/usr/bin/env python3
"""Verify immutable receipts for canonical FG reset-copy launch L11."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pathlib
import sys
from typing import Any


MODULE_SHA = "c11cd7474562e20762d272162c94378a17f8f77c7de8206d695e6c469383e840"
RAW_PARAMETER_SHA = "b18c2cc81aac4bc673b19c101dcb331dfa86693659bf1dbfe1a77d4a52f3e3a7"
CANON_PARAMETER_SHA = "7e578a6c893a77c02fed71dcf1cc94624cb5e995cbfa77a4ad94b6a50bd48d71"
FIXTURE_PARAMETER_SHA = "9b4595a88249e4192e7a66db345af68a8a53ec2e550bf86935ad5e974a48a723"
INPUT_SHA = "34ef70b219e8a0064118e4a4bcec17f666f36807dd6400a525c2cbad7c109dee"
REGION_SHA = "a637351f441a2d68a1cdce0fe99a8b9a22eecad02623481f879d0e19a8df3ed7"
OUTPUT_SHA = {
    165: "9da7039e9ee6a86fbad5be1b3a23be617ff1036787fab90c1a72028bb00b965b",
    90: "de8d9f99dfbdf3c8d4b500de0a8998879b467f5c9a41d659087217cc5ca511df",
}
RUNS = {
    "l11-poison-a5-process-a": 165,
    "l11-poison-a5-process-b": 165,
    "l11-poison-5a-process-a": 90,
    "l11-poison-5a-process-b": 90,
}
TEXTURE_WIDTH = 640
TEXTURE_HEIGHT = 360
OUTPUT_ORIGIN = (29, 19)
COPY_EXTENT = (587, 317)
TEXTURE_BYTES = 921600
REGION_BYTES = 744316
OUTSIDE_PIXELS = 44321
ROOT_KEY = "receipt_root_sha256"


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
            f"{label} root missing")
    body = copy.deepcopy(value)
    body.pop(ROOT_KEY, None)
    require(sha_bytes(canonical(body)) == expected, f"{label} root mismatch")


def write_json(path: pathlib.Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")


def in_region(x: int, y: int) -> bool:
    return (OUTPUT_ORIGIN[0] <= x < OUTPUT_ORIGIN[0] + COPY_EXTENT[0] and
            OUTPUT_ORIGIN[1] <= y < OUTPUT_ORIGIN[1] + COPY_EXTENT[1])


def untouched_intersection(output_a5: bytes, output_5a: bytes) -> int:
    require(len(output_a5) == TEXTURE_BYTES and len(output_5a) == TEXTURE_BYTES,
            "dual-poison artifact size drift")
    count = 0
    for y in range(TEXTURE_HEIGHT):
        for x in range(TEXTURE_WIDTH):
            if not in_region(x, y):
                continue
            offset = (y * TEXTURE_WIDTH + x) * 4
            a = output_a5[offset:offset + 4]
            b = output_5a[offset:offset + 4]
            count += a == bytes([165]) * 4 and b == bytes([90]) * 4
    return count


def verify_artifact(path: pathlib.Path, expected_sha: str,
                    expected_bytes: int) -> dict[str, Any]:
    require(path.is_file(), f"artifact missing: {path}")
    require(path.stat().st_size == expected_bytes, f"artifact size drift: {path}")
    digest = sha_file(path)
    require(digest == expected_sha, f"artifact hash drift: {path}")
    return {"file": path.name, "bytes": expected_bytes, "sha256": digest}


def verify_backend(backend: str, run_root: pathlib.Path,
                   repository: pathlib.Path) -> dict[str, Any]:
    require(backend in ("rtx", "mac"), "unknown backend")
    expected_backend = "rtx-d3d12-nvapi" if backend == "rtx" else "mac-metal"
    receipts: dict[str, dict[str, Any]] = {}
    receipt_files: list[dict[str, Any]] = []
    process_ids: list[int] = []
    for run_id, poison in RUNS.items():
        directory = run_root / run_id
        receipt_path = directory / "run-receipt.json"
        receipt = load(receipt_path)
        receipts[run_id] = receipt
        receipt_files.append({
            "run_id": run_id,
            "bytes": receipt_path.stat().st_size,
            "sha256": sha_file(receipt_path),
        })
        require(receipt.get("schema") == "neuralstudio.fg.early-l11-run.v1",
                f"{run_id}: schema drift")
        require(receipt.get("backend") == expected_backend,
                f"{run_id}: backend drift")
        require(receipt.get("run_id") == run_id, f"{run_id}: identity drift")
        require(receipt.get("poison_byte") == poison, f"{run_id}: poison drift")
        require(receipt.get("selected_launch") == 11,
                f"{run_id}: launch drift")
        module_key = "module_sha256" if backend == "rtx" else "source_module_sha256"
        require(receipt.get(module_key) == MODULE_SHA, f"{run_id}: module drift")
        require(receipt.get("packed_parameter_bytes") == 40,
                f"{run_id}: ABI size drift")
        require(receipt.get("frozen_selected_raw_parameter_sha256") ==
                RAW_PARAMETER_SHA, f"{run_id}: raw ABI drift")
        require(receipt.get("frozen_selected_address_free_parameter_sha256") ==
                CANON_PARAMETER_SHA, f"{run_id}: canonical ABI drift")
        require(receipt.get("fixture_address_free_parameter_sha256") ==
                FIXTURE_PARAMETER_SHA, f"{run_id}: fixture ABI drift")
        require(receipt.get("resource_extent") == [640, 360],
                f"{run_id}: resource extent drift")
        require(receipt.get("input_origin") == [17, 11],
                f"{run_id}: input origin drift")
        require(receipt.get("output_origin") == [29, 19],
                f"{run_id}: output origin drift")
        require(receipt.get("copy_extent") == [587, 317],
                f"{run_id}: copy extent drift")
        require(receipt.get("input_pre_sha256") == INPUT_SHA and
                receipt.get("input_post_sha256") == INPUT_SHA,
                f"{run_id}: input provenance/mutation")
        require(receipt.get("region_output_sha256") == REGION_SHA,
                f"{run_id}: region hash drift")
        require(receipt.get("output_sha256") == OUTPUT_SHA[poison],
                f"{run_id}: full output hash drift")
        require(receipt.get("output_bytes") == TEXTURE_BYTES,
                f"{run_id}: output size drift")
        require(receipt.get("declared_region_bytes") == REGION_BYTES,
                f"{run_id}: region size drift")
        require(receipt.get("outside_region_pixels") == OUTSIDE_PIXELS,
                f"{run_id}: outside-domain size drift")
        for field in ("input_mismatched_bytes",
                      "declared_region_mismatched_bytes",
                      "outside_region_mismatched_bytes",
                      "canary_mismatched_bytes",
                      "post_execution_expected_mismatched_bytes",
                      "declared_region_poison_pixels"):
            require(receipt.get(field) == 0, f"{run_id}: {field} != 0")
        require(receipt.get("input_immutable") is True,
                f"{run_id}: input immutability false")
        require(receipt.get("expected_bytes_fed_to_execution") is False,
                f"{run_id}: expected ingress")
        require(receipt.get("archived_intermediates_consumed") is False,
                f"{run_id}: archived ingress")
        require(receipt.get("address_fields_recorded") is False,
                f"{run_id}: address field recorded")
        require(receipt.get("result") == "pass", f"{run_id}: did not pass")
        process_id = receipt.get("process_id")
        require(isinstance(process_id, int) and process_id > 0,
                f"{run_id}: PID missing")
        process_ids.append(process_id)
        verify_artifact(directory / "input-after.rgba8", INPUT_SHA, TEXTURE_BYTES)
        verify_artifact(directory / "output.rgba8", OUTPUT_SHA[poison],
                        TEXTURE_BYTES)
        verify_artifact(directory / "region-output.rgba8", REGION_SHA,
                        REGION_BYTES)

    require(len(set(process_ids)) == 4, "fresh-process gate failed")
    require(len({r["region_output_sha256"] for r in receipts.values()}) == 1,
            "two-poison region invariance failed")
    require(len({r["input_pre_sha256"] for r in receipts.values()}) == 1,
            "input generator drift")
    for poison in (165, 90):
        pair = [r for r in receipts.values() if r["poison_byte"] == poison]
        require(len({r["output_sha256"] for r in pair}) == 1,
                f"poison {poison}: fresh-process output drift")
    a5 = (run_root / "l11-poison-a5-process-a/output.rgba8").read_bytes()
    five_a = (run_root / "l11-poison-5a-process-a/output.rgba8").read_bytes()
    untouched = untouched_intersection(a5, five_a)
    require(untouched == 0, "dual-poison declared-region coverage failed")

    source_files: list[dict[str, Any]] = []
    build_artifacts: list[dict[str, Any]] = []
    rig_receipt = None
    if backend == "rtx":
        rig_path = run_root / "rig-receipt.json"
        rig = load(rig_path)
        require(rig.get("schema") == "neuralstudio.fg.early-l11-rig.v1",
                "rig schema drift")
        require(rig.get("result") == "pass", "rig receipt did not pass")
        require(rig.get("fresh_process_count") == 4,
                "rig fresh-process count drift")
        require(rig.get("two_poison_region_invariant") is True,
                "rig poison gate drift")
        require(rig.get("module", {}).get("sha256") == MODULE_SHA,
                "rig module drift")
        remote_hashes = {x["file"]: x["sha256"] for x in rig["sources"]}
        local_sources = {
            "rtx_nvapi_oracle.cpp": repository / "src/fg_early_l11/rtx_nvapi_oracle.cpp",
            "contract.h": repository / "src/fg_early_l11/contract.h",
            "fg_compositor_l87/rtx_nvapi_oracle.cpp": repository / "src/fg_compositor_l87/rtx_nvapi_oracle.cpp",
            "fg_compositor_l87/compositor_l87_contract.h": repository / "src/fg_compositor_l87/compositor_l87_contract.h",
            "build-fg-early-l11-rtx.ps1": repository / "scripts/build-fg-early-l11-rtx.ps1",
        }
        for name, path in local_sources.items():
            digest = sha_file(path)
            require(remote_hashes.get(name) == digest,
                    f"rig/local source hash drift: {name}")
            source_files.append({"file": name, "sha256": digest})
        rig_receipt = {
            "bytes": rig_path.stat().st_size,
            "sha256": sha_file(rig_path),
            "job_id": rig["job_id"],
            "attempt_id": rig["attempt_id"],
            "gpu": rig["gpu"],
            "cuda_toolkit": rig["cuda_toolkit"],
        }
        binary = run_root / "build/fg-early-l11-rtx.exe"
        build_artifacts.append({
            "file": binary.name,
            "bytes": binary.stat().st_size,
            "sha256": sha_file(binary),
        })
    else:
        local_sources = [
            repository / "src/fg_early_l11/early_l11.metal",
            repository / "src/fg_early_l11/macos_probe.mm",
            repository / "src/fg_early_l11/contract.h",
            repository / "scripts/build-macos-fg-early-l11.sh",
        ]
        for path in local_sources:
            source_files.append({"file": path.name, "sha256": sha_file(path)})
        for path in (run_root / "build/early_l11.metallib",
                     run_root / "build/fg-early-l11-mac"):
            require(path.is_file(), f"Mac build artifact missing: {path}")
            build_artifacts.append({
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha_file(path),
            })

    result = {
        "schema": "neuralstudio.fg.early-l11-backend.v1",
        "backend": expected_backend,
        "selected_launch": 11,
        "module_token": 4,
        "module_sha256": MODULE_SHA,
        "packed_parameter_bytes": 40,
        "fixture": {
            "resource_extent": [640, 360],
            "input_origin": [17, 11],
            "output_origin": [29, 19],
            "copy_extent": [587, 317],
            "input_sha256": INPUT_SHA,
        },
        "run_count": 4,
        "fresh_process_count": 4,
        "process_ids_distinct": True,
        "poison_bytes": [165, 90],
        "full_output_sha256_by_poison": {
            "165": OUTPUT_SHA[165],
            "90": OUTPUT_SHA[90],
        },
        "region_output_sha256": REGION_SHA,
        "region_output_bytes": REGION_BYTES,
        "outside_region_pixels": OUTSIDE_PIXELS,
        "two_poison_region_invariant": True,
        "dual_poison_untouched_region_pixels": untouched,
        "declared_region_full_write_coverage": True,
        "outside_region_canary_mismatched_bytes": 0,
        "input_mismatched_bytes": 0,
        "input_immutable": True,
        "expected_or_archived_ingress": False,
        "address_fields_recorded": False,
        "run_receipts": receipt_files,
        "source_files": source_files,
        "build_artifacts": build_artifacts,
        "rig_receipt": rig_receipt,
        "result": "pass",
    }
    return rooted(result)


def validate_backend(value: dict[str, Any], backend: str) -> None:
    verify_root(value, backend)
    expected = "rtx-d3d12-nvapi" if backend == "rtx" else "mac-metal"
    require(value.get("schema") == "neuralstudio.fg.early-l11-backend.v1",
            f"{backend}: schema drift")
    require(value.get("backend") == expected, f"{backend}: backend drift")
    require(value.get("selected_launch") == 11, f"{backend}: launch drift")
    require(value.get("module_sha256") == MODULE_SHA, f"{backend}: module drift")
    require(value.get("fresh_process_count") == 4 and
            value.get("process_ids_distinct") is True,
            f"{backend}: fresh-process gate")
    require(value.get("poison_bytes") == [165, 90], f"{backend}: poison drift")
    require(value.get("region_output_sha256") == REGION_SHA,
            f"{backend}: region drift")
    require(value.get("dual_poison_untouched_region_pixels") == 0 and
            value.get("declared_region_full_write_coverage") is True,
            f"{backend}: write coverage")
    require(value.get("outside_region_canary_mismatched_bytes") == 0,
            f"{backend}: canary failure")
    require(value.get("input_immutable") is True and
            value.get("input_mismatched_bytes") == 0,
            f"{backend}: input mutation")
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
    require(abi.get("schema") == "neuralstudio.fg.early-l11-abi.v1",
            "ABI schema drift")
    require(abi.get("module_sha256") == MODULE_SHA, "ABI module drift")
    require(abi.get("packed_parameter_bytes") == 40, "ABI size drift")
    require(rtx["fixture"] == mac["fixture"], "fixture drift")
    require(rtx["region_output_sha256"] == mac["region_output_sha256"],
            "RTX/Metal region mismatch")
    require(rtx["full_output_sha256_by_poison"] ==
            mac["full_output_sha256_by_poison"],
            "RTX/Metal full output mismatch")
    result = {
        "schema": "neuralstudio.fg.early-l11-accepted.v1",
        "selected_unit": "canonical L11/module-000004/main_kernel reset presentation copy",
        "abi_receipt_root_sha256": abi[ROOT_KEY],
        "rtx_backend_root_sha256": rtx[ROOT_KEY],
        "mac_backend_root_sha256": mac[ROOT_KEY],
        "module_sha256": MODULE_SHA,
        "fixture_address_free_parameter_sha256": FIXTURE_PARAMETER_SHA,
        "input_sha256": INPUT_SHA,
        "region_output_sha256": REGION_SHA,
        "region_output_bytes": REGION_BYTES,
        "full_output_sha256_by_poison": rtx["full_output_sha256_by_poison"],
        "rtx_mac_region_mismatched_bytes": 0,
        "rtx_mac_full_output_mismatched_bytes_by_poison": {"165": 0, "90": 0},
        "four_fresh_processes_per_backend": True,
        "two_poison_region_invariant": True,
        "dual_poison_untouched_region_pixels": 0,
        "outside_region_canary_mismatched_bytes": 0,
        "input_immutable": True,
        "expected_or_archived_ingress": False,
        "address_fields_recorded": False,
        "claims": {
            "selected_reset_copy_exact_on_metal": True,
            "nonzero_origin_and_subrect_exact": True,
            "generated_frame_arithmetic": False,
            "connected_early_chain": False,
            "complete_frame": False,
            "netal_integration": False,
            "performance": False,
        },
        "result": "pass",
    }
    return rooted(result)


def validate_final(value: dict[str, Any], abi: dict[str, Any],
                   rtx: dict[str, Any], mac: dict[str, Any]) -> None:
    verify_root(value, "accepted")
    expected = make_final(rtx, mac, abi)
    require(value == expected, "accepted receipt content drift")


def make_negative(accepted: dict[str, Any], abi: dict[str, Any],
                  rtx: dict[str, Any], mac: dict[str, Any]) -> dict[str, Any]:
    mutations: list[tuple[str, dict[str, Any], dict[str, Any], dict[str, Any],
                          dict[str, Any]]] = []

    def add(name: str, a=accepted, b=abi, r=rtx, m=mac) -> None:
        mutations.append((name, copy.deepcopy(a), copy.deepcopy(b),
                          copy.deepcopy(r), copy.deepcopy(m)))

    add("accepted-launch-drift")
    mutations[-1][1]["selected_unit"] = "canonical L12"
    add("accepted-module-drift")
    mutations[-1][1]["module_sha256"] = "0" * 64
    add("accepted-region-drift")
    mutations[-1][1]["region_output_sha256"] = "1" * 64
    add("accepted-parity-drift")
    mutations[-1][1]["rtx_mac_region_mismatched_bytes"] = 1
    add("accepted-scope-overclaim")
    mutations[-1][1]["claims"]["complete_frame"] = True
    add("abi-size-drift")
    mutations[-1][2]["packed_parameter_bytes"] = 48
    add("rtx-fresh-process-drift")
    mutations[-1][3]["fresh_process_count"] = 3
    add("rtx-canary-drift")
    mutations[-1][3]["outside_region_canary_mismatched_bytes"] = 1
    add("mac-coverage-drift")
    mutations[-1][4]["dual_poison_untouched_region_pixels"] = 1
    add("mac-input-mutation")
    mutations[-1][4]["input_immutable"] = False
    add("rtx-forbidden-ingress")
    mutations[-1][3]["expected_or_archived_ingress"] = True
    add("accepted-root-corruption")
    mutations[-1][1][ROOT_KEY] = "f" * 64

    results: list[dict[str, Any]] = []
    for name, mutated_accepted, mutated_abi, mutated_rtx, mutated_mac in mutations:
        rejected = False
        reason = ""
        try:
            validate_final(mutated_accepted, mutated_abi, mutated_rtx, mutated_mac)
        except VerificationError as error:
            rejected = True
            reason = str(error)
        require(rejected, f"negative mutation accepted: {name}")
        results.append({"mutation": name, "rejected": True, "reason": reason})
    return rooted({
        "schema": "neuralstudio.fg.early-l11-negative-audit.v1",
        "accepted_receipt_root_sha256": accepted[ROOT_KEY],
        "mutation_count": len(results),
        "mutations": results,
        "all_mutations_rejected": True,
        "result": "pass",
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    backend_parser = subparsers.add_parser("backend")
    backend_parser.add_argument("--backend", choices=("rtx", "mac"), required=True)
    backend_parser.add_argument("--run-root", type=pathlib.Path, required=True)
    backend_parser.add_argument("--repository", type=pathlib.Path, required=True)
    backend_parser.add_argument("--output", type=pathlib.Path, required=True)
    final_parser = subparsers.add_parser("final")
    final_parser.add_argument("--rtx", type=pathlib.Path, required=True)
    final_parser.add_argument("--mac", type=pathlib.Path, required=True)
    final_parser.add_argument("--abi", type=pathlib.Path, required=True)
    final_parser.add_argument("--output", type=pathlib.Path, required=True)
    receipt_parser = subparsers.add_parser("receipt")
    receipt_parser.add_argument("--accepted", type=pathlib.Path, required=True)
    receipt_parser.add_argument("--rtx", type=pathlib.Path, required=True)
    receipt_parser.add_argument("--mac", type=pathlib.Path, required=True)
    receipt_parser.add_argument("--abi", type=pathlib.Path, required=True)
    negative_parser = subparsers.add_parser("negative")
    negative_parser.add_argument("--accepted", type=pathlib.Path, required=True)
    negative_parser.add_argument("--rtx", type=pathlib.Path, required=True)
    negative_parser.add_argument("--mac", type=pathlib.Path, required=True)
    negative_parser.add_argument("--abi", type=pathlib.Path, required=True)
    negative_parser.add_argument("--output", type=pathlib.Path, required=True)
    arguments = parser.parse_args()
    try:
        if arguments.command == "backend":
            value = verify_backend(arguments.backend, arguments.run_root,
                                   arguments.repository)
            write_json(arguments.output, value)
        elif arguments.command == "final":
            value = make_final(load(arguments.rtx), load(arguments.mac),
                               load(arguments.abi))
            write_json(arguments.output, value)
        elif arguments.command == "receipt":
            validate_final(load(arguments.accepted), load(arguments.abi),
                           load(arguments.rtx), load(arguments.mac))
            value = load(arguments.accepted)
        else:
            accepted = load(arguments.accepted)
            abi = load(arguments.abi)
            rtx = load(arguments.rtx)
            mac = load(arguments.mac)
            validate_final(accepted, abi, rtx, mac)
            value = make_negative(accepted, abi, rtx, mac)
            write_json(arguments.output, value)
        print(f"receipt_root_sha256={value.get(ROOT_KEY, 'not-rooted-negative-audit')}")
        print("verification_result=pass")
        return 0
    except VerificationError as error:
        print(f"verification_result=fail\nerror={error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
