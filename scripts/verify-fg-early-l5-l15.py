#!/usr/bin/env python3
"""Verify standalone selected-module parity for canonical FG L5/L15."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pathlib
import struct
import sys
from typing import Any


MODULE_SHA = "fd0665969da887dd92540b673009e0361a6c1de74fe54badf48f300b1fccbbf1"
ROM_SHA = "7cf4f8cece6b18b07605edc9cb601a153346a79b10bf3b93c7ec058d57d29750"
FIXTURE_PARAMETER_SHA = "770bf9ca1edcae17222a64a4583ac19a6c5e2b3ef2648d1c2145efc1f6cdb12c"
SENTINEL = 0x7FFFFFFF7FFFFFFF
PIXELS = 57600
FLOW_BYTES = 460800
DEPTH_BYTES = 230400
ATOMIC_BYTES = 460800
MASK_BYTES = 57600
ROOT_KEY = "receipt_root_sha256"
DOMAINS = {
    ("reset", "equal-high-tie"): {
        "launch": 5,
        "flow": "cfe29b27dc2a2e8484409577e6faf7906776c3b426296e532bb1f4d62a3ca57d",
        "depth": "bb7a01adb087b6d1b2c9fdee6a2f79fc9cd12568dc9f4284d29ea8f7a9b60b7c",
        "atomic": "263232475f4defe8d572f93b1cb283849a2600cc289a82cecb9ecd93a6351d28",
        "mask": "b10081cd219841a6ac25cdf800bbae4a47841792169e5cf3b08f90171e43a1f6",
        "sentinel": 28383, "candidate": 29217, "high": 2, "low": 9,
        "mask_zero": 33565, "mask_one": 24035,
    },
    ("reset", "priority-collision"): {
        "launch": 5,
        "flow": "867f3d0f108ef09bca09d1c71f1745f589df540955f916d837f79f4088c42be5",
        "depth": "6b8ac30433467992069960c44badfad5b37a56c14d5735044e2eb755271d4acd",
        "atomic": "d570db84aff862ed39b78476c8b19db6e1ae7cd29b0b6041ed3dbc866d9c83f3",
        "mask": "97d9d0b2c49dc3e58f3f62173cf994063756565f7fa4a84c8decfa1cc6f265b2",
        "sentinel": 28610, "candidate": 28990, "high": 6, "low": 9,
        "mask_zero": 28937, "mask_one": 28663,
    },
    ("generated", "equal-high-tie"): {
        "launch": 15,
        "flow": "a47d458f22f1e928d32ac52d07dcf8b053a51636857434f8316181326ca09d4a",
        "depth": "bb7a01adb087b6d1b2c9fdee6a2f79fc9cd12568dc9f4284d29ea8f7a9b60b7c",
        "atomic": "b1ffda4f6bd8805f557285e2f812b17250f2a5514237bbf8a336ec7da5188dd3",
        "mask": "8e0bf6dadd55ddccd57d49d67b0c761fa47eeaaa96ca8f85982196f833b66334",
        "sentinel": 28562, "candidate": 29038, "high": 2, "low": 9,
        "mask_zero": 28984, "mask_one": 28616,
    },
    ("generated", "priority-collision"): {
        "launch": 15,
        "flow": "bf00c583f1e8ebe488f87b669985dc9e627d800ec3a9ee6f5dd4218e6cedf38e",
        "depth": "15bc0555ab17409909a783933a93dc165b32dc2aedcca745ace651488d34e393",
        "atomic": "75ccfc5d1c4b1b185769992a63e39e7daffb9aa65fcb6865f307e92884509b88",
        "mask": "90eb9af21847d164e3239ad2bc7fa6e5618e8815148c9b67d411a86fb9f6ee0f",
        "sentinel": 28366, "candidate": 29234, "high": 6, "low": 9,
        "mask_zero": 33513, "mask_one": 24087,
    },
}
RUNS = {
    f"l{domain['launch']}-{phase}-{scenario}-poison-{poison_name}-process-{ordinal}":
        (phase, scenario, poison)
    for (phase, scenario), domain in DOMAINS.items()
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


def verify_file(path: pathlib.Path, digest: str, size: int) -> None:
    require(path.is_file(), f"artifact missing: {path}")
    require(path.stat().st_size == size, f"artifact size drift: {path}")
    require(sha_file(path) == digest, f"artifact hash drift: {path}")


def classify_atomic(data: bytes) -> dict[str, int]:
    sentinel = candidate = negative = 0
    highs: set[int] = set()
    lows: set[int] = set()
    for offset in range(0, len(data), 8):
        value = struct.unpack_from("<Q", data, offset)[0]
        if value == SENTINEL:
            sentinel += 1
            continue
        candidate += 1
        high = value >> 32
        low = value & 0xFFFFFFFF
        negative += bool(high & 0x80000000)
        highs.add(high)
        lows.add(low)
    return {"sentinel": sentinel, "candidate": candidate,
            "negative": negative, "high": len(highs), "low": len(lows)}


def classify_mask(data: bytes, poison: int) -> dict[str, int]:
    return {"zero": data.count(0), "one": data.count(255),
            "poison": data.count(poison) if poison not in (0, 255) else 0,
            "other": sum(value not in (0, 255) for value in data)}


def verify_backend(backend: str, root: pathlib.Path,
                   repository: pathlib.Path) -> dict[str, Any]:
    require(backend in ("rtx", "mac"), "unknown backend")
    expected_backend = "rtx-cuda-driver" if backend == "rtx" else "mac-metal"
    process_ids: list[int] = []
    run_receipts: list[dict[str, Any]] = []
    observed: dict[tuple[str, str], tuple[str, str]] = {}
    for run_id, (phase, scenario, poison) in RUNS.items():
        expected = DOMAINS[(phase, scenario)]
        directory = root / run_id
        receipt_path = directory / "run-receipt.json"
        receipt = load(receipt_path)
        run_receipts.append({"run_id": run_id,
                             "bytes": receipt_path.stat().st_size,
                             "sha256": sha_file(receipt_path)})
        require(receipt.get("schema") == "neuralstudio.fg.early-l5-l15-run.v1",
                f"{run_id}: schema drift")
        require(receipt.get("backend") == expected_backend,
                f"{run_id}: backend drift")
        require(receipt.get("run_id") == run_id and
                receipt.get("phase") == phase and
                receipt.get("scenario") == scenario,
                f"{run_id}: identity drift")
        require(receipt.get("selected_launch") == expected["launch"] and
                receipt.get("poison_byte") == poison,
                f"{run_id}: launch/poison drift")
        require(receipt.get("module_token") == 13 and
                receipt.get("module_sha256") == MODULE_SHA and
                receipt.get("packed_parameter_bytes") == 144,
                f"{run_id}: module/ABI drift")
        require(receipt.get("flow_pre_sha256") == expected["flow"] and
                receipt.get("flow_post_sha256") == expected["flow"] and
                receipt.get("depth_pre_sha256") == expected["depth"] and
                receipt.get("depth_post_sha256") == expected["depth"],
                f"{run_id}: input provenance/mutation")
        require(receipt.get("atomic_output_sha256") == expected["atomic"] and
                receipt.get("mask_output_sha256") == expected["mask"],
                f"{run_id}: output hash drift")
        require(receipt.get("atomic_output_bytes") == ATOMIC_BYTES and
                receipt.get("mask_output_bytes") == MASK_BYTES,
                f"{run_id}: output size drift")
        require(receipt.get("atomic_sentinel_u64") == expected["sentinel"] and
                receipt.get("atomic_candidate_u64") == expected["candidate"] and
                receipt.get("atomic_unique_high_words") == expected["high"] and
                receipt.get("atomic_unique_low_words") == expected["low"],
                f"{run_id}: atomic classification drift")
        require(receipt.get("mask_zero_pixels") == expected["mask_zero"] and
                receipt.get("mask_one_pixels") == expected["mask_one"] and
                receipt.get("mask_poison_pixels") == 0 and
                receipt.get("mask_other_pixels") == 0,
                f"{run_id}: mask classification drift")
        require(receipt.get("input_mismatched_bytes") == 0 and
                receipt.get("atomic_canary_mismatched_bytes") == 0 and
                receipt.get("mask_guard_row_mismatched_bytes") == 0,
                f"{run_id}: immutability/canary gate")
        require(receipt.get("collision_fixture") is True and
                receipt.get("equal_high_tie_fixture") is
                (scenario == "equal-high-tie"),
                f"{run_id}: collision/tie label drift")
        require(receipt.get("expected_or_archived_ingress") is False and
                receipt.get("address_fields_recorded") is False and
                receipt.get("result") == "pass",
                f"{run_id}: provenance/result gate")
        process = receipt.get("process_id")
        require(isinstance(process, int) and process > 0, f"{run_id}: PID missing")
        process_ids.append(process)

        flow = directory / "flow-after.bin"
        depth = directory / "depth-after.bin"
        atomic = directory / "atomic-output.bin"
        mask = directory / "mask-output.bin"
        verify_file(flow, expected["flow"], FLOW_BYTES)
        verify_file(depth, expected["depth"], DEPTH_BYTES)
        verify_file(atomic, expected["atomic"], ATOMIC_BYTES)
        verify_file(mask, expected["mask"], MASK_BYTES)
        atomic_class = classify_atomic(atomic.read_bytes())
        require(atomic_class == {
            "sentinel": expected["sentinel"],
            "candidate": expected["candidate"],
            "negative": 0,
            "high": expected["high"],
            "low": expected["low"],
        }, f"{run_id}: independently classified atomic output drift")
        mask_class = classify_mask(mask.read_bytes(), poison)
        require(mask_class == {
            "zero": expected["mask_zero"],
            "one": expected["mask_one"],
            "poison": 0,
            "other": 0,
        }, f"{run_id}: independently classified mask output drift")
        key = (phase, scenario)
        value = (receipt["atomic_output_sha256"], receipt["mask_output_sha256"])
        if key in observed:
            require(observed[key] == value, f"{run_id}: poison/process drift")
        observed[key] = value

    require(len(set(process_ids)) == 16, "fresh-process gate failed")
    require(len(observed) == 4, "domain coverage incomplete")

    source_files: list[dict[str, str]] = []
    build_artifacts: list[dict[str, Any]] = []
    rig_summary = None
    if backend == "rtx":
        rig_path = root / "rig-receipt.json"
        rig = load(rig_path)
        require(rig.get("schema") == "neuralstudio.fg.early-l5-l15-rig.v1" and
                rig.get("result") == "pass" and
                rig.get("fresh_process_count") == 16,
                "rig receipt rejected")
        local_sources = {
            "source_sha256": repository / "src/fg_early_l5_l15/rtx_oracle.cpp",
            "contract_sha256": repository / "src/fg_early_l5_l15/contract.h",
            "shared_helper_sha256": repository / "src/fg_conv/rtx_oracle.cpp",
            "harness_sha256": repository / "scripts/build-fg-early-l5-l15-rtx.ps1",
        }
        for field, path in local_sources.items():
            digest = sha_file(path)
            require(rig.get(field) == digest, f"rig/local source drift: {field}")
            source_files.append({"file": str(path.relative_to(repository)),
                                 "sha256": digest})
        binary = root / "build/fg-early-l5-l15-rtx.exe"
        require(sha_file(binary) == rig["binary_sha256"], "RTX binary drift")
        build_artifacts.append({"file": binary.name, "bytes": binary.stat().st_size,
                                "sha256": sha_file(binary)})
        rig_summary = {"job_id": rig["job_id"], "attempt_id": rig["attempt_id"],
                       "gpu": rig["gpu"], "cuda_toolkit": rig["cuda_toolkit"],
                       "file_sha256": sha_file(rig_path)}
    else:
        paths = [
            repository / "src/fg_early_l5_l15/macos_probe.mm",
            repository / "src/fg_early_l5_l15/early_l5_l15.metal",
            repository / "src/fg_early_l5_l15/contract.h",
            repository / "src/fg_conv/macos_probe.mm",
            repository / "scripts/build-macos-fg-early-l5-l15.sh",
        ]
        for path in paths:
            source_files.append({"file": str(path.relative_to(repository)),
                                 "sha256": sha_file(path)})
        for path, expected_sha in (
            (root / "build/early_l5_l15.metallib", None),
            (root / "build/fg-early-l5-l15-mac", None),
            (root / "build/sm120-rcp-correction-rom.bin", ROM_SHA),
        ):
            require(path.is_file(), f"Mac build artifact missing: {path}")
            digest = sha_file(path)
            if expected_sha is not None:
                require(digest == expected_sha, "RCP ROM identity drift")
            build_artifacts.append({"file": path.name, "bytes": path.stat().st_size,
                                    "sha256": digest})

    summaries = []
    for (phase, scenario), expected in DOMAINS.items():
        summaries.append({
            "phase": phase,
            "scenario": scenario,
            "selected_launch": expected["launch"],
            "flow_sha256": expected["flow"],
            "depth_sha256": expected["depth"],
            "atomic_output_sha256": expected["atomic"],
            "mask_output_sha256": expected["mask"],
            "atomic_sentinel_u64": expected["sentinel"],
            "atomic_candidate_u64": expected["candidate"],
            "atomic_unique_high_words": expected["high"],
            "atomic_unique_low_words": expected["low"],
            "mask_zero_pixels": expected["mask_zero"],
            "mask_one_pixels": expected["mask_one"],
        })
    result = {
        "schema": "neuralstudio.fg.early-l5-l15-backend.v1",
        "backend": expected_backend,
        "selected_launches": [5, 15],
        "module_token": 13,
        "module_sha256": MODULE_SHA,
        "packed_parameter_bytes": 144,
        "fixture_address_free_parameter_sha256": FIXTURE_PARAMETER_SHA,
        "run_count": 16,
        "fresh_process_count": 16,
        "poison_bytes": [165, 90],
        "domain_count": 4,
        "summaries": summaries,
        "atomic_signed64_order_exercised": True,
        "equal_high_distinct_low_ties_exercised": True,
        "mixed_high_priorities_exercised": True,
        "mask_output_domain": [0, 255],
        "mask_full_write_coverage": True,
        "mask_guard_row_mismatched_bytes": 0,
        "atomic_canary_mismatched_bytes": 0,
        "input_mismatched_bytes": 0,
        "input_immutable": True,
        "rom_sha256": ROM_SHA if backend == "mac" else None,
        "rom_immutable": True if backend == "mac" else None,
        "atomic_lowering": (
            "signed-high-xor/u32-min then equal-high low-u32-min and GPU pack"
            if backend == "mac" else "native atom.global.min.s64"),
        "expected_or_archived_ingress": False,
        "address_fields_recorded": False,
        "run_receipts": run_receipts,
        "source_files": source_files,
        "build_artifacts": build_artifacts,
        "rig": rig_summary,
        "result": "pass",
    }
    return rooted(result)


def validate_backend(value: dict[str, Any], backend: str) -> None:
    verify_root(value, backend)
    expected = "rtx-cuda-driver" if backend == "rtx" else "mac-metal"
    require(value.get("schema") == "neuralstudio.fg.early-l5-l15-backend.v1" and
            value.get("backend") == expected, f"{backend}: identity drift")
    require(value.get("selected_launches") == [5, 15] and
            value.get("module_sha256") == MODULE_SHA and
            value.get("packed_parameter_bytes") == 144,
            f"{backend}: selected ABI drift")
    require(value.get("fresh_process_count") == 16 and
            value.get("domain_count") == 4 and
            value.get("poison_bytes") == [165, 90],
            f"{backend}: process/domain gate")
    for field in ("atomic_signed64_order_exercised",
                  "equal_high_distinct_low_ties_exercised",
                  "mixed_high_priorities_exercised",
                  "mask_full_write_coverage", "input_immutable"):
        require(value.get(field) is True, f"{backend}: {field} false")
    require(value.get("mask_output_domain") == [0, 255] and
            value.get("mask_guard_row_mismatched_bytes") == 0 and
            value.get("atomic_canary_mismatched_bytes") == 0 and
            value.get("input_mismatched_bytes") == 0,
            f"{backend}: output/canary/input gate")
    if backend == "mac":
        require(value.get("rom_sha256") == ROM_SHA and
                value.get("rom_immutable") is True,
                "mac: ROM gate")
    require(value.get("expected_or_archived_ingress") is False and
            value.get("address_fields_recorded") is False and
            value.get("result") == "pass",
            f"{backend}: provenance/result gate")


def make_final(rtx: dict[str, Any], mac: dict[str, Any],
               abi: dict[str, Any]) -> dict[str, Any]:
    validate_backend(rtx, "rtx")
    validate_backend(mac, "mac")
    verify_root(abi, "ABI")
    require(abi.get("schema") == "neuralstudio.fg.early-l5-l15-abi.v1" and
            abi.get("canonical_launches") == [5, 15] and
            abi.get("module_sha256") == MODULE_SHA,
            "ABI identity drift")
    require(rtx["summaries"] == mac["summaries"], "RTX/Metal summary mismatch")
    return rooted({
        "schema": "neuralstudio.fg.early-l5-l15-accepted.v1",
        "selected_units": [
            "canonical L5/module-000013 reset atomic seed and mask",
            "canonical L15/module-000013 generated atomic seed and mask",
        ],
        "abi_receipt_root_sha256": abi[ROOT_KEY],
        "rtx_backend_root_sha256": rtx[ROOT_KEY],
        "mac_backend_root_sha256": mac[ROOT_KEY],
        "module_sha256": MODULE_SHA,
        "fixture_address_free_parameter_sha256": FIXTURE_PARAMETER_SHA,
        "domain_summaries": rtx["summaries"],
        "processes_per_backend": 16,
        "rtx_mac_atomic_mismatched_bytes": 0,
        "rtx_mac_mask_mismatched_bytes": 0,
        "equal_high_ties_exact": True,
        "mixed_priority_collisions_exact": True,
        "mask_full_write_coverage": True,
        "canary_mismatched_bytes": 0,
        "input_and_rom_immutable": True,
        "expected_or_archived_ingress": False,
        "address_fields_recorded": False,
        "claims": {
            "l5_standalone_exact_on_metal": True,
            "l15_standalone_exact_on_metal": True,
            "signed64_atomic_lowering_exact_for_accepted_domains": True,
            "connected_early_chain": False,
            "generated_frame": False,
            "netal_integration": False,
            "performance": False,
        },
        "result": "pass",
    })


def validate_final(value: dict[str, Any], rtx: dict[str, Any],
                   mac: dict[str, Any], abi: dict[str, Any]) -> None:
    verify_root(value, "accepted")
    require(value == make_final(rtx, mac, abi), "accepted receipt drift")


def make_negative(accepted: dict[str, Any], rtx: dict[str, Any],
                  mac: dict[str, Any], abi: dict[str, Any]) -> dict[str, Any]:
    cases = []

    def add(name: str):
        values = (copy.deepcopy(accepted), copy.deepcopy(rtx),
                  copy.deepcopy(mac), copy.deepcopy(abi))
        cases.append((name, *values))
        return values

    a, _, _, _ = add("accepted-module-drift"); a["module_sha256"] = "0" * 64
    a, _, _, _ = add("accepted-atomic-parity-drift"); a["rtx_mac_atomic_mismatched_bytes"] = 1
    a, _, _, _ = add("accepted-tie-drift"); a["equal_high_ties_exact"] = False
    a, _, _, _ = add("accepted-overclaim"); a["claims"]["generated_frame"] = True
    _, r, _, _ = add("rtx-process-drift"); r["fresh_process_count"] = 15
    _, r, _, _ = add("rtx-tie-domain-drift"); r["equal_high_distinct_low_ties_exercised"] = False
    _, r, _, _ = add("rtx-mask-domain-drift"); r["mask_output_domain"] = [0, 1]
    _, _, m, _ = add("mac-rom-drift"); m["rom_immutable"] = False
    _, _, m, _ = add("mac-canary-drift"); m["atomic_canary_mismatched_bytes"] = 1
    _, _, m, _ = add("mac-ingress-drift"); m["expected_or_archived_ingress"] = True
    _, _, m, _ = add("mac-summary-drift"); m["summaries"][0]["atomic_candidate_u64"] += 1
    _, _, _, b = add("abi-size-drift"); b["packed_parameter_bytes"] = 136
    _, _, _, b = add("abi-launch-drift"); b["canonical_launches"] = [5]
    a, _, _, _ = add("accepted-root-corruption"); a[ROOT_KEY] = "f" * 64
    results = []
    for name, ma, mr, mm, mb in cases:
        rejected = False
        reason = ""
        try:
            validate_final(ma, mr, mm, mb)
        except VerificationError as error:
            rejected = True
            reason = str(error)
        require(rejected, f"negative mutation accepted: {name}")
        results.append({"mutation": name, "rejected": True, "reason": reason})
    return rooted({
        "schema": "neuralstudio.fg.early-l5-l15-negative-audit.v1",
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
                                   arguments.repository)
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
