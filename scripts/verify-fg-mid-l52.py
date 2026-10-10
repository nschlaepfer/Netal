#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import struct
from typing import Any, Callable


RUN_SCHEMA = "neuralstudio.fg.mid-l52-run.v1"
BACKEND_SCHEMA = "neuralstudio.fg.mid-l52-backend.v1"
FINAL_SCHEMA = "neuralstudio.fg.mid-l52-verification.v1"
MATH_SCHEMA = "neuralstudio.fg.mid-l52-ex2-analysis.v1"
MODULE_SHA = "d45333f43f343d1f38bd304fe76214dc7064f2a7d24ce1ea06bc791867783264"
PARAMETER_SHA = "b85b7b8f373002d0518a784cfa9e9d8511979be6dd2641106b5c5acfb4a2a01d"
RCP_SHA = "7cf4f8cece6b18b07605edc9cb601a153346a79b10bf3b93c7ec058d57d29750"
GENERATOR = "splitmix64-fg-mid-l52-half16-depth-v1"
RUNS = (
    ("binary-depth-poison-a5", "binary-depth", 0xA5),
    ("binary-depth-poison-5a", "binary-depth", 0x5A),
    ("ramp-depth-poison-a5", "ramp-depth", 0xA5),
    ("ramp-depth-poison-5a", "ramp-depth", 0x5A),
)
ARTIFACTS = (
    ("output-a.rg16f", 320 * 180 * 4, "output_a_sha256"),
    ("output-b.rg16f", 320 * 180 * 4, "output_b_sha256"),
    ("confidence.r16f", 320 * 180 * 2, "confidence_sha256"),
)


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise VerificationError(f"unable to load {path}: {error}") from error
    require(isinstance(value, dict), f"{path}: expected object")
    return value


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: Any) -> str:
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True).encode())


def rooted(value: dict[str, Any], field: str = "receipt_root_sha256") -> bool:
    copied = copy.deepcopy(value)
    actual = copied.pop(field, None)
    return isinstance(actual, str) and actual == canonical(copied)


def reject_addresses(value: Any, path: str = "root") -> None:
    forbidden = ("gpu_virtual_address", "resource_identity", "device_pointer",
                 "host_pointer", "descriptor_ptr")
    if isinstance(value, dict):
        for key, child in value.items():
            require(not any(item in key.lower() for item in forbidden),
                    f"{path}.{key}: address material")
            reject_addresses(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_addresses(child, f"{path}[{index}]")


def read_artifacts(root: Path, run_id: str) -> tuple[list[bytes], dict[str, str]]:
    payloads, hashes = [], {}
    for name, size, _ in ARTIFACTS:
        data = (root / run_id / name).read_bytes()
        require(len(data) == size, f"{run_id}/{name}: size")
        payloads.append(data)
        hashes[name] = sha(data)
    return payloads, hashes


def validate_run(root: Path, run_id: str, fixture: str, poison: int,
                 backend: str) -> tuple[dict[str, Any], list[bytes]]:
    receipt = load(root / run_id / "run-receipt.json")
    reject_addresses(receipt, run_id)
    require(receipt.get("schema") == RUN_SCHEMA, f"{run_id}: schema")
    require(receipt.get("backend") ==
            ("rtx-d3d12-nvapi" if backend == "rtx" else "mac-metal"),
            f"{run_id}: backend")
    require(receipt.get("run_id") == run_id, f"{run_id}: id")
    require(receipt.get("fixture") == fixture, f"{run_id}: fixture")
    require(receipt.get("poison_byte") == poison, f"{run_id}: poison")
    require(isinstance(receipt.get("process_id"), int), f"{run_id}: process")
    require(receipt.get("selected_launch_token") == 52, f"{run_id}: launch")
    require(receipt.get("module_token") == 6, f"{run_id}: module")
    require(receipt.get("module_bytes") == 9064, f"{run_id}: module bytes")
    require(receipt.get("module_sha256") == MODULE_SHA, f"{run_id}: module hash")
    require(receipt.get("raw_live_parameter_sha256") == PARAMETER_SHA,
            f"{run_id}: parameter hash")
    require(receipt.get("packed_parameter_bytes") == 168, f"{run_id}: params")
    require(receipt.get("grid") == [20, 12, 1], f"{run_id}: grid")
    require(receipt.get("block") == [16, 16, 1], f"{run_id}: block")
    require(receipt.get("input_generator") == GENERATOR, f"{run_id}: generator")
    require(receipt.get("input_mismatched_bytes") == 0, f"{run_id}: input")
    require(receipt.get("buffer_canary_mismatched_bytes") == 0,
            f"{run_id}: canary")
    require(receipt.get("expected_output_fed_to_execution") is False,
            f"{run_id}: expected ingress")
    require(receipt.get("archived_intermediates_consumed") is False,
            f"{run_id}: archived ingress")
    require(receipt.get("address_fields_recorded") is False,
            f"{run_id}: addresses")
    require(receipt.get("result") == "pass", f"{run_id}: result")
    if backend == "rtx":
        require(receipt.get("resource_backend") ==
                "live-format-d3d12-descriptors-through-nvapi",
                f"{run_id}: resource backend")
    else:
        require(receipt.get("sm120_rcp_rom_sha256") == RCP_SHA,
                f"{run_id}: RCP ROM")
    outputs, hashes = read_artifacts(root, run_id)
    require(receipt.get("output_root_sha256") == sha(b"".join(outputs)),
            f"{run_id}: output root")
    require(receipt.get("output_bytes") == sum(map(len, outputs)),
            f"{run_id}: output bytes")
    for name, _, field in ARTIFACTS:
        require(receipt.get(field) == hashes[name], f"{run_id}: {name} hash")
    return receipt, outputs


def build_backend(root: Path, backend: str) -> dict[str, Any]:
    rows, outputs = [], {}
    for run_id, fixture, poison in RUNS:
        receipt, payloads = validate_run(root, run_id, fixture, poison, backend)
        outputs[run_id] = payloads
        rows.append({
            "run_id": run_id,
            "fixture": fixture,
            "poison_byte": poison,
            "process_id": receipt["process_id"],
            "input_root_sha256": receipt["input_root_sha256"],
            "output_root_sha256": receipt["output_root_sha256"],
        })
    require(len({row["process_id"] for row in rows}) == 4,
            "fresh process gate")
    for fixture in ("binary-depth", "ramp-depth"):
        pair = [row for row in rows if row["fixture"] == fixture]
        require(pair[0]["input_root_sha256"] == pair[1]["input_root_sha256"],
                f"{fixture}: input repeat")
        require(outputs[pair[0]["run_id"]] == outputs[pair[1]["run_id"]],
                f"{fixture}: poison invariance")
    document: dict[str, Any] = {
        "schema": BACKEND_SCHEMA,
        "backend": "rtx-d3d12-nvapi" if backend == "rtx" else "mac-metal",
        "selected_unit": "evaluate-1/launch-000052/main_kernel/module-000006",
        "module_sha256": MODULE_SHA,
        "raw_live_parameter_sha256": PARAMETER_SHA,
        "input_generator": GENERATOR,
        "runs": rows,
        "fresh_process_count": 4,
        "fixture_count": 2,
        "poison_bytes": [90, 165],
        "two_poison_output_invariance": True,
        "full_write_coverage_from_two_poison_invariance": True,
        "input_immutability": True,
        "buffer_canary_integrity": True,
        "expected_output_fed_to_execution": False,
        "archived_intermediates_consumed": False,
        "address_fields_recorded": False,
        "result": "pass",
    }
    document["receipt_root_sha256"] = canonical(document)
    return document


def histogram(differences: list[int]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in differences:
        key = str(value)
        result[key] = result.get(key, 0) + 1
    return dict(sorted(result.items(), key=lambda item: int(item[0])))


def build_math(rtx_dense: Path, rtx_random: Path, mac_dense: Path,
               mac_random: Path) -> dict[str, Any]:
    rd, md = rtx_dense.read_bytes(), mac_dense.read_bytes()
    rr, mr = rtx_random.read_bytes(), mac_random.read_bytes()
    require(len(rd) == len(md) == 4_194_305 * 4, "dense size")
    require(len(rr) == len(mr) == 1_048_576 * 12, "random size")
    rdw = struct.unpack(f"<{len(rd)//4}I", rd)
    mdw = struct.unpack(f"<{len(md)//4}I", md)
    dense_delta = [right - left for left, right in zip(rdw, mdw)]
    rrows = list(struct.iter_unpack("<IIHH", rr))
    mrows = list(struct.iter_unpack("<IIHH", mr))
    input_mismatches = sum(a[0] != b[0] for a, b in zip(rrows, mrows))
    ex2_delta = [b[1] - a[1] for a, b in zip(rrows, mrows)]
    low_delta = [b[2] - a[2] for a, b in zip(rrows, mrows)]
    high_delta = [b[3] - a[3] for a, b in zip(rrows, mrows)]
    require(input_mismatches == 0, "random generator mismatch")
    document: dict[str, Any] = {
        "schema": MATH_SCHEMA,
        "dense": {
            "domain": "[-2,0]-inclusive-step-2^-21",
            "records": len(rdw),
            "rtx_sha256": sha(rd),
            "mac_sha256": sha(md),
            "ex2_mismatched_words": sum(value != 0 for value in dense_delta),
            "mac_minus_rtx_ulp_histogram": histogram(dense_delta),
        },
        "random": {
            "domain": "1048576-deterministic-negative-finite-f32-values",
            "records": len(rrows),
            "rtx_sha256": sha(rr),
            "mac_sha256": sha(mr),
            "input_mismatched_words": input_mismatches,
            "ex2_mismatched_words": sum(value != 0 for value in ex2_delta),
            "ex2_mac_minus_rtx_ulp_histogram": histogram(ex2_delta),
            "low_confidence_half_mismatches": sum(value != 0 for value in low_delta),
            "low_confidence_half_delta_histogram": histogram(low_delta),
            "high_confidence_half_mismatches": sum(value != 0 for value in high_delta),
            "high_confidence_half_delta_histogram": histogram(high_delta),
        },
        "claims": {
            "metal_fast_ex2_general_fp32_exact": False,
            "final_half_general_random_exact": False,
            "selected_l52_fixture_domain_exact": True,
        },
        "expected_output_fed_to_execution": False,
        "archived_intermediates_consumed": False,
        "address_fields_recorded": False,
        "result": "pass",
    }
    require(document["dense"]["ex2_mismatched_words"] == 3_715_900,
            "dense counterexample drift")
    require(document["random"]["ex2_mismatched_words"] == 371_111,
            "random EX2 counterexample drift")
    require(document["random"]["low_confidence_half_mismatches"] == 45,
            "random low-half counterexample drift")
    require(document["random"]["high_confidence_half_mismatches"] == 3,
            "random high-half counterexample drift")
    document["receipt_root_sha256"] = canonical(document)
    return document


def build_final(rtx_root: Path, mac_root: Path,
                math_receipt: dict[str, Any]) -> dict[str, Any]:
    rtx, mac = build_backend(rtx_root, "rtx"), build_backend(mac_root, "mac")
    comparisons = []
    for run_id, fixture, poison in RUNS:
        rr = next(row for row in rtx["runs"] if row["run_id"] == run_id)
        mr = next(row for row in mac["runs"] if row["run_id"] == run_id)
        require(rr["input_root_sha256"] == mr["input_root_sha256"],
                f"{run_id}: input parity")
        artifacts = []
        for name, size, _ in ARTIFACTS:
            a = (rtx_root / run_id / name).read_bytes()
            b = (mac_root / run_id / name).read_bytes()
            mismatch = sum(x != y for x, y in zip(a, b))
            require(mismatch == 0 and len(a) == len(b) == size,
                    f"{run_id}/{name}: parity")
            artifacts.append({"artifact": name, "bytes": size,
                              "sha256": sha(a), "mismatched_bytes": 0})
        comparisons.append({"run_id": run_id, "fixture": fixture,
                            "poison_byte": poison, "artifacts": artifacts,
                            "mismatched_bytes": 0})
    document: dict[str, Any] = {
        "schema": FINAL_SCHEMA,
        "selected_launch_token": 52,
        "module_token": 6,
        "module_sha256": MODULE_SHA,
        "raw_live_parameter_sha256": PARAMETER_SHA,
        "input_generator": GENERATOR,
        "supported_domain": {
            "output_dimensions": [320, 180],
            "input_dimensions": [160, 96, 16],
            "input_crop": [0, 0, 3, 3],
            "depth_enabled": True,
            "optional_wrap_mask_enabled": False,
            "finite_half_features": True,
            "depth_fixtures": ["binary-depth", "ramp-depth"],
        },
        "comparisons": comparisons,
        "rtx_backend_root_sha256": rtx["receipt_root_sha256"],
        "mac_backend_root_sha256": mac["receipt_root_sha256"],
        "math_counterexample_root_sha256": math_receipt["receipt_root_sha256"],
        "sm120_rcp_rom_sha256": RCP_SHA,
        "fresh_process_count_per_backend": 4,
        "two_poison_output_invariance": True,
        "full_write_coverage_from_two_poison_invariance": True,
        "all_outputs_rtx_metal_exact": True,
        "input_immutability": True,
        "buffer_canary_integrity": True,
        "expected_output_fed_to_execution": False,
        "archived_intermediates_consumed": False,
        "address_fields_recorded": False,
        "claims": {
            "launch52_dependency_closed_fixture_domain_exact": True,
            "general_arbitrary_f32_ex2_exact": False,
            "connected_upstream_l51_to_l52_exact": False,
            "launch53_or_later_inferred": False,
            "complete_frame_exact": False,
        },
        "result": "pass",
    }
    document["receipt_root_sha256"] = canonical(document)
    return document


def validate_final(value: dict[str, Any]) -> None:
    reject_addresses(value)
    require(value.get("schema") == FINAL_SCHEMA, "final schema")
    require(rooted(value), "final root")
    require(value.get("all_outputs_rtx_metal_exact") is True, "final parity")
    require(value.get("two_poison_output_invariance") is True, "final poison")
    require(value.get("expected_output_fed_to_execution") is False,
            "final expected ingress")
    require(value.get("archived_intermediates_consumed") is False,
            "final archived ingress")
    require(value.get("result") == "pass", "final result")
    require(all(row.get("mismatched_bytes") == 0
                for row in value.get("comparisons", [])), "final comparisons")


def build_audit(value: dict[str, Any]) -> dict[str, Any]:
    validate_final(value)
    mutations: list[tuple[str, Callable[[dict[str, Any]], None]]] = [
        ("output_mismatch", lambda x: x["comparisons"][0].__setitem__("mismatched_bytes", 1)),
        ("module_drift", lambda x: x.__setitem__("module_sha256", "0" * 64)),
        ("poison_gate", lambda x: x.__setitem__("two_poison_output_invariance", False)),
        ("expected_ingress", lambda x: x.__setitem__("expected_output_fed_to_execution", True)),
        ("archived_ingress", lambda x: x.__setitem__("archived_intermediates_consumed", True)),
        ("address_leak", lambda x: x.__setitem__("gpu_virtual_address", "0x1")),
        ("result_drift", lambda x: x.__setitem__("result", "fail")),
    ]
    outcomes = []
    for name, mutate in mutations:
        changed = copy.deepcopy(value)
        mutate(changed)
        try:
            validate_final(changed)
        except VerificationError:
            outcomes.append({"mutation": name, "rejected": True})
            continue
        raise VerificationError(f"mutation accepted: {name}")
    document: dict[str, Any] = {
        "schema": "neuralstudio.fg.mid-l52-negative-audit.v1",
        "accepted_receipt_root_sha256": value["receipt_root_sha256"],
        "mutations": outcomes,
        "mutation_count": len(outcomes),
        "all_mutations_rejected": True,
        "address_fields_recorded": False,
        "result": "pass",
    }
    document["receipt_root_sha256"] = canonical(document)
    return document


def write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
    except FileExistsError as error:
        raise VerificationError(f"refusing to overwrite {path}") from error


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("backend")
    p.add_argument("--backend", choices=("rtx", "mac"), required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("math")
    for name in ("rtx-dense", "rtx-random", "mac-dense", "mac-random"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("final")
    p.add_argument("--rtx-root", type=Path, required=True)
    p.add_argument("--mac-root", type=Path, required=True)
    p.add_argument("--math-receipt", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("audit")
    p.add_argument("--accepted", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "backend":
            value = build_backend(args.root, args.backend)
        elif args.command == "math":
            value = build_math(args.rtx_dense, args.rtx_random,
                               args.mac_dense, args.mac_random)
        elif args.command == "final":
            math = load(args.math_receipt)
            require(math.get("schema") == MATH_SCHEMA and rooted(math),
                    "math receipt")
            value = build_final(args.rtx_root, args.mac_root, math)
        else:
            value = build_audit(load(args.accepted))
        write_exclusive(args.output, value)
        print(f"receipt_root_sha256={value['receipt_root_sha256']}")
        print("result=pass")
        return 0
    except (VerificationError, OSError) as error:
        print(f"verification_error={error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
