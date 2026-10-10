#!/usr/bin/env python3
"""Build address-free ABI/backend/final receipts for FG launch L55."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pathlib
import re
import sys
from typing import Any

MODULE_SHA = "66a9e6bda87ec8b63c5e10752e84d07435dc4392bbe42ec142e78b0d0b624e37"
RAW_PARAM_SHA = "f46662e2dadbf89180e182202dad123a674ae94b08f9289b8d00e80f8d9f66ff"
CANON_PARAM_SHA = "6e05e0cc5b2005a313d8b4cb03e1bc495028603d224a84e709a48351238766f4"
PTX_SHA = "5f1217226a5818846d655fee23cfbc05b08cd1d6d4c85d428b379b39b612f964"
DESCRIPTOR_ROOT = "d3c72b68dcec1d1bb307239c7a782a4c49e06c92b121e8e74a1f0a63ba509733"
CENSUS_ROOT = "324e579b133caf212ad9db44908a73ae6baa90ffbb4978696b51ae99cb114c26"
ROM_SHA = "7cf4f8cece6b18b07605edc9cb601a153346a79b10bf3b93c7ec058d57d29750"
GENERATOR = "splitmix64-fg-mid-l55-independent-resources-v1"
RUNS = {
    "poison-a5-process-a": 165,
    "poison-a5-process-b": 165,
    "poison-5a-process-a": 90,
    "poison-5a-process-b": 90,
}
OUTPUTS = {
    "surface_a": ("surface-a.rgba16f", 1_843_200),
    "surface_b": ("surface-b.rgba16f", 1_843_200),
    "surface_mask": ("surface-mask.r8", 230_400),
    "linear_a": ("linear-a.rgba16f", 1_884_160),
    "linear_b": ("linear-b.rgba16f", 1_884_160),
    "linear_mask": ("linear-mask.r16f", 471_040),
}
ROOT_KEY = "receipt_root_sha256"


class VerificationError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise VerificationError(message)


def load(path: pathlib.Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except Exception as error:
        raise VerificationError(f"unable to load {path}: {error}") from error
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


def add_root(value: dict[str, Any]) -> dict[str, Any]:
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


def write(path: pathlib.Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")


def verify_artifact(root: pathlib.Path, name: str, expected_sha: str,
                    expected_bytes: int, required: bool) -> bool:
    path = root / name
    if not path.exists():
        require(not required, f"required artifact missing: {path}")
        return False
    require(path.stat().st_size == expected_bytes, f"artifact size drift: {path}")
    require(sha_file(path) == expected_sha, f"artifact hash drift: {path}")
    return True


def verify_backend(backend: str, root: pathlib.Path) -> dict[str, Any]:
    expected_backend = "rtx-d3d12-nvapi" if backend == "rtx" else "mac-metal"
    receipts: dict[str, dict[str, Any]] = {}
    present: dict[str, list[str]] = {}
    pids: list[int] = []
    for run_id, poison in RUNS.items():
        run_root = root / run_id
        receipt = load(run_root / "run-receipt.json")
        receipts[run_id] = receipt
        require(receipt.get("schema") == "neuralstudio.fg.mid-l55-run.v1",
                f"{run_id} schema drift")
        require(receipt.get("backend") == expected_backend,
                f"{run_id} backend drift")
        require(receipt.get("run_id") == run_id and
                receipt.get("poison_byte") == poison,
                f"{run_id} identity/poison drift")
        module_key = "module_sha256" if backend == "rtx" else "source_module_sha256"
        require(receipt.get(module_key) == MODULE_SHA,
                f"{run_id} module drift")
        require(receipt.get("raw_parameter_sha256") == RAW_PARAM_SHA and
                receipt.get("address_free_parameter_sha256") == CANON_PARAM_SHA,
                f"{run_id} parameter provenance drift")
        require(receipt.get("input_generator") == GENERATOR,
                f"{run_id} generator drift")
        require(receipt.get("combined_output_bytes") == 8_156_160,
                f"{run_id} output byte count drift")
        require(receipt.get("input_mismatched_bytes") == 0 and
                receipt.get("canary_mismatched_bytes") == 0 and
                receipt.get("external_expected_tensors_consumed") is False and
                receipt.get("archived_intermediates_consumed") is False and
                receipt.get("address_fields_recorded") is False and
                receipt.get("result") == "pass",
                f"{run_id} correctness/provenance gate")
        require(set(receipt.get("output_hashes", {})) == set(OUTPUTS),
                f"{run_id} output set drift")
        pid = receipt.get("process_id")
        require(isinstance(pid, int) and pid > 0, f"{run_id} PID missing")
        pids.append(pid)
        names: list[str] = []
        for key, (name, byte_count) in OUTPUTS.items():
            if verify_artifact(run_root, name, receipt["output_hashes"][key],
                               byte_count,
                               backend == "mac" or run_id ==
                               "poison-a5-process-a"):
                names.append(name)
        present[run_id] = names
        if backend == "mac":
            architecture = receipt.get("architecture_rom", {})
            require(architecture.get("sha256") == ROM_SHA and
                    architecture.get("model_values") == 0 and
                    architecture.get("captured_intermediates") == 0,
                    f"{run_id} architecture ROM provenance")
    require(len(set(pids)) == 4, "fresh-process gate failed")
    require({entry["poison_byte"] for entry in receipts.values()} == {90, 165},
            "dual-poison gate failed")
    require(len({entry["combined_output_sha256"]
                 for entry in receipts.values()}) == 1,
            "combined two-poison output invariance failed")
    for key in OUTPUTS:
        require(len({entry["output_hashes"][key]
                     for entry in receipts.values()}) == 1,
                f"{key} two-poison output invariance failed")
    require(len({canonical(entry["input_hashes"])
                 for entry in receipts.values()}) == 1,
            "generated input drift")

    repository = pathlib.Path(__file__).resolve().parent.parent
    source_files: list[dict[str, Any]] = []
    build_artifacts: list[dict[str, Any]] = []
    remote_job = None
    if backend == "rtx":
        rig = load(root / "rig-receipt.json")
        require(rig.get("result") == "pass" and
                rig.get("fresh_process_count") == 4 and
                rig.get("two_poison_output_invariance") is True and
                rig.get("full_write_coverage_from_two_poison_invariance") is True and
                rig.get("module", {}).get("sha256") == MODULE_SHA,
                "rig receipt gate")
        remote_job = rig.get("remote_job_retained")
        rig_sources = {item["file"]: item["sha256"] for item in rig["sources"]}
        paths = {
            "rtx_oracle.cpp": repository / "src/fg_mid_l55/rtx_oracle.cpp",
            "fg_mid_l55_contract.h": repository / "src/fg_mid_l55/fg_mid_l55_contract.h",
            "fg_mid_l55_inputs.h": repository / "src/fg_mid_l55/fg_mid_l55_inputs.h",
            "build-fg-mid-l55-rtx.ps1": repository / "scripts/build-fg-mid-l55-rtx.ps1",
        }
        for name, path in paths.items():
            digest = sha_file(path)
            require(rig_sources.get(name) == digest,
                    f"rig/local source drift: {name}")
            source_files.append({"file": name, "sha256": digest})
        build_artifacts.append({"file": "rig-receipt.json",
                                "sha256": sha_file(root / "rig-receipt.json")})
    else:
        paths = [
            repository / "src/fg_mid_l55/fg_mid_l55.metal",
            repository / "src/fg_mid_l55/macos_probe.mm",
            repository / "src/fg_mid_l55/fg_mid_l55_contract.h",
            repository / "src/fg_mid_l55/fg_mid_l55_inputs.h",
            repository / "scripts/build-macos-fg-mid-l55.sh",
        ]
        for path in paths:
            source_files.append({"file": path.name,
                                 "sha256": sha_file(path)})
        for name in ("fg_mid_l55.metallib", "fg-mid-l55-mac",
                     "sm120-rcp-correction-rom.bin"):
            path = root / "build" / name
            require(path.is_file(), f"Mac build artifact missing: {path}")
            build_artifacts.append({"file": name, "bytes": path.stat().st_size,
                                    "sha256": sha_file(path)})
        require(build_artifacts[-1]["sha256"] == ROM_SHA,
                "Mac ROM build artifact drift")

    first = receipts["poison-a5-process-a"]
    return add_root({
        "schema": "neuralstudio.fg.mid-l55-backend.v1",
        "backend": backend,
        "fresh_process_count": 4,
        "poison_bytes": [90, 165],
        "module_sha256": MODULE_SHA,
        "raw_parameter_sha256": RAW_PARAM_SHA,
        "address_free_parameter_sha256": CANON_PARAM_SHA,
        "input_generator": GENERATOR,
        "input_hashes": first["input_hashes"],
        "output_hashes": first["output_hashes"],
        "combined_output_bytes": 8_156_160,
        "combined_output_sha256": first["combined_output_sha256"],
        "two_poison_output_invariance": True,
        "full_write_coverage_from_two_poison_invariance": True,
        "input_immutable": True,
        "canaries_intact": True,
        "expected_or_archived_tensor_ingress": False,
        "source_files": source_files,
        "build_artifacts": build_artifacts,
        "present_artifacts": present,
        "remote_job_retained": remote_job,
        "result": "pass",
    })


def build_abi(args: argparse.Namespace) -> dict[str, Any]:
    census = load(args.census)
    descriptor = load(args.descriptor)
    ptx = args.ptx.read_text()
    require(census.get("census_root_sha256") == CENSUS_ROOT,
            "selected census root drift")
    launch = next(item for item in census["ordered_launches"]
                  if item["launch_token"] == "launch-000055")
    require(launch.get("module_token") == "module-000003" and
            launch.get("module_sha256") == MODULE_SHA and
            launch.get("parameter_bytes") == 240 and
            launch.get("grid") == [40, 23, 1] and
            launch.get("block") == [256, 1, 1],
            "selected launch identity drift")
    require(sha_file(args.ptx) == PTX_SHA and
            ".target sm_120" in ptx and ".target sm_89" in ptx,
            "module3 PTX identity drift")
    require(descriptor.get("receipt_root_sha256") == DESCRIPTOR_ROOT,
            "descriptor root drift")
    resources = {item["nvapi_resource_token"]: item
                 for item in descriptor["resource_object_requests"]}
    expected = {
        "resource-000030": ("merged_texture_sampler", "DXGI_FORMAT_R16G16B16A16_FLOAT", 640, 360),
        "resource-000006": ("merged_texture_sampler", "DXGI_FORMAT_R16G16B16A16_FLOAT", 640, 360),
        "resource-000050": ("merged_texture_sampler", "DXGI_FORMAT_R16G16_FLOAT", 320, 180),
        "resource-000051": ("merged_texture_sampler", "DXGI_FORMAT_R16G16_FLOAT", 320, 180),
        "resource-000052": ("independent_surface", "DXGI_FORMAT_R16G16B16A16_FLOAT", 640, 360),
        "resource-000053": ("independent_surface", "DXGI_FORMAT_R16G16B16A16_FLOAT", 640, 360),
        "resource-000054": ("independent_surface", "DXGI_FORMAT_R8_UNORM", 640, 360),
    }
    for token, (kind, fmt, width, height) in expected.items():
        item = resources[token]
        require(item["request_type"] == kind and
                item["resource"]["format_name"] == fmt and
                item["resource"]["width"] == width and
                item["resource"]["height"] == height,
                f"descriptor binding drift: {token}")
    require("setp.ne.s32 %p30, %r12, 1" in ptx and
            "sust.p.2d.v4.b32.zero" in ptx and
            "cvt.rn.f16x2.f32" in ptx,
            "selected mode/output PTX drift")
    return add_root({
        "schema": "neuralstudio.fg.mid-l55-abi.v1",
        "selected_launch_token": 55,
        "descriptor_trace_launch_alias": 53,
        "module_token": 3,
        "module_bytes": 43880,
        "module_sha256": MODULE_SHA,
        "ptx_sha256": PTX_SHA,
        "ptx_architectures": ["sm_89", "sm_120"],
        "packed_parameter_bytes": 240,
        "raw_parameter_sha256": RAW_PARAM_SHA,
        "address_free_parameter_sha256": CANON_PARAM_SHA,
        "grid": [40, 23, 1],
        "block": [256, 1, 1],
        "bindings": [
            {"offset": 0, "meaning": "resource-000030 RGBA16F source A point-border"},
            {"offset": 8, "meaning": "resource-000006 RGBA16F source B point-border"},
            {"offset": 16, "meaning": "resource-000050 RG16F motion point-border"},
            {"offset": 24, "meaning": "allocation-000020 candidate A packed half2"},
            {"offset": 32, "meaning": "allocation-000021 candidate B packed half2"},
            {"offset": 40, "meaning": "resource-000051 RG16F blend point-border"},
            {"offset": 48, "meaning": "resource-000052 RGBA16F surface A"},
            {"offset": 56, "meaning": "resource-000053 RGBA16F surface B"},
            {"offset": 64, "meaning": "resource-000054 R8_UNORM mask surface"},
            {"offset": 72, "meaning": "allocation-000038 padded half4 output A"},
            {"offset": 80, "meaning": "allocation-000039 padded half4 output B"},
            {"offset": 88, "meaning": "allocation-000041 padded half mask output"},
        ],
        "selected_controls": {
            "physical_dimensions": [640, 368],
            "accumulation_mode": 1,
            "candidate_dimensions": [320, 180],
            "secondary_dimensions": [320, 180],
            "active_dimensions": [640, 360],
            "optional_ui_enabled": False,
            "optional_texture_handles_zero": True,
        },
        "selected_numerics": {
            "division": "SM120 div.approx.ftz via accepted RCP architecture ROM",
            "normalized_point_coordinate": "floor(floor(coord*4096)/4096*dimension)",
            "normalized_point_domain": "independently generated selected L55 half-offset domain",
            "surface_conversion": ["RGBA16_FLOAT", "R8_UNORM boolean"],
        },
        "source_roots": {
            "selected_module_census": CENSUS_ROOT,
            "resource_descriptor_provenance": DESCRIPTOR_ROOT,
        },
        "addresses_recorded": False,
        "result": "pass",
    })


def compare_outputs(rtx_root: pathlib.Path, mac_root: pathlib.Path) -> dict[str, int]:
    result: dict[str, int] = {}
    for key, (name, _) in OUTPUTS.items():
        left = (rtx_root / "poison-a5-process-a" / name).read_bytes()
        right = (mac_root / "poison-a5-process-a" / name).read_bytes()
        require(len(left) == len(right), f"{key} output size mismatch")
        count = sum(a != b for a, b in zip(left, right))
        require(count == 0, f"{key} RTX/Mac mismatched bytes: {count}")
        result[key] = count
    return result


def build_final(args: argparse.Namespace) -> dict[str, Any]:
    rtx = verify_backend("rtx", args.rtx_root)
    mac = verify_backend("mac", args.mac_root)
    abi = load(args.abi)
    verify_root(abi, "ABI")
    require(abi.get("module_sha256") == MODULE_SHA, "ABI module drift")
    require(rtx["input_hashes"] == mac["input_hashes"],
            "RTX/Mac generated input mismatch")
    require(rtx["output_hashes"] == mac["output_hashes"] and
            rtx["combined_output_sha256"] == mac["combined_output_sha256"],
            "RTX/Mac output receipt mismatch")
    mismatches = compare_outputs(args.rtx_root, args.mac_root)
    return add_root({
        "schema": "neuralstudio.fg.mid-l55-accepted.v1",
        "abi_receipt_sha256": sha_file(args.abi),
        "abi_receipt_root_sha256": abi[ROOT_KEY],
        "rtx_backend_receipt": rtx,
        "mac_backend_receipt": mac,
        "output_hashes": rtx["output_hashes"],
        "combined_output_bytes": 8_156_160,
        "combined_output_sha256": rtx["combined_output_sha256"],
        "rtx_mac_mismatched_bytes": mismatches,
        "failure_counterexamples": {
            "poison_pattern_is_not_write_coverage": 92,
            "native_metal_sampler_mismatched_bytes": 1112927,
            "boundary_down_heuristic_mismatched_bytes": 451328,
            "fixed12_point_sampler_mismatched_bytes": 0,
        },
        "claims": {
            "selected_l55_mode1_exact_on_metal": True,
            "all_six_outputs_exact": True,
            "four_fresh_processes_per_backend": True,
            "dual_poison_full_write_coverage": True,
            "input_and_rom_immutable": True,
            "canaries_intact": True,
            "expected_or_archived_tensor_ingress": False,
            "optional_ui_branch": False,
            "multi_sample_modes": False,
            "connected_graph": False,
            "full_generated_frame": False,
            "netal_integration": False,
            "performance_claim": False,
        },
        "remote_job_retained": rtx["remote_job_retained"],
        "result": "pass",
    })


def verify_receipt(args: argparse.Namespace) -> dict[str, Any]:
    accepted = load(args.accepted)
    abi = load(args.abi)
    verify_root(accepted, "accepted")
    verify_root(abi, "ABI")
    require(accepted.get("abi_receipt_sha256") == sha_file(args.abi) and
            accepted.get("abi_receipt_root_sha256") == abi.get(ROOT_KEY),
            "ABI receipt binding mismatch")
    require(abi.get("module_sha256") == MODULE_SHA and
            abi.get("selected_controls", {}).get("accumulation_mode") == 1,
            "ABI identity/control drift")
    require(accepted.get("result") == "pass" and abi.get("result") == "pass",
            "accepted/ABI result drift")
    require(len(abi.get("bindings", [])) == 12 and
            "resource-000030" in abi["bindings"][0].get("meaning", "") and
            "resource-000054" in abi["bindings"][8].get("meaning", ""),
            "ABI resource binding drift")
    require(abi.get("selected_numerics", {}).get("normalized_point_coordinate") ==
            "floor(floor(coord*4096)/4096*dimension)",
            "point-sampler primitive drift")
    backends = [accepted.get("rtx_backend_receipt"),
                accepted.get("mac_backend_receipt")]
    require(all(isinstance(value, dict) for value in backends),
            "backend receipt missing")
    for value in backends:
        verify_root(value, "backend")
        require(value.get("result") == "pass" and
                value.get("fresh_process_count") == 4 and
                value.get("full_write_coverage_from_two_poison_invariance") is True and
                value.get("expected_or_archived_tensor_ingress") is False,
                "backend receipt gate drift")
    require(backends[0]["output_hashes"] == backends[1]["output_hashes"] ==
            accepted.get("output_hashes"), "backend output binding drift")
    require(backends[0]["input_hashes"] == backends[1]["input_hashes"],
            "backend input binding drift")
    require(backends[0]["combined_output_sha256"] ==
            backends[1]["combined_output_sha256"] ==
            accepted.get("combined_output_sha256"),
            "combined output binding drift")
    require(all(value == 0 for value in
                accepted.get("rtx_mac_mismatched_bytes", {}).values()),
            "accepted output parity drift")
    claims = accepted.get("claims", {})
    require(claims.get("selected_l55_mode1_exact_on_metal") is True and
            claims.get("expected_or_archived_tensor_ingress") is False and
            claims.get("optional_ui_branch") is False and
            claims.get("multi_sample_modes") is False and
            claims.get("connected_graph") is False and
            claims.get("full_generated_frame") is False and
            claims.get("netal_integration") is False and
            claims.get("performance_claim") is False,
            "accepted claim boundary drift")
    require(accepted.get("failure_counterexamples") == {
        "poison_pattern_is_not_write_coverage": 92,
        "native_metal_sampler_mismatched_bytes": 1112927,
        "boundary_down_heuristic_mismatched_bytes": 451328,
        "fixed12_point_sampler_mismatched_bytes": 0,
    }, "failure counterexample drift")
    require(accepted.get("remote_job_retained") ==
            "D:\\CodexLab\\fg-mid-l55-20260904T120633Z",
            "remote job identity drift")
    return {ROOT_KEY: accepted[ROOT_KEY], "result": "pass"}


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    abi = commands.add_parser("abi")
    abi.add_argument("--census", type=pathlib.Path, required=True)
    abi.add_argument("--descriptor", type=pathlib.Path, required=True)
    abi.add_argument("--ptx", type=pathlib.Path, required=True)
    abi.add_argument("--output", type=pathlib.Path, required=True)
    backend = commands.add_parser("backend")
    backend.add_argument("--backend", choices=("rtx", "mac"), required=True)
    backend.add_argument("--root", type=pathlib.Path, required=True)
    backend.add_argument("--output", type=pathlib.Path, required=True)
    final = commands.add_parser("final")
    final.add_argument("--rtx-root", type=pathlib.Path, required=True)
    final.add_argument("--mac-root", type=pathlib.Path, required=True)
    final.add_argument("--abi", type=pathlib.Path, required=True)
    final.add_argument("--output", type=pathlib.Path, required=True)
    receipt = commands.add_parser("receipt")
    receipt.add_argument("--accepted", type=pathlib.Path, required=True)
    receipt.add_argument("--abi", type=pathlib.Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "abi":
            result = build_abi(args); write(args.output, result)
        elif args.command == "backend":
            result = verify_backend(args.backend, args.root); write(args.output, result)
        elif args.command == "final":
            result = build_final(args); write(args.output, result)
        else:
            result = verify_receipt(args)
        print("verification_result=pass")
        print(f"receipt_root_sha256={result[ROOT_KEY]}")
        return 0
    except VerificationError as error:
        print(f"verification_error={error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
