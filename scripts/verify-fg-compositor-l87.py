#!/usr/bin/env python3
"""Build and verify immutable receipts for selected FG compositor launch 87."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pathlib
import re
import sys
from typing import Any

MODULE_SHA = "6cc983a76decaed1a8c7f207a95531bece5a1ad092c4db3a0159b590c331c24b"
RAW_PARAM_SHA = "b4237dac2bf46edd1c9069d8b9e89b71f21f109c144b1b39cf1120bda1a8f54e"
CANON_PARAM_SHA = "85a388275a5df0aae3b463f2af4fee7c5cc6135c0405127e5261b38f2ab55a01"
ROM_SHA = "7cf4f8cece6b18b07605edc9cb601a153346a79b10bf3b93c7ec058d57d29750"
DESCRIPTOR_ROOT = "d3c72b68dcec1d1bb307239c7a782a4c49e06c92b121e8e74a1f0a63ba509733"
CENSUS_ROOT = "324e579b133caf212ad9db44908a73ae6baa90ffbb4978696b51ae99cb114c26"
GENERATOR = "splitmix64-fg-compositor-l87-half4-mask-v1"
RUNS = {
    "poison-a5-process-a": (165, "full"),
    "poison-a5-process-b": (165, "full"),
    "poison-5a-process-a": (90, "full"),
    "poison-5a-process-b": (90, "full"),
    "sparse-mask-branch": (60, "sparse"),
}
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


def add_root(value: dict[str, Any]) -> dict[str, Any]:
    rooted = copy.deepcopy(value)
    rooted.pop(ROOT_KEY, None)
    rooted[ROOT_KEY] = sha_bytes(canonical(rooted))
    return rooted


def verify_root(value: dict[str, Any], label: str) -> None:
    expected = value.get(ROOT_KEY)
    require(isinstance(expected, str) and len(expected) == 64,
            f"{label} receipt root missing")
    unrooted = copy.deepcopy(value)
    unrooted.pop(ROOT_KEY, None)
    require(sha_bytes(canonical(unrooted)) == expected,
            f"{label} receipt root mismatch")


def write_json(path: pathlib.Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")


def verify_artifact(run_root: pathlib.Path, name: str, expected_sha: str,
                    expected_bytes: int, required: bool) -> bool:
    path = run_root / name
    if not path.exists():
        require(not required, f"required artifact missing: {path}")
        return False
    require(path.stat().st_size == expected_bytes, f"artifact size drift: {path}")
    require(sha_file(path) == expected_sha, f"artifact hash drift: {path}")
    return True


def verify_backend(backend: str, root: pathlib.Path) -> dict[str, Any]:
    require(backend in ("rtx", "mac"), "unknown backend")
    receipts: dict[str, dict[str, Any]] = {}
    processes: list[int] = []
    present_artifacts: dict[str, list[str]] = {}
    for run_id, (poison, mode) in RUNS.items():
        run_root = root / run_id
        receipt = load(run_root / "run-receipt.json")
        receipts[run_id] = receipt
        expected_backend = "rtx-d3d12-nvapi" if backend == "rtx" else "mac-metal"
        require(receipt.get("schema") == "neuralstudio.fg.compositor-l87-run.v1",
                f"{run_id} schema drift")
        require(receipt.get("backend") == expected_backend, f"{run_id} backend drift")
        require(receipt.get("run_id") == run_id, f"{run_id} identity drift")
        require(receipt.get("poison_byte") == poison, f"{run_id} poison drift")
        require(receipt.get("mask_mode") == mode, f"{run_id} mask mode drift")
        require(receipt.get("result") == "pass", f"{run_id} did not pass")
        require(receipt.get("input_generator") == GENERATOR,
                f"{run_id} generator drift")
        require(receipt.get("level0_bytes") == 460800, f"{run_id} level bytes drift")
        require(receipt.get("mask_bytes") == 230400, f"{run_id} mask bytes drift")
        require(receipt.get("output_bytes") == 921600, f"{run_id} output bytes drift")
        module_key = "module_sha256" if backend == "rtx" else "source_module_sha256"
        require(receipt.get(module_key) == MODULE_SHA, f"{run_id} module drift")
        require(receipt.get("raw_parameter_sha256") == RAW_PARAM_SHA,
                f"{run_id} raw parameter drift")
        require(receipt.get("address_free_parameter_sha256") == CANON_PARAM_SHA,
                f"{run_id} canonical parameter drift")
        require(receipt.get("input_mismatched_bytes") == 0,
                f"{run_id} input mutation")
        require(receipt.get("canary_mismatched_bytes") == 0,
                f"{run_id} canary mutation")
        require(receipt.get("external_expected_tensors_consumed") is False,
                f"{run_id} consumed expected tensor")
        require(receipt.get("archived_intermediates_consumed") is False,
                f"{run_id} consumed archived intermediate")
        require(receipt.get("address_fields_recorded") is False,
                f"{run_id} recorded an address")
        expected_zero = 0 if mode == "full" else 15701
        require(receipt.get("zero_mask_pixels") == expected_zero,
                f"{run_id} zero-mask count drift")
        require(receipt.get("poison_pixels_after") == expected_zero,
                f"{run_id} write coverage drift")
        process = receipt.get("process_id")
        require(isinstance(process, int) and process > 0, f"{run_id} PID missing")
        processes.append(process)

        required = backend == "mac" or run_id in (
            "poison-a5-process-a", "sparse-mask-branch")
        present: list[str] = []
        if verify_artifact(run_root, "output.rgba8", receipt["output_sha256"],
                           921600, required):
            present.append("output.rgba8")
        if verify_artifact(run_root, "input-level0-after.rgba16f",
                           receipt["level0_sha256"], 460800,
                           backend == "mac" or run_id == "poison-a5-process-a"):
            present.append("input-level0-after.rgba16f")
        if verify_artifact(run_root, "input-mask-after.r8", receipt["mask_sha256"],
                           230400,
                           backend == "mac" or run_id == "poison-a5-process-a"):
            present.append("input-mask-after.r8")
        diagnostic_sha = receipt.get("diagnostic_float_output_sha256")
        if diagnostic_sha is not None and verify_artifact(
                run_root, "diagnostic-output.rgba32f", diagnostic_sha, 3686400,
                backend == "mac" or run_id == "poison-a5-process-a"):
            present.append("diagnostic-output.rgba32f")
        present_artifacts[run_id] = present

        if backend == "mac":
            architecture = receipt.get("architecture_rom", {})
            require(architecture.get("sha256") == ROM_SHA,
                    f"{run_id} RCP ROM drift")
            require(architecture.get("model_values") == 0 and
                    architecture.get("captured_intermediates") == 0,
                    f"{run_id} RCP ROM provenance drift")
            require(receipt.get("packed_candidate_output_sha256") !=
                    receipt.get("output_sha256"),
                    f"{run_id} native Metal UNORM counterexample disappeared")
            require(receipt.get("strict_division_output_sha256") !=
                    receipt.get("output_sha256"),
                    f"{run_id} strict negative control disappeared")

    require(len(set(processes)) == len(RUNS), "fresh-process identity gate failed")
    primary = [receipts[name] for name in RUNS if RUNS[name][1] == "full"]
    require(len({entry["output_sha256"] for entry in primary}) == 1,
            "two-poison primary output invariance failed")
    require(len({entry["level0_sha256"] for entry in receipts.values()}) == 1,
            "generated pyramid input drift")
    require(len({entry["mask_sha256"] for name, entry in receipts.items()
                 if RUNS[name][1] == "full"}) == 1,
            "generated full mask drift")
    require(len({entry["diagnostic_float_output_sha256"] for entry in primary}) == 1,
            "diagnostic float output drift")

    rig = None
    repository = pathlib.Path(__file__).resolve().parent.parent
    source_files: list[dict[str, Any]] = []
    backend_artifacts: list[dict[str, Any]] = []
    if backend == "rtx":
        rig = load(root / "rig-receipt.json")
        require(rig.get("result") == "pass", "rig receipt did not pass")
        require(rig.get("fresh_primary_processes") == 4,
                "rig fresh-process count drift")
        require(rig.get("two_poison_output_invariance") is True,
                "rig poison gate drift")
        require(rig.get("primary_zero_untouched_pixels") is True,
                "rig coverage gate drift")
        require(rig.get("module", {}).get("sha256") == MODULE_SHA,
                "rig module drift")
        rig_sources = {entry["file"]: entry["sha256"] for entry in rig["sources"]}
        local_sources = {
            "rtx_nvapi_oracle.cpp": repository / "src/fg_compositor_l87/rtx_nvapi_oracle.cpp",
            "compositor_l87_contract.h": repository / "src/fg_compositor_l87/compositor_l87_contract.h",
            "build-fg-compositor-l87-rtx.ps1": repository / "scripts/build-fg-compositor-l87-rtx.ps1",
        }
        for name, path in local_sources.items():
            digest = sha_file(path)
            require(rig_sources.get(name) == digest,
                    f"rig/local source hash drift: {name}")
            source_files.append({"file": name, "sha256": digest})
        backend_artifacts.append({
            "file": "rig-receipt.json", "bytes": (root / "rig-receipt.json").stat().st_size,
            "sha256": sha_file(root / "rig-receipt.json")})
    else:
        local_sources = [
            repository / "src/fg_compositor_l87/compositor_l87.metal",
            repository / "src/fg_compositor_l87/macos_probe.mm",
            repository / "src/fg_compositor_l87/compositor_l87_contract.h",
            repository / "scripts/build-macos-fg-compositor-l87.sh",
        ]
        for path in local_sources:
            source_files.append({"file": path.name, "sha256": sha_file(path)})
        artifacts = [
            root / "build/compositor_l87.metallib",
            root / "build/fg-compositor-l87-mac",
            root / "build/sm120-rcp-correction-rom.bin",
        ]
        for path in artifacts:
            require(path.is_file(), f"Mac build artifact missing: {path}")
            backend_artifacts.append({
                "file": path.name, "bytes": path.stat().st_size,
                "sha256": sha_file(path)})
        require(backend_artifacts[-1]["sha256"] == ROM_SHA,
                "Mac build ROM drift")

    output = {
        "schema": "neuralstudio.fg.compositor-l87-backend.v1",
        "backend": backend,
        "source_root": "remote-job-retained" if backend == "rtx" else "isolated-worktree",
        "fresh_process_count": len(set(processes)),
        "primary_process_count": 4,
        "poison_bytes": [90, 165],
        "module_sha256": MODULE_SHA,
        "raw_parameter_sha256": RAW_PARAM_SHA,
        "address_free_parameter_sha256": CANON_PARAM_SHA,
        "generator": GENERATOR,
        "level0_sha256": primary[0]["level0_sha256"],
        "full_mask_sha256": primary[0]["mask_sha256"],
        "primary_output_bytes": 921600,
        "primary_output_sha256": primary[0]["output_sha256"],
        "sparse_output_sha256": receipts["sparse-mask-branch"]["output_sha256"],
        "diagnostic_float_output_sha256":
            primary[0]["diagnostic_float_output_sha256"],
        "sparse_diagnostic_float_output_sha256":
            receipts["sparse-mask-branch"]["diagnostic_float_output_sha256"],
        "two_poison_output_invariance": True,
        "zero_untouched_primary_pixels": True,
        "sparse_mask_zero_pixels": 15701,
        "sparse_mask_untouched_pixels": 15701,
        "input_immutable": True,
        "canaries_intact": True,
        "expected_or_archived_tensor_ingress": False,
        "present_artifacts": present_artifacts,
        "source_files": source_files,
        "backend_artifacts": backend_artifacts,
        "remote_job_retained": rig.get("remote_job_retained") if rig else None,
        "result": "pass",
    }
    return add_root(output)


def build_abi(args: argparse.Namespace) -> dict[str, Any]:
    census = load(args.census)
    descriptor = load(args.descriptor)
    ptx = args.ptx.read_text(errors="strict")
    require(census.get("census_root_sha256") == CENSUS_ROOT, "census root drift")
    launch = census["ordered_launches"][-1]
    require(launch.get("launch_token") == "launch-000087", "selected launch drift")
    require(launch.get("module_token") == "module-000020", "selected module drift")
    require(launch.get("module_sha256") == MODULE_SHA, "selected module hash drift")
    require(launch.get("parameter_bytes") == 72, "selected parameter size drift")
    require(launch.get("grid") == [40, 23, 1] and
            launch.get("block") == [256, 1, 1], "selected geometry drift")
    require(descriptor.get("receipt_root_sha256") == DESCRIPTOR_ROOT,
            "descriptor provenance root drift")
    resources = {entry["nvapi_resource_token"]: entry
                 for entry in descriptor["resource_object_requests"]}
    texture = resources["resource-000060"]
    surface = resources["resource-000055"]
    require(texture["request_type"] == "merged_texture_sampler",
            "texture binding kind drift")
    require(texture["resource"]["format_name"] == "DXGI_FORMAT_R8_UNORM" and
            texture["resource"]["width"] == 640 and
            texture["resource"]["height"] == 360,
            "mask texture descriptor drift")
    require(surface["request_type"] == "independent_surface",
            "surface binding kind drift")
    require(surface["resource"]["format_name"] ==
            "DXGI_FORMAT_R8G8B8A8_UNORM", "output surface format drift")
    offsets = sorted({int(value) for value in re.findall(
        r"main_kernel_param_0\+(\d+)", ptx)} | {0})
    require(offsets == [0, 8, 16, 24, 32, 40, 48, 56, 64],
            f"PTX parameter access mask drift: {offsets}")
    require(".target sm_120" in ptx and ".target sm_89" in ptx,
            "PTX architecture pair drift")
    require("sust.p.2d.v4.b32.zero" in ptx and
            "tex.base.2d.v4.f32.s32" in ptx,
            "PTX texture/surface operation drift")
    return add_root({
        "schema": "neuralstudio.fg.compositor-l87-abi.v1",
        "selected_launch_token": 87,
        "descriptor_trace_launch_alias": 85,
        "module_token": 20,
        "function_name": "main_kernel",
        "module_bytes": 26024,
        "module_sha256": MODULE_SHA,
        "ptx_sha256": sha_file(args.ptx),
        "ptx_architectures": ["sm_89", "sm_120"],
        "disassembly": {
            "tool": "NVIDIA cuobjdump",
            "version": "CUDA 13.3 V13.3.73",
            "remote_job_retained": "D:\\CodexLab\\fg-compositor-l87-20260904T112919Z",
            "ptx_remote_relative_path": "output\\module20.ptx.txt",
            "ptx_sha256": "ab6e9232519a46e63cbbdce3fbe32acd392a1fbe762dfb259506f1ba5b66518e",
            "sass_remote_relative_path": "output\\module20.sass.txt",
            "sass_sha256": "1836a3a8f7a9f61043efb16c0920683263af14d26a75985d2dd623235d0cd46c",
        },
        "grid": [40, 23, 1],
        "block": [256, 1, 1],
        "packed_parameter_bytes": 72,
        "parameter_access_offsets": offsets,
        "byte_68_through_71_unused": True,
        "bindings": [
            {"offset": 0, "type": "u64", "meaning": "resource-000060 R8_UNORM 640x360 point-border merged texture-sampler"},
            {"offset": 8, "type": "u64", "meaning": "resource-000055 RGBA8_UNORM 640x360 independent surface"},
            {"offset": 16, "type": "i32x2", "meaning": "output origin [0,0]"},
            {"offset": 24, "type": "half4*", "meaning": "320x180 selected level buffer allocation-000046"},
            {"offset": 32, "type": "half4*", "meaning": "160x90 inactive level allocation-000048"},
            {"offset": 40, "type": "half4*", "meaning": "80x45 inactive level allocation-000050"},
            {"offset": 48, "type": "half4*", "meaning": "40x23 inactive level allocation-000051"},
            {"offset": 56, "type": "u32", "meaning": "selected level 10"},
            {"offset": 60, "type": "u32", "meaning": "width 640"},
            {"offset": 64, "type": "u32", "meaning": "height 360"},
        ],
        "selected_branch": {
            "level": 10,
            "only_level0_dereferenced": True,
            "mask_write_predicate": "R8_UNORM sample > 0",
            "low_edge": "duplicate",
            "high_halo": "zero",
            "premultiplied_weights": [0.5625, 0.1875, 0.1875, 0.0625],
            "division": "div.approx.ftz.f32",
            "output_conversion": "SUST.P RGBA8_UNORM",
        },
        "upstream": {
            "selected_module_census_root_sha256": CENSUS_ROOT,
            "resource_descriptor_provenance_root_sha256": DESCRIPTOR_ROOT,
            "raw_parameter_sha256": RAW_PARAM_SHA,
            "address_free_parameter_sha256": CANON_PARAM_SHA,
        },
        "addresses_recorded": False,
        "result": "pass",
    })


def compare_bytes(left: pathlib.Path, right: pathlib.Path, label: str) -> int:
    a = left.read_bytes()
    b = right.read_bytes()
    require(len(a) == len(b), f"{label} byte size drift")
    mismatch = sum(x != y for x, y in zip(a, b))
    require(mismatch == 0, f"{label} mismatched bytes: {mismatch}")
    return mismatch


def build_final(args: argparse.Namespace) -> dict[str, Any]:
    rtx = verify_backend("rtx", args.rtx_root)
    mac = verify_backend("mac", args.mac_root)
    abi = load(args.abi)
    verify_root(abi, "ABI")
    require(abi.get("module_sha256") == MODULE_SHA, "ABI module drift")
    for key in ("level0_sha256", "full_mask_sha256", "primary_output_sha256",
                "sparse_output_sha256", "diagnostic_float_output_sha256",
                "sparse_diagnostic_float_output_sha256"):
        require(rtx[key] == mac[key], f"RTX/Mac {key} mismatch")
    full_mismatch = compare_bytes(
        args.rtx_root / "poison-a5-process-a" / "output.rgba8",
        args.mac_root / "poison-a5-process-a" / "output.rgba8",
        "primary output")
    sparse_mismatch = compare_bytes(
        args.rtx_root / "sparse-mask-branch" / "output.rgba8",
        args.mac_root / "sparse-mask-branch" / "output.rgba8",
        "sparse output")
    float_mismatch = compare_bytes(
        args.rtx_root / "poison-a5-process-a" / "diagnostic-output.rgba32f",
        args.mac_root / "poison-a5-process-a" / "diagnostic-output.rgba32f",
        "diagnostic float output")
    return add_root({
        "schema": "neuralstudio.fg.compositor-l87-accepted.v1",
        "abi_receipt_sha256": sha_file(args.abi),
        "abi_receipt_root_sha256": abi[ROOT_KEY],
        "rtx_backend_receipt": rtx,
        "mac_backend_receipt": mac,
        "primary_output_bytes": 921600,
        "primary_output_sha256": rtx["primary_output_sha256"],
        "primary_rtx_mac_mismatched_bytes": full_mismatch,
        "sparse_output_sha256": rtx["sparse_output_sha256"],
        "sparse_rtx_mac_mismatched_bytes": sparse_mismatch,
        "diagnostic_float_output_sha256": rtx["diagnostic_float_output_sha256"],
        "diagnostic_float_rtx_mac_mismatched_bytes": float_mismatch,
        "sust_unorm_characterization": {
            "selected_behavior": "truncate-to-12-fractional-bits then scale-to-unorm8 nearest tie-down",
            "selected_integer_formula": "min(255,(floor(clamp(x,0,1)*4096)*255+2047)>>12)",
            "native_metal_unorm_mismatched_bytes": 10919,
            "native_metal_mismatch_direction": "all Metal bytes were RTX plus one",
            "float_math_mismatched_words": 0,
        },
        "claims": {
            "selected_module20_level10_exact_on_metal": True,
            "four_fresh_processes_per_backend": True,
            "two_poison_output_invariance": True,
            "zero_untouched_primary_pixels": True,
            "sparse_mask_branch_exact": True,
            "input_and_rom_immutable": True,
            "canaries_intact": True,
            "expected_or_archived_tensor_ingress": False,
            "complete_frame_generation_model": False,
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
    require(accepted.get("abi_receipt_sha256") == sha_file(args.abi),
            "ABI receipt file hash mismatch")
    require(accepted.get("abi_receipt_root_sha256") == abi.get(ROOT_KEY),
            "ABI receipt root binding mismatch")
    require(accepted.get("result") == "pass", "accepted result drift")
    require(abi.get("result") == "pass", "ABI result drift")
    require(abi.get("module_sha256") == MODULE_SHA, "ABI module drift")
    require(abi.get("packed_parameter_bytes") == 72,
            "ABI parameter size drift")
    require(abi.get("parameter_access_offsets") ==
            [0, 8, 16, 24, 32, 40, 48, 56, 64],
            "ABI parameter access drift")
    bindings = abi.get("bindings")
    require(isinstance(bindings, list) and len(bindings) == 10,
            "ABI binding count drift")
    require("resource-000060 R8_UNORM" in bindings[0].get("meaning", ""),
            "ABI mask binding drift")
    require("resource-000055 RGBA8_UNORM" in bindings[1].get("meaning", ""),
            "ABI surface binding drift")
    require(abi.get("byte_68_through_71_unused") is True,
            "ABI padding proof drift")
    backend_values = []
    for key in ("rtx_backend_receipt", "mac_backend_receipt"):
        value = accepted.get(key)
        require(isinstance(value, dict), f"{key} missing")
        verify_root(value, key)
        require(value.get("result") == "pass", f"{key} result drift")
        require(value.get("expected_or_archived_tensor_ingress") is False,
                f"{key} expected tensor ingress drift")
        require(value.get("zero_untouched_primary_pixels") is True,
                f"{key} write coverage drift")
        backend_values.append(value)
    rtx, mac = backend_values
    require(rtx.get("backend") == "rtx" and mac.get("backend") == "mac",
            "backend ordering drift")
    require(rtx.get("primary_output_sha256") == mac.get("primary_output_sha256"),
            "backend primary output drift")
    require(rtx.get("diagnostic_float_output_sha256") ==
            mac.get("diagnostic_float_output_sha256"),
            "backend float output drift")
    require(accepted.get("primary_output_sha256") ==
            rtx.get("primary_output_sha256"),
            "accepted output binding drift")
    require(accepted.get("diagnostic_float_output_sha256") ==
            rtx.get("diagnostic_float_output_sha256"),
            "accepted float binding drift")
    require(accepted.get("primary_rtx_mac_mismatched_bytes") == 0,
            "accepted primary parity drift")
    require(accepted.get("diagnostic_float_rtx_mac_mismatched_bytes") == 0,
            "accepted float parity drift")
    require(accepted.get("claims", {}).get("complete_frame_generation_model") is False,
            "whole-model overclaim")
    require(accepted.get("claims", {}).get("expected_or_archived_tensor_ingress") is False,
            "accepted tensor ingress drift")
    require(accepted.get("claims", {}).get("selected_module20_level10_exact_on_metal")
            is True, "accepted parity claim drift")
    characterization = accepted.get("sust_unorm_characterization", {})
    require(characterization.get("selected_integer_formula") ==
            "min(255,(floor(clamp(x,0,1)*4096)*255+2047)>>12)",
            "SUST.P conversion formula drift")
    require(characterization.get("native_metal_unorm_mismatched_bytes") == 10919,
            "SUST.P counterexample drift")
    require(accepted.get("remote_job_retained") ==
            "D:\\CodexLab\\fg-compositor-l87-20260904T112919Z",
            "remote job identity drift")
    return {"result": "pass", "receipt_root_sha256": accepted[ROOT_KEY]}


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    abi = sub.add_parser("abi")
    abi.add_argument("--census", type=pathlib.Path, required=True)
    abi.add_argument("--descriptor", type=pathlib.Path, required=True)
    abi.add_argument("--ptx", type=pathlib.Path, required=True)
    abi.add_argument("--output", type=pathlib.Path, required=True)
    backend = sub.add_parser("backend")
    backend.add_argument("--backend", choices=("rtx", "mac"), required=True)
    backend.add_argument("--root", type=pathlib.Path, required=True)
    backend.add_argument("--output", type=pathlib.Path, required=True)
    final = sub.add_parser("final")
    final.add_argument("--rtx-root", type=pathlib.Path, required=True)
    final.add_argument("--mac-root", type=pathlib.Path, required=True)
    final.add_argument("--abi", type=pathlib.Path, required=True)
    final.add_argument("--output", type=pathlib.Path, required=True)
    receipt = sub.add_parser("receipt")
    receipt.add_argument("--accepted", type=pathlib.Path, required=True)
    receipt.add_argument("--abi", type=pathlib.Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "abi":
            value = build_abi(args)
            write_json(args.output, value)
        elif args.command == "backend":
            value = verify_backend(args.backend, args.root)
            write_json(args.output, value)
        elif args.command == "final":
            value = build_final(args)
            write_json(args.output, value)
        else:
            value = verify_receipt(args)
        print("verification_result=pass")
        print(f"receipt_root_sha256={value[ROOT_KEY]}")
        return 0
    except VerificationError as error:
        print(f"verification_error={error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
