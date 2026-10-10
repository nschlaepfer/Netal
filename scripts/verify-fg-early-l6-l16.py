#!/usr/bin/env python3
"""Verify exact standalone parity for canonical FG L6/L16 module21."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pathlib
import sys
from typing import Any


MODULE_SHA = "11df926493fca0a981d560427b36c4dcf489ccb84d794501a70bda8d414ea8a2"
ROM_SHA = "7cf4f8cece6b18b07605edc9cb601a153346a79b10bf3b93c7ec058d57d29750"
FIXTURE_PARAM_SHA = "d93e4c2e556921152bb76fd7d9e3bd56e54cd708c558c9aa93df39374bfb9fc7"
LOWERING_ROOT = "18126bdcf0b45aac2449d974918bdd4d670bb5a58518c8b76e17e5365cb3989b"
BUFFER_BYTES = [57600, 86400, 21600, 5520, 1440]
ROOT_KEY = "receipt_root_sha256"
PHASES = {
    "reset": {
        "launch": 6,
        "flow": "37a75ef40eef3993afc38b212865cc04f420e4c471bffa660f7f40ba946a8807",
        "depth": "eafbecf7bbe9a5b185121f3bd32cb676885dfc7c53aa72f150a9a3011889302f",
        "input_mask": "69f90e80b122a25497b67e2858df407e695052f149b36c89695a9c2bb211a1b1",
        "candidates": "52d96f9d2a795a2cac4600741d6b9e52c5140bdae75593d0f8ce5f7f35fb4d9c",
        "vector": "d8f3cc05d2a54d9dffe5f74826f57d5fadfbf93b6ae07aa54aa333c3c2c3c2e1",
        "mask": "5f055e05774bf54d24b10a3dd7406587c127e46395eec9c8b4d406165865c0c9",
        "buffers": [
            "712c240df58f9aa4fc69d076f10c5d11a37e4cb6368b917005ecc37b7c12b7d2",
            "2c383c6eb6197d9145699ba330b11fe6f60b0fa97dcad02599d48d12d540bcb3",
            "d81b274fd38592a593041cb9f08d9878a3bb7524e1e8924c76a5c3a8453d9e72",
            "72e326215d731d608eff9805c1d393b4318832c4832f991871458ba59abf79dc",
            "81430d736f307a4e01d5adf91f484c235a680612617719d2c6b864c4fa631b3b",
        ],
    },
    "generated": {
        "launch": 16,
        "flow": "7167ce1f9369f10e3fb80255be4feaccf3671e4b408295416a52afceddac80f6",
        "depth": "163fa996bface28425e9c86eb13edbd3fc964dd75a2455fa17f329b0d6f64a5e",
        "input_mask": "c79146647f1cc92bcc0dff028073c63c71361a46d6f578e0b1a7d8527aad614c",
        "candidates": "84f7e496ab8123890710f14aff9198a4d78d7681c584839fe89a75051c5506f7",
        "vector": "78099580d89eb89f524decb2b3e346bd8ecc83b64ae34a46b52aa221a57e2ca8",
        "mask": "5f055e05774bf54d24b10a3dd7406587c127e46395eec9c8b4d406165865c0c9",
        "buffers": [
            "4ce03d337a9a7378bcaf62075b71fa531a1e808326504d5b52b7cb107057872f",
            "95a2e6f335e1ba027326177e35a933859f8a01f798967dead7a80cde5f0da399",
            "ae0c0bde6d2e216d3ad1e906edc620731fd9a55bf28e8ba9fe911e26ebb0bfa9",
            "0fe69f43e5f984739bc3167a8f5ed21680d93d953b71b2d64d1e4ae77b01723a",
            "d2dce7766ba988a8e3e8a089f1d94988c9c326198a923f6382e98bc72b1217fe",
        ],
    },
}
RUNS = {
    f"l{data['launch']}-{phase}-poison-{poison_name}-process-{ordinal}":
        (phase, poison)
    for phase, data in PHASES.items()
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
        raise VerificationError(f"unable to load {path}: {error}") from error
    require(isinstance(value, dict), f"JSON root is not object: {path}")
    return value


def canonical(value: Any, newline: bool = True) -> bytes:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return (text + ("\n" if newline else "")).encode()


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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
    body = copy.deepcopy(value); body.pop(ROOT_KEY, None)
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


def output_artifacts(root: pathlib.Path, phase: str, poison_name: str) -> list[bytes]:
    launch = PHASES[phase]["launch"]
    directory = root / f"l{launch}-{phase}-poison-{poison_name}-process-a"
    return [
        (directory / "vector-surface.bin").read_bytes(),
        (directory / "mask-surface.bin").read_bytes(),
        *[(directory / f"output-buffer-{index}.bin").read_bytes()
          for index in range(5)],
    ]


def verify_backend(backend: str, root: pathlib.Path,
                   repository: pathlib.Path,
                   lowering_path: pathlib.Path) -> dict[str, Any]:
    require(backend in ("rtx", "mac"), "unknown backend")
    expected_backend = "rtx-d3d12-nvapi" if backend == "rtx" else "mac-metal"
    processes: list[int] = []
    run_receipts: list[dict[str, Any]] = []
    for run_id, (phase, poison) in RUNS.items():
        expected = PHASES[phase]
        directory = root / run_id
        receipt_path = directory / "run-receipt.json"
        receipt = load(receipt_path)
        run_receipts.append({"run_id": run_id,
                             "bytes": receipt_path.stat().st_size,
                             "sha256": sha_file(receipt_path)})
        require(receipt.get("schema") == "neuralstudio.fg.early-l6-l16-run.v1" and
                receipt.get("backend") == expected_backend,
                f"{run_id}: schema/backend drift")
        require(receipt.get("run_id") == run_id and
                receipt.get("phase") == phase and
                receipt.get("selected_launch") == expected["launch"] and
                receipt.get("poison_byte") == poison,
                f"{run_id}: identity drift")
        require(receipt.get("module_token") == 21 and
                receipt.get("module_sha256") == MODULE_SHA and
                receipt.get("grid") == [20, 12, 1] and
                receipt.get("block") == [256, 1, 1] and
                receipt.get("packed_parameter_bytes") == 224,
                f"{run_id}: selected module/dispatch drift")
        require(receipt.get("loaded_byte_156") == 0 and
                receipt.get("padding_157_159") == [poison, poison, poison],
                f"{run_id}: loaded/padding byte drift")
        require(receipt.get("flow_pre_sha256") == expected["flow"] and
                receipt.get("flow_post_sha256") == expected["flow"] and
                receipt.get("depth_pre_sha256") == expected["depth"] and
                receipt.get("depth_post_sha256") == expected["depth"] and
                receipt.get("input_mask_pre_sha256") == expected["input_mask"] and
                receipt.get("input_mask_post_sha256") == expected["input_mask"] and
                receipt.get("candidates_pre_sha256") == expected["candidates"] and
                receipt.get("candidates_post_sha256") == expected["candidates"],
                f"{run_id}: input provenance/mutation")
        require(receipt.get("vector_surface_sha256") == expected["vector"] and
                receipt.get("mask_surface_sha256") == expected["mask"] and
                receipt.get("output_buffer_sha256") == expected["buffers"] and
                receipt.get("output_buffer_bytes") == BUFFER_BYTES,
                f"{run_id}: output drift")
        require(receipt.get("input_mismatched_bytes") == 0 and
                receipt.get("buffer_canary_mismatched_bytes") == 0,
                f"{run_id}: input/canary failure")
        require(receipt.get("expected_or_captured_output_ingress") is False and
                receipt.get("archived_intermediate_ingress") is False and
                receipt.get("address_fields_recorded") is False and
                receipt.get("result") == "pass",
                f"{run_id}: provenance/result gate")
        if backend == "mac":
            require(receipt.get("architecture_rom_sha256") == ROM_SHA and
                    receipt.get("vector_surface_rounding") == "toward-zero",
                    f"{run_id}: Mac architecture semantics drift")
        process = receipt.get("process_id")
        require(isinstance(process, int) and process > 0, f"{run_id}: PID missing")
        processes.append(process)
        verify_file(directory / "flow-after.bin", expected["flow"], 460800)
        verify_file(directory / "depth-after.bin", expected["depth"], 230400)
        verify_file(directory / "input-mask-after.bin", expected["input_mask"], 57600)
        verify_file(directory / "candidates-after.bin", expected["candidates"], 460800)
        verify_file(directory / "vector-surface.bin", expected["vector"], 230400)
        verify_file(directory / "mask-surface.bin", expected["mask"], 57600)
        for index, (digest, size) in enumerate(zip(expected["buffers"], BUFFER_BYTES)):
            verify_file(directory / f"output-buffer-{index}.bin", digest, size)
    require(len(set(processes)) == 8, "fresh-process gate failed")

    for phase in PHASES:
        a5 = output_artifacts(root, phase, "a5")
        five_a = output_artifacts(root, phase, "5a")
        require(len(a5) == len(five_a) == 7, "output artifact set drift")
        for index, (left, right) in enumerate(zip(a5, five_a)):
            require(left == right,
                    f"{phase}: output {index} retains poison-dependent bytes")

    lowering = load(lowering_path)
    verify_root(lowering, "lowering", newline=False)
    require(lowering.get(ROOT_KEY) == LOWERING_ROOT and
            lowering.get("division_sites_rewritten") == 24 and
            lowering.get("reciprocal_sites_rewritten") == 2 and
            lowering.get("mul_hi_u16_sites_repaired") == 1 and
            lowering.get("shared_u8_store_width_sites_repaired") == 2 and
            lowering.get("global_u8_store_width_sites_repaired") == 1 and
            lowering.get("expected_or_captured_output_embedded") is False,
            "lowering receipt drift")

    source_files = []
    build_artifacts = []
    rig_summary = None
    if backend == "rtx":
        rig_path = root / "rig-receipt.json"; rig = load(rig_path)
        require(rig.get("schema") == "neuralstudio.fg.early-l6-l16-rig.v1" and
                rig.get("fresh_process_count") == 8 and
                rig.get("true_d3d12_nvapi_resources") is True and
                rig.get("result") == "pass", "rig receipt drift")
        local = {
            "source_sha256": repository / "src/fg_early_l6_l16/rtx_nvapi_oracle.cpp",
            "contract_sha256": repository / "src/fg_early_l6_l16/contract.h",
            "base_source_sha256": repository / "src/fg_mid_l52/rtx_oracle.cpp",
            "harness_sha256": repository / "scripts/build-fg-early-l6-l16-rtx.ps1",
        }
        for field, path in local.items():
            digest = sha_file(path)
            require(rig.get(field) == digest, f"rig/local source drift: {field}")
            source_files.append({"file": str(path.relative_to(repository)),
                                 "sha256": digest})
        binary = root / "build/fg-early-l6-l16-rtx.exe"
        require(sha_file(binary) == rig["binary_sha256"], "RTX binary drift")
        build_artifacts.append({"file": binary.name, "bytes": binary.stat().st_size,
                                "sha256": sha_file(binary)})
        rig_summary = {"job_id": rig["job_id"], "attempt_id": rig["attempt_id"],
                       "gpu": rig["gpu"], "file_sha256": sha_file(rig_path),
                       "true_d3d12_nvapi_resources": True}
    else:
        paths = [
            repository / "src/fg_early_l6_l16/macos_probe.mm",
            repository / "src/fg_early_l6_l16/early_l6_l16.metal",
            repository / "src/fg_early_l6_l16/contract.h",
            repository / "src/fg_conv/macos_probe.mm",
            repository / "scripts/build-macos-fg-early-l6-l16.sh",
            repository / "scripts/lower-fg-early-module21-cumetal.py",
        ]
        for path in paths:
            source_files.append({"file": str(path.relative_to(repository)),
                                 "sha256": sha_file(path)})
        for path, expected in (
            (root / "build/kernel.metallib", None),
            (root / "build/probe", None),
            (root / "build/rcp.bin", ROM_SHA),
        ):
            require(path.is_file(), f"Mac artifact missing: {path}")
            digest = sha_file(path)
            if expected: require(digest == expected, "RCP ROM drift")
            build_artifacts.append({"file": path.name, "bytes": path.stat().st_size,
                                    "sha256": digest})
        require(sha_file(repository / "src/fg_early_l6_l16/early_l6_l16.metal") ==
                lowering["output_msl_sha256"], "lowered Metal source drift")

    result = {
        "schema": "neuralstudio.fg.early-l6-l16-backend.v1",
        "backend": expected_backend,
        "selected_launches": [6, 16],
        "module_token": 21,
        "module_sha256": MODULE_SHA,
        "packed_parameter_bytes": 224,
        "fixture_address_free_parameter_sha256": FIXTURE_PARAM_SHA,
        "selected_grid": [20, 12, 1],
        "selected_block": [256, 1, 1],
        "loaded_byte_156": 0,
        "padding_157_159_poison_invariant": True,
        "run_count": 8,
        "fresh_process_count": 8,
        "poison_bytes": [165, 90],
        "phase_outputs": {
            phase: {"vector_surface_sha256": data["vector"],
                    "mask_surface_sha256": data["mask"],
                    "output_buffer_sha256": data["buffers"]}
            for phase, data in PHASES.items()
        },
        "output_artifact_count_per_phase": 7,
        "all_output_bytes_poison_independent": True,
        "exact_write_coverage_all_outputs": True,
        "input_mismatched_bytes": 0,
        "buffer_canary_mismatched_bytes": 0,
        "input_immutable": True,
        "rom_sha256": ROM_SHA if backend == "mac" else None,
        "rom_immutable": True if backend == "mac" else None,
        "vector_surface_rounding": "toward-zero",
        "lowering_receipt_root_sha256": LOWERING_ROOT,
        "expected_or_captured_output_ingress": False,
        "archived_intermediate_ingress": False,
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
    expected = "rtx-d3d12-nvapi" if backend == "rtx" else "mac-metal"
    require(value.get("schema") == "neuralstudio.fg.early-l6-l16-backend.v1" and
            value.get("backend") == expected, f"{backend}: identity drift")
    require(value.get("selected_launches") == [6, 16] and
            value.get("module_sha256") == MODULE_SHA and
            value.get("selected_block") == [256, 1, 1],
            f"{backend}: selected module drift")
    require(value.get("fresh_process_count") == 8 and
            value.get("padding_157_159_poison_invariant") is True and
            value.get("all_output_bytes_poison_independent") is True and
            value.get("exact_write_coverage_all_outputs") is True,
            f"{backend}: process/padding/coverage gate")
    require(value.get("phase_outputs") == {
        phase: {"vector_surface_sha256": data["vector"],
                "mask_surface_sha256": data["mask"],
                "output_buffer_sha256": data["buffers"]}
        for phase, data in PHASES.items()}, f"{backend}: output root drift")
    require(value.get("input_immutable") is True and
            value.get("input_mismatched_bytes") == 0 and
            value.get("buffer_canary_mismatched_bytes") == 0,
            f"{backend}: input/canary gate")
    if backend == "mac":
        require(value.get("rom_sha256") == ROM_SHA and
                value.get("rom_immutable") is True, "mac: ROM gate")
    else:
        require(value.get("rig", {}).get("true_d3d12_nvapi_resources") is True,
                "rtx: typed resource gate")
    require(value.get("lowering_receipt_root_sha256") == LOWERING_ROOT and
            value.get("expected_or_captured_output_ingress") is False and
            value.get("archived_intermediate_ingress") is False and
            value.get("address_fields_recorded") is False and
            value.get("result") == "pass", f"{backend}: provenance/result gate")


def make_final(rtx: dict[str, Any], mac: dict[str, Any], abi: dict[str, Any],
               counterexamples: dict[str, Any]) -> dict[str, Any]:
    validate_backend(rtx, "rtx"); validate_backend(mac, "mac")
    verify_root(abi, "ABI"); verify_root(counterexamples, "counterexamples")
    require(abi.get("schema") == "neuralstudio.fg.early-l6-l16-abi.v1" and
            abi.get("module_sha256") == MODULE_SHA and
            abi.get("claims", {}).get("true_typed_resource_contract_recovered")
            is True, "ABI drift")
    require(counterexamples.get("all_failures_preserved") is True,
            "failure preservation drift")
    require(rtx["phase_outputs"] == mac["phase_outputs"],
            "RTX/Metal output mismatch")
    return rooted({
        "schema": "neuralstudio.fg.early-l6-l16-accepted.v1",
        "selected_units": [
            "canonical L6/module-000021 reset multiscale producer",
            "canonical L16/module-000021 generated multiscale producer"],
        "abi_receipt_root_sha256": abi[ROOT_KEY],
        "counterexamples_root_sha256": counterexamples[ROOT_KEY],
        "rtx_backend_root_sha256": rtx[ROOT_KEY],
        "mac_backend_root_sha256": mac[ROOT_KEY],
        "lowering_receipt_root_sha256": LOWERING_ROOT,
        "module_sha256": MODULE_SHA,
        "phase_outputs": rtx["phase_outputs"],
        "output_artifacts_per_phase": 7,
        "rtx_mac_mismatched_bytes_all_outputs": 0,
        "processes_per_backend": 8,
        "padding_157_159_poison_invariant": True,
        "exact_write_coverage_all_outputs": True,
        "input_and_rom_immutable": True,
        "canary_mismatched_bytes": 0,
        "expected_or_captured_output_ingress": False,
        "address_fields_recorded": False,
        "claims": {
            "l6_standalone_exact_on_metal": True,
            "l16_standalone_exact_on_metal": True,
            "true_typed_d3d12_oracle": True,
            "connected_early_chain": False,
            "generated_frame": False,
            "netal_integration": False,
            "performance": False},
        "result": "pass"})


def validate_final(value: dict[str, Any], rtx: dict[str, Any], mac: dict[str, Any],
                   abi: dict[str, Any], counterexamples: dict[str, Any]) -> None:
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
    a,_,_,_,_=add("accepted-module-drift");a["module_sha256"]="0"*64
    a,_,_,_,_=add("accepted-parity-drift");a["rtx_mac_mismatched_bytes_all_outputs"]=1
    a,_,_,_,_=add("accepted-overclaim");a["claims"]["generated_frame"]=True
    _,r,_,_,_=add("rtx-process-drift");r["fresh_process_count"]=7
    _,r,_,_,_=add("rtx-typed-resource-drift");r["rig"]["true_d3d12_nvapi_resources"]=False
    _,r,_,_,_=add("rtx-padding-drift");r["padding_157_159_poison_invariant"]=False
    _,_,m,_,_=add("mac-rom-drift");m["rom_immutable"]=False
    _,_,m,_,_=add("mac-rounding-drift");m["vector_surface_rounding"]="nearest-even"
    _,_,m,_,_=add("mac-coverage-drift");m["exact_write_coverage_all_outputs"]=False
    _,_,_,b,_=add("abi-resource-drift");b["claims"]["true_typed_resource_contract_recovered"]=False
    _,_,_,b,_=add("abi-byte156-drift");b["parameter_layout"][15]["meaning"]="loaded=1"
    _,_,_,_,c=add("counterexample-loss");c["all_failures_preserved"]=False
    _,r,_,_,_=add("rtx-ingress-drift");r["expected_or_captured_output_ingress"]=True
    a,_,_,_,_=add("accepted-root-corruption");a[ROOT_KEY]="f"*64
    results=[]
    for name,ma,mr,mm,mb,mc in cases:
        rejected=False;reason=""
        try:validate_final(ma,mr,mm,mb,mc)
        except VerificationError as error:rejected=True;reason=str(error)
        require(rejected,f"negative mutation accepted: {name}")
        results.append({"mutation":name,"rejected":True,"reason":reason})
    return rooted({"schema":"neuralstudio.fg.early-l6-l16-negative-audit.v1",
                   "accepted_receipt_root_sha256":accepted[ROOT_KEY],
                   "mutation_count":len(results),"mutations":results,
                   "all_mutations_rejected":True,"result":"pass"})


def main() -> int:
    parser=argparse.ArgumentParser();commands=parser.add_subparsers(dest="command",required=True)
    backend=commands.add_parser("backend");backend.add_argument("--backend",choices=("rtx","mac"),required=True);backend.add_argument("--run-root",type=pathlib.Path,required=True);backend.add_argument("--repository",type=pathlib.Path,required=True);backend.add_argument("--lowering",type=pathlib.Path,required=True);backend.add_argument("--output",type=pathlib.Path,required=True)
    for name in ("final","receipt","negative"):
        sub=commands.add_parser(name)
        if name in ("receipt","negative"):sub.add_argument("--accepted",type=pathlib.Path,required=True)
        for arg in ("rtx","mac","abi","counterexamples"):sub.add_argument(f"--{arg}",type=pathlib.Path,required=True)
        if name in ("final","negative"):sub.add_argument("--output",type=pathlib.Path,required=True)
    args=parser.parse_args()
    try:
        if args.command=="backend":
            value=verify_backend(args.backend,args.run_root,args.repository,args.lowering);write_json(args.output,value)
        else:
            rtx,mac,abi,counters=load(args.rtx),load(args.mac),load(args.abi),load(args.counterexamples)
            if args.command=="final":value=make_final(rtx,mac,abi,counters);write_json(args.output,value)
            elif args.command=="receipt":value=load(args.accepted);validate_final(value,rtx,mac,abi,counters)
            else:
                accepted=load(args.accepted);validate_final(accepted,rtx,mac,abi,counters);value=make_negative(accepted,rtx,mac,abi,counters);write_json(args.output,value)
        print(f"receipt_root_sha256={value[ROOT_KEY]}");print("verification_result=pass");return 0
    except VerificationError as error:
        print(f"verification_result=fail\nerror={error}",file=sys.stderr);return 2


if __name__=="__main__":raise SystemExit(main())
