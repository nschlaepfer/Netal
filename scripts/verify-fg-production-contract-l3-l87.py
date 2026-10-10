#!/usr/bin/env python3
"""Verify the 16-rendered-frame fg_host-contract RTX/Metal chain."""

from __future__ import annotations

import argparse, copy, hashlib, json
from pathlib import Path
from typing import Any

FRAME_OUTPUTS = [
"e7dbd1f46331372cf5be08f19ed7bdb24720a390f9d34b6da82e5aa7845fa214",
"c8a707cd4933fe21a66877d3d6272af7c8d2bcd270e2fbafe713091ddc236527",
"710563cf915cbf7659e54935b7cd604817c5f8b0bd4ed638037648fc2d12a097",
"b4e83a85afd99e33b43d0a9c7662d88dbfc2238ed731081182b22a0af0837dcf",
"7ec766aa1245aa048f1e6fb172662cdc2027d885e79bfd051c20ecbdf4c21d11",
"5f5570fb9238aeafe924f2390741af81135ba53047f26280c143a7d3d9d5dcdf",
"a1e1e23af7475d35a0f0e7deaee9a1562d73086da4063a96f5c94b63de25a694",
"68dc0ac480df81a929f2cd085ff7b0f948474034a2801addf2dd2adee6ac2fdf",
"093867624aa47e246d06d6d25a9860a0fac5b3b6fd370cca0f20c2f0613a5eb4",
"308aed8667910470dd4ab3e7e319f24ceae51c4e4f39f92d97df0d87839de89a",
"7b6d1a989568c3e72ded3b04f50087150e9d18e65f04f42ec87457f7b945befa",
"d2155168247af074c6ac9c67806774ceb617f48a10f63b0fb3d97efa204f357d",
"5bf9d686345643e44c06cd188663e7f5a03662f126cee414eb3094eae68f8fcd",
"94c2167a2a38197cc713605a770fe2842aa15848162ce9fe591b53643ab1cc2d",
"70ed17354acbbb534a7c38c17c70b7a352c3cb8b5c680ad0305a3106f7df396e"]
ALLOCATION37 = [
"42a790add781faee6551d2d1250b4ae34145cd6c9a41e4e8682b68301acb3f59",
"33faa1328791861c1e20a08582bacb31e7440f97c44bfc804268ae8ea7c8b627",
"fd02a9850c40f1cbc3e18465e442ad1fa665d3dcecf14349e8a1e60a49d6e06d",
"66ed857cceda216bf69550dc0bd318b11de07a58755368cb8bd1616e81a02cf3",
"e964d9e915c8d0ff04a4ed1bc9d2306accc4f41fbdcf59a3b0e1e23bc8609db7",
"c3ad0d4a9eafbd0e4363fcc667e79badd16d379324f085af929a8299d8218c02",
"95198a73d0080be70590a8b06f51095b97e071b62ff8a33150fe1e8ae0a95c75",
"b4f2454dcc79a81b6c4535e7462192fcc438e687d4e4f8230e46966f5e15150d",
"25d4622badbf8b1d9fd782f9ce52c281db84023a1947af4ce67efdc7311be86a",
"d5238025b243999a476b70884e621bd80fd7b4fe02875ae8f3ce5c0fe57b8428",
"710369167d835f687e9594f5200ae04d52501fdd381b47d39c827c17aeec1f05",
"36d24f2e7f2fbf70a489f5a8afcc4d044058313c0458fd67ba6f219c092bf30e",
"c91c6f8508775af0d98b636e48ef7fd70e4d60ed92d96e1627be1ce85f31713b",
"321d7974d29c19963d17054d1e8bdb55ff0b56b61ed3cf66c291479246b1bb21",
"1f21f7d89c732125b55a51c7739aa8a8301b7f7406f5939185550a8340fc2d87"]
ALLOCATION29 = [
"4d38cfb0052c45c9faf545a388c68a6fe33146789c285b327285fc94b7054ab2",
"bc9714e770d1226aa9dc34da2447b13b854574a3072f46c8745b9e490017f609",
"79afdf8a6b1666f48278a1d5898da717591aa4976e3b69adf2c4fdc77c453fed",
"cf87d1b1013fb6bb72235ef5491abf96c81b0badfaa29efa2b0f60c107e7ca11",
"498e9c5a11018799889e5b2bbb9a9bd3ed699ca70c81098e92bffbda43788a59",
"fb3fbd88589aae9a97c21e345553fa60e307e74d0eeb12f95807aef29f076a6e",
"4835f1d13b45be3521124b1604be80803a82730c889e33c6a36297bd22b33505",
"8d9bc333a51c0d9e4657b1597790dfdc607956a1729d7b73614bbe67654a1a9c",
"8b195e42d0b744a5eff82a0c4507656efa72d6f7988bc60ce662f4ede1d46f24",
"94ea7374683a1d1872fda678a7b2d8438a2f0c6d9a85dc4618d429e0eaacc3cb",
"030a09df5d287d56111464b49627109ce47ace954591acec86f4a1542ffc4c8c",
"f29c69d7a62f0a13604217a8d61112866571b75e2bf9535492034a77827f1068",
"9db19686371aaf7f3fb31b34b6acae510244397f93717ea2a4b40579ee15f991",
"e084f9ca8b231e2858d2fa05bd29694a1d8af330d44187d5fdad38551a492265",
"78f35edc7e51bf9426a26be07596c7847d8a76975cd359311ee19763dc7fb41b"]
CHECKPOINT_ROOT = "86b63bfdf21772fe343304d065ec91ebcab3362d52ccbb1d5a14eb6c483bb296"
SELECTED_MODULE_CENSUS_ROOT = "324e579b133caf212ad9db44908a73ae6baa90ffbb4978696b51ae99cb114c26"
ALLOCATION_PROVENANCE_ROOT = "8a687084ddbc9c0ebeefd4a12dabb98162691bb8dbf62a5505e531ef64c9c75c"
DESCRIPTOR_PROVENANCE_ROOT = "d3c72b68dcec1d1bb307239c7a782a4c49e06c92b121e8e74a1f0a63ba509733"

class VerificationError(RuntimeError): pass
def require(x: bool, m: str) -> None:
    if not x: raise VerificationError(m)
def load(p: Path) -> dict[str, Any]:
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise VerificationError(f"receipt read failed: {p}: {error}") from error
def digest(b: bytes) -> str: return hashlib.sha256(b).hexdigest()
def canonical(x: Any) -> str:
    return digest(json.dumps(x,sort_keys=True,separators=(",",":")).encode())
def keys() -> list[str]:
    return ([f"reset-launch-{i:06d}" for i in range(3,12)] +
            [f"generated-{f}-launch-{i:06d}" for f in range(1,16)
             for i in range(12,88)])
EXPECTED_KEYS=keys()

def validate_run(run: Path, backend: str,
                 receipt_override: dict[str,Any] | None=None) -> dict[str,str]:
    early=load(run/"run-receipt.json")
    tails=[load(run/f"generated-{f}/l20-l87/run-receipt.json")
           for f in range(1,16)]
    receipt=receipt_override or load(run/"temporal-receipt.json")
    require(receipt.get("schema")=="neuralstudio.fg.production-contract-l3-l87-run.v1",f"{run.name}: schema")
    require(receipt.get("backend")==backend,f"{run.name}: backend")
    early_backend = "mac-metal" if backend == "mac-metal" else "rtx-d3d12-nvapi"
    require(early.get("schema")=="neuralstudio.fg.production-contract-early-run.v1" and
            early.get("backend")==early_backend,f"{run.name}: early identity")
    require(early.get("rendered_frame_count")==16 and
            early.get("phase_launch_counts")==[9,120] and
            early.get("launch_count")==129 and early.get("l20_live_handoff") is True,
            f"{run.name}: early coverage")
    require(receipt.get("process_id")==early.get("process_id") and
            all(t.get("process_id")==early.get("process_id") for t in tails),f"{run.name}: same process")
    require(receipt.get("run_id")==early.get("run_id")==run.name and
            receipt.get("poison_byte")==early.get("poison_byte") and
            all(t.get("poison_byte")==early.get("poison_byte") for t in tails),
            f"{run.name}: run identity")
    require(receipt.get("rendered_frame_count")==16 and
            receipt.get("phase_launch_counts")==[9,1140] and
            receipt.get("launch_count")==1149,f"{run.name}: launch count")
    cp=receipt.get("checkpoint_sha256")
    require(isinstance(cp,dict) and set(cp)==set(EXPECTED_KEYS) and len(cp)==1149,f"{run.name}: checkpoints")
    expected=dict(early["checkpoint_sha256"])
    for f,t in enumerate(tails,1): expected.update({f"generated-{f}-{k}":v for k,v in t["checkpoint_sha256"].items()})
    require(cp==expected,f"{run.name}: checkpoint provenance")
    require(canonical(cp)==CHECKPOINT_ROOT,f"{run.name}: checkpoint root")
    require(receipt.get("allocation37_sha256")==ALLOCATION37 and
            receipt.get("allocation29_sha256")==ALLOCATION29,f"{run.name}: temporal state")
    require(receipt.get("final_output_sha256")==FRAME_OUTPUTS and
            receipt.get("final_output_bytes")==[921600]*15,f"{run.name}: outputs")
    require(early.get("allocation29_capture_bytes_consumed") is False and
            early.get("allocation29_runtime_source")=="generated-half-0.5-then-live-L52" and
            early.get("expected_or_archived_intermediate_ingress") is False and
            early.get("address_fields_recorded") is False and
            early.get("input_mismatched_bytes")==0 and early.get("canary_mismatched_bytes")==0 and
            early.get("allocation37_poison_halfwords")==0 and early.get("allocation29_poison_halfwords")==0,f"{run.name}: early integrity")
    for f,t in enumerate(tails,1):
        require(t.get("schema")=="neuralstudio.fg.connected-l20-l87-run.v1" and
                t.get("backend")==backend and t.get("launch_range")==[20,87] and
                t.get("launch_count")==68 and t.get("live_device_handoff_count")==67 and
                t.get("selected_module_census_root_sha256")==SELECTED_MODULE_CENSUS_ROOT and
                t.get("allocation_provenance_root_sha256")==ALLOCATION_PROVENANCE_ROOT and
                t.get("descriptor_provenance_root_sha256")==DESCRIPTOR_PROVENANCE_ROOT and
                t.get("address_fields_recorded") is False and
                t.get("input_model_mismatched_bytes")==0 and t.get("canary_mismatched_bytes")==0 and
                t.get("readonly_tail_mismatched_bytes")==0 and t.get("host_intermediate_ingress") is False and
                t.get("expected_or_archived_tensor_ingress") is False,f"{run.name}: tail {f}")
    require(receipt.get("host_intermediate_ingress") is False and
            receipt.get("expected_or_archived_tensor_ingress") is False and
            receipt.get("address_fields_recorded") is False and receipt.get("result")=="pass",f"{run.name}: policy")
    return cp

def backend(root:Path,name:str)->dict[str,Any]:
    runs=sorted(p for p in root.glob("fg-production-contract-l3-l87-*") if p.is_dir())
    require(len(runs)==4,f"{name}: four runs")
    maps=[validate_run(p,name) for p in runs];rs=[load(p/"temporal-receipt.json") for p in runs]
    require(len({r["process_id"] for r in rs})==4,f"{name}: fresh pids")
    require(sorted(r["poison_byte"] for r in rs)==[90,90,165,165],f"{name}: poisons")
    require(all(x==maps[0] for x in maps[1:]),f"{name}: invariance")
    return {"backend":name,"checkpoint_sha256":maps[0],"result":"pass"}

def verify(mac:Path,rtx:Path)->dict[str,Any]:
    m=backend(mac,"mac-metal");r=backend(rtx,"rtx-cuda-external-memory-d3d12-nvapi")
    require(m["checkpoint_sha256"]==r["checkpoint_sha256"],"cross-backend checkpoint mismatch")
    mr=sorted(mac.glob("fg-production-contract-l3-l87-*"))[0];rr=sorted(rtx.glob("fg-production-contract-l3-l87-*"))[0]
    for f in range(1,16):
        for rel,size,want in [("l20-l87/output.bin",921600,FRAME_OUTPUTS[f-1]),
                              ("allocation37.fp16",1474560,ALLOCATION37[f-1]),
                              ("allocation29-post-l52.r16f",115200,ALLOCATION29[f-1])]:
            a=mr/f"generated-{f}"/rel;b=rr/f"generated-{f}"/rel
            require(a.exists() and b.exists(),f"frame {f}: retrieved artifacts")
            av=a.read_bytes();bv=b.read_bytes();require(av==bv and len(av)==size and digest(av)==want,f"frame {f}: {rel}")
    result={"schema":"neuralstudio.fg.production-contract-l3-l87-verification.v1",
            "scope":"fg_host 640x360 contract, reset plus 15 generated frames",
            "rendered_frame_count":16,"generated_frame_count":15,
            "phase_launch_counts":[9,1140],"launch_count":1149,
            "checkpoint_root_sha256":CHECKPOINT_ROOT,
            "allocation37_sha256":ALLOCATION37,"allocation29_sha256":ALLOCATION29,
            "final_output_sha256":FRAME_OUTPUTS,"fresh_processes_per_backend":4,
            "same_process_all_launches":True,"two_poison_checkpoint_invariance":True,
            "live_temporal_bank_rotation":True,"input_model_immutability":True,
            "canary_integrity":True,"full_write_coverage":True,
            "expected_or_archived_tensor_ingress":False,"address_fields_recorded":False,
            "claims":{"production_contract_long16_exact":True,"production_game_capture":False,
                      "hudless_subrect_distortion":False,"netal_integration":False,"performance":False},
            "result":"pass"}
    result["verification_root_sha256"]=canonical(result);return result

def negative(mac:Path,rtx:Path)->dict[str,Any]:
    run=sorted(mac.glob("fg-production-contract-l3-l87-*"))[0];p=run/"temporal-receipt.json";original=load(p)
    muts={"checkpoint":lambda x:x["checkpoint_sha256"].__setitem__("generated-11-launch-000069","0"*64),
          "state":lambda x:x["allocation29_sha256"].__setitem__(2,"1"*64),
          "process":lambda x:x.__setitem__("process_id",x["process_id"]+1),
          "ingress":lambda x:x.__setitem__("expected_or_archived_tensor_ingress",True),
          "count":lambda x:x.__setitem__("launch_count",1148),
          "output":lambda x:x["final_output_sha256"].__setitem__(14,"f"*64)}
    rejected=[]
    for n,fn in muts.items():
        x=copy.deepcopy(original);fn(x)
        try:validate_run(run,"mac-metal",x)
        except VerificationError as e:rejected.append({"mutation":n,"observed_error":str(e)})
        else:raise VerificationError(f"mutation accepted: {n}")
    base=verify(mac,rtx)
    return {"schema":"neuralstudio.fg.production-contract-l3-l87-negative-audit.v1",
            "baseline_verification_root_sha256":base["verification_root_sha256"],
            "mutations":rejected,"all_mutations_rejected":True,"result":"pass"}

def main()->int:
    ap=argparse.ArgumentParser();ap.add_argument("mode",choices=("verify","negative-audit"));ap.add_argument("--mac",type=Path,required=True);ap.add_argument("--rtx",type=Path,required=True);ap.add_argument("--output",type=Path);a=ap.parse_args()
    result=verify(a.mac,a.rtx) if a.mode=="verify" else negative(a.mac,a.rtx)
    text=json.dumps(result,indent=2,sort_keys=True)+"\n"
    if a.output:
        if a.output.exists():raise VerificationError(f"preserve existing output: {a.output}")
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(text)
    print(text,end="");return 0
if __name__=="__main__":
    try:raise SystemExit(main())
    except VerificationError as e:raise SystemExit(f"verification_error={e}")
