#!/usr/bin/env python3
"""Verify the live L52/L53/L55 RTX/Metal runtime chain."""

from __future__ import annotations

import argparse, copy, hashlib, json
from pathlib import Path
from typing import Any, Callable

CHECKPOINT_ROOT = "2b13a70a5d44659a6912c872a01adfd0e9eceaa02b668743cef82a0ea088a156"


class VerificationError(RuntimeError): pass
def require(value: bool, message: str) -> None:
    if not value: raise VerificationError(message)
def load(path: Path) -> dict[str,Any]: return json.loads(path.read_text())
def sha(data: bytes) -> str: return hashlib.sha256(data).hexdigest()
def canonical(value: Any) -> str:
    return sha(json.dumps(value,sort_keys=True,separators=(",", ":")).encode())
def suffix(path: Path) -> str: return path.name.split("poison-",1)[1]


def validate_provenance(mac:Path,rtx:Path,provenance_path:Path,
                        override:dict[str,Any]|None=None)->None:
    value=override or load(provenance_path);repo=Path(__file__).resolve().parent.parent
    require(value.get("schema")=="neuralstudio.fg.runtime-live-alias-provenance.v1" and
            value.get("expected_or_archived_intermediate_ingress") is False and
            value.get("address_fields_recorded") is False and value.get("result")=="pass","provenance policy")
    for relative,want in value.get("local_source_sha256",{}).items():
        path=repo/relative;require(path.is_file() and sha(path.read_bytes())==want,"source pin: "+relative)
    require(sha((mac/"build/runtime_tail.mm").read_bytes())==value["mac"]["generated_tail_sha256"] and
            sha((mac/"fg-runtime-offline").read_bytes())==value["mac"]["binary_sha256"],"Mac build pins")
    archive=rtx.parent/(rtx.name+".tar.gz")
    require(archive.is_file() and sha(archive.read_bytes())==value["rtx"]["retrieved_archive_sha256"],"RTX archive pin")
    backend=load(rtx/"backend-receipt.json")
    require(backend.get("early_source_sha256")==value["rtx"]["generated_early_sha256"] and
            backend.get("generated_tail_source_sha256")==value["rtx"]["generated_tail_sha256"] and
            backend.get("binary_sha256")==value["rtx"]["binary_sha256"],"RTX build pins")


def validate(mac: Path, rtx: Path,
             mutation: Callable[[dict[str,Any]],None]|None=None,
             provenance:Path|None=None)->dict[str,Any]:
    mac_runs=sorted(p for p in mac.glob("fg-runtime-poison-*-process-*") if p.is_dir())
    rtx_runs=sorted(p for p in rtx.glob("fg-production-contract-l3-l87-poison-*-process-*") if p.is_dir())
    require(len(mac_runs)==4 and len(rtx_runs)==4,"four processes per backend")
    require([suffix(p) for p in mac_runs]==[suffix(p) for p in rtx_runs],"run pairing")
    mac_receipts=[load(p/"run-receipt.json") for p in mac_runs]
    rtx_receipts=[load(p/"temporal-receipt.json") for p in rtx_runs]
    if mutation is not None: mutation(mac_receipts[0])
    require(len({r["process_id"] for r in mac_receipts})==4 and
            len({r["process_id"] for r in rtx_receipts})==4 and
            all(isinstance(r["process_id"],int) and r["process_id"]>0
                for r in mac_receipts+rtx_receipts),"fresh process IDs")
    require(sorted(r["poison_byte"] for r in mac_receipts)==[90,90,165,165] and
            sorted(r["poison_byte"] for r in rtx_receipts)==[90,90,165,165],"poison coverage")
    maps=[];outputs=[]
    for mpath,rpath,mreceipt,rreceipt in zip(mac_runs,rtx_runs,mac_receipts,rtx_receipts):
        require(mreceipt.get("schema")=="neuralstudio.fg.runtime-run.v1" and
                mreceipt.get("ex2_mode")=="compact" and
                mreceipt.get("auxiliary_mode")=="live" and
                mreceipt.get("l52_l53_l55_live_early_resource_aliases") is True and
                mreceipt.get("l53_l55_auxiliary_profile")=="live-early-resource-aliases" and
                mreceipt.get("runtime_input_mismatched_bytes")==0 and
                mreceipt.get("runtime_canary_mismatched_bytes")==0 and
                mreceipt.get("resource_allocations_per_generated_frame")==[210]+[0]*14 and
                mreceipt.get("expected_or_archived_intermediate_ingress") is False and
                mreceipt.get("tail_probe_file_io") is False and
                mreceipt.get("result")=="pass",f"{mpath.name}: Metal contract")
        require(rreceipt.get("schema")=="neuralstudio.fg.production-contract-l3-l87-run.v1" and
                rreceipt.get("launch_count")==1149 and
                rreceipt.get("host_intermediate_ingress") is False and
                rreceipt.get("expected_or_archived_tensor_ingress") is False and
                rreceipt.get("address_fields_recorded") is False and
                rreceipt.get("result")=="pass",f"{rpath.name}: RTX contract")
        checkpoints=mreceipt.get("checkpoint_sha256")
        require(isinstance(checkpoints,dict) and len(checkpoints)==1149 and
                checkpoints==rreceipt.get("checkpoint_sha256") and
                canonical(checkpoints)==CHECKPOINT_ROOT,f"{mpath.name}: checkpoint parity")
        require(mreceipt.get("output_sha256")==rreceipt.get("final_output_sha256"),
                f"{mpath.name}: output parity")
        early=load(rpath/"run-receipt.json")
        require(early.get("input_mismatched_bytes")==0 and
                early.get("canary_mismatched_bytes")==0 and early.get("result")=="pass",
                f"{rpath.name}: RTX early integrity")
        for frame in range(1,16):
            tail=load(rpath/f"generated-{frame}/l20-l87/run-receipt.json")
            require(tail.get("input_model_mismatched_bytes")==0 and
                    tail.get("canary_mismatched_bytes")==0 and
                    tail.get("readonly_tail_mismatched_bytes")==0 and
                    tail.get("expected_or_archived_tensor_ingress") is False and
                    tail.get("result")=="pass",f"{rpath.name}: tail {frame}")
        maps.append(checkpoints);outputs.append(mreceipt["output_sha256"])
    require(all(value==maps[0] for value in maps[1:]) and
            all(value==outputs[0] for value in outputs[1:]),"process/poison invariance")
    mac_rep=mac_runs[[p.name for p in mac_runs].index("fg-runtime-poison-a5-process-a")]
    rtx_rep=rtx_runs[[p.name for p in rtx_runs].index("fg-production-contract-l3-l87-poison-a5-process-a")]
    for frame,expected in enumerate(outputs[0],1):
        mbytes=(mac_rep/f"generated-{frame}.rgba8").read_bytes()
        rbytes=(rtx_rep/f"generated-{frame}/l20-l87/output.bin").read_bytes()
        require(mbytes==rbytes and len(mbytes)==921600 and sha(mbytes)==expected,
                f"frame {frame}: retrieved output bytes")
    if provenance is not None: validate_provenance(mac,rtx,provenance)
    result={"schema":"neuralstudio.fg.runtime-live-alias-verification.v1",
            "scope":"640x360 compact runtime, reset plus 15 generated, live early-resource aliases",
            "checkpoint_root_sha256":CHECKPOINT_ROOT,
            "output_sha256":outputs[0],"selected_launch_count":1149,
            "fresh_processes_per_backend":4,"poison_bytes":[90,165],
            "l52_l53_l55_live_early_resource_aliases":True,
            "metal_resource_allocations_per_generated_frame":[210]+[0]*14,
            "retrieved_outputs_byte_exact":True,
            "expected_or_archived_intermediate_ingress":False,
            "address_fields_recorded":False,
            "source_build_pins_verified":provenance is not None,
            "claims":{"game_capture":False,"netal_integration":False,"performance":False},
            "result":"pass"}
    result["verification_root_sha256"]=canonical(result);return result


def negative(mac:Path,rtx:Path,provenance:Path|None=None)->dict[str,Any]:
    mutations:dict[str,Callable[[dict[str,Any]],None]]={
      "checkpoint":lambda x:x["checkpoint_sha256"].__setitem__("generated-5-launch-000068","0"*64),
      "output":lambda x:x["output_sha256"].__setitem__(10,"f"*64),
      "alias":lambda x:x.__setitem__("l52_l53_l55_live_early_resource_aliases",False),
      "ingress":lambda x:x.__setitem__("expected_or_archived_intermediate_ingress",True),
      "allocation":lambda x:x["resource_allocations_per_generated_frame"].__setitem__(8,1),
      "poison":lambda x:x.__setitem__("poison_byte",17),
      "process":lambda x:x.__setitem__("process_id",0)}
    rejected=[]
    for name,mutate in mutations.items():
        try:validate(mac,rtx,mutate,provenance)
        except (VerificationError,ValueError) as error:rejected.append({"mutation":name,"observed_error":str(error)})
        else:raise VerificationError("mutation accepted: "+name)
    if provenance is not None:
        changed=copy.deepcopy(load(provenance));first=sorted(changed["local_source_sha256"])[0]
        changed["local_source_sha256"][first]="f"*64
        try:validate_provenance(mac,rtx,provenance,changed)
        except VerificationError as error:rejected.append({"mutation":"source_pin","observed_error":str(error)})
        else:raise VerificationError("mutation accepted: source_pin")
    baseline=validate(mac,rtx,provenance=provenance)
    return {"schema":"neuralstudio.fg.runtime-live-alias-negative-audit.v1",
            "baseline_verification_root_sha256":baseline["verification_root_sha256"],
            "mutations":rejected,"all_mutations_rejected":True,"result":"pass"}


def main()->int:
    p=argparse.ArgumentParser();p.add_argument("mode",choices=("verify","negative-audit"));p.add_argument("--mac",type=Path,required=True);p.add_argument("--rtx",type=Path,required=True);p.add_argument("--provenance",type=Path);p.add_argument("--output",type=Path);a=p.parse_args()
    result=validate(a.mac,a.rtx,provenance=a.provenance) if a.mode=="verify" else negative(a.mac,a.rtx,a.provenance)
    text=json.dumps(result,indent=2,sort_keys=True)+"\n"
    if a.output:
        if a.output.exists():raise VerificationError("preserve existing output")
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(text)
    print(text,end="");return 0
if __name__=="__main__":
    try:raise SystemExit(main())
    except (VerificationError,OSError,json.JSONDecodeError) as error:raise SystemExit("verification_error="+str(error))
