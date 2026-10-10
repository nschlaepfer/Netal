#!/usr/bin/env python3
"""Verify runtime-u32 provenance against corrected schedule-bound u32 v2."""
from __future__ import annotations
import argparse,copy,hashlib,json
from pathlib import Path
from typing import Any

class Error(RuntimeError):pass
def req(v:bool,m:str)->None:
    if not v:raise Error(m)
def load(p:Path)->dict[str,Any]:return json.loads(p.read_text())
def root(v:Any)->str:return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",", ":")).encode()).hexdigest()

def verify(repo:Path,supersession:Path,override=None)->dict[str,Any]:
    value=override or load(supersession)
    runtime=load(repo/"evidence/specs/fg-runtime-v1/u32-promotion/promotion-receipt.json")
    runtime_verification=load(repo/"evidence/specs/fg-runtime-v1/u32-promotion/verification.json")
    v2=load(repo/"evidence/specs/fg-conv-hmma-u32-v2/verification.json")
    v2_receipt=load(repo/"evidence/specs/fg-conv-hmma-u32-v2/accepted-receipt.json")
    req(value.get("schema")=="neuralstudio.fg.runtime-u32-supersession.v2" and
        value.get("runtime_commit")=="df220131057b0d005bc2299706d0d7773a7cd2a0" and
        value.get("corrected_u32_commit")=="14b2cf442528949cf02547b557efa8951367ecbb" and
        value.get("runtime_verification_root_sha256")==runtime_verification["verification_root_sha256"] and
        value.get("runtime_checkpoint_root_sha256")==runtime["checkpoint_root_sha256"] and
        value.get("corrected_u32_verification_root_sha256")==v2["receipt_root_sha256"] and
        value.get("corrected_u32_backend_root_sha256")==v2["backend_root_sha256"] and
        value.get("superseded_u32_verification_root_sha256")==v2["supersedes_receipt_root_sha256"] and
        value.get("arithmetic_or_runtime_rerun_required") is False and value.get("result")=="pass",
        "supersession policy")
    req(v2.get("schema")=="neuralstudio.fg.conv-hmma-u32-verification.v2" and
        v2.get("total_mismatched_bytes")==0 and
        v2.get("claims",{}).get("schedule_specific_timing_validated") is True,
        "corrected v2 policy")
    req(runtime["u32_source_sha256"]==v2_receipt["source_sha256"]["src/fg_conv_simd_l20/hmma_exact_u32.metalh"] and
        runtime["override_source_sha256"]==v2_receipt["source_sha256"]["src/fg_conv_simd_l20/hmma_exact_u32_override.metalh"] and
        value.get("u32_source_sha256")==runtime["u32_source_sha256"] and
        value.get("u32_override_source_sha256")==runtime["override_source_sha256"],
        "arithmetic artifact identity")
    result={"schema":"neuralstudio.fg.runtime-u32-v2-supersession-verification.v1",
      "runtime_verification_root_sha256":runtime_verification["verification_root_sha256"],
      "runtime_checkpoint_root_sha256":runtime["checkpoint_root_sha256"],
      "corrected_u32_verification_root_sha256":v2["receipt_root_sha256"],
      "arithmetic_artifacts_identical":True,"runtime_rerun_required":False,"result":"pass"}
    result["verification_root_sha256"]=root(result);return result

def main()->int:
    p=argparse.ArgumentParser();p.add_argument("mode",choices=("verify","negative-audit"));p.add_argument("--repo",type=Path,required=True);p.add_argument("--supersession",type=Path,required=True);p.add_argument("--output",type=Path);a=p.parse_args()
    if a.mode=="verify":result=verify(a.repo,a.supersession)
    else:
        base=load(a.supersession);cases=[]
        for name,field,replacement in (("runtime-root","runtime_verification_root_sha256","0"*64),
            ("v2-root","corrected_u32_verification_root_sha256","f"*64),
            ("source","u32_source_sha256","0"*64),
            ("rerun","arithmetic_or_runtime_rerun_required",True)):
            changed=copy.deepcopy(base);changed[field]=replacement
            try:verify(a.repo,a.supersession,changed)
            except Error as e:cases.append({"mutation":name,"observed_error":str(e)})
            else:raise Error("mutation accepted: "+name)
        accepted=verify(a.repo,a.supersession)
        result={"schema":"neuralstudio.fg.runtime-u32-v2-supersession-negative.v1",
          "baseline_verification_root_sha256":accepted["verification_root_sha256"],
          "mutations":cases,"all_mutations_rejected":True,"result":"pass"}
    text=json.dumps(result,indent=2,sort_keys=True)+"\n"
    if a.output:
        if a.output.exists():raise Error("preserve output")
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(text)
    print(text,end="");return 0
if __name__=="__main__":
    try:raise SystemExit(main())
    except (Error,OSError,json.JSONDecodeError) as e:raise SystemExit("verification_error="+str(e))
