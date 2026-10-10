#!/usr/bin/env python3
"""Bind all 25 independently verified selected custom convolution launches."""
from __future__ import annotations
import argparse,hashlib,json,os,re,sys
from pathlib import Path
from typing import Any
class VerificationError(RuntimeError):pass
def require(c:bool,m:str)->None:
 if not c:raise VerificationError(m)
def sha(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def canon(v:Any)->str:return sha(json.dumps(v,sort_keys=True,separators=(",",":")).encode())
def load(p:Path)->dict[str,Any]:v=json.loads(p.read_text(encoding="utf-8-sig"));require(isinstance(v,dict),f"not object {p}");return v
def receipt_path(repo:Path,launch:int)->Path:
 if launch==57:return repo/"evidence/specs/fg-custom-conv-pre-v1/verification.json"
 if launch==58:return repo/"evidence/specs/fg-custom-conv0-c8-v1/verification.json"
 if launch==59:return repo/"evidence/specs/fg-custom-conv1-v1/verification.json"
 if 60<=launch<=67:return repo/f"evidence/specs/fg-custom-conv2-family-v1/layers/l{launch}-verification.json"
 if launch==68:return repo/"evidence/specs/fg-custom-bot0-v1/verification.json"
 if launch==69:return repo/"evidence/specs/fg-custom-block0-bot1-v1/verification.json"
 if launch==72:return repo/"evidence/specs/fg-custom-block1-conv0-v1/verification.json"
 if launch==73:return repo/"evidence/specs/fg-custom-block1-conv1-v1/verification.json"
 if 74<=launch<=81:return repo/f"evidence/specs/fg-custom-block1-conv2-family-v1/layers/l{launch}-verification.json"
 if launch==82:return repo/"evidence/specs/fg-custom-block1-bot0-v1/verification.json"
 if launch==83:return repo/"evidence/specs/fg-custom-block1-bot1-v1/verification.json"
 raise VerificationError(f"unexpected launch {launch}")
def address_free(v:Any)->None:
 def walk(n:Any)->None:
  if isinstance(n,dict):
   for k,c in n.items():
    if k!="address_fields_recorded":require(not any(x in k.lower()for x in("pointer","virtual_address","device_address","gpu_address")),f"address key {k}")
    walk(c)
  elif isinstance(n,list):
   for c in n:walk(c)
  elif isinstance(n,str):require(re.search(r"0x[0-9a-fA-F]{12,16}(?:\b|$)",n)is None,"address text")
 walk(v)
def validate(repo:Path)->dict[str,Any]:
 inventory_path=repo/"evidence/specs/fg-custom-conv-inventory-v1/inventory.json";inventory=load(inventory_path);require(inventory.get("receipt_root_sha256")=="93d869870d34327b1718ec37f418640936ac1d5829923218a61fc21344028f45","inventory root");expected=list(range(57,70))+list(range(72,84));rows=[]
 launches=inventory.get("launches");require(isinstance(launches,list)and len(launches)==25,"inventory count")
 by_launch={int(x["launch"]):x for x in launches};require(sorted(by_launch)==expected,"inventory coverage")
 for launch in expected:
  source=receipt_path(repo,launch);r=load(source);body=dict(r);root=body.pop("receipt_root_sha256",None);require(root==canon(body),f"L{launch}: receipt root");require(r.get("result")=="pass"and r.get("rtx_mac_exact")is True and r.get("rtx_mac_differing_bytes")==0 and r.get("address_fields_recorded")is False,f"L{launch}: parity");address_free(r);i=by_launch[launch];require(r.get("module_sha256")==i["module_sha256"]and r.get("weight_sha256")==i["weight_sha256"]and r.get("bias_sha256")==i["bias_sha256"],f"L{launch}: provenance");unit=r.get("selected_unit",r.get("layer_id"));require((f"{launch:06d}"in str(unit))or(unit==f"l{launch}"),f"L{launch}: identity");rows.append({"launch":launch,"function_name":i["function_name"],"module_token":i["module_token"],"module_sha256":i["module_sha256"],"weight_token":i["weight_token"],"weight_sha256":i["weight_sha256"],"bias_token":i["bias_token"],"bias_sha256":i["bias_sha256"],"output_sha256":r["output_sha256"],"verification_root_sha256":root,"source_receipt":str(source.relative_to(repo))})
 families={"block0_conv2_launches_60_67":load(repo/"evidence/specs/fg-custom-conv2-family-v1/family-verification.json")["receipt_root_sha256"],"block1_conv2_launches_74_81":load(repo/"evidence/specs/fg-custom-block1-conv2-family-v1/family-verification.json")["receipt_root_sha256"]}
 result={"schema":"neuralstudio.fg.custom-conv-25-aggregate.v1","inventory_receipt_root_sha256":inventory["receipt_root_sha256"],"launch_count":25,"launches":rows,"family_receipt_roots":families,"every_launch_executed_independently":True,"every_launch_rtx_mac_exact":True,"fresh_process_and_two_poison_gates_in_source_receipts":True,"connected_execution_claimed":False,"full_frame_claimed":False,"scope":"aggregate of 25 independent selected custom-convolution launch receipts; no connected custom block or full-frame claim","address_fields_recorded":False,"result":"pass"};result["receipt_root_sha256"]=canon(result);return result
def main()->int:
 p=argparse.ArgumentParser();p.add_argument("--repo",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
 try:v=validate(a.repo.resolve());a.output.parent.mkdir(parents=True,exist_ok=True);t=a.output.with_suffix(a.output.suffix+".tmp");t.write_text(json.dumps(v,indent=2,sort_keys=True)+"\n");os.replace(t,a.output);print(f"receipt_root_sha256={v['receipt_root_sha256']}");print("result=pass");return 0
 except(OSError,json.JSONDecodeError,VerificationError)as e:print(f"verification_error={e}",file=sys.stderr);return 1
if __name__=="__main__":raise SystemExit(main())
