#!/usr/bin/env python3
"""Verify the connected selected launch chain from L56 through L87."""
from __future__ import annotations
import argparse,hashlib,json,os,re,sys
from pathlib import Path
from typing import Any
RUN="neuralstudio.fg.connected-l56-l87-run.v1";BACKEND="neuralstudio.fg.connected-l56-l87-backend.v1";FINAL="neuralstudio.fg.connected-l56-l87-verification.v1";MANIFEST="2a1179d4a25be030f772cd90ca7d017ed6eac9291917f7d43ea9b0cd5ac6a294";OUTPUT_BYTES=640*360*4
CORE_INPUTS=["8132d194cbbb6a45e16986d7fb332ee3f4dfd3eb40d47bbb4842c96459868e3c","46b8fd0e814f87f94d41e1a20d4c8b72b69a4a5c78d918e9e8c619385a934ca4","5d42d679c9085725cb52d8e9e754dca03e87bb45c1a6785bfca020ed56973a20"];MASK_SHA="1fcc1e968752775f83abad77515c5bde83257518207ae028105fcf93251050ea";RCP_SHA="7cf4f8cece6b18b07605edc9cb601a153346a79b10bf3b93c7ec058d57d29750";SIG_SHA="394394a5258bad437495d68076be75d8413fa0ed5b752400e3947c332437b850";TAIL_MODULES=["3d8c6aeba3a82fdeb99672243a3d392776bd1742576ab44e8ef42af105c2fd76","07877ed3dcec1977a5ddfd7843314a5044ed9559bb2d222713906afdfc609d1c","143c6712c99db4a5792df697d68178afeea3657f4f0eeb12f905c90abf38a1b0","6cc983a76decaed1a8c7f207a95531bece5a1ad092c4db3a0159b590c331c24b"]
class E(RuntimeError):pass
def req(c:bool,m:str)->None:
 if not c:raise E(m)
def sha(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def canon(v:Any)->str:return sha(json.dumps(v,sort_keys=True,separators=(",",":")).encode())
def load(p:Path)->dict[str,Any]:v=json.loads(p.read_text(encoding="utf-8-sig"));req(isinstance(v,dict),f"not object {p}");return v
def write(p:Path,v:dict[str,Any])->None:p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+".tmp");t.write_text(json.dumps(v,indent=2,sort_keys=True)+"\n");os.replace(t,p)
def mask()->bytes:
 r=bytearray(640*360)
 for y in range(360):
  for x in range(640):
   island=(104<=x<224 and 72<=y<184)or(376<=x<552 and 196<=y<320);hole=(x*13+y*7)%47==0;r[y*640+x]=255 if island and not hole else 0
 return bytes(r)
def address_free(v:Any)->None:
 def f(n:Any)->None:
  if isinstance(n,dict):
   for k,x in n.items():
    if k!="address_fields_recorded":req(not any(q in k.lower()for q in("pointer","virtual_address","device_address","gpu_address")),"address key")
    f(x)
  elif isinstance(n,list):
   for x in n:f(x)
  elif isinstance(n,str):req(re.search(r"0x[0-9a-fA-F]{12,16}(?:\b|$)",n)is None,"address")
 f(v)
def expected_models()->list[dict[str,Any]]:
 repo=Path(__file__).resolve().parents[1];a=load(repo/"evidence/specs/fg-custom-conv-25-aggregate-v1/aggregate.json");req(a.get("receipt_root_sha256")=="6309c8c32c7fcb17434f429f16220b2385cfe6e8f34dfd329d5abede3641e1c0","aggregate")
 return [{"weight_token":x["weight_token"],"weight_sha256":x["weight_sha256"],"bias_token":x["bias_token"],"bias_sha256":x["bias_sha256"]}for x in a["launches"]]
def validate_backend(kind:str,root:Path)->dict[str,Any]:
 ident="rtx-hybrid-cuda-nvapi-d3d12"if kind=="rtx"else"mac-metal";models=expected_models();post=[[x["weight_sha256"],x["bias_sha256"]]for x in models];receipts=[];outputs=[]
 for poison in(0xA5,0x5A):
  for ord in("a","b"):
   rid=f"connected-l56-l87-poison-{poison:02x}-process-{ord}";r=load(root/rid/"run-receipt.json");o=(root/rid/"output.bin").read_bytes();req(r.get("schema")==RUN and r.get("backend")==ident and r.get("run_id")==rid and r.get("poison_byte")==poison,f"{rid}: identity");req(isinstance(r.get("process_id"),int),f"{rid}: pid");req(r.get("fixture")=="islands"if kind=="rtx"else True,f"{rid}: fixture");req(r.get("launch_range")==[56,87]and r.get("launch_count")==32 and sorted(r.get("checkpoint_sha256",{}))==[f"launch-{i:06d}"for i in range(56,88)],f"{rid}: checkpoints");req(r.get("model_manifest_sha256")==MANIFEST and r.get("model_state")==models,f"{rid}: model")
   if kind=="rtx":
    req(r.get("external_input_sha256")==CORE_INPUTS+[MASK_SHA]==r.get("external_input_post_sha256"),f"{rid}: inputs");req(r.get("model_state_immutable")is True and r.get("module_tail_sha256")==TAIL_MODULES,f"{rid}: modules");req(r.get("readonly_tail_mismatched_bytes")==0 and r.get("gpu_va_bridge_live")is True and r.get("texture_surface_handoffs_live")is True and r.get("host_intermediate_ingress")is False and r.get("expected_or_archived_tensor_ingress")is False,f"{rid}: bridge")
   else:
    req(r.get("external_input_pre_sha256")==CORE_INPUTS==r.get("external_input_post_sha256"),f"{rid}: core inputs");req(r.get("tail_input_sha256")==[MASK_SHA,RCP_SHA,SIG_SHA] and r.get("model_state_post_sha256")==post,f"{rid}: tail/model");req(r.get("archived_intermediate_ingress")is False and r.get("expected_output_file_opened")is False,f"{rid}: ingress")
   req(r.get("input_model_mismatched_bytes")==0 and r.get("canary_mismatched_bytes")==0 and r.get("in_process_handoff_count")==31 and r.get("address_fields_recorded")is False and r.get("result")=="pass",f"{rid}: gates");req(len(o)==OUTPUT_BYTES and r.get("final_output_bytes")==OUTPUT_BYTES and r.get("final_output_sha256")==sha(o),f"{rid}: output");address_free(r);receipts.append(r);outputs.append(o)
 req(len({r["process_id"]for r in receipts})==4,"fresh process");req(len({json.dumps(r["checkpoint_sha256"],sort_keys=True)for r in receipts})==1,"checkpoint poison invariance");req(len({sha(o)for o in outputs})==1,"output poison invariance");m=bytearray(outputs[0]);m[87]^=1;req(sha(m)!=receipts[0]["final_output_sha256"],"mutation")
 v={"schema":BACKEND,"backend":ident,"launch_range":[56,87],"launch_count":32,"checkpoint_sha256":receipts[0]["checkpoint_sha256"],"external_input_sha256":CORE_INPUTS+[MASK_SHA],"model_manifest_sha256":MANIFEST,"tail_module_sha256":TAIL_MODULES,"final_output_sha256":sha(outputs[0]),"final_output_bytes":OUTPUT_BYTES,"fresh_process_count":4,"two_poison_all_checkpoint_invariant":True,"full_write_coverage_from_two_poison_invariance":True,"input_model_immutability":True,"canary_integrity":True,"live_device_handoffs":True,"expected_or_archived_tensor_ingress":False,"output_mutation_rejected":True,"address_fields_recorded":False,"result":"pass"};v["receipt_root_sha256"]=canon(v);return v
def validate_final(rp:Path,mp:Path)->dict[str,Any]:
 r=load(rp);m=load(mp)
 for x,b in((r,"rtx-hybrid-cuda-nvapi-d3d12"),(m,"mac-metal")):req(x.get("schema")==BACKEND and x.get("backend")==b,f"{b}: identity");d=dict(x);h=d.pop("receipt_root_sha256",None);req(h==canon(d),f"{b}: root");address_free(x)
 req(r["checkpoint_sha256"]==m["checkpoint_sha256"],"checkpoint parity");req(r["final_output_sha256"]==m["final_output_sha256"],"final parity");v={"schema":FINAL,"launch_range":[56,87],"launch_count":32,"checkpoint_sha256":r["checkpoint_sha256"],"external_input_sha256":CORE_INPUTS+[MASK_SHA],"model_manifest_sha256":MANIFEST,"final_output_sha256":r["final_output_sha256"],"final_output_bytes":OUTPUT_BYTES,"rtx_backend_root_sha256":r["receipt_root_sha256"],"mac_backend_root_sha256":m["receipt_root_sha256"],"rtx_mac_checkpoint_mismatch_count":0,"rtx_mac_exact":True,"connected_in_process_handoffs_verified":31,"scope":"connected selected launches 56-87 for one generated islands fixture; no launches0-55, temporal sequence, Netal, or performance claim","complete_frame_claimed":False,"address_fields_recorded":False,"result":"pass"};v["receipt_root_sha256"]=canon(v);return v
def main()->int:
 p=argparse.ArgumentParser();s=p.add_subparsers(dest="cmd",required=True);b=s.add_parser("backend");b.add_argument("--backend",choices=("rtx","mac"),required=True);b.add_argument("--root",type=Path,required=True);b.add_argument("--output",type=Path,required=True);f=s.add_parser("final");f.add_argument("--rtx",type=Path,required=True);f.add_argument("--mac",type=Path,required=True);f.add_argument("--output",type=Path,required=True);a=p.parse_args()
 try:v=validate_backend(a.backend,a.root)if a.cmd=="backend"else validate_final(a.rtx,a.mac);write(a.output,v);print(f"receipt_root_sha256={v['receipt_root_sha256']}");print("result=pass");return 0
 except(OSError,json.JSONDecodeError,E)as e:print(f"verification_error={e}",file=sys.stderr);return 1
if __name__=="__main__":raise SystemExit(main())
