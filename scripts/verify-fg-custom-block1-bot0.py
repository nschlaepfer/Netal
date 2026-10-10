#!/usr/bin/env python3
"""Verify launch 82 custom block1 bottom convolution on RTX and Metal."""
from __future__ import annotations
import argparse,hashlib,json,os,re,struct,sys
from pathlib import Path
from typing import Any
MODULE_SHA256="0f80c161e89ca4354f932ea6c3f7a0d3206e0a4fff7530d1da0fe4b162102faa";WEIGHT_SHA256="4ab572a084310d9f79b7878aefd73a356900ea8428320b861c6de28c114977d6";BIAS_SHA256="3ae35c7933b00a52114c7a75f0184bd480c5c814d2204271bd46da8f88772913";MANIFEST_SHA256="2a1179d4a25be030f772cd90ca7d017ed6eac9291917f7d43ea9b0cd5ac6a294";OUTPUT_BYTES=3*184*320*16*2;RUN_SCHEMA="neuralstudio.fg.custom-block1-bot0-run.v1";BACKEND_SCHEMA="neuralstudio.fg.custom-block1-bot0-backend.v1";FINAL_SCHEMA="neuralstudio.fg.custom-block1-bot0-verification.v1"
class VerificationError(RuntimeError):pass
def require(c:bool,m:str)->None:
 if not c:raise VerificationError(m)
def sha(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def canon(v:Any)->str:return sha(json.dumps(v,sort_keys=True,separators=(",",":")).encode())
def load(p:Path)->dict[str,Any]:v=json.loads(p.read_text(encoding="utf-8-sig"));require(isinstance(v,dict),f"not object {p}");return v
def write(p:Path,v:dict[str,Any])->None:p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+".tmp");t.write_text(json.dumps(v,indent=2,sort_keys=True)+"\n");os.replace(t,p)
def address_free(v:Any,label:str)->None:
 def visit(n:Any)->None:
  if isinstance(n,dict):
   for k,c in n.items():
    if k!="address_fields_recorded":require(not any(x in k.lower()for x in("pointer","virtual_address","device_address","gpu_address")),f"{label}: address key")
    visit(c)
  elif isinstance(n,list):
   for c in n:visit(c)
  elif isinstance(n,str):require(re.search(r"0x[0-9a-fA-F]{12,16}(?:\b|$)",n)is None,f"{label}: address")
 visit(v)
def mix(v:int)->int:
 m=(1<<64)-1;v=(v+0x9E3779B97F4A7C15)&m;v=((v^(v>>30))*0xBF58476D1CE4E5B9)&m;v=((v^(v>>27))*0x94D049BB133111EB)&m;return(v^(v>>31))&m
def generated_input()->bytes:
 vals=(0,0x2800,0xA800,0x2C00,0xAC00,0x3000,0xB000,0x3400,0xB400,0x3800,0xB800);r=bytearray(92*160*32*2)
 for i in range(len(r)//2):struct.pack_into("<H",r,2*i,vals[mix((0xA4093822299F31D0+i*0xD1B54A32D192ED03)&((1<<64)-1))%len(vals)])
 return bytes(r)
INPUT_SHA256=sha(generated_input())
def validate_backend(backend:str,root:Path)->dict[str,Any]:
 ident="rtx-cuda-driver"if backend=="rtx"else"mac-metal";rs=[];outs=[]
 for poison in(0xA5,0x5A):
  for ord in("a","b"):
   rid=f"block1-bot0-poison-{poison:02x}-process-{ord}";r=load(root/rid/"run-receipt.json");o=(root/rid/"output.bin").read_bytes();require(r.get("schema")==RUN_SCHEMA and r.get("backend")==ident and r.get("run_id")==rid and r.get("poison_byte")==poison,f"{rid}: identity");require(isinstance(r.get("process_id"),int),f"{rid}: pid");require(r.get("trace_launch_token")=="launch-000082"and r.get("module_token")==65 and r.get("module_bytes")==5264 and r.get("module_sha256")==MODULE_SHA256,f"{rid}: module");require(r.get("weight_source_token")==198 and r.get("bias_source_token")==200 and r.get("model_manifest_sha256")==MANIFEST_SHA256,f"{rid}: provenance");require(r.get("weight_pre_sha256")==WEIGHT_SHA256==r.get("weight_post_sha256")and r.get("bias_pre_sha256")==BIAS_SHA256==r.get("bias_post_sha256"),f"{rid}: model");require(r.get("input_generator")=="splitmix64-fg-custom-block1-bot0-v1"and r.get("input_pre_sha256")==INPUT_SHA256==r.get("input_post_sha256"),f"{rid}: input");require(len(o)==OUTPUT_BYTES and r.get("output_bytes")==OUTPUT_BYTES and r.get("output_sha256")==sha(o),f"{rid}: output");require(r.get("input_mismatched_bytes")==0 and r.get("canary_mismatched_bytes")==0 and r.get("external_model_files_opened")is True and r.get("expected_output_file_opened")is False and r.get("address_fields_recorded")is False and r.get("result")=="pass",f"{rid}: gates")
   if backend=="rtx":require(r.get("grid")==[40,46,1]and r.get("block")==[32,2,6]and r.get("packed_parameter_bytes")==96 and r.get("function_name")=="custom_block1_conv_bot0_hf_kernel",f"{rid}: RTX")
   else:require(r.get("selected_rtx_grid")==[40,46,1]and r.get("selected_rtx_block")==[32,2,6]and r.get("metal_dispatch_threads")==[320,184,48]and r.get("metal_threads_per_threadgroup")==[8,4,1]and r.get("metal_function")=="fg_custom_block1_bot0",f"{rid}: Metal")
   address_free(r,rid);rs.append(r);outs.append(o)
 require(len({r["process_id"]for r in rs})==4,"fresh process");require(len({sha(o)for o in outs})==1,"poison");m=bytearray(outs[0]);m[61]^=1;require(sha(m)!=rs[0]["output_sha256"],"mutation")
 result={"schema":BACKEND_SCHEMA,"backend":ident,"selected_unit":"evaluate-1/launch-000082/custom_block1_conv_bot0_hf_kernel/module-65","module_sha256":MODULE_SHA256,"weight_sha256":WEIGHT_SHA256,"bias_sha256":BIAS_SHA256,"model_manifest_sha256":MANIFEST_SHA256,"input_sha256":INPUT_SHA256,"output_sha256":sha(outs[0]),"output_bytes":OUTPUT_BYTES,"fresh_process_count":4,"two_process_two_poison_invariant":True,"full_write_coverage_from_two_poison_invariance":True,"input_model_immutability":True,"canary_integrity":True,"expected_output_fed_to_execution":False,"output_mutation_rejected":True,"address_fields_recorded":False,"result":"pass"};result["receipt_root_sha256"]=canon(result);return result
def validate_final(rp:Path,mp:Path)->dict[str,Any]:
 r=load(rp);m=load(mp)
 for x,b in((r,"rtx-cuda-driver"),(m,"mac-metal")):require(x.get("schema")==BACKEND_SCHEMA and x.get("backend")==b,f"{b}: id");d=dict(x);h=d.pop("receipt_root_sha256",None);require(h==canon(d),f"{b}: root");address_free(x,b)
 require(r["output_sha256"]==m["output_sha256"],"parity");v={"schema":FINAL_SCHEMA,"selected_unit":r["selected_unit"],"module_sha256":MODULE_SHA256,"weight_sha256":WEIGHT_SHA256,"bias_sha256":BIAS_SHA256,"input_sha256":INPUT_SHA256,"output_sha256":r["output_sha256"],"output_bytes":OUTPUT_BYTES,"rtx_backend_root_sha256":r["receipt_root_sha256"],"mac_backend_root_sha256":m["receipt_root_sha256"],"rtx_mac_differing_bytes":0,"rtx_mac_exact":True,"scope":"one custom three-plane convolution launch; no custom-family or connected-group claim","address_fields_recorded":False,"result":"pass"};v["receipt_root_sha256"]=canon(v);return v
def main()->int:
 p=argparse.ArgumentParser();s=p.add_subparsers(dest="cmd",required=True);b=s.add_parser("backend");b.add_argument("--backend",choices=("rtx","mac"),required=True);b.add_argument("--root",type=Path,required=True);b.add_argument("--output",type=Path,required=True);f=s.add_parser("final");f.add_argument("--rtx",type=Path,required=True);f.add_argument("--mac",type=Path,required=True);f.add_argument("--output",type=Path,required=True);a=p.parse_args()
 try:v=validate_backend(a.backend,a.root)if a.cmd=="backend"else validate_final(a.rtx,a.mac);write(a.output,v);print(f"receipt_root_sha256={v['receipt_root_sha256']}");print("result=pass");return 0
 except(OSError,json.JSONDecodeError,VerificationError)as e:print(f"verification_error={e}",file=sys.stderr);return 1
if __name__=="__main__":raise SystemExit(main())
