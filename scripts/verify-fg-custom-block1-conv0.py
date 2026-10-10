#!/usr/bin/env python3
"""Verify selected custom block1 conv0 on RTX and Metal."""

from __future__ import annotations

import argparse, hashlib, json, os, re, struct, sys
from pathlib import Path
from typing import Any

MODULE_SHA256="5a33f08403b49bbabd4dfb04f476daaac8f9fa0964befc752dc4e7fd6d2471d6";WEIGHT_SHA256="255701f510034a1ec801207b0542c2d49919d131c35d1337a60ad16fdc9004e6";BIAS_SHA256="8ad9d15f85e8cea4e0a5b67792f39e170e75c7db010f727c7894e2c23be5d4c1";MANIFEST_SHA256="2a1179d4a25be030f772cd90ca7d017ed6eac9291917f7d43ea9b0cd5ac6a294";OUTPUT_BYTES=320*184*16*2;RUN_SCHEMA="neuralstudio.fg.custom-block1-conv0-run.v1";BACKEND_SCHEMA="neuralstudio.fg.custom-block1-conv0-backend.v1";FINAL_SCHEMA="neuralstudio.fg.custom-block1-conv0-verification.v1"
class VerificationError(RuntimeError):pass
def require(c:bool,m:str)->None:
    if not c:raise VerificationError(m)
def sha(data:bytes)->str:return hashlib.sha256(data).hexdigest()
def canon(v:Any)->str:return sha(json.dumps(v,sort_keys=True,separators=(",",":")).encode())
def load(path:Path)->dict[str,Any]:
    v=json.loads(path.read_text(encoding="utf-8-sig"));require(isinstance(v,dict),f"not object {path}");return v
def write(path:Path,v:dict[str,Any])->None:
    path.parent.mkdir(parents=True,exist_ok=True);t=path.with_suffix(path.suffix+".tmp");t.write_text(json.dumps(v,indent=2,sort_keys=True)+"\n");os.replace(t,path)
def address_free(v:Any,label:str)->None:
    def visit(n:Any,p:str)->None:
        if isinstance(n,dict):
            for k,c in n.items():
                if k!="address_fields_recorded":require(not any(x in k.lower() for x in ("pointer","virtual_address","device_address","gpu_address")),f"{label}: address key {p}.{k}")
                visit(c,f"{p}.{k}")
        elif isinstance(n,list):
            for i,c in enumerate(n):visit(c,f"{p}[{i}]")
        elif isinstance(n,str):require(re.search(r"0x[0-9a-fA-F]{12,16}(?:\b|$)",n)is None,f"{label}: address text")
    visit(v,label)
def splitmix(v:int)->int:
    mask=(1<<64)-1;v=(v+0x9E3779B97F4A7C15)&mask;v=((v^(v>>30))*0xBF58476D1CE4E5B9)&mask;v=((v^(v>>27))*0x94D049BB133111EB)&mask;return(v^(v>>31))&mask
def generated_input()->bytes:
    values=(0x0000,0x2800,0xA800,0x2C00,0xAC00,0x3000,0xB000,0x3400,0xB400,0x3800,0xB800);count=640*368*18;result=bytearray(count*2);seed=0xBE5466CF34E90C6C;step=0xD1B54A32D192ED03;mask=(1<<64)-1
    for i in range(count):struct.pack_into("<H",result,i*2,values[splitmix((seed+i*step)&mask)%len(values)])
    return bytes(result)
INPUT_SHA256=sha(generated_input())
def validate_backend(backend:str,root:Path)->dict[str,Any]:
    identity="rtx-cuda-driver" if backend=="rtx" else "mac-metal";receipts=[];outputs=[]
    for poison in(0xA5,0x5A):
        for ordinal in("a","b"):
            run_id=f"block1-conv0-poison-{poison:02x}-process-{ordinal}";r=load(root/run_id/"run-receipt.json");o=(root/run_id/"output.bin").read_bytes();require(r.get("schema")==RUN_SCHEMA and r.get("backend")==identity and r.get("run_id")==run_id and r.get("poison_byte")==poison,f"{run_id}: identity");require(isinstance(r.get("process_id"),int),f"{run_id}: pid");require(r.get("trace_launch_token")=="launch-000072" and r.get("module_token")==62 and r.get("module_bytes")==5096 and r.get("module_sha256")==MODULE_SHA256,f"{run_id}: module");require(r.get("packed_parameter_bytes")==80 and r.get("abi_scalars")==[16,3,3,18,1,368,640,18,1,184,320,16],f"{run_id}: ABI");require(r.get("input_shape_nhwc")==[1,368,640,18] and r.get("output_shape_nhwc")==[1,184,320,16],f"{run_id}: shape");require(r.get("fused_operations")==["conv3x3","leaky_relu","clamp_-6_6","average_pool_2x2"],f"{run_id}: fusion");require(r.get("weight_source_token")==158 and r.get("bias_source_token")==160 and r.get("model_capture_manifest_sha256")==MANIFEST_SHA256,f"{run_id}: provenance");require(r.get("weight_pre_sha256")==WEIGHT_SHA256==r.get("weight_post_sha256") and r.get("bias_pre_sha256")==BIAS_SHA256==r.get("bias_post_sha256"),f"{run_id}: model");require(r.get("input_generator")=="splitmix64-fg-custom-block1-conv0-v1" and r.get("input_pre_sha256")==INPUT_SHA256==r.get("input_post_sha256"),f"{run_id}: input");require(len(o)==OUTPUT_BYTES and r.get("output_bytes")==OUTPUT_BYTES and r.get("output_sha256")==sha(o),f"{run_id}: output");require(r.get("input_mismatched_bytes")==0 and r.get("canary_mismatched_bytes")==0 and r.get("per_run_write_coverage_conclusive")is False,f"{run_id}: guards");require(r.get("external_model_files_opened")is True and r.get("expected_output_file_opened")is False and r.get("address_fields_recorded")is False and r.get("result")=="pass",f"{run_id}: result")
            if backend=="rtx":require(r.get("grid")==[80,92,1] and r.get("block")==[32,2,2] and r.get("function_name")=="custom_block1_conv0_kernel",f"{run_id}: RTX")
            else:require(r.get("selected_rtx_grid")==[80,92,1] and r.get("selected_rtx_block")==[32,2,2] and r.get("metal_dispatch_threads")==[320,184,16] and r.get("metal_threads_per_threadgroup")==[8,4,1] and r.get("metal_function")=="fg_custom_block1_conv0",f"{run_id}: Metal")
            address_free(r,run_id);receipts.append(r);outputs.append(o)
    require(len({r["process_id"]for r in receipts})==4,"fresh process");require(len({sha(o)for o in outputs})==1,"two poison");m=bytearray(outputs[0]);m[57]^=1;require(sha(m)!=receipts[0]["output_sha256"],"mutation")
    result={"schema":BACKEND_SCHEMA,"backend":identity,"selected_unit":"evaluate-1/launch-000072/custom_block1_conv0_kernel/module-62","module_sha256":MODULE_SHA256,"weight_sha256":WEIGHT_SHA256,"bias_sha256":BIAS_SHA256,"model_manifest_sha256":MANIFEST_SHA256,"input_sha256":INPUT_SHA256,"output_sha256":sha(outputs[0]),"output_bytes":OUTPUT_BYTES,"fresh_process_count":4,"two_process_two_poison_invariant":True,"full_write_coverage_from_two_poison_invariance":True,"input_model_immutability":True,"canary_integrity":True,"expected_output_fed_to_execution":False,"output_mutation_rejected":True,"address_fields_recorded":False,"result":"pass"};result["receipt_root_sha256"]=canon(result);return result
def validate_final(rtx_path:Path,mac_path:Path)->dict[str,Any]:
    rtx=load(rtx_path);mac=load(mac_path)
    for r,b in((rtx,"rtx-cuda-driver"),(mac,"mac-metal")):
        require(r.get("schema")==BACKEND_SCHEMA and r.get("backend")==b,f"{b}: identity");x=dict(r);root=x.pop("receipt_root_sha256",None);require(root==canon(x),f"{b}: root");address_free(r,b)
    require(rtx["output_sha256"]==mac["output_sha256"],"parity")
    result={"schema":FINAL_SCHEMA,"selected_unit":rtx["selected_unit"],"module_sha256":MODULE_SHA256,"weight_sha256":WEIGHT_SHA256,"bias_sha256":BIAS_SHA256,"input_sha256":INPUT_SHA256,"output_sha256":rtx["output_sha256"],"output_bytes":OUTPUT_BYTES,"rtx_backend_root_sha256":rtx["receipt_root_sha256"],"mac_backend_root_sha256":mac["receipt_root_sha256"],"rtx_mac_differing_bytes":0,"rtx_mac_exact":True,"scope":"one custom fused convolution launch; no custom-family or connected-group claim","address_fields_recorded":False,"result":"pass"};result["receipt_root_sha256"]=canon(result);return result
def main()->int:
    p=argparse.ArgumentParser();s=p.add_subparsers(dest="command",required=True);b=s.add_parser("backend");b.add_argument("--backend",choices=("rtx","mac"),required=True);b.add_argument("--root",type=Path,required=True);b.add_argument("--output",type=Path,required=True);f=s.add_parser("final");f.add_argument("--rtx",type=Path,required=True);f.add_argument("--mac",type=Path,required=True);f.add_argument("--output",type=Path,required=True);a=p.parse_args()
    try:r=validate_backend(a.backend,a.root)if a.command=="backend"else validate_final(a.rtx,a.mac);write(a.output,r);print(f"receipt_root_sha256={r['receipt_root_sha256']}");print("result=pass");return 0
    except(OSError,json.JSONDecodeError,VerificationError)as e:print(f"verification_error={e}",file=sys.stderr);return 1
if __name__=="__main__":raise SystemExit(main())

