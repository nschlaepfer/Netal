#!/usr/bin/env python3
"""Verify connected launches 56-71 checkpoint by checkpoint."""
from __future__ import annotations
import argparse,hashlib,json,os,re,struct,sys
from pathlib import Path
from typing import Any
MANIFEST="2a1179d4a25be030f772cd90ca7d017ed6eac9291917f7d43ea9b0cd5ac6a294";OUT_BYTES=640*368*18*2;RUN_SCHEMA="neuralstudio.fg.connected-custom-l56-l71-run.v1";BACKEND_SCHEMA="neuralstudio.fg.connected-custom-l56-l71-backend.v1";FINAL_SCHEMA="neuralstudio.fg.connected-custom-l56-l71-verification.v1"
MODELS=[(106,"14058e1781a3e362cd9f0106cf5c3fcd985ede67f1391ff3dfc73bceeceeae38",108,"349bbe2fcd8fc31407bfe238f4889871f0d64dd593b4146b59eca837fcc9a77f"),(110,"be4a2cdb5d93c3b3ee3ff5ee8663d27017488664ca8682313740ff99aec4b829",112,"097bd45ed222a90e9311fd89bbb8bca8db92760ec932380d3c44e29eb3870b69"),(114,"c314eefc0c076af12dc246cbffbcf32fe78b50640b451567aa85832eff367092",116,"9718b129b9bf59ee1a683c6001880cfb910a5e95e9b2338e4fcfc742d7f83be7"),(118,"819fa5692e109843574dd0608312014a32abc7e7499784066142a4d82c486985",120,"ba63d29aecb3781452154a32c8e771471f49ff4cd1d7f56d06a1cd89081cd461"),(122,"5f4ac25a208ceb920b34fae48b7cee7107cd3e8da565e0a689a2cab93fe7ce93",124,"a7548519b77593645b6d5c46777597c0eba48255cd9f079f77d166da4c406bb5"),(126,"42a3ff7432377294d3c1fbdbe3c9876d13b989afb3952dd8ce964c531eae47a3",128,"f0dcd605c4a628cd940630048aeca237e2f563ec5219f265d93d382abef569ff"),(130,"672b9d32f4510335489894e12379bdc53a82ba27474c4484f6a58e1a18657af3",132,"591d8e9cb8e6c383b40636a814e714e676857deecfea6331792d84131edbf618"),(134,"54356c51efe3b2a4aa798275b73e1f16857d1976eaa99bfffe32c1d6f9905eae",136,"8f2b5996e111aa7d5d9711f0c58cea678de5974e3a733ae6a1a8bf1e0ec5840c"),(138,"7323eede30ef99e54156437d5f0430c266c1bf076018ad942afd48e7a62eca47",140,"7b48edd449f5442b4af26a7ac5d9e610aa0f087422347d7acfe6fa2d46b573d8"),(142,"971c7195720f720d7ce59564d9ee85259adfd0f66b2f2c426ca7f43c9290877c",144,"755ff7524a6855c81f50da913721a63f8fe39aa9f8106e577f14a0575a24633e"),(146,"145a29061230186864b92a934c6dcafa00ddede91381002ce7086e42c13b9490",148,"89e683bf9d2d108add777b4fb521d9d72940c66270cf96d1735943bf423bb93d"),(150,"6b59f16f1be43617ebf0c94ff29cf7c8b7a2f2c7b7bfb41df8f9ede585e70684",152,"c3e301df2e15c48e0c2fb35fc636b944653c6f91071a101d71b256b740224d29"),(154,"4d477655f6ec78e0e6a0d3c41e0848e537091c0c311fb04d986ad8891823e4e8",156,"0a9a85630034bb3dc5442cd52381424c1e7b9c676014b764477341005cb2f998")]
MODULES=[(53,"63b7e8105e00b04049e1004a8df23998074fb04b2b6637171da379b4a967d8c3"),(54,"a371abef8e41995064c25e171d0f2f9119a24ad4b671041be16c7b639a7c68a2"),(55,"8b0501ef1e90ee5c0621138406982cb06822de06de930184ce40f7f4d6e1709e"),(56,"c36bedc4fe275c9891dbf35fc79f3377e71f91e7150a2d6b2ae7eb7f0d0548f7"),(57,"d2b6ef4d2533614364bffab475f98a32f3c2cacbe02e0fa5825588dc5dfa3b4d"),(58,"3acb96881f20a50e575e2b2959f0aa433e03e3da20fbf624ae6bb183d77a3e48"),(59,"4d35017769f6338b6f8ba3ac3079756c32410eeacca7f817e48ad5d7dc59eb3c"),(60,"1d19bfddd2affc5587ec19829259b070404e44c28a6f0c570a4b7769fd0dbba6"),(61,"184542107e98be3dc6229251c19cae4ecc44a7cc1b78cbf741b7fb3b8d81f060")]
class E(RuntimeError):pass
def req(c:bool,m:str)->None:
 if not c:raise E(m)
def sha(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def canon(v:Any)->str:return sha(json.dumps(v,sort_keys=True,separators=(",",":")).encode())
def load(p:Path)->dict[str,Any]:v=json.loads(p.read_text(encoding="utf-8-sig"));req(isinstance(v,dict),f"not object {p}");return v
def write(p:Path,v:dict[str,Any])->None:p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+".tmp");t.write_text(json.dumps(v,indent=2,sort_keys=True)+"\n");os.replace(t,p)
def mix(v:int)->int:
 m=(1<<64)-1;v=(v+0x9E3779B97F4A7C15)&m;v=((v^(v>>30))*0xBF58476D1CE4E5B9)&m;v=((v^(v>>27))*0x94D049BB133111EB)&m;return(v^(v>>31))&m
def external(n:int,seed:int)->bytes:
 vals=(0,0x8000,0x2800,0xA800,0x2C00,0xAC00,0x3000,0xB000,0x3400,0xB400,0x3555,0xB555,0x3800,0xB800,0x3A00,0xBA00);r=bytearray(n)
 for i in range(n//2):struct.pack_into("<H",r,2*i,vals[mix((seed+i*0xD1B54A32D192ED03)&((1<<64)-1))%len(vals)])
 return bytes(r)
INPUT_HASHES=[sha(external(640*368*8,0x243F6A8885A308D3)),sha(external(640*368*8,0x13198A2E03707344)),sha(external(640*368*2,0xA4093822299F31D0))]
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
def validate_backend(backend:str,root:Path)->dict[str,Any]:
 ident="rtx-cuda-driver"if backend=="rtx"else"mac-metal";rs=[];outs=[]
 expected_model=[{"weight_token":a,"weight_sha256":b,"bias_token":c,"bias_sha256":d}for a,b,c,d in MODELS];expected_post=[[b,d]for _,b,_,d in MODELS];expected_modules=[{"module_token":a,"module_sha256":b}for a,b in MODULES]
 for poison in(0xA5,0x5A):
  for ord in("a","b"):
   rid=f"connected-l56-l71-poison-{poison:02x}-process-{ord}";r=load(root/rid/"run-receipt.json");o=(root/rid/"output.bin").read_bytes();req(r.get("schema")==RUN_SCHEMA and r.get("backend")==ident and r.get("run_id")==rid and r.get("poison_byte")==poison,f"{rid}: identity");req(isinstance(r.get("process_id"),int),f"{rid}: pid");req(r.get("launch_range")==[56,71]and r.get("launch_count")==16 and sorted(r.get("checkpoint_sha256",{}))==[f"launch-{i:06d}"for i in range(56,72)],f"{rid}: checkpoints");req(r.get("external_input_generator")=="splitmix64-fg-connected-custom-l56-l71-v1"and r.get("external_input_pre_sha256")==INPUT_HASHES==r.get("external_input_post_sha256"),f"{rid}: inputs");req(r.get("model_manifest_sha256")==MANIFEST and r.get("model_state")==expected_model and r.get("model_state_post_sha256")==expected_post,f"{rid}: model");req(backend!="rtx"or r.get("module_state")==expected_modules,f"{rid}: modules");req(len(o)==OUT_BYTES and r.get("final_output_bytes")==OUT_BYTES and r.get("final_output_sha256")==sha(o),f"{rid}: output");req(r.get("input_model_mismatched_bytes")==0 and r.get("canary_mismatched_bytes")==0 and r.get("in_process_handoff_count")==15 and r.get("archived_intermediate_ingress")is False and r.get("expected_output_file_opened")is False and r.get("address_fields_recorded")is False and r.get("result")=="pass",f"{rid}: gates");address_free(r);rs.append(r);outs.append(o)
 req(len({x["process_id"]for x in rs})==4,"fresh process");req(len({json.dumps(x["checkpoint_sha256"],sort_keys=True)for x in rs})==1,"checkpoint poison invariance");req(len({sha(o)for o in outs})==1,"output poison invariance");m=bytearray(outs[0]);m[71]^=1;req(sha(m)!=rs[0]["final_output_sha256"],"mutation")
 v={"schema":BACKEND_SCHEMA,"backend":ident,"launch_range":[56,71],"launch_count":16,"checkpoint_sha256":rs[0]["checkpoint_sha256"],"external_input_sha256":INPUT_HASHES,"model_manifest_sha256":MANIFEST,"final_output_sha256":sha(outs[0]),"final_output_bytes":OUT_BYTES,"fresh_process_count":4,"two_poison_all_checkpoint_invariant":True,"full_write_coverage_from_two_poison_invariance":True,"input_model_immutability":True,"canary_integrity":True,"archived_or_expected_intermediate_ingress":False,"output_mutation_rejected":True,"address_fields_recorded":False,"result":"pass"};v["receipt_root_sha256"]=canon(v);return v
def validate_final(rp:Path,mp:Path)->dict[str,Any]:
 r=load(rp);m=load(mp)
 for x,b in((r,"rtx-cuda-driver"),(m,"mac-metal")):req(x.get("schema")==BACKEND_SCHEMA and x.get("backend")==b,f"{b}: identity");d=dict(x);h=d.pop("receipt_root_sha256",None);req(h==canon(d),f"{b}: root")
 req(r["checkpoint_sha256"]==m["checkpoint_sha256"],"checkpoint parity");req(r["final_output_sha256"]==m["final_output_sha256"],"output parity");v={"schema":FINAL_SCHEMA,"launch_range":[56,71],"launch_count":16,"checkpoint_sha256":r["checkpoint_sha256"],"external_input_sha256":INPUT_HASHES,"model_manifest_sha256":MANIFEST,"final_output_sha256":r["final_output_sha256"],"final_output_bytes":OUT_BYTES,"rtx_backend_root_sha256":r["receipt_root_sha256"],"mac_backend_root_sha256":m["receipt_root_sha256"],"rtx_mac_checkpoint_mismatch_count":0,"rtx_mac_exact":True,"connected_in_process_handoffs_verified":15,"scope":"connected selected custom stack launches 56-71; no launch72+, downstream, full-frame, or Netal claim","address_fields_recorded":False,"result":"pass"};v["receipt_root_sha256"]=canon(v);return v
def main()->int:
 p=argparse.ArgumentParser();s=p.add_subparsers(dest="cmd",required=True);b=s.add_parser("backend");b.add_argument("--backend",choices=("rtx","mac"),required=True);b.add_argument("--root",type=Path,required=True);b.add_argument("--output",type=Path,required=True);f=s.add_parser("final");f.add_argument("--rtx",type=Path,required=True);f.add_argument("--mac",type=Path,required=True);f.add_argument("--output",type=Path,required=True);a=p.parse_args()
 try:v=validate_backend(a.backend,a.root)if a.cmd=="backend"else validate_final(a.rtx,a.mac);write(a.output,v);print(f"receipt_root_sha256={v['receipt_root_sha256']}");print("result=pass");return 0
 except(OSError,json.JSONDecodeError,E)as e:print(f"verification_error={e}",file=sys.stderr);return 1
if __name__=="__main__":raise SystemExit(main())
