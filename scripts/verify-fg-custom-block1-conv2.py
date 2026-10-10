#!/usr/bin/env python3
import argparse,hashlib,json,os,re,struct,sys
from pathlib import Path
from typing import Any
MODULE="ae48122fca1d4494c10e3e1b27b53282e1998346dc6d6a1c61c2658bcbd0e474";MANIFEST="2a1179d4a25be030f772cd90ca7d017ed6eac9291917f7d43ea9b0cd5ac6a294";BYTES=92*160*32*2;RUN="neuralstudio.fg.custom-block1-conv2-run.v1";BACK="neuralstudio.fg.custom-block1-conv2-backend.v1";FINAL="neuralstudio.fg.custom-block1-conv2-verification.v1";FAMILY="neuralstudio.fg.custom-block1-conv2-family.v1"
LAYERS={"l74":(74,166,168,"bf39c57db6f20c38951cc91584a98543feb8f396c1529667d7541ce6d4732162","87d1d98747c14b03644c43ef21aae52ad19a3058d0f5ce673d954bc3c9d3ebeb",False),"l75":(75,170,172,"1ee5ffe7c824fec11aade50a433dd5c7d18e3f13452c84154c920f9b42e81653","4cf2f999a3b60e58553c0fa0e32c134a2de378ee22d6d4d0a0140cb2e640b882",True),"l76":(76,174,176,"a11cfbf3ae8b114b6e98f39fc59d11df69d9fc52741255130019cb7cd22f93b4","0898c1586b71e2ea5bbd4d9cbaad3e15f416743544446cc8d2157a4f23f628fa",False),"l77":(77,178,180,"21f1e5dcf02d33ad5eec044251148b5f851d264f86529a63633edb8a8ae2fe75","ff53999e00052d0f16117a358f65efddb1d36814cc8820edc3b0f56fca630a07",True),"l78":(78,182,184,"87b22102aab453ff63e8f2245a9ee9d9adce55b1b489d43a6246ae7b6b9d23fd","2d134dee471fda6e55f345531803da054d8edf9c6a0385a637843f5e39d0619f",False),"l79":(79,186,188,"f07e7bd536bbc5bd4dbb7e4d87c9575073d4fb8b3c119a11dae7b1cfbdce8170","07d8006bd5c7857575653d74815dc300ca71048a72f105165f0a2534b7fec2b0",True),"l80":(80,190,192,"d819d65c7f2582134c098b99b53105dd64c630554a072ef579a2d4bc0b51c713","715c6b649120b4d1449a0b8654d19101ab660ca3f490d8b827778c817cf9c30b",False),"l81":(81,194,196,"ab3a82644cdaca91b322abe6f835a63ef073005fceacf1cb8a1e66096c3bdac3","a301af6f017ef2324d205769e43c4327fbc9e9aa20dce506b2663acc0adca06e",True)}
class E(RuntimeError):pass
def req(c,m):
 if not c:raise E(m)
def sha(b):return hashlib.sha256(b).hexdigest()
def canon(v):return sha(json.dumps(v,sort_keys=True,separators=(",",":")).encode())
def load(p):v=json.loads(p.read_text(encoding="utf-8-sig"));req(isinstance(v,dict),"object");return v
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+".tmp");t.write_text(json.dumps(v,indent=2,sort_keys=True)+"\n");os.replace(t,p)
def mix(v):m=(1<<64)-1;v=(v+0x9E3779B97F4A7C15)&m;v=((v^(v>>30))*0xBF58476D1CE4E5B9)&m;v=((v^(v>>27))*0x94D049BB133111EB)&m;return(v^(v>>31))&m
def inp(layer,res):
 launch=LAYERS[layer][0];vals=(0,0x2800,0xA800,0x2C00,0xAC00,0x3000,0xB000,0x3400,0xB400,0x3800,0xB800);r=bytearray(BYTES);seed=(0x9E3779B97F4A7C15 if res else 0x243F6A8885A308D3)^(launch*0xA5A3564E27F8862B);m=(1<<64)-1
 for i in range(BYTES//2):struct.pack_into("<H",r,2*i,vals[mix((seed+i*0xD1B54A32D192ED03)&m)%len(vals)])
 return bytes(r)
def af(v):
 def f(n):
  if isinstance(n,dict):
   for k,x in n.items():req(k=="address_fields_recorded" or not any(q in k.lower()for q in("pointer","address")),"address");f(x)
  elif isinstance(n,list):
   for x in n:f(x)
  elif isinstance(n,str):req(re.search(r"0x[0-9a-fA-F]{12,16}",n)is None,"address text")
 f(v)
def backend(kind,layer,root):
 launch,wt,bt,wh,bh,res=LAYERS[layer];identity="rtx-cuda-driver"if kind=="rtx"else"mac-metal";ai=sha(inp(layer,False));ri=sha(inp(layer,True));receipts=[];outs=[]
 for poison in(0xA5,0x5A):
  for ord in("a","b"):
   rid=f"{layer}-poison-{poison:02x}-process-{ord}";r=load(root/rid/"run-receipt.json");o=(root/rid/"output.bin").read_bytes();req(r.get("schema")==RUN and r.get("backend")==identity and r.get("layer_id")==layer and r.get("run_id")==rid and r.get("poison_byte")==poison,rid);req(r.get("trace_launch_token")==f"launch-{launch:06d}" and r.get("module_token")==64 and r.get("module_bytes")==4896 and r.get("module_sha256")==MODULE,"module");req(r.get("packed_parameter_bytes")==92 and r.get("abi_scalars")==[32,3,3,32,1,92,160,32,1,92,160,32,int(res)],"abi");req(r.get("residual_enabled")is res and r.get("weight_source_token")==wt and r.get("bias_source_token")==bt and r.get("model_capture_manifest_sha256")==MANIFEST,"provenance");req(r.get("weight_pre_sha256")==wh==r.get("weight_post_sha256") and r.get("bias_pre_sha256")==bh==r.get("bias_post_sha256"),"model");req(r.get("input_pre_sha256")==ai==r.get("input_post_sha256") and r.get("residual_pre_sha256")==ri==r.get("residual_post_sha256"),"inputs");req(len(o)==BYTES and r.get("output_bytes")==BYTES and r.get("output_sha256")==sha(o),"output");req(r.get("input_mismatched_bytes")==0 and r.get("canary_mismatched_bytes")==0 and r.get("expected_output_file_opened")is False and r.get("address_fields_recorded")is False and r.get("result")=="pass","gates");
   if kind=="rtx":req(r.get("grid")==[20,23,1] and r.get("block")==[32,2,4],"rtx")
   else:req(r.get("metal_dispatch_threads")==[160,92,32] and r.get("metal_function")=="fg_custom_block1_conv2","metal")
   af(r);receipts.append(r);outs.append(o)
 req(len({r["process_id"]for r in receipts})==4 and len({sha(o)for o in outs})==1,"repeat");m=bytearray(outs[0]);m[launch]^=1;req(sha(m)!=receipts[0]["output_sha256"],"mutation");v={"schema":BACK,"backend":identity,"layer_id":layer,"trace_launch_token":f"launch-{launch:06d}","module_sha256":MODULE,"weight_sha256":wh,"bias_sha256":bh,"input_sha256":ai,"residual_sha256":ri,"residual_enabled":res,"output_sha256":sha(outs[0]),"output_bytes":BYTES,"fresh_process_count":4,"two_process_two_poison_invariant":True,"full_write_coverage_from_two_poison_invariance":True,"input_model_immutability":True,"canary_integrity":True,"output_mutation_rejected":True,"address_fields_recorded":False,"result":"pass"};v["receipt_root_sha256"]=canon(v);return v
def final(rp,mp,layer):
 r=load(rp);m=load(mp)
 for x,b in((r,"rtx-cuda-driver"),(m,"mac-metal")):y=dict(x);root=y.pop("receipt_root_sha256",None);req(x.get("backend")==b and root==canon(y),"root");af(x)
 req(r["output_sha256"]==m["output_sha256"],"parity");v={"schema":FINAL,"layer_id":layer,"module_sha256":MODULE,"weight_sha256":r["weight_sha256"],"bias_sha256":r["bias_sha256"],"input_sha256":r["input_sha256"],"residual_sha256":r["residual_sha256"],"residual_enabled":r["residual_enabled"],"output_sha256":r["output_sha256"],"output_bytes":BYTES,"rtx_backend_root_sha256":r["receipt_root_sha256"],"mac_backend_root_sha256":m["receipt_root_sha256"],"rtx_mac_differing_bytes":0,"rtx_mac_exact":True,"address_fields_recorded":False,"result":"pass"};v["receipt_root_sha256"]=canon(v);return v
def family(d):
 members=[]
 for l in LAYERS:v=load(d/f"{l}-verification.json");x=dict(v);root=x.pop("receipt_root_sha256");req(root==canon(x)and v.get("rtx_mac_exact")is True,"member");members.append({"layer_id":l,"receipt_root_sha256":root,"output_sha256":v["output_sha256"],"residual_enabled":v["residual_enabled"]})
 v={"schema":FAMILY,"layer_count":8,"layers":members,"every_launch_executed_independently":True,"every_launch_rtx_mac_exact":True,"residual_off_and_on_validated":True,"address_fields_recorded":False,"result":"pass"};v["receipt_root_sha256"]=canon(v);return v
def main():
 p=argparse.ArgumentParser();s=p.add_subparsers(dest="cmd",required=True);b=s.add_parser("backend");b.add_argument("--backend",choices=("rtx","mac"),required=True);b.add_argument("--layer",choices=LAYERS,required=True);b.add_argument("--root",type=Path,required=True);b.add_argument("--output",type=Path,required=True);f=s.add_parser("final");f.add_argument("--layer",choices=LAYERS,required=True);f.add_argument("--rtx",type=Path,required=True);f.add_argument("--mac",type=Path,required=True);f.add_argument("--output",type=Path,required=True);g=s.add_parser("family");g.add_argument("--dir",type=Path,required=True);g.add_argument("--output",type=Path,required=True);a=p.parse_args()
 try:v=backend(a.backend,a.layer,a.root)if a.cmd=="backend"else(final(a.rtx,a.mac,a.layer)if a.cmd=="final"else family(a.dir));write(a.output,v);print(f"receipt_root_sha256={v['receipt_root_sha256']}");print("result=pass");return 0
 except(Exception,E)as e:print(f"verification_error={e}",file=sys.stderr);return 1
if __name__=="__main__":raise SystemExit(main())

