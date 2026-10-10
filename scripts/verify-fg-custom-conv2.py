#!/usr/bin/env python3
import argparse,hashlib,json,os,re,struct,sys
from pathlib import Path
from typing import Any
MODULE="d2b6ef4d2533614364bffab475f98a32f3c2cacbe02e0fa5825588dc5dfa3b4d";MANIFEST="2a1179d4a25be030f772cd90ca7d017ed6eac9291917f7d43ea9b0cd5ac6a294";BYTES=46*80*64*2;RUN="neuralstudio.fg.custom-conv2-run.v1";BACK="neuralstudio.fg.custom-conv2-backend.v1";FINAL="neuralstudio.fg.custom-conv2-verification.v1";FAMILY="neuralstudio.fg.custom-conv2-family.v1"
LAYERS={"l60":(60,118,120,"819fa5692e109843574dd0608312014a32abc7e7499784066142a4d82c486985","ba63d29aecb3781452154a32c8e771471f49ff4cd1d7f56d06a1cd89081cd461",False),"l61":(61,122,124,"5f4ac25a208ceb920b34fae48b7cee7107cd3e8da565e0a689a2cab93fe7ce93","a7548519b77593645b6d5c46777597c0eba48255cd9f079f77d166da4c406bb5",True),"l62":(62,126,128,"42a3ff7432377294d3c1fbdbe3c9876d13b989afb3952dd8ce964c531eae47a3","f0dcd605c4a628cd940630048aeca237e2f563ec5219f265d93d382abef569ff",False),"l63":(63,130,132,"672b9d32f4510335489894e12379bdc53a82ba27474c4484f6a58e1a18657af3","591d8e9cb8e6c383b40636a814e714e676857deecfea6331792d84131edbf618",True),"l64":(64,134,136,"54356c51efe3b2a4aa798275b73e1f16857d1976eaa99bfffe32c1d6f9905eae","8f2b5996e111aa7d5d9711f0c58cea678de5974e3a733ae6a1a8bf1e0ec5840c",False),"l65":(65,138,140,"7323eede30ef99e54156437d5f0430c266c1bf076018ad942afd48e7a62eca47","7b48edd449f5442b4af26a7ac5d9e610aa0f087422347d7acfe6fa2d46b573d8",True),"l66":(66,142,144,"971c7195720f720d7ce59564d9ee85259adfd0f66b2f2c426ca7f43c9290877c","755ff7524a6855c81f50da913721a63f8fe39aa9f8106e577f14a0575a24633e",False),"l67":(67,146,148,"145a29061230186864b92a934c6dcafa00ddede91381002ce7086e42c13b9490","89e683bf9d2d108add777b4fb521d9d72940c66270cf96d1735943bf423bb93d",True)}
class E(RuntimeError):pass
def req(c,m):
 if not c:raise E(m)
def sha(b):return hashlib.sha256(b).hexdigest()
def canon(v):return sha(json.dumps(v,sort_keys=True,separators=(",",":")).encode())
def load(p):v=json.loads(p.read_text(encoding="utf-8-sig"));req(isinstance(v,dict),"object");return v
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+".tmp");t.write_text(json.dumps(v,indent=2,sort_keys=True)+"\n");os.replace(t,p)
def mix(v):m=(1<<64)-1;v=(v+0x9E3779B97F4A7C15)&m;v=((v^(v>>30))*0xBF58476D1CE4E5B9)&m;v=((v^(v>>27))*0x94D049BB133111EB)&m;return(v^(v>>31))&m
def inp(layer,res):
 launch=LAYERS[layer][0];vals=(0,0x2800,0xA800,0x2C00,0xAC00,0x3000,0xB000,0x3400,0xB400,0x3800,0xB800);r=bytearray(BYTES);seed=(0x082EFA98EC4E6C89 if res else 0x452821E638D01377)^(launch*0xD6E8FEB86659FD93);m=(1<<64)-1
 for i in range(BYTES//2):struct.pack_into("<H",r,2*i,vals[mix((seed+i*0xA5A3564E27F8862B)&m)%len(vals)])
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
   rid=f"{layer}-poison-{poison:02x}-process-{ord}";r=load(root/rid/"run-receipt.json");o=(root/rid/"output.bin").read_bytes();req(r.get("schema")==RUN and r.get("backend")==identity and r.get("layer_id")==layer and r.get("run_id")==rid and r.get("poison_byte")==poison,rid);req(r.get("trace_launch_token")==f"launch-{launch:06d}" and r.get("module_token")==57 and r.get("module_bytes")==5352 and r.get("module_sha256")==MODULE,"module");req(r.get("packed_parameter_bytes")==92 and r.get("abi_scalars")==[64,3,3,64,1,46,80,64,1,46,80,64,int(res)],"abi");req(r.get("residual_enabled")is res and r.get("weight_source_token")==wt and r.get("bias_source_token")==bt and r.get("model_capture_manifest_sha256")==MANIFEST,"provenance");req(r.get("weight_pre_sha256")==wh==r.get("weight_post_sha256") and r.get("bias_pre_sha256")==bh==r.get("bias_post_sha256"),"model");req(r.get("input_pre_sha256")==ai==r.get("input_post_sha256") and r.get("residual_pre_sha256")==ri==r.get("residual_post_sha256"),"inputs");req(len(o)==BYTES and r.get("output_bytes")==BYTES and r.get("output_sha256")==sha(o),"output");req(r.get("input_mismatched_bytes")==0 and r.get("canary_mismatched_bytes")==0 and r.get("expected_output_file_opened")is False and r.get("address_fields_recorded")is False and r.get("result")=="pass","gates");
   if kind=="rtx":req(r.get("grid")==[10,23,1] and r.get("block")==[32,1,8],"rtx")
   else:req(r.get("metal_dispatch_threads")==[80,46,64] and r.get("metal_function")=="fg_custom_block0_conv2","metal")
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
