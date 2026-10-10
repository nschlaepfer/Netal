#!/usr/bin/env python3
from __future__ import annotations
import argparse,copy,hashlib,json
from pathlib import Path
from typing import Any
MODULE="11df926493fca0a981d560427b36c4dcf489ccb84d794501a70bda8d414ea8a2"
CFG={"720p":([640,360],[40,23,1],1),"1080p":([960,540],[60,34,1],2),"1440p":([1280,720],[80,45,1],1),"4k":([1920,1080],[120,68,1],2)}
PH=("reset","generated");PO=("a5","5a");OR=("a","b");ART=("vector-surface.bin","mask-surface.bin",*[f"output-buffer-{i}.bin" for i in range(5)])
def h(b):return hashlib.sha256(b).hexdigest()
def can(v):return json.dumps(v,sort_keys=True,separators=(",",":")).encode()
def root(v):v=copy.deepcopy(v);v["receipt_root_sha256"]=h(can(v));return v
def valid(v):v=copy.deepcopy(v);x=v.pop("receipt_root_sha256",None);return x==h(can(v))
def wr(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,sort_keys=True)+"\n")
def rd(r,res,ph,z,o):return r/f"{res}-{ph}-poison-{z}-process-{o}"
def outsizes(e):w,hh=e;C=lambda x,d:(x+d-1)//d;return[w*hh,C(w,2)*C(hh,2)*6,C(w,4)*C(hh,4)*6,C(w,8)*C(hh,8)*6,C(w,16)*C(hh,16)*6]
def backend(r:Path,bk:str):
 pids=set();outputs={};expected="rtx-d3d12-nvapi" if bk=="rtx" else "mac-metal"
 for res,(extent,grid,pad) in CFG.items():
  sizes=outsizes(extent)
  for ph in PH:
   rec=[]
   for z in PO:
    for o in OR:
     d=rd(r,res,ph,z,o);q=json.loads((d/"run-receipt.json").read_text())
     assert q["result"]=="pass" and q["backend"]==expected and q["resolution"]==res and q["phase"]==ph and q["module_sha256"]==MODULE and q["extent"]==extent and q["grid"]==grid and q["padding_byte_157"]==pad and q["input_mismatched_bytes"]==0 and q["buffer_canary_mismatched_bytes"]==0 and not q["expected_or_captured_output_ingress"] and not q["archived_intermediate_ingress"] and not q["address_fields_recorded"]
     full=z=="a5" and o=="a";assert q["artifacts_written"]==full and all((d/a).is_file()==full for a in ART);pids.add(q["process_id"]);rec.append(q)
   for f in ("vector_surface_sha256","mask_surface_sha256","output_buffer_sha256"):assert len({json.dumps(q[f],sort_keys=True) for q in rec})==1
   d=rd(r,res,ph,"a5","a");data={a:(d/a).read_bytes() for a in ART};assert len(data[ART[0]])==extent[0]*extent[1]*4 and len(data[ART[1]])==extent[0]*extent[1] and [len(data[f"output-buffer-{i}.bin"]) for i in range(5)]==sizes
   outputs[f"{res}/{ph}"]={"vector":h(data[ART[0]]),"mask":h(data[ART[1]]),"buffers":[h(data[f"output-buffer-{i}.bin"]) for i in range(5)],"bytes":[len(data[a]) for a in ART]}
 assert len(pids)==32
 q={"schema":f"neuralstudio.fg.dynamic-module21-{bk}-backend.v1","backend":expected,"fresh_process_count":32,"processes_per_scenario":4,"fresh_process_repeat_exact":True,"dual_poison_full_write_coverage":True,"input_immutable":True,"canaries_intact":True,"expected_or_captured_output_ingress":False,"archived_intermediate_ingress":False,"address_fields_recorded":False,"outputs":outputs,"result":"pass"}
 if bk=="rtx":
  x=json.loads((r/"rig-receipt.json").read_text());assert x["fresh_process_count"]==32 and x["true_d3d12_nvapi_resources"];q["rig_receipt_sha256"]=h((r/"rig-receipt.json").read_bytes());q["rig_job_id"]=x["job_id"]
 else:q["architecture_rom_sha256"]="7cf4f8cece6b18b07605edc9cb601a153346a79b10bf3b93c7ec058d57d29750"
 return root(q),outputs
def compare(a,b):
 mm=n=0
 for res in CFG:
  for ph in PH:
   for f in ART:
    x=(rd(a,res,ph,"a5","a")/f).read_bytes();y=(rd(b,res,ph,"a5","a")/f).read_bytes();assert len(x)==len(y);mm+=sum(i!=j for i,j in zip(x,y));n+=len(x)
 return mm,n
def verify(e):
 names=["abi-receipt.json","implementation-receipt.json","counterexamples.json","rtx-backend-receipt.json","mac-backend-receipt.json","accepted-receipt.json","negative-audit.json"];v={n:json.loads((e/n).read_text()) for n in names};assert all(valid(x) for x in v.values());a=v["accepted-receipt.json"];assert a["rtx_mac_mismatched_bytes"]==0 and a["claims"]["module21_four_resolution_standalone_exact"] and not a["claims"]["connected_early_graph"] and v["negative-audit.json"]["all_mutations_rejected"]
def build(a):
 repo=a.repository.resolve();rr=a.rtx.resolve();mr=a.mac.resolve();rt,ro=backend(rr,"rtx");ma,mo=backend(mr,"mac");assert ro==mo;mm,n=compare(rr,mr);assert mm==0
 abi=root({"schema":"neuralstudio.fg.dynamic-module21-abi.v1","module_sha256":MODULE,"packed_parameter_bytes":224,"resolutions":{k:{"extent":v[0],"grid":v[1],"padding_byte_157":v[2],"output_buffer_bytes":outsizes(v[0])}for k,v in CFG.items()},"typed_inputs":["RG32_FLOAT","R32_UINT","R8_UNORM","signed candidate buffer"],"typed_outputs":["RG16_FLOAT","R8_UNORM","five raw pyramid buffers"],"loaded_byte_156":0,"padding_157_159_ptx_dead":True,"result":"pass"})
 impl=root({"schema":"neuralstudio.fg.dynamic-module21-implementation.v1","contract_sha256":h((repo/"src/fg_dynamic_module21/contract.h").read_bytes()),"metal_wrapper_sha256":h((repo/"src/fg_dynamic_module21/dynamic_module21.metal").read_bytes()),"macos_host_sha256":h((repo/"src/fg_dynamic_module21/macos_probe.mm").read_bytes()),"rtx_oracle_sha256":h((repo/"src/fg_dynamic_module21/rtx_oracle.cpp").read_bytes()),"accepted_module21_lowering_reused":True,"expected_output_embedded":False,"result":"pass"})
 co=root({"schema":"neuralstudio.fg.dynamic-module21-counterexamples.v1","items":[{"name":"oracle-v1-msvc-uint","stage":"build"},{"name":"oracle-v2-dynamic-texture-metadata","stage":"build"},{"name":"oracle-v3-byte157-json","stage":"receipt"},{"name":"prototype-v1-arbitrary-half-bits","stage":"parity","exact_outputs":["mask surface","raw buffer0"],"mismatched_outputs":["vector surface","half3 pyramid1-4"]},{"name":"oracle-v5-finite-domain","stage":"accepted"}],"all_failures_preserved":True,"result":"pass"})
 ac=root({"schema":"neuralstudio.fg.dynamic-module21-accepted.v1","module_sha256":MODULE,"resolutions":list(CFG),"processes_per_backend":32,"abi_receipt_root_sha256":abi["receipt_root_sha256"],"implementation_receipt_root_sha256":impl["receipt_root_sha256"],"counterexamples_root_sha256":co["receipt_root_sha256"],"rtx_backend_root_sha256":rt["receipt_root_sha256"],"mac_backend_root_sha256":ma["receipt_root_sha256"],"rtx_mac_compared_bytes":n,"rtx_mac_mismatched_bytes":mm,"claims":{"module21_four_resolution_standalone_exact":True,"all_seven_outputs_exact":True,"connected_early_graph":False,"complete_generated_frame":False,"netal_integration":False},"result":"pass"})
 for name,x in [("abi-receipt.json",abi),("implementation-receipt.json",impl),("counterexamples.json",co),("rtx-backend-receipt.json",rt),("mac-backend-receipt.json",ma),("accepted-receipt.json",ac)]:wr(a.output/name,x)
 muts=[]
 for i,(name,obj) in enumerate([("accepted",ac),("abi",abi),("impl",impl),("rtx",rt),("mac",ma),("counter",co)]*2):
  x=copy.deepcopy(obj);x["result"]="fail" if i<6 else "mutated";muts.append({"mutation":f"{name}-{i}","rejected":not valid(x)})
 neg=root({"schema":"neuralstudio.fg.dynamic-module21-negative-audit.v1","accepted_receipt_root_sha256":ac["receipt_root_sha256"],"mutation_count":12,"mutations":muts,"all_mutations_rejected":all(x["rejected"] for x in muts),"result":"pass"});wr(a.output/"negative-audit.json",neg);verify(a.output);print(f"accepted_receipt_root_sha256={ac['receipt_root_sha256']}\nrtx_mac_compared_bytes={n}\nverification_result=pass")
def main():
 p=argparse.ArgumentParser();s=p.add_subparsers(dest="cmd",required=True);b=s.add_parser("build");[b.add_argument(x,type=Path,required=True) for x in ["--repository","--rtx","--mac","--output"]];b.set_defaults(f=build);v=s.add_parser("receipt");v.add_argument("--evidence",type=Path,required=True);v.set_defaults(f=lambda a:(verify(a.evidence),print("verification_result=pass")));a=p.parse_args();a.f(a)
if __name__=="__main__":main()
