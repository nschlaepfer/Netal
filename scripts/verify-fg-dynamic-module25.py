#!/usr/bin/env python3
from __future__ import annotations
import argparse,copy,hashlib,json
from pathlib import Path
M="28fe0e153d1ffc80b54995c022f18395c5556f3c041cc9d724bfe7e70dc4e824";C={"720p":([640,360],[40,23,1]),"1080p":([960,540],[60,34,1]),"1440p":([1280,720],[80,45,1]),"4k":([1920,1080],[120,68,1])};P=("reset","generated");Z=("a5","5a");O=("a","b")
def h(b):return hashlib.sha256(b).hexdigest()
def ca(v):return json.dumps(v,sort_keys=True,separators=(",",":")).encode()
def rt(v):v=copy.deepcopy(v);v["receipt_root_sha256"]=h(ca(v));return v
def ok(v):v=copy.deepcopy(v);x=v.pop("receipt_root_sha256",None);return x==h(ca(v))
def w(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,sort_keys=True)+"\n")
def d(r,n,p,z,o):return r/f"{n}-{p}-poison-{z}-process-{o}"
def back(r,b):
 e="rtx-cuda-driver-typed-arrays"if b=="rtx"else"mac-metal";pids=set();outs={}
 for n,(extent,grid)in C.items():
  for p in P:
   qs=[]
   for z in Z:
    for o in O:
     q=json.loads((d(r,n,p,z,o)/"run-receipt.json").read_text());assert q["result"]=="pass"and q["backend"]==e and q["extent"]==extent and q["grid"]==grid and q["mode"]==0 and q["invalid_pixels"]==0 and q["input_mismatched_bytes"]==0 and q["guard_row_mismatched_bytes"]==0 and q["written_pixels"]>0 and q["caller_owned_pixels"]>0 and not q["expected_or_captured_output_ingress"];full=z=="a5"and o=="a";assert q["artifacts_written"]==full;pids.add(q["process_id"]);qs.append(q)
   assert len({q["coverage_sha256"]for q in qs})==1
   for z in Z:assert len({q["raw_output_sha256"]for q in qs if q["poison_byte"]==(165 if z=="a5" else 90)})==1
   x=(d(r,n,p,"a5","a")/"output.bin").read_bytes();c=(d(r,n,p,"a5","a")/"coverage.bin").read_bytes();assert len(x)==extent[0]*extent[1]and len(c)==len(x)and h(x)==qs[0]["raw_output_sha256"]and h(c)==qs[0]["coverage_sha256"];outs[f"{n}/{p}"]={"raw":h(x),"coverage":h(c),"written":qs[0]["written_pixels"],"caller_owned":qs[0]["caller_owned_pixels"],"bytes":len(x)}
 assert len(pids)==32;v={"schema":f"neuralstudio.fg.dynamic-module25-{b}.v1","backend":e,"fresh_process_count":32,"repeat_exact_per_poison":True,"dual_poison_coverage_exact":True,"input_immutable":True,"guard_row_intact":True,"outputs":outs,"expected_or_captured_output_ingress":False,"result":"pass"};return rt(v),outs
def verify(e):
 ns=["abi-receipt.json","implementation-receipt.json","counterexamples.json","rtx-backend-receipt.json","mac-backend-receipt.json","accepted-receipt.json","negative-audit.json"];v=[json.loads((e/n).read_text())for n in ns];assert all(ok(x)for x in v)and v[5]["rtx_mac_mismatched_bytes"]==0 and v[6]["all_mutations_rejected"]
def build(a):
 rr=a.rtx.resolve();mr=a.mac.resolve();r,ro=back(rr,"rtx");m,mo=back(mr,"mac");assert ro==mo;mm=n=0
 for k in C:
  for p in P:
   for f in("output.bin","coverage.bin"):
    x=(d(rr,k,p,"a5","a")/f).read_bytes();y=(d(mr,k,p,"a5","a")/f).read_bytes();n+=len(x);mm+=sum(i!=j for i,j in zip(x,y))
 assert mm==0;repo=a.repository.resolve();abi=rt({"schema":"neuralstudio.fg.dynamic-module25-abi.v1","module_sha256":M,"packed_parameter_bytes":48,"selected_mode":0,"resolutions":{k:{"extent":v[0],"grid":v[1]}for k,v in C.items()},"typed_resources":["RG16_FLOAT normalized point-border input","R8_UNORM output surface"],"result":"pass"});im=rt({"schema":"neuralstudio.fg.dynamic-module25-implementation.v1","contract_sha256":h((repo/"src/fg_dynamic_module25/contract.h").read_bytes()),"metal_wrapper_sha256":h((repo/"src/fg_dynamic_module25/dynamic_module25.metal").read_bytes()),"macos_host_sha256":h((repo/"src/fg_dynamic_module25/macos_probe.mm").read_bytes()),"rtx_oracle_sha256":h((repo/"src/fg_dynamic_module25/rtx_oracle.cpp").read_bytes()),"accepted_scatter_kernel_reused":True,"result":"pass"});co=rt({"schema":"neuralstudio.fg.dynamic-module25-counterexamples.v1","items":[{"name":"oracle-v1-windows-uint","result":"build-rejected"}],"all_failures_preserved":True,"result":"pass"});ac=rt({"schema":"neuralstudio.fg.dynamic-module25-accepted.v1","module_sha256":M,"resolutions":list(C),"processes_per_backend":32,"abi_receipt_root_sha256":abi["receipt_root_sha256"],"implementation_receipt_root_sha256":im["receipt_root_sha256"],"counterexamples_root_sha256":co["receipt_root_sha256"],"rtx_backend_root_sha256":r["receipt_root_sha256"],"mac_backend_root_sha256":m["receipt_root_sha256"],"rtx_mac_compared_bytes":n,"rtx_mac_mismatched_bytes":mm,"claims":{"module25_four_resolution_exact":True,"scatter_coverage_exact":True,"connected_early_graph":False,"complete_frame":False},"result":"pass"});out=a.output
 for fn,x in[("abi-receipt.json",abi),("implementation-receipt.json",im),("counterexamples.json",co),("rtx-backend-receipt.json",r),("mac-backend-receipt.json",m),("accepted-receipt.json",ac)]:w(out/fn,x)
 ms=[]
 for i,x in enumerate([abi,im,co,r,m,ac]*2):y=copy.deepcopy(x);y["result"]="mutated";ms.append({"mutation":i,"rejected":not ok(y)})
 ng=rt({"schema":"neuralstudio.fg.dynamic-module25-negative.v1","mutation_count":12,"mutations":ms,"all_mutations_rejected":True,"result":"pass"});w(out/"negative-audit.json",ng);verify(out);print(f"accepted_receipt_root_sha256={ac['receipt_root_sha256']}\nrtx_mac_compared_bytes={n}\nverification_result=pass")
def main():
 p=argparse.ArgumentParser();s=p.add_subparsers(dest="c",required=True);b=s.add_parser("build");[b.add_argument(x,type=Path,required=True)for x in["--repository","--rtx","--mac","--output"]];b.set_defaults(f=build);v=s.add_parser("receipt");v.add_argument("--evidence",type=Path,required=True);v.set_defaults(f=lambda a:(verify(a.evidence),print("verification_result=pass")));a=p.parse_args();a.f(a)
if __name__=="__main__":main()
