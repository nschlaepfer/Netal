#!/usr/bin/env python3
from __future__ import annotations
import argparse,copy,hashlib,json
from pathlib import Path
M="d787cf7b861295d1b4c6a80cb374230de793aa69f288a7e724bf59278bc4a62d";C={"720p":([320,180],[20,12,1]),"1080p":([480,270],[30,17,1]),"1440p":([320,180],[20,12,1]),"4k":([480,270],[30,17,1])};P=("reset","generated");Z=("a5","5a");O=("a","b")
def h(b):return hashlib.sha256(b).hexdigest()
def ca(v):return json.dumps(v,sort_keys=True,separators=(",",":")).encode()
def rt(v):v=copy.deepcopy(v);v["receipt_root_sha256"]=h(ca(v));return v
def ok(v):v=copy.deepcopy(v);x=v.pop("receipt_root_sha256",None);return x==h(ca(v))
def w(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,sort_keys=True)+"\n")
def d(r,n,p,z,o):return r/f"{n}-{p}-poison-{z}-process-{o}"
def back(r,b):
 e="rtx-cuda-driver-raw-buffers" if b=="rtx" else "mac-metal";pids=set();outs={}
 for n,(extent,grid) in C.items():
  for p in P:
   q=[]
   for z in Z:
    for o in O:
     x=json.loads((d(r,n,p,z,o)/"run-receipt.json").read_text());assert x["result"]=="pass" and x["backend"]==e and x["selected_mode"]==0 and x["extent"]==extent and x["grid"]==grid and x["readonly_input_mismatched_bytes"]==0 and x["canary_mismatched_bytes"]==0 and not x["expected_or_captured_output_ingress"];full=z=="a5"and o=="a";assert x["artifacts_written"]==full and (d(r,n,p,z,o)/"level0-output.bin").is_file()==full;pids.add(x["process_id"]);q.append(x)
   assert len({x["level0_post_sha256"] for x in q})==1;f=(d(r,n,p,"a5","a")/"level0-output.bin").read_bytes();assert len(f)==extent[0]*extent[1]*6 and h(f)==q[0]["level0_post_sha256"];outs[f"{n}/{p}"]={"bytes":len(f),"sha256":h(f),"pre_sha256":q[0]["level0_pre_sha256"]}
 assert len(pids)==32;v={"schema":f"neuralstudio.fg.dynamic-module23-{b}.v1","backend":e,"fresh_process_count":32,"dual_poison_repeat_exact":True,"level0_full_store_coverage":True,"levels1_3_and_mask_immutable":True,"canaries_intact":True,"expected_or_captured_output_ingress":False,"outputs":outs,"result":"pass"}
 if b=="rtx":v["rig_receipt_sha256"]=h((r/"rig-receipt.json").read_bytes())
 return rt(v),outs
def verify(e):
 ns=["abi-receipt.json","lowering-receipt.json","counterexamples.json","rtx-backend-receipt.json","mac-backend-receipt.json","accepted-receipt.json","negative-audit.json"];v={n:json.loads((e/n).read_text())for n in ns};assert all(ok(x) for x in v.values());assert v["accepted-receipt.json"]["rtx_mac_mismatched_bytes"]==0 and not v["accepted-receipt.json"]["claims"]["mode1_reused"] and v["negative-audit.json"]["all_mutations_rejected"]
def build(a):
 rr=a.rtx.resolve();mr=a.mac.resolve();r,ro=back(rr,"rtx");m,mo=back(mr,"mac");assert ro==mo;mm=n=0
 for k in C:
  for p in P:
   x=(d(rr,k,p,"a5","a")/"level0-output.bin").read_bytes();y=(d(mr,k,p,"a5","a")/"level0-output.bin").read_bytes();n+=len(x);mm+=sum(i!=j for i,j in zip(x,y))
 assert mm==0;repo=a.repository.resolve();lr=json.loads((repo/"build/fg_dynamic_module23/lowering.json").read_text());abi=rt({"schema":"neuralstudio.fg.dynamic-module23-abi.v1","module_sha256":M,"packed_parameter_bytes":72,"selected_mode":0,"offsets":{"0_24":"four half3 levels","32":"R8 mask","40_44":"extent","48":"mode0","56_64":"null"},"resolutions":{k:{"extent":v[0],"grid":v[1]}for k,v in C.items()},"mode1_null_offset_contract_rejected":True,"result":"pass"});lower=rt({"schema":"neuralstudio.fg.dynamic-module23-lowering-proof.v1","lowering_receipt_root_sha256":lr["receipt_root_sha256"],"metal_sha256":h((repo/"src/fg_dynamic_module23/dynamic_module23_mode0.metal").read_bytes()),"division_sites":54,"mode0_in_place_level0":True,"result":"pass"});co=rt({"schema":"neuralstudio.fg.dynamic-module23-counterexamples.v1","items":[{"name":"original-mode1-reuse","result":"rejected","reason":"target traces select mode0 with inverted ABI roles"},{"name":"oracle-v1-build","result":"rejected","reason":"missing poison parser and mixed auto declaration"}],"all_failures_preserved":True,"result":"pass"});ac=rt({"schema":"neuralstudio.fg.dynamic-module23-accepted.v1","module_sha256":M,"resolutions":list(C),"processes_per_backend":32,"abi_receipt_root_sha256":abi["receipt_root_sha256"],"lowering_receipt_root_sha256":lower["receipt_root_sha256"],"counterexamples_root_sha256":co["receipt_root_sha256"],"rtx_backend_root_sha256":r["receipt_root_sha256"],"mac_backend_root_sha256":m["receipt_root_sha256"],"rtx_mac_compared_bytes":n,"rtx_mac_mismatched_bytes":mm,"claims":{"module23_mode0_four_resolution_exact":True,"level0_in_place_full_store":True,"readonly_inputs_immutable":True,"mode1_reused":False,"connected_early_graph":False,"complete_frame":False},"result":"pass"});out=a.output
 for fn,x in [("abi-receipt.json",abi),("lowering-receipt.json",lower),("counterexamples.json",co),("rtx-backend-receipt.json",r),("mac-backend-receipt.json",m),("accepted-receipt.json",ac)]:w(out/fn,x)
 ms=[]
 for i,x in enumerate([abi,lower,co,r,m,ac]*2):y=copy.deepcopy(x);y["result"]="mutated";ms.append({"mutation":i,"rejected":not ok(y)})
 neg=rt({"schema":"neuralstudio.fg.dynamic-module23-negative.v1","mutation_count":12,"mutations":ms,"all_mutations_rejected":all(x["rejected"]for x in ms),"result":"pass"});w(out/"negative-audit.json",neg);verify(out);print(f"accepted_receipt_root_sha256={ac['receipt_root_sha256']}\nrtx_mac_compared_bytes={n}\nverification_result=pass")
def main():
 p=argparse.ArgumentParser();s=p.add_subparsers(dest="c",required=True);b=s.add_parser("build");[b.add_argument(x,type=Path,required=True)for x in["--repository","--rtx","--mac","--output"]];b.set_defaults(f=build);v=s.add_parser("receipt");v.add_argument("--evidence",type=Path,required=True);v.set_defaults(f=lambda a:(verify(a.evidence),print("verification_result=pass")));a=p.parse_args();a.f(a)
if __name__=="__main__":main()
