#!/usr/bin/env python3
from __future__ import annotations
import argparse, copy, hashlib, json, struct
from pathlib import Path
from typing import Any, Callable

RUN_SCHEMA="neuralstudio.fg.early-l19-run.v1"; BACKEND_SCHEMA="neuralstudio.fg.early-l19-backend.v1"; FINAL_SCHEMA="neuralstudio.fg.early-l19-verification.v1"
MODULE="3f2308a693730b44c00e064ed25dab3b73a420f2e92d215ef733c4e10a30439c"; GENERATOR="fg-early-l19-independent-ten-inputs-v1"
RAW="6f8656c06f9d30222e3d26cefb0594433d386eaa79bba79eef95914fbe2314d5"; FROZEN="9211cb3fa6156db90144c2d845c4ad0d54b0592f7d057d56b43a2cae6c5eb07d"; FIXTURE_PARAM="5dd08659e6c4f5724657768ea8da24eb0d1a61431d64bd9b9df2e69b997cf890"
FIRST_ROOT="c1435760ec194050aa1dd4dfd2d82f71e9aa7874ce550328a4eb08cd3d2a0018"; TEMPORAL_ROOT="ebb4c6a84a4cf0e607ef62b39a55dc46ec0ec059f3d02bc1b77e4d084ef9a79b"
PROBE_SHA="3d486d3a086e6515a8663cfa6301eb90cdbba080a898d2a0b14ff02521d61c6b"; PROBE_BYTES=320*180*4
INPUTS={
"first-frame":["76c592ae233cf18ab86145dbdd4064d9ded3764431a17f3a985cf1bf975828be","92b26a48d369847e1a441333bf0351d5d533068a0ba5c6fcc10c7617ddb245ef","9d78b93c9f49c0e52361833cf2fa2a041680aa031c695b6288c2e8e7d198e0f2","0a11d1a0cf395eee4c8ac5eafb2657c4da9d8c0d91040824097ef723f8661cf2","b7eb44d3ec7a1c1ce970c4eb8e165c9d45756bc8151f13a32aa0fbed3dd87c5d","494279930662975cbef5ef1ad6c3d5115b748781809f2070809d9fb40bd705e2","69a13b293b84ff925c02204c56925db2cfae04b65630303dbbb4f664797d6f5a","91d2d62a5700cbf9404e3c575c3b6fc19e1e31c24c6fcfc0a2cf70d41314b447","2a589ae1f2fa2a6328223ff195a29c9244bec633dca49139f6f231e1d79c0eb2","317a8ba7007a5ecadad6ee379a48f5c557d3a4a6ebcc25d8a0177437e1e20b20"],
"fan-in":["76c592ae233cf18ab86145dbdd4064d9ded3764431a17f3a985cf1bf975828be","92b26a48d369847e1a441333bf0351d5d533068a0ba5c6fcc10c7617ddb245ef","9d78b93c9f49c0e52361833cf2fa2a041680aa031c695b6288c2e8e7d198e0f2","0a11d1a0cf395eee4c8ac5eafb2657c4da9d8c0d91040824097ef723f8661cf2","b7eb44d3ec7a1c1ce970c4eb8e165c9d45756bc8151f13a32aa0fbed3dd87c5d","494279930662975cbef5ef1ad6c3d5115b748781809f2070809d9fb40bd705e2","5748a3442e94cd17a03baf423095d1b75e0ee0b5ba4b4885eeaa9f18d59d0e5e","5e5f23f0c2f6b0d8d7877324c5957ae74da7235afecf053e51689af507d9a9ba","a69e304470d0a8a4019bed693f6cba8e2dd55e1c40e449664d03b4d85689ca56","317a8ba7007a5ecadad6ee379a48f5c557d3a4a6ebcc25d8a0177437e1e20b20"]}
OUTPUTS={"first-frame":{"tensor":"59a802d1b06e929156c96aa8535873fcd6c5e067c4a7bd6343aa97fb6cc68bfb","mask":"425dfe96e5e38dba57e63db84e1facb991bf6ca792f41f40351db2cbcde90d44","vector":"ac963698d881e5319dce78394006f017d0a367b8266fe0742581f64b68378fb8"},"fan-in":{"tensor":"87dc0fc3b1080eefa905a59391c7294868b81f7cef672f422be97647c241ff57","mask":"425dfe96e5e38dba57e63db84e1facb991bf6ca792f41f40351db2cbcde90d44","vector":"f3282cbf95b7e9b14a8cf61d8579e8f8bae854cfae07b2478d80b4bc1a027865"}}
SIZES={"tensor":1474560,"mask":57600,"vector":230400}; FILES={"tensor":"tensor-output.fp16","mask":"mask-output.r8","vector":"vector-output.rg16f"}
RUNS=tuple((f"l19-{f}-poison-{p}-process-{o}",f,int(p,16)) for f in ("first-frame","fan-in") for p in ("a5","5a") for o in ("a","b"))
class VerificationError(RuntimeError): pass
def req(c,m):
 if not c: raise VerificationError(m)
def sha(b): return hashlib.sha256(b).hexdigest()
def canon(v): return sha(json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode())
def load(p):
 try:v=json.loads(p.read_text())
 except Exception as e: raise VerificationError(f"load {p}: {e}") from e
 req(isinstance(v,dict),f"{p}: object"); return v
def reroot(v): v.pop("receipt_root_sha256",None);v["receipt_root_sha256"]=canon(v)
def rooted(v):
 x=copy.deepcopy(v);r=x.pop("receipt_root_sha256",None);return r==canon(x)
def address_free(v):
 s=json.dumps(v).lower()
 for x in ("gpu_virtual_address","device_pointer","host_pointer","descriptor_ptr"):req(x not in s,f"address field {x}")
def validate_run(root,rid,fixture,poison,backend):
 r=load(root/rid/"run-receipt.json");address_free(r);req(r.get("schema")==RUN_SCHEMA and r.get("backend")==backend and r.get("run_id")==rid and r.get("fixture")==fixture and r.get("poison_byte")==poison,f"{rid}: identity");req(isinstance(r.get("process_id"),int),f"{rid}: pid");req(r.get("selected_launch")==19 and (r.get("module_token")or r.get("source_module_token"))==5 and (r.get("module_sha256")or r.get("source_module_sha256"))==MODULE,f"{rid}: module");req(r.get("packed_parameter_bytes")==192 and r.get("frozen_selected_raw_parameter_sha256")==RAW and r.get("frozen_selected_address_free_parameter_sha256")==FROZEN and r.get("fixture_address_free_parameter_sha256")==FIXTURE_PARAM,f"{rid}: abi");req(r.get("input_generator")==GENERATOR and r.get("input_pre_sha256")==r.get("input_post_sha256")==INPUTS[fixture],f"{rid}: inputs");req(r.get("allocation29_first_frame_evidence_root_sha256")==FIRST_ROOT and r.get("allocation29_temporal_evidence_root_sha256")==TEMPORAL_ROOT and r.get("allocation29_runtime_source")=="independently-generated" and r.get("allocation29_capture_bytes_consumed") is False,f"{rid}: allocation29")
 out={}
 for k in FILES:
  b=(root/rid/FILES[k]).read_bytes();req(len(b)==SIZES[k] and r.get(f"{k}_output_bytes")==SIZES[k] and sha(b)==r.get(f"{k}_output_sha256")==OUTPUTS[fixture][k],f"{rid}: {k}");req(r.get(f"{k}_poison_pixels")==0,f"{rid}: {k} coverage");out[k]=b
 for k in ("input_mismatched_bytes","buffer_canary_mismatched_bytes","texture_canary_mismatched_bytes"):req(r.get(k)==0,f"{rid}: {k}")
 req(r.get("texture_resource_bounds_enforced") is True and r.get("complete_write_coverage_checked_by_dual_poison") is True and r.get("expected_output_fed_to_execution") is False and r.get("archived_intermediates_consumed") is False and r.get("address_fields_recorded") is False and r.get("result")=="pass",f"{rid}: gates");return r,out
def build_backend(root,backend):
 rows=[];outs={}
 for rid,f,p in RUNS:
  r,o=validate_run(root,rid,f,p,backend);outs[rid]=o;rows.append({"run_id":rid,"fixture":f,"poison_byte":p,"process_id":r["process_id"],**{f"{k}_output_sha256":r[f"{k}_output_sha256"] for k in FILES}})
 req(len({x["process_id"] for x in rows})==8,"fresh processes")
 for f in OUTPUTS:
  rs=[x for x in rows if x["fixture"]==f];first=outs[rs[0]["run_id"]];req(len(rs)==4 and all(outs[x["run_id"]]==first for x in rs),f"{f}: invariance")
 d={"schema":BACKEND_SCHEMA,"backend":backend,"selected_launch":19,"module_token":5,"module_sha256":MODULE,"input_generator":GENERATOR,"runs":rows,"fresh_process_count":8,"two_process_repeat_per_poison":True,"two_poison_output_invariance":True,"input_immutability":True,"complete_write_coverage":True,"buffer_and_texture_canaries":True,"allocation29_capture_bytes_consumed":False,"expected_output_fed_to_execution":False,"archived_intermediates_consumed":False,"address_fields_recorded":False,"result":"pass"};reroot(d);return d
MASK64=(1<<64)-1
def splitmix(v):v=(v+0x9e3779b97f4a7c15)&MASK64;v=((v^(v>>30))*0xbf58476d1ce4e5b9)&MASK64;v=((v^(v>>27))*0x94d049bb133111eb)&MASK64;return v^(v>>31)
def f32(v):return struct.unpack("<f",struct.pack("<f",v))[0]
RX=struct.unpack("<f",struct.pack("<I",0x3b4ccccd))[0];RY=struct.unpack("<f",struct.pack("<I",0x3bb60b61))[0]
def motion(pixel,ch):return f32((splitmix(0x3f84d5b5b5470917^((pixel*0x94d049bb133111eb)&MASK64)^((ch*0x9e3779b97f4a7c15)&MASK64))%9-4)/8)
def build_coordinate(path):
 b=path.read_bytes();req(len(b)==PROBE_BYTES and sha(b)==PROBE_SHA,"probe identity");vals=struct.unpack("<57600I",b);yd={-0.5:set(),0.5:set()};diff=0
 for y in range(180):
  for x in range(320):
   v=vals[y*320+x];actual=None if v==0 else ((v-1)%512,(v-1)//512);nx=f32(f32(x+.5)*RX);ny=f32(f32(y+.5)*RY);dx=motion(y*320+x,0);dy=motion(y*320+x,1);sx=f32(nx+f32(dx*RX));sy=f32(ny+f32(dy*RY));expected=None if not(0<=sx<1 and 0<=sy<1) else ((int(f32(sx*4096))*320)>>12,(int(f32(sy*4096))*180)>>12)
   if actual!=expected:
    req(actual is not None and expected is not None and actual[0]==expected[0] and dy in yd,"unexpected coordinate class");yd[dy].add((y,actual[1],expected[1]));diff+=1
 neg=sorted(a for a,_,_ in yd[-0.5]);pos=sorted(a for a,_,_ in yd[0.5]);req(neg==[58,97,103,110,116,123,129,142,148,155,161,168,174] and pos==[57,70,83,102,115,147,160] and diff==677,"coordinate exception map")
 d={"schema":"neuralstudio.fg.early-l19-coordinate-map.v1","probe_sha256":PROBE_SHA,"probe_bytes":PROBE_BYTES,"positions":57600,"fixed12_difference_count":677,"negative_half_y_retains_center_rows":neg,"positive_half_y_advances_rows":pos,"all_differences_classified":True,"probe_fed_to_model_execution":False,"expected_or_archived_output_consumed":False,"address_fields_recorded":False,"result":"pass"};reroot(d);return d
def validate_coord(v):req(rooted(v) and v.get("probe_sha256")==PROBE_SHA and v.get("fixed12_difference_count")==677 and v.get("all_differences_classified") is True and v.get("probe_fed_to_model_execution") is False and v.get("result")=="pass","coordinate receipt")
def build_final(rr,mr,c):
 validate_coord(c);r=build_backend(rr,"rtx-d3d12-nvapi");m=build_backend(mr,"mac-metal");cs=[]
 for rid,f,p in RUNS:
  row={"run_id":rid,"fixture":f,"poison_byte":p}
  for k in FILES:
   a=(rr/rid/FILES[k]).read_bytes();b=(mr/rid/FILES[k]).read_bytes();req(a==b,f"{rid}: {k} parity");row[f"{k}_bytes"]=len(a);row[f"{k}_sha256"]=sha(a);row[f"{k}_mismatched_bytes"]=0
  cs.append(row)
 d={"schema":FINAL_SCHEMA,"selected_launch":19,"module_token":5,"module_sha256":MODULE,"fixture_domain":["first-frame","fan-in"],"comparisons":cs,"rtx_backend_root_sha256":r["receipt_root_sha256"],"mac_backend_root_sha256":m["receipt_root_sha256"],"coordinate_map_root_sha256":c["receipt_root_sha256"],"fresh_process_count_per_backend":8,"two_process_repeat_per_poison":True,"two_poison_output_invariance":True,"input_immutability":True,"complete_write_coverage":True,"canary_integrity":True,"allocation29_capture_bytes_consumed":False,"expected_output_fed_to_execution":False,"archived_intermediates_consumed":False,"address_fields_recorded":False,"claims":{"standalone_l19_exact_on_fixture_domain":True,"allocation37_direct_l20_input_layout_exact":True,"connected_early_path_proven":False,"captured_allocation29_replayed":False,"complete_frame_metal_parity_proven":False,"netal_integrated":False,"performance_advantage_proven":False},"result":"pass"};reroot(d);return d
def validate_final(v,cr):
 address_free(v);req(rooted(v) and v.get("schema")==FINAL_SCHEMA and v.get("selected_launch")==19 and v.get("module_sha256")==MODULE and v.get("coordinate_map_root_sha256")==cr,"final identity");req(len(v.get("comparisons",[]))==8 and all(all(x.get(f"{k}_mismatched_bytes")==0 and x.get(f"{k}_sha256")==OUTPUTS[x["fixture"]][k] for k in FILES) for x in v["comparisons"]),"comparisons");req(all(v.get(k) is True for k in ("two_process_repeat_per_poison","two_poison_output_invariance","input_immutability","complete_write_coverage","canary_integrity")),"gates");req(v.get("allocation29_capture_bytes_consumed") is False and v.get("expected_output_fed_to_execution") is False and v.get("archived_intermediates_consumed") is False and v.get("result")=="pass","ingress/result");req(v.get("claims")=={"standalone_l19_exact_on_fixture_domain":True,"allocation37_direct_l20_input_layout_exact":True,"connected_early_path_proven":False,"captured_allocation29_replayed":False,"complete_frame_metal_parity_proven":False,"netal_integrated":False,"performance_advantage_proven":False},"claims")
def build_audit(v,cr):
 validate_final(v,cr);ms=[("tensor",lambda x:x["comparisons"][0].__setitem__("tensor_mismatched_bytes",1)),("mask",lambda x:x["comparisons"][0].__setitem__("mask_mismatched_bytes",1)),("vector",lambda x:x["comparisons"][0].__setitem__("vector_mismatched_bytes",1)),("module",lambda x:x.__setitem__("module_sha256","0"*64)),("coordinate",lambda x:x.__setitem__("coordinate_map_root_sha256","0"*64)),("coverage",lambda x:x.__setitem__("complete_write_coverage",False)),("canary",lambda x:x.__setitem__("canary_integrity",False)),("capture_ingress",lambda x:x.__setitem__("allocation29_capture_bytes_consumed",True)),("expected_ingress",lambda x:x.__setitem__("expected_output_fed_to_execution",True)),("scope",lambda x:x["claims"].__setitem__("connected_early_path_proven",True))];out=[]
 for n,m in ms:
  x=copy.deepcopy(v);m(x);reroot(x)
  try:validate_final(x,cr)
  except VerificationError:out.append({"mutation":n,"rejected":True});continue
  raise VerificationError(f"mutation accepted {n}")
 d={"schema":"neuralstudio.fg.early-l19-negative-audit.v1","accepted_receipt_root_sha256":v["receipt_root_sha256"],"coordinate_map_root_sha256":cr,"mutations":out,"mutation_count":len(out),"all_mutations_rejected":True,"address_fields_recorded":False,"result":"pass"};reroot(d);return d
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);f=p.open("x");json.dump(v,f,indent=2,sort_keys=True);f.write("\n");f.close()
def main():
 ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest="c",required=True);p=sp.add_parser("backend");p.add_argument("--backend",required=True);p.add_argument("--root",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p=sp.add_parser("coordinate");p.add_argument("--probe",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p=sp.add_parser("final");p.add_argument("--rtx-root",type=Path,required=True);p.add_argument("--mac-root",type=Path,required=True);p.add_argument("--coordinate-receipt",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p=sp.add_parser("audit");p.add_argument("--accepted",type=Path,required=True);p.add_argument("--coordinate-receipt",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=ap.parse_args()
 try:
  if a.c=="backend":v=build_backend(a.root,a.backend)
  elif a.c=="coordinate":v=build_coordinate(a.probe)
  elif a.c=="final":v=build_final(a.rtx_root,a.mac_root,load(a.coordinate_receipt))
  else:c=load(a.coordinate_receipt);validate_coord(c);v=build_audit(load(a.accepted),c["receipt_root_sha256"])
  write(a.output,v);print(f"receipt_root_sha256={v['receipt_root_sha256']}\nresult=pass");return 0
 except (VerificationError,OSError) as e:print(f"verification_error={e}");return 1
if __name__=="__main__":raise SystemExit(main())
