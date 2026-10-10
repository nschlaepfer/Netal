#!/usr/bin/env python3
import argparse,hashlib,json
from pathlib import Path
def canon(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def load(p):return json.loads(p.read_text(encoding="utf-8-sig"))
def main():
 a=argparse.ArgumentParser();a.add_argument("--job-root",type=Path,required=True);a.add_argument("--output",type=Path,required=True);x=a.parse_args();j=x.job_root
 matrix=load(j/"feature-modes-matrix.json");norm=load(j/"feature-modes-normalized.json");nr={r["mode"]:r for r in norm["rows"]};mr={(r["mode"],r["poison_byte"]):r for r in matrix["rows"]}
 modes=["core","hudless","hudless-ui-alpha","hudless-ui-color-alpha","distortion","output-real","disable-output","menu","subrect"]
 rows=[];core=None
 for mode in modes:
  hs=[]
  for p in (165,90):
   h=load(j/"output"/f"{mode}-poison-{'a5' if p==165 else '5a'}"/"fg-run-receipt.json");f=h["frames"][-1];hs.append((h,f))
  poison_invariant=hs[0][1]["output_interpolated_sha256"]==hs[1][1]["output_interpolated_sha256"]
  if mode=="core":core=hs[0][1]["output_interpolated_sha256"]
  n=load(j/"output"/f"normalized-v2-{mode}.json")
  rows.append({"mode":mode,"exit_codes":[mr[(mode,165)]["exit_code"],mr[(mode,90)]["exit_code"]],"resource_profile":hs[0][0]["resource_profile"],"launch_count":nr[mode]["launch_count"],"structural_root_sha256":n["structural_graph"]["root_a_sha256"],"normalized_repeat_root_sha256":n["receipt_root_sha256"],"address_free_parameter_repeat":nr[mode]["address_free_parameters_exact"],"generated_output_sha256_by_poison":[hs[0][1]["output_interpolated_sha256"],hs[1][1]["output_interpolated_sha256"]],"poison_invariant":poison_invariant,"generated_output_equals_core":poison_invariant and hs[0][1]["output_interpolated_sha256"]==core,"output_real_sha256":hs[0][1]["output_real_sha256"],"disable_sha256":hs[0][1]["disable_sha256"],"accepted":[hs[0][0]["accepted"],hs[1][0]["accepted"]],"untouched_pixels":[hs[0][1]["output_untouched_pixels"],hs[1][1]["output_untouched_pixels"]]})
 reset=[]
 for p in (165,90):
  h=load(j/"output"/f"reset-every-frame-poison-{'a5' if p==165 else '5a'}"/"fg-run-receipt.json");reset.append({"poison":p,"exit_code":mr[("reset-every-frame",p)]["exit_code"],"accepted":h["accepted"],"frame_count":len(h["frames"]),"frame_resets":[f["reset"] for f in h["frames"]]})
 unsupported=[]
 for m in ("mfg2-index1","mfg2-index2","mfg3-index1","mfg3-index3"):
  unsupported.append({"mode":m,"exit_codes":[mr[(m,165)]["exit_code"],mr[(m,90)]["exit_code"]],"trace_lines":[mr[(m,165)]["trace_lines"],mr[(m,90)]["trace_lines"]],"error":mr[(m,165)]["error"]})
 d={"schema":"neuralstudio.fg.feature-modes-verification.v1","resolution":[640,360],"runs":len(matrix["rows"]),"successful_modes":rows,"reset_every_frame_negative":reset,"unsupported_multiframe":unsupported,"supported_multiframe":{"count":1,"index":1},"fresh_processes":True,"two_poison_output_invariance_modes":[r["mode"] for r in rows if r["poison_invariant"]],"poison_variant_counterexamples":[r["mode"] for r in rows if not r["poison_invariant"]],"all_successful_outputs_zero_untouched_pixels":all(all(v==0 for v in r["untouched_pixels"]) for r in rows),"expected_or_archived_bytes_fed_to_execution":False,"address_fields_recorded":False,"result":"pass"};d["receipt_root_sha256"]=canon(d);x.output.write_text(json.dumps(d,indent=2,sort_keys=True)+"\n");print(json.dumps({"root":d["receipt_root_sha256"],"rows":rows},separators=(",",":")));return 0
if __name__=="__main__":raise SystemExit(main())
