#!/usr/bin/env python3
import argparse,hashlib,json
from difflib import SequenceMatcher
from pathlib import Path
def load(p):return json.loads(p.read_text(encoding="utf-8-sig"))
def canon(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument("--job-root",type=Path,required=True);a.add_argument("--output",type=Path,required=True);x=a.parse_args();modes=["core","hudless","hudless-ui-alpha","hudless-ui-color-alpha","distortion","output-real","disable-output","menu","reset-every-frame","subrect"];docs={m:load(x.job_root/"output"/f"normalized-v2-{m}.json") for m in modes}
 def seq(d):
  mods={z["encounter_token"]:z["blob_sha256"] for z in d["module_blobs"]["run_a"]}
  return [{"module_sha256":mods.get(z["module_token"]),"function":z["function_name"],"grid":z["grid"],"block":z["block"],"parameter_bytes":z["parameter_bytes"],"phase":z["phase"]} for z in d["selected_launches"]["run_a"]]
 base=seq(docs["core"]);rows=[]
 for mode in modes:
  s=seq(docs[mode]);bk=[canon(z) for z in base];sk=[canon(z) for z in s];ops=[]
  for tag,i,j,k,l in SequenceMatcher(a=bk,b=sk,autojunk=False).get_opcodes():
   if tag!="equal":ops.append({"operation":tag,"core_range":[i,j],"mode_range":[k,l],"core_launches":base[i:j],"mode_launches":s[k:l]})
  rows.append({"mode":mode,"launch_count":len(s),"sequence_root_sha256":canon(s),"equals_core":s==base,"delta_operations":ops})
 d={"schema":"neuralstudio.fg.feature-modes-structural-deltas.v1","core_sequence_root_sha256":canon(base),"rows":rows,"address_fields_recorded":False,"result":"pass"};d["receipt_root_sha256"]=canon(d);x.output.write_text(json.dumps(d,indent=2,sort_keys=True)+"\n");print(json.dumps({"root":d["receipt_root_sha256"],"rows":[{"mode":r["mode"],"count":r["launch_count"],"equal":r["equals_core"],"ops":len(r["delta_operations"])} for r in rows]},separators=(",",":")))
if __name__=="__main__":main()
