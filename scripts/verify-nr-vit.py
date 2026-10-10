#!/usr/bin/env python3
"""Seal an NR ViT-1D acceptance attempt: every block, poison and ordinal must report zero mismatches,
exactly-once coverage, intact canaries and immutable inputs, from four distinct processes, with the connected
chain (tokens from the previous block's live output) for blocks 32..38. Writes a receipt with a Merkle root."""
import argparse, hashlib, json, os, sys
def sha(b): return hashlib.sha256(b).hexdigest()
def main():
    p=argparse.ArgumentParser(); p.add_argument("--attempt",required=True); p.add_argument("--fixtures",required=True); p.add_argument("--output",required=True); a=p.parse_args()
    index=json.load(open(os.path.join(a.fixtures,"index.json")))
    runs=[]; pids=set(); ok=True; leaves=[]
    for poison in ("a5","5a"):
        for ordinal in ("a","b"):
            for block in range(31,39):
                d=os.path.join(a.attempt,f"{poison}-{ordinal}",f"block{block}"); r=json.load(open(os.path.join(d,"run.json")))
                pids.add(r["pid"]); stages=r["stages"]
                good=r["result"]=="pass" and r["immutable_input_errors"]==0 and all(s["mismatches"]==0 and s["coverage_errors"]==0 and s["canary_errors"]==0 for s in stages.values()) and set(stages)=={"expand","contract","q","k","v","attention","projection"}
                good=good and r["poison"]==int(poison,16) and (r["tokens_source"]=="fixture")==(block==31) and not r["isolated"]
                for name,s in stages.items():
                    exp=index["blocks"][str(block)]["files"][{"q":"q.e4m3","k":"k.e4m3","v":"v.e4m3"}.get(name,name+".e4m3")]["sha256"]
                    good=good and s["expected_sha256"]==exp and s["sha256"]==exp
                ok=ok and good
                leaf=sha(open(os.path.join(d,"run.json"),"rb").read()); leaves.append(leaf)
                runs.append({"poison":poison,"ordinal":ordinal,"block":block,"pid":r["pid"],"gpu_ms":r["gpu_ms"],"pass":good,"run_sha256":leaf})
    ok=ok and len(pids)==len(runs)          # every block run is its own fresh process
    build=open(os.path.join(a.attempt,"build-sha256.txt")).read()
    receipt={"schema":"neuralstudio.nr.vit1d-acceptance.v1","blocks":list(range(31,39)),"stages_per_block":7,"fresh_processes":len(pids),"connected_chain":True,"poisons":["a5","5a"],"ordinals":["a","b"],
             "runs":runs,"build_sha256_listing":build,"fixtures_index_sha256":sha(open(os.path.join(a.fixtures,"index.json"),"rb").read()),"result":"pass" if ok else "fail"}
    root=sha(json.dumps(receipt,sort_keys=True,separators=(",",":")).encode()); receipt["receipt_root_sha256"]=root
    json.dump(receipt,open(a.output,"w"),indent=1)
    print("result",receipt["result"],"processes",len(pids),"root",root)
    print("gpu_ms per block (a5-a):",[r["gpu_ms"] for r in runs if r["poison"]=="a5" and r["ordinal"]=="a"])
    sys.exit(0 if ok else 1)
main()
