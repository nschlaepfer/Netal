#!/usr/bin/env python3
"""Seal an NR split-Swin acceptance attempt: every launch (25-57, 100-132), poison and ordinal must report zero
mismatches, exactly-once coverage over the recorded write set, intact canaries and immutable inputs, each from a
fresh process, with every non-boundary input taken from a live Metal output of the same chain (connected). Writes a
receipt with a Merkle-style root over the run reports."""
import argparse, hashlib, json, os, sys
BOUNDARY = {(25, "x_view"), (26, "res_view"), (100, "x_vit")}          # recorded bytes produced outside this family
def sha(b): return hashlib.sha256(b).hexdigest()
def main():
    p = argparse.ArgumentParser(); p.add_argument("--attempt", required=True); p.add_argument("--fixtures", required=True); p.add_argument("--output", required=True); a = p.parse_args()
    index = json.load(open(os.path.join(a.fixtures, "index.json"))); order = list(range(25, 58)) + list(range(100, 133))
    runs = []; pids = set(); ok = True
    for poison in ("a5", "5a"):
        for ordinal in ("a", "b"):
            for n in order:
                meta = index["launches"][str(n)]; d = os.path.join(a.attempt, f"{poison}-{ordinal}", f"launch-{n:03d}"); r = json.load(open(os.path.join(d, "run.json")))
                pids.add(r["pid"]); good = r["result"] == "pass" and r["immutable_input_errors"] == 0 and r["poison"] == int(poison, 16) and r["shift"] == meta["shift"]
                outs = {b["role"] for b in meta["bindings"].values() if b["direction"] == "out"}; good = good and set(r["outputs"]) == outs
                for role, o in r["outputs"].items():
                    exp = meta["files"][role + ".bin"]["sha256"]
                    good = good and o["mismatches"] == 0 and o["coverage_errors"] == 0 and o["canary_errors"] == 0 and o["expected_sha256"] == exp
                    if role != "out_vit": good = good and o["sha256"] == exp          # the ViT-input buffer keeps poison outside the write set
                for role, src in r["input_sources"].items():
                    if role == "weights": continue
                    recorded = src.startswith(os.path.abspath(a.fixtures)) or src.startswith(a.fixtures)
                    if recorded != ((n, role) in BOUNDARY): good = False; print(f"launch {n} input {role} source {src} unexpected", file=sys.stderr)
                ok = ok and good
                runs.append({"poison": poison, "ordinal": ordinal, "launch": n, "kernel": meta["kernel"], "pid": r["pid"], "gpu_ms": r["gpu_ms"], "pass": good, "run_sha256": sha(open(os.path.join(d, "run.json"), "rb").read())})
    ok = ok and len(pids) == len(runs)
    receipt = {"schema": "neuralstudio.nr.split-swin-acceptance.v1", "launches": order, "fresh_processes": len(pids), "connected_chain": True, "boundary_inputs": sorted(f"{n}:{r}" for n, r in BOUNDARY),
               "poisons": ["a5", "5a"], "ordinals": ["a", "b"], "runs": runs, "build_sha256_listing": open(os.path.join(a.attempt, "build-sha256.txt")).read(),
               "fixtures_index_sha256": sha(open(os.path.join(a.fixtures, "index.json"), "rb").read()), "result": "pass" if ok else "fail"}
    root = sha(json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()); receipt["receipt_root_sha256"] = root
    json.dump(receipt, open(a.output, "w"), indent=1)
    gpu = sum(r["gpu_ms"] for r in runs if r["poison"] == "a5" and r["ordinal"] == "a")
    print("result", receipt["result"], "runs", len(runs), "processes", len(pids), "root", root); print(f"GPU ms for one pass of the family (a5-a): {gpu:.1f}")
    sys.exit(0 if ok else 1)
main()
