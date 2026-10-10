#!/usr/bin/env python3
"""Seal a connected launches-25..132 attempt (split-Swin -> final head -> ViT repack -> ViT-1D -> repack -> decoder
upsample -> split-Swin -> view): every probe run must pass with zero mismatches, exactly-once coverage, intact canaries
and immutable inputs; every input must be a live output of this chain except the two boundary views; every run is a
fresh process. Writes a receipt with a root hash."""
import argparse, hashlib, json, os, sys
BOUNDARY = {(25, "x_view"), (26, "res_view")}
def sha(b): return hashlib.sha256(b).hexdigest()
def main():
    p = argparse.ArgumentParser(); p.add_argument("--attempt", required=True); p.add_argument("--split-fixtures", default="build/nr_split/fixtures"); p.add_argument("--vit-fixtures", default="build/nr_vit/fixtures"); p.add_argument("--output", required=True); a = p.parse_args()
    sidx = json.load(open(os.path.join(a.split_fixtures, "index.json"))); vidx = json.load(open(os.path.join(a.vit_fixtures, "index.json")))
    runs = []; pids = []; ok = True
    def bad(msg):
        nonlocal ok; ok = False; print("FAIL:", msg, file=sys.stderr)
    for poison in ("a5", "5a"):
        for ordinal in ("a", "b"):
            base = os.path.join(a.attempt, f"{poison}-{ordinal}")
            for n in list(range(25, 58)) + list(range(100, 133)):
                meta = sidx["launches"][str(n)]; d = os.path.join(base, f"launch-{n:03d}"); r = json.load(open(os.path.join(d, "run.json"))); pids.append(r["pid"])
                good = r["result"] == "pass" and r["immutable_input_errors"] == 0 and r["poison"] == int(poison, 16) and r["shift"] == meta["shift"]
                for role, o in r["outputs"].items():
                    exp = meta["files"][role + ".bin"]["sha256"]
                    good = good and o["mismatches"] == 0 and o["coverage_errors"] == 0 and o["canary_errors"] == 0 and o["expected_sha256"] == exp and (role == "out_vit" or o["sha256"] == exp)
                for role, src in r["input_sources"].items():
                    if role == "weights": continue
                    live = src.startswith(base)
                    if live == ((n, role) in BOUNDARY): good = False; print(f"{poison}-{ordinal} launch {n} input {role} from {src}", file=sys.stderr)
                if not good: bad(f"{poison}-{ordinal} launch {n}")
                runs.append({"poison": poison, "ordinal": ordinal, "launch": n, "kernel": meta["kernel"], "pid": r["pid"], "gpu_ms": r["gpu_ms"], "pass": good, "run_sha256": sha(open(os.path.join(d, "run.json"), "rb").read())})
            for name, need_live in (("repack58", {"repack58"}), ("repack99", {"repack58", "repack99"})):
                d = os.path.join(base, name); r = json.load(open(os.path.join(d, "run.json"))); pids.append(r["pid"])
                good = r["result"] == "pass" and r["poison"] == int(poison, 16) and all(k["mismatches"] == 0 and k["coverage_errors_dwords"] == 0 and k["canary_errors"] == 0 and k["input_immutable"] for k in r["kernels"].values())
                for k, src in r["input_sources"].items():
                    if (src != "fixture") != (k in need_live): good = False; print(f"{poison}-{ordinal} {name} input {k} from {src}", file=sys.stderr)
                if not good: bad(f"{poison}-{ordinal} {name}")
                runs.append({"poison": poison, "ordinal": ordinal, "launch": 58 if name == "repack58" else 99, "kernel": name, "pid": r["pid"], "gpu_ms": None, "pass": good, "run_sha256": sha(open(os.path.join(d, "run.json"), "rb").read())})
            for block in range(31, 39):
                d = os.path.join(base, f"block{block}"); r = json.load(open(os.path.join(d, "run.json"))); pids.append(r["pid"]); st = r["stages"]
                good = r["result"] == "pass" and r["immutable_input_errors"] == 0 and not r["isolated"] and r["poison"] == int(poison, 16) and set(st) == {"expand", "contract", "q", "k", "v", "attention", "projection"}
                good = good and all(s["mismatches"] == 0 and s["coverage_errors"] == 0 and s["canary_errors"] == 0 for s in st.values()) and r["tokens_source"] != "fixture" and r["tokens_source"].startswith(base)
                for name, s in st.items():
                    exp = vidx["blocks"][str(block)]["files"][{"q": "q.e4m3", "k": "k.e4m3", "v": "v.e4m3"}.get(name, name + ".e4m3")]["sha256"]; good = good and s["expected_sha256"] == exp and s["sha256"] == exp
                if not good: bad(f"{poison}-{ordinal} vit block {block}")
                runs.append({"poison": poison, "ordinal": ordinal, "launch": [59 + 5 * (block - 31) + i for i in range(5)], "kernel": f"vit block {block}", "pid": r["pid"], "gpu_ms": r["gpu_ms"], "pass": good, "run_sha256": sha(open(os.path.join(d, "run.json"), "rb").read())})
    if len(set(pids)) != len(pids): bad("processes not distinct")
    receipt = {"schema": "neuralstudio.nr.middle-acceptance.v1", "launches": "25-132 (108 recorded launches)", "families": ["split-Swin 23-30", "final head", "ViT repack", "ViT-1D 31-38", "ViT repack", "decoder input upsample", "split-Swin 40-47", "output view"],
               "fresh_processes": len(set(pids)), "runs_expected": len(runs), "connected_chain": True, "boundary_inputs": sorted(f"{n}:{r}" for n, r in BOUNDARY), "poisons": ["a5", "5a"], "ordinals": ["a", "b"], "runs": runs,
               "split_fixtures_index_sha256": sha(open(os.path.join(a.split_fixtures, "index.json"), "rb").read()), "vit_fixtures_index_sha256": sha(open(os.path.join(a.vit_fixtures, "index.json"), "rb").read()), "result": "pass" if ok else "fail"}
    root = sha(json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()); receipt["receipt_root_sha256"] = root
    json.dump(receipt, open(a.output, "w"), indent=1)
    gpu = sum(r["gpu_ms"] or 0 for r in runs if r["poison"] == "a5" and r["ordinal"] == "a")
    print("result", receipt["result"], "runs", len(runs), "processes", len(set(pids)), "root", root); print(f"GPU ms for one pass of launches 25-132 (a5-a, repacks excluded): {gpu:.1f}")
    sys.exit(0 if ok else 1)
main()
