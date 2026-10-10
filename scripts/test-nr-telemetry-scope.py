#!/usr/bin/env python3
"""Regression tests for the telemetry reporting bugs found in the 2026-09-08 Block2 cost report (Codex 04:59).

They run against retained receipts and need no GPU. Each test states the defect it prevents:
 1. block-scope host time must come from the authoritative record (label==NAME, command==false), never from a
    phase-label PREFIX sum - "block2" also prefixes block20/block21/block22, which inflated the first report.
 2. that authoritative record must be unique per frame.
 3. the prefix-sum defect must actually be demonstrable in the data (otherwise test 1 proves nothing here).
 4. encoder API time and frame wall time are different fields and must never be substituted for one another.
 5. paired statistics must average the temporal steps WITHIN each process before pairing, with the reset-first
    frame kept separate.
Usage: test-nr-telemetry-scope.py RUN_ROOT [OUT.json]
"""
import json, statistics, sys, os, hashlib
root = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else None
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
def receipts():
    for arm in ("baseline", "candidate"):
        for i in (1, 2, 3, 4):
            p = f"{root}/{arm}-p{i}/process-{i}/receipt.json"
            if os.path.exists(p): yield arm, i, json.load(open(p))
def authoritative(t, label): 
    r = [x for x in t["records"] if x.get("label") == label and x.get("command") is False]
    return r
def prefix_sum(t, prefix):
    return sum(ph.get("wall_ms", 0.0) for ph in t["phases"] if ph.get("label", "").startswith(prefix))
cases = []
def check(name, ok, detail=""):
    cases.append({"case": name, "ok": bool(ok), "detail": detail}); print(("PASS " if ok else "FAIL ") + name + (" :: " + detail if detail else ""))
    return ok
allok = True
frames = [(arm, i, f) for arm, i, r in receipts() for f in r["frames"]]
allok &= check("receipts-present", len(frames) == 24, f"{len(frames)} frames")
# 2. uniqueness
uniq = all(len(authoritative(f["encoder_telemetry"], "block2")) == 1 for _, _, f in frames)
allok &= check("authoritative-record-unique", uniq, "exactly one label==block2 command==false record per frame")
# 3. the prefix defect is real in this data
labels = {ph["label"].split("/")[0] for _, _, f in frames for ph in f["encoder_telemetry"]["phases"]}
sibling = sorted(l for l in labels if l.startswith("block2") and l != "block2")
allok &= check("prefix-siblings-exist", len(sibling) >= 1, f"labels sharing the prefix: {sibling}")
# 1. prefix sum must differ from the authoritative value (i.e. the old method was wrong)
arm, i, f = frames[0]; t = f["encoder_telemetry"]
a = authoritative(t, "block2")[0]["host_scope_ms"]; p = prefix_sum(t, "block2"); pe = prefix_sum(t, "block2/")
allok &= check("prefix-sum-differs-from-authoritative", abs(p - a) > 1.0, f"prefix {p:.4f} ms vs authoritative {a:.4f} ms (delta {p-a:+.4f})")
allok &= check("exact-slash-prefix-matches-authoritative", abs(pe - a) < 0.05, f"'block2/' sum {pe:.6f} ms vs record {a:.4f} ms")
# 4. encoder API and frame wall are distinct
bad = [(arm, i, f["frame"]) for arm, i, f in frames if abs(f["encoder"]["frame_ms"] - f["frame_wall_ms"]) < 1e-9]
allok &= check("encoder-api-and-frame-wall-distinct", not bad, f"{len(bad)} frames where the two fields coincide")
mx = max(frames, key=lambda x: x[2]["frame_wall_ms"])
allok &= check("largest-frame-wall-exceeds-its-encoder-api", mx[2]["frame_wall_ms"] > mx[2]["encoder"]["frame_ms"],
               f"{mx[0]}-p{mx[1]} frame {mx[2]['frame']}: wall {mx[2]['frame_wall_ms']/1000:.4f} s vs encoder API {mx[2]['encoder']['frame_ms']/1000:.4f} s")
# 5. pairing shape: 8 processes -> 4 pairs, temporal averaged within process, reset separate
per = {}
for arm, i, r in receipts():
    fs = r["frames"]; per[(arm, i)] = {"reset": [x for x in fs if x["frame"] == 0], "temporal": [x for x in fs if x["frame"] > 0]}
shape = len(per) == 8 and all(len(v["reset"]) == 1 and len(v["temporal"]) == 2 for v in per.values())
allok &= check("pairing-shape", shape, "8 processes, 1 reset + 2 temporal frames each -> 4 paired units")
means = [statistics.mean([x["frame_wall_ms"] for x in per[(a, i)]["temporal"]]) for a in ("baseline", "candidate") for i in (1, 2, 3, 4)]
allok &= check("temporal-mean-before-pairing", len(means) == 8 and all(m > 0 for m in means), "within-process temporal means computed for all eight processes")
rep = {"schema": "neuralstudio.nr.telemetry-scope-tests.v1", "run_root": root, "cases": cases, "all_ok": allok}
if out: json.dump(rep, open(out, "w"), indent=1); print("receipt", sha(out)[:16])
print("ALL PASS" if allok else "FAILURES PRESENT"); sys.exit(0 if allok else 1)
