#!/usr/bin/env python3
"""Build a body descriptor from a tracked template by repointing every path at the recovered artefacts
(2026-09-07 recovery). Nothing is invented: the template supplies the layout (offsets, grids, kernels,
patch table, control words); this script only rewrites paths, adds the temporal/resident fields the
accepted descriptors carried, and VERIFIES that every referenced file exists and - where an identity is
known from Codex's pin ledger - that its sha256 matches. Any mismatch is a hard failure.

Usage: nr-build-descriptor.py --template T.json --output D.json --stage DIR [--temporal-libs DIR]
                              [--resident-lib FILE] [--ledger LEDGER.json] [--checks per-launch|final]
"""
import argparse, hashlib, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLD = "/private/tmp/claude-501/-Users-nico/deb7f1d0-b1d5-4951-9b30-84ff130544fc/scratchpad/nrwt/"
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
MAP = {   # historical build dir -> recovered build dir (identical library bytes, verified in build/nr_ptx/recovered-libraries.json)
    "build/nr_split/attempt-v13/": "build/nr_split/verify2-v13/", "build/nr_vit/attempt-v8/": "build/nr_vit/verify2-v8/",
    "build/nr_vit/tp-attempt-v8/": "build/nr_vit/verify-tp-v8/", "build/nr_tp/attempt-v5/": "build/nr_tp/verify-v5/",
    "build/nr_ptx/attempt-v17/": "build/nr_ptx/recovery-v17/",
}
def main():
    p = argparse.ArgumentParser(); p.add_argument("--template", required=True); p.add_argument("--output", required=True)
    p.add_argument("--stage", required=True, help="regenerated decoder stage directory (params-N.bin / weights-N.bin)")
    p.add_argument("--temporal-libs", default="", help="directory with the temporal post/copy libraries (adds lib_temporal + ex2_half_table to launches 155/156)")
    p.add_argument("--ex2-table", default="build/nr_ex2_half/table-v1/ex2-scaled-half.u32x2")
    p.add_argument("--resident-lib", default="", help="validator library for the resident body (adds resident_lib)")
    p.add_argument("--ledger", default=""); p.add_argument("--checks", default="")
    a = p.parse_args()
    if os.path.exists(a.output): sys.exit("preserve existing descriptor: " + a.output)
    d = json.load(open(a.template)); known = {}
    if a.ledger:
        for path, h in json.load(open(a.ledger))["artifacts"].items():
            local = path.replace(OLD, ROOT + "/") if path.startswith(OLD) else path
            for old, new in MAP.items(): local = local.replace(ROOT + "/" + old, ROOT + "/" + new)
            if "/decoder-chain/stage/" in local: local = os.path.join(ROOT, a.stage, os.path.basename(local))
            known[os.path.abspath(local)] = h   # full-path keys: basenames collide (weights.bin, tp.metallib)
    def fix(v):
        if not isinstance(v, str) or not v.startswith(OLD): return v
        rel = v[len(OLD):]
        for old, new in MAP.items():
            if rel.startswith(old): rel = new + rel[len(old):]
        if "/decoder-chain/stage/" in rel: rel = os.path.join(a.stage, os.path.basename(rel))
        return os.path.join(ROOT, rel)
    def walk(o):
        if isinstance(o, dict): return {k: walk(v) for k, v in o.items()}
        if isinstance(o, list): return [walk(x) for x in o]
        return fix(o)
    d = walk(d)
    D = d["decoder"]
    if a.checks: D["checks"] = a.checks
    if a.temporal_libs:
        for L in D["launches"]:
            if L["launch"] == 155:
                L["lib_temporal"] = os.path.join(ROOT, a.temporal_libs, "post.metallib")
                L["ex2_half_table"] = {"file": os.path.join(ROOT, a.ex2_table), "sha256": sha(a.ex2_table), "bytes": os.path.getsize(a.ex2_table),
                                       "provenance": "Codex recovery-tables-v2 handoff (verified copy in build/nr_tables/recovery-v1)"}
            if L["launch"] == 156: L["lib_temporal"] = os.path.join(ROOT, a.temporal_libs, "copy.metallib")
    if a.resident_lib:
        d["resident_lib"] = os.path.join(ROOT, a.resident_lib); d["resident_lib_sha256"] = sha(a.resident_lib)
    # verify every referenced file
    missing, mismatch, checked = [], [], 0
    def check(o, key=""):
        nonlocal checked
        if isinstance(o, dict): [check(v, k) for k, v in o.items()]
        elif isinstance(o, list): [check(x, key) for x in o]
        elif isinstance(o, str) and o.startswith(ROOT) and ("." in os.path.basename(o)):
            if key in ("file",) and "patches" not in key and not os.path.exists(o) and o.endswith(".e4m3"): return   # patch entries: name only (harness supplies live regions)
            if not os.path.exists(o): missing.append(o); return
            checked += 1; b = os.path.abspath(o)
            if b in known and sha(o) != known[b]: mismatch.append((o, known[b], sha(o)))
    check(d)
    rec = {"schema": "neuralstudio.nr.descriptor-build.v2", "ledger_paths_known": len(known), "template": a.template, "template_sha256": sha(a.template), "stage": a.stage,
           "temporal_libs": a.temporal_libs, "resident_lib": a.resident_lib, "files_checked": checked,
           "missing": missing, "ledger_mismatches": [{"path": m[0], "expected": m[1], "actual": m[2]} for m in mismatch]}
    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    json.dump(d, open(a.output, "w"), indent=1); json.dump(rec, open(a.output + ".build-receipt.json", "w"), indent=1)
    print(f"descriptor {a.output} sha {sha(a.output)[:16]} | files checked {checked} | missing {len(missing)} | ledger mismatches {len(mismatch)}")
    for m in missing[:6]: print("  MISSING", m)
    for m in mismatch[:6]: print("  MISMATCH", m[0], m[1][:12], m[2][:12])
    sys.exit(1 if (missing or mismatch) else 0)
if __name__ == "__main__": main()
