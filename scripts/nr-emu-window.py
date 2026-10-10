#!/usr/bin/env python3
"""Emulate one CTA of a recorded NR launch under optional emulator environment overrides and compare only the
1 KB tiles that CTA wrote against the oracle. Prints one summary line (mismatching bytes inside the written tiles,
bytes written where the oracle changed nothing) so semantic hypotheses can be tested one CTA at a time."""
import argparse, os, subprocess, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
def main():
    p = argparse.ArgumentParser(); p.add_argument("--launch", type=int, required=True); p.add_argument("--cta", required=True)
    p.add_argument("--output", required=True); p.add_argument("--env", action="append", default=[]); p.add_argument("--label", default="")
    a = p.parse_args()
    env = dict(os.environ)
    for kv in a.env: k, v = kv.split("=", 1); env[k] = v
    run = os.path.join(a.output, "run")
    if not os.path.exists(os.path.join(run, "diff.npz")):
        r = subprocess.run([sys.executable, os.path.join(HERE, "nr-vit-emulate-launch.py"), "--launch", str(a.launch), "--procs", "1", "--only", a.cta, "--output", run], env=env, capture_output=True, text=True)
        if r.returncode: print(r.stdout[-2000:], r.stderr[-4000:]); raise SystemExit(r.returncode)
    d = np.load(os.path.join(run, "diff.npz")); ch = d["changed"]; mine = d["mine"]; orc = d["oracle"]; pre = d["pre"]; extra = d["extra"]
    stored = d["stored"] if "stored" in d.files else np.concatenate([ch[mine != pre], extra])
    ins = np.isin(ch, stored); bad = ins & (mine != orc)          # stored bytes the oracle changed, with a different value
    mism = int(bad.sum()); ex = int(len(extra))                    # extra: stored with a new value where the oracle changed nothing
    tiles = np.unique(stored >> 10)
    per = {int(t): (int(((stored >> 10) == t).sum()), int((bad & ((ch >> 10) == t)).sum()) + int(((extra >> 10) == t).sum())) for t in tiles}
    line = f"{a.label or a.output} launch {a.launch} cta {a.cta}: stored {len(stored)} mism {mism} extra {ex} per_tile(stored,bad) {per}"
    print(line)
    with open(os.path.join(a.output, "window.txt"), "w") as f: f.write(line + "\n")
if __name__ == "__main__": main()
