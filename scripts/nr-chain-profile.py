#!/usr/bin/env python3
"""Profile a full-chain attempt from its receipts: per stage and per launch, GPU time (command-buffer GPU start->end), probe wall
time, and the remaining process/file overhead (harness step seconds minus probe wall). Read-only; distinguishes the runtime cost
that optimisation can address from orchestration overhead."""
import argparse, glob, json, os
def main():
    p = argparse.ArgumentParser(); p.add_argument("--attempt", required=True); a = p.parse_args()
    fc = json.load(open(os.path.join(a.attempt, "full-chain.json"))); steps = {s["label"]: s for s in fc["steps"]}
    rows = []
    for key, stage in fc["stages"].items():
        if "body" not in stage: continue
        body = os.path.join(a.attempt, f"body-{key}"); gpu = wall = 0.0; per = []
        for d in sorted(glob.glob(os.path.join(body, "launch-*"))):
            rj = os.path.join(d, "run", "run.json"); rp = os.path.join(d, "report.json")
            if not os.path.exists(rj): continue
            r = json.load(open(rj)); gpu += r["gpu_ms"]; wall += r["wall_ms"]; per.append((os.path.basename(d), round(r["gpu_ms"], 1), round(r["wall_ms"], 1)))
        step = steps.get(f"body-{key}", {}).get("seconds", 0.0)
        rows.append((key, "decoder+output 133-156", round(gpu / 1000, 2), round(wall / 1000, 2), round(step - wall / 1000, 1), per))
        mid = os.path.join(a.attempt, f"middle-{key}"); mgpu = mwall = 0.0; mper = []
        for d in sorted(glob.glob(os.path.join(mid, "*", "launch-*"))) + sorted(glob.glob(os.path.join(mid, "*", "block*"))):
            rj = os.path.join(d, "run.json")
            if not os.path.exists(rj): continue
            r = json.load(open(rj)); g = r.get("gpu_ms") or sum(v.get("gpu_ms", 0) for v in r.values() if isinstance(v, dict)); w = r.get("wall_ms") or 0.0
            mgpu += g or 0; mwall += w or 0; mper.append((os.path.basename(d), round(g or 0, 1)))
        mstep = steps.get(f"middle-{key}", {}).get("seconds", 0.0)
        rows.append((key, "middle 25-132", round(mgpu / 1000, 2), round(mwall / 1000, 2) if mwall else None, round(mstep - mwall / 1000, 1) if mwall else round(mstep, 1), mper))
    print(f"{'run':6s} {'stage':22s} {'gpu s':>7s} {'probe wall s':>13s} {'harness overhead s':>19s}")
    for key, stage, g, w, ov, per in rows:
        print(f"{key:6s} {stage:22s} {g:7.2f} {str(w):>13s} {str(ov):>19s}")
        if stage.startswith("decoder"): print("   top launches by GPU ms:", sorted(per, key=lambda x: -x[1])[:6])
    enc = steps.get("encoder", {}).get("seconds"); print("encoder driver (Codex, 4 chains incl. compiles) harness seconds:", enc)
if __name__ == "__main__": main()
