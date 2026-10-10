#!/usr/bin/env python3
"""CPU/log analysis of the accepted temporal-sequence receipts: per-stage timings as recorded (encoder wall from Codex's runtime, body middle/decoder GPU ms from
command-buffer timestamps, body wall) and the per-launch GPU breakdown from diag/frame-N/chain-run.json. Medians over frames grouped by reset / temporal.
No new measurements; diagnostic mode (per-launch command buffers, readbacks, hashing) and ambient load caveats apply - see the doc. Writes a JSON."""
import json, os, sys, glob, statistics as st
DIRS = ["build/nr_full/temporal-seq-v7", "build/nr_full/temporal-varied-v1/case1", "build/nr_full/temporal-varied-v1/case2", "build/nr_full/temporal-images-v1", "build/nr_full/temporal-motion-v2", "build/nr_full/temporal-spatial-v1", "build/nr_full/temporal-eight-v1"]
def med(v): return round(st.median(v), 1) if v else None
stages = {"reset": {"encoder_frame_ms": [], "middle_gpu_ms": [], "decoder_gpu_ms": [], "body_wall_ms": []}, "temporal": {"encoder_frame_ms": [], "middle_gpu_ms": [], "decoder_gpu_ms": [], "body_wall_ms": []}}
launch = {"reset": {}, "temporal": {}}; per_kind = {"reset": {}, "temporal": {}}; prepare = []; n_frames = 0; sources = []
for D in DIRS:
    for rp in sorted(glob.glob(os.path.join(D, "*", "receipt.json"))):
        R = json.load(open(rp)); sources.append(rp)
        for f in R["frames"]:
            if "body" not in f or f.get("result") != "pass": continue
            g = "reset" if f["temporal"]["body_reset"] else "temporal"; n_frames += 1
            stages[g]["encoder_frame_ms"].append(f["encoder"]["frame_ms"]); stages[g]["middle_gpu_ms"].append(f["body"]["middle_gpu_ms"]); stages[g]["decoder_gpu_ms"].append(f["body"]["decoder_gpu_ms"]); stages[g]["body_wall_ms"].append(f["body"]["wall_ms"])
            if f["frame"] == 0: prepare.append(f["encoder"]["prepare_ms"])
            cr = os.path.join(os.path.dirname(rp), "diag", f"frame-{f['frame']}", "chain-run.json")
            if os.path.exists(cr):
                C = json.load(open(cr)); items = list(C.get("launches", [])); dec = C.get("decoder", {}); items += list(dec.get("launches", [])) if isinstance(dec, dict) else []
                for L in items:
                    key = f"{L.get('launch')}:{L.get('kind', L.get('kernel', '?'))}"; launch[g].setdefault(key, []).append(L.get("gpu_ms", 0.0)); per_kind[g].setdefault(str(L.get("kind", L.get("kernel", "?"))), []).append(L.get("gpu_ms", 0.0))
out = {"schema": "neuralstudio.nr.temporal-timing-summary.v1", "frames_analysed": n_frames, "sources": sources, "encoder_prepare_ms_median_first_frame": med(prepare),
       "stage_medians_ms": {g: {k: med(v) for k, v in d.items()} | {"frames": len(d["middle_gpu_ms"])} for g, d in stages.items()},
       "top_launches_by_median_gpu_ms": {g: sorted([{"launch": k, "median_gpu_ms": med(v), "n": len(v)} for k, v in launch[g].items()], key=lambda x: -x["median_gpu_ms"])[:15] for g in launch},
       "gpu_ms_by_kind_sum_of_medians": {g: sorted([{"kind": k, "sum_of_per_launch_medians_ms": round(sum(med(launch[g][kk]) for kk in launch[g] if kk.split(":", 1)[1] == k), 1), "launches": sum(1 for kk in launch[g] if kk.split(":", 1)[1] == k)} for k in per_kind[g]], key=lambda x: -x["sum_of_per_launch_medians_ms"]) for g in per_kind}}
json.dump(out, open(sys.argv[1], "w"), indent=1)
print("frames", n_frames, "| encoder prepare (first frame) median", out["encoder_prepare_ms_median_first_frame"])
for g in ("reset", "temporal"): print(g, out["stage_medians_ms"][g])
for g in ("reset", "temporal"):
    print("--", g, "top launches"); [print("  ", x) for x in out["top_launches_by_median_gpu_ms"][g][:10]]
    print("--", g, "by kind"); [print("  ", x) for x in out["gpu_ms_by_kind_sum_of_medians"][g][:12]]
