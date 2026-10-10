#!/usr/bin/env python3
"""Connected acceptance run of the NR split-Swin family on Metal: launches 25-57 and 100-132 in recorded order,
one fresh probe process per launch, each launch fed from the previous launches' live Metal outputs; recorded
bytes are used only at the family boundaries (the block-23 input view, the residual view of launch 26, and the
ViT output consumed by launch 100). Writes <attempt>/<poison>-<ordinal>/launch-NNN/run.json for the verifier."""
import argparse, json, os, subprocess, sys
KTYPE = {"cc_split_swin_16h_ffwd_inpview_512_tilesync_fp8": "ffwd_view", "cc_split_swin_16h_ffwd_proj_inpview_512_chained_fp8": "conv_resview",
         "cc_split_swin_16h_qkv_512_chained_fp8": "qkv_attend", "cc_split_swin_16h_proj_512_chained_fp8": "conv", "cc_split_swin_16h_ffwd_512_chained_fp8": "ffwd",
         "cc_split_swin_16h_ffwd_proj_512_chained_fp8": "conv", "cc_split_swin_16h_proj_pool_512_chained_fp8": "conv_pool", "cc_split_swin_16h_final_head_512_wait_fp8": "final_head",
         "cc_dec_input_upsample_1024_512_tilesync_fp8": "dec_upsample", "cc_split_swin_16h_proj_512_outview_wait_fp8": "conv_outview"}
def main():
    p = argparse.ArgumentParser(); p.add_argument("--attempt", required=True); p.add_argument("--fixtures", required=True); p.add_argument("--poisons", default="a5,5a"); p.add_argument("--ordinals", default="a,b")
    p.add_argument("--launches", default="25-57,100-132"); p.add_argument("--isolated", action="store_true", help="every launch reads its recorded inputs (triage)"); a = p.parse_args()
    index = json.load(open(os.path.join(a.fixtures, "index.json"))); order = []
    for part in a.launches.split(","): lo, hi = [int(x) for x in part.split("-")]; order += list(range(lo, hi + 1))
    probe = os.path.join(a.attempt, "split_probe"); lib = os.path.join(a.attempt, "split.metallib"); failures = 0
    for poison in a.poisons.split(","):
        for ordinal in a.ordinals.split(","):
            # live outputs by pool offset: the recorded graph binds by pool offset, so a launch's input role maps to the
            # producer's output through the shared pool offset
            live = {}
            for n in order:
                meta = index["launches"][str(n)]; ktype = KTYPE[meta["kernel"]]; fx = os.path.join(a.fixtures, f"launch-{n:03d}")
                out = os.path.join(a.attempt, f"{poison}-{ordinal}", f"launch-{n:03d}")
                cmd = [probe, lib, ktype, poison, out, "--fixture", fx, "--tables", a.fixtures, "--shift", f"{meta['shift'][0]},{meta['shift'][1]}"]
                for off, b in meta["bindings"].items():
                    if b["direction"] == "in" and not a.isolated and b["pool_offset"] in live: cmd += ["--in", f"{b['role']}={live[b['pool_offset']]}"]
                r = subprocess.run(cmd, capture_output=True, text=True)
                ok = r.returncode == 0; failures += not ok
                status = "pass" if ok else "FAIL"
                tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr.strip()[-200:]
                print(f"{poison}-{ordinal} launch {n:3d} {ktype:13s} {status}  {tail}", flush=True)
                if not ok and r.returncode == 2: print(r.stderr[-500:])
                for off, b in meta["bindings"].items():
                    if b["direction"] == "out": live[b["pool_offset"]] = os.path.join(out, b["role"] + ".bin")
    print("failures", failures); sys.exit(1 if failures else 0)
if __name__ == "__main__": main()
