#!/usr/bin/env python3
"""Independent audit of a connected temporal-sequence matrix directory (build/nr_full/temporal-seq-*): for every process and frame, hash the RETAINED
bytes (final-N.rgba16le, diag/frame-N/tex_out_156.bin = the actual history texture after copy 156) and check, bit for bit: history == caller output
(0 half-value differences), the receipt's history_sha256_after == the actual dump, the next frame's history_sha256_before == the previous actual dump,
frames_since_reset advances by one per frame, and the recorded final hashes match the files. No floating-point comparison, no sign canonicalisation.
Usage: nr-temporal-seq-audit.py DIR [--expect-frame N SHA]... --out FILE"""
import argparse, hashlib, json, os, sys
import numpy as np
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--expect-frame", nargs=2, action="append", default=[]); ap.add_argument("--out", required=True); a = ap.parse_args()
    expect = {int(n): s for n, s in a.expect_frame}; report = {"schema": "neuralstudio.nr.temporal-sequence-audit.v2", "dir": a.dir, "runs": [], "result": "pass"}; ok_all = True
    for pair in sorted(d for d in os.listdir(a.dir) if os.path.isdir(os.path.join(a.dir, d)) and os.path.exists(os.path.join(a.dir, d, "receipt.json"))):
        R = json.load(open(os.path.join(a.dir, pair, "receipt.json"))); frames = []; prev_hist = None; prev_fsr = 0; ok = R.get("result") == "pass"
        for f in R["frames"]:
            n = f["frame"]; fin = os.path.join(a.dir, pair, f"final-{n}.rgba16le"); hist = os.path.join(a.dir, pair, "diag", f"frame-{n}", "tex_out_156.bin")
            fin_sha = sha(fin); hist_sha = sha(hist); fa = np.fromfile(fin, dtype=np.uint16); ha = np.fromfile(hist, dtype=np.uint16); diffs = int((fa != ha).sum())
            b = f["body"]; is_reset = bool(f.get("temporal", {}).get("body_reset", n == 0))   # v2: a reset frame anywhere in the sequence reads no history and restarts the counter at 1
            e = {"frame": n, "reset": is_reset, "final_sha256": fin_sha, "history_actual_sha256": hist_sha, "history_vs_output_half_differences": diffs,
                 "receipt_final_matches_file": f["surface_sha256_caller_readback"] == fin_sha, "receipt_history_after_matches_actual": b["history_sha256_after"] == hist_sha,
                 "history_before_matches_previous_actual": (b["history_sha256_before"] == "") if is_reset else (b["history_sha256_before"] == prev_hist), "frames_since_reset_after": b["frames_since_reset"],
                 "frames_since_reset_correct": b["frames_since_reset"] == (1 if is_reset else prev_fsr + 1), "reference_mismatching_values": f["reference"]["mismatching_values"], "expected_final": expect.get(n), "final_matches_expected": (fin_sha == expect[n]) if n in expect else None, "unreferenced": n not in expect, "receipt_result": f["result"]}
            good = diffs == 0 and e["receipt_final_matches_file"] and e["receipt_history_after_matches_actual"] and e["history_before_matches_previous_actual"] and e["frames_since_reset_correct"] and (e["final_matches_expected"] in (True, None)) and f["result"] == "pass"
            e["audit"] = "pass" if good else "FAIL"; ok = ok and good; frames.append(e); prev_hist = hist_sha; prev_fsr = b["frames_since_reset"]
        report["runs"].append({"pair": pair, "pid": R["pid"], "receipt_sha256": sha(os.path.join(a.dir, pair, "receipt.json")), "executable_sha256": R["executable_sha256"], "frames": frames, "audit": "pass" if ok else "FAIL"}); ok_all = ok_all and ok
    report["result"] = "pass" if ok_all else "FAIL"; report["receipt_root_rule"] = "sha256 of this document serialised with indent=1 without receipt_root_sha256"
    report["receipt_root_sha256"] = hashlib.sha256(json.dumps(report, indent=1).encode()).hexdigest(); json.dump(report, open(a.out, "w"), indent=1)
    for r in report["runs"]: print(r["pair"], r["audit"], [(x["frame"], x["history_vs_output_half_differences"], x["audit"]) for x in r["frames"]])
    print(report["result"], report["receipt_root_sha256"][:16]); sys.exit(0 if ok_all else 1)
if __name__ == "__main__": main()
