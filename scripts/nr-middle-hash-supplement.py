#!/usr/bin/env python3
"""Supplementary proof for the fused engine middle (Codex 05:43): compare every middle launch/stage hash recorded in a body runtime chain-run.json
(in-memory frame with diagnostics on) against a retained STRICT scalar middle run directory (.../middle-<key>/<key> of a strict attempt), plus the
final launch-132 view. Same 125-check rule as nr-middle-resident.py, no GPU. Usage: nr-middle-hash-supplement.py CHAIN_RUN_JSON REF_MIDDLE_DIR OUT_JSON"""
import hashlib, json, os, sys
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
def main():
    run_p, ref, out = sys.argv[1:4]; run = json.load(open(run_p)); failures = []; checks = 0; compared = []
    for L in run["launches"]:
        n, kind = L["launch"], L["kind"]
        if kind == "ptx": continue
        if kind == "vit_block":
            for stage, h in L["outputs"].items():
                f = os.path.join(ref, f"block{n}", f"{stage}.bin")
                if not os.path.exists(f): failures.append(f"block {n}: missing reference {stage}"); continue
                checks += 1; ok = sha(f) == h; compared.append({"launch": n, "role": stage, "reference": f, "equal": ok})
                if not ok: failures.append(f"block {n} {stage}: hash differs from the strict reference")
        elif kind.startswith("nr_vit_repack"):
            f = os.path.join(ref, "repack58" if n == 58 else "repack99", kind + ".bin"); checks += 1; ok = sha(f) == L["outputs"][kind]; compared.append({"launch": n, "role": kind, "reference": f, "equal": ok})
            if not ok: failures.append(f"repack {n}: hash differs")
        else:
            for role, h in L["outputs"].items():
                f = os.path.join(ref, f"launch-{n:03d}", f"{role}.bin")
                if not os.path.exists(f): failures.append(f"launch {n}: missing reference {role}"); continue
                checks += 1; ok = sha(f) == h; compared.append({"launch": n, "role": role, "reference": f, "equal": ok})
                if not ok: failures.append(f"launch {n} {role}: hash differs from the strict reference")
    fref = os.path.join(ref, "launch-132", "out_view.bin"); fmine = os.path.join(os.path.dirname(run_p), "out_view_132.bin"); checks += 1
    ok = os.path.exists(fmine) and sha(fref) == sha(fmine); compared.append({"launch": 132, "role": "out_view", "reference": fref, "equal": ok})
    if not ok: failures.append("final launch-132 view differs")
    rec = {"schema": "neuralstudio.nr.middle-hash-supplement.v1", "chain_run": os.path.abspath(run_p), "chain_run_sha256": sha(run_p), "reference_middle": os.path.abspath(ref), "checks": checks, "failures": failures, "compared": compared, "result": "pass" if checks == 126 and not failures else "fail"}
    rec["receipt_root_sha256"] = hashlib.sha256(json.dumps(rec, sort_keys=True, separators=(",", ":")).encode()).hexdigest(); json.dump(rec, open(out, "w"), indent=1)
    print("supplement", rec["result"], "checks", checks, "failures", failures[:4]); sys.exit(0 if rec["result"] == "pass" else 1)
if __name__ == "__main__": main()
