#!/usr/bin/env python3
"""Case-aware body validation: run the fused runtime on a fresh encoder run for one of Codex's color cases and compare the final surface with that
case's RTX output. Inputs: --case-dir (Codex layout: input-color.rgba16le, <key>/output.rgba16le, case-receipt.json), --encoder-run (fresh same-attempt
producer for this case: encoder/, block0/, native_exports.json), --key (poison-ordinal). The middle per-launch hashes cannot be checked against the
pinned strict references for a new input (they legitimately differ), so this runs in production mode with the final-surface comparison, coverage,
immutability and provenance; the receipt states exactly that. v2 (2026-09-06): the acceptance decision is scripts/nr_case_gate.py (fail-closed over runner
exit code, receipt completeness, retained failures, coverage, provenance binding and RTX equality; CPU negative tests in scripts/test-nr-case-gate.py)."""
import argparse, ast, hashlib, json, os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import nr_case_gate as gate
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); PY = sys.executable
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()
def main():
    p = argparse.ArgumentParser(); p.add_argument("--attempt", required=True); p.add_argument("--case-dir", required=True); p.add_argument("--encoder-run", required=True); p.add_argument("--key", required=True)
    p.add_argument("--reference-attempt", default=os.path.join(ROOT, "build/nr_chain/attempt-v7"), help="only for the descriptor/weights layout; middle hashes are NOT compared for a new input")
    p.add_argument("--split-attempt", required=True); p.add_argument("--vit-attempt", required=True); p.add_argument("--dec-lib", required=True); p.add_argument("--post-lib", required=True); p.add_argument("--copy-lib", required=True); p.add_argument("--probe", required=True)
    p.add_argument("--vit-tp-lib", default=""); p.add_argument("--split-tp-lib", default=""); a = p.parse_args(); att = os.path.abspath(a.attempt)
    if os.path.exists(att): raise SystemExit("preserve previous attempt: " + att)
    tex = os.path.join(a.case_dir, "input-color.rgba16le"); ref = os.path.join(a.case_dir, a.key, "output.rgba16le")
    for f in (tex, ref, os.path.join(a.encoder_run, "encoder/block22-middle.e4m3"), os.path.join(a.encoder_run, "block0/skip.e4m3"), os.path.join(a.encoder_run, "encoder/native_exports.json")):
        if not os.path.exists(f): raise SystemExit("missing live input: " + f)
    cmd = [PY, os.path.join(ROOT, "scripts/nr-body-resident.py"), "--attempt", att, "--reference-attempt", a.reference_attempt, "--key", a.key, "--encoder-run", a.encoder_run, "--split-attempt", a.split_attempt, "--vit-attempt", a.vit_attempt,
           "--dec-lib", a.dec_lib, "--post-lib", a.post_lib, "--copy-lib", a.copy_lib, "--probe", a.probe, "--production", "--tex-in", tex, "--reference-frame", ref] + (["--vit-tp-lib", a.vit_tp_lib] if a.vit_tp_lib else []) + (["--split-tp-lib", a.split_tp_lib] if a.split_tp_lib else [])
    r = subprocess.run(cmd, capture_output=True, text=True)
    if os.path.isdir(att): open(os.path.join(att, "validate.log"), "w").write(r.stdout + r.stderr)
    else: os.makedirs(att); open(os.path.join(att, "validate.log"), "w").write("runner produced no attempt directory\n" + r.stdout + r.stderr)
    rec = None; chain = None
    try: rec = json.load(open(os.path.join(att, "resident.json")))
    except Exception: rec = None
    try: chain = json.load(open(os.path.join(att, "chain.json")))
    except Exception: chain = None
    tw, th = (chain or {}).get("decoder", {}).get("tex_w", 0), (chain or {}).get("decoder", {}).get("tex_h", 0)
    result, reasons = gate.decide(rec, chain, r.returncode, tex, ref, os.path.abspath(a.encoder_run), a.key, int(tw) * int(th))
    L = (rec or {}).get("decoder_launches", []); L = ast.literal_eval(L) if isinstance(L, str) else L; fin = next((l for l in L if l.get("launch") == 156), {})
    out = {"schema": "neuralstudio.nr.case-validate.v2", "case_dir": os.path.abspath(a.case_dir), "key": a.key, "encoder_run": os.path.abspath(a.encoder_run), "input_texture_sha256": sha(tex), "rtx_reference_sha256": sha(ref),
           "case_receipt_sha256": sha(os.path.join(a.case_dir, "case-receipt.json")) if os.path.exists(os.path.join(a.case_dir, "case-receipt.json")) else None,
           "runner_returncode": r.returncode, "body_result": (rec or {}).get("result"), "body_checks": (rec or {}).get("checks"), "body_failures_all": (rec or {}).get("failures"),
           "body_failures_exempt_old_frame": [f for f in (rec or {}).get("failures", []) if gate.is_exempt(f)], "body_failures_retained": [f for f in (rec or {}).get("failures", []) if not gate.is_exempt(f)],
           "final_surface_sha256": fin.get("final_surface_sha256"), "final_surface_mismatching_values": fin.get("final_surface_mismatching_values"), "coverage_texture": fin.get("texture"),
           "note": "new input: comparisons against the pinned strict frame (per-launch/block hashes, middle view, final pool, restaged pre-image, repack) are exempt by scope and listed; every other failure, the runner exit code, receipt completeness, final-surface write coverage, immutability, provenance binding (case texture, producer run) and RTX final-surface equality are required",
           "gate_reasons": reasons, "result": result}
    json.dump(out, open(os.path.join(att, "case-validate.json"), "w"), indent=1); print("case", a.key, out["result"], "mismatching values", out["final_surface_mismatching_values"], "reasons", reasons[:5])
    sys.exit(0 if out["result"] == "pass" else 1)
if __name__ == "__main__": main()
