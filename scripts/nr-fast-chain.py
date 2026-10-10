#!/usr/bin/env python3
"""Fast production-path NR chain: original image -> Codex's Block0/encoder driver (invoked unchanged) -> GPU-resident middle 25-132
(`nr-middle-resident.py`) -> GPU-resident decoder/output 133-156 (`nr-chain-resident.py`). Every intermediate is this attempt's fresh
producer output; every stage's hashes are checked against a strict reference attempt (CPU-only, after execution). Measures the
production-path wall time per ordinal against the strict per-launch harness seconds recorded in the reference attempt. This driver is a
measurement/orchestration instrument; the strict harness (`test-nr-full-chain.py`) remains the acceptance instrument."""
import argparse, hashlib, json, os, subprocess, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()
def run(label, cmd, log, cwd):
    t0 = time.time()
    with open(log, "w") as f: r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=cwd)
    return {"label": label, "command": cmd, "exit_code": r.returncode, "seconds": round(time.time() - t0, 1), "log": log}
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--attempt", required=True); p.add_argument("--reference-attempt", default=os.path.join(ROOT, "build/nr_chain/attempt-v2"), help="strict full-chain attempt whose per-stage hashes must be reproduced")
    p.add_argument("--codex-checkout", required=True); p.add_argument("--candidate-bundle", required=True); p.add_argument("--minimum-free-gib", default="", help="passed through to Codex's encoder driver ONLY when a bounded storage budget has been agreed in the journal; default keeps the driver's own 22 GiB setting")
    p.add_argument("--poisons", default="a5,5a"); p.add_argument("--ordinals", default="a,b"); p.add_argument("--encoder-run", default="", help="resume: encoder run dir INSIDE this attempt")
    p.add_argument("--split-attempt", required=True); p.add_argument("--vit-attempt", required=True)
    p.add_argument("--dec-lib", required=True); p.add_argument("--post-lib", required=True); p.add_argument("--copy-lib", required=True); p.add_argument("--chain-probe", required=True); p.add_argument("--production", action="store_true", help="resident decoder in production mode (final checks only)"); p.add_argument("--vit-tp-lib", default="", help="three-product ViT library for the fused body"); p.add_argument("--split-tp-lib", default="", help="three-product split conv library for the fused body"); p.add_argument("--fused-probe", default="", help="run middle+decoder in one process (nr-body-resident.py) with this fused middle_chain_probe")
    a = p.parse_args(); att = os.path.realpath(a.attempt)
    if os.path.exists(att): raise SystemExit(f"preserve existing attempt: {att}")
    os.makedirs(att); steps = []; failures = []; stages = {}
    ref_att = os.path.realpath(a.reference_attempt); ref = json.load(open(os.path.join(ref_att, "full-chain.json")))
    if ref.get("result") != "pass": failures.append("reference attempt is not a passing strict receipt")
    ref_sec = {s["label"]: s["seconds"] for s in ref["steps"]}
    bundle = os.path.realpath(a.candidate_bundle); keys = [f"{P}-{o}" for P in a.poisons.split(",") for o in a.ordinals.split(",")]
    enc_run = os.path.realpath(a.encoder_run) if a.encoder_run else os.path.join(att, "encoder-run")
    if a.encoder_run and os.path.commonpath([enc_run, att]) != att: raise SystemExit("resume must use an encoder run inside this attempt (realpath)")
    if not a.encoder_run:
        steps.append(run("encoder", [PY, os.path.join(a.codex_checkout, "scripts/test-nr-live-encoder.py"), "--attempt", enc_run, "--candidate-bundle", bundle] + (["--minimum-free-gib", a.minimum_free_gib] if a.minimum_free_gib else []), os.path.join(att, "encoder.log"), a.codex_checkout))
        if steps[-1]["exit_code"]: failures.append("encoder driver failed")
    vpath = os.path.join(enc_run, "verification.json")
    if not os.path.exists(vpath) or not a.encoder_run:
        steps.append(run("encoder-verify", [PY, os.path.join(a.codex_checkout, "scripts/verify-nr-live-encoder.py"), "--output", vpath, enc_run], os.path.join(att, "encoder-verify.log"), a.codex_checkout))
        if steps[-1]["exit_code"]: failures.append("encoder verification failed (exit code)")
    ev = json.load(open(vpath)) if os.path.exists(vpath) else None
    if not ev or ev.get("result") != "pass": failures.append("encoder verification missing or not pass")
    enc_seconds = next((s["seconds"] for s in steps if s["label"] == "encoder"), None)
    for key in keys:
        P = key.split("-")[0]; run_dir = os.path.join(enc_run, key); entry = os.path.join(run_dir, "encoder/block22-middle.e4m3"); skip0 = os.path.join(run_dir, "block0/skip.e4m3")
        st = {"encoder_run_dir": run_dir}
        if not (os.path.exists(entry) and os.path.exists(skip0)): failures.append(f"{key}: encoder outputs missing"); stages[key] = st; continue
        st["encoder_entry_sha256"] = sha(entry); st["block0_skip_sha256"] = sha(skip0)
        ref_mid = os.path.join(ref_att, f"middle-{key}", key); ref_body = os.path.join(ref_att, f"body-{key}")
        if sha(os.path.join(ref_att, "encoder-run", key, "encoder/block22-middle.e4m3")) != st["encoder_entry_sha256"]: failures.append(f"{key}: encoder entry differs from the reference attempt's encoder entry")
        if a.fused_probe:
            body = os.path.join(att, f"body-{key}")
            steps.append(run(f"body-{key}", [PY, os.path.join(ROOT, "scripts/nr-body-resident.py"), "--attempt", body, "--reference-attempt", ref_att, "--key", key, "--encoder-run", run_dir, "--split-attempt", a.split_attempt, "--vit-attempt", a.vit_attempt,
                                              "--dec-lib", a.dec_lib, "--post-lib", a.post_lib, "--copy-lib", a.copy_lib, "--probe", a.fused_probe] + (["--production"] if a.production else []) + (["--vit-tp-lib", a.vit_tp_lib] if a.vit_tp_lib else []) + (["--split-tp-lib", a.split_tp_lib] if a.split_tp_lib else []), os.path.join(att, f"body-{key}.log"), ROOT))
            if steps[-1]["exit_code"]: failures.append(f"{key}: fused body failed")
            br = os.path.join(body, "resident.json")
            if os.path.exists(br):
                b = json.load(open(br)); fin = next((l for l in b["decoder_launches"] if l["launch"] == 156), {})
                st["fused_body"] = {k: b[k] for k in ("result", "mode", "checks", "middle_gpu_s", "decoder_gpu_s", "gpu_s_total", "probe_wall_s", "driver_wall_s", "receipt_root_sha256")}; st["fused_body"]["final_surface_sha256"] = fin.get("final_surface_sha256")
                if b["result"] != "pass": failures.append(f"{key}: fused body result {b['result']}")
                if fin.get("final_surface_sha256") != ref["comparison_reference"]["sha256"]: failures.append(f"{key}: final surface hash differs from the reference frame")
            else: failures.append(f"{key}: fused body receipt missing")
            sec = {s["label"]: s["seconds"] for s in steps}
            st["timing_s"] = {"encoder_amortised": round(enc_seconds / len(keys), 1) if enc_seconds else None, "body_step": sec.get(f"body-{key}"), "production_total": round((enc_seconds / len(keys) if enc_seconds else 0) + sec.get(f"body-{key}", 0), 1),
                              "gpu_total": st.get("fused_body", {}).get("gpu_s_total"), "strict_reference": {"encoder_amortised": round(ref_sec.get("encoder", 0) / len(keys), 1), "middle": ref_sec.get(f"middle-{key}"), "body": ref_sec.get(f"body-{key}"),
                                                                                                         "total": round(ref_sec.get("encoder", 0) / len(keys) + (ref_sec.get(f"middle-{key}") or 0) + (ref_sec.get(f"body-{key}") or 0), 1)}}
            stages[key] = st; continue
        mid = os.path.join(att, f"middle-{key}")
        steps.append(run(f"middle-{key}", [PY, os.path.join(ROOT, "scripts/nr-middle-resident.py"), "--attempt", mid, "--reference-middle", ref_mid, "--encoder-entry", entry, "--poison", P, "--split-attempt", a.split_attempt, "--vit-attempt", a.vit_attempt], os.path.join(att, f"middle-{key}.log"), ROOT))
        if steps[-1]["exit_code"]: failures.append(f"{key}: resident middle failed")
        mr = os.path.join(mid, "resident.json"); mid_out = os.path.join(mid, "run", "out_view_132.bin")
        if os.path.exists(mr):
            m = json.load(open(mr)); st["middle"] = {"result": m["result"], "checks": m["checks"], "gpu_s": round(m["gpu_ms_total"] / 1000, 2), "probe_wall_s": round(m["probe_wall_ms_total"] / 1000, 2), "driver_wall_s": m["driver_wall_s"], "receipt_root_sha256": m["receipt_root_sha256"]}
            if m["result"] != "pass": failures.append(f"{key}: resident middle result {m['result']}")
        else: failures.append(f"{key}: resident middle receipt missing")
        if not os.path.exists(mid_out): failures.append(f"{key}: middle output missing"); stages[key] = st; continue
        st["middle_out_sha256"] = sha(mid_out)
        body = os.path.join(att, f"body-{key}")
        steps.append(run(f"body-{key}", [PY, os.path.join(ROOT, "scripts/nr-chain-resident.py"), "--attempt", body, "--reference-body", ref_body, "--encoder-run", os.path.join(run_dir, "encoder"), "--block0-skip", skip0, "--middle-out", mid_out, "--poison", P,
                                          "--dec-lib", a.dec_lib, "--post-lib", a.post_lib, "--copy-lib", a.copy_lib, "--probe", a.chain_probe] + (["--production"] if a.production else []), os.path.join(att, f"body-{key}.log"), ROOT))
        if steps[-1]["exit_code"]: failures.append(f"{key}: resident decoder failed")
        br = os.path.join(body, "resident.json")
        if os.path.exists(br):
            b = json.load(open(br)); fin = next((l for l in b["launches"] if l["launch"] == 156), {})
            st["body"] = {"result": b["result"], "gpu_s": round(b["gpu_ms_total"] / 1000, 2), "probe_wall_s": round(b["probe_wall_ms_total"] / 1000, 2), "driver_wall_s": b["driver_wall_s"], "final_surface_sha256": fin.get("final_surface_sha256"), "final_surface_mismatching_values": fin.get("final_surface_mismatching_values"), "receipt_root_sha256": b["receipt_root_sha256"]}
            if b["result"] != "pass": failures.append(f"{key}: resident decoder result {b['result']}")
            if fin.get("final_surface_sha256") != ref["comparison_reference"]["sha256"]: failures.append(f"{key}: final surface hash differs from the reference frame")
        else: failures.append(f"{key}: resident decoder receipt missing")
        sec = {s["label"]: s["seconds"] for s in steps}
        st["timing_s"] = {"encoder_amortised": round(enc_seconds / len(keys), 1) if enc_seconds else None, "middle_step": sec.get(f"middle-{key}"), "body_step": sec.get(f"body-{key}"),
                          "production_total": round((enc_seconds / len(keys) if enc_seconds else 0) + sec.get(f"middle-{key}", 0) + sec.get(f"body-{key}", 0), 1),
                          "gpu_total": round((st.get("middle", {}).get("gpu_s", 0) or 0) + (st.get("body", {}).get("gpu_s", 0) or 0), 2),
                          "strict_reference": {"encoder_amortised": round(ref_sec.get("encoder", 0) / len(keys), 1), "middle": ref_sec.get(f"middle-{key}"), "body": ref_sec.get(f"body-{key}"),
                                               "total": round(ref_sec.get("encoder", 0) / len(keys) + (ref_sec.get(f"middle-{key}") or 0) + (ref_sec.get(f"body-{key}") or 0), 1)}}
        stages[key] = st
    receipt = {"schema": "neuralstudio.nr.fast-chain.v1", "attempt": att, "reference_attempt": ref_att, "reference_receipt_root_sha256": ref["receipt_root_sha256"], "candidate_bundle": {"path": bundle, "manifest_sha256": sha(os.path.join(bundle, "manifest.json")) if os.path.exists(os.path.join(bundle, "manifest.json")) else None},
               "encoder_run": enc_run, "steps": steps, "stages": stages, "failures": failures, "untested_modes": ref.get("untested_modes"),
               "decoder_mode": "production" if a.production else "checked", "fused": bool(a.fused_probe), "note": "measurement/orchestration instrument; strict acceptance is test-nr-full-chain.py. Encoder wall is Codex's driver for all ordinals, amortised per ordinal.",
               "result": "pass" if not failures else "fail"}
    receipt["receipt_root_sha256"] = hashlib.sha256(json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    json.dump(receipt, open(os.path.join(att, "fast-chain.json"), "w"), indent=1)
    print("fast chain", receipt["result"], "failures", failures[:8])
    for key, st in stages.items():
        t = st.get("timing_s", {}); print(key, "production", t.get("production_total"), "s (gpu", t.get("gpu_total"), ") vs strict", t.get("strict_reference", {}).get("total"), "s")
    sys.exit(0 if not failures else 1)
if __name__ == "__main__": main()
