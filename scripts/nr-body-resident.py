#!/usr/bin/env python3
"""Fused GPU-resident body: middle 25-132 and decoder/output 133-156 in ONE process on one resident pool (middle_chain_probe with a
"decoder" section). The middle section is checked exactly as nr-middle-resident.py (every launch/stage output hash vs the strict
reference, coverage); the pool is then restaged in-process to the strict decoder pre-image (this run's middle view kept, everything else
re-poisoned, live encoder skips and Block0 skip patched in from files, control words 0xff) and the decoder launches run as in
nr-chain-resident.py (checked: per-launch pool hashes; --production: final pool hash only, skipped checks listed). Comparisons are CPU-only,
after execution. The separate runners and the strict harness remain the acceptance instruments."""
import argparse, hashlib, json, os, subprocess, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); PY = sys.executable
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()
def main():
    p = argparse.ArgumentParser(); p.add_argument("--attempt", required=True); p.add_argument("--reference-attempt", required=True, help="strict full-chain attempt dir"); p.add_argument("--key", required=True, help="poison-ordinal, e.g. a5-a")
    p.add_argument("--encoder-run", required=True, help="encoder run dir for this key (contains encoder/ and block0/)"); p.add_argument("--split-attempt", required=True); p.add_argument("--vit-attempt", required=True)
    p.add_argument("--dec-lib", required=True); p.add_argument("--post-lib", required=True); p.add_argument("--copy-lib", required=True); p.add_argument("--probe", required=True, help="fused middle_chain_probe binary"); p.add_argument("--production", action="store_true"); p.add_argument("--vit-tp-lib", default=""); p.add_argument("--split-tp-lib", default=""); p.add_argument("--chain-only", action="store_true", help="write chain.json and exit (serve driver)"); p.add_argument("--check-only", action="store_true", help="skip the probe; check the existing run/ under --attempt (serve driver)"); p.add_argument("--tex-in", default=""); p.add_argument("--reference-frame", default="")
    a = p.parse_args(); att = os.path.abspath(a.attempt)
    if os.path.exists(att) and not a.check_only: raise SystemExit("preserve previous attempt: " + att)
    if not a.check_only: os.makedirs(att)
    P = a.key.split("-")[0]; ref = os.path.abspath(a.reference_attempt); ref_mid = os.path.join(ref, f"middle-{a.key}", a.key); ref_body = os.path.join(ref, f"body-{a.key}")
    entry = os.path.join(a.encoder_run, "encoder/block22-middle.e4m3"); skip0 = os.path.join(a.encoder_run, "block0/skip.e4m3")
    if a.check_only: mid = json.load(open(os.path.join(att, "chain.json"))); dec = {"tex_in": mid["decoder"]["tex_in"]}
    else:
      r = subprocess.run([PY, os.path.join(ROOT, "scripts/nr-middle-resident.py"), "--attempt", os.path.join(att, "middle-chain"), "--reference-middle", ref_mid, "--encoder-entry", entry, "--poison", P, "--split-attempt", a.split_attempt, "--vit-attempt", a.vit_attempt, "--chain-only"] + (["--vit-tp-lib", a.vit_tp_lib] if a.vit_tp_lib else []) + (["--split-tp-lib", a.split_tp_lib] if a.split_tp_lib else []), capture_output=True, text=True)
      if r.returncode: raise SystemExit("middle chain: " + r.stderr[-600:])
      r = subprocess.run([PY, os.path.join(ROOT, "scripts/nr-chain-resident.py"), "--attempt", os.path.join(att, "decoder-chain"), "--reference-body", ref_body, "--encoder-run", os.path.join(a.encoder_run, "encoder"), "--block0-skip", skip0, "--poison", P,
                          "--dec-lib", a.dec_lib, "--post-lib", a.post_lib, "--copy-lib", a.copy_lib, "--probe", a.probe, "--chain-only"] + (["--production"] if a.production else []) + (["--tex-in", a.tex_in] if a.tex_in else []), capture_output=True, text=True)
      if r.returncode: raise SystemExit("decoder chain: " + r.stderr[-600:])
      mid = json.load(open(os.path.join(att, "middle-chain", "chain.json"))); dec = json.load(open(os.path.join(att, "decoder-chain", "chain.json")))
      mid["decoder"] = {k: dec[k] for k in ("tables", "tex_in", "tex_w", "tex_h", "checks", "launches", "patches", "control")}

    cj = os.path.join(att, "chain.json")
    if not a.check_only: json.dump(mid, open(cj, "w"), indent=1)
    if a.chain_only: print("chain", cj); return
    if a.check_only: wall = float(open(os.path.join(att, "serve-wall-ms.txt")).read()) / 1000.0 if os.path.exists(os.path.join(att, "serve-wall-ms.txt")) else -1.0
    else:
        t0 = time.time(); r = subprocess.run([a.probe, cj, os.path.join(att, "run")], capture_output=True, text=True); wall = time.time() - t0
        open(os.path.join(att, "probe.log"), "w").write(r.stdout + r.stderr)
        if r.returncode: raise SystemExit("fused probe failed: " + r.stderr[-800:])
    run = json.load(open(os.path.join(att, "run", "chain-run.json"))); failures = []; checks = 0; per = []
    for L in run["launches"]:
        n, kind = L["launch"], L["kind"]
        if kind == "ptx":
            rr = json.load(open(os.path.join(ref_body, f"launch-{n:03d}", "report.json")))
            same = True if a.production else rr["emulated_post_sha256"] == L["pool_post_sha256"]; checks += 0 if a.production else 1
            cov_ok = True if a.production else (L["words_written_more_than_once"] == 0 and L["words_written_once"] >= rr["coverage"]["oracle_changed_words"])
            item = {"launch": n, "gpu_ms": L["gpu_ms"], "wall_ms": L["wall_ms"], "pool_hash_matches_reference": None if a.production else same}
            if not same: failures.append(f"launch {n}: pool hash differs from the strict reference")
            if not cov_ok: failures.append(f"launch {n}: coverage")
            if not (L["weights_immutable"] and L["params_immutable"]): failures.append(f"launch {n}: immutability")
            if "texture" in L:
                t = L["texture"]; item["texture"] = t; checks += 1
                if t["pixels_once"] != t["pixels"] or t["pixels_multi"] != 0: failures.append(f"launch {n}: texture not written exactly once")
                if n == 156:
                    to = np.fromfile(os.path.join(att, "run", "tex_out_156.bin"), dtype=np.uint16); refimg = np.fromfile(a.reference_frame if a.reference_frame else os.path.join(os.path.dirname(dec["tex_in"]), "output.rgba16le"), dtype=np.uint16)
                    item["final_surface_mismatching_values"] = int((to != refimg).sum()); item["final_surface_sha256"] = t["output_sha256"]; checks += 1
                    if item["final_surface_mismatching_values"]: failures.append("final surface differs from the recording")
                if n == 155 and t["input_after_sha256"] != run["decoder"]["texture_in_sha256"]: failures.append("input colour texture changed")
            per.append(item); continue
        if kind == "vit_block":
            for stage, h in L["outputs"].items():
                f = os.path.join(ref_mid, f"block{n}", f"{stage}.bin"); checks += 1
                if not os.path.exists(f): failures.append(f"block {n}: missing reference {stage}"); continue
                if sha(f) != h: failures.append(f"block {n} {stage}: hash differs from the strict reference")
            for stage, (once, multi, zero) in L["coverage"].items():
                if multi or zero: failures.append(f"block {n} {stage}: coverage once={once} multi={multi} zero={zero}")
        elif kind.startswith("nr_vit_repack"):
            f = os.path.join(ref_mid, "repack58" if n == 58 else "repack99", kind + ".bin"); checks += 1
            if sha(f) != L["outputs"][kind]: failures.append(f"repack {n}: hash differs")
            once, multi, zero = L["coverage_dwords"]
            if multi or zero: failures.append(f"repack {n}: coverage {L['coverage_dwords']}")
        else:
            for role, h in L["outputs"].items():
                f = os.path.join(ref_mid, f"launch-{n:03d}", f"{role}.bin"); checks += 1
                if not os.path.exists(f): failures.append(f"launch {n}: missing reference {role}"); continue
                if sha(f) != h: failures.append(f"launch {n} {role}: hash differs from the strict reference")
            for role, (once, multi, zero) in L["coverage"].items():
                if multi or (zero and role != "out_vit"): failures.append(f"launch {n} {role}: coverage once={once} multi={multi} zero={zero}")
    final_ref = sha(os.path.join(ref_mid, "launch-132", "out_view.bin")); checks += 1
    if sha(os.path.join(att, "run", "out_view_132.bin")) != final_ref: failures.append("middle final view differs from the strict reference")
    last = max(L["launch"] for L in run["launches"] if L["kind"] == "ptx"); ref_last = json.load(open(os.path.join(ref_body, f"launch-{last:03d}", "report.json")))["emulated_post_sha256"]; checks += 1
    if run["decoder"]["pool_final_sha256"] != ref_last: failures.append("final pool hash differs from the strict reference's last launch")
    # the restaged decoder pre-image must equal the strict body's pool_pre (same live inputs, same poison)
    pre_ref = json.load(open(os.path.join(ref_body, "body.json"))).get("pool_pre_133_sha256"); checks += 1 if pre_ref else 0
    if pre_ref and run["decoder"]["pool_pre_sha256"] != pre_ref: failures.append("restaged decoder pre-image differs from the strict body's pool_pre")
    fc = json.load(open(os.path.join(ref, "full-chain.json"))); sec = {s["label"]: s["seconds"] for s in fc["steps"]}
    receipt = {"schema": "neuralstudio.nr.body-resident.v1", "attempt": att, "key": a.key, "reference_attempt": ref, "reference_receipt_root_sha256": fc["receipt_root_sha256"], "mode": "production" if a.production else "checked",
               "checks_skipped_in_production": ["decoder per-launch pool hash", "decoder per-launch pool coverage scan"] if a.production else [], "checks": checks, "failures": failures, "decoder_launches": per,
               "middle_gpu_s": round(run["middle_gpu_ms"] / 1000, 3), "middle_wall_s": round(run["middle_wall_ms"] / 1000, 3), "decoder_gpu_s": round(run["decoder"]["gpu_ms"] / 1000, 3), "decoder_wall_s": round(run["decoder"]["wall_ms"] / 1000, 3),
               "gpu_s_total": round(run["gpu_ms_total"] / 1000, 3), "probe_wall_s": round(run["wall_ms_total"] / 1000, 3), "driver_wall_s": round(wall, 2), "decoder_pre_image_matches_strict": None if not pre_ref else run["decoder"]["pool_pre_sha256"] == pre_ref,
               "reference_strict_seconds": {"middle": sec.get(f"middle-{a.key}"), "body": sec.get(f"body-{a.key}")}, "result": "pass" if not failures else "fail"}
    receipt["receipt_root_sha256"] = hashlib.sha256(json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()).hexdigest(); json.dump(receipt, open(os.path.join(att, "resident.json"), "w"), indent=1)
    print("fused body", receipt["result"], "checks", checks, "failures", failures[:6]); print(f"GPU {receipt['gpu_s_total']} s (middle {receipt['middle_gpu_s']}, decoder {receipt['decoder_gpu_s']}), probe wall {receipt['probe_wall_s']} s, driver wall {wall:.1f} s"); sys.exit(0 if not failures else 1)   # 0 pass, 1 receipt written with failures listed; execution errors exit 2 (see main guard)
if __name__ == "__main__":
    try: main()
    except SystemExit as e:
        if isinstance(e.code, str): print(e.code, file=sys.stderr); sys.exit(2)   # execution error (no receipt or aborted): distinct from "failures listed"
        raise
