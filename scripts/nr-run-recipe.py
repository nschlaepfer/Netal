#!/usr/bin/env python3
"""Machine-checked run recipe for connected temporal sequences (v8, 2026-09-07: the recipe as run is copied into the output root as recipe-as-run.json; v7, 2026-09-06, Codex admission review 23:19: storage_floor_gib must be a real finite non-bool number >= 12 (the floor is never lowered); comparable receipts are re-measured from their own run-summary inventory (every listed replica directory must exist, the summary must be all-pass, the recomputed max per-process footprint must be <= the unmargined budget and the recipe's declared figure must not understate it) instead of trusting the recipe's number; v6, 2026-09-06, Codex admission review: admission numbers must be real finite non-bool numbers; the budget carries an `evidence` identity {harness_executable_sha256, descriptor_sha256, body_mode, comparable_receipts} that must equal the recipe's pinned build/descriptor/mode or the recipe is rejected; after every process the ACTUAL output footprint is measured and recorded, and an overrun (actual > budget) stops the configuration with an overrun receipt; v5, 2026-09-06: optional recipe-pinned `admission` budget - {per_frame_mib, fixed_mib, margin, basis} derived from the harness write sites and comparable receipts; when present the per-process check is free - budget >= floor before EVERY process (pilot and replicas) together with a fresh reservation check, replacing the fixed 3 GiB pilot reserve + pilot-measured replica rule, which stays the default without the field; the 12 GiB floor is never lowered; v4, 2026-09-06: `harness_extra_args` allowlist adds --body-checked (body checked mode on live frames); v3: ["--encoder-telemetry"] for the measurement builds, recorded in the summary; v2, 2026-09-06; v1 review findings by Codex, root d2256fbc…, applied: every pin
mandatory with structured rejections, strict types, binary resets, unique safe replica tags, finite storage floor, FULL motion-file scan for variant 0,
body-code-based status labels with raw fields retained, and `run` refuses to execute unless the caller holds the coordinated GPU reservation). A recipe pins EVERYTHING that decides
behaviour: harness build + executable sha, descriptor + sha, encoder assets + object sha, per-frame input/reference files (+ sizes/sha when given),
motion file + sha, reset sequence, encoder sampler variant, replica poison patterns, storage floor and failure policy. `preflight` rejects the
recipe BEFORE any GPU lock is taken; `run` preflights, then runs the pilot, admits storage from the measured pilot size, runs the replicas and
STOPS the configuration at the first failed replica. Every receipt is summarised with the status vocabulary: execution_completed,
reference_unavailable, comparison_passed, comparison_failed, rejected. audit_accepted and promoted are never set here (auditor/owner only).
Usage: nr-run-recipe.py preflight RECIPE.json | run RECIPE.json --out DIR [--owner NAME]"""
import argparse, hashlib, json, math, os, re, subprocess, sys, shutil
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG, MOT = 1920 * 1080 * 8, 1920 * 1080 * 4
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
RUNNER_SHA = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()
TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
def preflight(R):
    errs = []
    def need(cond, msg):
        if not cond: errs.append(msg)
    if not isinstance(R, dict): return ["recipe is not a JSON object"]
    need(R.get("schema") == "neuralstudio.nr.run-recipe.v1", f"unknown schema {R.get('schema')!r}")
    for k, t in (("name", str), ("harness_build", str), ("harness_executable_sha256", str), ("descriptor", str), ("descriptor_sha256", str), ("encoder_assets", str), ("encoder_object", str), ("encoder_object_sha256", str), ("frames", list), ("motion", dict), ("reset_sequence", str), ("encoder_sampler_variant", int), ("replicas", list), ("storage_floor_gib", (int, float)), ("failure_policy", str)):
        need(k in R, f"missing key {k}"); need(k in R and isinstance(R[k], t), f"key {k} has the wrong type")
    if errs: return errs
    H = lambda v: isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v) is not None
    for k in ("harness_executable_sha256", "descriptor_sha256", "encoder_object_sha256"): need(H(R[k]), f"{k} is not a 64-hex sha256")
    exe = os.path.join(ROOT, R["harness_build"], "nr_full_frame"); need(os.path.exists(exe), f"harness executable missing: {exe}")
    if os.path.exists(exe) and H(R["harness_executable_sha256"]): need(sha(exe) == R["harness_executable_sha256"], "harness executable sha mismatch")
    d = os.path.join(ROOT, R["descriptor"]); need(os.path.exists(d), f"descriptor missing: {d}")
    if os.path.exists(d) and H(R["descriptor_sha256"]): need(sha(d) == R["descriptor_sha256"], "descriptor sha mismatch")
    need(os.path.isdir(R["encoder_assets"]), f"encoder assets dir missing: {R['encoder_assets']}")
    need(os.path.exists(R["encoder_object"]), f"encoder object missing: {R['encoder_object']}")
    if os.path.exists(R["encoder_object"]) and H(R["encoder_object_sha256"]): need(sha(R["encoder_object"]) == R["encoder_object_sha256"], "encoder object sha mismatch")
    rec = os.path.join(ROOT, R["harness_build"], "build-sha256.txt"); need(os.path.exists(rec) and R["encoder_object_sha256"] in open(rec).read(), "harness build record does not contain the pinned encoder object sha")
    rs = R["reset_sequence"].split(","); need(all(x in ("0", "1") for x in rs), "reset_sequence must be a comma list of 0/1"); need(len(rs) == len(R["frames"]), "reset_sequence length != number of frames"); need(rs and rs[0] == "1", "first frame must be a reset")
    need(len(R["frames"]) >= 1, "frames must be non-empty")
    for i, f in enumerate(R["frames"]):
        need(isinstance(f, dict) and isinstance(f.get("input"), str), f"frame {i}: input path missing")
        if not isinstance(f, dict): continue
        need(H(f.get("input_sha256")), f"frame {i}: input_sha256 required (64 hex)")
        if isinstance(f.get("input"), str):
            need(os.path.exists(f["input"]), f"frame {i} input missing: {f['input']}")
            if os.path.exists(f["input"]): need(os.path.getsize(f["input"]) == IMG, f"frame {i} input size != 1920x1080 RGBA16F"); need(sha(f["input"]) == f.get("input_sha256"), f"frame {i} input sha mismatch")
        ref = f.get("reference"); need(ref is None or isinstance(ref, str), f"frame {i}: reference must be a path or null")
        if isinstance(ref, str) and ref:
            need(H(f.get("reference_sha256")), f"frame {i}: reference_sha256 required (64 hex)"); need(os.path.exists(ref), f"frame {i} reference missing: {ref}")
            if os.path.exists(ref): need(os.path.getsize(ref) == IMG, f"frame {i} reference size != RGBA16F image"); need(sha(ref) == f.get("reference_sha256"), f"frame {i} reference sha mismatch"); need(sha(ref) != f.get("input_sha256"), f"frame {i} reference equals its input (expected-data ingress?)")
        if R.get("comparison_references_required") is True: need(bool(ref), f"frame {i}: reference required by this recipe")
    m = R["motion"]; need(isinstance(m.get("file"), str) and H(m.get("sha256")), "motion: file path and 64-hex sha256 required")
    if isinstance(m.get("file"), str):
        need(os.path.exists(m["file"]), f"motion file missing: {m['file']}")
        if os.path.exists(m["file"]):
            need(os.path.getsize(m["file"]) == MOT, "motion file size != 1920x1080 RG16F"); need(sha(m["file"]) == m.get("sha256"), "motion sha mismatch")
            all_zero = not any(open(m["file"], "rb").read())   # FULL scan
            sv = R["encoder_sampler_variant"]; need(sv in (0, 3), "encoder_sampler_variant must be 0 or 3")
            if sv == 0: need(all_zero, "sampler variant 0 requires an all-zero motion file (full scan)")
            if sv == 3: need(m.get("nonzero_admitted") is True, "sampler variant 3 requires nonzero_admitted: true")
    reps = R["replicas"]; need(len(reps) >= 1, "replicas must be non-empty"); tags = []
    for r in reps:
        ok = isinstance(r, dict) and isinstance(r.get("tag"), str) and TAG_RE.fullmatch(r["tag"]) is not None and isinstance(r.get("poisons"), list)
        need(ok, f"replica entry malformed: {r!r}")
        if ok: need(r["tag"] not in tags, f"duplicate replica tag {r['tag']}"); tags.append(r["tag"]); need(len(r["poisons"]) == len(R["frames"]) and all(p in ("a5", "5a") for p in r["poisons"]), f"replica {r['tag']} poison pattern invalid")
    fl = R["storage_floor_gib"]; need(isinstance(fl, (int, float)) and not isinstance(fl, bool) and math.isfinite(fl) and 12 <= fl < 1e6, "storage_floor_gib must be a real finite non-bool number >= 12 GiB (the safety floor is never lowered)")
    need(R["failure_policy"] == "stop-config-on-any-failure", "failure_policy must be stop-config-on-any-failure")
    ad = R.get("admission")
    if ad is not None:
        need(isinstance(ad, dict) and all(k in ad for k in ("per_frame_mib", "fixed_mib", "margin", "basis")), "admission must carry per_frame_mib, fixed_mib, margin, basis")
        if isinstance(ad, dict):
            num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0
            need(all(num(ad.get(k)) for k in ("per_frame_mib", "fixed_mib", "margin")), "admission numbers must be real, finite, positive and not booleans")
            need(isinstance(ad.get("basis"), str) and len(ad["basis"]) > 20, "admission.basis must state the write sites / receipts it was derived from"); need(num(ad.get("margin")) and ad["margin"] >= 1.5, "admission.margin must be >= 1.5")
            ev = ad.get("evidence"); need(isinstance(ev, dict), "admission.evidence {harness_executable_sha256, descriptor_sha256, body_mode, comparable_receipts} required")
            if isinstance(ev, dict):
                need(ev.get("harness_executable_sha256") == R["harness_executable_sha256"], "admission.evidence.harness_executable_sha256 != the recipe's pinned harness (budget derived for another build)")
                need(ev.get("descriptor_sha256") == R["descriptor_sha256"], "admission.evidence.descriptor_sha256 != the recipe's pinned descriptor")
                mode = "checked" if "--body-checked" in R.get("harness_extra_args", []) else "production"; need(ev.get("body_mode") == mode, f"admission.evidence.body_mode must be {mode!r} for this recipe")
                cr = ev.get("comparable_receipts"); need(isinstance(cr, list) and len(cr) >= 1 and all(isinstance(c, dict) and isinstance(c.get("path"), str) and H(c.get("run_summary_sha256")) and num(c.get("measured_mib_per_process")) for c in cr), "admission.evidence.comparable_receipts must list >= 1 {path, run_summary_sha256, measured_mib_per_process}")
                if isinstance(cr, list):
                    for c in cr:
                        if isinstance(c, dict) and isinstance(c.get("path"), str):
                            rs = os.path.join(ROOT, c["path"], "run-summary.json"); need(os.path.exists(rs), f"comparable receipt missing: {rs}")
                            if os.path.exists(rs) and H(c.get("run_summary_sha256")): need(sha(rs) == c["run_summary_sha256"], f"comparable receipt sha mismatch: {rs}")
                            if os.path.exists(rs):   # v7: re-measure the comparable run from its OWN inventory; never trust the recipe's number
                                try: cs = json.load(open(rs))
                                except Exception: cs = None
                                need(isinstance(cs, dict) and cs.get("all_replicas_passed") is True and isinstance(cs.get("runs"), list) and len(cs["runs"]) >= 1, f"comparable receipt is not an all-pass run summary with runs: {rs}")
                                if isinstance(cs, dict) and isinstance(cs.get("runs"), list):
                                    sizes = []
                                    for run in cs["runs"]:
                                        d = os.path.join(ROOT, c["path"], str(run.get("tag"))); need(os.path.isdir(d), f"comparable replica directory missing: {d}"); sizes.append(dir_size_gib(d) * 1024.0 if os.path.isdir(d) else float("inf"))
                                    if sizes:
                                        mx = max(sizes); c["remeasured_max_mib_per_process"] = round(mx, 2)
                                        if num(ad.get("per_frame_mib")) and num(ad.get("fixed_mib")): need(mx <= ad["per_frame_mib"] * len(R["frames"]) + ad["fixed_mib"], f"comparable run {c['path']}: re-measured footprint {mx:.1f} MiB per process exceeds the unmargined budget")
                                        if num(c.get("measured_mib_per_process")): need(c["measured_mib_per_process"] >= mx * 0.9, f"comparable run {c['path']}: recipe declares {c['measured_mib_per_process']} MiB but re-measurement gives {mx:.1f} MiB")
    xa = R.get("harness_extra_args", []); need(isinstance(xa, list) and all(x in ("--encoder-telemetry", "--body-checked") for x in xa), "harness_extra_args may only contain --encoder-telemetry / --body-checked")
    if "--encoder-telemetry" in xa: need("TELEMETRY=1" in open(os.path.join(ROOT, R["harness_build"], "telemetry.txt")).read() if os.path.exists(os.path.join(ROOT, R["harness_build"], "telemetry.txt")) else False, "--encoder-telemetry requires a harness build linked with the telemetry header (telemetry.txt)")
    return errs
def dir_size_gib(path):
    return sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(path) for f in fs) / 2**30 if os.path.isdir(path) else 0.0
def held_reservation(owner):
    """the caller must already hold the coordinated GPU lock: same owner, holder alive, and this process inside the holder's process group"""
    try: h = json.load(open(os.path.expanduser("~/.nr-gpu-lock/holder.json")))   # written by scripts/nr-gpu-lock.py (META)
    except Exception: return False, "no lock metadata readable"
    if h.get("owner") != owner: return False, f"lock held by {h.get('owner')!r}, not {owner!r}"
    try: os.kill(int(h.get("pid") or 0), 0)
    except ProcessLookupError: return False, "lock holder process is gone"
    except Exception: pass
    pg = h.get("pgid")
    if pg and os.getpgid(0) != int(pg): return False, f"this process (pgid {os.getpgid(0)}) is not inside the reservation's process group {pg}"
    return True, f"reservation token {h.get('token')} held by {owner}"
def status_of(frame):
    """label from the ACTUAL observed fields (raw fields are retained beside it): rejected = no body result (admission/exception before the body);
    execution_failed = body ran and returned a non-zero code; reference_unavailable = body code 0 without a reference; comparison_passed /
    comparison_failed = body code 0 with a reference and 0 / >0 mismatching values."""
    b = frame.get("body")
    if not isinstance(b, dict) or "code" not in b: return "rejected"
    if b["code"] != 0: return "execution_failed"
    if frame.get("no_reference_no_parity_meaning") or not frame.get("reference", {}).get("file"): return "reference_unavailable"
    return "comparison_passed" if frame["reference"].get("mismatching_values") == 0 and frame.get("result") == "pass" else "comparison_failed"
def run_one(R, out, rep):
    D = os.path.join(out, rep["tag"]); os.makedirs(D, exist_ok=False); fr = R["frames"]; p = rep["poisons"]
    cmd = [os.path.join(ROOT, R["harness_build"], "nr_full_frame"), os.path.join(ROOT, R["descriptor"]), R["encoder_assets"], fr[0]["input"], p[0], "--reference", fr[0]["reference"], "--receipt", f"{D}/receipt.json", "--save-dir", D, "--diag-dir", f"{D}/diag", "--reset-sequence", R["reset_sequence"], "--motion", R["motion"]["file"], "--encoder-sampler-variant", str(R["encoder_sampler_variant"])]
    for i in range(1, len(fr)): cmd += ["--frame", fr[i]["input"], p[i], fr[i].get("reference") or ""]
    cmd += list(R.get("harness_extra_args", []))
    rc = subprocess.run(cmd, stdout=open(f"{D}/stdout.log", "w"), stderr=subprocess.STDOUT).returncode
    rec = json.load(open(f"{D}/receipt.json")) if os.path.exists(f"{D}/receipt.json") else {}
    frames = rec.get("frames", []); st = [status_of(f) for f in frames]; raw = [{"frame": f.get("frame"), "body_code": (f.get("body") or {}).get("code"), "error": f.get("error"), "mismatching_values": (f.get("reference") or {}).get("mismatching_values"), "reference_present": bool((f.get("reference") or {}).get("file"))} for f in frames]
    ok = rc == 0 and rec.get("result") == "pass" and all(s in ("comparison_passed", "reference_unavailable") for s in st) and len(st) == len(fr)
    return {"tag": rep["tag"], "exit_code": rc, "execution_completed": bool(rec), "frame_status": st, "frame_raw": raw, "receipt": f"{D}/receipt.json", "receipt_sha256": sha(f"{D}/receipt.json") if rec else None, "passed": ok}
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["preflight", "run"]); ap.add_argument("recipe"); ap.add_argument("--out"); ap.add_argument("--owner", default="Fable"); ap.add_argument("--no-lock-check", action="store_true", help="diagnostic only: skip the reservation check (never for evidence runs)"); a = ap.parse_args()
    R = json.load(open(a.recipe)); errs = preflight(R)
    if errs: print("PREFLIGHT REJECTED:"); [print("  -", e) for e in errs]; sys.exit(3)
    print("preflight ok:", R["name"])
    if a.cmd == "preflight": return
    okl, why = held_reservation(a.owner)
    if not okl and not a.no_lock_check: print("REFUSED: run requires the coordinated GPU reservation held by the caller (" + why + "); use nr-gpu-lock.py run --owner ... -- nr-run-recipe.py run ..."); sys.exit(4)
    print("reservation:", why)
    out = a.out; os.makedirs(out, exist_ok=False); floor = float(R["storage_floor_gib"]); shutil.copyfile(a.recipe, os.path.join(out, "recipe-as-run.json"))   # v8: the recipe exactly as run travels with the receipts (verifiers must not depend on the live spec file)
    free = shutil.disk_usage(ROOT).free / 2**30; print(f"storage free {free:.1f} GiB, floor {floor} GiB")
    reps = R["replicas"]; ad = R.get("admission")
    if ad:   # v5: recipe-pinned per-process budget, checked with a fresh reservation + free-space read before EVERY process; nothing is deleted, every output is retained
        budget = (ad["per_frame_mib"] * len(R["frames"]) + ad["fixed_mib"]) * ad["margin"] / 1024.0; results = []; log = open(os.path.join(out, "admission.log"), "a")
        for k, rep in enumerate(reps):
            okl, why = held_reservation(a.owner); free = shutil.disk_usage(ROOT).free / 2**30
            line = f"{__import__('datetime').datetime.utcnow().isoformat()}Z process {k+1}/{len(reps)} {rep['tag']}: reservation {'held' if okl else 'NOT held'} ({why}); free {free:.2f} GiB, budget {budget:.2f} GiB ({ad['per_frame_mib']} MiB x {len(R['frames'])} frames + {ad['fixed_mib']} MiB, x{ad['margin']}), floor {floor} GiB"; print(line); log.write(line + "\n"); log.flush()
            if not okl and not a.no_lock_check: print("REFUSED: reservation not held before process", k + 1); break
            if free - budget < floor: print(f"STORAGE ADMISSION REFUSED before process {k+1} (free {free:.2f} - budget {budget:.2f} < floor {floor})"); log.write("REFUSED\n"); break
            r = run_one(R, out, rep); actual = dir_size_gib(os.path.join(out, rep["tag"])); r["actual_output_gib"] = round(actual, 4); r["budget_gib"] = round(budget, 4); r["overrun"] = actual > budget; results.append(r)
            line = f"  measured {actual:.3f} GiB for {rep['tag']} (budget {budget:.2f}): {'OVERRUN' if r['overrun'] else 'within budget'}"; print(rep["tag"], r["passed"], r["frame_status"]); print(line); log.write(line + "\n"); log.flush()
            if r["overrun"]:   # fail closed: the derived budget was wrong for this configuration; stop, keep everything, write the overrun receipt
                json.dump({"schema": "neuralstudio.nr.admission-overrun.v1", "recipe": a.recipe, "recipe_sha256": sha(a.recipe), "process": rep["tag"], "actual_output_gib": actual, "budget_gib": budget, "admission": ad, "free_gib_before": free, "floor_gib": floor, "action": "configuration stopped; outputs retained; budget must be re-derived and re-reviewed"}, open(os.path.join(out, "ADMISSION-OVERRUN.json"), "w"), indent=1)
                print("ADMISSION OVERRUN: stopping the configuration; receipt written"); break
            if not r["passed"]: print("STOPPED at the first failed replica; evidence preserved"); break
        S = {"schema": "neuralstudio.nr.recipe-run.v7", "admission_overrun": any(r.get("overrun") for r in results), "runner_sha256": RUNNER_SHA, "recipe": a.recipe, "recipe_sha256": sha(a.recipe), "name": R["name"], "harness_extra_args": list(R.get("harness_extra_args", [])), "admission": ad, "admission_budget_gib": budget, "execution_completed": True, "runs": results, "replicas_requested": len(reps), "replicas_passed": sum(1 for r in results if r["passed"]), "all_replicas_passed": len(results) == len(reps) and all(r["passed"] for r in results) and not any(r.get("overrun") for r in results)}
        json.dump(S, open(os.path.join(out, "run-summary.json"), "w"), indent=1); print("RUN", "ALL PASS" if S["all_replicas_passed"] else "NOT ALL PASS", "->", os.path.join(out, "run-summary.json")); sys.exit(0 if S["all_replicas_passed"] else 1)
    if free - 3.0 < floor: print("STORAGE ADMISSION REFUSED for the pilot"); sys.exit(3)
    results = [run_one(R, out, reps[0])]; print("pilot", results[0]["passed"], results[0]["frame_status"])
    if results[0]["passed"] and len(reps) > 1:
        size = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(os.path.join(out, reps[0]["tag"])) for f in fs) / 2**30; free = shutil.disk_usage(ROOT).free / 2**30
        if free - size * (len(reps) - 1) < floor: print(f"STORAGE ADMISSION REFUSED for replicas (pilot {size:.2f} GiB x {len(reps)-1}, free {free:.1f})")
        else:
            for rep in reps[1:]:
                r = run_one(R, out, rep); results.append(r); print(rep["tag"], r["passed"], r["frame_status"])
                if not r["passed"]: print("STOPPED at the first failed replica; evidence preserved"); break
    S = {"schema": "neuralstudio.nr.recipe-run.v3", "runner_sha256": RUNNER_SHA, "recipe": a.recipe, "recipe_sha256": sha(a.recipe), "name": R["name"], "harness_extra_args": list(R.get("harness_extra_args", [])), "execution_completed": True, "runs": results, "replicas_requested": len(reps), "replicas_passed": sum(1 for r in results if r["passed"]), "all_replicas_passed": len(results) == len(reps) and all(r["passed"] for r in results), "audit_accepted": None, "promoted": None, "note": "audit_accepted and promoted are set only by the independent auditor / owner, never by this runner"}
    json.dump(S, open(os.path.join(out, "run-summary.json"), "w"), indent=1); print("RUN", "ALL PASS" if S["all_replicas_passed"] else "NOT ALL PASS", "->", os.path.join(out, "run-summary.json"))
    sys.exit(0 if S["all_replicas_passed"] else 1)
if __name__ == "__main__": main()
