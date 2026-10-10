#!/usr/bin/env python3
"""Live checked-mode checkpoint identity between recipe runs of the SAME live case on the v11 per-launch body and the resident body (v13+), both with
--body-checked. v2 (Codex 21:44): the pass is gated on the DECLARED inventory - both recipes must exist, name the same frames (inputs, references),
motion, reset sequence and sampler variant, differ only in harness build/descriptor/name, declare a non-empty replica list with identical tags and
poison patterns, and EVERY declared replica/frame must have a receipt frame with body_mode "checked", body code 0, comparison_passed, the declared
poison, and a chain-run.json on both sides; any missing item fails the gate (an empty directory can never pass). Several run directories per side may
be given (e.g. the base pair plus the complementary-poison pair) - the union must cover every declared replica of every recipe.
Usage: nr-live-checked-compare.py OUT.json --v11 RECIPE.json RUN_DIR [--v11 ...] --resident RECIPE.json RUN_DIR [--resident ...]"""
import json, os, subprocess, sys
here = os.path.dirname(os.path.abspath(__file__))
def poison_hex(v):   # receipts carry the poison byte as an integer (165 = a5, 90 = 5a) or as hex text
    try: return format(int(v), "02x") if not (isinstance(v, str) and not v.isdigit()) else v.lower().replace("0x", "")
    except Exception: return str(v)
def parse(argv):
    out = argv[1]; sides = {"v11": [], "resident": []}; i = 2
    while i < len(argv):
        k = argv[i]; assert k in ("--v11", "--resident"), k; sides[k[2:]].append((argv[i + 1], argv[i + 2])); i += 3
    return out, sides
out, sides = parse(sys.argv); problems = []; pairs = []
def load(side):
    inv = {}   # (replica tag) -> {recipe, run_dir, poisons, frames}
    for rec_path, run_dir in sides[side]:
        R = json.load(open(rec_path)); S = os.path.join(run_dir, "run-summary.json")
        if not os.path.exists(S): problems.append(f"{side}: {run_dir} has no run-summary.json"); continue
        summ = json.load(open(S))
        if summ.get("recipe_sha256") and summ["recipe_sha256"] != __import__("hashlib").sha256(open(rec_path, "rb").read()).hexdigest(): problems.append(f"{side}: run-summary recipe sha differs from {rec_path}")
        if "--body-checked" not in R.get("harness_extra_args", []): problems.append(f"{side}: recipe {rec_path} does not request --body-checked")
        for rep in R["replicas"]:
            tag = rep["tag"]; d = os.path.join(run_dir, tag); rc = os.path.join(d, "receipt.json")
            if tag in inv: problems.append(f"{side}: replica tag {tag} declared twice across the given runs"); continue
            entry = {"recipe": rec_path, "run_dir": run_dir, "poisons": rep["poisons"], "frames": {}, "key": (json.dumps(R["frames"], sort_keys=True), json.dumps(R["motion"], sort_keys=True), R["reset_sequence"], R["encoder_sampler_variant"], R["encoder_object_sha256"])}
            if not os.path.exists(rc): problems.append(f"{side}: replica {tag} of {rec_path} has no receipt ({rc})"); inv[tag] = entry; continue
            frames = json.load(open(rc)).get("frames", [])
            if len(frames) != len(R["frames"]): problems.append(f"{side}: replica {tag}: receipt has {len(frames)} frames, recipe declares {len(R['frames'])}")
            for k, fr in enumerate(frames):
                ok = fr.get("body_mode") == "checked" and (fr.get("body") or {}).get("code") == 0 and fr.get("status") == "comparison_passed" and poison_hex(fr.get("poison")) == rep["poisons"][k].lower()
                cr = os.path.join(d, "diag", f"frame-{k}", "chain-run.json")
                if not ok: problems.append(f"{side}: replica {tag} frame {k}: body_mode={fr.get('body_mode')} code={(fr.get('body') or {}).get('code')} status={fr.get('status')} poison={fr.get('poison')} (declared {rep['poisons'][k]})")
                if not os.path.exists(cr): problems.append(f"{side}: replica {tag} frame {k}: missing {cr}")
                entry["frames"][k] = cr
            inv[tag] = entry
    return inv
a, b = load("v11"), load("resident")
if not a or not b: problems.append("empty declared inventory on at least one side")
if set(a) != set(b): problems.append(f"replica tags differ: v11 {sorted(a)} vs resident {sorted(b)}")
keys = {e["key"] for e in list(a.values()) + list(b.values())}
if len(keys) > 1: problems.append("the recipes do not describe the same live case (frames/motion/reset/sampler/encoder object differ)")
rows = []
for tag in sorted(set(a) & set(b)):
    ea, eb = a[tag], b[tag]
    if ea["poisons"] != eb["poisons"]: problems.append(f"replica {tag}: poison patterns differ between sides")
    for k in range(len(ea["poisons"])):
        f11, f12 = ea["frames"].get(k), eb["frames"].get(k)
        if not (f11 and f12 and os.path.exists(f11) and os.path.exists(f12)): rows.append({"replica": tag, "frame": k, "poison": ea["poisons"][k], "ok": False, "why": "missing chain-run"}); continue
        tmp = os.path.join(out + ".d", f"{tag}-frame-{k}.json"); os.makedirs(out + ".d", exist_ok=True)
        subprocess.run([sys.executable, os.path.join(here, "nr-v12-checked-compare.py"), f11, f12, tmp], capture_output=True, text=True); c = json.load(open(tmp))
        ok = c["all_compared_match"] and not c["uncompared"] and c["matched"] >= 152
        rows.append({"replica": tag, "frame": k, "poison": ea["poisons"][k], "matched": c["matched"], "mismatched": len(c["mismatched"]), "uncompared": len(c["uncompared"]), "ok": ok, "detail": tmp}); print(tag, "frame", k, ea["poisons"][k], "matched", c["matched"], "mismatched", len(c["mismatched"]), "uncompared", len(c["uncompared"]))
# poison coverage per frame slot: every slot must be observed under both poisons (union over replicas)
slots = {}
for tag, e in a.items():
    for k, p in enumerate(e["poisons"]): slots.setdefault(k, set()).add(p)
coverage = {str(k): sorted(v) for k, v in sorted(slots.items())}; both = all(v == {"5a", "a5"} for v in slots.values()) and bool(slots)
declared = sum(len(e["poisons"]) for e in a.values()); passed = sum(1 for r in rows if r["ok"])
all_ok = not problems and bool(rows) and passed == declared == len(rows) and both
S = {"schema": "neuralstudio.nr.live-checked-compare.v2", "declared_frames": declared, "compared_frames": len(rows), "passed_frames": passed, "replicas": sorted(a), "poison_coverage_per_frame_slot": coverage, "both_poisons_every_slot": both, "problems": problems, "frames": rows, "all_ok": all_ok}
json.dump(S, open(out, "w"), indent=1)
print("LIVE CHECKED IDENTITY", "ALL MATCH (declared inventory complete, both poisons on every slot)" if all_ok else "NOT ACCEPTED", f"{passed}/{declared} declared frames; problems: {len(problems)}; both poisons every slot: {both}")
sys.exit(0 if all_ok else 1)
