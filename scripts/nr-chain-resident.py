#!/usr/bin/env python3
"""Run the decoder/output launches 133-156 GPU-resident (ptx_chain_probe) on the same poisoned live-input pool image and parameter bytes
as the strict per-launch harness, then check: every per-launch pool hash equals the reference strict attempt's per-launch post hash, the
final surface equals the recording, pool words and texture pixels written exactly once, weights/params/input texture immutable. Reports GPU
vs wall time against the reference attempt's harness seconds. Comparisons are CPU-only, after execution."""
import argparse, hashlib, json, os, struct, subprocess, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import importlib.util
spec = importlib.util.spec_from_file_location("emu_launch", os.path.join(ROOT, "scripts/nr-vit-emulate-launch.py")); E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)
from nr_vit_ref import tensor
SKIPS = [(90827776, 2211840, "block22-native.e4m3"), (93285376, 4423680, "block14-native.e4m3"), (97709056, 8847360, "block8-native.e4m3"), (141945856, 17694720, "block4-native.e4m3")]
MIDDLE = (123145216, 1105920); BLOCK0_SKIP = (388096, 71166976); CONTROL = (233984, 387616); POOL_BYTES = 212519936
def sha(b): return hashlib.sha256(b).hexdigest()
def main():
    p = argparse.ArgumentParser(); p.add_argument("--stage-only", action="store_true", help="write only stage/params-N.bin and stage/weights-N.bin (manifest-derived; no encoder run, no pool image, no chain.json)"); p.add_argument("--attempt", required=True); p.add_argument("--reference-body", required=True, help="strict attempt body dir (body-<key>) whose per-launch post hashes must be reproduced")
    p.add_argument("--encoder-run", required=True); p.add_argument("--block0-skip", required=True); p.add_argument("--middle-out", default="", help="middle output view file; omit with --chain-only for the fused runner (the view stays resident)"); p.add_argument("--chain-only", action="store_true", help="write chain.json (with the pool patches list) and exit"); p.add_argument("--poison", required=True)
    p.add_argument("--production", action="store_true", help="production mode: no per-launch pool hash/coverage scan in the probe; final pool hash, final surface, texture exactly-once and immutability still checked; the per-launch (checked) mode remains the acceptance instrument"); p.add_argument("--tex-in", default="", help="input colour texture (RGBA16F 1920x1080 raw) for the post block; default: the pinned capture"); p.add_argument("--reference-frame", default="", help="expected final surface (RGBA16F raw) for the CPU comparison; default: the pinned recording"); p.add_argument("--dec-lib", required=True); p.add_argument("--post-lib", required=True); p.add_argument("--copy-lib", required=True); p.add_argument("--probe", required=True); p.add_argument("--tables", default=os.path.join(ROOT, "build/nr_ptx/tables"))
    a = p.parse_args(); att = os.path.abspath(a.attempt)
    if os.path.exists(att): raise SystemExit("preserve previous attempt: " + att)
    os.makedirs(os.path.join(att, "stage")); poison = int(a.poison, 16)
    M = json.load(open(E.MANIFEST)); kn = {k["selection_id"]: k["name"] for k in M["selected_kernels"]}; weights = {w["destination"]["byte_offset"]: w for w in M["base_graph_weight_bindings"]}
    # --stage-only (2026-09-07 recovery): the 24 launch parameter/weight files depend ONLY on the pinned graph manifest and weight tensors, not on any
    # encoder run; this mode regenerates them (and nothing else) when the archived encoder outputs are unavailable. The normal path is unchanged.
    ne = {} if a.stage_only else json.load(open(os.path.join(a.encoder_run, "native_exports.json")))["outputs"]
    img = bytearray([poison]) * POOL_BYTES; inputs = []
    for off, n, f in ([] if a.stage_only else SKIPS):
        b = open(os.path.join(a.encoder_run, f), "rb").read(); assert len(b) == n and sha(b) == ne[f]["sha256"], f; img[off:off + n] = b; inputs.append({"file": f, "sha256": sha(b), "live": True})
    if a.middle_out: b = open(a.middle_out, "rb").read(); assert len(b) == MIDDLE[1]; img[MIDDLE[0]:MIDDLE[0] + MIDDLE[1]] = b; inputs.append({"file": a.middle_out, "sha256": sha(b), "live": True})
    elif not (a.chain_only or a.stage_only): raise SystemExit("--middle-out is required unless --chain-only")
    if a.stage_only: pass
    else: b = open(a.block0_skip, "rb").read(); assert len(b) == BLOCK0_SKIP[1] - BLOCK0_SKIP[0]; img[BLOCK0_SKIP[0]:BLOCK0_SKIP[1]] = b; inputs.append({"file": a.block0_skip, "sha256": sha(b), "live": True})
    img[CONTROL[0]:CONTROL[1]] = b"\xff" * (CONTROL[1] - CONTROL[0]); pre = os.path.join(att, "stage", "pool_pre.bin")
    if not a.stage_only: open(pre, "wb").write(bytes(img))   # stage-only: no 212 MB pool image is needed
    launches = []
    for n in range(133, 157):
        L = next(l for l in M["launches"] if l["launch_id"] == f"launch_{n:03d}"); kname = kn[L["selection_id"]]
        params = bytearray(L["arguments"]["bytes"]); params[:] = bytes.fromhex(L["arguments"]["constant_bytes_hex"]); wbinds = []; textures = []
        for bnd in L["arguments"]["bindings"]:
            t = bnd["target"]
            if t["id"] == "allocation_003": struct.pack_into("<Q", params, bnd["argument_offset"], E.POOL + t["byte_offset"])
            elif t["id"] == "allocation_001": wbinds.append((bnd["argument_offset"], t["byte_offset"]))
            elif t["kind"] in ("texture", "surface"): struct.pack_into("<Q", params, bnd["argument_offset"], (0x300000000 if t["kind"] == "texture" else 0x400000000) + int(t["id"][-3:])); textures.append(t["id"])
        wbytes = b"\0" * 16
        if wbinds:
            base = min(o for _, o in wbinds); blobs = {o: tensor(weights[o]["tensor_name"]).tobytes() for _, o in wbinds}; wb = bytearray(max(o - base + len(x) for o, x in blobs.items()))
            for o, x in blobs.items(): wb[o - base:o - base + len(x)] = x
            for ao, o in wbinds: struct.pack_into("<Q", params, ao, E.WBASE + (o - base))
            wbytes = bytes(wb)
        pp = os.path.join(att, "stage", f"params-{n}.bin"); wp = os.path.join(att, "stage", f"weights-{n}.bin"); open(pp, "wb").write(bytes(params)); open(wp, "wb").write(wbytes)
        short = "ptx_cg2r_copy" if kname == "cg2r_copy_kernel" else "ptx_" + kname[len("cc_tinlayout_fused_"):-4]
        lib = a.copy_lib if n == 156 else a.post_lib if n == 155 else a.dec_lib
        launches.append({"launch": n, "lib": os.path.abspath(lib), "kernel": short, "params": pp, "weights": wp, "grid": L["grid"], "block": L["block"], "textures": "post" if n == 155 else "copy" if n == 156 else ""})
    if a.stage_only:
        import glob as _g; fs = sorted(_g.glob(os.path.join(att, "stage", "*.bin")))
        json.dump({"schema": "neuralstudio.nr.decoder-stage.v1", "manifest": E.MANIFEST, "files": {os.path.basename(f): sha(open(f, "rb").read()) for f in fs}}, open(os.path.join(att, "stage-receipt.json"), "w"), indent=1)
        print("stage-only:", len(fs), "files ->", os.path.join(att, "stage")); return
    chain = {"pool_pre": pre, "tables": os.path.abspath(a.tables), "poison": a.poison, "tex_in": os.path.abspath(a.tex_in) if a.tex_in else os.path.join(E.ORACLE, "capture/input-color.rgba16le"), "tex_w": 1920, "tex_h": 1080, "checks": "final" if a.production else "per-launch", "launches": launches,
             "patches": [{"file": os.path.join(a.encoder_run, f), "offset": off, "bytes": n} for off, n, f in SKIPS] + [{"file": a.block0_skip, "offset": BLOCK0_SKIP[0], "bytes": BLOCK0_SKIP[1] - BLOCK0_SKIP[0]}], "control": list(CONTROL), "inputs": inputs}
    cj = os.path.join(att, "chain.json"); json.dump(chain, open(cj, "w"), indent=1)
    if a.chain_only: print("chain", cj); return
    t0 = time.time(); r = subprocess.run([a.probe, cj, os.path.join(att, "run")], capture_output=True, text=True); wall = time.time() - t0
    open(os.path.join(att, "probe.log"), "w").write(r.stdout + r.stderr)
    if r.returncode: raise SystemExit("chain probe failed: " + r.stderr[-800:])
    run = json.load(open(os.path.join(att, "run", "chain-run.json"))); ref = a.reference_body; failures = []; per = []
    for L in run["launches"]:
        n = L["launch"]; rp = os.path.join(ref, f"launch-{n:03d}", "report.json"); rr = json.load(open(rp))
        same = True if a.production else rr["emulated_post_sha256"] == L["pool_post_sha256"]; cov_ok = True if a.production else (L["words_written_more_than_once"] == 0 and L["words_written_once"] >= rr["coverage"]["oracle_changed_words"])
        item = {"launch": n, "pool_hash_matches_reference": same, "gpu_ms": L["gpu_ms"], "wall_ms": L["wall_ms"], "reference_gpu_ms": rr["probe"]["gpu_ms"], "words_once": L["words_written_once"], "words_multi": L["words_written_more_than_once"], "weights_immutable": L["weights_immutable"], "params_immutable": L["params_immutable"]}
        if not same: failures.append(f"launch {n}: pool hash differs from the strict reference")
        if not cov_ok: failures.append(f"launch {n}: coverage")
        if not (L["weights_immutable"] and L["params_immutable"]): failures.append(f"launch {n}: immutability")
        if "texture" in L:
            t = L["texture"]; item["texture"] = t
            if t["pixels_once"] != t["pixels"] or t["pixels_multi"] != 0: failures.append(f"launch {n}: texture not written exactly once")
            if n == 156:
                to = np.fromfile(os.path.join(att, "run", "tex_out_156.bin"), dtype=np.uint16); refimg = np.fromfile(a.reference_frame if a.reference_frame else os.path.join(E.ORACLE, "capture/output.rgba16le"), dtype=np.uint16)
                item["final_surface_mismatching_values"] = int((to != refimg).sum()); item["final_surface_sha256"] = t["output_sha256"]
                if item["final_surface_mismatching_values"]: failures.append("final surface differs from the recording")
            if n == 155 and t["input_after_sha256"] != run["texture_in_sha256"]: failures.append("input colour texture changed")
        per.append(item)
    last = max(run["launches"], key=lambda L: L["launch"])["launch"]; ref_last = json.load(open(os.path.join(ref, f"launch-{last:03d}", "report.json")))["emulated_post_sha256"]
    final_pool_ok = run.get("pool_final_sha256") == ref_last
    if not final_pool_ok: failures.append("final pool hash differs from the strict reference's last launch")
    ref_body = json.load(open(os.path.join(ref, "body.json"))); ref_harness = None
    fc = os.path.join(os.path.dirname(ref.rstrip("/")), "full-chain.json")
    if os.path.exists(fc): ref_harness = next((s["seconds"] for s in json.load(open(fc))["steps"] if s["label"] == "body-" + os.path.basename(ref.rstrip("/"))[5:]), None)
    receipt = {"schema": "neuralstudio.nr.chain-resident.v1", "attempt": att, "reference_body": os.path.abspath(ref), "reference_body_sha256": sha(open(os.path.join(ref, "body.json"), "rb").read()), "inputs": inputs, "poison": a.poison, "launches": per,
               "mode": "production" if a.production else "checked", "tex_in": os.path.abspath(a.tex_in) if a.tex_in else "pinned capture", "reference_frame": os.path.abspath(a.reference_frame) if a.reference_frame else "pinned recording", "checks_skipped_in_production": ["per-launch pool hash", "per-launch pool coverage scan"] if a.production else [], "final_pool_sha256": run.get("pool_final_sha256"), "final_pool_matches_reference": final_pool_ok, "gpu_ms_total": run["gpu_ms_total"], "probe_wall_ms_total": run["wall_ms_total"], "driver_wall_s": round(wall, 1), "reference_harness_s": ref_harness, "reference_gpu_ms_total": sum((l["gpu_ms"] or 0) for l in ref_body["launches"]),
               "failures": failures, "result": "pass" if not failures else "fail"}
    receipt["receipt_root_sha256"] = hashlib.sha256(json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    json.dump(receipt, open(os.path.join(att, "resident.json"), "w"), indent=1)
    print("resident chain", receipt["result"], "failures", failures); print(f"GPU {run['gpu_ms_total']/1000:.2f} s, probe wall {run['wall_ms_total']/1000:.2f} s, driver wall {wall:.1f} s; reference GPU {receipt['reference_gpu_ms_total']/1000:.2f} s, reference harness {ref_harness} s")
    sys.exit(0 if not failures else 1)
if __name__ == "__main__": main()
