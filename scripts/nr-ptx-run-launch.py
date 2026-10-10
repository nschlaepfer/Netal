#!/usr/bin/env python3
"""Run one recorded NR launch through its transliterated Metal kernel and compare the resulting pool image with the RTX oracle:
prepares the pre-launch pool image, the weight blob and the parameter bytes exactly as the emulator does, runs ptx_probe,
then reports mismatching bytes inside the oracle's changed set and bytes written outside it."""
import argparse, json, os, struct, subprocess, sys, hashlib, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("emu", os.path.join(ROOT, "scripts/nr-vit-emulate-launch.py")); emu = importlib.util.module_from_spec(spec); spec.loader.exec_module(emu)
from nr_vit_ref import tensor
def main():
    p = argparse.ArgumentParser(); p.add_argument("--launch", type=int, required=True); p.add_argument("--lib", required=True); p.add_argument("--probe", required=True); p.add_argument("--kernel", required=True)
    p.add_argument("--output", required=True); p.add_argument("--tables", default=os.path.join(ROOT, "build/nr_split/fixtures"))
    p.add_argument("--pool-pre", default="", help="pool image to start from (connected chain) instead of the recorded pre-launch state"); p.add_argument("--keep-post", action="store_true"); p.add_argument("--tex-in", default="", help="input texture file (RGBA16F 1920x1080); default capture/input-color (launch 155); launch 156 needs the produced launch-155 texture"); p.add_argument("--tex-poison", default="a5", help="poison byte (hex) for the output texture before the launch"); p.add_argument("--body", action="store_true", help="connected body on a poisoned pool image: pass requires exact oracle-changed bytes, no extra writes and clean coverage, not whole-image equality"); a = p.parse_args()
    if os.path.exists(a.output): raise SystemExit("preserve previous run: " + a.output)
    m = json.load(open(emu.MANIFEST)); L = next(l for l in m["launches"] if l["launch_id"] == f"launch_{a.launch:03d}")
    kname = {k["selection_id"]: k["name"] for k in m["selected_kernels"]}[L["selection_id"]]; weights = {w["destination"]["byte_offset"]: w for w in m["base_graph_weight_bindings"]}
    recorded_before = emu.pool_state(a.launch - 1); after = emu.pool_state(a.launch)
    before = bytearray(open(a.pool_pre, "rb").read()) if a.pool_pre else recorded_before
    params = bytearray(L["arguments"]["bytes"]); params[:] = bytes.fromhex(L["arguments"]["constant_bytes_hex"]); wtensor = None; wbytes = b"\0" * 16; textures = []; wbinds = []
    for bnd in L["arguments"]["bindings"]:
        t = bnd["target"]
        if t["id"] == "allocation_003": struct.pack_into("<Q", params, bnd["argument_offset"], emu.POOL + t["byte_offset"])
        elif t["id"] == "allocation_001": wbinds.append((bnd["argument_offset"], t["byte_offset"]))
        elif t["kind"] in ("texture", "surface"): struct.pack_into("<Q", params, bnd["argument_offset"], (0x300000000 if t["kind"] == "texture" else 0x400000000) + int(t["id"][-3:])); textures.append(t["id"])   # opaque non-null handles; the Metal kernel samples texture(0) / writes texture(1)
        else: raise SystemExit("unsupported binding " + str(t))
    if wbinds:      # one blob holding every weight tensor of the launch at its relative offset (the post block binds two adjacent records)
        base = min(o for _, o in wbinds); blobs = {o: tensor(weights[o]["tensor_name"]).tobytes() for _, o in wbinds}; wbytes = bytearray(max(o - base + len(b) for o, b in blobs.items()))
        for o, b in blobs.items(): wbytes[o - base:o - base + len(b)] = b
        for ao, o in wbinds: struct.pack_into("<Q", params, ao, emu.WBASE + (o - base))
        wtensor = "+".join(weights[o]["tensor_name"] for _, o in sorted(wbinds, key=lambda x: x[1])); wbytes = bytes(wbytes)
    os.makedirs(a.output); stage = os.path.join(a.output, "stage"); os.makedirs(stage)
    open(os.path.join(stage, "pool_pre.bin"), "wb").write(bytes(before)); open(os.path.join(stage, "weights.bin"), "wb").write(wbytes); open(os.path.join(stage, "params.bin"), "wb").write(bytes(params))
    g, b = L["grid"], L["block"]; run = os.path.join(a.output, "run")
    r = subprocess.run([a.probe, a.lib, a.kernel, os.path.join(stage, "pool_pre.bin"), os.path.join(stage, "weights.bin"), os.path.join(stage, "params.bin"), a.tables, f"{g[0]},{g[1]},{g[2]}", f"{b[0]},{b[1]},{b[2]}", run] + ([a.tex_in or os.path.join(emu.ORACLE, "capture/input-color.rgba16le"), "1920,1080", a.tex_poison] if textures else []), capture_output=True, text=True)
    print(r.stdout.strip()); 
    if r.returncode != 0: print(r.stderr[-2000:]); raise SystemExit("probe failed")
    mine = np.frombuffer(open(os.path.join(run, "pool_post.bin"), "rb").read(), dtype=np.uint8); pre = np.frombuffer(bytes(before), dtype=np.uint8); orc = np.frombuffer(bytes(after), dtype=np.uint8); rpre = np.frombuffer(bytes(recorded_before), dtype=np.uint8)
    changed = np.nonzero(orc != rpre)[0]; written = np.nonzero(mine != pre)[0]; mism = changed[mine[changed] != orc[changed]]; extra = np.setdiff1d(written, changed)
    whole_image_mismatch = int((mine != orc).sum())          # connected chains: every byte of the pool must equal the oracle's post-launch image
    extra_bad = extra[mine[extra] != orc[extra]]              # body mode: bytes written where the oracle changed nothing must still equal the oracle post-state (same-value writes are legitimate)
    cov = np.frombuffer(open(os.path.join(run, "cov.bin"), "rb").read(), dtype=np.uint32)          # per 4-byte word
    tex = {}
    if textures:      # the final output image: texture_004 as written by the post block; launch 156 copies it to surface_001 = capture/output.rgba16le (identity copy assumed)
        to = np.fromfile(os.path.join(run, "tex_out.bin"), dtype=np.uint16); ref = np.fromfile(os.path.join(emu.ORACLE, "capture/output.rgba16le"), dtype=np.uint16)
        bad = np.nonzero(to != ref)[0]; pb = int(a.tex_poison, 16); unwritten = int((to == (pb | (pb << 8))).sum())
        tcov = np.fromfile(os.path.join(run, "tex_coverage.bin"), dtype=np.uint32); pr = json.load(open(os.path.join(run, "run.json")))
        tex = {"textures": textures, "reference": "capture/output.rgba16le" + (" (final surface_001: the recorded output itself)" if "surface_001" in textures else " (texture_004 vs the recorded final surface: informational; the strict gate is launch 156)"), "input_texture": a.tex_in or "capture/input-color.rgba16le", "values": int(len(ref)), "mismatching_values": int(len(bad)), "first_mismatches": [{"x": int(i // 4 % 1920), "y": int(i // 4 // 1920), "c": int(i % 4), "mine": f"{int(to[i]):04x}", "oracle": f"{int(ref[i]):04x}"} for i in bad[:12]],
               "poison_byte": a.tex_poison, "poison_values_left": unwritten, "pixels": int(tcov.size), "pixels_not_written": int((tcov == 0).sum()), "pixels_written_more_than_once": int((tcov > 1).sum()),
               "input_texture_immutable": pr.get("texture_in_sha256") == pr.get("texture_in_after_sha256") and bool(pr.get("texture_in_sha256")), "output_sha256": hashlib.sha256(to.tobytes()).hexdigest()}
        rep_tex_ok = len(bad) == 0 and tex["pixels_not_written"] == 0 and tex["pixels_written_more_than_once"] == 0 and tex["input_texture_immutable"]
    else: rep_tex_ok = True
    cwords = np.unique(changed // 4); cov_words = np.nonzero(cov)[0]
    cov_missing = int((cov[cwords] == 0).sum()); cov_multi = int((cov > 1).sum()); cov_outside = int(len(np.setdiff1d(cov_words, cwords)))
    os.remove(os.path.join(run, "cov.bin"))
    rep = {"schema": "neuralstudio.nr.ptx-launch-check.v1", "launch": a.launch, "kernel": kname, "metal_kernel": a.kernel, "weights": wtensor, "grid": g, "block": b, "oracle_changed_bytes": int(len(changed)),
           "mismatching_bytes": int(len(mism)), "first_mismatch_offsets": mism[:16].tolist(), "bytes_written_outside_oracle_change": int(len(extra)), "first_extra_offsets": extra[:8].tolist(), "extra_bytes_not_equal_oracle_post": int(len(extra_bad)), "first_extra_bad_offsets": extra_bad[:8].tolist(), "whole_image_mismatch": whole_image_mismatch, "pool_pre_source": a.pool_pre or "recorded",
           "texture": tex, "coverage": {"oracle_changed_words": int(len(cwords)), "changed_words_not_written": cov_missing, "words_written_more_than_once": cov_multi, "written_words_outside_oracle_change": cov_outside},
           "probe": json.load(open(os.path.join(run, "run.json"))), "pre_sha256": hashlib.sha256(before).hexdigest(), "post_sha256": hashlib.sha256(after).hexdigest(), "emulated_post_sha256": hashlib.sha256(mine.tobytes()).hexdigest(),
           "result": "pass" if len(mism) == 0 and (len(extra) == 0 or (a.body and len(extra_bad) == 0)) and cov_missing == 0 and cov_multi == 0 and cov_outside == 0 and (a.body or whole_image_mismatch == 0) and rep_tex_ok else "fail", "body_mode": bool(a.body)}
    json.dump(rep, open(os.path.join(a.output, "report.json"), "w"), indent=1); os.remove(os.path.join(stage, "pool_pre.bin"))
    if not a.keep_post: os.remove(os.path.join(run, "pool_post.bin"))
    print(json.dumps({k: rep[k] for k in ("launch", "kernel", "oracle_changed_bytes", "mismatching_bytes", "bytes_written_outside_oracle_change", "coverage", "texture", "result")}))
if __name__ == "__main__": main()
