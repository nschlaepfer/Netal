#!/usr/bin/env python3
"""CPU emulation of launch 155 (post block) for chosen CTAs on Codex's archived second-frame packet (temporal branch active), with the PTX warp
emulator's models; compares the emulated 8x8 tiles with the captured expected output and (optionally) a Metal replay output. Hypothesis switches
come from the emulator's environment (EMU_SUST_RNE, EMU_EX2_POS, ...). Diagnostic only.
Usage: nr-post155-emulate-cta.py --packet DIR --params-base F --weights F --ctas "1,0 60,40" --out DIR [--metal-output F]"""
import argparse, importlib.util, json, os, struct, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, "scripts"))
spec = importlib.util.spec_from_file_location("PW", os.path.join(ROOT, "scripts/ptx_warp_emu.py")); PW = importlib.util.module_from_spec(spec); spec.loader.exec_module(PW)
POOL = 0x100000000; WBASE = 0x200000000
def half_img(path, ch):
    a = np.fromfile(path, dtype=np.uint16).reshape(1080, 1920, ch).view(np.float16).astype(np.float32)
    if ch == 2: a = np.concatenate([a, np.zeros((1080, 1920, 1), np.float32), np.ones((1080, 1920, 1), np.float32)], axis=2)   # RG16F sampled as (r, g, 0, 1)
    return a
def main():
    p = argparse.ArgumentParser(); p.add_argument("--packet", required=True); p.add_argument("--params-base", required=True); p.add_argument("--weights", required=True); p.add_argument("--ctas", default="1,0"); p.add_argument("--out", required=True); p.add_argument("--metal-output", default="")
    a = p.parse_args(); os.makedirs(a.out, exist_ok=True); pk = a.packet
    ptx = open(os.path.join(ROOT, "build/nr_dec/ptx/cc_tinlayout_fused_post_block_swin_1h_32_fp8.ptx")).read(); entry = "cc_tinlayout_fused_post_block_swin_1h_32_fp8"
    base = bytearray(open(a.params_base, "rb").read()); assert len(base) == 184
    struct.pack_into("<I", base, 88, 1); struct.pack_into("<I", base, 96, 6)
    for off, v in ((124, 1920), (128, 1080), (132, 1 / 1920), (136, 1 / 1080), (148, 1920), (152, 1080), (156, 1 / 1920), (160, 1 / 1080)): struct.pack_into("<f", base, off, float(v))
    hcol = struct.unpack_from("<Q", base, 56)[0]; hsurf = struct.unpack_from("<Q", base, 16)[0]; hhist = struct.unpack_from("<Q", base, 88)[0]; hmot = struct.unpack_from("<Q", base, 96)[0]
    PW.TEXTURES.clear(); PW.OUT_IMAGES.clear()
    PW.TEXTURES[hcol] = (half_img(os.path.join(pk, "input-color.rgba16le"), 4), 1920, 1080); PW.TEXTURES[hhist] = (half_img(os.path.join(pk, "history.rgba16le"), 4), 1920, 1080); PW.TEXTURES[hmot] = (half_img(os.path.join(pk, "motion.rg16le"), 2), 1920, 1080)
    PW.OUT_IMAGES[hsurf] = {"img": np.full((1080, 1920, 4), 0xa5a5, dtype=np.uint16), "written": np.zeros((1080, 1920), dtype=bool)}
    gmem = {POOL: bytearray(open(os.path.join(pk, "post-input-pool.bin"), "rb").read()), WBASE: bytearray(open(a.weights, "rb").read())}
    tables = PW.load_tables(os.path.join(ROOT, "build/nr_vit/tables")); expected = np.fromfile(os.path.join(pk, "expected-output.rgba16le"), dtype=np.uint16).reshape(1080, 1920, 4)
    metal = np.fromfile(a.metal_output, dtype=np.uint16).reshape(1080, 1920, 4) if a.metal_output else None
    report = {"schema": "neuralstudio.nr.post155-cta-emulation.v1", "env": {k: v for k, v in os.environ.items() if k.startswith("EMU_")}, "ctas": []}
    for spec_ in a.ctas.split():
        cx, cy = [int(v) for v in spec_.split(",")]; t0 = time.time()
        PW.run_cta(ptx, entry, bytes(base), (cx, cy, 0), (32, 1, 1), (241, 145, 1), gmem, tables)
        wr = PW.OUT_IMAGES[hsurf]["written"]; ys, xs = np.nonzero(wr); tile = {"cta": [cx, cy], "pixels_written": int(wr.sum()), "seconds": round(time.time() - t0, 1)}
        if wr.sum():
            got = PW.OUT_IMAGES[hsurf]["img"][wr]; exp = expected[wr]; tile["bbox"] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            tile["mismatch_vs_expected"] = int((got != exp).sum()); tile["values"] = int(got.size)
            if metal is not None: tile["mismatch_vs_metal"] = int((got != metal[wr]).sum()); tile["metal_vs_expected"] = int((metal[wr] != exp).sum())
            mm = np.nonzero((got != exp).any(axis=1))[0][:4]; tile["first"] = [{"pixel": [int(xs[i]), int(ys[i])], "emu": [hex(int(v)) for v in got[i]], "expected": [hex(int(v)) for v in exp[i]], "metal": ([hex(int(v)) for v in metal[ys[i], xs[i]]] if metal is not None else None)} for i in mm]
        PW.OUT_IMAGES[hsurf]["written"][:] = False; report["ctas"].append(tile); print(json.dumps(tile))
    json.dump(report, open(os.path.join(a.out, "emulation.json"), "w"), indent=1)
if __name__ == "__main__": main()
