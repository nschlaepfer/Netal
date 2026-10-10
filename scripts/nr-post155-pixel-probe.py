#!/usr/bin/env python3
"""Diagnostic: emulate one CTA of launch 155 on Codex's captured second-frame packet and dump the tail-copy temporal registers of one lane
(taps, Catmull-Rom weights, normaliser, blend) as f32 bit patterns, then a 1-ulp sensitivity table of the stored value. Reuses the driver's setup.
Usage: nr-post155-pixel-probe.py --packet DIR --params-base F --weights F --cta X,Y --warp W --lane L --out F"""
import argparse, importlib.util, json, os, struct, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, "scripts"))
spec = importlib.util.spec_from_file_location("drv", os.path.join(ROOT, "scripts/nr-post155-emulate-cta.py")); drv = importlib.util.module_from_spec(spec); spec.loader.exec_module(drv); PW = drv.PW
def f32(bits): return float(np.frombuffer(np.uint32(bits).tobytes(), dtype=np.float32)[0])
def bits(x): return int(np.frombuffer(np.float32(x).tobytes(), dtype=np.uint32)[0])
def rz16(x): return int(np.frombuffer(np.float32(x).tobytes(), dtype=np.uint32)[0]) >> 13 if False else _rz16(x)
def _rz16(x):
    b = bits(x); s = b >> 31; e = ((b >> 23) & 0xff) - 127; m = b & 0x7fffff
    if e < -14: return s << 15   # (subnormal handling irrelevant here)
    return (s << 15) | ((e + 15) << 10) | (m >> 13)
def main():
    p = argparse.ArgumentParser(); p.add_argument("--packet", required=True); p.add_argument("--params-base", required=True); p.add_argument("--weights", required=True); p.add_argument("--cta", default="53,91"); p.add_argument("--warp", type=int, default=0); p.add_argument("--lane", type=int, default=13); p.add_argument("--out", required=True)
    a = p.parse_args(); pk = a.packet
    ptx = open(os.path.join(ROOT, "build/nr_dec/ptx/cc_tinlayout_fused_post_block_swin_1h_32_fp8.ptx")).read(); entry = "cc_tinlayout_fused_post_block_swin_1h_32_fp8"
    base = bytearray(open(a.params_base, "rb").read()); struct.pack_into("<I", base, 88, 1); struct.pack_into("<I", base, 96, 6)
    for off, v in ((124, 1920), (128, 1080), (132, 1 / 1920), (136, 1 / 1080), (148, 1920), (152, 1080), (156, 1 / 1920), (160, 1 / 1080)): struct.pack_into("<f", base, off, float(v))
    hcol = struct.unpack_from("<Q", base, 56)[0]; hsurf = struct.unpack_from("<Q", base, 16)[0]; hhist = struct.unpack_from("<Q", base, 88)[0]; hmot = struct.unpack_from("<Q", base, 96)[0]
    PW.TEXTURES.clear(); PW.OUT_IMAGES.clear()
    PW.TEXTURES[hcol] = (drv.half_img(os.path.join(pk, "input-color.rgba16le"), 4), 1920, 1080); PW.TEXTURES[hhist] = (drv.half_img(os.path.join(pk, "history.rgba16le"), 4), 1920, 1080); PW.TEXTURES[hmot] = (drv.half_img(os.path.join(pk, "motion.rg16le"), 2), 1920, 1080)
    PW.OUT_IMAGES[hsurf] = {"img": np.full((1080, 1920, 4), 0xa5a5, dtype=np.uint16), "written": np.zeros((1080, 1920), dtype=bool)}
    gmem = {drv.POOL: bytearray(open(os.path.join(pk, "post-input-pool.bin"), "rb").read()), drv.WBASE: bytearray(open(a.weights, "rb").read())}
    tables = PW.load_tables(os.path.join(ROOT, "build/nr_vit/tables"))
    cx, cy = [int(v) for v in a.cta.split(",")]
    cta = PW.run_cta(ptx, entry, bytes(base), (cx, cy, 0), (32, 1, 1), (241, 145, 1), gmem, tables)
    R = cta.warps[a.warp].R; L = a.lane
    def g(n): return int(R[f"%r{n}"][L]) & 0xffffffff
    names = {"tap0_rgba": (5855, 5858), "tap1_rgba": (5859, 5862), "tap2_rgba": (5863, 5866), "tap3_rgba": (5867, 5870), "tap4_rgba": (5871, 5874),
             "w_tap0..4": (5875, 5879), "sum_partials": (5880, 5883), "rcp_sum": (5884, 5884), "pv_b_chain": (5897, 5901), "pv_b_x_rcp": (5902, 5902),
             "ex2_arg,ex2,plus1,rcp,x_conf": (5903, 5907), "conf_clamped": (5909, 5909), "diff_rgb": (5910, 5912), "out_rgb(after fma)": (5955, 5953)}
    rep = {"cta": [cx, cy], "warp": a.warp, "lane": L, "env": {k: v for k, v in os.environ.items() if k.startswith("EMU_")}, "registers": {}}
    for nm, (lo, hi) in names.items():
        rng = range(lo, hi + 1) if hi >= lo else range(lo, hi - 1, -1)
        rep["registers"][nm] = [{"r": n, "bits": f"{g(n):#010x}", "f32": f32(g(n))} for n in rng]
    conf = f32(g(5909)); diff_b = f32(g(5912)); out_b = f32(g(5953)); pvr = f32(g(5902))
    cur_b = float(np.float32(pvr) - np.float32(diff_b))                                    # cur = pv*rcp - diff (approximation, diff is exact at the fma below)
    def final(c, d, cu): return float(np.float32(np.float64(np.float32(c)) * np.float64(np.float32(d)) + np.float64(np.float32(cu))))
    rep["blend_check"] = {"conf": conf, "diff_b": diff_b, "cur_b_recovered": cur_b, "fma_recomputed": final(conf, diff_b, cur_b), "out_b_bits": f"{g(5953):#010x}", "out_b_rz16": f"{_rz16(out_b):#06x}"}
    def ulp(x, k): return f32((bits(x) + k) & 0xffffffff)
    sens = []
    for nm, val, which in (("conf", conf, 0), ("diff_b", diff_b, 1), ("cur_b", cur_b, 2)):
        for k in (-2, -1, 1, 2):
            args = [conf, diff_b, cur_b]; args[which] = ulp(val, k); f = final(*args); sens.append({"perturb": nm, "ulps": k, "final_bits": f"{bits(f):#010x}", "rz16": f"{_rz16(f):#06x}"})
    rep["sensitivity"] = sens
    json.dump(rep, open(a.out, "w"), indent=1); print(json.dumps(rep["blend_check"])); 
    for nm in ("tap0_rgba", "tap1_rgba", "tap2_rgba", "tap3_rgba", "tap4_rgba", "w_tap0..4", "sum_partials", "rcp_sum", "pv_b_chain", "pv_b_x_rcp", "ex2_arg,ex2,plus1,rcp,x_conf", "conf_clamped", "diff_rgb", "out_rgb(after fma)"): print(f"{nm:30s}", " ".join(f"{e['bits']}={e['f32']:.9g}" for e in rep["registers"][nm]))
    for e in sens: print(e)
if __name__ == "__main__": main()
