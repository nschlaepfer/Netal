#!/usr/bin/env python3
"""Fit the RTX texture unit's bilinear rule (coordinate -> 8-bit fraction, and the weight/blend arithmetic) against Codex's step-texture
micro-probe (build/nr_tex_micro/<run>: 303 tex.2d queries on a 1920x1080 RGBA16F texture with R = x>=384, G = y>=1, B = (x+y)&1, A = 1).
Joint search over coordinate models x blend models, scored on all 909 returned values. Diagnostic; writes a JSON receipt."""
import argparse, itertools, json, math, os, sys
import numpy as np
W, H = 1920, 1080
def texel(x, y):
    x = min(max(x, 0), W - 1); y = min(max(y, 0), H - 1); return (float(x >= 384), float(y >= 1), float((x + y) & 1), 1.0)
def rnd(v, mode):
    if mode == "trunc": return math.floor(v)
    if mode == "nup": return math.floor(v + 0.5)
    if mode == "neven": return round(v)
    if mode == "ceil": return math.ceil(v)
    raise ValueError(mode)
COORD_MODES = ["trunc", "nup", "neven"]
def coord(u, dim, cm):
    """returns (x0, fx256). cm = (order, pre_bits, m1, m2): order 'sub_first' quantises (u*dim - 0.5); 'sub_after' quantises u*dim then subtracts 128/256.
    pre_bits: an optional first quantisation of the unnormalised coordinate to 2^-pre_bits (m1) before the 8-bit fraction (m2); 0 = none."""
    order, pre, m1, m2 = cm; ue = float(u) * dim
    if order == "sub_first": ue -= 0.5
    if pre: ue = rnd(ue * (1 << pre), m1) / (1 << pre)
    T = rnd(ue * 256, m2)
    if order == "sub_after": T -= 128
    return T >> 8, T & 255
def blend(t, fx, fy, bm):
    """t = (a, b, c, d) texel values; bm = (form, pmode): form '2stage_y' = lerp in x then y, '2stage_x' = y then x, '4prod' = four products;
    pmode = rounding of each fixed-point product to 1/256 ('exact' keeps full precision)."""
    a, b, c, d = t; form, pm = bm
    def P(v): return v if pm == "exact" else rnd(v, pm)
    if form == "2stage_y":
        top = (a * (256 - fx) + b * fx); bot = (c * (256 - fx) + d * fx)                     # /256 units
        return (P(top * (256 - fy) / 256) + P(bot * fy / 256)) / 256
    if form == "2stage_x":
        l = (a * (256 - fy) + c * fy); r = (b * (256 - fy) + d * fy)
        return (P(l * (256 - fx) / 256) + P(r * fx / 256)) / 256
    if form == "2stage_y_both":
        top = P((a * (256 - fx) + b * fx) / 256) ; bot = P((c * (256 - fx) + d * fx) / 256)   # first stage also rounded to 1/256 (0/1 texels: exact anyway)
        return (P(top * (256 - fy)) + P(bot * fy)) / 256 / 1
    if form == "4prod":
        w = ((256 - fx) * (256 - fy), fx * (256 - fy), (256 - fx) * fy, fx * fy)
        return sum(P(wi / 256) * ti for wi, ti in zip(w, t)) / 256
    raise ValueError(form)
def main():
    p = argparse.ArgumentParser(); p.add_argument("--run", required=True); p.add_argument("--out", required=True); a = p.parse_args()
    Q = np.fromfile(os.path.join(a.run, "queries.bin"), dtype="<f4").reshape(-1, 2); S = np.fromfile(os.path.join(a.run, "samples.bin"), dtype="<u4").reshape(-1, 4).view(np.float32)
    cms = [(order, pre, m1, m2) for order in ("sub_first", "sub_after") for pre in (0, 9, 10, 11, 12, 13, 14, 16) for m1 in COORD_MODES for m2 in COORD_MODES if pre or m1 == "trunc"]
    bms = [(form, pm) for form in ("2stage_y", "2stage_x", "4prod") for pm in ("exact", "trunc", "nup", "neven")]
    results = []
    for cm in cms:
        cx = [coord(Q[k, 0], W, cm) for k in range(len(Q))]; cy = [coord(Q[k, 1], H, cm) for k in range(len(Q))]
        for bm in bms:
            bad = 0; ex = []
            for k in range(len(Q)):
                (x0, fx), (y0, fy) = cx[k], cy[k]; t = [texel(x0, y0), texel(x0 + 1, y0), texel(x0, y0 + 1), texel(x0 + 1, y0 + 1)]
                for ch in range(3):
                    v = blend(tuple(tt[ch] for tt in t), fx, fy, bm)
                    if abs(v - float(S[k, ch])) > 1e-9:
                        bad += 1
                        if len(ex) < 12: ex.append({"query": k, "channel": ch, "x0": x0, "fx": fx, "y0": y0, "fy": fy, "actual_x256": float(S[k, ch]) * 256, "model_x256": v * 256})
            results.append({"unequal_values": bad, "coordinate_model": {"order": cm[0], "pre_bits": cm[1], "pre_mode": cm[2], "fraction_mode": cm[3]}, "blend_model": {"form": bm[0], "product_rounding": bm[1]}, "first_counterexamples": ex})
    results.sort(key=lambda r: r["unequal_values"])
    rep = {"schema": "neuralstudio.nr.tex-micro-fit.v1", "run": a.run, "values": int(S.size - len(Q)), "models_tested": len(results), "best": results[:15]}
    json.dump(rep, open(a.out, "w"), indent=1)
    for r in results[:15]: print(r["unequal_values"], r["coordinate_model"], r["blend_model"])
    print("counterexamples of the best:"); [print("  ", e) for e in results[0]["first_counterexamples"]]
if __name__ == "__main__": main()
