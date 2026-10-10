#!/usr/bin/env python3
"""Generate mma.sync m16n8k32 e4m3 operand sets on which the candidate accumulation associations disagree, as an RTX probe
request: random realistic fragments, D predicted under every candidate (fp32 chain seeded with C = "seq"; products
first then C = "c_last"; two/four interleaved chains; pairwise trees). Cases are kept only if at least two
products-first candidates differ, so the hardware answer selects among them."""
import argparse, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from ptx_warp_emu import mma_accumulate
from nr_fp8 import E4M3
ORDERS = ("seq", "c_last", "halves", "quads", "tree", "tree_c_first", "rev")
def frag(A, B, C):
    lanes = []
    for l in range(32):
        g, t = l // 4, l % 4
        pk = lambda vals: int(sum(int(v) << (8 * j) for j, v in enumerate(vals)))
        a = [pk(A[g, 4 * t:4 * t + 4]), pk(A[g + 8, 4 * t:4 * t + 4]), pk(A[g, 16 + 4 * t:16 + 4 * t + 4]), pk(A[g + 8, 16 + 4 * t:16 + 4 * t + 4])]
        b = [pk(B[4 * t:4 * t + 4, g]), pk(B[16 + 4 * t:16 + 4 * t + 4, g])]
        c = [int(C[g, 2 * t]) | (int(C[g, 2 * t + 1]) << 16), int(C[g + 8, 2 * t]) | (int(C[g + 8, 2 * t + 1]) << 16)]
        lanes.append((a, b, c))
    return lanes
def dfrag(D):
    return [[int(D[l // 4, 2 * (l % 4)]) | (int(D[l // 4, 2 * (l % 4) + 1]) << 16), int(D[l // 4 + 8, 2 * (l % 4)]) | (int(D[l // 4 + 8, 2 * (l % 4) + 1]) << 16)] for l in range(32)]
def main():
    p = argparse.ArgumentParser(); p.add_argument("--cases", type=int, default=48); p.add_argument("--seed", type=int, default=7); p.add_argument("--output", required=True); a = p.parse_args()
    rng = np.random.default_rng(a.seed); cases = []; tries = 0
    e4 = np.array(E4M3, dtype=np.float32)
    while len(cases) < a.cases and tries < 200000:
        tries += 1
        # e4m3 bytes: exponent field 4..11 (values ~2^-3 .. 2^4 with mantissa), random sign; C halves in +-[0.25, 64)
        # bimodal magnitudes: a few large products dominate each sum while many tiny products sit below the fp32 LSB of the
        # running sum, so the association decides the fp32 result; sums stay far below the half range
        def frag_bytes(shape, large_frac):
            big = rng.random(shape) < large_frac
            ex = np.where(big, rng.integers(11, 15, shape), rng.integers(1, 4, shape))
            return (ex << 3 | rng.integers(0, 8, shape) | rng.integers(0, 2, shape) << 7).astype(np.int64)
        Ab = frag_bytes((16, 32), 0.08); Bb = frag_bytes((32, 8), 0.08)
        Cv = (np.exp2(rng.uniform(-2.0, 9.0, (16, 8))) * rng.choice([-1.0, 1.0], (16, 8))).astype(np.float16); Cb = Cv.view(np.uint16).astype(np.int64)
        A = e4[Ab]; B = e4[Bb]; C = Cv.astype(np.float32)
        P = [np.outer(A[:, k], B[k, :]).astype(np.float32) for k in range(32)]
        D = {o: mma_accumulate(P, C, o).astype(np.float16).view(np.uint16).astype(np.int64) for o in ORDERS}
        if any(not np.isfinite(mma_accumulate(P, C, o).astype(np.float16)).all() for o in ORDERS): continue      # keep every prediction finite
        pf = [D[o] for o in ("c_last", "halves", "quads", "tree", "tree_c_first")]
        diff = np.zeros((16, 8), dtype=bool)
        for x in pf[1:]: diff |= (x != pf[0])
        if not diff.any(): continue
        lanes = frag(Ab, Bb, Cb); dl = {o: dfrag(D[o]) for o in ORDERS}
        rows, cols = np.nonzero(diff); elems = [{"row": int(r), "col": int(c), "lane": int(4 * (r % 8) + c // 2), "reg": int(r // 8), "half": int(c % 2), "predictions": {o: f"{int(D[o][r, c]):04x}" for o in ORDERS}} for r, c in zip(rows, cols)]
        cases.append({"id": len(cases), "lanes": [{"lane": l, "a": [f"{v:08x}" for v in lanes[l][0]], "b": [f"{v:08x}" for v in lanes[l][1]], "c": [f"{v:08x}" for v in lanes[l][2]], "d": {o: [f"{v:08x}" for v in dl[o][l]] for o in ORDERS}} for l in range(32)], "discriminating_elements": elems})
    out = {"schema": "neuralstudio.nr.mma-operand-probe.v2", "instruction": "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16", "fragment_layout": "same as neuralstudio.nr.mma-operand-probe.v1 (raw 32-bit registers per lane: a[4], b[2], c[2] -> d[2])",
           "candidates": {"seq": "fp32 FMA chain seeded with C, products in k order (the pre-2026-09-05 emulator/runtime rule; rejected on launch 151's element)", "c_last": "fp32 chain over products in k order from zero, then + C", "halves": "two chains of 16 (k<16 from C, k>=16 from 0), then summed", "quads": "four chains of 8 (C in chain 0), summed ((q0+q1)+q2)+q3", "tree": "pairwise tree over the 32 products, C added last", "tree_c_first": "C+p0 then pairwise tree", "rev": "fp32 chain seeded with C, products in reverse k order"},
           "tries": tries, "cases": cases}
    os.makedirs(os.path.dirname(a.output), exist_ok=True); json.dump(out, open(a.output, "w"), indent=1)
    per = {o: 0 for o in ORDERS}
    print(f"wrote {a.output}: {len(cases)} cases from {tries} random MMAs; discriminating elements per case: {[len(c['discriminating_elements']) for c in cases][:12]}...")
    # how often do the candidates pairwise disagree over all kept elements
    import itertools; pfo = ("c_last", "halves", "quads", "tree", "tree_c_first")
    for o1, o2 in itertools.combinations(pfo, 2):
        n = sum(1 for c in cases for e in c["discriminating_elements"] if e["predictions"][o1] != e["predictions"][o2]); print(f"  {o1} vs {o2}: {n} elements differ")
if __name__ == "__main__": main()
