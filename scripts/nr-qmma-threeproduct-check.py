#!/usr/bin/env python3
"""CPU model of the three-product exact QMMA proposal (no Metal, no GPU) against Codex's RTX 5090 registers of the 64-case operand probe.
Per output element: (1) e4m3 -> integer m = x * 2^9, balanced split m = l + 512 h; (2) P0 = sum la*lb, P1 = sum ha*hb, P2 = sum (la+ha)(lb+hb)
accumulated in binary32 (numpy float32, sequential) and checked exact against integer sums; (3) reconstruction AB*2^18 = P0 + 512 (P2-P0-P1) + 512^2 P1
checked against the exact integer product sum; (4) certificate: with max_exp = max(operand exponent sums, C exponent) and G = 2^(max_exp-25),
the rule's truncated sum T = S - R where R sums the truncation remainders of terms not aligned to G; R in (-n_neg*G, n_pos*G); certified iff
RNE_half is constant over [S - n_pos*G, S + n_neg*G]; (5) certified results are compared with the RTX register (must never differ); uncertified
outputs are counted (they would take the accepted float-scaled rule)."""
import argparse, importlib.util, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
spec = importlib.util.spec_from_file_location("rulecheck", os.path.join(HERE, "nr-qmma-rule-check.py")); RC = importlib.util.module_from_spec(spec); spec.loader.exec_module(RC)
def e4m3_int(b):
    """value * 2^9 as an exact integer, and floor(log2|x|) (None for zero)"""
    (sig, q), neg = RC.e4m3_fields(b)
    if sig == 0: return 0, None
    m = sig << (q + 9); return (-m if neg else m), q + sig.bit_length() - 1
def split(m):
    h = (m + 256) >> 9            # balanced: l in [-256, 256)
    return m - 512 * h, h
def rne_half_of_scaled(value, scale): return RC.fixed_to_half_rne(value, scale)
def main():
    p = argparse.ArgumentParser(); p.add_argument("--rtx-dir", required=True); p.add_argument("--key", default="5a-a"); p.add_argument("--cases", type=int, default=64); a = p.parse_args()
    inp = np.fromfile(f"{a.rtx_dir}/output/{a.key}.input-after.bin", dtype=np.uint32).reshape(a.cases, 32, 8); out = np.fromfile(f"{a.rtx_dir}/output/{a.key}.bin", dtype=np.uint32).reshape(a.cases, 32, 2)
    f32 = np.float32; stats = {"outputs": 0, "decomp_bound_violations": 0, "partial_products_inexact": 0, "reconstruction_mismatch": 0, "certified": 0, "certified_wrong": 0, "uncertified": 0, "certified_sign_only": 0, "tiles_fully_certified": 0, "rule_reference_mismatch": 0}
    lmax = hmax = lhmax = 0; pmax = 0
    for ci in range(a.cases):
        lanes = inp[ci]; A = np.zeros((16, 32), dtype=np.int64); B = np.zeros((32, 8), dtype=np.int64); C = np.zeros((16, 8), dtype=np.int64)
        for l in range(32):
            g, t = l // 4, l % 4; a4 = [int(x) for x in lanes[l][0:4]]; b2 = [int(x) for x in lanes[l][4:6]]; c2 = [int(x) for x in lanes[l][6:8]]
            for j in range(4):
                A[g, 4 * t + j] = (a4[0] >> (8 * j)) & 255; A[g + 8, 4 * t + j] = (a4[1] >> (8 * j)) & 255; A[g, 16 + 4 * t + j] = (a4[2] >> (8 * j)) & 255; A[g + 8, 16 + 4 * t + j] = (a4[3] >> (8 * j)) & 255
                B[4 * t + j, g] = (b2[0] >> (8 * j)) & 255; B[16 + 4 * t + j, g] = (b2[1] >> (8 * j)) & 255
            C[g, 2 * t] = c2[0] & 0xffff; C[g, 2 * t + 1] = c2[0] >> 16; C[g + 8, 2 * t] = c2[1] & 0xffff; C[g + 8, 2 * t + 1] = c2[1] >> 16
        tile_all = True
        for r in range(16):
            for c in range(8):
                stats["outputs"] += 1; arow = [int(x) for x in A[r]]; bcol = [int(x) for x in B[:, c]]; cbits = int(C[r, c])
                l = r % 8 * 4 + c // 2; reg = r // 8; actual = (int(out[ci, l, reg]) >> (16 * (c % 2))) & 0xffff
                ref = RC.rule_reference(arow, bcol, cbits)
                if ref != actual: stats["rule_reference_mismatch"] += 1
                ma = [e4m3_int(b) for b in arow]; mb = [e4m3_int(b) for b in bcol]
                la, ha = zip(*[split(m) for m, _ in ma]); lb, hb = zip(*[split(m) for m, _ in mb])
                lmax = max(lmax, max(abs(x) for x in la + lb)); hmax = max(hmax, max(abs(x) for x in ha + hb)); lhmax = max(lhmax, max(abs(x + y) for x, y in zip(la + lb, ha + hb)))
                if max(abs(x) for x in la + lb) > 256 or max(abs(x) for x in ha + hb) > 448 or lhmax > 448: stats["decomp_bound_violations"] += 1
                P0i = sum(x * y for x, y in zip(la, lb)); P1i = sum(x * y for x, y in zip(ha, hb)); P2i = sum((x + u) * (y + v) for x, u, y, v in zip(la, ha, lb, hb))
                acc0 = acc1 = acc2 = f32(0)
                for x, y, u, v in zip(la, lb, ha, hb): acc0 = f32(acc0 + f32(f32(x) * f32(y))); acc1 = f32(acc1 + f32(f32(u) * f32(v))); acc2 = f32(acc2 + f32(f32(x + u) * f32(y + v)))
                if int(acc0) != P0i or int(acc1) != P1i or int(acc2) != P2i: stats["partial_products_inexact"] += 1
                pmax = max(pmax, abs(P0i), abs(P1i), abs(P2i))
                recon = P0i + 512 * (P2i - P0i - P1i) + 512 * 512 * P1i                     # = sum(ma*mb) = AB * 2^18
                exact_prod = sum(m1 * m2 for (m1, _), (m2, _) in zip(ma, mb))
                if recon != exact_prod: stats["reconstruction_mismatch"] += 1
                # exact sum at grid 2^-24 (C may be a half subnormal with lowest bit 2^-24)
                (cs, cq), cn = RC.half_fields(cbits); cval = (-cs if cn else cs) << (cq + 24) if cs else 0
                S = (recon << 6) + cval
                # rule's alignment: max_exp over nonzero products of operand-exponent sums, and C
                maxexp = None; terms = []
                for (m1, e1), (m2, e2) in zip(ma, mb):
                    if m1 == 0 or m2 == 0: continue
                    ex = e1 + e2; maxexp = ex if maxexp is None else max(maxexp, ex); terms.append(m1 * m2 << 6)          # units 2^-24
                if cs: ce = cq + cs.bit_length() - 1; maxexp = ce if maxexp is None else max(maxexp, ce); terms.append(cval)
                if maxexp is None:
                    stats["certified"] += 1; (stats.__setitem__("certified_wrong", stats["certified_wrong"] + 1) if actual != 0 else None); continue
                gexp = maxexp - 25; Gunits = 1 << (gexp + 24) if gexp + 24 >= 0 else None   # grid in units of 2^-24
                n_pos = n_neg = 0
                if Gunits is None: n_pos = n_neg = 0        # grid finer than 2^-24: every term aligned
                else:
                    for t in terms:
                        if t % Gunits: (n_pos := n_pos + 1) if t > 0 else (n_neg := n_neg + 1)
                        # re-bind after walrus in conditional expression
                    # (python quirk: recompute cleanly)
                    n_pos = sum(1 for t in terms if t % Gunits and t > 0); n_neg = sum(1 for t in terms if t % Gunits and t < 0)
                G = Gunits or 0
                lo, hi = S - n_pos * G, S + n_neg * G
                # RNE_half over the interval [lo, hi] (integers in units 2^-24): constant iff the half of lo == half of hi and no tie boundary crossed;
                # halves are monotone in value, so constancy over the closed interval reduces to equal endpoints
                h_lo = rne_half_of_scaled(lo, -24); h_hi = rne_half_of_scaled(hi, -24)
                if n_pos == 0 and n_neg == 0: stats["certified_sign_only"] += 1
                if h_lo == h_hi:
                    stats["certified"] += 1
                    if h_lo != actual: stats["certified_wrong"] += 1
                else: stats["uncertified"] += 1; tile_all = False
        if tile_all: stats["tiles_fully_certified"] += 1
    stats["max_abs_l"] = lmax; stats["max_abs_h"] = hmax; stats["max_abs_l_plus_h"] = lhmax; stats["max_abs_partial_sum"] = pmax; stats["partial_sum_bound_2^23"] = pmax < 2 ** 23
    print(json.dumps(stats, indent=1)); json.dump(stats, open(os.path.join(ROOT, "build/nr_ptx/qmma-threeproduct-check.json"), "w"), indent=1)
if __name__ == "__main__": main()
