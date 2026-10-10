#!/usr/bin/env python3
"""Adversarial QMMA fragment generator for the primitive test (qmma_probe): writes cases in the RTX operand-probe layout (per case, per lane
a[4] b[2] c[2] raw registers) and the expected d[2] registers from the accepted rule (scripts/nr-qmma-rule-check.py, 0/4096 vs RTX). Case classes
from Codex's review: max-magnitude mixed-sign operands; signed cancellation; half-subnormal / zero / midpoint-boundary priors; all-zero products with
C only; sparse rows/cols (mask zeros); one non-aligned term of each sign; C as the only non-aligned term; permutations of the same terms
(classification invariance); dense random. Output: <out>/cases.input.bin, <out>/cases.expected.bin, <out>/cases.json (class per case)."""
import argparse, importlib.util, json, os, random, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("rc", os.path.join(HERE, "nr-qmma-rule-check.py")); RC = importlib.util.module_from_spec(spec); spec.loader.exec_module(RC)
FINITE = [b for b in range(256) if (b & 0x7f) != 0x7f]
def e4(e, m, neg=False): return (0x80 if neg else 0) | (e << 3) | m
def pack_case(A, B, C):
    """A 16x32 bytes, B 32x8 bytes, C 16x8 half bits -> 32 lanes x 8 uint32"""
    out = np.zeros((32, 8), dtype=np.uint32)
    for l in range(32):
        g, t = l // 4, l % 4
        def pk(vals): return sum(int(v) << (8 * j) for j, v in enumerate(vals))
        out[l, 0] = pk(A[g, 4 * t:4 * t + 4]); out[l, 1] = pk(A[g + 8, 4 * t:4 * t + 4]); out[l, 2] = pk(A[g, 16 + 4 * t:16 + 4 * t + 4]); out[l, 3] = pk(A[g + 8, 16 + 4 * t:16 + 4 * t + 4])
        out[l, 4] = pk(B[4 * t:4 * t + 4, g]); out[l, 5] = pk(B[16 + 4 * t:16 + 4 * t + 4, g])
        out[l, 6] = int(C[g, 2 * t]) | (int(C[g, 2 * t + 1]) << 16); out[l, 7] = int(C[g + 8, 2 * t]) | (int(C[g + 8, 2 * t + 1]) << 16)
    return out
def expected(A, B, C):
    D = np.zeros((16, 8), dtype=np.int64)
    for r in range(16):
        for c in range(8): D[r, c] = RC.rule_reference([int(x) for x in A[r]], [int(x) for x in B[:, c]], int(C[r, c]))
    out = np.zeros((32, 2), dtype=np.uint32)
    for l in range(32):
        g, t = l // 4, l % 4; out[l, 0] = int(D[g, 2 * t]) | (int(D[g, 2 * t + 1]) << 16); out[l, 1] = int(D[g + 8, 2 * t]) | (int(D[g + 8, 2 * t + 1]) << 16)
    return out
def main():
    p = argparse.ArgumentParser(); p.add_argument("--out", required=True); p.add_argument("--seed", type=int, default=11); p.add_argument("--per-class", type=int, default=8); a = p.parse_args()
    rng = random.Random(a.seed); os.makedirs(a.out, exist_ok=True); cases = []; ins = []; exps = []
    def rand_bytes(shape, pool): return np.array([rng.choice(pool) for _ in range(int(np.prod(shape)))], dtype=np.int64).reshape(shape)
    def half_bits(v): return int(np.float16(v).view(np.uint16))
    maxmag = [e4(15, 6, n) for n in (False, True)] + [e4(15, 5, n) for n in (False, True)]
    for cls in ("max_magnitude_mixed_sign", "signed_cancellation", "subnormal_zero_priors", "midpoint_boundary_priors", "zero_products_c_only", "sparse_masks", "one_nonaligned_each_sign", "c_only_nonaligned", "permutation_pairs", "dense_random"):
        for i in range(a.per_class):
            A = rand_bytes((16, 32), FINITE); B = rand_bytes((32, 8), FINITE); C = np.array([[half_bits(rng.uniform(-64, 64)) for _ in range(8)] for _ in range(16)], dtype=np.int64)
            if cls == "max_magnitude_mixed_sign": A = rand_bytes((16, 32), maxmag); B = rand_bytes((32, 8), maxmag); C = np.array([[half_bits(rng.choice([65504.0, -65504.0, 0.0, 1.0])) for _ in range(8)] for _ in range(16)], dtype=np.int64)
            elif cls == "signed_cancellation":
                # pairs of equal-magnitude opposite-sign terms plus a tiny residual term
                for r in range(16):
                    for k in range(0, 32, 2): A[r, k] = A[r, k] & 0x7f; A[r, k + 1] = A[r, k] | 0x80
                for k in range(0, 32, 2): B[k + 1, :] = B[k, :]
                A[:, 31] = e4(1, 0); B[31, :] = e4(0, 1)   # residual 2^-9 * 2^-9 products cannot be paired (k=30,31 pair broken)
            elif cls == "subnormal_zero_priors": C = np.array([[rng.choice([0, 0x8000, 1, 0x3ff, 0x8001, 0x0400, 0x83ff]) for _ in range(8)] for _ in range(16)], dtype=np.int64)
            elif cls == "midpoint_boundary_priors":
                # choose C so that S lands near a half rounding midpoint: C = -(exact product sum) + midpoint-ish value
                for r in range(16):
                    for c in range(8):
                        s = sum(RC.e4m3_float(int(A[r, k])) * RC.e4m3_float(int(B[k, c])) for k in range(32)); target = rng.choice([0.5, 1.5, 2.5, 1.0 + 2 ** -11, 3.0 + 2 ** -10]) * rng.choice([1, -1])
                        C[r, c] = half_bits(max(-65504.0, min(65504.0, target - s)))
            elif cls == "zero_products_c_only":
                A[:] = rng.choice([0, 0x80]); C = np.array([[half_bits(rng.uniform(-2000, 2000)) if rng.random() < 0.7 else rng.choice([0, 0x8000, 1]) for _ in range(8)] for _ in range(16)], dtype=np.int64)
            elif cls == "sparse_masks":
                mask = np.array([[rng.random() < 0.15 for _ in range(32)] for _ in range(16)]); A = np.where(mask, A, 0); bm = np.array([[rng.random() < 0.3 for _ in range(8)] for _ in range(32)]); B = np.where(bm, B, 0)
            elif cls == "one_nonaligned_each_sign":
                # one huge product sets max_exp; one tiny product 20+ binades below (non-aligned) with chosen sign; the rest zero
                A[:] = 0; B[:] = 0; A[:, 0] = e4(15, 0); B[0, :] = e4(15, 0); A[:, 5] = e4(0, 1, i % 2 == 1); B[5, :] = e4(1, 3); C = np.array([[half_bits(rng.uniform(-1, 1)) for _ in range(8)] for _ in range(16)], dtype=np.int64)
            elif cls == "c_only_nonaligned":
                A[:] = 0; B[:] = 0; A[:, 0] = e4(15, 0); B[0, :] = e4(15, 0); C = np.array([[rng.choice([1, 0x8001, 3, 0x0201, 0x8201]) for _ in range(8)] for _ in range(16)], dtype=np.int64)
            elif cls == "permutation_pairs" and i % 2 == 1:
                prevA, prevB, prevC = cases[-1]["_A"], cases[-1]["_B"], cases[-1]["_C"]; perm = list(range(32)); rng.shuffle(perm); A = prevA[:, perm]; B = prevB[perm, :]; C = prevC
            cases.append({"class": cls, "index": i, "_A": A, "_B": B, "_C": C}); ins.append(pack_case(A, B, C)); exps.append(expected(A, B, C))
    np.concatenate(ins).astype(np.uint32).tofile(os.path.join(a.out, "cases.input.bin")); np.concatenate(exps).astype(np.uint32).tofile(os.path.join(a.out, "cases.expected.bin"))
    json.dump({"schema": "neuralstudio.nr.qmma-adversarial.v1", "seed": a.seed, "cases": [{"class": c["class"], "index": c["index"]} for c in cases]}, open(os.path.join(a.out, "cases.json"), "w"), indent=1)
    print("wrote", len(cases), "cases to", a.out)
if __name__ == "__main__": main()
