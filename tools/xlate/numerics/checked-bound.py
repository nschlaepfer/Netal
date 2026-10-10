"""The implementable bound: split K=16 into two K=8 MMAs, bound each half by its own partial sum.

  value MMAs : acc = A0.B0 then acc = A1.B1 + acc        (2 simdgroup ops, what the kernel does anyway)
  bound MMAs : P0 = |A0|.|B0|, P1 = |A1|.|B1|            (2 more, same shape)
  e = 0.5*8*ulp32(P0) + 0.5*8*ulp32(P0+P1) + 0.5*ulp32(|v|)

Every term is available from four simdgroup ops and a handful of scalar instructions. No per-step
sequential work, which is the whole reason the running bound was not implementable.
"""
import numpy as np
from fractions import Fraction
rng = np.random.default_rng(3)

def ulp32(x):
    x = abs(float(x))
    if x == 0: return float(np.nextafter(np.float32(0), np.float32(1)))
    return float(np.ldexp(1.0, np.frexp(x)[1] - 24))

def run(K, n, sa, sb, label):
    fb = {"half(1 split)": 0, "split K=8": 0, "running": 0}
    unsafe = dict.fromkeys(fb, 0); differ = 0
    half = K // 2
    for _ in range(n):
        a = np.float16(rng.standard_normal(K) * sa)
        b = np.float16(rng.standard_normal(K) * sb)
        v = np.float32(0); P = np.float32(0); running = 0.0
        P0 = None
        for i in range(K):
            v = np.float32(np.float32(a[i]) * np.float32(b[i]) + v)
            P = np.float32(abs(np.float32(a[i]) * np.float32(b[i])) + P)
            running += 0.5 * ulp32(P)
            if i == half - 1: P0 = float(P)
        exact = sum(Fraction(float(np.float32(x)) * float(np.float32(y))) for x, y in zip(a, b))
        ref, got = np.float16(float(exact)), np.float16(v)
        wrong = got.view(np.uint16) != ref.view(np.uint16)
        if wrong: differ += 1
        bounds = {
            "half(1 split)": 0.5 * K * ulp32(P),
            "split K=8":     0.5 * half * ulp32(P0) + 0.5 * half * ulp32(P) + 0.5 * ulp32(float(v)),
            "running":       running,
        }
        for name, e in bounds.items():
            lo, hi = np.float16(float(v) - e), np.float16(float(v) + e)
            if lo.view(np.uint16) != hi.view(np.uint16): fb[name] += 1
            elif wrong: unsafe[name] += 1
    eff = lambda f: 1.0 / ((1 - f) / 18.0 + f)
    cells = "  ".join(f"{k} {100*fb[k]/n:5.2f}%{'!' if unsafe[k] else ''} -> {eff(fb[k]/n):4.1f}x" for k in fb)
    print(f"{label:24s} K={K:3d} raw {100*differ/n:5.3f}%   {cells}")

print("implementable bound vs the ideal, with effective speedup against an 18x fast path\n")
for K in (16, 32):
    run(K, 2500, 1.0, 0.02, "conv-like")
run(16, 2500, 1.0, 1.0, "unit scale")
run(16, 2500, 4.0, 0.25, "wide range")
run(16, 2500, 0.05, 0.05, "small magnitudes")
