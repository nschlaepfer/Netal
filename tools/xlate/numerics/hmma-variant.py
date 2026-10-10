"""Which reading of the alignment rule matches FG's measured 117-byte divergence?

"aligns sixteen exact FP16 products plus the prior FP16 accumulator at the largest exponent, keeps
two alignment bits, truncates discarded alignment bits, and rounds once to FP16" admits more than
one reading, and they differ enormously. FG measured a raw 16x8 simdgroup tile differing from its
scalar oracle by 117 bytes of a 983,040-byte tensor - about 0.012% of positions. That number is the
calibration: the correct reading should reproduce roughly that disagreement rate against fp32.
"""
import numpy as np
from fractions import Fraction
rng = np.random.default_rng(7)

def terms(a, b, c):
    t = [Fraction(float(np.float32(x)) * float(np.float32(y))) for x, y in zip(a, b)]
    t.append(Fraction(float(np.float32(c))))
    return t

def grid(ts, guard=2):
    mags = [abs(t) for t in ts if t != 0]
    if not mags: return None
    emax = max(m.numerator.bit_length() - m.denominator.bit_length() for m in mags)
    return Fraction(2) ** (emax - (10 + guard))

def trunc(x, g):
    q = x / g
    return (int(q) if q >= 0 else -int(-q)) * g

def per_term(a, b, c):        # each term truncated onto the grid, then summed
    ts = terms(a, b, c); g = grid(ts)
    if g is None: return np.float16(0)
    return np.float16(float(sum(trunc(t, g) for t in ts)))

def sum_then_trunc(a, b, c):  # summed exactly, the sum truncated once
    ts = terms(a, b, c); g = grid(ts)
    if g is None: return np.float16(0)
    return np.float16(float(trunc(sum(ts), g)))

def exact_sum(a, b, c):       # no truncation at all: exact sum, one rounding to fp16
    return np.float16(float(sum(terms(a, b, c))))

def fp32(a, b, c):
    acc = np.float32(c)
    for v in (np.float32(a) * np.float32(b)): acc = np.float32(acc + v)
    return np.float16(acc)

N = 6000
A = np.float16(rng.standard_normal((N, 16)))
B = np.float16(rng.standard_normal((N, 16)) * 0.02)
C = np.float16(rng.standard_normal(N))
print("reading of the rule                      disagreement with fp32-then-round")
for name, fn in (("per-term truncate onto grid", per_term),
                 ("exact sum, truncate once", sum_then_trunc),
                 ("exact sum, no truncation", exact_sum)):
    d = sum(1 for i in range(N)
            if fn(A[i], B[i], C[i]).view(np.uint16) != fp32(A[i], B[i], C[i]).view(np.uint16))
    print(f"  {name:38s} {100*d/N:6.3f}%")
