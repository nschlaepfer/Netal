# A provably exact fast path for NVIDIA HMMA on Apple simdgroup matrices

FG's convolutions cost 74.8 of its 83.8 ms GPU frame because their kernel is a scalar correctness
oracle. A raw 16x8 simdgroup tile measured ~18x faster and was rejected for differing by 117 bytes.
This is the route that keeps most of the 18x **and cannot be wrong**.

## 1. The hardware is not a black box

`tools/xlate/numerics/mma-is-sequential-fma.mm`, M3 Max, 4,096 independent 8x8 products, three
distributions (wide exponent spread, unit scale, conv-like):

> **`simdgroup_multiply_accumulate` with half operands and an fp32 accumulator is bit-for-bit
> identical to a sequential fp32 FMA chain in k order. 0 differing bit patterns out of 262,144
> outputs, in every distribution.**

Against an exactly-rounded reference both disagree on the same 33-36% of *fp32* outputs and on the
same elements. So the MMA is exactly `for k: acc = fma(a[k], b[k], acc)`, and its error is therefore
boundable rather than mysterious. That single fact is what makes everything below possible, and it
is worth as much to SR and RR as it is to FG.

(The 33-36% at fp32 collapses to ~0.01% once rounded to fp16, which is why FG measured 117 bytes of
983,040 rather than a third of the tensor.)

## 2. The predicate

After K fused steps the accumulated error is bounded by the sum of each step's rounding, and each
FMA's addition rounds by at most half an ulp of its running partial. Bound the partials with a
second MMA over `|A|` and `|B|`, and the whole bound comes from four simdgroup ops:

    v  = A0.B0 ; v = A1.B1 + v          two value MMAs - what the kernel does anyway
    P0 = |A0|.|B0| ; P = |A1|.|B1| + P0 two bound MMAs, same shape
    e  = 0.5*8*ulp32(P0) + 0.5*8*ulp32(P) + 0.5*ulp32(|v|)

Accept `half(v)` when `half(v - e) == half(v + e)`. Inside that interval the fp16 result cannot
depend on the fp32 error, so **accepting is safe by construction** - there is no tuning constant and
no empirical threshold. Otherwise fall back to the exact scalar helper for that element.

## 3. What it costs

`tools/xlate/numerics/checked-bound.py`, 2,500 stages per row, against the hardware model proven in
section 1:

| distribution | K | raw fp32-vs-exact | fallback | effective vs an 18x fast path |
|---|---:|---:|---:|---:|
| conv-like (act N(0,1), w 0.02) | 16 | 0.040% | **2.64%** | **12.4x** |
| unit scale | 16 | 0.000% | 2.44% | 12.7x |
| wide dynamic range | 16 | 0.000% | 2.00% | 13.4x |
| small magnitudes | 16 | 0.000% | 1.84% | 13.7x |
| conv-like | 32 | 0.120% | 5.24% | 9.5x |

**Zero unsafe accepts in every row** - the predicate never accepted a value that differed from the
exact reference, which is what "safe by construction" has to mean in practice as well as in theory.

At K=16, FG's stage size, that is **74.8 ms -> about 6 ms** for the custom block, bit-exact.

## 4. Why not just use a wider accumulator

Because the exact sum of sixteen fp16 products needs about 26 significand bits and fp32 has 24, a
compensated (two-float) accumulator would remove the fallback entirely - but it has to sit *inside*
the K loop, and the MMA does not expose its intermediate partials. The checked path works with the
instruction as it is. A two-float accumulator built from scalar FMAs is a separate candidate and
should be measured against 12.4x, not assumed to beat it.

## 5. Limits

Section 1 is a hardware measurement. Sections 2 and 3 are exact-arithmetic evaluations of the model
that section 1 proved, not runs of FG's kernels: the fallback rates come from the verified sequential
FMA model over synthetic conv-shaped data. The receipt is still the three-way test on real FG L20
state, which needs FG's fixtures in a unit manifest. What has changed is that the route is no longer
a hypothesis about whether exactness is reachable - it is, by construction, and the remaining
question is only what it measures on the real tensors.
