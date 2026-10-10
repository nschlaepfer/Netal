# HMMA.16816.F16 with f16 accumulate: the rule, exactly

The single most load-bearing primitive fact in both the SR and FG lanes, recovered by Codex and
validated byte-exact against the RTX 5090 across all 42 FG convolution launches. It lives in
`Mac-fg-research:src/fg_conv_simd_l20/analyze_truncation.cc` and `src/fg_conv/hmma_exact.metalh`;
recorded here so nobody has to re-derive it from the rejection ledger.

For one `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16` output - sixteen fp16 products plus an
fp16 prior:

    for each product k:
        significand_k = sig(a_k) * sig(b_k)          exact, up to 22 bits
        exponent_k    = exp(a_k) + exp(b_k)          fp16 subnormals count as exponent -14
    prior:
        significand_c = sig(c) << 13                 widened to 24 bits
        exponent_c    = exp(c)
    max_exp = max over all NONZERO terms of exponent

    for each term:
        numerator = significand << 2                 two guard bits
        aligned   = numerator >> (max_exp - exponent) TRUNCATION; shift > 31 gives 0
        total    += sign * aligned                   exact integer sum

    result = RNE_fp16(total, scale = max_exp - 25)  one rounding, ties-to-even
             with fp16 subnormal handling at exponent < -14

## What follows from it

- The internal grid is `2^(max_exp - 25)`. A term loses bits only when shifted by more than its two
  zero guard bits - that is, when it sits three or more binades below the largest term.
- Each shifted term discards **less than one grid unit**, so the total discarded is under 17 units.
  Measured on L20's live input: at most 3.
- One grid unit is about 2^-15 of an fp16 ULP at the result's magnitude. **So NVIDIA's result equals
  "exact sum, round once" except within ~17 grid units of an fp16 midpoint** - which is the 0.09% of
  stages that every exact-sum method (double-float scalar, split-b MMA, split4 MMA) misses, all
  landing within 20 of each other against the oracle.
- Therefore the two halves of the problem are separable: **compute the exact sum with shared
  operands** (split-precision MMA does this at 8.65x), and **guard only against the truncation**,
  whose bound is ~7x tighter than the fp32-accumulation bound because it is a different, smaller
  quantity.

## What a simdgroup cannot give you

`max_exp` itself. It is a maximum over 17 terms spread across the lanes, and MMAs sum. `exp(Σ|a||b|)`
bounds it from above within four binades, which is enough for the guard and not enough to replicate
the truncation exactly in-lane. Closing that would make the guard unnecessary; it is the open
problem.
