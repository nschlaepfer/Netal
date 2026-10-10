# The custom block: conv2 family SIMD rewrite

The custom L56-L83 block is 74.8 of FG's 83.8 ms GPU frame. The conv2 families - `module 57`
launches 60-67 and `module 64` launches 74-81 - are **16 of its 25 launches** and share one kernel,
so they are the largest single target reachable by one rewrite.

`custom_conv2_simd.metal`, measured against the shipped scalar kernel on the real shape (80x46,
64 in, 64 out, 9 tiles x 4 channel stages, leaky 0.01, clamp +/-6, residual), synthetic weights:

| | ms per launch | vs scalar | differing |
|---|---:|---:|---:|
| scalar `fg_custom_block0_conv2` | 1.128 - 1.131 | 1.0x | - |
| **`fg_custom_block0_conv2_simd`** | **0.059 - 0.062** | **19.1x** | 342 of 235,520 (**0.145%**) |

At 1.13 ms each the conv2 families are about **18 ms** of the 74.8, so the other nine custom
launches carry the remaining ~57 ms at larger spatial extents.

The 0.145% divergence is the same class as L20's 0.19% - fp32 accumulation inside each K=16 stage
against NVIDIA's f16 accumulate - and it is exactly what the guard in `../fg-l20/` is built to
close. The weight fragment layout differs from L20's in packing but not in structure: L20 uses
`lane*4 + frag`, conv2 uses `(frag>=2?64:0) + lane*2 + (frag&1)`. Same `mma.sync.m16n8k16.row.col`
B-fragment convention, so the port was mechanical.

## Status and what is not done

This is the fast path only, and it is not yet exact. It was measured with synthetic weights against
the same scalar kernel, which validates the rewrite and its speedup on the real shape but is **not**
a live-token receipt - the eight distinct block0 weight/bias pairs and the block1 set still need
binding, and the result then needs the four-processes-two-poisons treatment the branch uses.

Next is the guard: port the bounded/compacted/resume pipeline from `../fg-l20/`, and repair with the
guarded scalar helper rather than the CLZ one, since that is 1.74x cheaper and already exact.

## The guard pipeline ported to conv2: 3.0-3.4x, bit-exact, first build

The simplest 3.3x design from L20 - plain SIMD, prior-free fp32 bound, per-element flags, compaction,
indirect resuming repair - ported to the conv2 family. Same synthetic weights as above, final output
compared byte-for-byte against the scalar kernel:

| | ms | vs scalar | mismatches | flags | first uncertain stage |
|---|---:|---:|---:|---:|---:|
| scalar `fg_custom_block0_conv2` | 1.32 - 1.65 | 1.0x | - | - | - |
| **guarded pipeline** | **0.440 - 0.491** | **3.0 - 3.4x** | **0** | 16.92% | 13.2 of 36 |

(The scalar reference swings with clock state; the guarded pipeline holds within 0.05 ms.)

The port was mechanical: the bounded kernel is `custom_conv2_simd.metal` plus the guard block, the
repair is the shipped scalar kernel plus a list index and a resume point, and compaction is shared.
Nothing specific to L20 survived the move, which is what "the design generalises" has to mean.

At 3.3x the sixteen conv2 launches go from ~18 ms to ~5.5 ms, exact. The other nine custom launches
carry the remaining ~57 ms.

## block1 conv1 (module 63, launch 73): 4.03x, bit-exact, first build

A different shape from conv2: a 16 -> 32 conv at 320x184 **fused with a 2x2 average pool** to
160x92, where the scalar kernel runs four full convs per output and pools them with half rounding at
each add. It splits cleanly. `cv()` returns half, so a half intermediate at full resolution is
lossless; the guard pipeline produces it, and a fourth kernel repeats the pooling arithmetic
verbatim.

| | ms | vs scalar | mismatches | flags |
|---|---:|---:|---:|---:|
| scalar `fg_custom_block1_conv1` | 2.305 - 2.311 | 1.0x | - | - |
| **bounded -> compact -> listed repair -> pool** | **0.572** | **4.03x** | **0 of 471,040** | 7.35% of 1,884,160 intermediates |

Better than conv2's 3.3x: the conv is a single K=16 stage per tile, so the per-stage guard overhead
is spread over more useful work, and the pool is nearly free. Every one of the three timing runs
landed at 0.572 ms.

Second port, first build, exact. The pattern holds across two different epilogues.

## conv2 on live weights: the synthetic result holds

Four of the eight block0 weight/bias pairs (tokens 118/120, 122/124, 126/128, 130/132 - sizes
73,728 and 128 bytes confirm which is which), same harness, final output byte-for-byte against the
scalar kernel:

| pair | scalar ms | guarded ms | vs scalar | mismatches | flags |
|---|---:|---:|---:|---:|---:|
| 118/120 | 1.689 | 0.494 | 3.42x | 0 | 16.15% |
| 122/124 | 1.476 | 0.491 | 3.01x | 0 | 16.49% |
| 126/128 | 1.460 | 0.430 | 3.40x | 0 | 15.96% |
| 130/132 | 1.320 | 0.378 | 3.49x | 0 | 16.22% |

Exact on every pair; flag rate within half a point of synthetic. This is the scalar-oracle receipt.
The branch's own standard adds four fresh processes and two poison values, which is a harness
matter, not a numerics one.

## Custom block, scored

| launches | ms today (approx) | status | ms at measured ratio |
|---|---:|---|---:|
| conv2 x16 (modules 57, 64) | ~18 | **exact, live weights, 3.0-3.5x** | ~5.5 |
| block1 conv1 (63) | ~2.3 | **exact, synthetic, 4.03x** | ~0.6 |
| remaining 8 (54, 55, 56, 58, 59, 62, 65, 66) | ~55 | guarded scalar helper available at 1.74x | ~32 |
| **total** | **74.8** | | **~38** |

Porting the remaining eight at ~3.5x would put the block near 22 ms. The ceiling with byte-exactness
against NVIDIA's truncation is about there; under 10 ms is not reachable with this gate, for the
reasons in `../fg-l20/RESULTS.md`.

## block0 conv1 (module 56, launch 59): 3.9-4.0x, bit-exact, synthetic and live

The conv+pool template retargeted: 32 -> 64 at 160x92, two K=16 stages per tile, pooled to 80x46.
Weight strides (2304, 256, 128).

| weights | scalar ms | guarded ms | vs scalar | mismatches | flags |
|---|---:|---:|---:|---:|---:|
| synthetic | 2.445 - 3.318 | 0.622 - 0.710 | 3.9 - 4.7x | 0 of 235,520 | 11.32% |
| **live 114/116** | 2.442 - 2.447 | **0.612 - 0.614** | **3.98 - 3.99x** | **0** | 10.92% |

Third port, first build, exact on live weights. About 1.8 ms off the block.

## block0 conv0-c8 (module 55, launch 58): 3.9-4.0x, bit-exact, live - the largest launch so far

Conv+pool template: 32 -> 32 at 320x184, two stages, pooled to 160x92.

| weights | scalar ms | guarded ms | vs scalar | mismatches | flags |
|---|---:|---:|---:|---:|---:|
| synthetic | 4.647 - 4.680 | 1.197 - 1.199 | 3.88 - 3.90x | 0 of 471,040 | 11.33% |
| **live 110/112** | 4.636 - 4.647 | **1.161 - 1.164** | **3.98 - 4.00x** | **0** | 10.54% |

Fourth port, first build. About 3.5 ms off the block.

### Running tally, custom block

| launch(es) | scalar | guarded | receipt |
|---|---:|---:|---|
| conv2 x16 | ~18 | ~5.5 | live, 4 of 8 pairs |
| conv0-c8 | 4.64 | 1.16 | live |
| conv1 | 2.44 | 0.61 | live |
| block1 conv1 | 2.31 | 0.57 | synthetic |
| **ported so far** | **~27.4** | **~7.9** | |
| remaining 6 (54, 58, 59, 62, 65, 66) | ~47 | 1.74x helper: ~27 | |

## block0 convPre (module 54, launch 57): 3.2x, bit-exact, live - the largest single launch

Conv+pool template at the full 640x368 extent: ten real input channels zero-padded to K=16 (input
stride is 10, so the a-load carries a channel mask), 32 out, one stage, pooled to 320x184. 7.5M
intermediates.

| weights | scalar ms | guarded ms | vs scalar | mismatches | flags |
|---|---:|---:|---:|---:|---:|
| synthetic | 6.817 - 6.892 | 2.191 - 2.193 | 3.11 - 3.15x | 0 of 1,884,160 | 6.03% |
| **live 106/108** | 6.787 - 6.791 | **2.122 - 2.129** | **3.19 - 3.20x** | **0** | 5.09% |

Fifth port, first build. About 4.7 ms off the block. The lower ratio than the 4x launches is
consistent with the intermediate buffer - 15 MB written then read - becoming the bandwidth item at
this extent; the flag rate is the lowest yet.

### Running tally

| launch(es) | scalar | guarded | receipt |
|---|---:|---:|---|
| conv2 x16 | ~18 | ~5.5 | live |
| convPre | 6.79 | 2.12 | live |
| conv0-c8 | 4.64 | 1.16 | live |
| conv1 | 2.44 | 0.61 | live |
| block1 conv1 | 2.31 | 0.57 | synthetic |
| **ported (20 of 25)** | **~34.2** | **~10.0** | |
| remaining 5 (58, 59, 62, 65, 66) | ~40 | | |

## block1 conv0 (module 62, launch 72): 3.0-3.6x, bit-exact, live

Eighteen input channels at 640x368: a K=16 stage plus a K=8 tail (channels 16-17 in k-block 0,
everything else zero), each rounded to half separately, matching the scalar kernel's two HMMA calls
per tile. Weights are plain row-major `(oc*9+tile)*18 + k`, not fragment-ordered. Pooled to 320x184,
16 out.

| weights | scalar ms | guarded ms | vs scalar | mismatches | flags |
|---|---:|---:|---:|---:|---:|
| synthetic | 7.10 - 7.23 | 2.765 - 2.768 | 2.56 - 2.61x | 0 of 942,080 | 16.17% |
| **live 158/160** | 6.67 - 7.97 | **2.236** | **2.98 - 3.57x** | **0** | 9.79% |

Sixth port, first build. The live weights flag far less than synthetic (9.8% vs 16.2%), so the real
ratio is the higher one. About 4.7 ms off the block.

## block0 bottom0 (module 58, launch 68): 6.2x, bit-exact on three planes, live

Bilinear upsample 80x46 -> 160x92 of a 64-channel tensor, then a 3x3 conv 64 -> 96 in four stages,
leaky, clamp on branches 1-2 only, scattered to three 32-channel planes. The upsample is a fixed
half-arithmetic expression; a separate kernel materialises it once and the conv reads it as input.

| weights | scalar ms | pipeline ms | vs scalar | mismatches | flags |
|---|---:|---:|---:|---:|---:|
| synthetic | 10.98 - 11.00 | 2.100 - 2.102 | 5.23 - 5.24x | 0 of 1,413,120 | 16.94% |
| **live 150/152** | 11.04 - 11.14 | **1.779 - 1.780** | **6.20 - 6.26x** | **0** | 12.07% |

The highest ratio of any port, and for an algorithmic reason on top of operand sharing: the scalar
kernel recomputes `up()` for every one of the 36 (tile, stage) pairs at every output, so the same
bilinear sample is evaluated dozens of times. Materialising it once removes that. About 9.3 ms off
the block - the largest single saving.

### Running tally

| launch(es) | scalar | guarded | receipt |
|---|---:|---:|---|
| conv2 x16 | ~18 | ~5.5 | live |
| bottom0 | 11.0 | 1.78 | live |
| block1 conv0 | ~7.0 | 2.24 | live |
| convPre | 6.79 | 2.12 | live |
| conv0-c8 | 4.64 | 1.16 | live |
| conv1 | 2.44 | 0.61 | live |
| block1 conv1 | 2.31 | 0.57 | synthetic |
| **ported (22 of 25)** | **~52** | **~14** | |
| remaining 3 (59, 65, 66) | | | |

## block1 bottom0 (module 65, launch 82): 6.0-6.7x, bit-exact on three planes, live

bottom0's template at block1 dimensions: bilinear upsample 160x92 -> 320x184 of a 32-channel tensor,
conv 32 -> 48 in two stages (weight strides 2304/256/128), leaky, no clamp, scattered to three
16-channel planes. Files `custom_block1_bot0_pipeline.metal`, harness `b1b0_pipe.mm`.

| weights | scalar ms | pipeline ms | vs scalar | mismatches | flags |
|---|---:|---:|---:|---:|---:|
| synthetic | 10.95 - 10.99 | 1.815 - 1.816 | 6.03 - 6.05x | 0 of 2,826,240 | 11.27% |
| **live 198/200** | 10.93 - 10.98 | **1.647** | **6.64 - 6.67x** | **0** | 8.72% |

Eighth port, first build. About 9.3 ms off the block.

## block0 bottom1 (module 59, launch 69): 1.94-1.97x, bit-exact, live

New shape: three input planes (each upsampled 160x92 -> 320x184 at 32 channels) and the plane that
feeds the conv depends on the output channel (0-3 -> plane 0, 4 -> plane 1, 5-7 -> plane 2). That
breaks the MMA's shared-A structure - A is the pixel x channel operand and cannot differ per output
column - so the fast pass runs once per plane with a `plane` constant, computes all eight channels
from that plane, and writes only the channels the plane owns (outputs, flags and resume state). The
listed repair selects the plane per channel. Clamp +-6 on channels 4-7; three differently shaped
outputs (4 / 1 / 3 channels). Files `custom_block0_bot1_pipeline.metal`, harness `b0b1_pipe.mm`.

| weights | scalar ms | pipeline ms | vs scalar | mismatches | flags |
|---|---:|---:|---:|---:|---:|
| synthetic | 1.896 - 1.897 | 0.978 - 0.981 | 1.93 - 1.94x | 0 of 471,040 | 11.12% |
| **live 154/156** | 1.894 | **0.959 - 0.962** | **1.97x** | **0** | 9.42% |

Why only 2x, when bottom0 gets 6x: the fast pass is bound by A-tile loads (the same 9 taps x 32
channels per pixel per pass), and every output pixel genuinely needs all three planes' taps. Three
passes read exactly the useful A bytes but produce 24 channels to keep 8, so the fast pass costs
three groups' worth for one group of output - and the scalar kernel is cheap here (8 channels). A
single pass over a zero-padded K'=96 (three planes concatenated in K, weights zeroed off-plane)
would be numerically identical (adding exact zeros to an fp32 chain changes nothing) but reads the
same A bytes; it would save only the redundant B loads, so it was not built. About 0.9 ms off.

## block1 bottom1 (module 66, launch 83): 1.96-1.99x, bit-exact, live

Same three-plane structure at 640x368 with 16-channel latents (one K=16 stage per tap), then the
skip add: channels 0-3 `a = half(a + half(2*skip))` from the 4-channel skip; channels 4-7 clamp +-6
then `half(a + half(1*skip))` from the 1- and 3-channel skips; output interleaved 8 channels. Files
`custom_block1_bot1_pipeline.metal`, harness `b1b1_pipe.mm`.

| weights | scalar ms | pipeline ms | vs scalar | mismatches | flags |
|---|---:|---:|---:|---:|---:|
| synthetic | 3.589 - 3.590 | 1.827 - 1.836 | 1.96 - 1.97x | 0 of 1,884,160 | 7.00% |
| **live 202/204** | 3.582 - 3.584 | **1.798 - 1.800** | **1.99x** | **0** | 5.55% |

Same reason as block0 bottom1. About 1.8 ms off.

## block1 conv1 live receipt (tokens 162/164)

| weights | scalar ms | guarded ms | vs scalar | mismatches | flags |
|---|---:|---:|---:|---:|---:|
| **live 162/164** | 2.309 - 2.314 | **0.572 - 0.578** | **3.99 - 4.04x** | **0 of 471,040** | 7.35% |

Every port now has a live-weight receipt.

### Final tally: all 25 custom launches ported, bit-exact, live

Standalone harness times (min of 6, GPU timestamps), live weights:

| launch(es) | scalar | guarded | ratio |
|---|---:|---:|---:|
| conv2 x16 | ~18 | ~5.5 | 3.0-3.5x |
| bottom0 | 11.0 | 1.78 | 6.2x |
| block1 bottom0 | 10.9 | 1.65 | 6.7x |
| block1 conv0 | ~7.0 | 2.24 | 3.0-3.6x |
| convPre | 6.79 | 2.12 | 3.2x |
| conv0-c8 | 4.64 | 1.16 | 4.0x |
| block1 bottom1 | 3.58 | 1.80 | 2.0x |
| conv1 | 2.44 | 0.61 | 4.0x |
| block1 conv1 | 2.31 | 0.57 | 4.0x |
| block0 bottom1 | 1.89 | 0.96 | 2.0x |
| **all 25** | **~68.6** | **~18.4** | **~3.7x** |

The standalone scalar sum (~68.6 ms) is below the 74.8 ms the block measures inside the live FG
frame (dispatch gaps, cold caches, concurrent work), so the in-frame number for the ported block
should be read as roughly 18-20 ms, not 18.4; the ports are not yet wired into the FG runtime, so
that end-to-end number is unmeasured. What is measured: every launch is byte-identical to the
shipped scalar kernel on live weights, and the sum of the fast paths is 3.7x below the sum of the
scalar kernels.

Under 10 ms does not exist on this path with the byte-exact gate intact. The guard ceiling is
architectural (documented in fg-l20/RESULTS.md and above): the fast pass itself, not the bound,
sets the ~3.3-4x on plain convolutions, and the three-plane bottoms sit at 2x for the A-load reason
above. The two known routes below 10 ms both relax the gate: split-b MMA (8.65x, 0.096% of
halfwords differ, all by NVIDIA's per-term truncation - one fp16 ULP) or raw SIMD (17-19x, 0.19%).
Whether one ULP on 0.1% of intermediates is acceptable for a generated frame is a product decision,
and the right next measurement is PSNR of split-b against exact on a real frame, not more guards.

## Second design: where the time actually was, and a proof (conv2, 8 live pairs)

A correction first. Every earlier entry says the ~3.3-4x ceiling is the fast pass. It is not. Timing
the conv2 pipeline's three dispatches separately (`conv2_attrib.mm`, live weights 118/120, min of 8):

| stage | ms | share |
|---|---:|---:|
| bounded fast pass | 0.148 | 40% |
| compaction + indirect | 0.011 | 3% |
| listed scalar repair | **0.215** | **58%** |

The repair of 16% of the elements cost 1.45x the whole fast pass. Two more measurements said why.
Block-ordered compaction (each 256-element block ranked by prefix sum, one atomic per block, so adjacent
repair lanes share a pixel) left the repair unchanged at 0.213 ms: locality was not the problem. A
single-stage shortcut that settles an element after one exact stage when it agrees with the fast pass
(90% of flagged elements) moved it only to 0.227 ms even though it cut the executed stages ~7x: the
problem was **lane divergence**. A SIMD group runs its longest lane, and with one element in ten
needing the full replay, 97% of 32-lane groups still ran it. Splitting the shortcut and the replay into
two dispatches, each uniform, is what worked:

| repair | ms | mismatches |
|---|---:|---:|
| listed replay from the first flagged stage | 0.207 | 0 |
| + one-stage shortcut, same dispatch | 0.199 | 0 |
| **shortcut pass -> second list -> uniform replay** | **0.094** | **0** |

Why the shortcut is sound: an unflagged stage is certified - given the same prior, the fast pass's half
equals NVIDIA's. If the first flagged stage, recomputed exactly from its (certified) prior, equals the
half the fast pass produced there, and no later stage was flagged, both chains coincide to the end and
the fast output already written is correct. The fast pass now also stores that half and the number of
flagged stages (`custom_conv2_bounded_v2.metal`; repair in `custom_conv2_repair.metal`).

Fast-pass attribution on the same bench (diagnostic variants, not sound): baseline 0.148; bit-trick
ulp32 0.122 (frexp/exp2 cost 0.026); no ulp32 at all 0.095; no bound MMAs 0.108; no double-float fold
0.123; no guard at all 0.066; raw SIMD 0.059. The bit-trick ulp32 is exact for everything reachable -
products of two halves are never fp32 subnormal, and a zero bound means zero error - and is adopted.

### The bound was not a proof, and now is

Reading `numerics/HMMA_F16_RULE.md` against the shipped bound `4 ulp32(bnd0) + 4 ulp32(bnd1) +
2 ulp32(value)`: the prior participates in `max_exp`, so when the prior dominates (most late stages) all
sixteen products are truncated against the prior's grid, up to 17 units of `2^(max_exp-25)`, and the
only prior-scaled term in the bound is `2 ulp32(value)`. The shipped guards are exact on every tensor
measured (about 10^8 element-stages, zero mismatches) but they rely on the truncation being small in
practice (measured at most 3 units on L20), not on a bound. The formal bound:

    3.5 ulp32(bnd0') + 3.5 ulp32(bnd1') + 4.25 ulp32(M) + 1.5 ulp32(value)
    bnd' = bnd * (1 + 2^-20)     M = max(bnd0', bnd1', |prior|)

- 3.5: a K=8 FMA chain seeded at zero rounds 7 times (the first FMA is exact - a product of two halves
  fits fp32), each by at most half an ulp of a partial that is at most the true bound; the nudge keeps
  the computed bound's exponent from falling below a partial's when it rounds under a power of two.
- 4.25: 17 terms each truncated by less than one grid unit, grid = 2^(max_exp-25), and
  2^max_exp <= max(largest product, |prior|) <= M, so 17 grid <= 4.25 ulp32(M).
- 1.5: the double-float combination rounds once (0.5 ulp) and each interval endpoint once more, at most
  1 ulp of the value when an endpoint crosses a binade.
- fp16 subnormals: the oracle gives a subnormal operand exponent -14 whatever its magnitude, so a
  subnormal product can hold max_exp while lying far below 2^max_exp and `2^max_exp <= M` fails. The
  bound matrices therefore carry `max(|x|, 2^-14)` for nonzero operands (the exponent proxy): the
  proxy product is >= 2^(exp_a+exp_b) for every nonzero term and still >= |a||b|, so both the chain
  bound and M stay valid. Live conv2 weight tensors hold 18-49 subnormals each; flagging any tile that
  contains one (tried first) flagged 100% of elements because a weight tile is shared by every pixel.
  The proxy changes the flag count by zero (53,434 -> 53,434 on 118/120) and costs 0.012 ms.

### Numbers, all 8 live conv2 weight pairs (`conv2_pipe_v2.mm`, synthetic N(0,1) activations)

| design | pipeline, 8 launches | vs scalar | flagged | pass 2 | mismatches |
|---|---:|---:|---:|---:|---:|
| shipped (RESULTS above) | ~2.9 ms | 3.0-3.5x | 16% | | 0 |
| second design, empirical 4/4/2 bound | 1.857 ms | **5.07x** | 15-16% | 2.1% | 0 |
| **second design, formal bound (default)** | 2.133 ms | **4.75x** (4.5x steady clock) | 21-23% | 3.7% | **0** |

The formal bound flags 40% more elements and costs 12% - each extra flag is now one exact stage in a
uniform dispatch, which is why a proof became affordable. `FORMAL` is a function constant; the
empirical bound stays available for measurement. Per conv2 launch: 0.37 -> 0.25 ms. Over the 16 conv2
launches, about 2 ms off the block. The same changes port mechanically to the other nine families.

## Second design on all 25 launches, and conv2 designs three to six

Every family ported (`*_pipeline_v2.metal`, harness `*_pipe_v2.mm`; pass 2 is each family's original
listed kernel unchanged). Serial, uncontended, live weights, min of 6; mismatches are on the final
output against the shipped scalar kernel.

| launch(es) | scalar | first design | **formal** | empirical | flagged (formal) | pass 2 |
|---|---:|---:|---:|---:|---:|---:|
| conv2 x16 (design six, 8 pairs x2) | 18.1 | 5.5 | **3.32** (5.4x) | 3.06 (6.0x) | 22.7% | 0.9% |
| convPre | 6.78 | 2.12 | **1.78** (3.8x) | 1.60 (4.2x) | 7.8% | 0.50% |
| conv0-c8 | 4.63 | 1.16 | **0.76** (6.1x) | 0.67 (6.9x) | 14.9% | 1.7% |
| conv1 | 2.44 | 0.61 | **0.41** (6.0x) | 0.36 (6.8x) | 15.3% | 1.7% |
| block1 conv1 | 2.31 | 0.57 | **0.43** (5.4x) | 0.39 (5.9x) | 9.4% | 0.68% |
| block1 conv0 | 6.51 | 2.24 | **1.43** (4.6x) | 1.26 (5.2x) | 9.2% | 0.72% |
| bottom0 | 10.96 | 1.78 | **1.12** (9.8x) | 0.94 (11.7x) | 17.6% | 2.5% |
| block1 bottom0 | 10.87 | 1.65 | **1.12** (9.7x) | 0.98 (11.0x) | 12.5% | 1.3% |
| block0 bottom1 | 1.90 | 0.96 | **0.87** (2.2x) | 0.82 (2.3x) | 13.4% | 1.4% |
| block1 bottom1 | 3.58 | 1.80 | **1.67** (2.2x) | 1.57 (2.3x) | 7.9% | 0.53% |
| **all 25** | **~68.6** | **~18.4** | **12.9 (5.3x)** | **11.6 (5.9x)** | | |

All 25 launches byte-exact under both bounds. The formal bound (a proof) costs 11% over the
empirical one. The two bottom1 families barely moved because their cost is three upsample dispatches
plus three fast passes, not the repair.

### conv2, designs three to six (each exact on all 8 live pairs)

| design | change | pipeline per launch | vs scalar |
|---|---|---:|---:|
| two | two-pass repair, formal bound | 0.250 | 4.5x |
| three | second shortcut slot: the fast pass also records the prior and result at the second flagged stage; pass 1 checks it too (the stages between two flags are certified, so that prior is exact). Pass 2 falls from 3.7% to 0.9% of elements | 0.253 | 4.5x |
| four | lane-parallel exact stage (16 lanes per element, shuffle reductions, `hmma_exact_lanes.metalh`) | 0.291 | 3.9x |
| five | repair with `clz` instead of the leading-bit loops and half4 vector loads for the 16 contiguous activations (`hmma_exact_fast.metalh`, oracle header untouched) | 0.222 | 5.1x |
| **six** | two 8-channel output groups per simdgroup (activation tiles loaded once for 16 channels) and a shorter exact fold (two two_sums, errors added once; slack 1.5 -> 1.5625 for the extra rounding) | **0.205** | **5.5x** |

Design three did not move the clock although it cut pass 2 four-fold, which disproved the estimate
that pass 2 was 80% of the repair. Design four was the wrong diagnosis: the repair passes are not
latency-bound (16x the threads made pass 1 2.6x slower), they are instruction-bound, and the
data-dependent leading-bit loop in the oracle helper - up to 27 iterations, run to the slowest lane -
was the instruction. Design five is that fix. Stage times for design six on 118/120: fast pass 0.134,
compaction 0.007, pass 1 0.021, pass 2 0.048 ms.

### Dispatch overhead on M3 Max (`numerics/overhead.mm`, empty 64-thread dispatches, min of 10)

| form | per dispatch |
|---|---:|
| one encoder per dispatch, or one encoder with memoryBarrier between dispatches, or no barriers | 3.2-3.9 us |
| indirect dispatch (own encoder or concurrent encoder + barrier) | 6-9 us |
| a command buffer with a single dispatch | 7 us |

Encoder boundaries cost the same as in-encoder barriers, so there is nothing to gain from merging
encoders. Indirect dispatches cost about 5 us more than direct ones; each pipeline has two, so about
0.25 ms of the 12.9 is dispatch overhead. Stage times measured in separate command buffers carry the
7 us command-buffer floor each, which is why their sums exceed the single-command-buffer pipeline.

## Correction: block1 conv2 was never ported

The connected-block probe (`Mac-fg-research:src/fg_connected_custom_l56_l83/macos_connected_l56_l83.mm`)
dispatches the custom block as launches 56-83: eight `fg_custom_block0_conv2` (80x46, 64 channels,
tokens 118-148) and eight `fg_custom_block1_conv2` (module 64, 160x92, 32 channels, two K=16 stages
per tap, tokens 166-196). Every tally above wrote "conv2 x16" and priced all sixteen at block0 conv2's
ported cost. block1 conv2 had no port: 8.7 ms of scalar was being counted as ~1.7 ms. Now ported
(`custom_block1_conv2_bounded_v6.metal`, `custom_block1_conv2_repair_v5.metal`, harness
`b1c2_pipe_v6.mm`), design six:

| weights | bound | scalar, 8 launches | pipeline | vs scalar | flagged | pass 2 | mismatches |
|---|---|---:|---:|---:|---:|---:|---:|
| synthetic | formal | 1.31 (x1) | 0.223 | 5.9x | 15.6% | 0.43% | 0 |
| **live 166-196, 8 pairs** | **formal** | 8.66 | **1.48** | **5.85x** | 14.4-15.8% | 0.4% | **0 of 8 x 471,040** |
| live 166-196, 8 pairs | empirical | 8.66 | 1.32 | 6.6x | | | 0 |

Corrected block sum, formal: 12.9 - 1.66 (the eight launches wrongly priced) + 1.48 = **12.7 ms**;
empirical 11.5 ms. The number moved little because block1 conv2 happens to cost what block0 conv2
costs; the claim was unsupported until this measurement. conv2 design seven (four output groups per
simdgroup, `custom_conv2_bounded_v7.metal`) measured 0.138 vs 0.134 ms for the fast pass - register
pressure - and is not adopted.

## The block, end to end: 68.3 -> 11.9 ms, all 28 checkpoints byte-identical to the RTX 5090

`connected/macos_connected_pipelines.mm` (built by `connected/build_connected_pipelines.sh` in the
Mac-fg-research worktree) runs launches 56-83 exactly as Codex's connected probe does - same external
inputs, same model identity checks, same guarded buffers, one command buffer - with every custom
convolution replaced by its design-six pipeline (FORMAL bound), and compares all 28 launch checkpoints
against `evidence/specs/fg-connected-custom-l56-l83-v1/verification.json`, the hashes that matched the
RTX 5090 run of NVIDIA's own modules.

| path | block GPU time (min of 6, two runs) | checkpoints matching RTX |
|---|---:|---:|
| scalar oracles (Codex's connected probe, re-timed here) | 68.3 / 68.0 ms | 28/28 |
| **design-six pipelines, formal bound** | **11.92 / 11.93 ms** | **28/28** |

5.7x on the measured block, and the receipt is against RTX, not against our own scalar kernels. This
supersedes every standalone-sum estimate above (the sum predicted ~12.7 ms; the block measures 11.9
because the scalar kernels are slower in the block than standalone while the pipelines are not).

Per-launch flagged / pass-2 fractions inside the block, live activations: convPre 9.2%/0.07%,
conv0 14.5%/0.39%, conv1 15.3%/0.41%, conv2 15.6%/0.52%, bottom0 17.6%/0.68%, block1 conv0
6.7%/0.15%, block1 conv1 10.3%/0.22%, block1 conv2 19.0%/0.49%, block1 bottom0 18.5%/0.51%,
block0 bottom1 6.9%/0.20%, block1 bottom1 3.7%/0.11%.

### Two things the block taught that the standalone harnesses could not

1. **The bottom kernels' bilinear formula.** The block's bottom launches use
   `fg_connected_block*_bot*` (`src/fg_connected_custom_l56_l83/connected_*.metal`), whose 2x upsample
   blends with one half `fma` per direction; the `custom_*` scalar kernels the standalone harnesses
   compared against round each product first. Same weights, different last bit: on the real l67
   checkpoint the two disagree on 15.6% of upsampled halves. The pipelines' upsample pre-kernels now
   carry the connected formula (`connected_upsamples.metal`); the repair passes read the materialised
   tensor and needed no change. Any future bottom-branch kernel must be validated against the
   connected variant, not `custom_*`.
2. **Design six on all nine remaining families** (`*_pipeline_v3.metal`, harness `*_pipe_v3.mm`),
   each exact on synthetic and live weights under both bounds, interleaved against `_v2`:
   convPre 4.09x, conv0-c8 7.09x, conv1 6.89x, block1 conv1 5.85x, block1 conv0 5.10x, bottom0 12.5x,
   block1 bottom0 11.2x, block0 bottom1 2.9x, block1 bottom1 2.4x (formal). The two bottom1 kernels
   are now one fused dispatch over the three planes (weights loaded once; `_v3p` keeps the three-pass
   form, 10% slower). Pass 2 fell 3-4x everywhere from the second shortcut slot. block1 conv0 and
   convPre keep scalar activation loads in the repair (18- and 10-halfword rows are not 8-byte aligned
   at every stage start).

What remains for 10 ms is 1.9 ms across 28 launches with about 130 dispatches: the next measurement
is a per-launch split of the 11.9 inside the block.

### Inside the block: per-launch split and the price of the proof

`connected/macos_connected_pipelines.mm` now also times each launch in its own command buffer (each
figure carries the ~7 us command-buffer floor) and takes FORMAL as its fifth argument.

| launch | scalar | pipeline | | launch | scalar | pipeline |
|---|---:|---:|---|---|---:|---:|
| convPre | 6.81 | **1.66** | | block1 conv0 | 6.51 | **1.32** |
| conv0 | 4.71 | 0.68 | | block1 conv1 | 2.31 | 0.42 |
| conv1 | 2.48 | 0.36 | | block1 conv2 x8 | 8.60 | **1.52** |
| conv2 x8 | 8.97 | **1.65** | | block1 bottom0 | 10.83 | 1.04 |
| bottom0 | 10.84 | 0.91 | | block1 bottom1 | 3.50 | **1.48** |
| bottom1 | 1.86 | 0.76 | | initial, upsample x3, central | 0.19 | 0.19 |
| | | | | **sum** | **67.6** | **11.98** |

Formal vs empirical bound, in the block, interleaved twice: **12.12 / 12.12 ms formal, 11.02 / 11.04 ms
empirical**, both 28/28 RTX-exact. The proof costs 1.1 ms (9%). (This session's clock ran slightly
warmer than the 11.95 ms runs above; the scalar baseline moved the same way.)

Two more conv2 experiments, both exact on all 8 live pairs, interleaved against design six:

| | fast pass | pass 2 | pipeline | verdict |
|---|---:|---:|---:|---|
| design six | 0.1338 | 0.0480 | 0.205 | |
| eight: write resume state only for flagged elements (`custom_conv2_bounded_v8.metal`) | 0.1335 | 0.0476 | 0.206 | **no gain** - the ~300 MB of side-state stores across the block were never on the critical path; the estimate was wrong |
| nine: lane-parallel pass 2 with the clz helper (`custom_conv2_repair_v6.metal`, `hmma_exact_lanes_fast.metalh`) | 0.1337 | **0.0380** | **0.198** | adopted: pass 2 is latency-bound at 0.5% of elements, so the 16-lane form that lost at 3.7% now wins 20% |

The block at 10 ms with the proof intact is not reachable from what is measured here: the fast pass is
~65% of every pipeline and within ~2x of the guard-free SIMD floor, and its guard overhead is spread over
many small terms (bound MMAs and proxy ~0.03, ulp/compare ~0.03, fold ~0.02 ms per conv2 launch). The
remaining exactness-neutral items are lane-parallel pass 2 on all families (~0.3 ms) and fusing the 2x2
pool into the fast pass for the five pooled families (unmeasured, ~0.25 ms). With the empirical bound
those land near 10.4 ms; with the formal bound near 11.5 ms.

### Pool fused into the fast pass: convPre 1.66 -> 1.25 ms, exact

`custom_conv_pre_pipeline_pf.metal`: the simdgroup's eight pixels form a 4x2 block (matrix row bit0 =
x parity, bit2 = y parity), so every 2x2 pool window is complete inside the group; the epilogue pools
with three `simd_shuffle_xor` (lanes ^2, ^16, ^18) in the oracle's addition order and writes the pooled
output directly. The intermediate is still written (the repair needs the window's siblings); pass 2
rewrites intermediates as before and a pass 3 over list2 re-pools only those windows, so the 15 MB pool
read and its dispatch are gone. Interleaved against design six on live 106/108, FORMAL, contended:

| | pipeline ms | mismatches |
|---|---:|---:|
| design six (`_v3`) | 1.653 / 1.668 | 0 |
| **pool fused (`_pf`)** | **1.248 / 1.247** | **0 of 1,884,160** |

-25%, well beyond the pool dispatch alone: the 4x2 block's 3x3 taps touch 24 distinct pixels instead of
the 8x1 block's 30, a 20% smaller cache footprint for the activation loads. The mapping is being tested
on the non-pooled families next.

## Correctness: the exponent floor (a hole in the formal bound, found by an outside review)

The oracle initialises `maximum_exponent = -21` (`hmma_exact.metalh`), so the alignment grid is never
finer than `2^(-21-25) = 2^-46`, however small the terms. The formal bound's truncation term used
`M = max(bnd0', bnd1', |prior|)` and argued `2^max_exp <= M`; when the floor binds, `2^max_exp = 2^-21 > M`
and the term is too small. Reproduced under the model (`numerics/guard_floor_test.metal`): prior 0, one
product `2^-25` (a = 2^-13, b = 2^-12), fifteen products `2^-48` (a = b = 2^-24, subnormal). The fifteen
small products truncate to zero on the 2^-46 grid, the exact tie 2^-25 rounds to even (half 0x0000); the
fp32 value 2^-25 + 15 * 2^-48 rounds to 0x0001; the old radius (4.55e-14) accepts both endpoints as
0x0001. **An unsafe accept.** Fix: `M = max(2^-21, ...)`, radius 2.72e-13, flagged. Applied to all
eleven bounded kernels in the block; the empirical bound gets the same floor as an additive constant
(`4.25 * ulp32(2^-21)`), though it remains a non-proof for the other reasons recorded above. The block
re-verifies 28/28 with flag counts unchanged on live data: the floor only bites on sums below 2^-21.
This was not observed live; it is a hole in the derivation, closed. Credit: external review of commit
127e669 forwarded by the user.

## Pool fusion on all five pooled families, lane-parallel pass 2 where it wins

`make_pf.py` applies the convPre transformation to any pooled family (`*_pipeline_pf.metal`); all exact,
interleaved against `_v3` on live weights, FORMAL: conv0 0.66 -> 0.61, conv1 0.355 -> 0.330, block1 conv0
1.30 -> 1.23, block1 conv1 0.40 -> 0.34 ms. The 4x2 pixel mapping alone is neutral on conv2 (0.1365 vs
0.1346), so the gain is the pool dispatch and its intermediate read, not the cache footprint.

Lane-parallel pass 2 (`*_pipeline_v4.metal`, agents, all exact, interleaved vs `_v3`): conv2 -0.010,
block0 bottom1 -0.011, block1 bottom1 -0.005 ms; flat on convPre/conv0/conv1/block1 conv1 (9-18 stage
chains); slower on block1 conv0 (+0.02) and block1 bottom0 (+0.027, 9.5k-element list). The crossover
is list length: below ~4k elements the scalar pass 2 leaves the GPU idle and 16 lanes per element pay;
above it the 16x threads and shuffles cost more than they hide. Adopted for conv2, block1 conv2, and the
two bottom1 kernels only.

### Block, end to end, with fused pools and the floor fix

| bound | block GPU time | checkpoints |
|---|---:|---:|
| formal (proof, floor-fixed) | **11.60 ms** | 28/28 |
| empirical | **10.05 ms** | 28/28 |

## Two proposals from the external review, measured on conv2

**Row-statistics certificate** (`custom_conv2_bounded_v11.metal`): replaces the two bound MMAs with a
row max of proxied |a| (two shuffles, shared by both output groups) times a host-precomputed sum of
proxied |w| per (tile, stage, K=8 block, channel), and takes the truncation grid from exponents
(`max(-21, e(prior), eA0 + eB0, eA1 + eB1)`, `17 * 2^(E-25)`) instead of from the sum bound. Exact on all
8 live pairs. Fast pass 0.1334 -> 0.1276 ms (-4%), but the max-times-sum chain bound is looser: flags
22.8% -> 30.4%, pass 2 0.92% -> 1.59%, pipeline **0.198 -> 0.213 ms**. Not adopted: the repair it adds
costs more than the MMAs it removes, which is what the flag-rate arithmetic predicted. A hybrid
(cheap certificate first, MMA certificate on rejection) would refine nearly every simdgroup at a 30%
scattered flag rate, so it was not built.

**Unsigned-32 exact repair** (`hmma_exact_u32.metalh`): the oracle's 64-bit aligned sum replaced by two
unsigned 32-bit sign buckets (each aligned numerator < 2^27, sixteen plus the prior < 2^32 per bucket),
clz-based conversion, same rule. `numerics/hmma_u32_test.metal`: **16,777,216 random operand vectors
(normals, subnormals, zeros, max magnitudes, signs), 0 mismatches against the oracle.** conv2 pass 1
0.0223 -> 0.0193 ms (-13%), pipeline 0.198 -> 0.196. Adopted for every scalar exact stage in the block
(pass 1 everywhere, pass 2 where it is scalar).

## Block, end to end (fused pools, floor fix, u32 repair, lane-parallel pass 2 where it wins)

| bound | block GPU time, two runs | checkpoints matching RTX |
|---|---:|---:|
| **formal (proof)** | **11.24 / 11.06 ms** | 28/28 |
| **empirical** | **10.00 / 10.02 ms** | 28/28 |

The GPU ran warm for these (scalar baseline 71 ms against 68 earlier in the day), so both are
conservative by roughly 4%. From 74.8 ms in the FG frame (68-71 ms in this probe) to 11 with a proof
or 10 without one, every launch byte-identical to the RTX 5090.

## Toward sub-10: bottom1 own-column fold (adopted), exponent-based truncation term (rejected)

**Own-column fold** (`custom_block0_bot1_pipeline_v5.metal`, `custom_block1_bot1_pipeline_v5.metal`): the
fused bottom1 kernels still need all three planes' MMAs (each plane's A differs), but each lane's two
output columns belong to exactly one plane each, so the fold, certificate and checkpoint bookkeeping now
run once per lane per stage on the owning plane's partials instead of three times. Same arithmetic,
identical flag counts, exact. Interleaved against `_v4` on live weights, FORMAL (contended clock):
block1 bottom1 1.515 -> 1.358 ms, block0 bottom1 0.753 -> 0.678 ms (-10% each). The MMAs and operand
loads, not the fold, dominate these two kernels.

**Exponent-based truncation term** (`custom_conv2_bounded_v16.metal`): keeps the bound MMAs but takes the
truncation grid from `max(-21, e(prior), eA0 + eB0, eA1 + eB1)` (row max of proxied |a| by two shuffles,
host table of per-block max weight exponents) instead of `ulp32(max(bnd, |prior|))`. Sound, exact on
all 8 pairs, flags 22.75% -> 20.95% only, fast pass 0.134 -> 0.147 ms; pipeline 0.195 -> 0.211.
Rejected. It also settles where the formal bound's extra flags over the empirical one come from: in the
prior-dominated regime the truncation term is 17 grid units at the prior's own exponent, which is already
exact; only per-term low bits could shrink it. The proof's cost in the block (~1 ms) is the price of
NVIDIA's truncation rule, not of a loose estimate.

## Sub-10 push: packed weight tables everywhere, tail specialisation, fused indirects where they pay

All exact, all with identical flag counts to their predecessors, all interleaved on live weights:

| change | families | per-pipeline gain |
|---|---|---|
| packed weight/proxy `half4` table (`*_wq`, design thirteen): one coalesced load per (kb, group) per stage replaces 8 gathers + 8 `bound_operand` | all eleven | conv2 -5.2%, convPre -7.3%, conv0 -8.3%, conv1 -7.9%, block1 conv0 -7%, block1 conv1 -6.6%, bottom0 -5.9%, block1 bottom0 -5.5%, block0 bottom1 -2.6%, block1 bottom1 -1.7%, block1 conv2 -4.9% |
| K=8 tail specialised (`custom_block1_conv0_pipeline_pf2`): only kb=0 MMAs/loads in the tail, fold with t1 = 0 | block1 conv0 | **-19%** (1.23 -> 1.00 ms) |
| own-column fold in the fused bottom1 kernels (`_v5`) | both bottom1 | -10% each |
| fused indirect arguments (`compact_ordered_ind`, pass-1 `_ind`/`_v8`): the compaction and pass 1 write the dispatch arguments, two tiny dispatches gone; device-scope fence, Metal 3.2; 200-1200 runs stable | conv2 (few hundred pass-1 groups): -2.5%. convPre/conv0/conv1/block1 conv0 (2k-9k groups): neutral to -2% slower - the per-group barrier + fence scales with group count. Kept on conv2 only; bottom1s and block1 conv2 pending | |

Block probe with everything above except the pending indirects, FORMAL, under heavy agent contention
(scalar baseline 73.6 ms vs 68 quiet): **10.23 ms, 28/28 RTX-exact**.

## Second external review (on 9b93d7a): what held, what did not

- **Lane-parallel exact helper missed the -21 floor.** `sig != 0 ? exponent : -21` per lane, then a max,
  floors only if some product is zero; with sixteen nonzero products all below 2^-21 it picked the
  smaller exponent. Real, in the exact repair of conv2, block1 conv2 and both bottom1s. Fixed
  (`max(exponent, -21)`); `numerics/hmma_lanes_test.metal` adds an all-tiny mode and carries the old
  helper inline: 8.4M vectors, old helper 1 divergence from the oracle, fixed 0. The earlier 16.8M-vector
  test covered the u32 helper only; this one was missing.
- **v11/v16 statistics index double-counted `matrix_column`** (`oc0` already contains it) and read past
  the table. Fixed in v16 and rerun: flags 22.75% -> 20.71%, fast pass +10%, pipeline 0.199 -> 0.204 ms.
  Still rejected, now on correct data; v11 not rerun (its chain bound is the loose part).
- **n*delta truncation allowance**: the term holding max_exp is never shifted, and when the floor binds
  the prior is exact on the 2^-46 grid, so at most n nonzero products lose a grid unit. Coefficient
  4.25 -> 4.0 on K=16 stages, 2.5 on convPre (ten live channels), 0.5 on block1 conv0's two-product
  tail. Applied to all eleven kernels, exact, flags convPre 9.2 -> 8.0%, block1 conv0 6.7 -> 5.8%.
- **Re-pool ownership**: benign as built - pass 3 runs after pass 2 completes, every writer reads the same
  final intermediates and writes the identical pooled value.
- **Directed interval endpoints** (`custom_conv2_bounded_v17.metal`: step each endpoint one float outward,
  fold slack 1.5625 -> 0.5625): sound, exact, and useless - flags 52,612 -> 52,610, fast pass slower.
  conv2's flags come from the cancellation regime (bnd >> value), where only the chain terms and the M
  term matter; the value-scaled term is noise. Not adopted.

### Block, GPU quiet (scalar baseline 68 ms), min of 15, two runs each

| bound | block GPU time | checkpoints |
|---|---:|---:|
| **formal (proof)** | **10.08 / 10.06 ms** | 28/28 |
| **empirical** | **9.41 / 9.40 ms** | 28/28 |

## Where the sub-10 push ended (2026-09-04, evening)

Launch 70's three plane upsamples fused into one dispatch (`connected/custom_upsample_planes3.metal`,
per-element arithmetic verbatim, plane by z): 0.138 -> 0.131 ms, checkpoint 70 unchanged.

Block probe, min of 15, three runs in a row with desktop compositing active (scalar baseline 72 ms
against 68 when the machine is idle):

| bound | block GPU time | checkpoints matching RTX |
|---|---:|---:|
| **formal (proof)** | **9.98 / 10.05 ms** | 28/28 |
| **empirical** | **9.35 ms** | 28/28 |

Earlier in the day with the GPU idle the same design read 10.06-10.08 formal / 9.40 empirical before
the upsample fusion. The proven path is at the 10 ms line to within the measurement noise of a machine
in use (about +-0.05 ms); the empirical path is under it by 0.6 ms. From 74.8 ms in the FG frame.

What did not move the needle and is recorded above so nobody retries it blind: 4 output groups per
simdgroup, conditional side-state writes, the row-statistics certificate (even index-corrected), the
exponent-based truncation grid (even index-corrected), directed interval endpoints, lane-parallel pass 2
on long lists, fused indirect arguments on families with thousands of pass-1 groups, and the 4x2 pixel
mapping without a pool to fuse.

## Into the FG runtime: 102.4 -> 79.3 ms per generated frame, same receipt as the accepted build

The guarded block now runs inside Codex's FG session (`Mac-fg-research` branch `fg-guarded-runtime`,
commit 5071c3d: `src/fg_connected_l20_l87/guarded_block.inc` + a switchable swap of the 21 dispatch
lines in the runtime tail; copies in `final/runtime/`). Gate: the runtime's own four-process offline
receipt against the accepted u32/compact-EX2/live-auxiliary build: **checkpoint root equal (1,149
checkpoints, 16 frames), all 15 output hashes equal, zero Metal allocations after warm-up.**

Two things the live frames taught that the synthetic connected receipt could not:

1. **A stage result that rounds to zero is +0 on RTX.** The oracle helpers kept the sign (-0); the
   runtime's connected bottom kernels carry a post-stage `-0 -> +0` rule for exactly this (Codex's
   "live zero" overlays), and three live elements hit it. Applied in every fast pass (the rounded
   value and both interval endpoints are zero-normalised) and in the exact helpers. The standalone
   `custom_*` scalar oracles do not carry the rule, so per-family harness comparisons can now differ
   on rounds-to-zero elements; the runtime receipt is the authority.
2. conv1's packed weight table was built with conv2's strides (4608/512) instead of conv0's (2304/256)
   in the runtime host; every frame's l59 differed until fixed. The per-family harness had it right.

Live benchmark (`benchmark-macos-fg-runtime-live.sh`, 45 frames over 3 processes, output hashes gated):

| runtime | p50 | p95 | p99 |
|---|---:|---:|---:|
| accepted u32 build (scalar block) | 102.36 ms | 103.61 | 108.62 |
| **guarded block** | **79.34 ms** | 85.32 | 86.11 |

The block fell from ~68 to ~10 GPU ms; the frame fell 23 ms. The rest is the other 62 launches and the
runtime's synchronous phase waits, which is where the frame work continues.

## The rest of the FG frame: 79.3 -> 16.2 ms, and the frame is now the GPU (2026-09-04, night)

Per-span profile of the 79 ms frame (`FG_RUNTIME_PROFILE_SPANS=1`): 24 command buffers, **15.7 ms of
GPU** in total (block 11.7, U-Net prefix L20-L51 3.6, everything else 0.4), 23.7 ms of commit+wait.
The other ~45 ms were host time. A CPU sample showed `std::vector<uint8_t>` constructors and
destructors running byte by byte: the runtime hosts were built with **no `-O` flag at all**. Two
changes, both gated by the same four-process offline receipt and the 45-frame live benchmark (output
hashes equal to the accepted build on every frame):

| runtime (same kernels, same receipt) | p50 | p95 | p99 |
|---|---:|---:|---:|
| guarded block, -O0 hosts (previous row) | 79.34 ms | 85.32 | 86.11 |
| `-O2` hosts (`FG_RUNTIME_CXX_OPT`, default `-O2`) | 22.6-26.5 ms | 23.6-29.3 | 23.9-33.5 |
| **+ deferred intra-frame waits** | **16.2-16.4 ms** | 16.6-16.9 | 16.9-17.0 |

Deferred waits: in non-audit, non-profiling frames every intermediate command buffer is committed
without a CPU wait (one queue, in-order execution); the output copy waits for everything and
`NSFGSpanFlush()` at the end of `Evaluate` guarantees no GPU work outlives the call (the reset path
writes buffers from the CPU). Safe because a non-audit frame never touches GPU-visible memory from the
CPU: `Guarded`/`NSFGCachedRawBuffer`/textures only re-fill under audit, counters are blit-filled.
Interleaved A/B (v7 / v6 / v7): 16.35 / 22.57 / 16.18 ms p50. `NSFG_SYNC_WAITS=1` restores the old
behaviour. fgwt commit `314b7f3` (branch `fg-guarded-runtime`, pushed to `origin`).

What is left is GPU time: block 11.7 ms (9.98 in the idle probe), prefix U-Net 3.6 ms, early
launches and tail under 1 ms combined.

## Frame generation in the game: what RE9 actually asks for (2026-09-04, night)

With `FrameGeneration=On` in RE9's `config.ini` (the game rewrites it to `DLSS 2x`) under GPTK 4.0b2,
D3DMetal maps DLSS-G onto a **Metal 4 MetalFX frame interpolator** (`_M4FXFrameInterpolatorEffect`,
`newFrameInterpolatorWithDevice:compiler:`). The bridge now hooks that seam observe-only (own IMPs,
registry and mutex; original called exactly once; `event=fg_*` lines) and the SR hook declines
interpolator classes explicitly. Observed contract at 1728x972 output, DLSS Balanced render 1002x564:

| input | value |
|---|---|
| color / prevColor / ui / output | 1728x1117 textures (content 1728x972), format 90 (BGR10A2), prevColor nil |
| depth | 1002x564, format 260 (Depth32Float_Stencil8) |
| motion | 1002x564, RG16Float, motion scale (1002, 564) |
| distortion | 864x558, RGBA16Float |
| per frame | deltaTime 25.3 ms steady, jitter passed through, near 0.1, far 999.9, fov 90 deg, UI composited |
| pacing | ~38 source fps with FG on (26 ms/encode, SR sustained alongside); 60-70 fps after FG stopped |

3,480 interpolated encodes passed through without a fallback or error before the encodes stopped and
the game exited (no crash report). So **frame generation is running in the game today, through
MetalFX, watched by Netal**. It is not Netal's network: the exact FG runtime is pinned at 640x360,
and at 1080p the DLSS-G graph runs at 9x the pixel count (Codex's resolution matrix: 87 of 89
launch grids scale), i.e. ~140 ms per generated frame at today's 16 ms - not a real-time option on
M3 Max even before the dynamic-resolution lowering exists. The app's Frame Generation row shows
"Observed" with the interpolator's output size and source fps, never "Active".

## Netal FG in the game (2026-09-04, 19:44): the exact network generates RE9's frames

Route: GPTK 4 maps DLSS-G onto a Metal 4 MetalFX frame interpolator; the bridge now serves that seam.
Pieces, each gated before the game saw it:

- **Metal 4 shim** for the runtime (`src/fg_runtime/metal4_shim.mm` on `fg-guarded-runtime`): stand-in
  queue/command buffer/encoder objects answer the runtime's Metal 3 selectors and forward into one
  `MTL4ComputeCommandEncoder` on the game's command buffer (argument table per dispatch, 1 MB constants
  ring per slot for `setBytes:`, barrier after every command, residency set learned from every bound
  resource; 191 dispatches, 633 resident resources per frame). Gate: `fg-runtime-metal4-probe`, the
  accepted 16-frame sequence on MTL4 command buffers, **15/15 output hashes identical, 3 passes**,
  1.4 ms CPU encode, 16.5 ms commit-to-complete. Two bugs found by the probe: argument tables cap at
  31 buffer binds, and the constants ring must itself be resident.
- **x86_64 runtime** (Rosetta, for the CrossOver game process): accepted hash, 29 ms p50 CPU-bound.
- **Bundle** (`scripts/build-macos-fg-runtime-bundle.sh`): universal static archive + 56 metallibs +
  model tokens + tables + manifest, 17 MB; the Netal bridge links it.
- **Session + adapter** (`src/macos_host/dlss_fg_session.mm` on Netal dev): game colour (BGR10A2,
  1728x1117 with a 1728x972 content rect) box-sampled to RGBA8 640x360; motion (RG16F 1002x564,
  scale 1002,564) converted to 640x360 pixel units; depth D32S8 point-sampled to R32F; generated frame
  bilinear-upsampled back over the content rect, rows outside pass the real frame through; reset
  frames present the real frame. Offline gate at RE9 geometry: reset exact, 1.2/1023 red error
  against the mean of a sliding gradient, 1.4 ms encode, ~16 ms GPU.
- **Hook**: `TryServeFG` in the interpolator's encode, Metal 4 only, fail-open (original called once on
  any refusal, disarm after 240 consecutive fallbacks), `--fg` on `mac-dlss.sh`; interpolator classes
  already loaded at injection are hooked by a class scan (the first attempt missed an interpolator
  created inside the 75 s settle window).

Live: `event=fg_served` frames 1..360+ with `fallbacks=0`; source 22-23 fps (43-45 ms per real frame)
against ~38 fps with MetalFX interpolating, so the exact 640x360 network costs the game roughly 18 ms
more per source frame than Apple's interpolator at this size. Presented rate about 46 fps. This is
"Netal frame generation running in the game": exact network, resolution-decoupled adapter.

## First in-game defect: blurred rectangles were a baked test fixture (2026-09-04, 20:05)

In RE9 every generated frame carried two soft-edged blurred rectangles with a dotted texture. A live
dump (`NEURALSTUDIO_FG_DUMP_DIR`) of the runtime's 640x360 inputs/outputs and an offline replay
(`fg-runtime-dump-replay`) reproduced the in-game output byte-for-byte from a fresh reset, so there
was no race and no shim fault. Ablating motion, depth and the previous frame changed nothing (the
scene was static); the block input (l56) and output (l83) showed no rectangles; the tail did. Launch
84's `inpainting_mask_texture_sampler` (`d3d-resource-000003`, R8 640x360, no producer among the 83
selected launches, DLSS-G-owned, zero in the core feature mode) was fed by `TailMask()`: the
connected receipt's "islands" fixture, two rectangles at (104..224, 72..184) and (376..552,
196..320) with a `(13x+7y) mod 47` hole pattern, exactly the boxes on screen. Live frames now bind a
zero mask; on the dumped frames the generated image moved from 1.55 to 0.22 mean levels off the
real frame with no pixel more than 8 off. The accepted runtime receipt and every earlier FG frame
hash include the fixture, so the live-mask build gets its own receipt.

## Resolution-decoupled compositor: generated frames at the game's resolution (2026-09-04, 21:15)

The 640x360 network output was upsampled 2.7x to 1728x972, so every generated frame was soft. The
network's *pixel-producing* stage is only two launches: 55 (candidate-warped surfaces from the two real
frames using launch 53's flow candidates and launch 54's blend weights) and 84 (fine flows and a
sigmoid blend over those surfaces). Both address by normalised coordinates, so the FG session now
re-evaluates them at the game's size over the retained real frames (`dlss_fg_session.mm`,
`fg_fullres_candidates` + `fg_fullres_blend`), with the runtime exporting the four resources it needs
(`NSFGSessionCopyLastTailResources`). Everything else - the whole network - still runs at 640x360.

Gates: at native 640x360 geometry the compositor reproduces the runtime's own output to within 4/1023
(mean 1.0, i.e. the runtime's 8-bit quantisation), no outliers. Live in RE9 at 1728x972, dumped frames:

| path | gradient energy of generated / real frame | mean abs diff vs real (8-bit) |
|---|---:|---:|
| 640x360 upsample | 0.44 | 6.0 |
| full-resolution compositor | 0.92 | 0.5 |

Encode cost unchanged (1.4 ms CPU); GPU cost of the two full-size passes is inside measurement noise
with the game running (needs commit feedback for a clean number). Inpainting (launches 85-87) is not
reproduced: with the zero mask it touches only two corner pixels.

## Hardware lowering for the game path: 16.2 -> 5.6 ms per generated frame (2026-09-04, 22:30)

Goal: a generated frame cheap enough that FG adds frames instead of trading them (render ~15 ms at
1728x972 native, so generation must be well under ~5 ms). The exact runtime stays the oracle; the game
path is gated by PSNR against it on dumped RE9 frames (replay tool + `fg-mode-psnr.py`).

| step | frame p50 (benchmark host) | GPU | PSNR vs exact |
|---|---:|---:|---:|
| exact (u32, zero mask) | 16.2 ms | 15.7 | - |
| block HW (`NSFG_BLOCK_MODE=hw`: fast passes with HW=true, no bound/flags/repair/re-pool) | 9.8 | | 77 dB |
| + prefix HW via fp32-FMA override of the HMMA helper | 8.9 | 7.4 (block 4.8, prefix 2.2) | 77 dB |
| + upsample kernels vectorised per pixel (bit-identical; exact receipt unchanged) | 7.4 | 6.0 (block 3.5) | 77 dB |
| + one encoder for the 33 prefix launches | 7.3 | 6.0 | (no GPU change: the time was in the kernels) |
| + dedicated prefix conv kernels (`conv_3x3_hw`, `conv_1x1_hw`: contiguous half4 loads, oc-major dispatch) | **5.6** | **4.3** (block 3.4, prefix 0.5, rest 0.4) | 77 dB (max 1/255) |

Findings: (1) the HMMA emulation cost nothing visible - the hardware modes are within 1 level of 255 of the
exact frames; (2) the block's scalar bilinear upsamples were 1.8 of 5.1 ms (30 GB/s, channel-strided
writes); vectorising them is bit-identical and also speeds up the exact path; (3) the prefix's 435M MACs
are under 0.1 ms of arithmetic - its 2.2 ms was gather loops and per-element index maths, and the packed
weight layout is contiguous per stage under a fixed permutation, so the hardware kernels load it as four
half4. Per-dispatch GPU timestamps (`NSFG_BLOCK_TIMING=1`, stage-boundary counters) drove the ranking.
Next: the eleven block fast passes (3.4 ms; conv2 x8 0.51, block1 conv2 x8 0.48, convPre 0.45, block1
conv0 0.35, bottoms 1.0).

## In the game with the hardware lowering, and the Metal 3 seam (2026-09-04, 23:30)

Host cost first: under Rosetta the bridge's per-frame encode was 8.5 ms of CPU (SHA-256 of the ROM and
sigmoid tables every frame, payload copies); cached reads and audit-only read-backs bring it to ~1.3 ms
(x86 probe), receipt unchanged (v16). Then RE9 at 1728x972 native, FrameRate=Max120, same scene, Netal
FG toggled through `netal-control` seconds apart (60-frame windows from `fg_pacing`):

| interpolator | source frame | source fps | output fps |
|---|---:|---:|---:|
| MetalFX frame interpolator (Netal paused) | 18.5-20.8 ms | 48-54 | ~100-108 |
| Netal FG, hardware lowering, full-resolution compositor | 20.9-23.0 ms | 44-48 | ~88-96 |

The game is GPU-bound, so generation cost lands on the source frame directly: Netal's generated frame
costs ~2-3 ms more GPU than MetalFX's at this resolution (runtime 640x360 ~4.3 ms + imports, colour copy
and the two full-resolution compositor kernels at 1728x972). Earlier in the session, before the host fix,
the same comparison read 40 vs 60 (a lighter scene); the gap closed from ~8 ms to ~2.5 ms per source
frame. `fg_served` counters: 0 fallbacks, fullres every generated frame. What is left is GPU work: the
eleven block fast passes (3.4 ms) and the compositor at output resolution.

Integration for other titles: the runtime shim now accepts an `MTLCommandBuffer` as the frame target
(real compute/blit encoders from the game's buffer, hazard-tracked, no argument tables or residency), the
FG session has `encodeToCommandBuffer:` and the bridge serves `_MFXFrameInterpolatorEffect` (Metal 3
MetalFX titles) as well as the Metal 4 effect (`fg_served api=metal3|metal4`). `scripts/netal-native.sh`
stages the FG bundle into the game's container and arms FG by default when it exists (`--no-fg`). The
Metal 3 path is built and receipt-clean (v17) but has not yet been exercised by a Metal 3 title: RE9
under GPTK 4 creates its interpolators on the Metal 4 class.

## Fused bilinear fetch (MLX-DLSS output-head fusion): exact, slower on M3 Max - rejected (2026-09-10)

The idea (iamwavecut/MLX-DLSS@0ca2dea, `Sources/DLSSMLX/FrameGenerationFusedConv.swift`, `docs/frame-generation.md`
"Native output-head fusion"): the convolution that follows a 2x bilinear upsample fetches its enlarged input inside the
kernel instead of reading a materialised tensor, and still rounds the interpolated values through fp16. Their M2 Max
figures (whole warm FG graph 4.44 -> 3.77 ms at 960x540, 14.3 -> 12.1 ms at 1920x1080) are equivalence to their own
earlier implementation, not RTX exactness, and say nothing about this lane.

Where it applies: the block's four upsample -> convolution boundaries, launches 68 (bottom0, 80x46x64 -> 160x92), 69
(bottom1, three 160x92x32 planes -> 320x184), 82 (block1 bottom0, 160x92x32 -> 320x184) and 83 (block1 bottom1, three
320x184x16 planes -> 640x368). NVIDIA's kernels already evaluate `up()` inline for every (tap, stage); these pipelines
materialise it once with the vectorised connected pre-kernels. Launch 70's upsample feeds `central`, not a convolution.
MLX-DLSS fused only its linear output head, the analogue of 69 and 83.

The rounding is kept by construction: the fused fetch is the connected formula (`wx1*b` rounded, one half fma per row,
`wy1*bot` rounded, one half fma) evaluated on the low-resolution source at the point of use, operand for operand the
arithmetic of `connected_upsamples.metal`. MLX-DLSS's own form (fp32 blend, half4 round trip) is a different formula and
was not used. Kernels `custom_*_pipeline_fu.metal`, generated by `src/fg_conv/make_fu.py` from the `_wq` references
(which are unchanged) by anchored substitutions of the activation fetch only: half2 per lane pair in the fast pass, half4
in pass 1, scalar in the lane-parallel pass 2; fetch in `fu_upsample.metalh`. Runtime switch `NSFG_FUSED_UPSAMPLE=1`
(or a launch list) in `guarded_block.inc`, off by default. Branch `fg-fused-upsample`, commit 6c3ad20, evidence in
`evidence/specs/fg-fused-upsample-v1/`. `-fno-fast-math -ffp-contract=off`, Metal 32023.883; timed runs under the NR GPU lock.

### Exactness: every gate passed

| gate | result |
|---|---|
| operand check: the fused fetch at every element vs the materialised tensor, scalar / half2 / half4 forms | 19,783,680 operands x 3 forms, byte-identical |
| connected block probe (launches 56-83), 2 processes x 20 runs of 15 | 28/28 RTX checkpoints in every run of both variants |
| bottom flagged / pass-2 counts (bot0, bot1, b1 bot0, b1 bot1) | identical: 244,119/9,426, 31,430/946, 513,534/14,264, 67,296/2,055 |
| runtime receipt: 4 fresh processes, both poisons, 1,149 checkpoints x 16 frames, 15 outputs | rebuilt baseline, fused 68+69+82+83 and fused 69+83 all give root fe4aa401 and the live-zero-mask outputs |
| complete-frame benchmark: pre / measured / post output hashes of every process | 24 processes, all equal to the receipt |

### The block (probe, one command buffer, variants alternated; scalar reference 69.8 / 70.7 ms, slightly warm)

| launch (with its upsample dispatches), min of 15 | materialised | fused | delta |
|---|---:|---:|---:|
| 68 bottom0 | 0.802 / 0.798 | 0.828 / 0.829 | +0.026 / +0.031 |
| 69 bottom1 | 0.311 / 0.310 | 0.326 / 0.326 | +0.014 / +0.017 |
| 82 block1 bottom0 | 0.873 / 0.855 | 0.910 / 0.906 | +0.037 / +0.051 |
| 83 block1 bottom1 | 0.612 / 0.613 | 0.649 / 0.648 | +0.037 / +0.036 |
| all 28 launches, own command buffers | 8.797 / 8.769 | 8.908 / 8.899 | +0.111 / +0.130 |
| whole block, median of 10 per-round minima | 8.971 / 8.881 | 9.009 / 9.324 | +0.04 / +0.44 |

Two processes, "/" separated. The second process's block rounds were noisy; its per-launch split was not.

### Inside the runtime: where the time went (`NSFG_BLOCK_TIMING`, median per frame, 84 frames per variant)

| | materialised | fused | delta |
|---|---:|---:|---:|
| upsample pre-kernels 68 / 69 (x3) / 82 / 83 (x3) | 0.010 / 0.046 / 0.012 / 0.098 | - | -0.166 |
| fast passes 68 / 69 / 82 / 83 | 0.651 / 0.241 / 0.656 / 0.455 | 0.680 / 0.281 / 0.702 / 0.566 | +0.226 |
| repair passes 1 and 2, four families | 0.383 | 0.408 | +0.025 |
| bottom families | 2.553 | 2.637 | +0.084 |
| block (124 -> 116 dispatches) | 9.360 | 9.458 | +0.098 |

Net per family: 68 +0.033, 69 -0.003, 82 +0.042, 83 +0.013 ms. The linear heads come closest to break-even; none gains.

### The complete generated frame (benchmark host, 640x360, 6 alternating process pairs, 90 measured frames per variant)

| candidate | materialised p50 / p95 | fused p50 / p95 | delta p50 | pairs with fused slower |
|---|---:|---:|---:|---:|
| 68 + 69 + 82 + 83 | 13.174 / 13.629 | 13.327 / 13.802 | +0.153 ms | 6 of 6 |
| 69 + 83 (MLX-DLSS's scope) | 13.595 / 14.000 | 13.671 / 14.130 | +0.076 ms | 4 of 6, the other two within 0.03 ms |

The machine ran warmer during the second A/B; each A/B is alternated within itself. Baseline context: on this build and
toolchain the materialised path with fusion off measures ~13.2 ms p50 for the frame and ~8.8-8.9 ms for the block in the
probe (16.2 and ~10.0 on 2026-09-04); the closure note below says what that difference is not attributed to.

**Rejected.** Why the MLX-DLSS gain does not transfer: there it removed an MLX operation (allocation, graph node,
dispatch) in front of a scalar per-pixel kernel. Here the buffers are preallocated, the eight pre-kernel dispatches
already stream (0.166 ms for 19.8M operands), and the SIMD fast pass reads each upsampled operand many times - once per
tap x stage x output-channel group x plane: 54 reads per materialised operand on 68, 9 on 69, 27 on 82, 9 on 83 (50.9M,
50.9M, 50.9M and 101.7M fused evaluations per frame) - and each read becomes four texel loads and six half operations
instead of one load. A per-simdgroup local materialisation (the 3x10 upsampled neighbourhood of each 8-pixel row) would
still evaluate 30 positions where the pre-kernel evaluates 8, per output-channel group, and was not built.

Reproduction notes from this run: `xcrun` on this Mac has no Metal toolchain (Command Line Tools only; the
`~/Library/Developer/Toolchains/Metal.xctoolchain` link is dangling), so the kernels were built with the mounted
`metaltc-17F109` image through a PATH `xcrun` shim. Homebrew bash 5.3.9 deadlocks on the build wrappers' multi-KB
python heredocs; `/bin/bash` runs them once `build-macos-fg-runtime.sh` expands its optional HMMA flags portably
(fixed on the branch). The connected probe on `fg-guarded-runtime` still dispatched the upsample pre-kernels in their
pre-1cb64ff 3-D shape (64x redundant threads on launch 68) and did not specialise `HW`; both fixed on the branch before
any timing.

### Closed 2026-09-10: rejected for the tested configuration

Tested configuration: M3 Max, macOS 27, Metal 32023.883, the 640x360 exact lane, fused launches 68+69+82+83 and 69+83.
The materialised upsample stays the default: nothing enables `NSFG_FUSED_UPSAMPLE` except the A/B harness's fused arm.
Preserved in the NeuralStudio repository (local refs, not pushed):

| ref | commit | contents |
|---|---|---|
| tag `fg-fused-upsample-evaluated-2026-09-10` | 6c3ad20 | the commit exactly as evaluated and cited above |
| branch `fg-build-probe-fixes` | f7f1228 | only the fixes worth keeping without fusion: portable empty-array expansion and an optional per-process receipt prefix (GPU lock runner) in `build-macos-fg-runtime.sh`; the probe's runtime upsample dispatch shapes and `HW=false` specialisation |
| branch `fg-fused-upsample` | f7f1228 -> 6a7fa8a -> fe096fa | the fixes, then the fusion work (tree-identical to 6c3ad20), then the closure note and the five `.log` evidence files `.gitignore` had dropped |

Every baseline figure in this entry (frame ~13.2 ms p50, block ~8.8-9.4 ms) is the materialised path with fusion off.
The difference from the 16.2 ms frame of 2026-09-04 was not investigated here and is not attributed to anything; it is
not a fusion effect, since fusion only ever added time.

