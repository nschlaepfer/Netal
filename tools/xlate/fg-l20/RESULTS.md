# FG L20 on the real layer: what the MMA-bounded guard actually does

Measured 2026-09-04 against the live L20 (160x96, 3x3, 48 -> 32, weight token 63) from
`NeuralStudio-fg-research/build/fg_default_readback/eval0-default-all-a5-a/`, M3 Max.
`guard_bench.mm` builds three kernels against the same weights and input and compares every one of
491,520 output halfwords.

| kernel | ms | vs scalar | mismatching halfwords | flagged | uncovered |
|---|---:|---:|---:|---:|---:|
| `fg_conv_3x3` scalar CLZ oracle | 1.74 - 1.83 | 1.0x | - | - | - |
| `fg_conv_3x3_simd` raw | 0.096 - 0.103 | **17 - 19x** | 952 (0.19%) | - | - |
| `fg_conv_3x3_simd_bounded` (this work) | 0.207 | **8.4x** | 952 | 37.5% | **0** |
| `fg_conv_3x3` + double-float helper | 1.00 - 1.31 | 1.4 - 1.8x | 455 (0.09%) | - | - |

The scalar reference reproduces the branch's own 1.762 ms, and the raw SIMD reproduces its 18.0x, so
the harness agrees with the existing receipts before any of this is read.

## What worked

**The guard is sound and it is cheap.** All 952 real mismatches are flagged; none escape. And the
guard costs **2x the raw path** (0.103 -> 0.207 ms) where the shipped guarded kernel cost **52x**
(0.103 -> 5.39 ms). Moving the error bound from a per-lane scalar scan over 32 operands to a second
MMA over `|a|` and `|b|` is worth about 26x on the guard itself, which was the prediction and it
held.

## What did not

**The end-to-end two-pass is 2.02x, not the 9-12x I predicted.** The prediction was made on a single
K=16 stage. L20 runs **27** of them (3 channel stages x 9 spatial taps), and an element is condemned
if any one stage's bound cannot decide. At ~1.7% per stage that compounds to 37%, and 37% of the
scalar path costs more than the fast path saves.

The bound is ~240x more conservative than reality per stage: 0.007% of stages actually round
differently, and the guard flags 1.7%. The gap is cancellation - the bound uses `ulp32(P)` where `P`
is the sum of magnitudes, and `P/|v|` runs about 9x on this data.

**Splitting the bound into two K=8 halves made it worse**, not better: the flag rate moved only
37.5% -> 33.2% while the kernel went 0.207 -> 0.785 ms, because reading a partial out of the
accumulator mid-chain forces a spill. Rejected and reverted.

**The double-float helper is not exact here.** It is a drop-in for `fg_hmma_f16_bits_16` and it
disagrees with the CLZ oracle on 455 halfwords. That corrects an earlier conclusion of mine: I had
calibrated "exact sum, one rounding" as the reading of NVIDIA's rule that best matched FG's measured
117-byte divergence, and on synthetic data it was bit-identical to my own oracle. Against the real
oracle it is not. **The two alignment bits and the truncation are real**; they just bite on ~0.09%
of stages rather than the 44% a per-term truncation would give. My synthetic oracle implemented the
wrong one of the three readings, and only the real layer could show that.

## Where this leaves the lane

The guard mechanism is fixed and proven. The blocker has moved from "the guard costs more than it
saves" to "the bound cannot be made tight enough to survive 27 compounding stages", which is a
different and better-posed problem. Two routes worth trying, in order:

1. **Correct the flagged stage in place** instead of condemning the element. The operands for a
   flagged element live across the simdgroup's lanes, so this needs `simd_shuffle` to gather 16
   values - affordable only if divergence is handled, since any flagged lane makes its whole
   simdgroup pay.
2. **A tighter rigorous bound.** The looseness is entirely `ulp32(P)` versus the running partials.
   Anything that bounds the partials without reading them mid-chain would move the flag rate
   directly, and a 10x tightening turns 37% into roughly 5% and the two-pass into roughly 6x.

Raw SIMD at 17-19x with 0.19% divergence remains available to any mode that does not require
byte-exactness. Under the current gate it is not.

## Ledger

Interleaved three rounds each, same weights and input, on the live L20.

| candidate | kernel ms | flags | uncovered | two-pass | verdict |
|---|---:|---:|---:|---:|---|
| single bound, `8*ulp32(P)` | 0.207 | 37.47% | 0 | 2.02x | baseline |
| split K=8, partial read mid-chain | 0.785 | 33.22% | 0 | 1.28x | **rejected** - spill |
| split K=8, two independent accumulators | **0.168 - 0.199** | **33.22%** | 0 | **2.24 - 2.36x** | accepted |

**The spill is gone.** Two bound accumulators that are neither seeded from each other nor read until
both chains close cost 0.168-0.199 ms against 0.785 for the version that read a partial mid-chain -
the same arithmetic, 4x the speed, and identical flag counts to the halfword (163,282), which is
what confirms only the codegen changed.

**The tightening itself is small.** 37.47% -> 33.22% of elements flagged, two-pass 2.02x -> 2.30x.
Nowhere near the 10x bound tightening the 6x target needs.

A correction to the previous commit: the file it recorded as the bounded kernel was the *spilling*
version. `git checkout --` silently no-ops on an untracked file, and stderr was discarded, so the
revert I reported did not happen. The table above is measured against the file now in this
directory.

### Why the bound resists tightening

Flag rate per stage is roughly `(P/|v|)/512`, where `P` bounds the partials and 512 is the ratio of
the fp16 decision margin to eight fp32 ulps. Measured flag rate is 1.49% per stage, so `P/|v|` is
about 7.6 on this data - cancellation. Getting to a 4% overall flag rate needs `P/|v|` near 0.76,
which no rigorous bound can deliver, because `P` genuinely is the worst case when the products
cancel.

The remaining looseness is not in the bound's form but in worst-case-per-FMA accounting: sixteen
half-ulp roundings are assumed to align, and in practice they partly cancel. That gap cannot be
closed soundly, which points the lane at item 2 - correcting the flagged stage rather than
condemning the element - rather than at further bound work.

## Correction: every two-pass number above was an estimate, and the estimate was wrong

Built the masked second pass and measured the real thing. Three runs, identical to the millisecond:

| | ms | vs scalar | final output |
|---|---:|---:|---|
| scalar reference | 1.739 | 1.0x | - |
| **real two-pass** (bounded + masked) | **1.918** | **0.91x** | **0 mismatching halfwords** |
| what I had been quoting | 0.762 | 2.28x | - |

**The two-pass is slower than just running the scalar kernel.** Not 2.02x, not 2.30x, not 2.36x -
0.91x. Every two-pass figure in this file and in commits 0c42c6b and 68e7f0a was an estimate that
assumed the masked pass costs its mask share, and that assumption is false.

The cause is divergence, and it was predictable from the dispatch geometry alone. Threads go out as
`(160, 96, 32)` in threadgroups of `(8, 4, 1)` - 32 threads each. With 33.22% of elements flagged,
the chance that a threadgroup contains no flagged element is `(1 - 0.3322)^32`, which is about one
in 170,000. Essentially **every** threadgroup runs the full convolution, so the early-out saves
nothing and the mask read plus the second encoder cost 0.18 ms on top.

The exactness result stands and is worth keeping: the final output is byte-identical to the oracle
across all 491,520 halfwords. The guard is sound and the second pass repairs exactly what it should.
Only the performance premise was fiction.

### What this changes

- Item 1's remaining value is now clearly zero: tightening 33.22% to, say, 25% still leaves
  `(1-0.25)^32` at about one in 10,000, so divergence still eats the whole layer. **The flag rate
  would have to fall below roughly 3% before a per-element mask helps at all**, and the bound
  analysis above says it cannot.
- Item 2 changes shape. The useful version is not "correct the flagged stage in place" but
  **compact the flagged elements first** - a prefix-sum over the mask producing a dense index list,
  then a dispatch sized to that list. Compaction is what makes the mask share real instead of
  notional, and only then does 33% cost 33%.
- The lesson generalises past this lane: a mask over a scattered 33% of a SIMD dispatch is not a
  33% dispatch. Anything in this project that plans a guarded fast path needs a compaction step in
  the plan from the start, or its speedup is imaginary.

## Compaction, and a bound that does not carry the prior: 2.65x, bit-exact

All measured end to end on the live L20, three runs each, final output compared byte-for-byte
against the scalar oracle over all 491,520 halfwords.

| pipeline | ms | vs scalar | flags | mismatches |
|---|---:|---:|---:|---:|
| masked two-pass | 1.918 | 0.91x | 33.22% | 0 |
| **compacted** three-pass | 0.811 - 0.825 | **2.13x** | 33.22% | 0 |
| compacted + prior-free bound | 0.588 - 0.595 | 2.92x | 16.13% | **94** - unsound, rejected |
| **compacted + prior-free bound + combination term** | **0.653 - 0.662** | **2.65x** | **18.41%** | **0** |

**Compaction is what makes a mask worth having.** A prefix-append into a dense index list plus an
indirect dispatch turns 0.91x into 2.13x with no change to the arithmetic. Threadgroups over the
compacted list are entirely live or entirely dead; over the raw mask they were all live.

**And it reverses the previous section's conclusion.** With compaction the flag rate maps almost
linearly onto time, so tightening the bound is valuable again - 33% gives 2.13x, 18% gives 2.65x,
and 5% would give about 6x. The claim that item 1 was worth zero was true only of the masked design.

**The prior-free bound.** Seeding the value accumulators at zero instead of at the running
accumulator, computing two independent K=8 sums, and folding in the prior with double-float
arithmetic means the error bound no longer carries `|prior|` - which dominates it after a few stages
and was most of the 7.6x gap between `P` and `|v|`. Flags nearly halve.

**It failed soundness first, and the failure is instructive.** The version without a term for the
combination step let 94 halfwords through: the double-float fold rounds when its low word is
absorbed, and `hi + lo` rounds once more into fp32. A bound that covers only some of the error is
not a bound. Adding `2 * ulp32(value)` restores exactness and costs 2.3 points of flag rate.

### Where the remaining time is

At 18.41%: fast pass ~0.19 ms, compaction ~0.05, repair ~0.32. The repair still dominates, and
reaching the 10 ms block target needs roughly 3% flags. The two surviving terms are
`4*ulp32(B0) + 4*ulp32(B1)` - eight FMA roundings per half, each charged at half an ulp of that
half's total product magnitude. Charging them against their own running partials instead would be
worth about 1.8x if it could be done without reading a partial mid-chain, which is exactly what
spilled last time.

## Per-element flags and a resuming repair: 3.37x

| pipeline | flags | ms | vs scalar | mismatches |
|---|---:|---:|---:|---:|
| compacted + prior-free bound | 18.41% | 0.657 | 2.65x | 0 |
| **+ per-element flags + resume** | **9.68%** | **0.517 - 0.522** | **3.37x** | **0** |

**A bug that had been inflating the flag rate the whole time.** `thread_elements()[0]` and `[1]` of
a `simdgroup_float8x8` are two *different output channels*, and every version until now kept one
`suspect_element` flag per row block. Flagging either channel condemned both. Splitting the flag,
the first-uncertain stage and the resume accumulator to per-element takes 18.41% to 9.68% - almost
exactly the halving that implies, since the two channels go uncertain nearly independently.

It also made the resume record wrong for one of each pair, which is how it surfaced: the resuming
repair returned 41,414 mismatching halfwords until the tracking was split.

**Resuming rather than restarting.** The fast pass now records, per element, the stage at which it
first became uncertain and the accumulator as it stood going in. Everything earlier already produced
the right half, so the repair replays from there. The mean first-uncertain stage is 9.4 of 27, so
65% of each flagged element is replayed rather than 100%.

At 3.37x the custom block is about 22 ms rather than 74.8.

### Remaining cost, measured

Of the 0.52 ms: the fast pass is ~0.19, compaction ~0.05, repair the rest. The raw unguarded SIMD is
0.098 ms, so **the guard roughly doubles the fast pass** - two extra MMAs for `|a|.|b|` plus the
per-element bound arithmetic. Reaching the 10 ms block target needs about 0.23 ms total, which the
fast pass alone nearly consumes; from here the fast pass matters as much as the flag rate does.

## Scoping correction: L20 is not in the 74.8 ms

The previous section claimed "the custom block would be about 22 ms rather than 74.8". That is
wrong, and it overstates the impact of everything above by roughly eight times.

From `docs/FG_RECONSTRUCTION_STATUS.md`, the runtime span profile: *"The custom L56-L83 core alone
is about 74.756 ms GPU; generic L20-L51 is about 9 ms."* And from the module table, the custom block
is launches **57-83**, modules 54-66. **L20 is launch 20, module 28 - the generic encoder/decoder
family.**

So the 3.37x measured here applies to the **9 ms** group, not the 74.8 ms one. It is worth about
6.3 ms of the 83.8 ms GPU frame - real, and about an eighth of what the last commit implied.

### Does it transfer?

The custom kernels do share the same primitive: `custom_conv_pre.metal`, `custom_conv2.metal`,
`custom_block1_conv1.metal` and `custom_block1_bot1.metal` all call `fg_hmma_f16_bits_16`, and the
optimization route already counts 42 convolution-family launches over both groups. So the guard
technique is applicable in principle.

What does not transfer for free is the fast path. Everything above is built on `l20_simd.metal` - a
SIMD/MMA rewrite of one specific layer's loop structure - and **no equivalent exists for any of the
25 custom launches**. Each needs its own, and the conv2 families alone are 16 of the 25.

### What that implies for the target

The 10 ms goal is against the custom block, and reaching it now means either 25 SIMD rewrites, or a
change to the shared scalar helper that benefits every launch without one. The second is the far
better lever and it is only partly explored: the branch already has `hmma_exact_u32` at 1.28-1.56x
isolated, and the question of whether an exact scalar helper can go faster still is open. That is
where the 74.8 ms is reachable from, not from more work on L20.

## The guard inside the shared helper: 1.74x on every launch, and why it stops there

`hmma_guarded_override.metalh` moves the guard into `fg_hmma_f16_bits_16` itself. The helper holds
all sixteen operands in registers, so an uncertain stage repairs itself on the spot and the
accumulator stays correct - **nothing compounds**, unlike the SIMD version where 1.5% per stage
became 9.68% of elements. And it needs no SIMD rewrite, so it reaches all 42 convolution-family
launches through the existing `-include` override, including the 25 custom ones.

| | ms | vs CLZ scalar | mismatches |
|---|---:|---:|---:|
| CLZ scalar reference | 1.738 - 1.739 | 1.0x | - |
| **guarded helper** | **1.002** | **1.74x** | **0** |
| its fast path with the fallback removed | 0.996 | 1.75x | 500 (unsound; ablation only) |

### The fallback is free, and that is the finding

Removing the exact fallback entirely saves **0.006 ms**. The guard and its repair cost essentially
nothing; the 1.74x ceiling is the fp32 fast path itself, which still costs 57% of the full bit
decomposition. That is not what an arithmetic-bound kernel looks like.

**These convolutions are memory-bound.** The scalar kernel reads 27 stages x 32 halfwords per output
element - 1,728 bytes - and at 491,520 outputs that is about 849 MB for one layer. At 1.74 ms that
is roughly 488 GB/s against the M3 Max's ~400 GB/s of bandwidth, so it is at or past the roof even
before cache reuse is counted.

Which explains the shape of everything measured so far:

- the guarded helper gains 1.74x rather than the 4-5x its arithmetic saving implies, because the
  arithmetic was never the constraint;
- the SIMD path gains 18x not because MMAs are fast but because **a simdgroup shares its operands
  across 32 lanes**, cutting the memory traffic by that factor. Its advantage is bandwidth, not math.

### What that means for the 74.8 ms

A better scalar helper cannot get the custom block to 10 ms - it is bandwidth-limited at about
1.74x, which is 74.8 -> 43 ms. Reaching the target needs the operand sharing, which means the
SIMD/MMA rewrite for each custom kernel, 25 of them, with the conv2 families accounting for 16.

The helper is still worth taking: it is exact, it is a one-line include change per build, and it
costs nothing per kernel. Applied across both groups it is about 36 ms off the 83.8 ms GPU frame -
74.8 -> 43 and 9 -> 5.2 - with the generic group able to do better still via the L20 SIMD guard at
3.37x.

## Split-precision MMA: the exact-sum floor, and what the residual actually is

A fresh premise. fp16 x fp16 products are 22 bits wide and eight of them summed need 25, so the raw
SIMD path loses bits on essentially every stage - the 0.19% is only how often that crosses an fp16
boundary. Rather than guard the loss, remove it: split the weight into hi (top 5 fraction bits) and
lo (exact remainder) on the host, run two MMAs per k-block against a 16- and a 17-bit product, and
combine the two partial sums with the prior in double-float. NVIDIA's own 3xTF32 trick, applied to
fp16.

| kernel | ms | vs oracle | differing vs oracle |
|---|---:|---:|---:|
| double-float scalar (computes the exact sum) | 1.004 | 1.73x | 455 |
| raw SIMD | 0.102 | 17.1x | 952 |
| **split-b** | **0.200** | **8.65x** | **471** |
| split4 (a and b, 8 MMAs) | 0.250 | 7.0x | 474 |

**Split-b vs the double-float scalar: 62 of 491,520 elements differ.** Splitting the activations as
well moves 471 to 474 - nothing. The split MMA is computing the exact sum, at 8.65x, with no guard,
no compaction and no repair pass.

**So the ~455-474 floor is not fp32 precision. It is NVIDIA's truncation rule.** Three independent
routes to the exact sum - double-float scalar, split-b, split4 - all land within 20 of each other
against the oracle, and none reach zero. Whatever HMMA.16816.F16 does to terms far below the largest
exponent, it is not "sum exactly and round once", and no method that computes the true sum can
match it. Closing the last 0.09% requires replicating that rule, which requires the maximum term
exponent per output element per stage - the one quantity a simdgroup layout does not hand any single
lane.

A bug worth recording from the first build: the bias lives in the same buffer as the weights, and
splitting the whole buffer truncated it. 63% of outputs wrong, from five cleared bits in 512 halves.

### Where that leaves the choices

- **Exact today:** the guarded/compacted/resuming pipeline, 3.37x. Now known to be paying for the
  truncation semantics, not for fp32.
- **Best non-exact:** split-b at 8.65x and 0.096% divergence, in about forty lines with no second
  pass. Or raw SIMD at 17x and 0.19%.
- **Exact and fast** needs the truncation rule pinned on hardware and then a way to get the maximum
  exponent per element cheaply. The first is a rig job; the second is open.

## Truncation-aware guard on the split kernel: sound, and worse

Since split-b computes the exact sum, the only remaining error is NVIDIA's truncation, and its bound
is a different quantity: under 17 grid units at `2^(max_exp - 25)`. I expected that to be much
tighter than the fp32-accumulation bound. Measured through the full compacted pipeline:

| pipeline | flags | ms | vs scalar | mismatches |
|---|---:|---:|---:|---:|
| plain SIMD + prior-free fp32 bound + compaction + resume | 9.68% | 0.461 - 0.528 | 3.30 - 3.77x | 0 |
| **split-b + truncation bound + compaction + resume** | **13.76%** | 0.605 | 2.87x | 0 |

Sound - every mismatch flagged - and slower. Two reasons, one arithmetic and one structural.

Arithmetic: 17 units at `2^(e-26)` against `8 * ulp32` at `2^(e-21)` is 17/32 - about 2x tighter per
unit, not 7x. I miscounted.

Structural, and decisive: `max_exp` is the maximum over **all 17 terms including the prior**, so the
truncation bound cannot be made prior-free. The fp32 bound could, because accumulation error depends
only on the products. After a few stages the prior dominates the sum, so `exp(bound)` tracks the
prior and the truncation guard is inherently looser than a prior-free fp32 guard. Making it prior-free
would be unsound.

### What the fresh approach established, net

- **Split-precision MMA is the best non-exact path**: 8.65x at 0.096% divergence, ~40 lines, no
  second pass. Twice as accurate as raw SIMD at half its speed.
- **The 0.09% floor is NVIDIA's truncation rule**, not precision - three independent exact-sum
  routes agree to within 20 elements of each other and none reach zero.
- **The exact SIMD ceiling is about 3.4x** with a guard, and the guard that gets there is the
  prior-free fp32 one on the plain SIMD kernel, not a smarter one on the split kernel. The split
  kernel's accuracy is wasted under a guard, because the guard's flag rate is set by the bound, not
  by the actual error.
- **Exact and fast requires `max_exp` per element per stage.** It is a max-plus matrix product -
  tropical, not linear - and the MMA computes sums. `Σ 2^(e_a + e_b)` pins it to within four binades,
  which is what `exp(Σ|a||b|)` already gives; pinning it exactly needs a base large enough that the
  runner-up terms cannot carry, and fp32's exponent range does not accommodate that across fp16's
  full product range. Open, and not obviously closable.

## max_exp from MMAs: exact on 483,360 of 483,360

The open problem was `max_exp` - NVIDIA's maximum over the 17 term exponents - which no single lane
of a simdgroup holds. It is a max-plus product and MMAs sum. But `Σ_k 2^(e_a[r,k] + e_b[k,c])` is an
ordinary product of `2^e_a` and `2^e_b`, and its leading bit lies in `[max_exp, max_exp+4]`, because
at most 17 terms can share the top. In **base 32** the runner-ups cannot carry into the leading digit
- `17 * 32^(m-1) < 32^m` - so `floor(log32)` is exact. Shifting every exponent against a stage-wide
maximum (one `simd_max`, since 32 lanes x 4 values covers all 128 operands) keeps `32^(shift)` inside
fp32's range, with a clamp at -12 for terms so far below they cannot matter.

Measured on L20 stage 0 against a scalar truth kernel (`maxexp_probe.metal`, `maxexp_bench.mm`):

| method | leading position minus true max_exp |
|---|---|
| base-2 sum | +0: 6.5%, +1: 50.7%, +2: 41.6%, +3: 1.2% |
| **base-32, stage-shifted** | **+0: 100.0000%** |

Two MMAs and one reduction per stage, and the quantity that was "not obviously closable" two
sections ago is exact everywhere.

What it buys: the truncation guard's bound becomes exactly `17 * 2^(max_exp - 25)` with none of the
up-to-four-binade slack of `exp(Σ|a||b|)`, and the prior's exponent is already in the lane. Combined
with split-b's exact sum, the only thing left to guard is the truncation itself, bounded tightly.

## Exact max_exp in the guard: the tightest yet, and the same speed - which is the finding

| guard | flags | ms | vs scalar | mismatches |
|---|---:|---:|---:|---:|
| plain SIMD + prior-free fp32 bound | 9.68% | 0.518 - 0.528 | 3.30 - 3.77x | 0 |
| split-b + truncation bound via `exp(Σ|a||b|)` | 13.76% | 0.605 | 2.87x | 0 |
| **split-b + truncation bound via exact max_exp** | **6.97%** | 0.518 - 0.531 | 3.27 - 3.36x | 0 |

The exact max_exp does what it should: 17 grid units at `2^(max_exp - 25)` with no binade slack
takes flags from 9.68% to 6.97%, the lowest of any sound guard. The repair pass shrinks
accordingly. And the total does not move, because the fast pass that produces it - four value
MMAs, two max_exp MMAs, exponent extraction, a `simd_max`, a double-float combination - costs about
0.40 ms against 0.10 for raw SIMD, and that is what the lower flag rate bought back.

**Every guard variant lands at ~3.3x on L20.** Three different bounds, three different flag rates,
one speed. The ceiling is not the guard's tightness; it is the cost of computing an exact sum and a
sound bound in the same pass, which is roughly 4x the raw kernel however it is arranged. A fourth
guard design will not move it.

What would: a fast pass that is exact without a guard - which needs per-term truncation, which the
MMA cannot do - or a decision that byte-exactness against NVIDIA's truncation rule is the
reconstruction standard and not the shipping standard. The first is closed by the instruction's
shape. The second is not mine to make.

The value now is in **generalising the simplest 3.3x design** - plain SIMD, prior-free fp32 bound,
per-element flags, compaction, resume - to the 25 custom launches where the 74.8 ms is.
