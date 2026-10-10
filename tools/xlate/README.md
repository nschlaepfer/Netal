# xlate — one harness for porting CUDA kernels to Metal and making them fast

Two features have now been reconstructed from NVIDIA binaries: Super Resolution on `dev`, Frame
Generation on `Mac-fg-research`. They were built independently and they rediscovered the same
things — the same `mma.sync.aligned.m16n8k16` arithmetic, the same SM120 transcendental behaviour,
the same texture-sampling rules, the same "prove it exact, then make it fast" loop. FG's own
`src/fg_conv/README.md` says its kernel "carries forward the exact **SR** HMMA arithmetic
hypothesis": the same instruction, characterised twice.

Ray reconstruction and neural rendering are next, and they are built from the same parts again.
This directory exists so the third feature costs less than the second did.

It is two things: a **registry** of what has been proven and where it lives, and a **runner** that
enforces the discipline both lanes arrived at the hard way.

## Why a runner at all — three failures, each of which happened

**A candidate that never executes.** `ExactLibrary()` prefers `<name>_exact.metallib`, so the
obvious-looking library is not the live one. Two rounds of SR optimisation were spent on a library
that never ran; the tell was a *null kernel* producing bit-exact receipts. On 2026-09-04
`try-hiluma-candidate.sh` was still doing this — its header says "output" and its body swaps
`hiluma_input_depthinv.metallib`. **So the runner builds and runs the null control itself, first,
and refuses to report a number if the receipts do not move.**

**A measurement that cannot decide.** Three consecutive runs of one unmodified library measured
6.43, 7.41 and 6.22 ms. Every single-run A/B at that resolution is a coin flip, and one was: a
descriptor change looked like a 0.3 ms win against a stale baseline and lost all three pairs once
interleaved. **So the runner always interleaves, always measures the incumbent's own spread, and
reports a delta only when it clears that spread.**

**A negative result that cost the same as a positive one and was thrown away.** Four hypotheses
died on the SR output pass in one afternoon. Each was worth recording; none had anywhere to go.
FG's `evidence/specs/<unit>/failure-ledger.md` is the right answer and it should not be
per-lane. **So rejection writes a ledger entry automatically. A dead end is an output, not a
non-result.**

## Runner

    tools/xlate/try-candidate.sh units/<unit>.json <candidate.metal|.metallib> <label>

The unit manifest names the bundle, the artifact the candidate replaces, the oracle hashes, the
probe and its environment, and which env var multiplies the timed dispatch. Everything else is the
same for every lane, which is the point.

Order is fixed and not skippable: null control → exactness → interleaved timing → verdict. Timing
never runs on a candidate that failed exactness, because a fast wrong kernel is not a result.

## Registry: what is proven, where it lives, and who still needs it

| primitive | proven in | form | portable to |
|---|---|---|---|
| `mma.sync.aligned.m16n8k16.row.col` exact arithmetic | SR (`pwin_gemm3`, hardware simdgroup + `split_k16` rounding schedule) and FG (`fg_hmma_f16_bits_16`, scalar oracle) | SR's is fast and shipping; FG's is exact-but-scalar | **FG's convolutions** — see below |
| `ex2.approx.ftz.f32` | FG, exhaustive over all 2,139,095,040 finite negative patterns, 6.3 MB compact | complete | **SR**, which currently memoises and drifts |
| `lg2.approx.ftz.f32` | neither — SR memoises 79,911 observed keys | incomplete | everything |
| `rcp.approx.ftz.f32` | SR, `1.0f/x` + 2-bit ROM correction | complete for the domain in use | FG uses the same ROM |
| texture fetch / `tld4` gather | SR (CuMetal emulation, packed-texel and gather hoists) | shipping | RR, which is sampler-heavy |
| Blackwell rounded-zero rule (`-0 -> +0`) | FG, per HMMA stage, lookup-free | complete | SR's emulation had the same bug |

### The two entries that are worth acting on now

**FG's EX2 into SR.** SR's documented limit is that "the exact mode's lg2/ex2 tables cover the
captured sequences only — unseen content drifts to 67-80 dB after frame 0". That is a quality bug
in every real game, because game content is always unseen. Its EX2 call sites take negative inputs
in `[-3.47, -0.38]`, which is entirely inside FG's proven domain. This is not an optimisation; it
is the difference between exact on four captures and exact on anything.

**SR's HMMA lowering into FG.** FG spends 74.8 of its 83.8 ms GPU frame in convolutions whose
kernel is a scalar correctness oracle, and measured a 16x8 simdgroup tile at ~18x before rejecting
it over a 117-byte difference. SR met that wall and did not reject the fast path: it *checked* it
(`build-pwin-fastwmma-candidates.py --variant checked`) and fell back to exact emulation only where
the check fails. 117 bytes of a 983,040-byte tensor is 0.01% of positions.

One thing gates that port and is not yet answered: SR's hardware path accumulates in **fp32**, while
FG's selected instruction is `f16.f16.f16.f16` — a *half* accumulator with two alignment bits and
truncation, which keeps fewer bits than fp32, not more. Whether the checked lowering already covers
that variant decides whether this is a port or a research problem. It is one experiment; it is
`units/hmma-accumulate-variant.md`.

## What this is not

It does not translate PTX automatically, and nothing here should be read as suggesting a
general-purpose CUDA-to-Metal compiler is the goal. Every unit in both lanes was recovered by
reading the selected PTX and proving one kernel at a time against a hardware oracle. What compounds
is the *parts list* and the *loop* — not the translation.

## Ray reconstruction and neural rendering

They are not the same kind of next step, and treating them as one would waste the harness.

**Ray reconstruction is the same network family as SR.** NVIDIA ships it through the same runtime,
and it is a U-Net with windowed attention over G-buffer inputs — noisy radiance, albedo, normals,
roughness — instead of colour alone. Every primitive in the parts list applies unchanged: the same
HMMA, the same transcendentals, the same texture emulation, the same window/shift structure the SR
layers already implement. What is genuinely new is input plumbing, not arithmetic. That makes RR
the cheapest third feature and the one that proves whether the parts list actually compounds.

**Neural rendering is a different shape.** Neural texture compression, radiance caches and RTX
Neural Shaders are small MLPs evaluated inside a shader through cooperative vectors, not a U-Net
run once per frame. The parts list still helps — the arithmetic is the same silicon — but the
structure, the scheduling and the integration point are all different, and Rigel's reverse
engineering found Metal 4's tensor path executing on the shader cores with no dedicated matrix
datapath before M5. So the numerics port over; the performance story does not, and it should be
started only after RR has tested whether the registry pays.

**The interchangeable claim is testable, and RR is the test.** If the parts list is real, RR's
arithmetic layer should be assembly rather than rediscovery, and the work should collapse to input
handling plus the network's own structure. If it is not real, that shows up immediately as another
independent characterisation of the same instruction — which is exactly what happened between SR
and FG, and exactly what this directory exists to stop happening a third time.

## FG custom block

`fg-custom/` holds the guarded exact pipelines for FG's 25 custom convolution launches (74.8 -> ~10 ms,
28/28 launch checkpoints byte-identical to the RTX 5090): `fg-custom/README.md` for the design and the
selected kernels in `fg-custom/final/`, `fg-custom/RESULTS.md` for the ledger, `fg-custom/archive/` for
every earlier design the ledger cites.
