# Parts list

Where each proven SM120 primitive actually lives. Moving one between lanes is a port, not a copy —
the branches are deliberately isolated — so this records the authoritative implementation, its
proven domain, and what it would take to reuse it.

| primitive | authoritative implementation | proven over | notes for reuse |
|---|---|---|---|
| `mma.sync.m16n8k16` — fast | `dev`: `tools/pwin_pipeline/pwin_gemm3.metal`, `simdgroup_multiply_accumulate` with `split_k16` selecting where the fp32 accumulator rounds back to fp16 | four RTX SR receipts, shipping in `_hw7` | fp32 accumulator. FG's instruction is `f16.f16.f16.f16`; see `units/hmma-accumulate-variant.md` |
| `mma.sync.m16n8k16` — exact | `Mac-fg-research`: `fg_hmma_f16_bits_16` | all 17 generic + 25 custom FG launches | scalar, one output element per thread; a correctness oracle, ~70x off a tensor kernel |
| `ex2.approx.ftz.f32` | `Mac-fg-research`: `evidence/specs/fg-sm120-ex2-compact-v1` | **all 2,139,095,040 finite negative FP32 patterns**, 6.3 MB compact | **cross-validated on SR's own data: 29,851/29,851 inside the window.** Port it |
| `lg2.approx.ftz.f32` | nowhere — `dev` memoises 79,911 observed keys in a 4 MB open-addressed table | the four SR captures only | tractable: agrees with `fast::log2` to 9.5e-07 **absolute**; needs a mantissa-keyed correction, not an output-ULP one |
| `rcp.approx.ftz.f32` | `dev`: `cm_rcp_approx_ftz_f32` — `1.0f/x` plus a 2-bit ROM correction, 98 call sites in the output kernel | the SR domain in use; 0.26 ms of a 2.1 ms dispatch | needs the RNE base, not a plain `fdiv`: division drifts inside large kernels |
| Blackwell rounded-zero (`-0 -> +0`) | `Mac-fg-research`, per HMMA stage, lookup-free | FG temporal frames 2 and 11 | SR's CuMetal emulation had the same bug and fixed it separately |
| texture fetch / `tld4` gather | `dev`: `cm_texture_texel`, `cm_tex_base_2d_s32` | SR receipts; two hoists worth 0.7 ms | descriptor vectorisation was tried and lost — the backend already coalesces |

## Measured 2026-09-04

Both entries above were checked rather than assumed. `numerics/mufu-window.mm` runs Metal's
`fast::exp2`/`fast::log2` over SR's recorded RTX pairs and histograms the difference.

**EX2 — the port is validated before it is written.** FG's compact form encodes
`RTX_bits = fast_bits - code + 1` with a two-bit code, so RTX must lie within `{+1, 0, -1, -2}` ULP
of `fast::exp2`. All **29,851 of 29,851** of SR's independently recorded RTX ex2 outputs fall inside
that window, distributed across exactly the four deltas FG documented. Two features, two capture
sets, one instruction, complete agreement. FG's backend covers SR's entire observed domain, so the
port fixes SR's drift on unseen content with no new audit required.

**LG2 — tractable, and the earlier reading of it was wrong.** Against `fast::log2` the ULP deltas
sprawl over 2,151 distinct values, which looks fatal until you notice where:

| \|log2(x)\| | count | max ULP delta |
|---|---|---|
| >= 1 | 38,623 | **2** |
| < 1 | 13,742 | 11 |
| < 2^-3 | 10,804 | 74 |
| < 2^-6 | 16,742 | 1,768,632 |

The blow-up is entirely an artifact of ULP shrinking to nothing as `log2(x) -> 0`. In absolute terms
the two agree to a median of 3.2e-08 and a maximum of 9.5e-07 across the whole domain - about
1.15e-07 of full scale. So `lg2` does not need a different instruction model, it needs a correction
keyed on the **mantissa** (the scale-invariant part of `log2(2^e * m) = e + log2(m)`) rather than on
output ULP, with more than two bits near `m = 1`. That is the same decomposition FG used for EX2.

## The gap worth closing first


`lg2` is the only primitive with no exhaustive backend, and it is the reason for SR's documented
limit: *"the exact mode's lg2/ex2 tables cover the captured sequences only — unseen content drifts
to 67-80 dB after frame 0."* Its observed domain is narrow (79,911 keys, inputs in
`[0.00313, 14.51]`, exponents 118-130), and FG's EX2 work shows the shape of the answer: decompose
by exponent region, find the region where a unit table indexed by a fixed-point transform is exact,
and audit exhaustively against the hardware.

Closing it plus porting EX2 removes the drift entirely, which is worth more than any millisecond
currently on the table: an upscaler that is exact on four captures and approximate on real game
content is approximate.
