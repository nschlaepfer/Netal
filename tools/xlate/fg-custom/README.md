# FG custom block (launches 56-83): guarded exact pipelines

Bit-exact Metal replacements for the 25 custom convolution launches of DLSS Frame Generation's L56-L83
block, gated against the RTX 5090 receipt. Every launch's output is byte-identical to NVIDIA's own module
run (`Mac-fg-research:evidence/specs/fg-connected-custom-l56-l83-v1/verification.json`, 28/28 checkpoints).

| | block GPU time (M3 Max, 640x360) |
|---|---:|
| Codex's scalar oracles | 68 ms idle machine (74.8 in the FG frame) |
| these pipelines, proven bound | **~10.0 ms** |
| these pipelines, empirical bound | **~9.4 ms** |

`RESULTS.md` is the ledger: every design, accepted or rejected, with the numbers that decided it.

## Layout

- `final/` - the selected kernel per family and its harness, the shared headers, the compaction kernels,
  and `connected/` (the end-to-end probe). `final/build_and_verify.sh <Mac-fg-research checkout>` stages
  the kernels into that checkout, builds the block, and checks all 28 checkpoints against RTX.
- `archive/` - every earlier design, kept because the ledger cites them by file name.

## The design (one paragraph)

Each convolution runs as a guarded fast pass on `simdgroup_multiply_accumulate` (bit-for-bit a
sequential fp32 FMA chain), 8 pixels x 16 output channels per simdgroup, with the K=16 stage split into
two zero-seeded K=8 partials folded into the fp16 prior by two exact two_sums. A per-element certificate
bounds the fp32 chain error (3.5 ulp32 per K=8 partial: seven roundings), NVIDIA's per-term truncation
(n grid units of `2^(max_exp-25)`, `max_exp` floored at -21, bounded through `M = max(2^-21, bnd0, bnd1,
|prior|)` with fp16-subnormal operands proxied to 2^-14) and the fold's own rounding; an element whose
interval straddles an fp16 rounding boundary is flagged with the stage, the prior and the fast result at
its first two flags. Flags compact into a list; pass 1 recomputes one exact stage per flagged element
(the u32-bucket oracle) and settles ~90% of them; the rest replay exactly from their resume point in a
uniform pass 2 (lane-parallel where lists are short). Pooled families fuse the 2x2 pool into the fast
pass (4x2 pixel blocks, shuffle pool, a re-pool pass over the replayed elements); bottom families
materialise the bilinear upsample once with the connected block's fma formula; the three-plane bottom1
kernels run all three planes' MMAs but fold only each lane's owning plane. Weights and their bound
proxies are packed on the host into one half4 per lane per stage.

## Per family (final/)

| launches | kernel file | fast pass / pass 1 / pass 2 (+epilogue) |
|---|---|---|
| 57 convPre | `custom_conv_pre_pipeline_wq.metal` | `fg_custom_block0_conv_pre_{bounded,repair_pass1,listed,repool}_wq` |
| 58 conv0 | `custom_conv0_c8_pipeline_wq.metal` | `fg_custom_block0_conv0_c8_*_wq` |
| 59 conv1 | `custom_conv1_pipeline_wq.metal` | `fg_custom_block0_conv1_*_wq` |
| 60-67 conv2 x8 | `custom_conv2_bounded_v13.metal`, `custom_conv2_repair_v8.metal` (pass 1, writes its own indirect args), `custom_conv2_repair_v6.metal` (pass 2, 16 lanes/element) | `fg_custom_block0_conv2_bounded_v13`, `_repair_pass1_v8`, `_repair_pass2_v6` |
| 68 bottom0 | `custom_bot0_pipeline_wq.metal` + `connected_upsamples.metal` | `fg_connected_block0_bot0_upsample`, `fg_custom_block0_bot0_*_wq` |
| 69 bottom1 | `custom_block0_bot1_pipeline_wq.metal` | fused three-plane `fg_custom_block0_bot1_bounded_wq`, `_repair_pass1_wq`, `_listed_wq` (lanes) |
| 70 upsample x3 | `connected/custom_upsample_planes3.metal` | one dispatch for the three planes |
| 72 block1 conv0 | `custom_block1_conv0_pipeline_wq.metal` | K=8 tail specialised; `fg_custom_block1_conv0_*_wq` |
| 73 block1 conv1 | `custom_block1_conv1_pipeline_wq.metal` | `fg_custom_block1_conv1_*_wq` |
| 74-81 block1 conv2 x8 | `custom_block1_conv2_bounded_wq.metal`, `custom_block1_conv2_repair_ind.metal` | `fg_custom_block1_conv2_bounded_wq`, `_repair_pass1_ind`, `_repair_pass2_ind` |
| 82 block1 bottom0 | `custom_block1_bot0_pipeline_wq.metal` | `fg_custom_block1_bot0_*_wq` |
| 83 block1 bottom1 | `custom_block1_bot1_pipeline_wq.metal` | fused, skip-add epilogue |

Shared: `compact_ordered.metal` (block-ordered compaction), `compact_ordered_ind.metal` (also writes the
indirect arguments; Metal 3.2 for the device-scope fence), `compact_suspects.metal` (`fg_indirect_from_counter`),
`hmma_exact_fast.metalh` (oracle with clz), `hmma_exact_u32.metalh` (two unsigned buckets; exact on 16.8M
vectors), `hmma_exact_lanes_fast.metalh` (16 lanes per element; exact on 8.4M vectors incl. the -21 floor).
`make_pf.py` generates the pool-fused variant of a pooled family from its pipeline file.

Host-side weight tables: each `*_pipe_wq.mm` carries `build_wq_<family>()` and a verifier (bitwise weights,
proxies, every slot covered once); the probe (`connected/macos_connected_pipelines.mm`) builds all eleven.

## Two bounds

`FORMAL` (function constant 0, default on) is the proof above. Off, the bound is the earlier `4/4/2` form:
exact on every tensor measured (about 10^9 element-stages, including the RTX receipt) but not a proof for
all inputs; it costs 6-7% less. The probe takes it as its fifth argument.

## What is proven and what is measured

Proven: the fast pass accepts an element only if every value in its certified interval rounds to the same
fp16 as the fast value, under the documented HMMA rule (`../numerics/HMMA_F16_RULE.md`) and the measured
fact that Apple's MMA is a sequential fp32 FMA chain. Measured: 28/28 checkpoints against RTX on the
receipt's inputs, per-family byte-exactness against the scalar oracles on live weights, the helper tests
in `../numerics/`. Not covered: the FG runtime integration (these pipelines run in the connected-block
probe; the live FG runtime still dispatches the scalar oracles).
