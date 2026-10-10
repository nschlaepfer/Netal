# FG launch 20 convolution reconstruction

This directory owns the first selected generic Frame Generation convolution:
evaluate-1 launch 20, module token 28, `k_conv_fp16_nhwc`.

The RTX oracle loads the exact 16,512-byte selected module through the CUDA
Driver API. It independently generates input, packed FP16 weights, replicated
bias, and output poison. It never opens an archived model buffer or expected
output. Every run proves input/weight immutability, guard integrity, and local
write coverage. The runner requires two processes for each of two poisons and
rejects output differences, including differences caused by the separately
poisoned 27,648-byte gap in the packed parameter buffer.

The current layout hypothesis is derived from the selected PTX and must remain
a hypothesis until post-run semantic verification passes:

- NHWC input: `1x96x160x48` FP16.
- NHWC output: `1x96x160x32` FP16.
- Active weights: `3x3x48x32` FP16 in native
  `mma.sync.m16n8k16.row.col` B-fragment lane order.
- A same-sized zero gap follows the active weights.
- The last 1,024 bytes contain four groups of eight biases, each repeated
  sixteen times.

CuMetal's strict PTX-to-Metal path currently rejects the selected
`mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16` opcode. That failure is a
compiler boundary, not permission to substitute MPS, MetalFX, or a different
model.

The native Metal kernel explicitly carries forward the exact SR HMMA
arithmetic hypothesis: each selected MMA call aligns sixteen exact FP16
products plus the prior FP16 accumulator at the largest exponent, keeps two
alignment bits, truncates discarded alignment bits, and rounds once to FP16.
For this launch, the live PTX chain orders spatial tiles as
`0,3,6,1,4,7,2,5,8` within each 16-channel stage. This transferred primitive
is accepted here only because the live token-63 weights and an independently
generated dense input match the exact selected RTX module byte for byte.
