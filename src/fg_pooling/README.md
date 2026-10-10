# FG selected pooling family

This directory reconstructs the five `k_pooling` launches selected by the
pinned Frame Generation runtime. The selected 2,032-byte module is module 29,
function 27, SHA-256
`a29a3d7041cbe7e44707e74dddf7aff61871a2397d1c097ef913bd753a149977`.

The family is NHWC 2x2 max pooling with stride two. Channels are processed as
packed FP16 pairs. The exact live contracts are:

| launch | input | output | grid | block |
|---:|---:|---:|---:|---:|
| 21 | 160x96x32 | 80x48x32 | 20x12x1 | 4x4x16 |
| 23 | 80x48x64 | 40x24x64 | 10x6x1 | 4x4x16 |
| 25 | 40x24x128 | 20x12x128 | 5x3x1 | 4x4x16 |
| 27 | 20x12x128 | 10x6x128 | 3x2x1 | 4x4x16 |
| 29 | 10x6x256 | 5x3x256 | 2x1x1 | 4x4x16 |

`rtx_oracle.cpp` loads the exact selected fatbin through the CUDA Driver API.
`macos_probe.mm` and `pooling.metal` generate and execute the same finite FP16
fixture independently, without accepting tensor artifacts from the RTX run.
The fixture includes normal values, subnormals, positive zero, negative zero,
and ties; NaN and infinity behavior is intentionally not claimed.
