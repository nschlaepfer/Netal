# Selected FG `k_upscale` family

This directory reconstructs the five live `evaluate_1` bilinear-upscale
launches selected by the pinned Frame Generation runtime. It is not a generic
replacement inferred from a framework graph: every launch identity, shape,
module hash, and operation order comes from the fresh-process NvAPI capture.

## Live family

| Launch | Module | Input HxW | Output HxW | Channels | Grid | Block |
| --- | --- | --- | --- | ---: | --- | --- |
| 32 | `d6962e...530fd` | 3x5 | 6x10 | 256 | 1x2x2 | 32x5x5 |
| 36 | `070c7f...a2b8` | 6x10 | 12x20 | 128 | 1x3x2 | 16x8x8 |
| 40 | `070c7f...a2b8` | 12x20 | 24x40 | 128 | 1x5x3 | 16x8x8 |
| 44 | `547a05...98fc` | 24x40 | 48x80 | 64 | 1x8x5 | 8x11x11 |
| 48 | `2b91bf...a74a` | 48x80 | 96x160 | 32 | 1x10x6 | 4x16x16 |

Every launch uses the same 40-byte little-endian packed ABI:
`input_ptr:u64`, `output_ptr:u64`, `input_height:u32`, `input_width:u32`,
`output_height:u32`, `output_width:u32`, `physical_output_height:u32`, and
`physical_output_width:u32`. The address-free receipt binds hashes of each
parameter record after replacing the two process-local pointer values with
zero bytes.

The four extracted SM89 PTX payloads are identical except for the fixed
channel-vector bound and the corresponding NHWC stride shifts. Each CUDA
thread evaluates eight FP16 channels. The Metal kernel expresses the same
align-corners coordinate transform and the selected driver's stable packed
FP16 lowering:

1. round the right horizontal product;
2. fuse the left product into the horizontal add;
3. repeat that sequence for the bottom row and round its `y` product; and
4. fuse the top `1 - y` product into the final add.

The selected PTX uses `div.approx.ftz.f32` for align-corners scale. A dedicated
RTX opcode probe found that seven of the ten live height/width extent pairs
produce the strict IEEE quotient and three produce its adjacent float32 value:
24->48 is -1 ULP, 40->80 is +1 ULP, and 80->160 is -1 ULP. Metal starts from
strict division and applies only these three selected-path corrections. This
is recovered opcode behavior, not an oracle output or captured tensor.

## Current numerical contract

The acceptance fixture covers positive and negative finite FP16 normals plus
both signed zeros, with magnitudes bounded below 4 and signs varying within
channels across space. Values include half-ULP neighbors around common values.
Subnormals, infinities, NaNs, and padded physical extents remain unsupported
until separate poison/provenance receipts accept them. This domain is stated
explicitly rather than silently relying on Metal fast-math.

`upscale.metal` is built with fast math and FP contraction disabled. CUDA and
Metal generate the tensor inputs independently. The verifier recomputes the
input generator, checks all output bytes, canaries, input immutability, full
write coverage, four unique processes per backend, and byte equality across
all poison/process/backend combinations.
