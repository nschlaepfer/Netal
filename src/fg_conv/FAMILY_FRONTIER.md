# Generic FG convolution family frontier

The selected evaluate-1 encoder/decoder contains 17 convolution launches over
16 distinct PTX modules. Live module, ABI, and post-evaluate-0 weight evidence
reduce them to two structural families:

- Four padded 3x3 convolutions use a 48-byte, nine-parameter ABI.
- Thirteen 1x1 convolutions use a 40-byte, seven-parameter ABI.
- The first sixteen launches apply FP16 leaky-ReLU with slope bits `0x211f`.
- The final 32-to-16 launch is linear.

Every captured weight buffer has the same dependency-closed shape: `N` active
FP16 weights, `N` positive-zero FP16 gap values, then output biases in groups
of eight with each group repeated sixteen times. All active weight words are
nonzero in this capture. The PTX activation census came from fresh rig job
`D:\CodexLab\fg-conv-family-census-20260904T102127Z`; the job is retained.

| Trace launch | Module | Weight token | Spatial | In -> out | ABI scalars | Activation | Parity |
|---:|---:|---:|---:|---:|---|---|---|
| 20 | 28 | 63 | 160x96, 3x3 | 48 -> 32 | 160,96,162,98,160,96 | leaky | exact dense + sparse |
| 22 | 30 | 65 | 80x48, 1x1 | 32 -> 64 | 80,48,80,48 | leaky | exact dense |
| 24 | 31 | 67 | 40x24, 3x3 | 64 -> 128 | 40,24,42,26,40,24 | leaky | exact dense |
| 26 | 32 | 69 | 20x12, 3x3 | 128 -> 128 | 20,12,22,14,20,12 | leaky | exact dense |
| 28 | 33 | 71 | 10x6, 3x3 | 128 -> 256 | 10,6,12,8,10,6 | leaky | exact dense, schedule-specialized |
| 30 | 34 | 73 | 5x3, 1x1 | 256 -> 512 | 5,3,5,3 | leaky | exact dense |
| 31 | 35 | 75 | 5x3, 1x1 | 512 -> 256 | 5,3,5,3 | leaky | exact dense |
| 34 | 38 | 77 | 10x6, 1x1 | 256 -> 256 | 10,6,10,6 | leaky | exact dense |
| 35 | 39 | 79 | 10x6, 1x1 | 256 -> 128 | 10,6,10,6 | leaky | exact dense |
| 38 | 42 | 81 | 20x12, 1x1 | 128 -> 128 | 20,12,20,12 | leaky | exact dense |
| 39 | 42 | 83 | 20x12, 1x1 | 128 -> 128 | 20,12,20,12 | leaky | exact dense |
| 42 | 43 | 85 | 40x24, 1x1 | 128 -> 128 | 40,24,40,24 | leaky | exact dense |
| 43 | 44 | 87 | 40x24, 1x1 | 128 -> 64 | 40,24,40,24 | leaky | exact dense |
| 46 | 47 | 89 | 80x48, 1x1 | 64 -> 64 | 80,48,80,48 | leaky | exact dense |
| 47 | 48 | 91 | 80x48, 1x1 | 64 -> 32 | 80,48,80,48 | leaky | exact dense |
| 50 | 51 | 93 | 160x96, 1x1 | 32 -> 32 | 160,96,160,96 | leaky | exact dense |
| 51 | 52 | 95 | 160x96, 1x1 | 32 -> 16 | 160,96,160,96 | linear | exact dense |

## Proven reusable primitive

All seventeen selected launches use distinct weight sets across sixteen
modules, varying dimensions, channels, dispatch geometries, and activation
behavior. Every module is byte-exact with
the same scalar implementation of one Blackwell
`mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16` output: align sixteen
exact FP16 products plus the prior FP16 accumulator, retain two alignment
bits, truncate discarded bits, then round once to FP16 RNE.

This independently proves the complete selected generic convolution family.
It does not prove any connected multi-kernel group.

## Fastest family closure

1. Connect the convolution family to the accepted pooling, elementwise, and
   upscale units for group-level parity.
2. Preserve launch 28's distinct accumulator schedule during graph assembly;
   the common schedule produced 12,352 differing bytes and was rejected.

Evidence anchors:

- live trace SHA-256: `72c0f352b8c70160333e81aee394b7fec6c638e663f85689d62cd27885b134d4`
- post-evaluate-0 manifest SHA-256: `2a1179d4a25be030f772cd90ca7d017ed6eac9291917f7d43ea9b0cd5ac6a294`
- launch-20 final receipt root: `66c686ed6210f9fa7b5da98823c052885c315d4234f8bf3db295587ca118b657`
- launch-51 final receipt root: `9eeb57ae29691cbff3307eee616be6b875dd675cee10167227bd6028cb6c5d23`
- complete 1x1-family receipt root: `100f98d331cf879e734d5303e4f1f0f4693da0f96d19e47f8793c93c88d90d71`
- complete 3x3-family receipt root: `972a6888b346a87423949b6cfa33c6dc91e3f93c303ebfc9e420a355c14550b3`
- all-17 convolution aggregate root: `b01e03d08e309049ef89ed489b4051213a455b952182068492fe9ec5443a87fd`
