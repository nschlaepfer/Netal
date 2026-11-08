# Synthetic FP8 matrix lab

This optional experiment compares direct FP8-input TensorOps with FP16 on
physical A20 Pro devices running iOS 27. It contains procedural inputs and
original public-API code, with no model weights or vendor implementation.
The existing iOS 26 and macOS SR builds do not depend on it.

Build with an SDK/compiler supporting MSL 4.1:

```sh
python3 scripts/test-netal-ios-fp8-lab.py
python3 scripts/build-netal-ios-fp8-lab.py --output-dir build/fp8-lab
```

Choose a new output directory each time. The builder emits a static archive,
public header, metallib and source-hashed build receipt. It does not sign,
install or contact a device. SDK 26 cannot build this experiment; a compiler
failure does not become an unavailable-hardware success.

Link the archive with Foundation and Metal, bundle the metallib, and call
`NetalRunFP8MatmulLabWithStop` on a background queue. Use its predicate to stop
when cancelled or inactive. Three samples and eight repetitions are a useful
first diagnostic. Bounds are 3–15 samples and 1–128 repetitions. A predicate or
critical thermal condition stops between completed commands; it cannot cancel
in-flight work. A GPU timeout retains resources until completion and requires
closing the app before another run.

The five K/N pairs (16/32, 32/48, 64/256, 640/160, 160/768) represent actual
SR operator dimensions, including K padding and an N tail. Every fixture uses
1024 rows, exact-grid and bounded-noise inputs, and identical mathematical
inputs/bias for FP16, E4M3 and E5M2. FP8 accumulates into FP32; the output is
`accumulator * scaleA * scaleB + bias`, with zero initial accumulation. Scales
are known scalar fixture scales, rather than learned or calibrated model scales.

GPU pack/unpack is checked against an independent host format reference,
including finite saturation, ties and subnormals. For each matrix, eight named
rows and all logical columns are checked against a CPU double-precision
oracle. All output elements must be finite; prefix/suffix guards and output
padding must remain intact. A 0.002 absolute bound applies to matmul error
relative to quantized operands. Quantization error against the original
inputs is recorded separately and has no image-quality acceptance threshold.
The sampled oracle is not an exhaustive proof for every matrix element.

Timing uses driver GPU start/end timestamps, with warmup excluded and rotating
mode order. Missing/invalid timestamps fail. Receipts separate prepacked GEMM
from packing both operands plus GEMM. The latter also includes diagnostic FP32
unpack writes, and both include buffer barriers. Source generation, validation,
allocation and shader compilation are outside these GPU measurements. Storage
and allocation figures distinguish packed operands from diagnostic allocations.

A successful receipt proves execution of the tested FP8-input TensorOps API.
It does not prove which native hardware instruction the driver selected,
accelerator throughput, full SR inference, DLSS quality, gameplay FPS, or a
2–3 ms frame budget. No FP8-to-half fallback is present in these matmul kernels.
A19 Pro and other chips remain excluded from this experiment.

The API and storage formats are specified by Apple's
[Metal Shading Language specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
and the installed SDK's public MetalPerformancePrimitives headers. Build and
host-test success are distinct from a physical-device runtime result.
The independent host format reference follows
[OCP OFP8 revision 1.0](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-06-20-pdf).
