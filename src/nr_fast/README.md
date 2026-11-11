# Netal Native Metal DLSS 5 Neural Rendering Runtime (`src/nr_fast`)

`src/nr_fast` implements Netal's standalone, pure-source Native Metal compute runtime (`namespace netal_nr_fast`) for DLSS 5 Neural Rendering U-Net + ViT hybrid transformers on Apple Silicon (`Apple9` / Metal 3.1+).

It executes the complete end-to-end pipeline—16-channel PCG/Catmull-Rom feature preprocessing, 31-block (`Small31`) or 71-block (`HT71`) hierarchical Swin-window + global-ViT transformer evaluation, temporal confidence blending with persistent history ping-pong textures, and optional separable Gaussian detail-split upsampling—inside a **single non-blocking `MTLComputeCommandEncoder`** with zero intermediate CPU synchronization and zero explicit tensor transpose dispatches.

---

## Files

| Path | Role |
| --- | --- |
| `src/nr_fast/nr_fast_runtime.h` | Public C++/ObjC++ API (`netal_nr_fast::Context`, `FrameInputs`, `FrameTicket`, `Report`, model validation & E4M3 helpers; pure C++20 compatible when included from non-ObjC++ translation units) |
| `src/nr_fast/nr_fast_runtime.mm` | SafeTensors / `.dlssmodel` loader, SHA-256 manifest verifier, 256-entry E4M3 LUT unpacker, load-time weight layout & 12-bit bias swizzler, untracked private scratch arena, residency-set manager, and single-encoder graph scheduler |
| `src/nr_fast/nr_fast_kernels.metal` | Native Metal 3.1 compute kernels (`-std=metal3.1 -fmetal-math-mode=safe` with `#pragma clang fp contract(off)` / `reassociate(off)`) |
| `src/nr_fast/nr_fast_cli.mm` | Standalone CLI (`netal-nr-fast-cli`: `render-image`, `eval-tensor`, `bench`, `validate-model`) |
| `scripts/build-netal-nr-fast.py` | Atomic out-of-tree builder producing `nr_fast_kernels.metallib`, `libnetal_nr_fast.a`, `netal-nr-fast-cli`, and `build-receipt.json` |
| `scripts/test-netal-nr-fast.py` | Model-free CI test suite covering atomic builds, safetensors fault injection, bit-exact `F8_E4M3` round-trip, synthetic `Small31`/`HT71` GPU contracts, and upsampled detail-split rendering |
| `scripts/compare-nr-native-vs-mlx.py` | Numerical parity and GPU latency benchmark comparing `netal-nr-fast-cli` against `mlxdlss` |

---

## Architecture & Kernel Design

### 1. Weight Loading, Validation & Dynamic Topology Inference
- **Supported Sources**:
  1. `.dlssmodel` directories (`manifest.json` + SHA-256 & byte-length verification + `weights.safetensors`).
  2. Standalone `.safetensors` files with `F16`, `F32`, or lossless 8-bit `F8_E4M3` / `U8` (`mlxdlss-native-e4m3-v1`) weights. E4M3 weight bytes decode to IEEE-754 binary16 via a verified 256-entry table (`uint16_t h = (b & 0x7f) << 7; half c = as_type<half>(h) * 256.0h; (b & 0x80) ? -c : c`), producing **0.0 bit-identical outputs** vs FP16 `.dlssmodel` bundles.
  3. `Context::PrepareSynthetic(device, TopologyKind::{Small31, HT71}, ...)` for deterministic, model-free CI and benchmarking without external checkpoints.
- **Strict SafeTensors Validation**: Rejects truncated headers, malformed JSON, negative or overflowing tensor dimensions, out-of-bounds `data_offsets`, overlapping byte intervals, non-finite `F32` scalars, and (optionally) symlinks.
- **Dynamic Mirrored Topology Inference**: Infers 5-stage encoder/decoder depths plus global ViT block count directly from `block0..blockN` tensor shapes (e.g., `[4, 4, 6, 8, 8]` + `8` global = 71-block `ht`; `[2, 2, 2, 2, 2]` + `8` global = 31-block `small`) and reads `block<post>.layer0.blend_scale` (defaulting to `0.73974609375f`).
- **Load-Time GPU Weight Preparation**:
  - Applies the 12-bit `simdgroup_matrix` fragment swizzle to `attn_bias` **only** when `headCount == 1` or `headCount == 16` (leaving `headCount` in `{2, 4, 8}` unswizzled).
  - Pre-transposes 5D `ffn_expand_weight` `[G, 4, G, 32, 32]` (axes `2, 3, 0, 1, 4`) into contiguous `[C, 4*C]` FP16, keeps `ffn_branch_projection_weight` as contiguous `[G, 128, 32]`, pre-scales global `attn_scale` by `sqrt(32)`, and packs `out_gain` `[16, 4]` + `out_conv_weight` `[16, 4]` into `[32, 8]` `headWeight`.
  - Uploads packed weights into `MTLResourceStorageModePrivate | MTLResourceHazardTrackingModeUntracked` memory and registers persistent weight, scratch, and history allocations in an `MTLResidencySet` on macOS 15+.

### 2. Native Metal 3.1 Compute Kernels
- **Exact E4M3 Activation Rounding (`nr_fast_e4m3`)**: Implements the branch-free magic-number FP16-to-E4M3 rounder (`(as_type<half>(bits) + magic) - magic`) protected by `-fmetal-math-mode=safe` and `#pragma clang fp contract(off)` / `reassociate(off)`.
- **Cooperative SRAM-Staged 1-Head 32ch Swin Window Kernels (`nr_fast_fused_window_1h32*`)**: 8 simdgroups (`256` threads) per `8x8` window keep activations in `simdgroup_matrix<half, 8, 8>` registers across FFN, cosine residual, QKV projection, XOR norm, bit-affine `(bits << 5) ^ 0x8000` window attention, output projection, and fused `2x2` downsample or `2x2` upsample/head epilogues while cooperatively staging per-phase weights into idle threadgroup SRAM (`12 KB` total).
- **Multi-Head Window Attention & Branched FFN (`nr_fast_window_attn_block_64`, `nr_fast_branched_ffn_64/128`, `nr_fast_window_attention_mh`)**: Dedicated register-tiled 64ch/128ch blocks plus general `{2, 4, 8, 16}`-head `8x8` window attention.
- **Adaptive Tiled `simdgroup_matrix` GEMM & Zero-Transpose Grouped GEMM (`nr_fast_gemm_f16`, `nr_fast_gemm_f16_bm64`, `nr_fast_grouped_gemm_f16`)**: Automatically selects `BM=64, BN=64, BK=32` for high-row-count stages (`M >= 1024`, halving B-matrix bandwidth) and `BM=32, BN=64, BK=32` for compact global token counts (`M < 1024`), with fused epilogues (`none`, `e4m3`, `quad_gate`, `quad_gate_e4m3`, `cosine_residual`, `cosine_residual_e4m3`). `nr_fast_grouped_gemm_f16` strides directly over `[M, G, K] -> [M, G, N]` with zero reshape/transpose copies.
- **On-Chip Global ViT Attention (`nr_fast_global_attn_resident` + streamed `denom`/`values` fallback)**: Single-pass resident global attention for `tokens <= 1024` (up to `2048x2048` network resolution) and two-pass streamed attention for larger grids, enforcing `[-3, +3]` symmetric logit clamp, bit-affine weights, sequential FP16 denominator accumulation, and zero `N x N` attention matrix allocation.
- **GPU Preprocessor & Detail-Split Postprocessor**:
  - `nr_fast_pre_features`: Exact PCG RXS-M-XS (`0x108ef2d9`) Box-Muller noise (`ch 0..2`), constant `1` (`ch 3`), centered current RGB (`ch 4..6`), closest-depth dilated 5-tap Catmull-Rom reprojected history RGB (`ch 7..9`), and profile/tone/structure controls (`ch 10..15`).
  - `nr_fast_post_temporal` + `nr_fast_detail_blur_h` / `nr_fast_detail_blur_v_compose`: Evaluates `predicted = clamp(cur + 0.25 * head.rgb, 0, 1)`, blends temporal history via `clamp(sigmoid(head.w) * blend_scale, 0, 1)`, updates double-buffered `history_tex`, and applies arbitrary-resolution bilinear upsampling + separable Gaussian low/high-frequency detail recomposition (`colour_strength * lowPass + detail_strength * highPass`) with optional linear <-> sRGB transfer.

---

## Build, Test, and CLI Usage

```bash
# 1. Build metallib, static library, CLI, and SHA-256 build receipt into a fresh directory:
python3 scripts/build-netal-nr-fast.py --output /Users/example/scratch/nr_fast_build

# 2. Run the model-free CI suite (build check, safetensors fault injection, FP8 round-trip, Small31/HT71 GPU contracts):
python3 scripts/test-netal-nr-fast.py

# 3. Render an image with real weights (either .dlssmodel or native FP8 .safetensors):
/Users/example/scratch/nr_fast_build/netal-nr-fast-cli render-image \
  --model /Users/example/models/dlssnr-weights-native-fp8.safetensors \
  --input /Users/example/in.png \
  --output /Users/example/out.png \
  --profile cinematic \
  --detail-strength 2.0 \
  --colour-strength 0.9

# 4. Run numerical parity + latency comparison against MLX-DLSS:
python3 scripts/compare-nr-native-vs-mlx.py \
  --mlx-bin /Users/example/bin/mlxdlss \
  --model /Users/example/models/NeuralRendering.dlssmodel \
  --fp8-model /Users/example/models/dlssnr-weights-native-fp8.safetensors \
  --bench-resolutions
```

---

## C++ / Objective-C++ Integration

```cpp
#include "nr_fast_runtime.h"

std::string err;
auto ctx = netal_nr_fast::Context::LoadFromPath(
    mtl_device, "/Users/example/models/NeuralRendering.dlssmodel", {}, &err);

netal_nr_fast::FrameInputs frame{};
frame.color_texture = input_rgba_tex;
frame.motion_texture = motion_rg_tex;   // optional
frame.depth_texture = depth_r_tex;      // optional
frame.output_texture = output_rgba_tex;
frame.params.output_width = 1920;
frame.params.output_height = 1080;
frame.params.network_width = 1280;
frame.params.network_height = 704;
frame.params.geometry_mode = netal_nr_fast::GeometryMode::ScaledNetwork;
frame.params.profile = netal_nr_fast::ControlProfile::Standard;
frame.params.detail_strength = 1.35f;
frame.params.colour_strength = 1.0f;
frame.params.frame_index = frame_idx;
frame.params.reset = (frame_idx == 0);

netal_nr_fast::FrameTicket ticket;
netal_nr_fast::Report report;
// Encodes the entire frame in a single MTLComputeCommandEncoder without committing `cmd_buf`:
ctx->encode(frame, cmd_buf, ticket, report, &err);
```

---

## Measured Verification & Benchmarks (Apple M4 Max, 32-core GPU)

### Numerical Parity vs `MLX-DLSS` (`mlxdlss --execution metal-fused --precision float16`, 71-block `HT`)

| Verification Case | MAE | RMSE | PSNR | Max Abs Diff | MLX GPU Time | Netal GPU p50 | Speedup |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Raw Tensor `[1, 256, 384, 16] -> [1, 256, 384, 4]` | `0.000000` | `0.000000` | `> 99 dB` | `0.000000` (bit-exact) | `49.05 ms` | `10.16 ms` | **4.83x** |
| Native FP8 `.safetensors` vs `.dlssmodel` (`[1, 256, 384, 16]`) | `0.000000` | `0.000000` | `> 99 dB` | `0.000000` (bit-exact) | — | `10.10 ms` | **1.00x (exact)** |
| `render-image` (`384x256`, `standard`) | `0.001888` | `0.002900` | `50.75 dB` | `0.019608` | `45.62 ms` | `11.39 ms` | **4.00x** |
| `render-image` (`384x256`, `cinematic`, `detail=2.5, colour=0.8, intensity=0.9`) | `0.004882` | `0.006240` | `44.10 dB` | `0.027451` | `51.82 ms` | `11.34 ms` | **4.57x** |

### End-to-End Temporal Frame Latencies on Apple M4 Max (32-core GPU)

| Variant | Network Size | Output Size | GPU Min | GPU p50 | CPU Encode p50 | Dispatches (1 Encoder) |
| --- | --- | --- | --- | --- | --- | --- |
| `Small31` (31 blocks, synthetic) | `960x544` | `1920x1080` | `16.31 ms` | `16.97 ms` (`58.9 FPS`) | `0.054 ms` | `151` |
| `Small31` (31 blocks, synthetic) | `1280x704` | `1920x1080` | `24.35 ms` | `25.07 ms` (`39.9 FPS`) | `0.055 ms` | `151` |
| `Small31` (31 blocks, synthetic) | `1920x1088` | `1920x1088` | `54.68 ms` | `56.54 ms` (`17.7 FPS`) | `0.063 ms` | `150` |
| `HT71` (71 blocks, real FP8 weights) | `960x544` | `1920x1080` | `33.87 ms` | `38.80 ms` (`25.8 FPS`) | `0.121 ms` | `351` |
| `HT71` (71 blocks, real FP8 weights) | `1280x704` | `1920x1080` | `68.79 ms` | `95.45 ms` (`10.5 FPS`) | `0.166 ms` | `351` |
| `HT71` (71 blocks, real FP8 weights) | `1920x1088` | `1920x1088` | `179.25 ms` | `198.67 ms` (`5.0 FPS`) | `0.186 ms` | `350` |

---

## License & Provenance

`src/nr_fast/*` is licensed under `MIT AND Apache-2.0`.

Portions of the `mlxdlss.neural-rendering-transformer.v1` tensor schema, dynamic topology inference rules, 12-bit `simdgroup_matrix` attention-bias swizzle, bit-affine `(bits << 5) ^ 0x8000` window/global softmax, vendor XOR cosine normalization, magic-number E4M3 activation rounding, PCG `0x108ef2d9` Gaussian noise math, 5-tap Catmull-Rom reprojection, and Gaussian detail-split composition are clean-room native Metal / C++ adaptations of:

- **WaveCut / MLX-DLSS Contributors** (`https://github.com/iamwavecut/MLX-DLSS`), licensed under the **Apache License, Version 2.0**.
