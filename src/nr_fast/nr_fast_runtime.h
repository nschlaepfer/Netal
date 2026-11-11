// SPDX-License-Identifier: MIT AND Apache-2.0
// Copyright (c) Netal Contributors.
//
// Portions of the neural-rendering transformer topology, 12-bit attention-bias
// fragment swizzle, bit-affine softmax, vendor cosine normalization, E4M3
// rounding math, and deterministic PCG feature contract are adapted from:
//   WaveCut / MLX-DLSS Contributors (https://github.com/iamwavecut/MLX-DLSS)
//   Licensed under the Apache License, Version 2.0.

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#ifdef __OBJC__
#import <Metal/Metal.h>
using NRMetalDevice = id<MTLDevice>;
using NRMetalCommandQueue = id<MTLCommandQueue>;
using NRMetalCommandBuffer = id<MTLCommandBuffer>;
using NRMetalTexture = id<MTLTexture>;
using NRMetalBuffer = id<MTLBuffer>;
#else
using NRMetalDevice = void*;
using NRMetalCommandQueue = void*;
using NRMetalCommandBuffer = void*;
using NRMetalTexture = void*;
using NRMetalBuffer = void*;
#endif

namespace netal_nr_fast {

enum class TopologyKind : uint32_t {
  Small31 = 0,  // depths [2, 2, 2, 2, 2], 8 global blocks -> 31 blocks
  HT71 = 1,     // depths [4, 4, 6, 8, 8], 8 global blocks -> 71 blocks
  Custom = 2,
};

enum class ControlProfile : uint32_t {
  Standard = 0,   // style 0, tone 1, structure 1
  Natural = 1,    // style 1, tone 1, structure 1
  Cinematic = 2,  // style 2, tone 1, structure 1
  Neutral = 3,    // style 0, tone 0, structure 0
};

enum class GeometryMode : uint32_t {
  // Reflect-then-clamp pad output extent to multiple-of-64 (>= 320), matching
  // MLX-DLSS vendor-aligned geometry when network dimensions are unspecified
  // or >= output dimensions without downscaling.
  VendorAligned = 0,
  // Downsample/resample full-resolution input to explicit internal network
  // extent (any multiple of 32 >= 128, including multiples of 64) and upsample
  // neural detail delta back to full output resolution.
  ScaledNetwork = 1,
};

struct TopologySpec {
  TopologyKind kind = TopologyKind::HT71;
  int depths[5] = {4, 4, 6, 8, 8};
  int global_block_count = 8;
  int total_blocks = 71;
  float blend_scale = 0.73974609375f;
  std::string name = "ht";
  std::string spec_string = "4,4,6,8,8";
};

struct LoadOptions {
  // Optional path to compiled nr_fast_kernels.metallib. If empty, searches
  // adjacent to the executable or falls back to runtime Metal source compile
  // only when explicitly provided.
  std::string metallib_path;
  bool reject_symlinks = false;
  bool verify_manifest_sha256 = true;
  bool use_residency_set = true;
};

struct SyntheticOptions {
  std::string metallib_path;
  uint64_t seed = 0x4E52464153543031ULL;
  float weight_scale = 0.08f;
  bool use_residency_set = true;
};

struct FrameParams {
  // Output resolution (defaults tooutput_texture dimensions when 0).
  uint32_t output_width = 0;
  uint32_t output_height = 0;
  // Internal network resolution. When 0, uses VendorAligned extent
  // (align64(max(320, output_dim))). Otherwise any multiple of 32 >= 128
  // (including all multiples of 64 >= 128).
  uint32_t network_width = 0;
  uint32_t network_height = 0;
  GeometryMode geometry_mode = GeometryMode::VendorAligned;

  ControlProfile profile = ControlProfile::Standard;
  // Optional explicit overrides (< 0 uses profile default).
  int32_t style_index_override = -1;
  float local_tone_override = -1.0f;
  float local_structure_override = -1.0f;

  float detail_strength = 1.0f;
  float colour_strength = 1.0f;
  float detail_radius = 4.0f;
  float intensity = 1.0f;

  // Multiplier converting raw motion_texture.xy samples into normalized UV
  // displacement (history_uv = current_uv + mv * motion_scale). Defaults to
  // (1, 1) for normalized UV motion; set (1.0f / mw, 1.0f / mh) for pixel-space
  // optical flow or MetalFX scaled motion vectors.
  float motion_scale_x = 1.0f;
  float motion_scale_y = 1.0f;

  uint32_t frame_index = 0;
  bool reset = true;
  bool depth_inverted = false;
  bool use_closest_depth = false;
  // When true, converts linear scene RGB input -> sRGB before NR and converts
  // composed sRGB output -> linear RGB when writing output_texture.
  // When false (default), treats input/output textures as sRGB [0, 1].
  bool srgb_transfer = false;
};

struct FrameInputs {
  NRMetalTexture color_texture = nullptr;    // Required: RGB/RGBA read texture
  NRMetalTexture motion_texture = nullptr;   // Optional: RG motion texture
  NRMetalTexture depth_texture = nullptr;    // Optional: R depth texture
  NRMetalTexture output_texture = nullptr;   // Required unless export_delta_buffer/texture != nullptr
  // Optional destination buffer (>= hist_w * hist_h * sizeof(float4)) receiving
  // post-temporal RGB delta (`nr_rgb - cur_rgb`) for async reuse via encode_compose_delta().
  NRMetalBuffer export_delta_buffer = nullptr;
  // Optional destination RGBA16Float texture (hist_w x hist_h, ShaderWrite | ShaderRead)
  // receiving pre-baked Gaussian colour/detail delta (`intensity * (colour*lowPass + detail*highPass)`)
  // at network resolution for single-pass (<0.4 ms) hardware-bilinear display composition.
  NRMetalTexture export_baked_delta_texture = nullptr;
  FrameParams params;
};

struct FrameTicket {
  struct State {
    std::atomic<bool> completed{false};
    std::atomic<bool> succeeded{false};
    double gpu_start_s = 0.0;
    double gpu_end_s = 0.0;
    double gpu_ms = 0.0;
    std::string error_message;
  };
  std::shared_ptr<State> state = std::make_shared<State>();

  bool is_completed() const { return state && state->completed.load(std::memory_order_acquire); }
  bool succeeded() const { return state && state->succeeded.load(std::memory_order_acquire); }
  double gpu_ms() const { return state ? state->gpu_ms : 0.0; }
};

struct Report {
  uint32_t output_width = 0;
  uint32_t output_height = 0;
  uint32_t network_width = 0;
  uint32_t network_height = 0;
  uint32_t block_count = 0;
  uint32_t dispatch_count = 0;
  double cpu_encode_ms = 0.0;
  double gpu_ms = 0.0;
  float blend_scale = 0.0f;
  bool temporal_active = false;
  bool synthetic_weights = false;
  std::string topology;
  std::string weight_format;
};

struct SafeTensorSummary {
  size_t tensor_count = 0;
  size_t f16_count = 0;
  size_t f32_count = 0;
  size_t fp8_e4m3_count = 0;
  size_t total_weight_bytes_fp16 = 0;
  TopologySpec topology;
  std::string source_format;
};

// Validates a standalone .safetensors or .dlssmodel path on CPU without GPU
// execution; exposed for unit and fault-injection tests.
bool ValidateModelFile(
    const std::string& path,
    SafeTensorSummary* out_summary,
    std::string* out_error,
    bool reject_symlinks = false,
    bool verify_manifest_sha256 = true);

// Decodes a single mlxdlss-native-e4m3-v1 byte to IEEE-754 binary16 bits.
uint16_t DecodeNativeE4M3ToFp16Bits(uint8_t byte_val);

// Encodes an IEEE-754 binary16 value (given as bits) to an E4M3 byte (for testing).
uint8_t EncodeFp16BitsToNativeE4M3(uint16_t fp16_bits);

class Context {
 public:
  ~Context();
  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;

  // Load weights from either a .dlssmodel directory (manifest.json + SHA-256 +
  // weights.safetensors) or a standalone .safetensors file (F16, F32, or
  // mlxdlss-native-e4m3-v1 F8_E4M3 / U8).
  static std::unique_ptr<Context> LoadFromPath(
      NRMetalDevice device,
      const std::string& model_path,
      const LoadOptions& options,
      std::string* out_error);

  // Build deterministic, bounded FP16 weights in memory for CI tests and
  // weight-free M4 Max benchmarks using the exact same Metal kernels & graph.
  static std::unique_ptr<Context> PrepareSynthetic(
      NRMetalDevice device,
      TopologyKind kind,
      const SyntheticOptions& options,
      std::string* out_error);

  const TopologySpec& topology() const;
  bool is_synthetic() const;
  const std::string& weight_format() const;
  size_t weight_buffer_bytes() const;
  size_t scratch_buffer_bytes() const;
  NRMetalTexture history_texture() const;

  void reset_history();

  // Non-blocking single-encoder GPU graph submission. Encodes feature prep ->
  // 31/71 transformer blocks -> temporal blend + history update -> optional
  // Gaussian detail-split composition into `cmd` WITHOUT committing or waiting.
  bool encode(
      const FrameInputs& inputs,
      NRMetalCommandBuffer cmd,
      FrameTicket& out_ticket,
      Report& out_report,
      std::string* out_error);

  // Synchronous wrapper: creates command buffer on `queue` (or internal queue
  // when `queue == nullptr`), encodes, commits, waits, and records exact
  // GPUEndTime - GPUStartTime in `out_report.gpu_ms`.
  bool run_sync(
      const FrameInputs& inputs,
      NRMetalCommandQueue queue,
      Report& out_report,
      std::string* out_error);

  // Lightweight (~0.2 ms) cached-delta upsampling and detail-split composition
  // onto a live display frame. Uses dedicated compose scratch buffers separate
  // from `scratch_buf`, so `encode()` on a background worker queue and
  // `encode_compose_delta()` on a display queue never contend on GPU memory.
  bool encode_compose_delta(
      NRMetalTexture color_texture,
      NRMetalBuffer delta_net_buffer,
      uint32_t delta_width,
      uint32_t delta_height,
      const FrameParams& params,
      NRMetalTexture output_texture,
      NRMetalCommandBuffer cmd,
      std::string* out_error);

  // Single-dispatch hardware-bilinear composition of a worker-baked RGBA16Float
  // delta texture onto a ShaderWrite output texture (zero blur scratch buffers).
  bool encode_compose_baked_delta(
      NRMetalTexture color_texture,
      NRMetalTexture baked_delta_texture,
      const FrameParams& params,
      NRMetalTexture output_texture,
      NRMetalCommandBuffer cmd,
      std::string* out_error);

  // Raw transformer core evaluation: [1, H, W, 16] -> [1, H, W, 4] on Metal
  // buffers (float32 or float16) inside a single compute encoder, for exact
  // apples-to-apples parity verification against MLX-DLSS.
  bool encode_tensor(
      NRMetalBuffer input_nhwc16,
      bool input_is_fp16,
      uint32_t height,
      uint32_t width,
      NRMetalBuffer output_nhwc4,
      bool output_is_fp16,
      NRMetalCommandBuffer cmd,
      FrameTicket& out_ticket,
      Report& out_report,
      std::string* out_error);

  bool eval_tensor_sync(
      const void* input_nhwc16,
      bool input_is_fp16,
      uint32_t height,
      uint32_t width,
      void* output_nhwc4,
      bool output_is_fp16,
      NRMetalCommandQueue queue,
      Report& out_report,
      std::string* out_error);

 private:
  struct Impl;
  explicit Context(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace netal_nr_fast
