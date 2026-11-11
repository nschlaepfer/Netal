#pragma once
// Self-contained GPU multi-scale optical-flow and scene-cut synthesizer for
// spatial MetalFX scalers and color-only CAMetalLayer games where motionTexture == nil.
// Zero CPU readback on the hot path: encodes directly onto the caller's MTLCommandBuffer.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>

namespace netal_nr_adapter {

struct FlowConfig {
  float scene_cut_threshold = 0.18f;
  uint32_t max_quarter_search_radius = 24; // +/-24 quarter-res px = +/-96 full-res px
};

struct FlowTelemetry {
  uint64_t frame_index = 0;
  float mean_luma_diff = 0.0f;
  bool scene_cut = false;
  bool valid = false;
};

struct FlowOutput {
  // Full-resolution MTLPixelFormatRG16Float flow texture:
  // PixelsFullRes, CurrentToPrevious, y-down (history_uv = current_uv + flow_px / float2(W, H)).
  id<MTLTexture> flow_texture = nil;
  id<MTLBuffer> scene_cut_flag_buffer = nil;
  NSUInteger scene_cut_flag_offset = 0;
  id<MTLBuffer> telemetry_buffer = nil;
  NSUInteger telemetry_offset = 0;
  uint64_t frame_index = 0;
  bool history_reset = false;
};

class OpticalFlowSynthesizer {
public:
  static constexpr size_t kRingSlots = 4;

  struct alignas(8) DeviceTelemetrySlot {
    float mean_luma_diff;
    uint32_t scene_cut;
    uint64_t frame_index;
  };
  static_assert(sizeof(DeviceTelemetrySlot) == 16);

private:
  struct PyramidParams {
    uint32_t full_width;
    uint32_t full_height;
    uint32_t half_width;
    uint32_t half_height;
    uint32_t quarter_width;
    uint32_t quarter_height;
    uint32_t has_previous;
    uint32_t force_reset;
    uint32_t num_groups;
    float scene_cut_threshold;
    uint64_t frame_index;
  };

  struct CoarseParams {
    uint32_t half_width;
    uint32_t half_height;
    uint32_t quarter_width;
    uint32_t quarter_height;
    uint32_t coarse_width;
    uint32_t coarse_height;
    int32_t max_radius;
  };

  struct UpsampleParams {
    uint32_t full_width;
    uint32_t full_height;
    uint32_t half_width;
    uint32_t half_height;
    uint32_t quarter_width;
    uint32_t quarter_height;
    uint32_t coarse_width;
    uint32_t coarse_height;
  };

  id<MTLDevice> device_ = nil;
  id<MTLComputePipelineState> luma_pyramid_pso_ = nil;
  id<MTLComputePipelineState> scene_cut_reduce_pso_ = nil;
  id<MTLComputePipelineState> coarse_search_pso_ = nil;
  id<MTLComputePipelineState> refine_upsample_pso_ = nil;

  FlowConfig config_{};
  uint32_t width_ = 0;
  uint32_t height_ = 0;
  uint32_t half_width_ = 0;
  uint32_t half_height_ = 0;
  uint32_t quarter_width_ = 0;
  uint32_t quarter_height_ = 0;
  uint32_t coarse_width_ = 0;
  uint32_t coarse_height_ = 0;
  uint32_t pyramid_groups_ = 0;

  id<MTLTexture> luma_half_[2] = {nil, nil};
  id<MTLTexture> luma_quarter_[2] = {nil, nil};
  id<MTLTexture> coarse_flow_ = nil;
  id<MTLTexture> flow_ring_[kRingSlots] = {nil, nil, nil, nil};
  id<MTLBuffer> partial_diffs_ = nil;
  id<MTLBuffer> scene_cut_flags_ = nil;   // Private/Shared GPU-side flags (256B-aligned per slot)
  id<MTLBuffer> telemetry_ring_ = nil;    // Shared ring buffer for non-blocking telemetry inspection

  uint32_t parity_ = 0;
  uint64_t frame_counter_ = 0;
  bool has_history_ = false;

  mutable std::mutex telemetry_mutex_;
  FlowTelemetry latest_telemetry_{};

  static constexpr NSUInteger kSlotStride = 256;

  id<MTLTexture> make_private_tex(MTLPixelFormat fmt, uint32_t w, uint32_t h, NSString* label) {
    MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:fmt
                                                                                    width:w
                                                                                   height:h
                                                                                mipmapped:NO];
    desc.storageMode = MTLStorageModePrivate;
    desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    id<MTLTexture> tex = [device_ newTextureWithDescriptor:desc];
    if (tex) tex.label = label;
    return tex;
  }

  bool ensure_resources(uint32_t w, uint32_t h, std::string* error) {
    if (w == width_ && h == height_ && luma_half_[0] != nil && flow_ring_[0] != nil &&
        partial_diffs_ != nil && scene_cut_flags_ != nil && telemetry_ring_ != nil) {
      return true;
    }
    const uint32_t hw = std::max(1u, (w + 1u) / 2u);
    const uint32_t hh = std::max(1u, (h + 1u) / 2u);
    const uint32_t qw = std::max(1u, (w + 3u) / 4u);
    const uint32_t qh = std::max(1u, (h + 3u) / 4u);
    const uint32_t cw = std::max(1u, (qw + 3u) / 4u);
    const uint32_t ch = std::max(1u, (qh + 3u) / 4u);
    const uint32_t groups_x = (qw + 15u) / 16u;
    const uint32_t groups_y = (qh + 15u) / 16u;
    const uint32_t num_groups = std::max(1u, groups_x * groups_y);

    id<MTLTexture> nh[2] = {
      make_private_tex(MTLPixelFormatR16Float, hw, hh, @"netal.nr.flow.luma_half.0"),
      make_private_tex(MTLPixelFormatR16Float, hw, hh, @"netal.nr.flow.luma_half.1"),
    };
    id<MTLTexture> nq[2] = {
      make_private_tex(MTLPixelFormatR16Float, qw, qh, @"netal.nr.flow.luma_quarter.0"),
      make_private_tex(MTLPixelFormatR16Float, qw, qh, @"netal.nr.flow.luma_quarter.1"),
    };
    id<MTLTexture> ncoarse = make_private_tex(MTLPixelFormatRGBA16Float, cw, ch, @"netal.nr.flow.coarse");
    id<MTLTexture> nring[kRingSlots];
    for (size_t i = 0; i < kRingSlots; ++i) {
      nring[i] = make_private_tex(MTLPixelFormatRG16Float, w, h,
                                  [NSString stringWithFormat:@"netal.nr.flow.full.%zu", i]);
    }
    id<MTLBuffer> npdiffs = [device_ newBufferWithLength:size_t(num_groups) * sizeof(float)
                                                 options:MTLResourceStorageModePrivate];
    id<MTLBuffer> nflags = [device_ newBufferWithLength:kRingSlots * kSlotStride
                                                options:MTLResourceStorageModePrivate];
    id<MTLBuffer> ntelem = [device_ newBufferWithLength:kRingSlots * kSlotStride
                                                options:MTLResourceStorageModeShared];
    if (!nh[0] || !nh[1] || !nq[0] || !nq[1] || !ncoarse || !npdiffs || !nflags || !ntelem) {
      if (error) *error = "failed to allocate optical-flow GPU pyramids or buffers";
      return false;
    }
    for (size_t i = 0; i < kRingSlots; ++i) {
      if (!nring[i]) {
        if (error) *error = "failed to allocate optical-flow output ring texture";
        return false;
      }
    }
    std::memset(ntelem.contents, 0, kRingSlots * kSlotStride);

    width_ = w;
    height_ = h;
    half_width_ = hw;
    half_height_ = hh;
    quarter_width_ = qw;
    quarter_height_ = qh;
    coarse_width_ = cw;
    coarse_height_ = ch;
    pyramid_groups_ = num_groups;
    luma_half_[0] = nh[0];
    luma_half_[1] = nh[1];
    luma_quarter_[0] = nq[0];
    luma_quarter_[1] = nq[1];
    coarse_flow_ = ncoarse;
    for (size_t i = 0; i < kRingSlots; ++i) flow_ring_[i] = nring[i];
    partial_diffs_ = npdiffs;
    scene_cut_flags_ = nflags;
    telemetry_ring_ = ntelem;
    parity_ = 0;
    has_history_ = false;
    return true;
  }

public:
  OpticalFlowSynthesizer() = default;

  bool init(id<MTLDevice> device, id<MTLLibrary> library,
            const FlowConfig& config = FlowConfig{}, std::string* error = nullptr) {
    if (!device || !library) {
      if (error) *error = "OpticalFlowSynthesizer requires a valid Metal device and library";
      return false;
    }
    if (!std::isfinite(config.scene_cut_threshold) || config.scene_cut_threshold <= 0.0f ||
        config.scene_cut_threshold > 2.0f || config.max_quarter_search_radius < 4u ||
        config.max_quarter_search_radius > 24u) {
      if (error) *error = "invalid FlowConfig threshold or search radius";
      return false;
    }
    auto build_pso = [&](NSString* name) -> id<MTLComputePipelineState> {
      id<MTLFunction> fn = [library newFunctionWithName:name];
      if (!fn) {
        if (error) *error = std::string("missing Metal function: ") + name.UTF8String;
        return nil;
      }
      NSError* ns_err = nil;
      id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&ns_err];
      if (!pso && error) {
        *error = std::string("pipeline creation failed for ") + name.UTF8String + ": " +
                 (ns_err ? ns_err.localizedDescription.UTF8String : "unknown");
      }
      return pso;
    };

    auto p1 = build_pso(@"nr_flow_luma_pyramid");
    auto p2 = p1 ? build_pso(@"nr_flow_scene_cut_reduce") : nil;
    auto p3 = p2 ? build_pso(@"nr_flow_coarse_search") : nil;
    auto p4 = p3 ? build_pso(@"nr_flow_refine_upsample") : nil;
    if (!p1 || !p2 || !p3 || !p4) return false;

    device_ = device;
    luma_pyramid_pso_ = p1;
    scene_cut_reduce_pso_ = p2;
    coarse_search_pso_ = p3;
    refine_upsample_pso_ = p4;
    config_ = config;
    reset();
    return true;
  }

  bool init_from_metallib(id<MTLDevice> device, NSString* metallib_path,
                          const FlowConfig& config = FlowConfig{}, std::string* error = nullptr) {
    if (!device || !metallib_path || metallib_path.length == 0) {
      if (error) *error = "metallib path required";
      return false;
    }
    NSError* ns_err = nil;
    id<MTLLibrary> lib = [device newLibraryWithURL:[NSURL fileURLWithPath:metallib_path] error:&ns_err];
    if (!lib) {
      if (error) *error = ns_err ? ns_err.localizedDescription.UTF8String : "failed to load metallib";
      return false;
    }
    return init(device, lib, config, error);
  }

  bool init_from_source_file(id<MTLDevice> device, NSString* source_path,
                             const FlowConfig& config = FlowConfig{}, std::string* error = nullptr) {
    if (!device || !source_path || source_path.length == 0) {
      if (error) *error = "Metal source path required";
      return false;
    }
    NSError* ns_err = nil;
    NSString* src = [NSString stringWithContentsOfFile:source_path encoding:NSUTF8StringEncoding error:&ns_err];
    if (!src) {
      if (error) *error = ns_err ? ns_err.localizedDescription.UTF8String : "cannot read Metal source";
      return false;
    }
    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    id<MTLLibrary> lib = [device newLibraryWithSource:src options:opts error:&ns_err];
    if (!lib) {
      if (error) *error = ns_err ? ns_err.localizedDescription.UTF8String : "Metal source compile failed";
      return false;
    }
    return init(device, lib, config, error);
  }

  bool ready() const {
    return device_ != nil && luma_pyramid_pso_ != nil && scene_cut_reduce_pso_ != nil &&
           coarse_search_pso_ != nil && refine_upsample_pso_ != nil;
  }

  void reset() {
    has_history_ = false;
    parity_ = 0;
  }

  static FlowTelemetry read_telemetry_slot(id<MTLBuffer> buffer, NSUInteger offset) {
    FlowTelemetry out{};
    if (!buffer || !buffer.contents || offset + sizeof(DeviceTelemetrySlot) > buffer.length) {
      return out;
    }
    DeviceTelemetrySlot slot{};
    std::memcpy(&slot, static_cast<const uint8_t*>(buffer.contents) + offset, sizeof(slot));
    out.frame_index = slot.frame_index;
    out.mean_luma_diff = slot.mean_luma_diff;
    out.scene_cut = slot.scene_cut != 0u;
    out.valid = true;
    return out;
  }

  FlowTelemetry latest_completed_telemetry() const {
    std::lock_guard<std::mutex> lock(telemetry_mutex_);
    return latest_telemetry_;
  }

  // Encodes multi-scale pyramid + scene-cut analysis + coarse search + edge-aware subpixel upsampling
  // into `command_buffer` with ZERO CPU readback or synchronization. Never commits or waits.
  bool encode(id<MTLCommandBuffer> command_buffer, id<MTLTexture> current_color,
              bool force_reset, FlowOutput& out, std::string* error = nullptr) {
    out = FlowOutput{};
    if (!ready()) {
      if (error) *error = "OpticalFlowSynthesizer is not initialized";
      return false;
    }
    if (!command_buffer || !current_color) {
      if (error) *error = "command_buffer and current_color are required";
      return false;
    }
    const uint32_t w = static_cast<uint32_t>(current_color.width);
    const uint32_t h = static_cast<uint32_t>(current_color.height);
    if (w == 0 || h == 0 || w > 16384 || h > 16384) {
      if (error) *error = "unsupported color texture dimensions";
      return false;
    }
    if ((current_color.usage & MTLTextureUsageShaderRead) == 0) {
      if (error) *error = "current_color texture requires MTLTextureUsageShaderRead";
      return false;
    }
    const bool resized = (w != width_ || h != height_);
    if (!ensure_resources(w, h, error)) return false;

    const bool had_history = has_history_ && !resized;
    const bool effective_reset = force_reset || !had_history;
    const uint32_t curr_idx = parity_ & 1u;
    const uint32_t prev_idx = (parity_ ^ 1u) & 1u;
    const uint64_t frame_idx = ++frame_counter_;
    const size_t slot = static_cast<size_t>((frame_idx - 1u) % kRingSlots);
    const NSUInteger slot_offset = static_cast<NSUInteger>(slot * kSlotStride);

    id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
    if (!enc) {
      if (error) *error = "failed to create MTLComputeCommandEncoder for optical flow";
      return false;
    }
    enc.label = @"Netal NR Optical Flow & Scene-Cut Synthesizer";

    // 1. Perceptual luma pyramid (half-res + quarter-res) + per-threadgroup luma diff sums.
    const PyramidParams pyr_params{
      width_, height_, half_width_, half_height_, quarter_width_, quarter_height_,
      had_history ? 1u : 0u,
      effective_reset ? 1u : 0u,
      pyramid_groups_,
      config_.scene_cut_threshold,
      frame_idx,
    };
    [enc setComputePipelineState:luma_pyramid_pso_];
    [enc setTexture:current_color atIndex:0];
    [enc setTexture:luma_half_[curr_idx] atIndex:1];
    [enc setTexture:luma_quarter_[curr_idx] atIndex:2];
    [enc setTexture:luma_quarter_[prev_idx] atIndex:3];
    [enc setBuffer:partial_diffs_ offset:0 atIndex:0];
    [enc setBytes:&pyr_params length:sizeof(pyr_params) atIndex:1];
    [enc dispatchThreadgroups:MTLSizeMake((quarter_width_ + 15u) / 16u, (quarter_height_ + 15u) / 16u, 1)
        threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];

    // 2. Finalize frame-wide mean luma difference into telemetry ring + GPU-side scene-cut flag.
    [enc setComputePipelineState:scene_cut_reduce_pso_];
    [enc setBuffer:partial_diffs_ offset:0 atIndex:0];
    [enc setBuffer:scene_cut_flags_ offset:slot_offset atIndex:1];
    [enc setBuffer:telemetry_ring_ offset:slot_offset atIndex:2];
    [enc setBytes:&pyr_params length:sizeof(pyr_params) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

    // 3. Quarter-res SRAM-cached hierarchical 8x8 patch search (+/-24 quarter-res px = +/-96 full-res px)
    //    with half-res luma polish.
    const CoarseParams coarse_params{
      half_width_, half_height_, quarter_width_, quarter_height_, coarse_width_, coarse_height_,
      static_cast<int32_t>(config_.max_quarter_search_radius),
    };
    [enc setComputePipelineState:coarse_search_pso_];
    [enc setTexture:luma_quarter_[curr_idx] atIndex:0];
    [enc setTexture:luma_quarter_[prev_idx] atIndex:1];
    [enc setTexture:luma_half_[curr_idx] atIndex:2];
    [enc setTexture:luma_half_[prev_idx] atIndex:3];
    [enc setTexture:coarse_flow_ atIndex:4];
    [enc setBuffer:scene_cut_flags_ offset:slot_offset atIndex:0];
    [enc setBytes:&coarse_params length:sizeof(coarse_params) atIndex:1];
    [enc dispatchThreadgroups:MTLSizeMake((coarse_width_ + 7u) / 8u, (coarse_height_ + 7u) / 8u, 1)
        threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];

    // 4. Edge-aware median/bilateral refinement + parabolic subpixel upsampling to RG16Float full-res.
    const UpsampleParams up_params{
      width_, height_, half_width_, half_height_, quarter_width_, quarter_height_, coarse_width_, coarse_height_,
    };
    id<MTLTexture> out_tex = flow_ring_[slot];
    [enc setComputePipelineState:refine_upsample_pso_];
    [enc setTexture:current_color atIndex:0];
    [enc setTexture:luma_half_[curr_idx] atIndex:1];
    [enc setTexture:luma_half_[prev_idx] atIndex:2];
    [enc setTexture:coarse_flow_ atIndex:3];
    [enc setTexture:out_tex atIndex:4];
    [enc setBuffer:scene_cut_flags_ offset:slot_offset atIndex:0];
    [enc setBytes:&up_params length:sizeof(up_params) atIndex:1];
    [enc dispatchThreadgroups:MTLSizeMake((quarter_width_ + 15u) / 16u, (quarter_height_ + 15u) / 16u, 1)
        threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];

    [enc endEncoding];

    parity_ ^= 1u;
    has_history_ = true;

    out.flow_texture = out_tex;
    out.scene_cut_flag_buffer = scene_cut_flags_;
    out.scene_cut_flag_offset = slot_offset;
    out.telemetry_buffer = telemetry_ring_;
    out.telemetry_offset = slot_offset;
    out.frame_index = frame_idx;
    out.history_reset = effective_reset;

    if ([(id)command_buffer respondsToSelector:@selector(addCompletedHandler:)]) {
      id<MTLBuffer> telem_buf = telemetry_ring_;
      std::mutex* mtx = &telemetry_mutex_;
      FlowTelemetry* latest = &latest_telemetry_;
      [command_buffer addCompletedHandler:^(id<MTLCommandBuffer> completed) {
        if (completed.status != MTLCommandBufferStatusCompleted) return;
        FlowTelemetry sample = OpticalFlowSynthesizer::read_telemetry_slot(telem_buf, slot_offset);
        if (sample.valid) {
          std::lock_guard<std::mutex> lock(*mtx);
          if (sample.frame_index >= latest->frame_index) *latest = sample;
        }
      }];
    }
    return true;
  }
};

} // namespace netal_nr_adapter
