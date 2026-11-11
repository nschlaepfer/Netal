#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "netal_nr_optical_flow.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

inline float half_to_float(uint16_t h) {
  __fp16 fp;
  std::memcpy(&fp, &h, sizeof(fp));
  return static_cast<float>(fp);
}

// Rich, non-periodic multi-scale procedural scene so every 8x8 quarter-res patch has unique structure.
inline void eval_scene(float x, float y, int scene_id, uint8_t rgba[4]) {
  float v = 0.0f;
  if (scene_id == 0) {
    float s1 = 0.5f + 0.5f * std::sin(x * 0.031f + 0.7f * std::cos(y * 0.027f));
    float s2 = 0.5f + 0.5f * std::cos(x * 0.089f - y * 0.073f + 0.4f * std::sin((x + y) * 0.019f));
    float s3 = 0.5f + 0.5f * std::sin(x * 0.0113f + y * 0.0149f);
    float gx = x * 0.0045f, gy = y * 0.0045f;
    float radial = 0.5f + 0.5f * std::sin(std::sqrt((gx - 4.1f) * (gx - 4.1f) + (gy - 2.3f) * (gy - 2.3f)) * 9.0f);
    v = 0.35f * s1 + 0.30f * s2 + 0.20f * s3 + 0.15f * radial;
  } else {
    // Abruptly different high-contrast checkerboard/diagonal scene for scene-cut test.
    int cx = static_cast<int>(std::floor(x / 48.0f));
    int cy = static_cast<int>(std::floor(y / 48.0f));
    v = ((cx ^ cy) & 1) ? 0.95f : 0.05f;
  }
  v = std::clamp(v, 0.0f, 1.0f);
  rgba[0] = static_cast<uint8_t>(std::lround(v * 255.0f));
  rgba[1] = static_cast<uint8_t>(std::lround(std::clamp(v * 0.9f + 0.05f, 0.0f, 1.0f) * 255.0f));
  rgba[2] = static_cast<uint8_t>(std::lround(std::clamp(1.0f - v * 0.5f, 0.0f, 1.0f) * 255.0f));
  rgba[3] = 255u;
}

id<MTLTexture> make_scene_texture(id<MTLDevice> device, uint32_t w, uint32_t h,
                                  float shift_x, float shift_y, int scene_id) {
  // Convention check:
  // Canonical NR FrameInputs mapping: history_uv = current_uv + flow_px / float2(W, H)
  // i.e. curr(x, y) == prev(x + flow_x, y + flow_y).
  // So setting curr(x, y) = scene(x + shift_x, y + shift_y) relative to prev(x, y) = scene(x, y)
  // yields ground-truth CurrentToPrevious flow = (shift_x, shift_y) in PixelsFullRes (y-down).
  std::vector<uint8_t> pixels(size_t(w) * size_t(h) * 4u);
  for (uint32_t y = 0; y < h; ++y) {
    for (uint32_t x = 0; x < w; ++x) {
      eval_scene(float(x) + shift_x, float(y) + shift_y, scene_id,
                 &pixels[(size_t(y) * w + x) * 4u]);
    }
  }
  MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                  width:w
                                                                                 height:h
                                                                              mipmapped:NO];
  desc.storageMode = MTLStorageModeShared;
  desc.usage = MTLTextureUsageShaderRead;
  id<MTLTexture> tex = [device newTextureWithDescriptor:desc];
  [tex replaceRegion:MTLRegionMake2D(0, 0, w, h)
         mipmapLevel:0
           withBytes:pixels.data()
         bytesPerRow:size_t(w) * 4u];
  return tex;
}

id<MTLTexture> make_scene_texture_rgba16f(id<MTLDevice> device, uint32_t w, uint32_t h,
                                          float shift_x, float shift_y, int scene_id) {
  std::vector<__fp16> pixels(size_t(w) * size_t(h) * 4u);
  for (uint32_t y = 0; y < h; ++y) {
    for (uint32_t x = 0; x < w; ++x) {
      uint8_t rgba8[4];
      eval_scene(float(x) + shift_x, float(y) + shift_y, scene_id, rgba8);
      size_t off = (size_t(y) * w + x) * 4u;
      pixels[off + 0] = static_cast<__fp16>(rgba8[0] / 255.0f);
      pixels[off + 1] = static_cast<__fp16>(rgba8[1] / 255.0f);
      pixels[off + 2] = static_cast<__fp16>(rgba8[2] / 255.0f);
      pixels[off + 3] = static_cast<__fp16>(1.0f);
    }
  }
  MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                                                  width:w
                                                                                 height:h
                                                                              mipmapped:NO];
  desc.storageMode = MTLStorageModeShared;
  desc.usage = MTLTextureUsageShaderRead;
  id<MTLTexture> tex = [device newTextureWithDescriptor:desc];
  [tex replaceRegion:MTLRegionMake2D(0, 0, w, h)
         mipmapLevel:0
           withBytes:pixels.data()
         bytesPerRow:size_t(w) * 8u];
  return tex;
}

struct FlowStats {
  double mean_dx = 0.0;
  double mean_dy = 0.0;
  double mae_dx = 0.0;
  double mae_dy = 0.0;
  float mean_luma_diff = 0.0f;
  bool scene_cut = false;
  uint32_t gpu_scene_cut_flag = 0;
};

FlowStats run_pair_and_readback(id<MTLDevice> device, id<MTLCommandQueue> queue,
                                netal_nr_adapter::OpticalFlowSynthesizer& synth,
                                id<MTLTexture> prev_tex, id<MTLTexture> curr_tex,
                                float expected_dx, float expected_dy) {
  synth.reset();
  const uint32_t w = static_cast<uint32_t>(curr_tex.width);
  const uint32_t h = static_cast<uint32_t>(curr_tex.height);

  // Frame 1: seed history.
  {
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    netal_nr_adapter::FlowOutput out1;
    std::string err;
    if (!synth.encode(cb, prev_tex, true, out1, &err)) {
      throw std::runtime_error("frame 1 encode failed: " + err);
    }
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) {
      throw std::runtime_error("frame 1 command buffer failed");
    }
  }

  // Frame 2: estimate flow prev <- curr + copy output RG16Float + GPU scene_cut flag for test verification.
  const size_t row_bytes = size_t(w) * 4u;
  id<MTLBuffer> readback = [device newBufferWithLength:row_bytes * h options:MTLResourceStorageModeShared];
  id<MTLBuffer> flag_readback = [device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
  netal_nr_adapter::FlowOutput out2;
  {
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    std::string err;
    if (!synth.encode(cb, curr_tex, false, out2, &err)) {
      throw std::runtime_error("frame 2 encode failed: " + err);
    }
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromTexture:out2.flow_texture
              sourceSlice:0
              sourceLevel:0
             sourceOrigin:MTLOriginMake(0, 0, 0)
               sourceSize:MTLSizeMake(w, h, 1)
                 toBuffer:readback
        destinationOffset:0
   destinationBytesPerRow:row_bytes
 destinationBytesPerImage:row_bytes * h];
    [blit copyFromBuffer:out2.scene_cut_flag_buffer
            sourceOffset:out2.scene_cut_flag_offset
                toBuffer:flag_readback
       destinationOffset:0
                    size:sizeof(uint32_t)];
    [blit endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) {
      throw std::runtime_error("frame 2 command buffer failed");
    }
  }

  const auto telem = netal_nr_adapter::OpticalFlowSynthesizer::read_telemetry_slot(
      out2.telemetry_buffer, out2.telemetry_offset);
  const auto* raw = static_cast<const uint16_t*>(readback.contents);

  // Measure interior region (excluding 64px border where shifted scene wraps/clamps at edges).
  const uint32_t margin = 64u;
  double sum_dx = 0.0, sum_dy = 0.0, err_dx = 0.0, err_dy = 0.0;
  size_t count = 0;
  for (uint32_t y = margin; y + margin < h; ++y) {
    for (uint32_t x = margin; x + margin < w; ++x) {
      size_t idx = (size_t(y) * w + x) * 2u;
      float dx = half_to_float(raw[idx + 0]);
      float dy = half_to_float(raw[idx + 1]);
      sum_dx += dx;
      sum_dy += dy;
      err_dx += std::fabs(dx - expected_dx);
      err_dy += std::fabs(dy - expected_dy);
      ++count;
    }
  }

  FlowStats stats{};
  stats.mean_dx = sum_dx / double(count);
  stats.mean_dy = sum_dy / double(count);
  stats.mae_dx = err_dx / double(count);
  stats.mae_dy = err_dy / double(count);
  stats.mean_luma_diff = telem.mean_luma_diff;
  stats.scene_cut = telem.scene_cut;
  stats.gpu_scene_cut_flag = *static_cast<const uint32_t*>(flag_readback.contents);
  return stats;
}

} // namespace

int main(int argc, char** argv) {
  @autoreleasepool {
    try {
      if (argc != 2) {
        std::fprintf(stderr, "usage: %s <optical_flow.metallib>\n", argv[0]);
        return 2;
      }
      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      if (!device) {
        std::printf("{\"skipped\": true, \"reason\": \"no Metal device\"}\n");
        return 0;
      }
      id<MTLCommandQueue> queue = [device newCommandQueue];
      if (!queue) throw std::runtime_error("no Metal command queue");

      netal_nr_adapter::OpticalFlowSynthesizer synth;
      std::string err;
      if (!synth.init_from_metallib(device, [NSString stringWithUTF8String:argv[1]],
                                    netal_nr_adapter::FlowConfig{}, &err)) {
        throw std::runtime_error("synth init failed: " + err);
      }

      const uint32_t W = 1920, H = 1080;
      id<MTLTexture> base = make_scene_texture(device, W, H, 0.0f, 0.0f, 0);
      id<MTLTexture> shift_pos8_neg4 = make_scene_texture(device, W, H, 8.0f, -4.0f, 0);
      id<MTLTexture> shift_neg12_pos6 = make_scene_texture(device, W, H, -12.0f, 6.0f, 0);
      id<MTLTexture> shift_large = make_scene_texture(device, W, H, 48.0f, -32.0f, 0);
      id<MTLTexture> base_f16 = make_scene_texture_rgba16f(device, W, H, 0.0f, 0.0f, 0);
      id<MTLTexture> shift_odd_f16 = make_scene_texture_rgba16f(device, W, H, 5.0f, -3.0f, 0);
      id<MTLTexture> cut_tex = make_scene_texture(device, W, H, 0.0f, 0.0f, 1);

      const auto static_stats = run_pair_and_readback(device, queue, synth, base, base, 0.0f, 0.0f);
      const auto t1_stats = run_pair_and_readback(device, queue, synth, base, shift_pos8_neg4, 8.0f, -4.0f);
      const auto t2_stats = run_pair_and_readback(device, queue, synth, base, shift_neg12_pos6, -12.0f, 6.0f);
      const auto t3_stats = run_pair_and_readback(device, queue, synth, base, shift_large, 48.0f, -32.0f);
      const auto odd_f16_stats = run_pair_and_readback(device, queue, synth, base_f16, shift_odd_f16, 5.0f, -3.0f);
      const auto cut_stats = run_pair_and_readback(device, queue, synth, base, cut_tex, 0.0f, 0.0f);

      // Benchmark 1920x1080 pure-GPU execution time using GPU-private textures (realistic game hot path).
      MTLTextureDescriptor* pdesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                       width:W
                                                                                      height:H
                                                                                   mipmapped:NO];
      pdesc.storageMode = MTLStorageModePrivate;
      pdesc.usage = MTLTextureUsageShaderRead;
      id<MTLTexture> priv_a = [device newTextureWithDescriptor:pdesc];
      id<MTLTexture> priv_b = [device newTextureWithDescriptor:pdesc];
      {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit copyFromTexture:base toTexture:priv_a];
        [blit copyFromTexture:shift_pos8_neg4 toTexture:priv_b];
        [blit endEncoding];
        netal_nr_adapter::FlowOutput warm;
        for (int i = 0; i < 8; ++i) {
          synth.encode(cb, (i & 1) ? priv_b : priv_a, i == 0, warm, &err);
        }
        [cb commit];
        [cb waitUntilCompleted];
      }

      std::vector<double> gpu_ms_samples;
      constexpr int kBenchIters = 40;
      gpu_ms_samples.reserve(kBenchIters);
      for (int i = 0; i < kBenchIters; ++i) {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        netal_nr_adapter::FlowOutput fout;
        if (!synth.encode(cb, (i & 1) ? priv_b : priv_a, false, fout, &err)) {
          throw std::runtime_error("bench encode failed: " + err);
        }
        [cb commit];
        [cb waitUntilCompleted];
        double ms = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
        if (ms > 0.0) gpu_ms_samples.push_back(ms);
      }
      std::sort(gpu_ms_samples.begin(), gpu_ms_samples.end());
      double median_ms = gpu_ms_samples.empty() ? 0.0 : gpu_ms_samples[gpu_ms_samples.size() / 2];
      double p90_ms = gpu_ms_samples.empty() ? 0.0 : gpu_ms_samples[(gpu_ms_samples.size() * 9) / 10];

      std::printf(
          "{\n"
          "  \"device\": \"%s\",\n"
          "  \"resolution\": \"%ux%u\",\n"
          "  \"static\": {\"mean_dx\": %.5f, \"mean_dy\": %.5f, \"mae_dx\": %.5f, \"mae_dy\": %.5f, \"scene_cut\": %s},\n"
          "  \"shift_8_neg4\": {\"mean_dx\": %.5f, \"mean_dy\": %.5f, \"mae_dx\": %.5f, \"mae_dy\": %.5f, \"scene_cut\": %s},\n"
          "  \"shift_neg12_6\": {\"mean_dx\": %.5f, \"mean_dy\": %.5f, \"mae_dx\": %.5f, \"mae_dy\": %.5f, \"scene_cut\": %s},\n"
          "  \"shift_48_neg32\": {\"mean_dx\": %.5f, \"mean_dy\": %.5f, \"mae_dx\": %.5f, \"mae_dy\": %.5f, \"scene_cut\": %s},\n"
          "  \"shift_5_neg3_rgba16f\": {\"mean_dx\": %.5f, \"mean_dy\": %.5f, \"mae_dx\": %.5f, \"mae_dy\": %.5f, \"scene_cut\": %s},\n"
          "  \"scene_cut\": {\"mean_dx\": %.5f, \"mean_dy\": %.5f, \"mae_dx\": %.5f, \"mae_dy\": %.5f, \"mean_luma_diff\": %.5f, \"scene_cut\": %s, \"gpu_flag\": %u},\n"
          "  \"benchmark_1080p\": {\"median_gpu_ms\": %.5f, \"p90_gpu_ms\": %.5f, \"samples\": %zu}\n"
          "}\n",
          device.name.UTF8String, W, H,
          static_stats.mean_dx, static_stats.mean_dy, static_stats.mae_dx, static_stats.mae_dy,
          static_stats.scene_cut ? "true" : "false",
          t1_stats.mean_dx, t1_stats.mean_dy, t1_stats.mae_dx, t1_stats.mae_dy,
          t1_stats.scene_cut ? "true" : "false",
          t2_stats.mean_dx, t2_stats.mean_dy, t2_stats.mae_dx, t2_stats.mae_dy,
          t2_stats.scene_cut ? "true" : "false",
          t3_stats.mean_dx, t3_stats.mean_dy, t3_stats.mae_dx, t3_stats.mae_dy,
          t3_stats.scene_cut ? "true" : "false",
          odd_f16_stats.mean_dx, odd_f16_stats.mean_dy, odd_f16_stats.mae_dx, odd_f16_stats.mae_dy,
          odd_f16_stats.scene_cut ? "true" : "false",
          cut_stats.mean_dx, cut_stats.mean_dy, cut_stats.mae_dx, cut_stats.mae_dy,
          cut_stats.mean_luma_diff, cut_stats.scene_cut ? "true" : "false", cut_stats.gpu_scene_cut_flag,
          median_ms, p90_ms, gpu_ms_samples.size());
      return 0;
    } catch (const std::exception& e) {
      std::fprintf(stderr, "FAIL: %s\n", e.what());
      return 1;
    }
  }
}
