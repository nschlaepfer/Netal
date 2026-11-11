// SPDX-License-Identifier: MIT AND Apache-2.0
// Copyright (c) Netal Contributors.
//
// Command-line driver for Netal's native Metal DLSS 5 Neural Rendering engine.

#import "nr_fast_runtime.h"

#import <CoreGraphics/CoreGraphics.h>
#import <Foundation/Foundation.h>
#import <ImageIO/ImageIO.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace {

using namespace netal_nr_fast;

bool ParseWxH(const std::string& s, uint32_t* w, uint32_t* h) {
  size_t x = s.find('x');
  if (x == std::string::npos) x = s.find('X');
  if (x == std::string::npos || x == 0 || x + 1 >= s.size()) return false;
  char* e1 = nullptr;
  char* e2 = nullptr;
  unsigned long pw = std::strtoul(s.substr(0, x).c_str(), &e1, 10);
  unsigned long ph = std::strtoul(s.substr(x + 1).c_str(), &e2, 10);
  if (!e1 || *e1 != '\0' || !e2 || *e2 != '\0' || pw == 0 || ph == 0) return false;
  *w = static_cast<uint32_t>(pw);
  *h = static_cast<uint32_t>(ph);
  return true;
}

bool ParseProfile(const std::string& s, ControlProfile* out) {
  if (s == "standard") { *out = ControlProfile::Standard; return true; }
  if (s == "natural") { *out = ControlProfile::Natural; return true; }
  if (s == "cinematic") { *out = ControlProfile::Cinematic; return true; }
  if (s == "neutral") { *out = ControlProfile::Neutral; return true; }
  return false;
}

bool ParseTopology(const std::string& s, TopologyKind* out) {
  if (s == "small" || s == "small31") { *out = TopologyKind::Small31; return true; }
  if (s == "ht" || s == "ht71") { *out = TopologyKind::HT71; return true; }
  return false;
}

double Percentile(std::vector<double> vals, double p) {
  if (vals.empty()) return 0.0;
  std::sort(vals.begin(), vals.end());
  double idx = (p / 100.0) * static_cast<double>(vals.size() - 1);
  size_t lo = static_cast<size_t>(std::floor(idx));
  size_t hi = static_cast<size_t>(std::ceil(idx));
  double t = idx - static_cast<double>(lo);
  return vals[lo] * (1.0 - t) + vals[hi] * t;
}

bool LoadImageRGBA32F(
    const std::string& path,
    std::vector<float>* out_rgba,
    uint32_t* out_w,
    uint32_t* out_h,
    std::string* out_err) {
  NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]];
  CGImageSourceRef src = CGImageSourceCreateWithURL((__bridge CFURLRef)url, nullptr);
  if (!src) {
    if (out_err) *out_err = "cannot open image: " + path;
    return false;
  }
  CGImageRef img = CGImageSourceCreateImageAtIndex(src, 0, nullptr);
  CFRelease(src);
  if (!img) {
    if (out_err) *out_err = "cannot decode image: " + path;
    return false;
  }

  const uint32_t w = static_cast<uint32_t>(CGImageGetWidth(img));
  const uint32_t h = static_cast<uint32_t>(CGImageGetHeight(img));
  out_rgba->assign(static_cast<size_t>(w) * h * 4u, 0.0f);
  CGColorSpaceRef cs = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
  const uint32_t bitmap_info =
      static_cast<uint32_t>(kCGImageAlphaNoneSkipLast) |
      static_cast<uint32_t>(kCGBitmapFloatComponents) |
      static_cast<uint32_t>(kCGBitmapByteOrder32Host);
  CGContextRef ctx = CGBitmapContextCreate(
      out_rgba->data(),
      w,
      h,
      32,
      static_cast<size_t>(w) * 16u,
      cs,
      bitmap_info);
  CGColorSpaceRelease(cs);
  if (!ctx) {
    CGImageRelease(img);
    if (out_err) *out_err = "failed to create CoreGraphics float32 bitmap context";
    return false;
  }
  CGContextDrawImage(ctx, CGRectMake(0, 0, w, h), img);
  CGContextRelease(ctx);
  CGImageRelease(img);

  for (size_t i = 0; i < static_cast<size_t>(w) * h; ++i) {
    (*out_rgba)[i * 4u + 3u] = 1.0f;
  }
  *out_w = w;
  *out_h = h;
  return true;
}

bool SavePNGFromRGBA32F(
    const std::string& path,
    const float* rgba,
    uint32_t w,
    uint32_t h,
    int bit_depth,
    std::string* out_err) {
  CGColorSpaceRef cs = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
  CGDataProviderRef provider = nullptr;
  CGImageRef img = nullptr;
  std::vector<uint8_t> buf8;
  std::vector<uint16_t> buf16;

  if (bit_depth == 16) {
    buf16.resize(static_cast<size_t>(w) * h * 4u);
    for (size_t i = 0; i < static_cast<size_t>(w) * h; ++i) {
      for (size_t c = 0; c < 3; ++c) {
        float v = std::clamp(rgba[i * 4u + c], 0.0f, 1.0f);
        buf16[i * 4u + c] = static_cast<uint16_t>(std::round(v * 65535.0f));
      }
      buf16[i * 4u + 3u] = 65535u;
    }
    CFDataRef data = CFDataCreate(
        kCFAllocatorDefault,
        reinterpret_cast<const UInt8*>(buf16.data()),
        static_cast<CFIndex>(buf16.size() * sizeof(uint16_t)));
    provider = CGDataProviderCreateWithCFData(data);
    CFRelease(data);
    img = CGImageCreate(
        w,
        h,
        16,
        64,
        static_cast<size_t>(w) * 8u,
        cs,
        kCGImageAlphaNoneSkipLast | kCGBitmapByteOrder16Host,
        provider,
        nullptr,
        false,
        kCGRenderingIntentDefault);
  } else {
    buf8.resize(static_cast<size_t>(w) * h * 4u);
    for (size_t i = 0; i < static_cast<size_t>(w) * h; ++i) {
      for (size_t c = 0; c < 3; ++c) {
        float v = std::clamp(rgba[i * 4u + c], 0.0f, 1.0f);
        buf8[i * 4u + c] = static_cast<uint8_t>(std::round(v * 255.0f));
      }
      buf8[i * 4u + 3u] = 255u;
    }
    CFDataRef data = CFDataCreate(
        kCFAllocatorDefault, buf8.data(), static_cast<CFIndex>(buf8.size()));
    provider = CGDataProviderCreateWithCFData(data);
    CFRelease(data);
    img = CGImageCreate(
        w,
        h,
        8,
        32,
        static_cast<size_t>(w) * 4u,
        cs,
        static_cast<uint32_t>(kCGImageAlphaNoneSkipLast) | static_cast<uint32_t>(kCGBitmapByteOrder32Big),
        provider,
        nullptr,
        false,
        kCGRenderingIntentDefault);
  }
  CGDataProviderRelease(provider);
  CGColorSpaceRelease(cs);
  if (!img) {
    if (out_err) *out_err = "failed to construct CGImage for output";
    return false;
  }

  NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]];
  CGImageDestinationRef dst =
      CGImageDestinationCreateWithURL((__bridge CFURLRef)url, CFSTR("public.png"), 1, nullptr);
  if (!dst) {
    CGImageRelease(img);
    if (out_err) *out_err = "cannot create PNG destination: " + path;
    return false;
  }
  CGImageDestinationAddImage(dst, img, nullptr);
  bool ok = CGImageDestinationFinalize(dst);
  CFRelease(dst);
  CGImageRelease(img);
  if (!ok && out_err) *out_err = "failed to finalize PNG: " + path;
  return ok;
}

bool WriteJsonOutput(NSDictionary* dict, const std::string& json_path, std::string* out_err) {
  NSError* err = nil;
  NSData* data = [NSJSONSerialization dataWithJSONObject:dict
                                                 options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys
                                                   error:&err];
  if (!data) {
    if (out_err) *out_err = "failed to serialize JSON summary";
    return false;
  }
  std::string text(static_cast<const char*>([data bytes]), [data length]);
  text.push_back('\n');
  std::fwrite(text.data(), 1, text.size(), stdout);
  std::fflush(stdout);
  if (!json_path.empty()) {
    std::ofstream out(json_path, std::ios::binary);
    if (!out.is_open()) {
      if (out_err) *out_err = "cannot write JSON report: " + json_path;
      return false;
    }
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
  }
  return true;
}

std::unique_ptr<Context> CreateContextFromArgs(
    id<MTLDevice> device,
    const std::string& model_path,
    const std::string& synthetic_spec,
    const std::string& metallib_path,
    uint64_t seed,
    std::string* out_err) {
  std::string syn = synthetic_spec;
  if (syn.empty() && model_path.compare(0, 10, "synthetic:") == 0) {
    syn = model_path.substr(10);
  }
  if (!syn.empty()) {
    TopologyKind kind = TopologyKind::HT71;
    if (!ParseTopology(syn, &kind)) {
      if (out_err) *out_err = "invalid synthetic topology '" + syn + "' (expected small|ht)";
      return nullptr;
    }
    SyntheticOptions sopt;
    sopt.metallib_path = metallib_path;
    sopt.seed = seed;
    return Context::PrepareSynthetic(device, kind, sopt, out_err);
  }
  if (model_path.empty()) {
    if (out_err) *out_err = "either --model <path> or --synthetic <small|ht> is required";
    return nullptr;
  }
  LoadOptions lopt;
  lopt.metallib_path = metallib_path;
  return Context::LoadFromPath(device, model_path, lopt, out_err);
}

id<MTLTexture> MakeSharedRGBA32FTexture(id<MTLDevice> dev, uint32_t w, uint32_t h, MTLTextureUsage usage) {
  MTLTextureDescriptor* td =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                         width:w
                                                        height:h
                                                     mipmapped:NO];
  td.storageMode = MTLStorageModeShared;
  td.usage = usage;
  return [dev newTextureWithDescriptor:td];
}

int RunRenderImage(int argc, char** argv) {
  std::string input_path, output_path, model_path, synthetic_spec, metallib_path, json_path, dump_rgb_path;
  uint32_t net_w = 0, net_h = 0;
  uint32_t req_out_w = 0, req_out_h = 0;
  ControlProfile profile = ControlProfile::Standard;
  float detail_strength = 1.0f;
  float colour_strength = 1.0f;
  float detail_radius = 4.0f;
  float intensity = 1.0f;
  uint32_t frame_index = 0;
  bool srgb_transfer = false;
  int bit_depth = 8;
  int warmup = 0;
  int repeat = 1;
  uint64_t seed = 0x4E52464153543031ULL;

  for (int i = 0; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--srgb") {
      srgb_transfer = true;
      continue;
    }
    auto need_val = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "error: option %s requires a value\n", arg.c_str());
        std::exit(2);
      }
      return argv[++i];
    };
    if (arg == "--input") input_path = need_val();
    else if (arg == "--output") output_path = need_val();
    else if (arg == "--model") model_path = need_val();
    else if (arg == "--synthetic") synthetic_spec = need_val();
    else if (arg == "--metallib") metallib_path = need_val();
    else if (arg == "--json") json_path = need_val();
    else if (arg == "--dump-rgb-f32") dump_rgb_path = need_val();
    else if (arg == "--net-size") {
      if (!ParseWxH(need_val(), &net_w, &net_h)) {
        std::fprintf(stderr, "error: invalid --net-size (expected WxH)\n");
        return 2;
      }
    } else if (arg == "--out-size") {
      if (!ParseWxH(need_val(), &req_out_w, &req_out_h)) {
        std::fprintf(stderr, "error: invalid --out-size (expected WxH)\n");
        return 2;
      }
    } else if (arg == "--profile") {
      if (!ParseProfile(need_val(), &profile)) {
        std::fprintf(stderr, "error: invalid --profile\n");
        return 2;
      }
    } else if (arg == "--detail-strength") detail_strength = std::stof(need_val());
    else if (arg == "--colour-strength") colour_strength = std::stof(need_val());
    else if (arg == "--detail-radius") detail_radius = std::stof(need_val());
    else if (arg == "--intensity") intensity = std::stof(need_val());
    else if (arg == "--frame-index") frame_index = static_cast<uint32_t>(std::stoul(need_val()));
    else if (arg == "--srgb-transfer") srgb_transfer = (std::stoi(need_val()) != 0);
    else if (arg == "--bit-depth") bit_depth = std::stoi(need_val());
    else if (arg == "--warmup") warmup = std::stoi(need_val());
    else if (arg == "--repeat") repeat = std::max(1, std::stoi(need_val()));
    else if (arg == "--seed") seed = std::stoull(need_val());
    else {
      std::fprintf(stderr, "error: unknown render-image argument '%s'\n", arg.c_str());
      return 2;
    }
  }

  if (input_path.empty() || output_path.empty()) {
    std::fprintf(stderr, "error: render-image requires --input and --output\n");
    return 2;
  }

  std::string err;
  std::vector<float> in_rgba;
  uint32_t w = 0, h = 0;
  if (!LoadImageRGBA32F(input_path, &in_rgba, &w, &h, &err)) {
    std::fprintf(stderr, "error: %s\n", err.c_str());
    return 1;
  }

  id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  id<MTLCommandQueue> queue = [dev newCommandQueue];
  auto ctx = CreateContextFromArgs(dev, model_path, synthetic_spec, metallib_path, seed, &err);
  if (!ctx) {
    std::fprintf(stderr, "error: %s\n", err.c_str());
    return 1;
  }

  const uint32_t out_w = req_out_w > 0 ? req_out_w : w;
  const uint32_t out_h = req_out_h > 0 ? req_out_h : h;

  id<MTLTexture> in_tex = MakeSharedRGBA32FTexture(dev, w, h, MTLTextureUsageShaderRead);
  id<MTLTexture> out_tex =
      MakeSharedRGBA32FTexture(dev, out_w, out_h, MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite);
  [in_tex replaceRegion:MTLRegionMake2D(0, 0, w, h)
            mipmapLevel:0
              withBytes:in_rgba.data()
            bytesPerRow:static_cast<size_t>(w) * 16u];

  FrameInputs fi{};
  fi.color_texture = in_tex;
  fi.output_texture = out_tex;
  fi.params.output_width = out_w;
  fi.params.output_height = out_h;
  fi.params.network_width = net_w;
  fi.params.network_height = net_h;
  fi.params.geometry_mode =
      ((net_w > 0 && (net_w != w || net_h != h)) || out_w != w || out_h != h)
          ? GeometryMode::ScaledNetwork
          : GeometryMode::VendorAligned;
  fi.params.profile = profile;
  fi.params.detail_strength = detail_strength;
  fi.params.colour_strength = colour_strength;
  fi.params.detail_radius = detail_radius;
  fi.params.intensity = intensity;
  fi.params.frame_index = frame_index;
  fi.params.reset = true;
  fi.params.srgb_transfer = srgb_transfer;

  Report rep{};
  for (int i = 0; i < warmup; ++i) {
    if (!ctx->run_sync(fi, queue, rep, &err)) {
      std::fprintf(stderr, "error: %s\n", err.c_str());
      return 1;
    }
  }

  std::vector<double> gpu_times;
  std::vector<double> cpu_times;
  for (int i = 0; i < repeat; ++i) {
    if (!ctx->run_sync(fi, queue, rep, &err)) {
      std::fprintf(stderr, "error: %s\n", err.c_str());
      return 1;
    }
    gpu_times.push_back(rep.gpu_ms);
    cpu_times.push_back(rep.cpu_encode_ms);
  }

  std::vector<float> out_rgba(static_cast<size_t>(out_w) * out_h * 4u, 0.0f);
  [out_tex getBytes:out_rgba.data()
        bytesPerRow:static_cast<size_t>(out_w) * 16u
         fromRegion:MTLRegionMake2D(0, 0, out_w, out_h)
        mipmapLevel:0];

  double mean_rgb = 0.0;
  for (size_t px = 0; px < static_cast<size_t>(out_w) * out_h; ++px) {
    for (size_t c = 0; c < 3; ++c) {
      float v = out_rgba[px * 4u + c];
      if (!std::isfinite(v)) {
        std::fprintf(stderr, "error: non-finite pixel at index %zu\n", px);
        return 1;
      }
      mean_rgb += v;
    }
  }
  mean_rgb /= static_cast<double>(static_cast<size_t>(out_w) * out_h * 3u);

  if (!dump_rgb_path.empty()) {
    std::vector<float> rgb3(static_cast<size_t>(out_w) * out_h * 3u);
    for (size_t px = 0; px < static_cast<size_t>(out_w) * out_h; ++px) {
      rgb3[px * 3u + 0u] = out_rgba[px * 4u + 0u];
      rgb3[px * 3u + 1u] = out_rgba[px * 4u + 1u];
      rgb3[px * 3u + 2u] = out_rgba[px * 4u + 2u];
    }
    std::ofstream df(dump_rgb_path, std::ios::binary);
    df.write(reinterpret_cast<const char*>(rgb3.data()), static_cast<std::streamsize>(rgb3.size() * sizeof(float)));
  }

  if (!SavePNGFromRGBA32F(output_path, out_rgba.data(), out_w, out_h, bit_depth, &err)) {
    std::fprintf(stderr, "error: %s\n", err.c_str());
    return 1;
  }

  NSDictionary* summary = @{
    @"command" : @"render-image",
    @"device" : [dev name],
    @"topology" : [NSString stringWithUTF8String:rep.topology.c_str()],
    @"weight_format" : [NSString stringWithUTF8String:rep.weight_format.c_str()],
    @"synthetic" : @(rep.synthetic_weights),
    @"block_count" : @(rep.block_count),
    @"dispatch_count" : @(rep.dispatch_count),
    @"output_size" : @[ @(out_w), @(out_h) ],
    @"network_size" : @[ @(rep.network_width), @(rep.network_height) ],
    @"gpu_ms_p50" : @(Percentile(gpu_times, 50.0)),
    @"gpu_ms_p95" : @(Percentile(gpu_times, 95.0)),
    @"cpu_encode_ms_p50" : @(Percentile(cpu_times, 50.0)),
    @"mean_rgb" : @(mean_rgb),
    @"weight_buffer_mb" : @(static_cast<double>(ctx->weight_buffer_bytes()) / (1024.0 * 1024.0)),
    @"scratch_buffer_mb" : @(static_cast<double>(ctx->scratch_buffer_bytes()) / (1024.0 * 1024.0)),
  };
  if (!WriteJsonOutput(summary, json_path, &err)) {
    std::fprintf(stderr, "error: %s\n", err.c_str());
    return 1;
  }
  return 0;
}

int RunEvalTensor(int argc, char** argv) {
  std::string input_path, output_path, model_path, synthetic_spec, metallib_path, json_path;
  uint32_t height = 0, width = 0;
  bool fp16_io = false;
  int warmup = 0;
  int repeat = 1;
  uint64_t seed = 0x4E52464153543031ULL;

  for (int i = 0; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--fp16-io") {
      fp16_io = true;
      continue;
    }
    auto need_val = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "error: option %s requires a value\n", arg.c_str());
        std::exit(2);
      }
      return argv[++i];
    };
    if (arg == "--input") input_path = need_val();
    else if (arg == "--output") output_path = need_val();
    else if (arg == "--height") height = static_cast<uint32_t>(std::stoul(need_val()));
    else if (arg == "--width") width = static_cast<uint32_t>(std::stoul(need_val()));
    else if (arg == "--model") model_path = need_val();
    else if (arg == "--synthetic") synthetic_spec = need_val();
    else if (arg == "--metallib") metallib_path = need_val();
    else if (arg == "--warmup") warmup = std::stoi(need_val());
    else if (arg == "--repeat") repeat = std::max(1, std::stoi(need_val()));
    else if (arg == "--seed") seed = std::stoull(need_val());
    else if (arg == "--json") json_path = need_val();
    else {
      std::fprintf(stderr, "error: unknown eval-tensor argument '%s'\n", arg.c_str());
      return 2;
    }
  }

  if (input_path.empty() || output_path.empty() || height == 0 || width == 0) {
    std::fprintf(stderr, "error: eval-tensor requires --input, --output, --height, --width\n");
    return 2;
  }

  const size_t elem_sz = fp16_io ? sizeof(uint16_t) : sizeof(float);
  const size_t in_bytes = static_cast<size_t>(height) * width * 16u * elem_sz;
  const size_t out_bytes = static_cast<size_t>(height) * width * 4u * elem_sz;

  std::ifstream in_f(input_path, std::ios::binary | std::ios::ate);
  if (!in_f.is_open() || static_cast<size_t>(in_f.tellg()) != in_bytes) {
    std::fprintf(stderr, "error: input tensor size mismatch (expected %zu bytes)\n", in_bytes);
    return 1;
  }
  in_f.seekg(0, std::ios::beg);
  std::vector<uint8_t> in_data(in_bytes);
  in_f.read(reinterpret_cast<char*>(in_data.data()), static_cast<std::streamsize>(in_bytes));

  std::string err;
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  id<MTLCommandQueue> queue = [dev newCommandQueue];
  auto ctx = CreateContextFromArgs(dev, model_path, synthetic_spec, metallib_path, seed, &err);
  if (!ctx) {
    std::fprintf(stderr, "error: %s\n", err.c_str());
    return 1;
  }

  id<MTLBuffer> in_buf =
      [dev newBufferWithBytes:in_data.data() length:in_bytes options:MTLResourceStorageModeShared];
  id<MTLBuffer> out_buf = [dev newBufferWithLength:out_bytes options:MTLResourceStorageModeShared];

  Report rep{};
  auto run_once = [&]() -> bool {
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    FrameTicket ticket;
    if (!ctx->encode_tensor(
            in_buf, fp16_io, height, width, out_buf, fp16_io, cmd, ticket, rep, &err)) {
      return false;
    }
    [cmd commit];
    [cmd waitUntilCompleted];
    if (cmd.status != MTLCommandBufferStatusCompleted) {
      err = "Metal command buffer failed";
      return false;
    }
    rep.gpu_ms = (cmd.GPUEndTime > cmd.GPUStartTime && cmd.GPUStartTime > 0.0)
                     ? (cmd.GPUEndTime - cmd.GPUStartTime) * 1000.0
                     : ticket.gpu_ms();
    return true;
  };

  for (int i = 0; i < warmup; ++i) {
    if (!run_once()) {
      std::fprintf(stderr, "error: %s\n", err.c_str());
      return 1;
    }
  }
  std::vector<double> gpu_times;
  std::vector<double> cpu_times;
  for (int i = 0; i < repeat; ++i) {
    if (!run_once()) {
      std::fprintf(stderr, "error: %s\n", err.c_str());
      return 1;
    }
    gpu_times.push_back(rep.gpu_ms);
    cpu_times.push_back(rep.cpu_encode_ms);
  }

  std::ofstream out_f(output_path, std::ios::binary);
  if (!out_f.is_open()) {
    std::fprintf(stderr, "error: cannot open output path %s\n", output_path.c_str());
    return 1;
  }
  out_f.write(static_cast<const char*>([out_buf contents]), static_cast<std::streamsize>(out_bytes));

  NSDictionary* summary = @{
    @"command" : @"eval-tensor",
    @"device" : [dev name],
    @"topology" : [NSString stringWithUTF8String:rep.topology.c_str()],
    @"weight_format" : [NSString stringWithUTF8String:rep.weight_format.c_str()],
    @"synthetic" : @(rep.synthetic_weights),
    @"block_count" : @(rep.block_count),
    @"dispatch_count" : @(rep.dispatch_count),
    @"shape" : @[ @1, @(height), @(width), @4 ],
    @"gpu_ms_p50" : @(Percentile(gpu_times, 50.0)),
    @"gpu_ms_p95" : @(Percentile(gpu_times, 95.0)),
    @"cpu_encode_ms_p50" : @(Percentile(cpu_times, 50.0)),
  };
  if (!WriteJsonOutput(summary, json_path, &err)) {
    std::fprintf(stderr, "error: %s\n", err.c_str());
    return 1;
  }
  return 0;
}

int RunBench(int argc, char** argv) {
  std::string topo_str = "ht";
  std::string model_path, metallib_path, json_path;
  uint32_t net_w = 0, net_h = 0;
  uint32_t out_w = 0, out_h = 0;
  bool temporal = false;
  bool verify_contracts = false;
  ControlProfile profile = ControlProfile::Standard;
  float detail_strength = 1.0f;
  float colour_strength = 1.0f;
  int warmup = 2;
  int repeat = 5;
  uint64_t seed = 0x4E52464153543031ULL;

  for (int i = 0; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--temporal") { temporal = true; continue; }
    if (arg == "--verify-contracts") { verify_contracts = true; continue; }
    auto need_val = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "error: option %s requires a value\n", arg.c_str());
        std::exit(2);
      }
      return argv[++i];
    };
    if (arg == "--topology") topo_str = need_val();
    else if (arg == "--model") model_path = need_val();
    else if (arg == "--metallib") metallib_path = need_val();
    else if (arg == "--net-size") {
      if (!ParseWxH(need_val(), &net_w, &net_h)) return 2;
    } else if (arg == "--out-size") {
      if (!ParseWxH(need_val(), &out_w, &out_h)) return 2;
    } else if (arg == "--profile") {
      if (!ParseProfile(need_val(), &profile)) return 2;
    } else if (arg == "--detail-strength") detail_strength = std::stof(need_val());
    else if (arg == "--colour-strength") colour_strength = std::stof(need_val());
    else if (arg == "--warmup") warmup = std::stoi(need_val());
    else if (arg == "--repeat") repeat = std::max(1, std::stoi(need_val()));
    else if (arg == "--seed") seed = std::stoull(need_val());
    else if (arg == "--json") json_path = need_val();
    else {
      std::fprintf(stderr, "error: unknown bench argument '%s'\n", arg.c_str());
      return 2;
    }
  }

  if (net_w == 0 || net_h == 0) {
    std::fprintf(stderr, "error: bench requires --net-size WxH\n");
    return 2;
  }
  if (out_w == 0 || out_h == 0) {
    out_w = net_w;
    out_h = net_h;
  }

  std::string err;
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  id<MTLCommandQueue> queue = [dev newCommandQueue];
  auto ctx = CreateContextFromArgs(
      dev, model_path, model_path.empty() ? topo_str : "", metallib_path, seed, &err);
  if (!ctx) {
    std::fprintf(stderr, "error: %s\n", err.c_str());
    return 1;
  }

  id<MTLTexture> color_tex = MakeSharedRGBA32FTexture(dev, out_w, out_h, MTLTextureUsageShaderRead);
  id<MTLTexture> motion_tex = MakeSharedRGBA32FTexture(dev, out_w, out_h, MTLTextureUsageShaderRead);
  id<MTLTexture> output_tex =
      MakeSharedRGBA32FTexture(dev, out_w, out_h, MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite);

  std::vector<float> color_init(static_cast<size_t>(out_w) * out_h * 4u);
  std::vector<float> motion_init(static_cast<size_t>(out_w) * out_h * 4u, 0.0f);
  for (uint32_t y = 0; y < out_h; ++y) {
    for (uint32_t x = 0; x < out_w; ++x) {
      size_t idx = (static_cast<size_t>(y) * out_w + x) * 4u;
      float fx = static_cast<float>(x) / static_cast<float>(out_w);
      float fy = static_cast<float>(y) / static_cast<float>(out_h);
      color_init[idx + 0u] = 0.5f + 0.4f * std::sin(fx * 18.0f + fy * 7.0f);
      color_init[idx + 1u] = 0.5f + 0.4f * std::cos(fx * 11.0f - fy * 15.0f);
      color_init[idx + 2u] = 0.5f + 0.35f * std::sin((fx + fy) * 23.0f);
      color_init[idx + 3u] = 1.0f;
      // Non-zero normalized motion vector (-2 px horizontal, +1 px vertical):
      motion_init[idx + 0u] = -2.0f / static_cast<float>(out_w);
      motion_init[idx + 1u] = 1.0f / static_cast<float>(out_h);
    }
  }
  [color_tex replaceRegion:MTLRegionMake2D(0, 0, out_w, out_h)
               mipmapLevel:0
                 withBytes:color_init.data()
               bytesPerRow:static_cast<size_t>(out_w) * 16u];
  [motion_tex replaceRegion:MTLRegionMake2D(0, 0, out_w, out_h)
                mipmapLevel:0
                  withBytes:motion_init.data()
                bytesPerRow:static_cast<size_t>(out_w) * 16u];

  // Poison output texture with NaNs before execution when verifying contracts.
  if (verify_contracts) {
    std::vector<float> poison(
        static_cast<size_t>(out_w) * out_h * 4u, std::numeric_limits<float>::quiet_NaN());
    [output_tex replaceRegion:MTLRegionMake2D(0, 0, out_w, out_h)
                  mipmapLevel:0
                    withBytes:poison.data()
                  bytesPerRow:static_cast<size_t>(out_w) * 16u];
  }

  FrameInputs fi{};
  fi.color_texture = color_tex;
  fi.motion_texture = temporal ? motion_tex : nil;
  fi.output_texture = output_tex;
  fi.params.output_width = out_w;
  fi.params.output_height = out_h;
  fi.params.network_width = net_w;
  fi.params.network_height = net_h;
  fi.params.geometry_mode =
      (net_w != out_w || net_h != out_h) ? GeometryMode::ScaledNetwork : GeometryMode::VendorAligned;
  fi.params.profile = profile;
  fi.params.detail_strength = detail_strength;
  fi.params.colour_strength = colour_strength;

  bool contracts_ok = true;
  double temporal_l1_delta = 0.0;
  double reset_max_err = 0.0;

  if (verify_contracts) {
    // Frame 0 (reset = true):
    fi.params.frame_index = 0;
    fi.params.reset = true;
    Report r0{};
    if (!ctx->run_sync(fi, queue, r0, &err)) {
      std::fprintf(stderr, "error: %s\n", err.c_str());
      return 1;
    }
    std::vector<float> frame0_out(static_cast<size_t>(out_w) * out_h * 4u);
    [output_tex getBytes:frame0_out.data()
             bytesPerRow:static_cast<size_t>(out_w) * 16u
              fromRegion:MTLRegionMake2D(0, 0, out_w, out_h)
             mipmapLevel:0];

    // Check all pixels finite in [0, 1] (poison overwritten):
    for (size_t i = 0; i < frame0_out.size(); ++i) {
      if (!std::isfinite(frame0_out[i]) || frame0_out[i] < 0.0f || frame0_out[i] > 1.0f) {
        contracts_ok = false;
        break;
      }
    }

    // Check input immutability:
    std::vector<float> color_after(color_init.size());
    [color_tex getBytes:color_after.data()
            bytesPerRow:static_cast<size_t>(out_w) * 16u
             fromRegion:MTLRegionMake2D(0, 0, out_w, out_h)
            mipmapLevel:0];
    if (std::memcmp(color_init.data(), color_after.data(), color_init.size() * sizeof(float)) != 0) {
      contracts_ok = false;
    }

    if (temporal) {
      // Run frames 1..3 with reset = false and non-zero motion:
      for (uint32_t f = 1; f <= 3; ++f) {
        fi.params.frame_index = f;
        fi.params.reset = false;
        if (!ctx->run_sync(fi, queue, r0, &err)) {
          std::fprintf(stderr, "error: %s\n", err.c_str());
          return 1;
        }
      }
      std::vector<float> frame3_out(frame0_out.size());
      [output_tex getBytes:frame3_out.data()
               bytesPerRow:static_cast<size_t>(out_w) * 16u
                fromRegion:MTLRegionMake2D(0, 0, out_w, out_h)
               mipmapLevel:0];
      for (size_t px = 0; px < static_cast<size_t>(out_w) * out_h; ++px) {
        for (size_t c = 0; c < 3; ++c) {
          float v = frame3_out[px * 4u + c];
          if (!std::isfinite(v)) contracts_ok = false;
          temporal_l1_delta += std::abs(static_cast<double>(v - frame0_out[px * 4u + c]));
        }
      }
      temporal_l1_delta /= static_cast<double>(static_cast<size_t>(out_w) * out_h * 3u);
      if (temporal_l1_delta <= 0.0) contracts_ok = false;

      // Reset frame (reset = true, frame_index = 0) must reproduce frame0_out bit-identically:
      fi.params.frame_index = 0;
      fi.params.reset = true;
      if (!ctx->run_sync(fi, queue, r0, &err)) {
        std::fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
      }
      std::vector<float> reset_out(frame0_out.size());
      [output_tex getBytes:reset_out.data()
               bytesPerRow:static_cast<size_t>(out_w) * 16u
                fromRegion:MTLRegionMake2D(0, 0, out_w, out_h)
               mipmapLevel:0];
      for (size_t i = 0; i < frame0_out.size(); ++i) {
        reset_max_err = std::max(reset_max_err, std::abs(static_cast<double>(reset_out[i] - frame0_out[i])));
      }
      if (reset_max_err > 1e-6) contracts_ok = false;
    }

    if (!contracts_ok) {
      std::fprintf(stderr, "error: runtime GPU contract verification failed\n");
      return 1;
    }
  }

  Report rep{};
  ctx->reset_history();
  for (int i = 0; i < warmup; ++i) {
    fi.params.frame_index = static_cast<uint32_t>(i);
    fi.params.reset = (i == 0) || !temporal;
    if (!ctx->run_sync(fi, queue, rep, &err)) {
      std::fprintf(stderr, "error: %s\n", err.c_str());
      return 1;
    }
  }

  std::vector<double> gpu_times;
  std::vector<double> cpu_times;
  for (int i = 0; i < repeat; ++i) {
    fi.params.frame_index = static_cast<uint32_t>(warmup + i);
    fi.params.reset = (!temporal && i == 0);
    if (!temporal) fi.params.reset = true;
    if (!ctx->run_sync(fi, queue, rep, &err)) {
      std::fprintf(stderr, "error: %s\n", err.c_str());
      return 1;
    }
    gpu_times.push_back(rep.gpu_ms);
    cpu_times.push_back(rep.cpu_encode_ms);
  }

  NSDictionary* summary = @{
    @"command" : @"bench",
    @"device" : [dev name],
    @"topology" : [NSString stringWithUTF8String:rep.topology.c_str()],
    @"weight_format" : [NSString stringWithUTF8String:rep.weight_format.c_str()],
    @"synthetic" : @(rep.synthetic_weights),
    @"temporal" : @(temporal),
    @"contracts_verified" : @(verify_contracts && contracts_ok),
    @"temporal_l1_delta" : @(temporal_l1_delta),
    @"reset_max_err" : @(reset_max_err),
    @"block_count" : @(rep.block_count),
    @"dispatch_count" : @(rep.dispatch_count),
    @"network_size" : @[ @(rep.network_width), @(rep.network_height) ],
    @"output_size" : @[ @(rep.output_width), @(rep.output_height) ],
    @"gpu_ms_min" : @(Percentile(gpu_times, 0.0)),
    @"gpu_ms_p50" : @(Percentile(gpu_times, 50.0)),
    @"gpu_ms_p95" : @(Percentile(gpu_times, 95.0)),
    @"cpu_encode_ms_p50" : @(Percentile(cpu_times, 50.0)),
    @"fps_p50" : @(1000.0 / std::max(1e-6, Percentile(gpu_times, 50.0))),
    @"weight_buffer_mb" : @(static_cast<double>(ctx->weight_buffer_bytes()) / (1024.0 * 1024.0)),
    @"scratch_buffer_mb" : @(static_cast<double>(ctx->scratch_buffer_bytes()) / (1024.0 * 1024.0)),
  };
  if (!WriteJsonOutput(summary, json_path, &err)) {
    std::fprintf(stderr, "error: %s\n", err.c_str());
    return 1;
  }
  return 0;
}

int RunValidateModel(int argc, char** argv) {
  std::string model_path, json_path;
  bool reject_symlinks = false;
  bool verify_sha256 = true;
  bool check_e4m3_lut = false;

  for (int i = 0; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--reject-symlinks") { reject_symlinks = true; continue; }
    if (arg == "--no-sha256") { verify_sha256 = false; continue; }
    if (arg == "--check-e4m3-lut") { check_e4m3_lut = true; continue; }
    auto need_val = [&]() -> std::string {
      if (i + 1 >= argc) std::exit(2);
      return argv[++i];
    };
    if (arg == "--model") model_path = need_val();
    else if (arg == "--json") json_path = need_val();
    else {
      std::fprintf(stderr, "error: unknown validate-model argument '%s'\n", arg.c_str());
      return 2;
    }
  }

  if (check_e4m3_lut) {
    // Verify all finite non-NaN E4M3 codes (0x00..0x7E, 0x80..0xFE) round-trip cleanly.
    for (int b = 0; b < 256; ++b) {
      if ((b & 0x7f) == 0x7f) continue;  // 0x7f / 0xff are NaN in E4M3
      uint16_t fp16 = DecodeNativeE4M3ToFp16Bits(static_cast<uint8_t>(b));
      uint8_t re = EncodeFp16BitsToNativeE4M3(fp16);
      if ((b == 0x80 && re == 0x80) || re == static_cast<uint8_t>(b)) {
        continue;
      }
      std::fprintf(stderr, "error: E4M3 round-trip mismatch at byte 0x%02x -> 0x%02x\n", b, re);
      return 1;
    }
  }

  if (model_path.empty() && check_e4m3_lut) {
    std::printf("{\"e4m3_lut_roundtrip\":\"pass\"}\n");
    return 0;
  }

  SafeTensorSummary sum{};
  std::string err;
  if (!ValidateModelFile(model_path, &sum, &err, reject_symlinks, verify_sha256)) {
    std::fprintf(stderr, "error: %s\n", err.c_str());
    return 1;
  }

  NSDictionary* out = @{
    @"status" : @"valid",
    @"tensor_count" : @(sum.tensor_count),
    @"f16_count" : @(sum.f16_count),
    @"f32_count" : @(sum.f32_count),
    @"fp8_e4m3_count" : @(sum.fp8_e4m3_count),
    @"topology" : [NSString stringWithUTF8String:sum.topology.name.c_str()],
    @"spec" : [NSString stringWithUTF8String:sum.topology.spec_string.c_str()],
    @"total_blocks" : @(sum.topology.total_blocks),
    @"blend_scale" : @(sum.topology.blend_scale),
    @"source_format" : [NSString stringWithUTF8String:sum.source_format.c_str()],
  };
  if (!WriteJsonOutput(out, json_path, &err)) {
    std::fprintf(stderr, "error: %s\n", err.c_str());
    return 1;
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  setenv("AGX_RELAX_CDM_CTXSTORE_TIMEOUT", "1", 0);
  @autoreleasepool {
    if (argc < 2) {
      std::fprintf(
          stderr,
          "usage: netal-nr-fast-cli <render-image | eval-tensor | bench | validate-model> [options]\n");
      return 2;
    }
    std::string cmd = argv[1];
    if (cmd == "render-image") return RunRenderImage(argc - 2, argv + 2);
    if (cmd == "eval-tensor") return RunEvalTensor(argc - 2, argv + 2);
    if (cmd == "bench") return RunBench(argc - 2, argv + 2);
    if (cmd == "validate-model") return RunValidateModel(argc - 2, argv + 2);
    std::fprintf(stderr, "error: unknown command '%s'\n", cmd.c_str());
    return 2;
  }
}
