// Metal 4 gate for the FG runtime: the accepted 16-frame synthetic sequence, encoded through the
// Metal 4 shim into MTL4 command buffers on an MTL4 queue, must produce the accepted output hashes.
//   fg-runtime-metal4-probe <libs> <model> <rom> <sigmoid> <ex2-a> <ex2-b> <expected-json|-> <frames>
#import <CommonCrypto/CommonDigest.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "fg_runtime.h"
#include "../fg_host/synthetic_contract.h"

#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;

std::string Sha256(const std::vector<std::uint8_t>& bytes) {
  std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
  CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size()), digest.data());
  std::ostringstream result;
  result << std::hex << std::setfill('0');
  for (const auto byte : digest) result << std::setw(2) << unsigned(byte);
  return result.str();
}

id<MTLTexture> Texture(id<MTLDevice> device, MTLPixelFormat format, MTLTextureUsage usage,
                       const std::vector<std::uint8_t>& bytes, std::size_t bpp) {
  auto* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:640 height:360 mipmapped:NO];
  d.storageMode = MTLStorageModeShared;
  d.usage = usage;
  id<MTLTexture> t = [device newTextureWithDescriptor:d];
  if (t == nil) throw std::runtime_error("texture allocation");
  [t replaceRegion:MTLRegionMake2D(0, 0, 640, 360) mipmapLevel:0 withBytes:bytes.data() bytesPerRow:640 * bpp];
  return t;
}

std::vector<std::uint8_t> TextureBytes(id<MTLTexture> t, std::size_t bpp) {
  std::vector<std::uint8_t> r(640u * 360u * bpp);
  [t getBytes:r.data() bytesPerRow:640 * bpp fromRegion:MTLRegionMake2D(0, 0, 640, 360) mipmapLevel:0];
  return r;
}

template <typename T>
std::vector<std::uint8_t> Bytes(const std::vector<T>& v) {
  std::vector<std::uint8_t> r(v.size() * sizeof(T));
  std::memcpy(r.data(), v.data(), r.size());
  return r;
}

void Require(NSFGStatus s, const char* op, NSFGSession* session = nullptr) {
  if (s == NSFG_STATUS_OK) return;
  std::string m = std::string(op) + ": " + NSFGStatusName(s);
  if (session) m += std::string(": ") + NSFGSessionLastError(session);
  throw std::runtime_error(m);
}

std::vector<std::string> ExpectedHashes(const char* path) {
  std::vector<std::string> out;
  if (std::strcmp(path, "-") == 0) return out;
  std::ifstream in(path);
  std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  const auto key = text.find("\"output_sha256\"");
  if (key == std::string::npos) throw std::runtime_error("expected receipt has no output_sha256");
  auto pos = text.find('[', key);
  const auto end = text.find(']', pos);
  while (true) {
    const auto q1 = text.find('"', pos);
    if (q1 == std::string::npos || q1 > end) break;
    const auto q2 = text.find('"', q1 + 1);
    out.push_back(text.substr(q1 + 1, q2 - q1 - 1));
    pos = q2 + 1;
  }
  return out;
}
}  // namespace

int main(int argc, const char* argv[]) {
  @autoreleasepool {
    if (argc != 9) {
      std::cerr << "usage=fg-runtime-metal4-probe <libs> <model> <rom> <sigmoid> <ex2-a> <ex2-b> <expected-json|-> <passes>\n";
      return 2;
    }
    try {
      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      id<MTL4CommandQueue> queue = [device newMTL4CommandQueue];
      id<MTL4CommandAllocator> allocator = [device newCommandAllocator];
      id<MTLSharedEvent> event = [device newSharedEvent];
      if (device == nil || queue == nil || allocator == nil || event == nil)
        throw std::runtime_error("Metal 4 unavailable");
      const int passes = std::atoi(argv[8]);
      MTL4CommitOptions* commit_options = [[MTL4CommitOptions alloc] init];
      [commit_options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
        if (feedback.error != nil)
          std::cerr << "gpu_error=" << feedback.error.localizedDescription.UTF8String << '\n';
        else if (std::getenv("NSFG_M4_DEBUG") != nullptr)
          std::cerr << "gpu_ms=" << (feedback.GPUEndTime - feedback.GPUStartTime) * 1000.0 << '\n';
      }];

      std::array<neuralstudio::fg::SyntheticFrame, 16> frames;
      std::array<id<MTLTexture>, 16> color{}, motion{}, depth{};
      for (std::uint32_t f = 0; f < 16; ++f) {
        frames[f] = neuralstudio::fg::MakeSyntheticFrame(640, 360, f);
        color[f] = Texture(device, MTLPixelFormatRGBA8Unorm, MTLTextureUsageShaderRead, frames[f].backbuffer_rgba8, 4);
        motion[f] = Texture(device, MTLPixelFormatRG16Float, MTLTextureUsageShaderRead, Bytes(frames[f].motion_rg16f), 4);
        depth[f] = Texture(device, MTLPixelFormatR32Float, MTLTextureUsageShaderRead, Bytes(frames[f].depth_r32f), 4);
      }
      const std::vector<std::uint8_t> poison(640u * 360u * 4u, 0xa5);
      std::array<id<MTLTexture>, 15> targets{};
      for (auto& t : targets)
        t = Texture(device, MTLPixelFormatRGBA8Unorm, MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite, poison, 4);

      NSFGCreateInfo create{};
      create.metal_device = (__bridge void*)device;
      create.metal_command_queue = nullptr;
      create.metallib_directory = argv[1];
      create.model_directory = argv[2];
      create.rcp_rom_path = argv[3];
      create.sigmoid_lut_path = argv[4];
      create.ex2_unit_table_path = argv[5];
      create.ex2_fraction_table_path = argv[6];
      create.width = 640;
      create.height = 360;
      create.ex2_mode = NSFG_EX2_COMPACT;
      create.auxiliary_mode = NSFG_AUXILIARY_LIVE_EARLY_RESOURCES;
      create.poison_byte = 0xa5;
      create.metal4 = true;
      NSFGSession* session = nullptr;
      Require(NSFGSessionCreate(&create, &session), "create");

      uint64_t signal = 0;
      auto run_frame = [&](const NSFGFrameInfo& info, NSFGFrameResult* result) {
        id<MTL4CommandBuffer> cb = [device newCommandBuffer];
        [cb beginCommandBufferWithAllocator:allocator];
        const auto t0 = Clock::now();
        Require(NSFGSessionEvaluateMetal4(session, &info, (__bridge void*)cb, result), "evaluate", session);
        const double encode_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        [cb endCommandBuffer];
        id<MTL4CommandBuffer> list[1] = {cb};
        [queue commit:list count:1 options:commit_options];
        [queue signalEvent:event value:++signal];
        if (![event waitUntilSignaledValue:signal timeoutMS:20000]) throw std::runtime_error("GPU timeout");
        const double total_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        [allocator reset];
        return std::make_pair(encode_ms, total_ms);
      };

      const auto expected = ExpectedHashes(argv[7]);
      bool all_ok = true;
      for (int pass = 0; pass < passes; ++pass) {
        if (pass > 0) Require(NSFGSessionInvalidate(session), "invalidate", session);
        NSFGFrameInfo reset{};
        reset.current_backbuffer_rgba8 = (__bridge void*)color[0];
        reset.previous_backbuffer_rgba8 = (__bridge void*)color[0];
        reset.motion_rg16f = (__bridge void*)motion[0];
        reset.depth_r32f = (__bridge void*)depth[0];
        reset.frame_id = 0;
        reset.reset = true;
        run_frame(reset, nullptr);
        std::vector<double> encode, total;
        int mismatches = 0;
        for (std::uint64_t f = 1; f < 16; ++f) {
          NSFGFrameInfo in{};
          in.current_backbuffer_rgba8 = (__bridge void*)color[f];
          in.previous_backbuffer_rgba8 = (__bridge void*)color[f - 1];
          in.motion_rg16f = (__bridge void*)motion[f];
          in.depth_r32f = (__bridge void*)depth[f];
          in.generated_target_rgba8 = (__bridge void*)targets[f - 1];
          in.frame_id = f;
          NSFGFrameResult r{};
          const auto ms = run_frame(in, &r);
          encode.push_back(ms.first);
          total.push_back(ms.second);
          const auto hash = Sha256(TextureBytes(targets[f - 1], 4));
          if (!expected.empty() && (f - 1 >= expected.size() || expected[f - 1] != hash)) {
            ++mismatches;
            std::cout << "frame=" << f << " hash=" << hash << " expected="
                      << (f - 1 < expected.size() ? expected[f - 1] : "?") << '\n';
          }
        }
        std::sort(encode.begin(), encode.end());
        std::sort(total.begin(), total.end());
        std::cout << "pass=" << pass << " frames=15 mismatches=" << mismatches
                  << " encode_p50_ms=" << std::fixed << std::setprecision(2) << encode[7]
                  << " commit_to_complete_p50_ms=" << total[7] << '\n';
        if (mismatches) all_ok = false;
      }
      NSFGSessionDestroy(session);
      std::cout << "result=" << (all_ok ? "pass" : "fail") << '\n';
      return all_ok ? 0 : 3;
    } catch (const std::exception& e) {
      std::cerr << "error=" << e.what() << '\n';
      return 1;
    }
  }
}
