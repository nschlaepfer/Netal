#import <CommonCrypto/CommonDigest.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "fg_runtime.h"

#include <array>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

#include "../fg_host/synthetic_contract.h"

namespace {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

struct SequenceResult {
  std::vector<std::string> output_sha256;
  std::vector<double> evaluate_wall_ms;
  std::vector<std::uint64_t> allocation_count;
  std::vector<std::vector<NSFGCommandSpan>> command_spans;
  double sequence_wall_ms = 0.0;
};

std::string Sha256(const std::vector<std::uint8_t>& bytes) {
  std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
  CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size()), digest.data());
  std::ostringstream result;
  result << std::hex << std::setfill('0');
  for (const auto byte : digest) result << std::setw(2) << unsigned(byte);
  return result.str();
}

id<MTLTexture> Texture(id<MTLDevice> device, MTLPixelFormat format,
                       MTLTextureUsage usage,
                       const std::vector<std::uint8_t>& bytes,
                       std::size_t bytes_per_pixel) {
  if (bytes.size() != 640u * 360u * bytes_per_pixel)
    throw std::runtime_error("benchmark texture size drift");
  auto* descriptor = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:format
                                    width:640
                                   height:360
                                mipmapped:NO];
  descriptor.storageMode = MTLStorageModeShared;
  descriptor.usage = usage;
  id<MTLTexture> result = [device newTextureWithDescriptor:descriptor];
  if (result == nil) throw std::runtime_error("benchmark texture allocation");
  [result replaceRegion:MTLRegionMake2D(0, 0, 640, 360)
             mipmapLevel:0
               withBytes:bytes.data()
             bytesPerRow:640 * bytes_per_pixel];
  return result;
}

std::vector<std::uint8_t> TextureBytes(id<MTLTexture> texture,
                                       std::size_t bytes_per_pixel) {
  std::vector<std::uint8_t> result(640u * 360u * bytes_per_pixel);
  [texture getBytes:result.data()
         bytesPerRow:640 * bytes_per_pixel
          fromRegion:MTLRegionMake2D(0, 0, 640, 360)
         mipmapLevel:0];
  return result;
}

template <typename T>
std::vector<std::uint8_t> Bytes(const std::vector<T>& values) {
  std::vector<std::uint8_t> result(values.size() * sizeof(T));
  std::memcpy(result.data(), values.data(), result.size());
  return result;
}

void Require(NSFGStatus status, const char* operation,
             NSFGSession* session = nullptr) {
  if (status == NSFG_STATUS_OK) return;
  std::string message = std::string(operation) + ": " + NSFGStatusName(status);
  if (session != nullptr)
    message += std::string(": ") + NSFGSessionLastError(session);
  throw std::runtime_error(message);
}

SequenceResult RunSequence(
    NSFGSession* session,
    const std::array<id<MTLTexture>, 16>& color,
    const std::array<id<MTLTexture>, 16>& motion,
    const std::array<id<MTLTexture>, 16>& depth,
    const std::array<id<MTLTexture>, 15>& targets,
    bool invalidate_first, bool collect_timing) {
  if (invalidate_first)
    Require(NSFGSessionInvalidate(session), "invalidate", session);
  NSFGFrameInfo reset{};
  reset.current_backbuffer_rgba8 = (__bridge void*)color[0];
  reset.previous_backbuffer_rgba8 = (__bridge void*)color[0];
  reset.motion_rg16f = (__bridge void*)motion[0];
  reset.depth_r32f = (__bridge void*)depth[0];
  reset.frame_id = 0;
  reset.reset = true;
  Require(NSFGSessionEvaluate(session, &reset, nullptr), "reset", session);
  SequenceResult result;
  result.output_sha256.reserve(15);
  result.evaluate_wall_ms.reserve(15);
  result.allocation_count.reserve(15);
  result.command_spans.reserve(15);
  const auto sequence_start = Clock::now();
  for (std::uint64_t frame = 1; frame < 16; ++frame) {
    NSFGFrameInfo input{};
    input.current_backbuffer_rgba8 = (__bridge void*)color[frame];
    input.previous_backbuffer_rgba8 = (__bridge void*)color[frame - 1];
    input.motion_rg16f = (__bridge void*)motion[frame];
    input.depth_r32f = (__bridge void*)depth[frame];
    input.generated_target_rgba8 = (__bridge void*)targets[frame - 1];
    input.frame_id = frame;
    NSFGFrameResult frame_result{};
    const auto start = Clock::now();
    Require(NSFGSessionEvaluate(session, &input, &frame_result),
            "evaluate", session);
    const auto end = Clock::now();
    result.evaluate_wall_ms.push_back(collect_timing
        ? std::chrono::duration<double, std::milli>(end - start).count()
        : 0.0);
    result.allocation_count.push_back(
        frame_result.resource_allocations_during_evaluate);
    std::vector<NSFGCommandSpan> spans;
    if (collect_timing) {
      const auto span_count = NSFGSessionLastCommandSpanCount(session);
      spans.resize(span_count);
      for (std::size_t index = 0; index < span_count; ++index)
        Require(NSFGSessionCopyLastCommandSpan(session, index, &spans[index]),
                "copy command span", session);
    }
    result.command_spans.push_back(std::move(spans));
    result.output_sha256.push_back(Sha256(TextureBytes(targets[frame - 1], 4)));
  }
  result.sequence_wall_ms = std::chrono::duration<double, std::milli>(
      Clock::now() - sequence_start).count();
  return result;
}

std::string JsonArray(const std::vector<std::string>& values) {
  std::ostringstream result;
  result << '[';
  for (std::size_t index = 0; index < values.size(); ++index)
    result << (index ? ", " : "") << '"' << values[index] << '"';
  result << ']';
  return result.str();
}

template <typename T>
std::string NumberArray(const std::vector<T>& values) {
  std::ostringstream result;
  result << '[' << std::setprecision(17);
  for (std::size_t index = 0; index < values.size(); ++index)
    result << (index ? ", " : "") << values[index];
  result << ']';
  return result.str();
}

std::string SpanArray(
    const std::vector<std::vector<NSFGCommandSpan>>& frames) {
  std::ostringstream result;
  result << '[' << std::setprecision(17);
  for (std::size_t frame = 0; frame < frames.size(); ++frame) {
    result << (frame ? "," : "") << "\n    [";
    for (std::size_t index = 0; index < frames[frame].size(); ++index) {
      const auto& span = frames[frame][index];
      result << (index ? "," : "")
             << "{\"domain\":" << static_cast<unsigned>(span.domain)
             << ",\"ordinal\":" << span.ordinal
             << ",\"cpu_commit_wait_ms\":" << span.cpu_commit_wait_ms
             << ",\"gpu_ms\":" << span.gpu_ms << '}';
    }
    result << ']';
  }
  result << "\n  ]";
  return result.str();
}

}  // namespace

int main(int argc, const char* argv[]) {
  @autoreleasepool {
    if (argc != 12) {
      std::cerr << "usage=fg-runtime-benchmark <libs> <model> <rom> <sigmoid> "
                   "<ex2-a> <ex2-b> <output-json> <run-id> <poison> "
                   "<reference|compact> <internal|live>\n";
      return 2;
    }
    try {
      const std::string mode = argv[10];
      if (mode != "reference" && mode != "compact")
        throw std::runtime_error("invalid EX2 mode");
      const std::string auxiliary_mode = argv[11];
      if (auxiliary_mode != "internal" && auxiliary_mode != "live")
        throw std::runtime_error("invalid auxiliary mode");
      const auto poison_value = std::stoul(argv[9], nullptr, 16);
      if (poison_value > 255u) throw std::runtime_error("invalid poison");
      const auto poison = static_cast<std::uint8_t>(poison_value);
      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      id<MTLCommandQueue> queue = [device newCommandQueue];
      if (device == nil || queue == nil) throw std::runtime_error("Metal unavailable");

      std::array<neuralstudio::fg::SyntheticFrame, 16> frames;
      std::array<std::vector<std::uint8_t>, 16> motion_bytes;
      std::array<std::vector<std::uint8_t>, 16> depth_bytes;
      std::array<id<MTLTexture>, 16> color{};
      std::array<id<MTLTexture>, 16> motion{};
      std::array<id<MTLTexture>, 16> depth{};
      for (std::uint32_t frame = 0; frame < 16; ++frame) {
        frames[frame] = neuralstudio::fg::MakeSyntheticFrame(640, 360, frame);
        motion_bytes[frame] = Bytes(frames[frame].motion_rg16f);
        depth_bytes[frame] = Bytes(frames[frame].depth_r32f);
        color[frame] = Texture(device, MTLPixelFormatRGBA8Unorm,
                               MTLTextureUsageShaderRead,
                               frames[frame].backbuffer_rgba8, 4);
        motion[frame] = Texture(device, MTLPixelFormatRG16Float,
                                MTLTextureUsageShaderRead,
                                motion_bytes[frame], 4);
        depth[frame] = Texture(device, MTLPixelFormatR32Float,
                               MTLTextureUsageShaderRead,
                               depth_bytes[frame], 4);
      }
      const std::vector<std::uint8_t> output_initial(640u * 360u * 4u, poison);
      std::array<id<MTLTexture>, 15> targets{};
      for (auto& target : targets)
        target = Texture(device, MTLPixelFormatRGBA8Unorm,
                         MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
                         output_initial, 4);

      NSFGCreateInfo create{};
      create.metal_device = (__bridge void*)device;
      create.metal_command_queue = (__bridge void*)queue;
      create.metallib_directory = argv[1];
      create.model_directory = argv[2];
      create.rcp_rom_path = argv[3];
      create.sigmoid_lut_path = argv[4];
      create.ex2_unit_table_path = argv[5];
      create.ex2_fraction_table_path = argv[6];
      create.width = 640;
      create.height = 360;
      create.ex2_mode = mode == "compact" ? NSFG_EX2_COMPACT
                                           : NSFG_EX2_REFERENCE;
      create.auxiliary_mode = auxiliary_mode == "live"
          ? NSFG_AUXILIARY_LIVE_EARLY_RESOURCES
          : NSFG_AUXILIARY_INTERNAL_SMALL_FLOW;
      create.poison_byte = poison;
      create.enable_audit = false;
      const bool profiling_enabled =
          std::getenv("FG_RUNTIME_PROFILE_SPANS") != nullptr;
      create.enable_profiling = profiling_enabled;
      const auto create_start = Clock::now();
      NSFGSession* session = nullptr;
      Require(NSFGSessionCreate(&create, &session), "create");
      const auto create_ms = std::chrono::duration<double, std::milli>(
          Clock::now() - create_start).count();

      const auto pre = RunSequence(session, color, motion, depth, targets,
                                   false, false);
      const auto measured = RunSequence(session, color, motion, depth, targets,
                                        true, true);
      const auto post = RunSequence(session, color, motion, depth, targets,
                                    true, false);
      NSFGSessionStats stats{};
      Require(NSFGSessionGetStats(session, &stats), "stats", session);
      NSFGSessionDestroy(session);

      std::size_t input_mismatches = 0;
      for (std::size_t frame = 0; frame < 16; ++frame) {
        input_mismatches +=
            TextureBytes(color[frame], 4) != frames[frame].backbuffer_rgba8;
        input_mismatches += TextureBytes(motion[frame], 4) != motion_bytes[frame];
        input_mismatches += TextureBytes(depth[frame], 4) != depth_bytes[frame];
      }
      const bool comparable = pre.output_sha256 == measured.output_sha256 &&
                              measured.output_sha256 == post.output_sha256;
      const bool no_measured_allocations = std::accumulate(
          measured.allocation_count.begin(), measured.allocation_count.end(),
          std::uint64_t{0}) == 0;
      const bool pass = comparable && no_measured_allocations &&
                        input_mismatches == 0 &&
                        stats.input_mismatched_bytes == 0 &&
                        stats.canary_mismatched_bytes == 0;
      const auto thermal_state = static_cast<long>(
          [NSProcessInfo processInfo].thermalState);
      const auto os = [NSProcessInfo processInfo].operatingSystemVersionString;
      const auto* output_path = argv[7];
      std::ofstream receipt(output_path);
      if (!receipt) throw std::runtime_error("unable to create benchmark receipt");
      receipt << std::setprecision(17)
              << "{\n  \"schema\": \"neuralstudio.fg.runtime-benchmark-run.v1\",\n"
              << "  \"run_id\": \"" << argv[8] << "\",\n"
              << "  \"process_id\": " << getpid() << ",\n"
              << "  \"ex2_mode\": \"" << mode << "\",\n"
              << "  \"auxiliary_mode\": \"" << auxiliary_mode << "\",\n"
              << "  \"poison_byte\": " << unsigned(poison) << ",\n"
              << "  \"device_name\": \"" << device.name.UTF8String << "\",\n"
              << "  \"os_version\": \"" << os.UTF8String << "\",\n"
              << "  \"thermal_state_after\": " << thermal_state << ",\n"
              << "  \"power_mode\": \"not-observed\",\n"
              << "  \"profile\": \"640x360-accepted-internal-small-flow\",\n"
              << "  \"timing_class\": \"synchronous-evaluate-wall-not-input-latency\",\n"
              << "  \"counter_attachment\": false,\n"
              << "  \"profiling_enabled\": "
              << (profiling_enabled ? "true" : "false") << ",\n"
              << "  \"warmup_generated_frames\": 15,\n"
              << "  \"measured_generated_frames\": 15,\n"
              << "  \"postcheck_generated_frames\": 15,\n"
              << "  \"session_create_wall_ms\": " << create_ms << ",\n"
              << "  \"warmup_sequence_wall_ms\": " << pre.sequence_wall_ms << ",\n"
              << "  \"measured_evaluate_wall_ms\": "
              << NumberArray(measured.evaluate_wall_ms) << ",\n"
              << "  \"measured_metal_resource_allocations\": "
              << NumberArray(measured.allocation_count) << ",\n"
              << "  \"measured_command_spans\": "
              << SpanArray(measured.command_spans) << ",\n"
              << "  \"pre_output_sha256\": " << JsonArray(pre.output_sha256) << ",\n"
              << "  \"measured_output_sha256\": "
              << JsonArray(measured.output_sha256) << ",\n"
              << "  \"post_output_sha256\": " << JsonArray(post.output_sha256) << ",\n"
              << "  \"output_hashes_comparable\": "
              << (comparable ? "true" : "false") << ",\n"
              << "  \"caller_owned_input_mismatches\": " << input_mismatches << ",\n"
              << "  \"expected_or_archived_intermediate_ingress\": false,\n"
              << "  \"address_fields_recorded\": false,\n"
              << "  \"result\": \"" << (pass ? "pass" : "fail") << "\"\n}\n";
      std::cout << "mode=" << mode << '\n'
                << "measured_frames=15\n"
                << "output_hashes_comparable=" << (comparable ? "true" : "false") << '\n'
                << "result=" << (pass ? "pass" : "fail") << '\n';
      return pass ? 0 : 3;
    } catch (const std::exception& error) {
      std::cerr << "error=" << error.what() << '\n';
      return 1;
    }
  }
}
