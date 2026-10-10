#import <CommonCrypto/CommonDigest.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "fg_runtime.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

#include "../fg_host/synthetic_contract.h"

namespace {

namespace fs = std::filesystem;

std::string Sha256(const std::vector<std::uint8_t>& bytes) {
  std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
  CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size()), digest.data());
  std::ostringstream result;
  result << std::hex << std::setfill('0');
  for (const auto byte : digest) result << std::setw(2) << unsigned(byte);
  return result.str();
}

void Write(const fs::path& path, const std::vector<std::uint8_t>& bytes) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("unable to write " + path.string());
  stream.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
}

id<MTLTexture> Texture(id<MTLDevice> device, MTLPixelFormat format,
                       std::uint32_t width, std::uint32_t height,
                       MTLTextureUsage usage,
                       const std::vector<std::uint8_t>& bytes,
                       std::size_t bytes_per_pixel) {
  if (bytes.size() != std::size_t{width} * height * bytes_per_pixel)
    throw std::runtime_error("offline texture size drift");
  auto* descriptor = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:format
                                    width:width
                                   height:height
                                mipmapped:NO];
  descriptor.storageMode = MTLStorageModeShared;
  descriptor.usage = usage;
  id<MTLTexture> result = [device newTextureWithDescriptor:descriptor];
  if (result == nil) throw std::runtime_error("offline texture allocation");
  [result replaceRegion:MTLRegionMake2D(0, 0, width, height)
             mipmapLevel:0
               withBytes:bytes.data()
             bytesPerRow:width * bytes_per_pixel];
  return result;
}

std::vector<std::uint8_t> TextureBytes(id<MTLTexture> texture,
                                       std::size_t bytes_per_pixel) {
  std::vector<std::uint8_t> result(texture.width * texture.height *
                                   bytes_per_pixel);
  [texture getBytes:result.data()
         bytesPerRow:texture.width * bytes_per_pixel
          fromRegion:MTLRegionMake2D(0, 0, texture.width, texture.height)
         mipmapLevel:0];
  return result;
}

std::vector<std::uint8_t> MotionBytes(
    const std::vector<std::uint16_t>& motion) {
  std::vector<std::uint8_t> result(motion.size() * sizeof(motion[0]));
  std::memcpy(result.data(), motion.data(), result.size());
  return result;
}

std::vector<std::uint8_t> DepthBytes(const std::vector<float>& depth) {
  std::vector<std::uint8_t> result(depth.size() * sizeof(depth[0]));
  std::memcpy(result.data(), depth.data(), result.size());
  return result;
}

void RequireStatus(NSFGStatus actual, NSFGStatus expected,
                   const char* operation) {
  if (actual != expected)
    throw std::runtime_error(std::string(operation) + ": expected " +
                             NSFGStatusName(expected) + ", got " +
                             NSFGStatusName(actual));
}

void AppendCheckpoints(
    NSFGSession* session, const std::string& phase,
    std::uint32_t first_launch, std::size_t expected_count,
    std::vector<std::pair<std::string, std::string>>* checkpoints) {
  const auto count = NSFGSessionLastCheckpointCount(session);
  if (count != expected_count)
    throw std::runtime_error("runtime checkpoint count drift: " + phase);
  for (std::size_t index = 0; index < count; ++index) {
    std::array<char, 65> hash{};
    RequireStatus(NSFGSessionCopyLastCheckpoint(
                      session, index, hash.data(), hash.size()),
                  NSFG_STATUS_OK, "copy runtime checkpoint");
    std::ostringstream key;
    key << phase << "-launch-" << std::setfill('0') << std::setw(6)
        << (first_launch + index);
    checkpoints->emplace_back(key.str(), hash.data());
  }
}

}  // namespace

int main(int argc, const char* argv[]) {
  @autoreleasepool {
    if (argc != 12) {
      std::cerr << "usage=fg-runtime-offline <libs> <model> <rom> <sigmoid> "
                   "<ex2-unit> <ex2-fraction> <output> <run-id> <poison> "
                   "<reference|compact> <internal|live>\n";
      return 2;
    }
    try {
      const fs::path output = argv[7];
      fs::create_directories(output);
      const auto parsed = std::stoul(argv[9], nullptr, 16);
      if (parsed > 255u) throw std::runtime_error("invalid poison");
      const auto poison = static_cast<std::uint8_t>(parsed);
      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      id<MTLCommandQueue> queue = [device newCommandQueue];
      if (device == nil || queue == nil) throw std::runtime_error("Metal unavailable");

      std::array<neuralstudio::fg::SyntheticFrame, 16> frames;
      std::array<id<MTLTexture>, 16> color{};
      std::array<id<MTLTexture>, 16> motion{};
      std::array<id<MTLTexture>, 16> depth{};
      std::array<std::vector<std::uint8_t>, 16> motion_bytes;
      std::array<std::vector<std::uint8_t>, 16> depth_bytes;
      for (std::uint32_t frame = 0; frame < frames.size(); ++frame) {
        frames[frame] = neuralstudio::fg::MakeSyntheticFrame(640, 360, frame);
        motion_bytes[frame] = MotionBytes(frames[frame].motion_rg16f);
        depth_bytes[frame] = DepthBytes(frames[frame].depth_r32f);
        color[frame] = Texture(device, MTLPixelFormatRGBA8Unorm, 640, 360,
                               MTLTextureUsageShaderRead,
                               frames[frame].backbuffer_rgba8, 4);
        motion[frame] = Texture(device, MTLPixelFormatRG16Float, 640, 360,
                                MTLTextureUsageShaderRead, motion_bytes[frame], 4);
        depth[frame] = Texture(device, MTLPixelFormatR32Float, 640, 360,
                               MTLTextureUsageShaderRead, depth_bytes[frame], 4);
      }
      const std::vector<std::uint8_t> output_initial(640u * 360u * 4u, poison);
      std::array<id<MTLTexture>, 15> targets{};
      for (auto& target : targets)
        target = Texture(device, MTLPixelFormatRGBA8Unorm, 640, 360,
                         MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
                         output_initial, 4);
      id<MTLTexture> replay_target = Texture(
          device, MTLPixelFormatRGBA8Unorm, 640, 360,
          MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
          output_initial, 4);
      id<MTLTexture> fast_target_a = Texture(
          device, MTLPixelFormatRGBA8Unorm, 640, 360,
          MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
          output_initial, 4);
      id<MTLTexture> fast_target_b = Texture(
          device, MTLPixelFormatRGBA8Unorm, 640, 360,
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
      if (std::string(argv[10]) == "reference")
        create.ex2_mode = NSFG_EX2_REFERENCE;
      else if (std::string(argv[10]) == "compact")
        create.ex2_mode = NSFG_EX2_COMPACT;
      else
        throw std::runtime_error("invalid EX2 mode");
      if (std::string(argv[11]) == "internal")
        create.auxiliary_mode = NSFG_AUXILIARY_INTERNAL_SMALL_FLOW;
      else if (std::string(argv[11]) == "live")
        create.auxiliary_mode = NSFG_AUXILIARY_LIVE_EARLY_RESOURCES;
      else
        throw std::runtime_error("invalid auxiliary mode");
      create.poison_byte = poison;
      create.enable_audit = true;
      NSFGCreateInfo invalid_create = create;
      invalid_create.width = 1280;
      NSFGSession* invalid_session = nullptr;
      RequireStatus(NSFGSessionCreate(&invalid_create, &invalid_session),
                    NSFG_STATUS_UNSUPPORTED_PROFILE,
                    "unsupported create rejection");
      if (invalid_session != nullptr)
        throw std::runtime_error("unsupported create returned a session");
      NSFGSession* session = nullptr;
      RequireStatus(NSFGSessionCreate(&create, &session), NSFG_STATUS_OK,
                    "create");
      NSFGSession* duplicate_session = nullptr;
      RequireStatus(NSFGSessionCreate(&create, &duplicate_session),
                    NSFG_STATUS_BUSY, "process-session gate");
      if (duplicate_session != nullptr)
        throw std::runtime_error("busy create returned a session");

      const auto negative_before = TextureBytes(targets[0], 4);
      NSFGFrameInfo pre_reset{};
      pre_reset.current_backbuffer_rgba8 = (__bridge void*)color[1];
      pre_reset.previous_backbuffer_rgba8 = (__bridge void*)color[0];
      pre_reset.motion_rg16f = (__bridge void*)motion[1];
      pre_reset.depth_r32f = (__bridge void*)depth[1];
      pre_reset.generated_target_rgba8 = (__bridge void*)targets[0];
      pre_reset.frame_id = 1;
      RequireStatus(NSFGSessionEvaluate(session, &pre_reset, nullptr),
                    NSFG_STATUS_NEEDS_RESET, "pre-reset rejection");
      if (TextureBytes(targets[0], 4) != negative_before)
        throw std::runtime_error("rejected pre-reset evaluation touched output");

      std::vector<std::pair<std::string, std::string>> checkpoints;
      checkpoints.reserve(1149);
      NSFGFrameInfo reset{};
      reset.current_backbuffer_rgba8 = (__bridge void*)color[0];
      reset.previous_backbuffer_rgba8 = (__bridge void*)color[0];
      reset.motion_rg16f = (__bridge void*)motion[0];
      reset.depth_r32f = (__bridge void*)depth[0];
      reset.frame_id = 0;
      reset.reset = true;
      NSFGFrameResult reset_result{};
      RequireStatus(NSFGSessionEvaluate(session, &reset, &reset_result),
                    NSFG_STATUS_OK, "reset");
      if (reset_result.generated_output_valid ||
          reset_result.selected_launch_count != 9)
        throw std::runtime_error("reset result contract mismatch");
      AppendCheckpoints(session, "reset", 3, 9, &checkpoints);

      NSFGFrameInfo wrong_order = pre_reset;
      wrong_order.frame_id = 2;
      RequireStatus(NSFGSessionEvaluate(session, &wrong_order, nullptr),
                    NSFG_STATUS_OUT_OF_ORDER_FRAME,
                    "out-of-order rejection");
      if (TextureBytes(targets[0], 4) != negative_before)
        throw std::runtime_error("rejected out-of-order evaluation touched output");

      NSFGFrameInfo alias = pre_reset;
      alias.generated_target_rgba8 = alias.current_backbuffer_rgba8;
      RequireStatus(NSFGSessionEvaluate(session, &alias, nullptr),
                    NSFG_STATUS_INVALID_ARGUMENT, "alias rejection");

      std::array<std::string, 15> output_sha{};
      std::array<std::uint64_t, 15> allocation_counts{};
      std::uint64_t evaluate_allocations = 0;
      for (std::uint64_t frame = 1; frame < 16; ++frame) {
        NSFGFrameInfo input{};
        input.current_backbuffer_rgba8 = (__bridge void*)color[frame];
        input.previous_backbuffer_rgba8 = (__bridge void*)color[frame - 1];
        input.motion_rg16f = (__bridge void*)motion[frame];
        input.depth_r32f = (__bridge void*)depth[frame];
        input.generated_target_rgba8 = (__bridge void*)targets[frame - 1];
        input.frame_id = frame;
        NSFGFrameResult result{};
        const auto status = NSFGSessionEvaluate(session, &input, &result);
        if (status != NSFG_STATUS_OK)
          throw std::runtime_error(std::string("frame evaluation failed: ") +
                                   NSFGSessionLastError(session));
        if (!result.generated_output_valid ||
            result.selected_launch_count != 76)
          throw std::runtime_error("generated result contract mismatch");
        AppendCheckpoints(session, "generated-" + std::to_string(frame),
                          12, 76, &checkpoints);
        evaluate_allocations += result.resource_allocations_during_evaluate;
        allocation_counts[frame - 1] =
            result.resource_allocations_during_evaluate;
        const auto bytes = TextureBytes(targets[frame - 1], 4);
        output_sha[frame - 1] = Sha256(bytes);
        Write(output / ("generated-" + std::to_string(frame) + ".rgba8"), bytes);
      }

      const auto invalidated_output_before = TextureBytes(targets[0], 4);
      RequireStatus(NSFGSessionInvalidate(session), NSFG_STATUS_OK,
                    "invalidate");
      RequireStatus(NSFGSessionEvaluate(session, &pre_reset, nullptr),
                    NSFG_STATUS_NEEDS_RESET, "post-invalidate rejection");
      if (TextureBytes(targets[0], 4) != invalidated_output_before)
        throw std::runtime_error("post-invalidate rejection touched output");
      RequireStatus(NSFGSessionEvaluate(session, &reset, nullptr),
                    NSFG_STATUS_OK, "replay reset");
      NSFGFrameInfo replay = pre_reset;
      replay.generated_target_rgba8 = (__bridge void*)replay_target;
      NSFGFrameResult replay_result{};
      RequireStatus(NSFGSessionEvaluate(session, &replay, &replay_result),
                    NSFG_STATUS_OK, "reset replay");
      const bool reset_replay_exact =
          TextureBytes(replay_target, 4) == TextureBytes(targets[0], 4) &&
          replay_result.resource_allocations_during_evaluate == 0;

      std::uint64_t input_mismatches = 0;
      for (std::size_t frame = 0; frame < frames.size(); ++frame) {
        const auto color_after = TextureBytes(color[frame], 4);
        const auto motion_after = TextureBytes(motion[frame], 4);
        const auto depth_after = TextureBytes(depth[frame], 4);
        input_mismatches += color_after != frames[frame].backbuffer_rgba8;
        input_mismatches += motion_after != motion_bytes[frame];
        input_mismatches += depth_after != depth_bytes[frame];
      }
      NSFGSessionStats stats{};
      RequireStatus(NSFGSessionGetStats(session, &stats), NSFG_STATUS_OK,
                    "stats");
      bool pass = input_mismatches == 0 && stats.input_mismatched_bytes == 0 &&
                        stats.canary_mismatched_bytes == 0 &&
                        stats.reset_count == 2 && stats.generated_frame_count == 16 &&
                        allocation_counts[0] != 0 &&
                        std::all_of(allocation_counts.begin() + 1,
                                    allocation_counts.end(),
                                    [](std::uint64_t value) { return value == 0; }) &&
                        stats.resource_allocations_after_warmup == 0 &&
                        reset_replay_exact && checkpoints.size() == 1149;

      NSFGSessionDestroy(session);
      session = nullptr;

      create.enable_audit = false;
      NSFGSession* fast_session = nullptr;
      RequireStatus(NSFGSessionCreate(&create, &fast_session), NSFG_STATUS_OK,
                    "non-audit create");
      RequireStatus(NSFGSessionEvaluate(fast_session, &reset, nullptr),
                    NSFG_STATUS_OK, "non-audit reset");
      NSFGFrameInfo fast_a = pre_reset;
      fast_a.generated_target_rgba8 = (__bridge void*)fast_target_a;
      NSFGFrameResult fast_a_result{};
      RequireStatus(NSFGSessionEvaluate(fast_session, &fast_a, &fast_a_result),
                    NSFG_STATUS_OK, "non-audit frame1");
      NSFGFrameInfo fast_b{};
      fast_b.current_backbuffer_rgba8 = (__bridge void*)color[2];
      fast_b.previous_backbuffer_rgba8 = (__bridge void*)color[1];
      fast_b.motion_rg16f = (__bridge void*)motion[2];
      fast_b.depth_r32f = (__bridge void*)depth[2];
      fast_b.generated_target_rgba8 = (__bridge void*)fast_target_b;
      fast_b.frame_id = 2;
      NSFGFrameResult fast_b_result{};
      RequireStatus(NSFGSessionEvaluate(fast_session, &fast_b, &fast_b_result),
                    NSFG_STATUS_OK, "non-audit frame2");
      const bool non_audit_fast_path_exact =
          TextureBytes(fast_target_a, 4) == TextureBytes(targets[0], 4) &&
          TextureBytes(fast_target_b, 4) == TextureBytes(targets[1], 4) &&
          fast_a_result.resource_allocations_during_evaluate != 0 &&
          fast_b_result.resource_allocations_during_evaluate == 0;
      NSFGSessionDestroy(fast_session);
      pass = pass && non_audit_fast_path_exact;

      std::ofstream receipt(output / "run-receipt.json");
      receipt << "{\n  \"schema\": \"neuralstudio.fg.runtime-run.v1\",\n"
              << "  \"backend\": \"mac-metal-runtime-prototype\",\n"
              << "  \"run_id\": \"" << argv[8] << "\",\n"
              << "  \"process_id\": " << getpid() << ",\n"
              << "  \"poison_byte\": " << unsigned(poison) << ",\n"
              << "  \"ex2_mode\": \"" << argv[10] << "\",\n"
              << "  \"auxiliary_mode\": \"" << argv[11] << "\",\n"
              << "  \"rendered_frame_count\": 16,\n"
              << "  \"reset_launch_count\": 9,\n"
              << "  \"generated_launch_count\": 1140,\n"
              << "  \"caller_owned_input_mismatches\": " << input_mismatches << ",\n"
              << "  \"runtime_input_mismatched_bytes\": " << stats.input_mismatched_bytes << ",\n"
              << "  \"runtime_canary_mismatched_bytes\": " << stats.canary_mismatched_bytes << ",\n"
              << "  \"metal_resource_allocations_total\": " << stats.resource_allocations_total << ",\n"
              << "  \"metal_resource_allocations_after_warmup\": " << stats.resource_allocations_after_warmup << ",\n"
              << "  \"resource_allocations_visible_during_evaluate\": " << evaluate_allocations << ",\n"
              << "  \"resource_allocations_per_generated_frame\": [";
      for (std::size_t index = 0; index < allocation_counts.size(); ++index)
        receipt << (index ? ", " : "") << allocation_counts[index];
      receipt << "],\n  \"persistent_tail_resources\": true,\n"
              << "  \"no_resource_allocation_after_warmup\": true,\n"
              << "  \"tail_audit_readbacks_enabled\": true,\n"
              << "  \"tail_probe_file_io\": false,\n"
              << "  \"caller_owned_current_previous_rgba8\": true,\n"
              << "  \"caller_owned_motion_rg16f_depth_r32f\": true,\n"
              << "  \"caller_owned_generated_target_rgba8\": true,\n"
              << "  \"l53_l55_auxiliary_profile\": \""
              << (std::string(argv[11]) == "live"
                      ? "live-early-resource-aliases"
                      : "accepted-internal-small-flow") << "\",\n"
              << "  \"l53_l55_auxiliary_caller_owned\": false,\n"
              << "  \"l52_l53_l55_live_early_resource_aliases\": "
              << (std::string(argv[11]) == "live" ? "true" : "false")
              << ",\n"
              << "  \"model_and_architecture_buffers_retained\": true,\n"
              << "  \"cpu_allocations_counted\": false,\n"
              << "  \"expected_or_archived_intermediate_ingress\": false,\n"
              << "  \"rejected_output_unchanged\": true,\n"
              << "  \"reset_replay_exact\": " << (reset_replay_exact ? "true" : "false") << ",\n"
              << "  \"non_audit_fast_path_exact\": " << (non_audit_fast_path_exact ? "true" : "false") << ",\n"
              << "  \"checkpoint_sha256\": {";
      for (std::size_t index = 0; index < checkpoints.size(); ++index)
        receipt << (index ? "," : "") << "\n    \""
                << checkpoints[index].first << "\": \""
                << checkpoints[index].second << "\"";
      receipt << "\n  },\n  \"output_sha256\": [";
      for (std::size_t index = 0; index < output_sha.size(); ++index)
        receipt << (index ? ", " : "") << "\"" << output_sha[index] << "\"";
      receipt << "],\n  \"result\": \"" << (pass ? "pass" : "fail") << "\"\n}\n";
      std::cout << "output_root_sha256=";
      for (const auto& hash : output_sha) std::cout << hash;
      std::cout << "\nresult=" << (pass ? "pass" : "fail") << '\n';
      return pass ? 0 : 3;
    } catch (const std::exception& error) {
      std::cerr << "error=" << error.what() << '\n';
      return 1;
    }
  }
}
