/* NeuralStudio DLSSNR worker protocol - SPDX-License-Identifier: MIT */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace neuralstudio::vulkan_bridge::dlssnr_worker {

constexpr std::uint32_t kMagic = 0x4E534E52u;  // NSNR
constexpr std::uint16_t kVersion = 2;
constexpr std::uint32_t kSlotCount = 3;
constexpr std::uint32_t kResourcesPerSlot = 4;
constexpr std::uint32_t kSharedHandleCount =
    1 + kSlotCount * kResourcesPerSlot;
constexpr std::uint32_t kMaximumPacketBytes = 4096;

enum class MessageType : std::uint16_t {
  hello = 1,
  hello_ack = 2,
  configure = 3,
  configured = 4,
  evaluate = 5,
  evaluated = 6,
  release = 7,
  released = 8,
  inject_crash = 9,
  inject_timeout = 10,
  inject_device_loss = 11,
  device_lost = 12,
  shutdown = 13,
  shutdown_ack = 14,
  set_options = 15,
  options_set = 16,
};

enum class Status : std::uint32_t {
  ok = 0,
  invalid_packet = 1,
  invalid_state = 2,
  adapter_mismatch = 3,
  shared_handle_open_failed = 4,
  streamline_init_failed = 5,
  evaluate_failed = 6,
  gpu_timeout = 7,
  device_lost = 8,
  internal_error = 9,
};

enum class SharedHandleIndex : std::uint32_t {
  timeline_fence = 0,
  slot0_input = 1,
  slot0_output = 2,
  slot0_depth = 3,
  slot0_motion = 4,
};

struct Options {
  float intensity = 2.0f;
  float local_tone_strength = 1.0f;
  float local_structure_strength = 1.0f;
  float global_tone_strength = 1.0f;
  float skin_structure_strength = 1.0f;
  std::uint32_t style = 0;
  std::uint32_t render_preset = 0;
  std::uint32_t use_auto_mask = 0;
  std::uint32_t performance_mode = 3;
};

struct FrameConstants {
  float motion_scale_x = 1.0f;
  float motion_scale_y = 1.0f;
  float jitter_x = 0.0f;
  float jitter_y = 0.0f;
  float camera_near = 0.1f;
  float camera_far = 1000.0f;
  float camera_fov_radians = 1.0f;
  float camera_aspect = 1.0f;
  std::uint32_t depth_inverted = 0;
  std::uint32_t camera_motion_included = 1;
  std::uint32_t motion_vectors_3d = 0;
  std::uint32_t reserved = 0;
};

[[nodiscard]] constexpr std::size_t ResourceHandleIndex(
    std::uint32_t slot, std::uint32_t resource) noexcept {
  return 1u + static_cast<std::size_t>(slot) * kResourcesPerSlot + resource;
}

struct Packet {
  std::uint32_t magic = kMagic;
  std::uint16_t version = kVersion;
  MessageType type = MessageType::hello;
  std::uint32_t size = sizeof(Packet);
  Status status = Status::ok;
  std::uint64_t nonce = 0;
  std::uint64_t sequence = 0;
  std::uint32_t process_id = 0;
  std::uint32_t slot = 0;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint32_t reset = 0;
  std::uint32_t feature_creations = 0;
  std::uint32_t successful_evaluations = 0;
  std::uint32_t debug_layer_available = 0;
  std::uint32_t debug_severe_messages = 0;
  std::uint64_t adapter_luid = 0;
  std::uint64_t frame_id = 0;
  std::uint64_t input_ready = 0;
  std::uint64_t model_complete = 0;
  std::uint64_t copyback_complete = 0;
  std::uint64_t retired = 0;
  Options options{};
  FrameConstants constants{};
  std::array<std::uint64_t, kSharedHandleCount> shared_handles{};
  std::array<std::uint64_t, 8> reserved{};
};

static_assert(std::is_trivially_copyable_v<Packet>);
static_assert(sizeof(Packet) <= kMaximumPacketBytes);

[[nodiscard]] inline bool Valid(const Packet& packet) noexcept {
  return packet.magic == kMagic && packet.version == kVersion &&
         packet.size == sizeof(Packet);
}

[[nodiscard]] inline bool ValidConstants(
    const FrameConstants& constants) noexcept {
  const auto finite_range = [](float value, float low, float high) {
    return value >= low && value <= high;
  };
  return finite_range(constants.motion_scale_x, -16384.0f, 16384.0f) &&
         finite_range(constants.motion_scale_y, -16384.0f, 16384.0f) &&
         finite_range(constants.jitter_x, -16.0f, 16.0f) &&
         finite_range(constants.jitter_y, -16.0f, 16.0f) &&
         finite_range(constants.camera_near, 0.00001f, 1000000.0f) &&
         finite_range(constants.camera_far, constants.camera_near,
                      1000000000.0f) &&
         finite_range(constants.camera_fov_radians, 0.01f, 3.13f) &&
         finite_range(constants.camera_aspect, 0.01f, 100.0f) &&
         constants.depth_inverted <= 1 &&
         constants.camera_motion_included <= 1 &&
         constants.motion_vectors_3d <= 1 && constants.reserved == 0;
}

[[nodiscard]] inline bool ValidTimeline(const Packet& packet) noexcept {
  return packet.input_ready != 0 &&
         packet.input_ready < packet.model_complete &&
         packet.model_complete < packet.copyback_complete &&
         packet.copyback_complete < packet.retired;
}

[[nodiscard]] inline bool ValidOptions(const Options& options) noexcept {
  const auto finite_range = [](float value, float low, float high) {
    return value >= low && value <= high;
  };
  return finite_range(options.intensity, 0.0f, 4.0f) &&
         finite_range(options.local_tone_strength, 0.0f, 4.0f) &&
         finite_range(options.local_structure_strength, 0.0f, 4.0f) &&
         finite_range(options.global_tone_strength, 0.0f, 4.0f) &&
         finite_range(options.skin_structure_strength, 0.0f, 4.0f) &&
         options.style <= 3 && options.render_preset <= 3 &&
         options.use_auto_mask <= 1 && options.performance_mode <= 3;
}

}  // namespace neuralstudio::vulkan_bridge::dlssnr_worker
