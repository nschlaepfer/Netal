#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "../fg_host/synthetic_contract.h"

namespace fg_production_contract_l3_l87 {

inline constexpr std::uint32_t kFullWidth = 640;
inline constexpr std::uint32_t kFullHeight = 360;
inline constexpr std::uint32_t kHalfWidth = 320;
inline constexpr std::uint32_t kHalfHeight = 180;
inline constexpr std::uint32_t kModelWidth = 160;
inline constexpr std::uint32_t kModelHeight = 96;
inline constexpr std::size_t kSourceBytes =
    std::size_t{kFullWidth} * kFullHeight * 4u;
inline constexpr std::size_t kHistoryBytes =
    std::size_t{kFullWidth} * kFullHeight * 8u;
inline constexpr std::size_t kExternalVectorBytes =
    std::size_t{kFullWidth} * kFullHeight * 4u;
inline constexpr std::size_t kExternalScalarBytes =
    std::size_t{kFullWidth} * kFullHeight * 4u;
inline constexpr std::size_t kHalfColorBytes =
    std::size_t{kHalfWidth} * kHalfHeight * 8u;
inline constexpr std::size_t kHalfScalarBytes =
    std::size_t{kHalfWidth} * kHalfHeight * 2u;
inline constexpr std::size_t kDepthBytes =
    std::size_t{kHalfWidth} * kHalfHeight * 4u;
inline constexpr std::size_t kMaskBytes =
    std::size_t{kHalfWidth} * kHalfHeight;
inline constexpr std::size_t kAtomicBytes =
    std::size_t{kHalfWidth} * kHalfHeight * 8u;
inline constexpr std::size_t kVectorBytes =
    std::size_t{kHalfWidth} * kHalfHeight * 4u;
inline constexpr std::size_t kModelInputBytes =
    std::size_t{kModelWidth} * kModelHeight * 48u * 2u;
inline constexpr std::size_t kL20OutputBytes =
    std::size_t{kModelWidth} * kModelHeight * 32u * 2u;
inline constexpr std::size_t kAuxMaskBytes =
    std::size_t{kHalfWidth} * kHalfHeight;
inline constexpr std::size_t kAuxVectorBytes =
    std::size_t{kHalfWidth} * kHalfHeight * 4u;
inline constexpr std::uint8_t kCanaryValue = 0xc7;
inline constexpr std::uint64_t kAtomicSentinel =
    0x7fffffff7fffffffull;
inline constexpr char kGenerator[] =
    "neuralstudio-fg-host-MakeSyntheticFrame-v1";

struct ExternalInputs {
  std::vector<std::uint8_t> source_reset;
  std::vector<std::uint8_t> source_generated;
  std::vector<std::uint8_t> source_generated2;
  std::vector<std::uint8_t> vector_reset;
  std::vector<std::uint8_t> vector_generated;
  std::vector<std::uint8_t> vector_generated2;
  std::vector<std::uint8_t> scalar_reset;
  std::vector<std::uint8_t> scalar_generated;
  std::vector<std::uint8_t> scalar_generated2;
};

struct RenderedFrameInput {
  std::vector<std::uint8_t> source;
  std::vector<std::uint8_t> motion;
  std::vector<std::uint8_t> depth;
};

inline RenderedFrameInput GenerateRenderedFrame(std::uint32_t frame_id) {
  const auto frame = neuralstudio::fg::MakeSyntheticFrame(
      kFullWidth, kFullHeight, frame_id);
  RenderedFrameInput result;
  result.source = frame.backbuffer_rgba8;
  result.motion.resize(frame.motion_rg16f.size() * sizeof(std::uint16_t));
  result.depth.resize(frame.depth_r32f.size() * sizeof(float));
  std::memcpy(result.motion.data(), frame.motion_rg16f.data(),
              result.motion.size());
  std::memcpy(result.depth.data(), frame.depth_r32f.data(),
              result.depth.size());
  return result;
}

inline std::uint64_t Mix(std::uint64_t value) {
  value += 0x9e3779b97f4a7c15ull;
  value = (value ^ (value >> 30u)) * 0xbf58476d1ce4e5b9ull;
  value = (value ^ (value >> 27u)) * 0x94d049bb133111ebull;
  return value ^ (value >> 31u);
}

inline void Store16(std::vector<std::uint8_t>* bytes, std::size_t offset,
                    std::uint16_t value) {
  std::memcpy(bytes->data() + offset, &value, sizeof(value));
}

inline void Store32(std::vector<std::uint8_t>* bytes, std::size_t offset,
                    std::uint32_t value) {
  std::memcpy(bytes->data() + offset, &value, sizeof(value));
}

inline void Store64(std::vector<std::uint8_t>* bytes, std::size_t offset,
                    std::uint64_t value) {
  std::memcpy(bytes->data() + offset, &value, sizeof(value));
}

inline std::vector<std::uint8_t> Source(std::uint64_t salt) {
  std::vector<std::uint8_t> result(kSourceBytes);
  for (std::size_t pixel = 0; pixel < result.size() / 4u; ++pixel) {
    const auto mixed = Mix(salt ^ pixel * 0xd1b54a32d192ed03ull);
    result[pixel * 4u] = static_cast<std::uint8_t>(mixed);
    result[pixel * 4u + 1u] = static_cast<std::uint8_t>(mixed >> 13u);
    result[pixel * 4u + 2u] = static_cast<std::uint8_t>(mixed >> 31u);
    result[pixel * 4u + 3u] = static_cast<std::uint8_t>(mixed >> 47u);
  }
  return result;
}

inline std::vector<std::uint8_t> Vector(std::uint64_t salt) {
  constexpr std::uint16_t values[] = {
      0x0000,0x8000,0x1800,0x9800,0x2000,0xa000,0x2800,0xa800,
      0x3000,0xb000,0x3400,0xb400,0x3800,0xb800};
  std::vector<std::uint8_t> result(kExternalVectorBytes);
  for (std::size_t pixel = 0; pixel < result.size() / 4u; ++pixel) {
    const auto mixed = Mix(salt ^ pixel * 0x94d049bb133111ebull);
    Store16(&result,pixel*4u,values[mixed % (sizeof(values)/2u)]);
    Store16(&result,pixel*4u+2u,
            values[(mixed>>11u) % (sizeof(values)/2u)]);
  }
  return result;
}

inline std::vector<std::uint8_t> Scalar(std::uint64_t salt) {
  std::vector<std::uint8_t> result(kExternalScalarBytes);
  for (std::size_t pixel = 0; pixel < result.size() / 4u; ++pixel) {
    const auto mixed = Mix(salt ^ pixel * 0x9e3779b97f4a7c15ull);
    const float value = static_cast<float>(64u + (mixed & 255u)) / 512.0f;
    Store32(&result,pixel*4u,std::bit_cast<std::uint32_t>(value));
  }
  return result;
}

inline ExternalInputs Generate() {
  const auto reset = neuralstudio::fg::MakeSyntheticFrame(
      kFullWidth, kFullHeight, 0);
  const auto generated1 = neuralstudio::fg::MakeSyntheticFrame(
      kFullWidth, kFullHeight, 1);
  const auto generated2 = neuralstudio::fg::MakeSyntheticFrame(
      kFullWidth, kFullHeight, 2);
  auto motion_bytes = [](const std::vector<std::uint16_t>& values) {
    std::vector<std::uint8_t> result(values.size() * sizeof(values[0]));
    std::memcpy(result.data(), values.data(), result.size());
    return result;
  };
  auto depth_bytes = [](const std::vector<float>& values) {
    std::vector<std::uint8_t> result(values.size() * sizeof(values[0]));
    std::memcpy(result.data(), values.data(), result.size());
    return result;
  };
  return {
      reset.backbuffer_rgba8,
      generated1.backbuffer_rgba8,
      generated2.backbuffer_rgba8,
      motion_bytes(reset.motion_rg16f),
      motion_bytes(generated1.motion_rg16f),
      motion_bytes(generated2.motion_rg16f),
      depth_bytes(reset.depth_r32f),
      depth_bytes(generated1.depth_r32f),
      depth_bytes(generated2.depth_r32f),
  };
}

inline std::vector<std::uint8_t> InitialHistory() {
  return std::vector<std::uint8_t>(kHistoryBytes);
}

inline std::vector<std::uint8_t> InitialGeneratedDepth() {
  constexpr std::uint32_t values[] = {0x3e800000u,0x3f000000u,0x3f400000u};
  std::vector<std::uint8_t> result(kDepthBytes);
  for (std::size_t pixel = 0; pixel < kDepthBytes / 4u; ++pixel)
    Store32(&result,pixel*4u,values[Mix(pixel) % 3u]);
  return result;
}

inline std::vector<std::uint8_t> InitialGeneratedMask() {
  return std::vector<std::uint8_t>(kMaskBytes, 1u);
}

inline std::vector<std::uint8_t> InitialAtomic() {
  std::vector<std::uint8_t> result(kAtomicBytes);
  for (std::size_t pixel = 0; pixel < kAtomicBytes / 8u; ++pixel)
    Store64(&result,pixel*8u,kAtomicSentinel);
  return result;
}

inline std::vector<std::uint8_t> InitialAllocation29() {
  std::vector<std::uint8_t> result(kHalfScalarBytes);
  for (std::size_t pixel = 0; pixel < kHalfScalarBytes / 2u; ++pixel)
    Store16(&result,pixel*2u,0x3800u);
  return result;
}

}  // namespace fg_production_contract_l3_l87
