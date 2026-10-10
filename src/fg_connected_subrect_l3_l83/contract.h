#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "../fg_connected_l3_l19/contract.h"

namespace fg_connected_subrect_l3_l83 {

inline constexpr std::uint32_t kResourceWidth = 640;
inline constexpr std::uint32_t kResourceHeight = 360;
inline constexpr std::uint32_t kOriginX = 32;
inline constexpr std::uint32_t kOriginY = 24;
inline constexpr std::uint32_t kWidth = 512;
inline constexpr std::uint32_t kHeight = 288;
inline constexpr std::uint32_t kGridX = 32;
inline constexpr std::uint32_t kGridY = 18;
inline constexpr std::size_t kGuardBytes = 256;
inline constexpr std::uint8_t kGuardValue = 0xc7;
inline constexpr std::array<std::uint32_t, 4> kLevelWidths = {
    256, 128, 64, 32};
inline constexpr std::array<std::uint32_t, 4> kLevelHeights = {
    144, 72, 36, 18};
inline constexpr std::array<std::size_t, 4> kLevelBytes = {
    294912, 73728, 18432, 4608};
inline constexpr std::size_t kMaskBytes =
    std::size_t{kWidth} * kHeight;
inline constexpr std::size_t kOutputBytes =
    std::size_t{kResourceWidth} * kResourceHeight * 4u;
inline constexpr std::size_t kRegionBytes =
    std::size_t{kWidth} * kHeight * 4u;
inline constexpr std::size_t kPhysicalPixels =
    std::size_t{kWidth} * kHeight;
inline constexpr std::size_t kModelInputBytes =
    128u * 96u * 48u * 2u;
inline constexpr std::uint32_t kEarlyWidth = kWidth / 2u;
inline constexpr std::uint32_t kEarlyHeight = kHeight / 2u;
inline constexpr std::uint32_t kTensorWidth = 128u;
inline constexpr std::uint32_t kTensorHeight = 96u;
inline constexpr std::size_t kEarlyPixels =
    std::size_t{kEarlyWidth} * kEarlyHeight;
inline constexpr std::size_t kEarlyColorBytes = kEarlyPixels * 8u;
inline constexpr std::size_t kEarlyMetricBytes = kEarlyPixels * 2u;
inline constexpr std::size_t kEarlyFlowBytes = kEarlyPixels * 8u;
inline constexpr std::size_t kEarlyDepthBytes = kEarlyPixels * 4u;
inline constexpr std::size_t kEarlyMaskBytes = kEarlyPixels;
inline constexpr std::size_t kEarlyVectorBytes = kEarlyPixels * 4u;
inline constexpr std::size_t kEarlyAtomicBytes = kEarlyPixels * 8u;
inline constexpr std::array<std::size_t, 5> kEarlyPyramidBytes = {
    36864u, 55296u, 13824u, 3456u, 864u};
inline constexpr std::size_t kHistoryBytes =
    std::size_t{kResourceWidth} * kResourceHeight * 8u;
inline constexpr std::size_t kSourceBytes = kOutputBytes;
inline constexpr std::size_t kExternalVectorBytes =
    std::size_t{kResourceWidth} * kResourceHeight * 4u;
inline constexpr std::size_t kExternalScalarBytes = kExternalVectorBytes;
inline constexpr std::uint64_t kEarlyAtomicSentinel =
    0x7fffffff7fffffffull;
inline constexpr std::size_t kL51Bytes = 128u * 96u * 16u * 2u;
inline constexpr std::size_t kL52VectorBytes =
    std::size_t{kWidth / 2u} * (kHeight / 2u) * 4u;
inline constexpr std::size_t kL52ScalarBytes =
    std::size_t{kWidth / 2u} * (kHeight / 2u) * 2u;
inline constexpr std::size_t kTailMotionBytes = kPhysicalPixels * 16u;
inline constexpr std::size_t kTailMaskBytes = kPhysicalPixels;
inline constexpr std::size_t kTailScratchBytes = kPhysicalPixels * 16u;
inline constexpr std::size_t kInputABytes = kPhysicalPixels * 8u;
inline constexpr std::size_t kInputBBytes = kPhysicalPixels * 8u;
inline constexpr std::size_t kInputCBytes = kPhysicalPixels * 2u;
inline constexpr std::size_t kL56Bytes = kPhysicalPixels * 20u;
inline constexpr std::size_t kL57Bytes = 144u * 256u * 32u * 2u;
inline constexpr std::size_t kL58Bytes = 72u * 128u * 32u * 2u;
inline constexpr std::size_t kLow64Bytes = 36u * 64u * 64u * 2u;
inline constexpr std::size_t kL68PlaneBytes = 72u * 128u * 32u * 2u;
inline constexpr std::size_t kL69Out0Bytes = 144u * 256u * 4u * 2u;
inline constexpr std::size_t kL69Out1Bytes = 144u * 256u * 2u;
inline constexpr std::size_t kL69Out2Bytes = 144u * 256u * 3u * 2u;
inline constexpr std::size_t kL70Out0Bytes = kPhysicalPixels * 4u * 2u;
inline constexpr std::size_t kL70Out1Bytes = kPhysicalPixels * 2u;
inline constexpr std::size_t kL70Out2Bytes = kPhysicalPixels * 3u * 2u;
inline constexpr std::size_t kL71Bytes = kPhysicalPixels * 18u * 2u;
inline constexpr std::size_t kL72Bytes = 144u * 256u * 16u * 2u;
inline constexpr std::size_t kBlock1Bytes = 72u * 128u * 32u * 2u;
inline constexpr std::size_t kL82PlaneBytes = kL72Bytes;
inline constexpr std::size_t kL83Bytes = kPhysicalPixels * 8u * 2u;
inline constexpr std::size_t kRcpRomBytes = 2097152;
inline constexpr char kRcpRomSha256[] =
    "7cf4f8cece6b18b07605edc9cb601a153346a79b10bf3b93c7ec058d57d29750";
inline constexpr std::size_t kModule19Bytes = 57792;
inline constexpr char kModule19Sha256[] =
    "143c6712c99db4a5792df697d68178afeea3657f4f0eeb12f905c90abf38a1b0";
inline constexpr char kRawL83ParameterSha256[] =
    "584fc6edd0a24dea7a56fc22855b247e6e56a119349adb0d728d62d6067475ee";
inline constexpr char kGenerator[] =
    "splitmix64-fg-subrect-module19-mode1-v1";

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

inline std::uint16_t RgbHalf(std::uint32_t fixture, std::uint32_t level,
                             std::uint32_t x, std::uint32_t y,
                             std::uint32_t channel) {
  const auto mixed = Mix(
      0xd6e8feb86659fd93ull ^
      (std::uint64_t{fixture + 1u} * 0xa0761d6478bd642full) ^
      (std::uint64_t{level + 1u} * 0xe7037ed1a0b428dbull) ^
      (std::uint64_t{y} * 0x8ebc6af09c88c6e3ull) ^
      (std::uint64_t{x} * 0x589965cc75374cc3ull) ^
      (std::uint64_t{channel + 1u} * 0x1d8e4e27c47d124full));
  const std::uint16_t sign = (mixed >> 63u) != 0u ? 0x8000u : 0u;
  const std::uint16_t exponent =
      static_cast<std::uint16_t>(12u + ((mixed >> 19u) & 3u));
  return static_cast<std::uint16_t>(sign | (exponent << 10u) |
                                    (mixed & 0x03ffu));
}

inline std::uint16_t AlphaHalf(std::uint32_t fixture, std::uint32_t level,
                               std::uint32_t x, std::uint32_t y) {
  constexpr std::array<std::uint16_t, 5> values = {
      0x0000u, 0x3000u, 0x3400u, 0x3800u, 0x3c00u};
  const auto mixed = Mix(0xbb67ae8584caa73bull ^
                         (std::uint64_t{fixture} << 48u) ^
                         (std::uint64_t{level} << 40u) ^
                         std::uint64_t{y} * 0x94d049bb133111ebull ^ x);
  return values[mixed % values.size()];
}

inline std::array<std::vector<std::uint8_t>, 4> Pyramid(
    std::uint32_t fixture) {
  std::array<std::vector<std::uint8_t>, 4> result;
  for (std::uint32_t level = 0; level < result.size(); ++level) {
    result[level].resize(kLevelBytes[level]);
    for (std::uint32_t y = 0; y < kLevelHeights[level]; ++y) {
      for (std::uint32_t x = 0; x < kLevelWidths[level]; ++x) {
        const std::size_t offset =
            (std::size_t{y} * kLevelWidths[level] + x) * 8u;
        for (std::uint32_t channel = 0; channel < 3u; ++channel)
          Store16(&result[level], offset + channel * 2u,
                  RgbHalf(fixture, level, x, y, channel));
        Store16(&result[level], offset + 6u,
                AlphaHalf(fixture, level, x, y));
      }
    }
  }
  return result;
}

inline std::vector<std::uint8_t> Mask(std::uint32_t fixture) {
  std::vector<std::uint8_t> result(kMaskBytes);
  for (std::uint32_t y = 0; y < kHeight; ++y) {
    for (std::uint32_t x = 0; x < kWidth; ++x) {
      const auto mixed = Mix(0x3c6ef372fe94f82bull ^
                             (std::uint64_t{fixture} << 52u) ^
                             std::uint64_t{y} * kWidth + x);
      result[std::size_t{y} * kWidth + x] =
          ((x + y + fixture) % 19u == 0u || (mixed & 15u) < 3u) ? 0u : 255u;
    }
  }
  return result;
}

inline std::array<std::uint8_t, 4> StablePixel(
    std::uint32_t fixture, std::uint32_t x, std::uint32_t y) {
  const auto mixed = Mix(0xa54ff53a5f1d36f1ull ^
                         (std::uint64_t{fixture} << 56u) ^
                         std::uint64_t{y} * kWidth + x);
  return {static_cast<std::uint8_t>(mixed),
          static_cast<std::uint8_t>(mixed >> 17u),
          static_cast<std::uint8_t>(mixed >> 39u), 255u};
}

inline std::vector<std::uint8_t> InitialOutput(
    std::uint32_t fixture, std::uint8_t poison,
    const std::vector<std::uint8_t>& mask) {
  std::vector<std::uint8_t> result(kOutputBytes, poison);
  for (std::uint32_t y = 0; y < kHeight; ++y) {
    for (std::uint32_t x = 0; x < kWidth; ++x) {
      const std::size_t mask_index = std::size_t{y} * kWidth + x;
      if (mask[mask_index] != 0u) continue;
      const auto pixel = StablePixel(fixture, x, y);
      const std::size_t output =
          (std::size_t{y + kOriginY} * kResourceWidth + x + kOriginX) * 4u;
      std::memcpy(result.data() + output, pixel.data(), pixel.size());
    }
  }
  return result;
}

inline std::vector<std::uint8_t> Region(
    const std::vector<std::uint8_t>& full) {
  std::vector<std::uint8_t> result(kRegionBytes);
  for (std::uint32_t y = 0; y < kHeight; ++y) {
    const std::size_t source =
        (std::size_t{y + kOriginY} * kResourceWidth + kOriginX) * 4u;
    const std::size_t destination = std::size_t{y} * kWidth * 4u;
    std::memcpy(result.data() + destination, full.data() + source,
                std::size_t{kWidth} * 4u);
  }
  return result;
}

struct EarlyExternalInputs {
  std::vector<std::uint8_t> source_reset;
  std::vector<std::uint8_t> source_generated;
  std::vector<std::uint8_t> vector_reset;
  std::vector<std::uint8_t> vector_generated;
  std::vector<std::uint8_t> scalar_reset;
  std::vector<std::uint8_t> scalar_generated;
};

inline EarlyExternalInputs GenerateEarlyExternal() {
  const auto values = fg_connected_l3_l19::Generate();
  return {values.source_reset, values.source_generated,
          values.vector_reset, values.vector_generated,
          values.scalar_reset, values.scalar_generated};
}

inline std::vector<std::uint8_t> InitialHistory() {
  return std::vector<std::uint8_t>(kHistoryBytes);
}

inline std::vector<std::uint8_t> InitialGeneratedDepth() {
  constexpr std::array<std::uint32_t, 3> values = {
      0x3e800000u, 0x3f000000u, 0x3f400000u};
  std::vector<std::uint8_t> result(kEarlyDepthBytes);
  for (std::size_t pixel = 0; pixel < kEarlyPixels; ++pixel) {
    const auto value = values[Mix(pixel) % values.size()];
    std::memcpy(result.data() + pixel * 4u, &value, sizeof(value));
  }
  return result;
}

inline std::vector<std::uint8_t> InitialGeneratedMask() {
  return std::vector<std::uint8_t>(kEarlyMaskBytes, 1u);
}

inline std::vector<std::uint8_t> InitialAtomic() {
  std::vector<std::uint8_t> result(kEarlyAtomicBytes);
  for (std::size_t pixel = 0; pixel < kEarlyPixels; ++pixel)
    std::memcpy(result.data() + pixel * 8u, &kEarlyAtomicSentinel,
                sizeof(kEarlyAtomicSentinel));
  return result;
}

inline std::vector<std::uint8_t> InitialAllocation29() {
  std::vector<std::uint8_t> result(kEarlyMetricBytes);
  constexpr std::uint16_t half = 0x3800u;
  for (std::size_t pixel = 0; pixel < kEarlyPixels; ++pixel)
    std::memcpy(result.data() + pixel * 2u, &half, sizeof(half));
  return result;
}

inline std::vector<std::uint8_t> FullRegion(
    const std::vector<std::uint8_t>& full, std::uint32_t bytes_per_pixel) {
  if (full.size() != std::size_t{kResourceWidth} * kResourceHeight *
                         bytes_per_pixel)
    throw std::runtime_error("full-region source size drift");
  std::vector<std::uint8_t> result(
      std::size_t{kWidth} * kHeight * bytes_per_pixel);
  for (std::uint32_t y = 0; y < kHeight; ++y) {
    const auto source =
        (std::size_t{y + kOriginY} * kResourceWidth + kOriginX) *
        bytes_per_pixel;
    const auto destination = std::size_t{y} * kWidth * bytes_per_pixel;
    std::memcpy(result.data() + destination, full.data() + source,
                std::size_t{kWidth} * bytes_per_pixel);
  }
  return result;
}

inline std::vector<std::uint8_t> HistoryRegion(
    const std::vector<std::uint8_t>& full, std::uint32_t bytes_per_pixel) {
  if (full.size() != std::size_t{kResourceWidth} * kResourceHeight *
                         bytes_per_pixel)
    throw std::runtime_error("history-region source size drift");
  std::vector<std::uint8_t> result(
      std::size_t{kWidth} * kHeight * bytes_per_pixel);
  for (std::uint32_t y = 0; y < kHeight; ++y)
    std::memcpy(result.data() + std::size_t{y} * kWidth * bytes_per_pixel,
                full.data() + std::size_t{y} * kResourceWidth *
                                  bytes_per_pixel,
                std::size_t{kWidth} * bytes_per_pixel);
  return result;
}

inline std::vector<std::uint8_t> Module5ActiveRows(
    const std::vector<std::uint8_t>& full, std::uint32_t bytes_per_pixel) {
  constexpr std::uint32_t kActiveRows = 120u;
  if (full.size() != kEarlyPixels * bytes_per_pixel)
    throw std::runtime_error("module5 active-row source size drift");
  return {full.begin(),
          full.begin() + std::size_t{kEarlyWidth} * kActiveRows *
                             bytes_per_pixel};
}

inline std::size_t Module5UntouchedRowMismatches(
    const std::vector<std::uint8_t>& full, std::uint8_t poison,
    std::uint32_t bytes_per_pixel) {
  constexpr std::uint32_t kActiveRows = 120u;
  if (full.size() != kEarlyPixels * bytes_per_pixel)
    throw std::runtime_error("module5 untouched-row source size drift");
  const auto begin = std::size_t{kEarlyWidth} * kActiveRows *
                     bytes_per_pixel;
  std::size_t mismatches = 0;
  for (std::size_t index = begin; index < full.size(); ++index)
    mismatches += full[index] != poison;
  return mismatches;
}

}  // namespace fg_connected_subrect_l3_l83
