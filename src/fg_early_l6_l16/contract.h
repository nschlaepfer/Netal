#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace fg_early_l6_l16 {

inline constexpr std::uint32_t kWidth = 320;
inline constexpr std::uint32_t kHeight = 180;
inline constexpr std::size_t kPixels = std::size_t{kWidth} * kHeight;
inline constexpr std::size_t kFlowBytes = kPixels * 2u * 4u;
inline constexpr std::size_t kDepthBytes = kPixels * 4u;
inline constexpr std::size_t kInputMaskBytes = kPixels;
inline constexpr std::size_t kCandidateBytes = kPixels * 8u;
inline constexpr std::size_t kVectorSurfaceBytes = kPixels * 2u * 2u;
inline constexpr std::size_t kMaskSurfaceBytes = kPixels;
inline constexpr std::array<std::size_t, 5> kOutputBufferBytes = {
    57600, 86400, 21600, 5520, 1440};
inline constexpr std::size_t kModuleBytes = 38472;
inline constexpr char kModuleSha256[] =
    "11df926493fca0a981d560427b36c4dcf489ccb84d794501a70bda8d414ea8a2";
inline constexpr char kInputGenerator[] =
    "fg-early-l6-l16-cross-bank-pyramid-v1";
inline constexpr std::uint64_t kCandidateSentinel =
    0x7fffffff7fffffffull;

enum class Phase { kReset, kGenerated };

inline std::string PhaseName(Phase phase) {
  return phase == Phase::kReset ? "reset" : "generated";
}

inline Phase ParsePhase(const std::string& value) {
  if (value == "reset") return Phase::kReset;
  if (value == "generated") return Phase::kGenerated;
  throw std::runtime_error("invalid L6/L16 phase: " + value);
}

inline std::uint64_t Mix(std::uint64_t value) {
  value += 0x9e3779b97f4a7c15ull;
  value = (value ^ (value >> 30u)) * 0xbf58476d1ce4e5b9ull;
  value = (value ^ (value >> 27u)) * 0x94d049bb133111ebull;
  return value ^ (value >> 31u);
}

inline void Store32(std::vector<std::uint8_t>* bytes, std::size_t offset,
                    std::uint32_t value) {
  std::memcpy(bytes->data() + offset, &value, sizeof(value));
}

inline void Store64(std::vector<std::uint8_t>* bytes, std::size_t offset,
                    std::uint64_t value) {
  std::memcpy(bytes->data() + offset, &value, sizeof(value));
}

inline std::vector<std::uint8_t> Flow(Phase phase) {
  constexpr std::array<float, 5> x_values = {
      -2.0f / float(kWidth), -1.0f / float(kWidth), 0.0f,
      1.0f / float(kWidth), 2.0f / float(kWidth)};
  constexpr std::array<float, 5> y_values = {
      -2.0f / float(kHeight), -1.0f / float(kHeight), 0.0f,
      1.0f / float(kHeight), 2.0f / float(kHeight)};
  const std::uint32_t phase_shift = phase == Phase::kReset ? 0u : 2u;
  std::vector<std::uint8_t> result(kFlowBytes);
  for (std::uint32_t y = 0; y < kHeight; ++y) {
    for (std::uint32_t x = 0; x < kWidth; ++x) {
      const std::size_t pixel = std::size_t{y} * kWidth + x;
      const auto mixed = Mix(0x243f6a8885a308d3ull ^
                             std::uint64_t{x / 8u} * 0x9e3779b97f4a7c15ull ^
                             std::uint64_t{y / 8u} * 0xbf58476d1ce4e5b9ull);
      const auto fx = x_values[(x + 2u * y + phase_shift + (mixed & 1u)) % 5u];
      const auto fy = y_values[(2u * x + y + phase_shift +
                                ((mixed >> 9u) & 1u)) % 5u];
      Store32(&result, pixel * 8u, std::bit_cast<std::uint32_t>(fx));
      Store32(&result, pixel * 8u + 4u, std::bit_cast<std::uint32_t>(fy));
    }
  }
  return result;
}

inline std::vector<std::uint8_t> Depth(Phase phase) {
  constexpr std::array<std::uint32_t, 4> values = {
      0x3e800000u, 0x3f000000u, 0x3f200000u, 0x3f400000u};
  const std::uint32_t shift = phase == Phase::kReset ? 0u : 1u;
  std::vector<std::uint8_t> result(kDepthBytes);
  for (std::uint32_t y = 0; y < kHeight; ++y) {
    for (std::uint32_t x = 0; x < kWidth; ++x) {
      const std::size_t pixel = std::size_t{y} * kWidth + x;
      Store32(&result, pixel * 4u,
              values[((x / 5u) + 3u * (y / 5u) + shift) % values.size()]);
    }
  }
  return result;
}

inline std::vector<std::uint8_t> InputMask(Phase phase) {
  const std::uint32_t shift = phase == Phase::kReset ? 0u : 3u;
  std::vector<std::uint8_t> result(kInputMaskBytes);
  for (std::uint32_t y = 0; y < kHeight; ++y) {
    for (std::uint32_t x = 0; x < kWidth; ++x) {
      const std::size_t pixel = std::size_t{y} * kWidth + x;
      const auto mixed = Mix(0x13198a2e03707344ull + pixel *
                             0x94d049bb133111ebull);
      result[pixel] = ((mixed >> shift) & 7u) < 5u ? 255u : 0u;
    }
  }
  return result;
}

inline std::vector<std::uint8_t> Candidates(Phase phase) {
  constexpr std::array<std::uint32_t, 4> high_values = {
      0x3faaaaaau, 0x40000000u, 0x40555554u, 0x40800000u};
  constexpr std::array<std::uint16_t, 5> half_values = {
      0xc000u, 0xbc00u, 0x0000u, 0x3c00u, 0x4000u};
  const std::uint32_t phase_shift = phase == Phase::kReset ? 0u : 1u;
  std::vector<std::uint8_t> result(kCandidateBytes);
  for (std::uint32_t y = 0; y < kHeight; ++y) {
    for (std::uint32_t x = 0; x < kWidth; ++x) {
      const std::size_t pixel = std::size_t{y} * kWidth + x;
      const auto mixed = Mix(0xa4093822299f31d0ull + pixel *
                             0xd1b54a32d192ed03ull);
      if ((mixed & 31u) == 0u) {
        Store64(&result, pixel * 8u, kCandidateSentinel);
        continue;
      }
      const std::uint32_t high =
          (high_values[((x / 4u) + (y / 4u) + phase_shift) % 4u] &
           0xfffffffeu) |
          std::uint32_t((mixed >> 5u) & 1u);
      const std::uint16_t low_x = half_values[(x + phase_shift) % 5u];
      const std::uint16_t low_y = half_values[(y + 2u * phase_shift) % 5u];
      const std::uint64_t value =
          (std::uint64_t{high} << 32u) |
          (std::uint64_t{low_x} << 16u) | low_y;
      Store64(&result, pixel * 8u, value);
    }
  }
  return result;
}

}  // namespace fg_early_l6_l16
