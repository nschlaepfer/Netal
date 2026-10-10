#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace fg_mid_l53 {

inline constexpr std::uint32_t kWidth = 320;
inline constexpr std::uint32_t kHeight = 180;
inline constexpr std::size_t kPixelCount = std::size_t{kWidth} * kHeight;
inline constexpr std::size_t kFlowF32Bytes = kPixelCount * 2u * 4u;
inline constexpr std::size_t kFlowF16Bytes = kPixelCount * 2u * 2u;
inline constexpr std::size_t kDepthBytes = kPixelCount * 4u;
inline constexpr std::size_t kOutputBytes = kPixelCount * 8u;
inline constexpr std::size_t kModuleBytes = 98408;
inline constexpr char kModuleSha256[] =
    "5a8e0284aab8ac14fc82b0504bbeef25d2fce1d13a1c11d8bdb3f91fee8145fc";
inline constexpr char kRawParameterSha256[] =
    "5a68951844e42596a4d20bc8d3d3db6ecff94927d7f6e96d66f7264c3fb895c7";
inline constexpr char kInputGenerator[] =
    "splitmix64-fg-mid-l53-two-flow-depth-v1";
inline constexpr std::uint64_t kInitialOutput =
    0x7fffffff7fffffffull;

inline std::uint64_t Mix(std::uint64_t value) {
  value += 0x9E3779B97F4A7C15ull;
  value = (value ^ (value >> 30u)) * 0xBF58476D1CE4E5B9ull;
  value = (value ^ (value >> 27u)) * 0x94D049BB133111EBull;
  return value ^ (value >> 31u);
}

inline void Store16(std::vector<std::uint8_t>* bytes, std::size_t offset,
                    std::uint16_t value) {
  (*bytes)[offset] = static_cast<std::uint8_t>(value);
  (*bytes)[offset + 1u] = static_cast<std::uint8_t>(value >> 8u);
}

inline void Store32(std::vector<std::uint8_t>* bytes, std::size_t offset,
                    std::uint32_t value) {
  for (std::size_t lane = 0; lane < 4u; ++lane) {
    (*bytes)[offset + lane] =
        static_cast<std::uint8_t>(value >> (lane * 8u));
  }
}

inline void Store64(std::vector<std::uint8_t>* bytes, std::size_t offset,
                    std::uint64_t value) {
  for (std::size_t lane = 0; lane < 8u; ++lane) {
    (*bytes)[offset + lane] =
        static_cast<std::uint8_t>(value >> (lane * 8u));
  }
}

inline std::vector<std::uint8_t> FlowF32(std::uint64_t seed) {
  constexpr std::array<std::uint32_t, 9> values = {
      0xbc000000u,  // -1/128
      0xbb800000u,  // -1/256
      0xbb000000u,  // -1/512
      0xba800000u,  // -1/1024
      0x00000000u,
      0x3a800000u,  // 1/1024
      0x3b000000u,  // 1/512
      0x3b800000u,  // 1/256
      0x3c000000u,  // 1/128
  };
  std::vector<std::uint8_t> result(kFlowF32Bytes);
  for (std::size_t pixel = 0; pixel < kPixelCount; ++pixel) {
    const auto mixed = Mix(seed + pixel * 0xD1B54A32D192ED03ull);
    Store32(&result, pixel * 8u, values[mixed % values.size()]);
    Store32(&result, pixel * 8u + 4u,
            values[(mixed >> 11u) % values.size()]);
  }
  return result;
}

inline std::vector<std::uint8_t> FlowF16(std::uint64_t seed,
                                         bool pixel_flow) {
  constexpr std::array<std::uint16_t, 9> small_values = {
      0xa000, 0x9c00, 0x9800, 0x9400, 0x0000,
      0x1400, 0x1800, 0x1c00, 0x2000,
  };
  constexpr std::array<std::uint16_t, 9> pixel_values = {
      0xc400, 0xc000, 0xbc00, 0xb800, 0x0000,
      0x3800, 0x3c00, 0x4000, 0x4400,
  };
  const auto& values = pixel_flow ? pixel_values : small_values;
  std::vector<std::uint8_t> result(kFlowF16Bytes);
  for (std::size_t pixel = 0; pixel < kPixelCount; ++pixel) {
    const auto mixed = Mix(seed + pixel * 0x94D049BB133111EBull);
    Store16(&result, pixel * 4u, values[mixed % values.size()]);
    Store16(&result, pixel * 4u + 2u,
            values[(mixed >> 9u) % values.size()]);
  }
  return result;
}

inline std::vector<std::uint8_t> Depth(std::uint64_t seed) {
  constexpr std::array<std::uint32_t, 6> values = {
      0x3e800000u,  // 0.25
      0x3f000000u,  // 0.5
      0x3f400000u,  // 0.75
      0x3f800000u,  // 1.0
      0x3fc00000u,  // 1.5
      0x40000000u,  // 2.0
  };
  std::vector<std::uint8_t> result(kDepthBytes);
  for (std::uint32_t y = 0; y < kHeight; ++y) {
    for (std::uint32_t x = 0; x < kWidth; ++x) {
      const auto tile = std::uint64_t{x / 16u} +
                        std::uint64_t{3u * (y / 16u)};
      const auto mixed = Mix(seed + tile * 0x9E3779B97F4A7C15ull);
      Store32(&result, (std::size_t{y} * kWidth + x) * 4u,
              values[mixed % values.size()]);
    }
  }
  return result;
}

inline std::vector<std::uint8_t> InitialOutput() {
  std::vector<std::uint8_t> result(kOutputBytes);
  for (std::size_t pixel = 0; pixel < kPixelCount; ++pixel) {
    Store64(&result, pixel * 8u, kInitialOutput);
  }
  return result;
}

inline std::vector<std::uint8_t> CurrToPrevFlow() {
  return FlowF32(0x243F6A8885A308D3ull);
}

inline std::vector<std::uint8_t> PrevToCurrFlow(bool pixel_flow) {
  return FlowF16(0x13198A2E03707344ull, pixel_flow);
}

inline std::vector<std::uint8_t> CurrDepth() {
  return Depth(0xA4093822299F31D0ull);
}

inline std::vector<std::uint8_t> PrevDepth() {
  return Depth(0x082EFA98EC4E6C89ull);
}

}  // namespace fg_mid_l53
