#pragma once

#include "module10_contract.h"
#include "module9_contract.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace fg_distortion_connected {

inline constexpr std::uint32_t kWidth = 320;
inline constexpr std::uint32_t kHeight = 180;
inline constexpr std::size_t kPixels = std::size_t{kWidth} * kHeight;
inline constexpr std::size_t kRgba16Bytes = kPixels * 8u;
inline constexpr std::size_t kRg16Bytes = kPixels * 4u;
inline constexpr std::size_t kR32Bytes = kPixels * 4u;
inline constexpr std::size_t kR8Bytes = kPixels;
inline constexpr char kInputGenerator[] =
    "fg-distortion-six-launch-independent-nonzero-v1";

inline std::vector<std::uint8_t> Distortion(std::uint32_t variant) {
  std::vector<std::uint8_t> result(kRgba16Bytes);
  for (std::uint32_t y = 0; y < kHeight; ++y) {
    for (std::uint32_t x = 0; x < kWidth; ++x) {
      const std::size_t pixel = std::size_t{y} * kWidth + x;
      const int vx = int((x + variant) % 5u) - 2;
      const int vy = int((y + variant * 2u) % 5u) - 2;
      const int rx = int((x + y + variant * 3u) % 5u) - 2;
      const int ry = int((x + y * 2u + variant) % 5u) - 2;
      const float values[4] = {
          float(vx) / 320.0f,
          float(vy) / 180.0f,
          float(rx) / 320.0f,
          float(ry) / 180.0f,
      };
      for (std::size_t channel = 0; channel < 4u; ++channel) {
        fg_distortion_module10::Store16(
            result, pixel * 8u + channel * 2u,
            fg_early_l19::FloatToHalf(values[channel]));
      }
    }
  }
  return result;
}

inline std::vector<std::uint8_t> Mask(std::uint32_t variant) {
  std::vector<std::uint8_t> result(kR32Bytes);
  for (std::uint32_t y = 0; y < kHeight; ++y) {
    for (std::uint32_t x = 0; x < kWidth; ++x) {
      const std::size_t pixel = std::size_t{y} * kWidth + x;
      const std::uint32_t value =
          ((x * (17u + variant) + y * (31u + variant * 2u)) ^
           (0x5a5a5a5au + variant * 0x11111111u));
      fg_distortion_module9::S32(result, pixel * 4u, value);
    }
  }
  return result;
}

inline std::vector<std::uint8_t> Flow32() {
  std::vector<std::uint8_t> result(kPixels * 8u);
  for (std::uint32_t y = 0; y < kHeight; ++y) {
    for (std::uint32_t x = 0; x < kWidth; ++x) {
      const std::size_t pixel = std::size_t{y} * kWidth + x;
      const float vx = float(int(x % 5u) - 2) / 320.0f;
      const float vy = float(int(y % 5u) - 2) / 180.0f;
      fg_distortion_module9::S32(result, pixel * 8u,
                                 std::bit_cast<std::uint32_t>(vx));
      fg_distortion_module9::S32(result, pixel * 8u + 4u,
                                 std::bit_cast<std::uint32_t>(vy));
    }
  }
  return result;
}

inline std::vector<std::uint8_t> Flow16() {
  std::vector<std::uint8_t> result(kRg16Bytes);
  for (std::uint32_t y = 0; y < kHeight; ++y) {
    for (std::uint32_t x = 0; x < kWidth; ++x) {
      const std::size_t pixel = std::size_t{y} * kWidth + x;
      const float vx = float(int((x + y) % 5u) - 2) / 2.0f;
      const float vy = float(int((x + y * 2u) % 5u) - 2) / 2.0f;
      fg_distortion_module10::Store16(result, pixel * 4u,
                                      fg_early_l19::FloatToHalf(vx));
      fg_distortion_module10::Store16(result, pixel * 4u + 2u,
                                      fg_early_l19::FloatToHalf(vy));
    }
  }
  return result;
}

} // namespace fg_distortion_connected
