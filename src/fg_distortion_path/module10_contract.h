#pragma once

#include "../fg_early_l19/contract.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace fg_distortion_module10 {

inline constexpr std::uint32_t kWidth = 320;
inline constexpr std::uint32_t kHeight = 180;
inline constexpr std::size_t kPixels = std::size_t{kWidth} * kHeight;
inline constexpr std::size_t kRgba16Bytes = kPixels * 8u;
inline constexpr std::size_t kModuleBytes = 5984;
inline constexpr char kModuleSha256[] =
    "0f7627574281429873003bd700c909d09c895770d8464af0914559d866fe9d3c";
inline constexpr char kResetRawParameterSha256[] =
    "e170b309e44af03949713aa49e319aa27448cc33a002f7a415006d2b3acbfa71";
inline constexpr char kGeneratedRawParameterSha256[] =
    "9c50edeea15f0b497785e9cb6a7e8ca9324992df32f12ce3bba553f55199b4a5";
inline constexpr char kResetAddressFreeParameterSha256[] =
    "a9709a973e4e90362fe8752cfaf629fe4653a2258ffc936a7e58c1bed0b59c16";
inline constexpr char kGeneratedAddressFreeParameterSha256[] =
    "5e7ccebe87f838586b00a647ff7f463d3626a9c5f13432bcb6cab96e12276f1e";
inline constexpr char kFixtureAddressFreeParameterSha256[] =
    "bedcc830e93471dc1685e06e218bad3a5b718f3b02ff720ad51ce555319ca514";
inline constexpr char kInputGenerator[] =
    "fg-distortion-module10-independent-nonzero-v1";

inline void Store16(std::vector<std::uint8_t> &bytes, std::size_t offset,
                    std::uint16_t value) {
  bytes[offset] = static_cast<std::uint8_t>(value);
  bytes[offset + 1u] = static_cast<std::uint8_t>(value >> 8u);
}

inline std::vector<std::uint8_t> GenerateInput() {
  std::vector<std::uint8_t> result(kRgba16Bytes);
  for (std::uint32_t y = 0; y < kHeight; ++y) {
    for (std::uint32_t x = 0; x < kWidth; ++x) {
      const std::size_t pixel = std::size_t{y} * kWidth + x;
      const float values[4] = {
          float(int(x % 17u) - 8) / 64.0f,
          float(int(y % 13u) - 6) / 32.0f,
          float(int((x + y) % 19u) - 9) / 32.0f,
          1.0f,
      };
      for (std::size_t channel = 0; channel < 4u; ++channel) {
        Store16(result, pixel * 8u + channel * 2u,
                fg_early_l19::FloatToHalf(values[channel]));
      }
    }
  }
  return result;
}

} // namespace fg_distortion_module10
