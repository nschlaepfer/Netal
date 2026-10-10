#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace fg_custom_conv_pre {

inline constexpr std::uint32_t kInputWidth = 640;
inline constexpr std::uint32_t kInputHeight = 368;
inline constexpr std::uint32_t kInputChannels = 10;
inline constexpr std::uint32_t kOutputWidth = 320;
inline constexpr std::uint32_t kOutputHeight = 184;
inline constexpr std::uint32_t kOutputChannels = 32;
inline constexpr std::size_t kInputBytes =
    std::size_t{kInputWidth} * kInputHeight * kInputChannels * 2;
inline constexpr std::size_t kOutputBytes =
    std::size_t{kOutputWidth} * kOutputHeight * kOutputChannels * 2;
inline constexpr std::size_t kWeightBytes = 9216;
inline constexpr std::size_t kBiasBytes = 64;
inline constexpr std::size_t kModuleBytes = 4192;
inline constexpr char kModuleSha256[] =
    "a371abef8e41995064c25e171d0f2f9119a24ad4b671041be16c7b639a7c68a2";
inline constexpr char kWeightSha256[] =
    "14058e1781a3e362cd9f0106cf5c3fcd985ede67f1391ff3dfc73bceeceeae38";
inline constexpr char kBiasSha256[] =
    "349bbe2fcd8fc31407bfe238f4889871f0d64dd593b4146b59eca837fcc9a77f";
inline constexpr char kManifestSha256[] =
    "2a1179d4a25be030f772cd90ca7d017ed6eac9291917f7d43ea9b0cd5ac6a294";

inline std::uint64_t SplitMix64(std::uint64_t value) {
  value += 0x9E3779B97F4A7C15ull;
  value = (value ^ (value >> 30u)) * 0xBF58476D1CE4E5B9ull;
  value = (value ^ (value >> 27u)) * 0x94D049BB133111EBull;
  return value ^ (value >> 31u);
}

inline std::vector<std::uint8_t> GenerateInput() {
  constexpr std::array<std::uint16_t, 11> values = {
      0x0000, 0x2800, 0xa800, 0x2c00, 0xac00, 0x3000,
      0xb000, 0x3400, 0xb400, 0x3800, 0xb800,
  };
  std::vector<std::uint8_t> result(kInputBytes);
  constexpr std::uint64_t seed = 0x243F6A8885A308D3ull;
  constexpr std::uint64_t step = 0xD1B54A32D192ED03ull;
  for (std::size_t index = 0; index < result.size() / 2; ++index) {
    const std::uint16_t bits =
        values[SplitMix64(seed + std::uint64_t{index} * step) % values.size()];
    result[index * 2] = static_cast<std::uint8_t>(bits);
    result[index * 2 + 1] = static_cast<std::uint8_t>(bits >> 8u);
  }
  return result;
}

}  // namespace fg_custom_conv_pre
