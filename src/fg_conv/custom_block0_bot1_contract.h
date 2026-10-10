#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace fg_custom_block0_bot1 {

inline constexpr std::size_t kInputPlaneBytes = 92u * 160u * 32u * 2u;
inline constexpr std::size_t kOutput0Bytes = 184u * 320u * 4u * 2u;
inline constexpr std::size_t kOutput1Bytes = 184u * 320u * 1u * 2u;
inline constexpr std::size_t kOutput2Bytes = 184u * 320u * 3u * 2u;
inline constexpr std::size_t kOutputBytes =
    kOutput0Bytes + kOutput1Bytes + kOutput2Bytes;
inline constexpr std::size_t kWeightBytes = 4608;
inline constexpr std::size_t kBiasBytes = 16;
inline constexpr std::size_t kModuleBytes = 6976;
inline constexpr char kModuleSha256[] =
    "4d35017769f6338b6f8ba3ac3079756c32410eeacca7f817e48ad5d7dc59eb3c";
inline constexpr char kWeightSha256[] =
    "4d477655f6ec78e0e6a0d3c41e0848e537091c0c311fb04d986ad8891823e4e8";
inline constexpr char kBiasSha256[] =
    "0a9a85630034bb3dc5442cd52381424c1e7b9c676014b764477341005cb2f998";
inline constexpr char kManifestSha256[] =
    "2a1179d4a25be030f772cd90ca7d017ed6eac9291917f7d43ea9b0cd5ac6a294";

inline std::uint64_t Mix(std::uint64_t value) {
  value += 0x9E3779B97F4A7C15ull;
  value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ull;
  value = (value ^ (value >> 27)) * 0x94D049BB133111EBull;
  return value ^ (value >> 31);
}

inline std::vector<std::uint8_t>Input(std::uint64_t seed) {
  constexpr std::array<std::uint16_t, 11> values = {
      0,      0x2800, 0xa800, 0x2c00, 0xac00, 0x3000,
      0xb000, 0x3400, 0xb400, 0x3800, 0xb800};
  std::vector<std::uint8_t> result(kInputPlaneBytes);
  for (std::size_t index = 0; index < result.size() / 2; ++index) {
    const auto bits =
        values[Mix(seed + std::uint64_t{index} * 0xD1B54A32D192ED03ull) %
               values.size()];
    result[2 * index] = static_cast<std::uint8_t>(bits);
    result[2 * index + 1] = static_cast<std::uint8_t>(bits >> 8);
  }
  return result;
}

inline std::vector<std::uint8_t>Input0() {
  return Input(0x8CB92BA72F3D8DD7ull);
}
inline std::vector<std::uint8_t>Input1() {
  return Input(0xDB4F0B9175AE2165ull);
}
inline std::vector<std::uint8_t>Input2() {
  return Input(0xBBE0563303A4615Full);
}

}  // namespace fg_custom_block0_bot1
