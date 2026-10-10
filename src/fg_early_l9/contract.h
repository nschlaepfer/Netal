#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace fg_early_l9 {

inline constexpr std::uint32_t kWidth = 320;
inline constexpr std::uint32_t kHeight = 180;
inline constexpr std::size_t kPixelCount = std::size_t{kWidth} * kHeight;
inline constexpr std::size_t kInputBytes = kPixelCount * 4u * 2u;
inline constexpr std::size_t kDeadMaskBytes = kPixelCount;
inline constexpr std::size_t kOutputBytes = kPixelCount * 2u * 2u;
inline constexpr std::size_t kModuleBytes = 4584;
inline constexpr char kModuleSha256[] =
    "540f686ffcd4508212b19296992a69ed81ca95772fd229d0a252905c8015419c";

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

inline std::vector<std::uint8_t> Input() {
  constexpr std::array<std::uint16_t, 21> values = {
      0x0000, 0x8000, 0x0001, 0x8001, 0x1400, 0x9400, 0x2000,
      0xa000, 0x2800, 0xa800, 0x3000, 0xb000, 0x3400, 0xb400,
      0x3800, 0xb800, 0x3c00, 0xbc00, 0x4000, 0xc000, 0x4200,
  };
  std::vector<std::uint8_t> result(kInputBytes);
  for (std::size_t index = 0; index < kPixelCount * 4u; ++index) {
    const auto mixed = Mix(0x243F6A8885A308D3ull +
                           index * 0xD1B54A32D192ED03ull);
    Store16(&result, index * 2u, values[mixed % values.size()]);
  }
  return result;
}

inline std::vector<std::uint8_t> DeadMask(std::uint8_t poison) {
  std::vector<std::uint8_t> result(kDeadMaskBytes);
  for (std::size_t index = 0; index < result.size(); ++index) {
    result[index] = static_cast<std::uint8_t>(
        poison ^ std::uint8_t(Mix(0x13198A2E03707344ull + index) >> 56u));
  }
  return result;
}

}  // namespace fg_early_l9
