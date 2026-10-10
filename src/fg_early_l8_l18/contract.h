#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace fg_early_l8_l18 {

inline constexpr std::uint32_t kWidth = 320;
inline constexpr std::uint32_t kHeight = 180;
inline constexpr std::size_t kPixelCount = std::size_t{kWidth} * kHeight;
inline constexpr std::size_t kInputBytes = kPixelCount * 2u * 2u;
inline constexpr std::size_t kOutputBytes = kPixelCount;
inline constexpr std::size_t kModuleBytes = 6640;
inline constexpr char kModuleSha256[] =
    "28fe0e153d1ffc80b54995c022f18395c5556f3c041cc9d724bfe7e70dc4e824";

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
  constexpr std::array<std::uint16_t, 9> values = {
      0xc400, 0xc000, 0xbc00, 0xb800, 0x0000,
      0x3800, 0x3c00, 0x4000, 0x4400,
  };
  std::vector<std::uint8_t> result(kInputBytes);
  for (std::size_t pixel = 0; pixel < kPixelCount; ++pixel) {
    const auto mixed = Mix(0x243F6A8885A308D3ull +
                           pixel * 0xD1B54A32D192ED03ull);
    Store16(&result, pixel * 4u, values[mixed % values.size()]);
    Store16(&result, pixel * 4u + 2u,
            values[(mixed >> 9u) % values.size()]);
  }
  return result;
}

}  // namespace fg_early_l8_l18
