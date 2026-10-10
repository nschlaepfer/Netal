#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace fg_early_l7_l17 {

inline constexpr std::uint32_t kWidth = 320;
inline constexpr std::uint32_t kHeight = 180;
inline constexpr std::size_t kPixelCount = std::size_t{kWidth} * kHeight;
inline constexpr std::array<std::uint32_t, 4> kLevelWidths = {160, 80, 40, 20};
inline constexpr std::array<std::uint32_t, 4> kLevelHeights = {90, 45, 23, 12};
inline constexpr std::array<std::size_t, 4> kLevelBytes = {
    160u * 90u * 3u * 2u,
    80u * 45u * 3u * 2u,
    40u * 23u * 3u * 2u,
    20u * 12u * 3u * 2u,
};
inline constexpr std::size_t kMaskBytes = kPixelCount;
inline constexpr std::size_t kOutputBytes = kPixelCount * 2u * 2u;
inline constexpr std::size_t kModuleBytes = 56480;
inline constexpr char kModuleSha256[] =
    "d787cf7b861295d1b4c6a80cb374230de793aa69f288a7e724bf59278bc4a62d";

enum class Fixture { kDense, kSparse };

inline std::string FixtureName(Fixture fixture) {
  return fixture == Fixture::kDense ? "dense" : "sparse";
}

inline Fixture ParseFixture(const std::string& name) {
  if (name == "dense") return Fixture::kDense;
  if (name == "sparse") return Fixture::kSparse;
  throw std::runtime_error("unknown module23 fixture: " + name);
}

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

inline std::vector<std::uint8_t> Level(std::size_t level) {
  constexpr std::array<std::uint16_t, 11> values = {
      0x0000, 0x2800, 0xa800, 0x3000, 0xb000, 0x3400,
      0xb400, 0x3800, 0xb800, 0x3c00, 0xbc00,
  };
  constexpr std::array<std::uint16_t, 6> weights = {
      0x0000, 0x3000, 0x3400, 0x3800, 0x3a00, 0x3c00,
  };
  std::vector<std::uint8_t> result(kLevelBytes[level]);
  const std::size_t pixels = std::size_t{kLevelWidths[level]} *
                             kLevelHeights[level];
  for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
    const auto mixed = Mix(0x243F6A8885A308D3ull + level * 0x100000001B3ull +
                           pixel * 0xD1B54A32D192ED03ull);
    Store16(&result, pixel * 6u, values[mixed % values.size()]);
    Store16(&result, pixel * 6u + 2u,
            values[(mixed >> 10u) % values.size()]);
    Store16(&result, pixel * 6u + 4u,
            weights[(mixed >> 20u) % weights.size()]);
  }
  return result;
}

inline std::array<std::vector<std::uint8_t>, 4> Levels() {
  return {Level(0), Level(1), Level(2), Level(3)};
}

inline std::vector<std::uint8_t> Mask(Fixture fixture) {
  std::vector<std::uint8_t> result(kMaskBytes, 1u);
  if (fixture == Fixture::kSparse) {
    for (std::size_t pixel = 0; pixel < result.size(); ++pixel) {
      const auto mixed = Mix(0x13198A2E03707344ull +
                             pixel * 0x94D049BB133111EBull);
      result[pixel] = (mixed & 7u) < 5u ? 1u : 0u;
    }
  }
  return result;
}

}  // namespace fg_early_l7_l17
