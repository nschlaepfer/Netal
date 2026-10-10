#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace fg_dynamic_module14 {

inline constexpr char kModuleSha256[] =
    "540f686ffcd4508212b19296992a69ed81ca95772fd229d0a252905c8015419c";
inline constexpr std::size_t kModuleBytes = 4584;

struct Resolution {
  const char* name;
  std::uint32_t width;
  std::uint32_t height;
  std::uint32_t grid_x;
  std::uint32_t grid_y;
  const char* raw_parameter_sha256;
  std::uint64_t fixture_salt;
};

inline constexpr std::array<Resolution, 4> kResolutions{{
    {"720p", 640, 360, 40, 23,
     "e1f743baeb62a509d45537354a13acf4a8c3c61f6ea8f08d4469d46f70365c4f",
     0x7207207207207207ull},
    {"1080p", 960, 540, 60, 34,
     "36bfa4cc6d18f4fa17d4e0191bf4c4e6abdfebb07e865ff540d447319146fbfe",
     0x1080108010801080ull},
    {"1440p", 1280, 720, 80, 45,
     "7c9719e0b5d5ee388bc0d40de8c17f52a3cedd2c60125804b07ad279817e80d6",
     0x1440144014401440ull},
    // The captured 4K configuration deliberately selects a 960x540 internal
    // module-14 surface.  Do not substitute the presentation dimensions.
    {"4k", 960, 540, 60, 34,
     "36bfa4cc6d18f4fa17d4e0191bf4c4e6abdfebb07e865ff540d447319146fbfe",
     0x4b4b4b4b4b4b4b4bull},
}};

inline const Resolution& FindResolution(std::string_view name) {
  for (const auto& resolution : kResolutions) {
    if (name == resolution.name) return resolution;
  }
  throw std::runtime_error("unknown module14 resolution");
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

inline std::vector<std::uint8_t> InputActive(const Resolution& resolution) {
  constexpr std::array<std::uint16_t, 25> values = {
      0x0000, 0x8000, 0x0001, 0x8001, 0x1400, 0x9400, 0x2000,
      0xa000, 0x2800, 0xa800, 0x3000, 0xb000, 0x3400, 0xb400,
      0x3555, 0x3800, 0xb800, 0x3c00, 0xbc00, 0x4000, 0xc000,
      0x4200, 0x4400, 0x7bff, 0xfbff,
  };
  const std::size_t pixel_count =
      std::size_t{resolution.width} * resolution.height;
  std::vector<std::uint8_t> result(pixel_count * 8u);
  for (std::size_t index = 0; index < pixel_count * 4u; ++index) {
    const auto mixed = Mix(resolution.fixture_salt +
                           index * 0xD1B54A32D192ED03ull);
    Store16(&result, index * 2u, values[mixed % values.size()]);
  }
  // A deterministic pre-store division counterexample: approximate reciprocal
  // multiplication and precise division differ by one FP32 ULP before the RZ
  // surface conversion.  It is generated independently on both backends.
  Store16(&result, 0, 0x0000);
  Store16(&result, 2, 0x0001);
  Store16(&result, 4, 0x1400);
  Store16(&result, 6, 0x3c00);
  return result;
}

inline std::vector<std::uint8_t> GuardedInput(
    const Resolution& resolution, std::uint8_t poison) {
  auto result = InputActive(resolution);
  const std::size_t active_bytes = result.size();
  result.resize(active_bytes + std::size_t{resolution.width} * 8u);
  for (std::size_t index = active_bytes; index < result.size(); ++index) {
    result[index] = static_cast<std::uint8_t>(poison ^
        std::uint8_t(Mix(resolution.fixture_salt + index) >> 56u));
  }
  return result;
}

inline std::vector<std::uint8_t> DeadMask(
    const Resolution& resolution, std::uint8_t poison) {
  const std::size_t bytes =
      std::size_t{resolution.width} * (resolution.height + 1u);
  std::vector<std::uint8_t> result(bytes);
  for (std::size_t index = 0; index < bytes; ++index) {
    result[index] = static_cast<std::uint8_t>(poison ^
        std::uint8_t(Mix(0x13198A2E03707344ull +
                         resolution.fixture_salt + index) >> 56u));
  }
  return result;
}

inline std::vector<std::uint8_t> PoisonedOutput(
    const Resolution& resolution, std::uint8_t poison) {
  return std::vector<std::uint8_t>(
      std::size_t{resolution.width} * (resolution.height + 1u) * 4u,
      poison);
}

}  // namespace fg_dynamic_module14
