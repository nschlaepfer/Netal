#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace fg_early_l4_l14 {

inline constexpr std::uint32_t kInputWidth = 640;
inline constexpr std::uint32_t kInputHeight = 360;
inline constexpr std::uint32_t kOutputWidth = 320;
inline constexpr std::uint32_t kOutputHeight = 180;
inline constexpr std::uint32_t kOutputRowsWithCanary = 181;
inline constexpr std::size_t kModuleBytes = 12632;
inline constexpr char kModuleSha256[] =
    "9e772df29e9434e4eff1edf1cde32706ea641c2b0156a459a080565a7483f381";
inline constexpr char kResetTraceRawParameterSha256[] =
    "2b822288e9427f60708f9e0f537f25484eb3c11309efc3755eccb693879249ce";
inline constexpr char kGeneratedTraceRawParameterSha256[] =
    "35284822ea5908b342267ed168ea193541f1b5d66a168182dfe0e26dcdb6a83c";
inline constexpr char kResetAddressFreeParameterSha256[] =
    "59686bcaf449fd6d217aceb7c6faef7e1b2de50561eb504a9b1c8deed072ac20";
inline constexpr char kGeneratedAddressFreeParameterSha256[] =
    "c884fbfbf81a56cc50b3271754d228d2e2c3ba7eb1f05c5565318dbb291a21af";
inline constexpr char kResetFixtureAddressFreeParameterSha256[] =
    "e6ea4f8ae223c3b1683356c5168e4931ca00dc0f734acbfd678724a8b1d52c65";
inline constexpr char kGeneratedFixtureAddressFreeParameterSha256[] =
    "29144317759be7f820383ed92ea0191a1f157a02dcf3e74f3eea2b80c81123b6";
inline constexpr char kInputGenerator[] =
    "fg-early-l4-l14-independent-inputs-v1";

enum class Fixture { kBranchGrid, kCenterTie };

inline Fixture ParseFixture(std::string_view name) {
  if (name == "branch-grid") return Fixture::kBranchGrid;
  if (name == "center-tie") return Fixture::kCenterTie;
  throw std::runtime_error("unsupported L4/L14 fixture");
}

inline const char* FixtureName(Fixture fixture) {
  return fixture == Fixture::kBranchGrid ? "branch-grid" : "center-tie";
}

inline std::uint16_t FloatToHalf(float value) {
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  const std::uint32_t sign = (bits >> 16u) & 0x8000u;
  const std::uint32_t exponent = (bits >> 23u) & 0xffu;
  std::uint32_t mantissa = bits & 0x7fffffu;
  if (exponent == 0xffu) {
    return static_cast<std::uint16_t>(
        sign | (mantissa == 0u ? 0x7c00u : 0x7e00u));
  }
  int half_exponent = static_cast<int>(exponent) - 127 + 15;
  if (half_exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
  if (half_exponent <= 0) {
    if (half_exponent < -10) return static_cast<std::uint16_t>(sign);
    mantissa |= 0x800000u;
    const unsigned shift = static_cast<unsigned>(14 - half_exponent);
    std::uint32_t rounded = mantissa >> shift;
    const std::uint32_t remainder = mantissa & ((1u << shift) - 1u);
    const std::uint32_t halfway = 1u << (shift - 1u);
    if (remainder > halfway || (remainder == halfway && (rounded & 1u))) {
      ++rounded;
    }
    return static_cast<std::uint16_t>(sign | rounded);
  }
  std::uint32_t rounded = mantissa >> 13u;
  const std::uint32_t remainder = mantissa & 0x1fffu;
  if (remainder > 0x1000u ||
      (remainder == 0x1000u && (rounded & 1u))) {
    ++rounded;
    if (rounded == 0x400u) {
      rounded = 0;
      ++half_exponent;
      if (half_exponent >= 31) {
        return static_cast<std::uint16_t>(sign | 0x7c00u);
      }
    }
  }
  return static_cast<std::uint16_t>(
      sign | (static_cast<std::uint32_t>(half_exponent) << 10u) | rounded);
}

inline void Store16(std::vector<std::uint8_t>* bytes, std::size_t offset,
                    std::uint16_t value) {
  (*bytes)[offset] = static_cast<std::uint8_t>(value);
  (*bytes)[offset + 1u] = static_cast<std::uint8_t>(value >> 8u);
}

inline void Store32(std::vector<std::uint8_t>* bytes, std::size_t offset,
                    float value) {
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  for (unsigned byte = 0; byte < 4; ++byte) {
    (*bytes)[offset + byte] =
        static_cast<std::uint8_t>(bits >> (byte * 8u));
  }
}

inline std::vector<std::uint8_t> VectorInput(Fixture fixture) {
  std::vector<std::uint8_t> result(
      static_cast<std::size_t>(kInputWidth) * kInputHeight * 4u);
  for (std::uint32_t y = 0; y < kInputHeight; ++y) {
    for (std::uint32_t x = 0; x < kInputWidth; ++x) {
      const std::size_t offset =
          (static_cast<std::size_t>(y) * kInputWidth + x) * 4u;
      const int a = fixture == Fixture::kBranchGrid
                        ? static_cast<int>((x * 5u + y * 3u) % 17u) - 8
                        : static_cast<int>((x + y * 2u) % 9u) - 4;
      const int b = fixture == Fixture::kBranchGrid
                        ? static_cast<int>((x * 7u + y * 11u) % 13u) - 6
                        : static_cast<int>((x * 3u + y) % 11u) - 5;
      Store16(&result, offset, FloatToHalf(static_cast<float>(a) * 0.125f));
      Store16(&result, offset + 2u,
              FloatToHalf(static_cast<float>(b) * 0.125f));
    }
  }
  return result;
}

inline std::vector<std::uint8_t> ScalarInput(Fixture fixture) {
  std::vector<std::uint8_t> result(
      static_cast<std::size_t>(kInputWidth) * kInputHeight * 4u);
  for (std::uint32_t y = 0; y < kInputHeight; ++y) {
    for (std::uint32_t x = 0; x < kInputWidth; ++x) {
      float value = 0.5f;
      if (fixture == Fixture::kBranchGrid) {
        const std::uint32_t rank =
            (x * 73u + y * 151u + ((x ^ y) * 29u)) & 255u;
        value = static_cast<float>(64u + rank) * (1.0f / 512.0f);
      }
      Store32(&result,
              (static_cast<std::size_t>(y) * kInputWidth + x) * 4u,
              value);
    }
  }
  return result;
}

inline std::size_t VectorInputBytes() {
  return static_cast<std::size_t>(kInputWidth) * kInputHeight * 4u;
}

inline std::size_t ScalarInputBytes() {
  return static_cast<std::size_t>(kInputWidth) * kInputHeight * 4u;
}

inline std::size_t VectorOutputPayloadBytes() {
  return static_cast<std::size_t>(kOutputWidth) * kOutputHeight * 8u;
}

inline std::size_t DepthOutputPayloadBytes() {
  return static_cast<std::size_t>(kOutputWidth) * kOutputHeight * 4u;
}

}  // namespace fg_early_l4_l14
