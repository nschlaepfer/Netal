#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace fg_early_l19 {

inline constexpr std::uint32_t kInputWidth = 320;
inline constexpr std::uint32_t kInputHeight = 180;
inline constexpr std::uint32_t kTensorWidth = 160;
inline constexpr std::uint32_t kTensorHeight = 96;
inline constexpr std::uint32_t kTensorChannels = 48;
inline constexpr std::size_t kTensorBytes =
    std::size_t{kTensorWidth} * kTensorHeight * kTensorChannels * 2u;
inline constexpr std::size_t kMaskOutputBytes =
    std::size_t{kInputWidth} * kInputHeight;
inline constexpr std::size_t kVectorOutputBytes =
    std::size_t{kInputWidth} * kInputHeight * 4u;
inline constexpr std::size_t kModuleBytes = 14448;
inline constexpr char kModuleSha256[] =
    "3f2308a693730b44c00e064ed25dab3b73a420f2e92d215ef733c4e10a30439c";
inline constexpr char kFrozenRawParameterSha256[] =
    "6f8656c06f9d30222e3d26cefb0594433d386eaa79bba79eef95914fbe2314d5";
inline constexpr char kFrozenAddressFreeParameterSha256[] =
    "9211cb3fa6156db90144c2d845c4ad0d54b0592f7d057d56b43a2cae6c5eb07d";
inline constexpr char kFixtureAddressFreeParameterSha256[] =
    "5dd08659e6c4f5724657768ea8da24eb0d1a61431d64bd9b9df2e69b997cf890";
inline constexpr char kAllocation29FirstFrameEvidenceRootSha256[] =
    "c1435760ec194050aa1dd4dfd2d82f71e9aa7874ce550328a4eb08cd3d2a0018";
inline constexpr char kAllocation29TemporalEvidenceRootSha256[] =
    "ebb4c6a84a4cf0e607ef62b39a55dc46ec0ec059f3d02bc1b77e4d084ef9a79b";
inline constexpr char kInputGenerator[] =
    "fg-early-l19-independent-ten-inputs-v1";

enum class Fixture { kFirstFrame, kFanIn };

inline Fixture ParseFixture(std::string_view name) {
  if (name == "first-frame") return Fixture::kFirstFrame;
  if (name == "fan-in") return Fixture::kFanIn;
  throw std::runtime_error("unsupported L19 fixture");
}

inline const char* FixtureName(Fixture fixture) {
  return fixture == Fixture::kFirstFrame ? "first-frame" : "fan-in";
}

inline std::uint64_t SplitMix64(std::uint64_t value) {
  value += 0x9e3779b97f4a7c15ull;
  value = (value ^ (value >> 30u)) * 0xbf58476d1ce4e5b9ull;
  value = (value ^ (value >> 27u)) * 0x94d049bb133111ebull;
  return value ^ (value >> 31u);
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
  const auto bits = std::bit_cast<std::uint32_t>(value);
  for (unsigned byte = 0; byte < 4u; ++byte) {
    (*bytes)[offset + byte] =
        static_cast<std::uint8_t>(bits >> (byte * 8u));
  }
}

inline std::vector<std::uint8_t> Half4Input(std::uint64_t salt) {
  std::vector<std::uint8_t> result(
      std::size_t{kInputWidth} * kInputHeight * 8u);
  for (std::uint32_t y = 0; y < kInputHeight; ++y) {
    for (std::uint32_t x = 0; x < kInputWidth; ++x) {
      const std::size_t pixel = std::size_t{y} * kInputWidth + x;
      for (std::uint32_t channel = 0; channel < 4u; ++channel) {
        const auto mixed = SplitMix64(salt ^ pixel * 0x9e3779b97f4a7c15ull ^
                                      std::uint64_t{channel + 1u} *
                                          0xbf58476d1ce4e5b9ull);
        const float value = static_cast<float>(1u + (mixed % 15u)) / 16.0f;
        Store16(&result, pixel * 8u + channel * 2u, FloatToHalf(value));
      }
    }
  }
  return result;
}

inline std::vector<std::uint8_t> BinaryMask(std::uint64_t salt) {
  std::vector<std::uint8_t> result(
      std::size_t{kInputWidth} * kInputHeight);
  for (std::size_t pixel = 0; pixel < result.size(); ++pixel) {
    result[pixel] = (SplitMix64(salt ^ pixel) & 3u) == 0u ? 0u : 255u;
  }
  return result;
}

inline std::vector<std::uint8_t> Half2Input(std::uint64_t salt,
                                            bool zero) {
  std::vector<std::uint8_t> result(
      std::size_t{kInputWidth} * kInputHeight * 4u);
  for (std::size_t pixel = 0; pixel < result.size() / 4u; ++pixel) {
    for (std::uint32_t channel = 0; channel < 2u; ++channel) {
      float value = 0.0f;
      if (!zero) {
        const auto mixed = SplitMix64(salt ^ pixel * 0x94d049bb133111ebull ^
                                      channel * 0x9e3779b97f4a7c15ull);
        value = static_cast<float>(static_cast<int>(mixed % 9u) - 4) / 8.0f;
      }
      Store16(&result, pixel * 4u + channel * 2u, FloatToHalf(value));
    }
  }
  return result;
}

inline std::vector<std::uint8_t> Half1Input(std::uint64_t salt,
                                            bool uniform_half) {
  std::vector<std::uint8_t> result(
      std::size_t{kInputWidth} * kInputHeight * 2u);
  for (std::size_t pixel = 0; pixel < result.size() / 2u; ++pixel) {
    float value = 0.5f;
    if (!uniform_half) {
      value = static_cast<float>(1u + (SplitMix64(salt ^ pixel) % 15u)) /
              16.0f;
    }
    Store16(&result, pixel * 2u, FloatToHalf(value));
  }
  return result;
}

inline std::vector<std::uint8_t> Float2Input(bool zero) {
  std::vector<std::uint8_t> result(
      std::size_t{kInputWidth} * kInputHeight * 8u);
  for (std::size_t pixel = 0; pixel < result.size() / 8u; ++pixel) {
    float x = 0.0f;
    float y = 0.0f;
    if (!zero) {
      x = static_cast<float>(static_cast<int>(pixel % 3u) - 1) *
          std::bit_cast<float>(0x3b4ccccdu);
      y = static_cast<float>(static_cast<int>((pixel / 3u) % 3u) - 1) *
          std::bit_cast<float>(0x3bb60b61u);
    }
    Store32(&result, pixel * 8u, x);
    Store32(&result, pixel * 8u + 4u, y);
  }
  return result;
}

struct Inputs {
  std::array<std::vector<std::uint8_t>, 10> textures;
};

inline Inputs GenerateInputs(Fixture fixture) {
  const bool first_frame = fixture == Fixture::kFirstFrame;
  Inputs result;
  result.textures[0] = Half4Input(0x243f6a8885a308d3ull);
  result.textures[1] = Half4Input(0x13198a2e03707344ull);
  result.textures[2] = BinaryMask(0xa4093822299f31d0ull);
  result.textures[3] = BinaryMask(0x082efa98ec4e6c89ull);
  result.textures[4] = Half2Input(0x452821e638d01377ull, false);
  result.textures[5] = Half1Input(0xbe5466cf34e90c6cull, false);
  result.textures[6] = Half1Input(0xc0ac29b7c97c50ddull, first_frame);
  result.textures[7] = Float2Input(first_frame);
  result.textures[8] = Half2Input(0x3f84d5b5b5470917ull, first_frame);
  result.textures[9] = BinaryMask(0x9216d5d98979fb1bull);
  return result;
}

inline constexpr std::array<std::size_t, 10> kInputBytes = {
    std::size_t{kInputWidth} * kInputHeight * 8u,
    std::size_t{kInputWidth} * kInputHeight * 8u,
    std::size_t{kInputWidth} * kInputHeight,
    std::size_t{kInputWidth} * kInputHeight,
    std::size_t{kInputWidth} * kInputHeight * 4u,
    std::size_t{kInputWidth} * kInputHeight * 2u,
    std::size_t{kInputWidth} * kInputHeight * 2u,
    std::size_t{kInputWidth} * kInputHeight * 8u,
    std::size_t{kInputWidth} * kInputHeight * 4u,
    std::size_t{kInputWidth} * kInputHeight,
};

}  // namespace fg_early_l19
