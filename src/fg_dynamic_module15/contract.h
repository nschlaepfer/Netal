#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace fg_dynamic_module15 {

inline constexpr char kModuleSha256[] =
    "9e772df29e9434e4eff1edf1cde32706ea641c2b0156a459a080565a7483f381";
inline constexpr std::size_t kModuleBytes = 12632;
inline constexpr std::size_t kPackedParameterBytes = 160;
inline constexpr char kGenerator[] = "fg-dynamic-module15-branch-tie-v1";

struct Resolution {
  const char* name;
  std::uint32_t input_width;
  std::uint32_t input_height;
  std::uint32_t output_width;
  std::uint32_t output_height;
  std::uint32_t grid_x;
  std::uint32_t grid_y;
  std::uint32_t inverse_x_bits;
  std::uint32_t inverse_y_bits;
  std::uint32_t lower_x_bits;
  std::uint32_t lower_y_bits;
  std::uint32_t upper_x_bits;
  std::uint32_t upper_y_bits;
  std::uint32_t output_scale_x_bits;
  std::uint32_t output_scale_y_bits;
  const char* reset_raw_sha256;
  const char* generated_raw_sha256;
  const char* reset_address_free_sha256;
  const char* generated_address_free_sha256;
};

inline constexpr std::array<Resolution, 4> kResolutions = {{
    {"720p", 1280, 720, 640, 360, 40, 23,
     0x3a4ccccd, 0x3ab60b61, 0x39cccccd, 0x3a360b61,
     0x3f7fe666, 0x3f7fd27d, 0x3523d70a, 0x3601742e,
     "b07cb1a8589f53bb215567a4e350d095af1c349f835f2ce2fd193f881d261ae4",
     "52ae7695610f72ebceb61c096bb7c1789135656034d1ae94532086978838a383",
     "b0fce9db2b5718e69c816380a59408e5552b6be452fd8bcea7c5ffc9707e9930",
     "26575a19da94504c4bae6eb21a021c730bf882673849a084180ddb2c377f68cc"},
    {"1080p", 1920, 1080, 960, 540, 60, 34,
     0x3a088889, 0x3a72b9d6, 0x39888889, 0x39f2b9d6,
     0x3f7feeef, 0x3f7fe1a8, 0x3491a2b4, 0x356623e0,
     "2ee9d9d547bb7af218906304a1cb71b2bb30a78f41fa58b79b75c9c113c58e97",
     "cef79207030577bcf4cda92c0e404389eed4909b0f6e09c87f8bc1b4b43a2fcc",
     "1788f069350f9896fa7c7be1530ff90a8d6bad33c4379d3d91181e15e44b0472",
     "ae465a1b32cc29548c2942087f675cb362f070a299fcacf77920362807f32fc9"},
    {"1440p", 2560, 1440, 1280, 720, 80, 45,
     0x39cccccd, 0x3a360b61, 0x394ccccd, 0x39b60b61,
     0x3f7ff333, 0x3f7fe93e, 0x3423d70a, 0x3501742e,
     "a744a01df885f1abf569ecde2e16b74748af3fe9e01e4920d6bf9d6e4b5edecf",
     "78f11c9415b7aa4c97b4955d8aef62f65402fb980c65255b193a12724c8d997c",
     "316cb00aa4ae44a7d282330042a91dc462627582aa429ebb533fcfb9bbe23892",
     "9a3aa6c1755ae16a248dcc58ef129adb18303e4e992c2f88d0c7775d3ffd7269"},
    {"4k", 3840, 2160, 1920, 1080, 120, 68,
     0x39888889, 0x39f2b9d6, 0x39088889, 0x3972b9d6,
     0x3f7ff778, 0x3f7ff0d4, 0x3391a2b4, 0x346623e0,
     "8a2ed8dac885206589505eec4a1520fa0bc0fea4db842f2111580f7008372064",
     "93a4ed4b32798b1141a0fc0bf224da2ad43e29cc95dd254cb1233957bacbdc6d",
     "78a2dd677a28fa8ac23a4ebc8f67642a17cdb0b8647bfdb520e4f65035d0ad8e",
     "ff281a810b9297147ad4ea38c1c0e1899b5e07dcb5abeccd39123a6a57d4c896"},
}};

inline const Resolution& ParseResolution(const std::string& name) {
  for (const auto& resolution : kResolutions)
    if (name == resolution.name) return resolution;
  throw std::runtime_error("unknown resolution: " + name);
}

inline std::uint64_t Mix(std::uint64_t value) {
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
  if (exponent == 0xffu)
    return static_cast<std::uint16_t>(
        sign | (mantissa == 0u ? 0x7c00u : 0x7e00u));
  int half_exponent = static_cast<int>(exponent) - 127 + 15;
  if (half_exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
  if (half_exponent <= 0) {
    if (half_exponent < -10) return static_cast<std::uint16_t>(sign);
    mantissa |= 0x800000u;
    const unsigned shift = static_cast<unsigned>(14 - half_exponent);
    std::uint32_t rounded = mantissa >> shift;
    const std::uint32_t remainder = mantissa & ((1u << shift) - 1u);
    const std::uint32_t halfway = 1u << (shift - 1u);
    if (remainder > halfway || (remainder == halfway && (rounded & 1u)))
      ++rounded;
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
      if (half_exponent >= 31)
        return static_cast<std::uint16_t>(sign | 0x7c00u);
    }
  }
  return static_cast<std::uint16_t>(
      sign | (static_cast<std::uint32_t>(half_exponent) << 10u) | rounded);
}

inline void Store16(std::vector<std::uint8_t>* bytes, std::size_t offset,
                    std::uint16_t value) {
  std::memcpy(bytes->data() + offset, &value, sizeof(value));
}

inline void Store32(std::vector<std::uint8_t>* bytes, std::size_t offset,
                    float value) {
  std::memcpy(bytes->data() + offset, &value, sizeof(value));
}

inline std::vector<std::uint8_t> VectorInput(
    const Resolution& resolution, bool generated) {
  const std::size_t pixels =
      std::size_t{resolution.input_width} * resolution.input_height;
  std::vector<std::uint8_t> result(pixels * 4u);
  for (std::uint32_t y = 0; y < resolution.input_height; ++y) {
    for (std::uint32_t x = 0; x < resolution.input_width; ++x) {
      const std::size_t offset =
          (std::size_t{y} * resolution.input_width + x) * 4u;
      const auto mixed = Mix((generated ? 0x13198a2e03707344ull
                                       : 0x243f6a8885a308d3ull) ^
                             (std::uint64_t{y} << 32u) ^ x);
      const int a = static_cast<int>(mixed % 33u) - 16;
      const int b = static_cast<int>((mixed >> 11u) % 29u) - 14;
      Store16(&result, offset, FloatToHalf(static_cast<float>(a) * 0.0625f));
      Store16(&result, offset + 2u,
              FloatToHalf(static_cast<float>(b) * 0.0625f));
    }
  }
  return result;
}

inline std::vector<std::uint8_t> ScalarInput(
    const Resolution& resolution, bool generated) {
  const std::size_t pixels =
      std::size_t{resolution.input_width} * resolution.input_height;
  std::vector<std::uint8_t> result(pixels * 4u);
  for (std::uint32_t y = 0; y < resolution.input_height; ++y) {
    for (std::uint32_t x = 0; x < resolution.input_width; ++x) {
      float value = 0.5f;
      if (x >= 64u || y >= 64u) {
        const auto mixed = Mix((generated ? 0xa4093822299f31d0ull
                                         : 0x082efa98ec4e6c89ull) +
                               std::uint64_t{y} * resolution.input_width + x);
        value = static_cast<float>(32u + (mixed & 255u)) * (1.0f / 512.0f);
      }
      Store32(&result, (std::size_t{y} * resolution.input_width + x) * 4u,
              value);
    }
  }
  return result;
}

inline std::size_t VectorInputBytes(const Resolution& resolution) {
  return std::size_t{resolution.input_width} * resolution.input_height * 4u;
}

inline std::size_t ScalarInputBytes(const Resolution& resolution) {
  return std::size_t{resolution.input_width} * resolution.input_height * 4u;
}

inline std::size_t VectorOutputBytes(const Resolution& resolution) {
  return std::size_t{resolution.output_width} * resolution.output_height * 8u;
}

inline std::size_t DepthOutputBytes(const Resolution& resolution) {
  return std::size_t{resolution.output_width} * resolution.output_height * 4u;
}

}  // namespace fg_dynamic_module15
