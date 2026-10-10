#pragma once

#include "../fg_early_l19/contract.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace fg_dynamic_module5 {

inline constexpr char kModuleSha256[] =
    "3f2308a693730b44c00e064ed25dab3b73a420f2e92d215ef733c4e10a30439c";
inline constexpr std::size_t kModuleBytes = 14448;
inline constexpr char kInputGenerator[] =
    "fg-dynamic-module5-independent-ten-inputs-v1";

struct Resolution {
  const char* name;
  std::uint32_t motion_width;
  std::uint32_t motion_height;
  std::uint32_t work_width;
  std::uint32_t work_height;
  std::uint32_t tensor_width;
  std::uint32_t tensor_height;
  std::uint32_t grid_x;
  std::uint32_t grid_y;
  std::uint32_t origin_x;
  std::uint32_t reserved_origin_x;
  std::uint32_t origin_y;
  std::uint32_t reserved_origin_y;
  const char* raw_parameter_sha256;
  const char* address_free_parameter_sha256;
  std::uint64_t fixture_salt;
};

inline constexpr std::array<Resolution, 4> kResolutions{{
    {"720p", 640, 360, 640, 360, 320, 192, 40, 24, 0, 0, 6, 6,
     "31452fa6c5411ee2372faa3fd7846d81add56279812d45300b28f4871d752938",
     "b0c7faa583b12174bf72f7c03665fff788bc8b7298b754913d60beac4ce93b1a",
     0x7207207207207207ull},
    {"1080p", 960, 540, 960, 540, 480, 288, 60, 36, 0, 0, 9, 9,
     "a0bb155954511a39e1f567ef798ac9265ec1b91e5cc8dc12fe25a32e91d33f59",
     "b2910867274200a5ad634e4ec86b8804491e8f02888f18d04d87a3c0088d2419",
     0x1080108010801080ull},
    {"1440p", 1280, 720, 1280, 720, 640, 384, 80, 48, 0, 0, 12, 12,
     "f32b37ca6be9d175995b367f9475cf2c349776118985f49207ec6fe3957fcddb",
     "6bd52da18317b2cf2b88ffe3e7132779a367c2e629d8a9fbc613ed2051e877e8",
     0x1440144014401440ull},
    // The 4K trace has a 1920x1080 motion-vector normalization extent while
    // the selected module operates on a 960x540 work surface.
    {"4k", 1920, 1080, 960, 540, 480, 288, 60, 36, 0, 0, 9, 9,
     "32571c8e622d6628b61a1552653e927b239dc0ff11973d5b02d8cfe32479de75",
     "e0f68a98f41f292f2f645d469cc0ef889463280a8975de34bd437ca94b93b5a4",
     0x4b4b4b4b4b4b4b4bull},
}};

inline const Resolution& FindResolution(std::string_view name) {
  for (const auto& resolution : kResolutions) {
    if (name == resolution.name) return resolution;
  }
  throw std::runtime_error("unsupported dynamic module5 resolution");
}

enum class Fixture { kFirstFrame, kFanIn };

inline Fixture ParseFixture(std::string_view name) {
  if (name == "first-frame") return Fixture::kFirstFrame;
  if (name == "fan-in") return Fixture::kFanIn;
  throw std::runtime_error("unsupported dynamic module5 fixture");
}

inline const char* FixtureName(Fixture fixture) {
  return fixture == Fixture::kFirstFrame ? "first-frame" : "fan-in";
}

inline std::uint64_t Salt(const Resolution& resolution, std::uint64_t value) {
  return fg_early_l19::SplitMix64(resolution.fixture_salt ^ value);
}

inline std::vector<std::uint8_t> Half4Input(
    const Resolution& resolution, std::uint64_t salt) {
  const std::size_t pixels =
      std::size_t{resolution.work_width} * resolution.work_height;
  std::vector<std::uint8_t> result(pixels * 8u);
  for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
    for (std::uint32_t channel = 0; channel < 4u; ++channel) {
      const auto mixed = fg_early_l19::SplitMix64(
          Salt(resolution, salt) ^ pixel * 0x9e3779b97f4a7c15ull ^
          std::uint64_t{channel + 1u} * 0xbf58476d1ce4e5b9ull);
      const float value = static_cast<float>(1u + (mixed % 15u)) / 16.0f;
      fg_early_l19::Store16(
          &result, pixel * 8u + channel * 2u,
          fg_early_l19::FloatToHalf(value));
    }
  }
  return result;
}

inline std::vector<std::uint8_t> BinaryMask(
    const Resolution& resolution, std::uint64_t salt) {
  const std::size_t pixels =
      std::size_t{resolution.work_width} * resolution.work_height;
  std::vector<std::uint8_t> result(pixels);
  for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
    result[pixel] = (fg_early_l19::SplitMix64(
        Salt(resolution, salt) ^ pixel) & 3u) == 0u ? 0u : 255u;
  }
  return result;
}

inline std::vector<std::uint8_t> Half2Input(
    const Resolution& resolution, std::uint64_t salt, bool zero) {
  const std::size_t pixels =
      std::size_t{resolution.work_width} * resolution.work_height;
  std::vector<std::uint8_t> result(pixels * 4u);
  for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
    for (std::uint32_t channel = 0; channel < 2u; ++channel) {
      float value = 0.0f;
      if (!zero) {
        const auto mixed = fg_early_l19::SplitMix64(
            Salt(resolution, salt) ^ pixel * 0x94d049bb133111ebull ^
            std::uint64_t{channel} * 0x9e3779b97f4a7c15ull);
        value = static_cast<float>(static_cast<int>(mixed % 9u) - 4) / 8.0f;
      }
      fg_early_l19::Store16(
          &result, pixel * 4u + channel * 2u,
          fg_early_l19::FloatToHalf(value));
    }
  }
  return result;
}

inline std::vector<std::uint8_t> Half1Input(
    const Resolution& resolution, std::uint64_t salt, bool uniform_half) {
  const std::size_t pixels =
      std::size_t{resolution.work_width} * resolution.work_height;
  std::vector<std::uint8_t> result(pixels * 2u);
  for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
    float value = 0.5f;
    if (!uniform_half) {
      value = static_cast<float>(1u + (fg_early_l19::SplitMix64(
          Salt(resolution, salt) ^ pixel) % 15u)) / 16.0f;
    }
    fg_early_l19::Store16(
        &result, pixel * 2u, fg_early_l19::FloatToHalf(value));
  }
  return result;
}

inline std::vector<std::uint8_t> Float2Input(
    const Resolution& resolution, bool zero) {
  const std::size_t pixels =
      std::size_t{resolution.work_width} * resolution.work_height;
  std::vector<std::uint8_t> result(pixels * 8u);
  for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
    float x = 0.0f;
    float y = 0.0f;
    if (!zero) {
      x = static_cast<float>(static_cast<int>(pixel % 3u) - 1) /
          static_cast<float>(resolution.work_width);
      y = static_cast<float>(static_cast<int>((pixel / 3u) % 3u) - 1) /
          static_cast<float>(resolution.work_height);
    }
    fg_early_l19::Store32(&result, pixel * 8u, x);
    fg_early_l19::Store32(&result, pixel * 8u + 4u, y);
  }
  return result;
}

struct Inputs {
  std::array<std::vector<std::uint8_t>, 10> textures;
};

inline Inputs GenerateInputs(const Resolution& resolution, Fixture fixture) {
  const bool first_frame = fixture == Fixture::kFirstFrame;
  Inputs result;
  result.textures[0] = Half4Input(resolution, 0x243f6a8885a308d3ull);
  result.textures[1] = Half4Input(resolution, 0x13198a2e03707344ull);
  result.textures[2] = BinaryMask(resolution, 0xa4093822299f31d0ull);
  result.textures[3] = BinaryMask(resolution, 0x082efa98ec4e6c89ull);
  result.textures[4] = Half2Input(resolution, 0x452821e638d01377ull, false);
  result.textures[5] = Half1Input(resolution, 0xbe5466cf34e90c6cull, false);
  result.textures[6] = Half1Input(resolution, 0xc0ac29b7c97c50ddull, first_frame);
  result.textures[7] = Float2Input(resolution, first_frame);
  result.textures[8] = Half2Input(resolution, 0x3f84d5b5b5470917ull, first_frame);
  result.textures[9] = BinaryMask(resolution, 0x9216d5d98979fb1bull);
  return result;
}

inline std::size_t InputBytes(const Resolution& resolution,
                              std::size_t index) {
  constexpr std::array<std::size_t, 10> pixel_bytes{{8, 8, 1, 1, 4,
                                                      2, 2, 8, 4, 1}};
  if (index >= pixel_bytes.size()) throw std::runtime_error("input index OOB");
  return std::size_t{resolution.work_width} * resolution.work_height *
         pixel_bytes[index];
}

inline std::size_t TensorBytes(const Resolution& resolution) {
  return std::size_t{resolution.tensor_width} * resolution.tensor_height *
         48u * 2u;
}

inline std::size_t MaskBytes(const Resolution& resolution) {
  return std::size_t{resolution.work_width} * resolution.work_height;
}

inline std::size_t VectorBytes(const Resolution& resolution) {
  return MaskBytes(resolution) * 4u;
}

}  // namespace fg_dynamic_module5
