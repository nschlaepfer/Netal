#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace fg_resolution_l22_l24 {

inline constexpr char kModule22Sha256[] =
    "7d6602bbed88409e9ad2ef71293869dd22f7153f6387339ed65e91c12341c67a";
inline constexpr char kModule24Sha256[] =
    "5a41f9d65d1f6f933ae42f36b493f60687a646b73929291bafbe9d00297b6bdf";
inline constexpr std::size_t kModule22Bytes = 15120;
inline constexpr std::size_t kModule24Bytes = 25880;
inline constexpr char kGenerator[] =
    "fg-resolution-l22-l24-half3-mask-v1";

struct Resolution {
  const char* name;
  std::uint32_t external_width;
  std::uint32_t external_height;
  std::uint32_t working_width;
  std::uint32_t working_height;
  std::uint32_t selected_level;
  std::uint32_t extra_level_count;
};

inline constexpr std::array<Resolution, 4> kResolutions = {{
    {"720p", 1280, 720, 640, 360, 10, 1},
    {"1080p", 1920, 1080, 960, 540, 10, 1},
    {"1440p", 2560, 1440, 1280, 720, 11, 2},
    {"4k", 3840, 2160, 1920, 1080, 11, 2},
}};

inline const Resolution& ParseResolution(const std::string& name) {
  for (const auto& resolution : kResolutions) {
    if (name == resolution.name) return resolution;
  }
  throw std::runtime_error("unknown resolution: " + name);
}

inline std::uint32_t CeilDiv(std::uint32_t value, std::uint32_t divisor) {
  return (value + divisor - 1u) / divisor;
}

inline std::uint64_t Mix(std::uint64_t value) {
  value += 0x9e3779b97f4a7c15ull;
  value = (value ^ (value >> 30u)) * 0xbf58476d1ce4e5b9ull;
  value = (value ^ (value >> 27u)) * 0x94d049bb133111ebull;
  return value ^ (value >> 31u);
}

inline void Store16(std::vector<std::uint8_t>* bytes, std::size_t offset,
                    std::uint16_t value) {
  std::memcpy(bytes->data() + offset, &value, sizeof(value));
}

inline std::vector<std::uint8_t> Half3(std::uint32_t width,
                                       std::uint32_t height,
                                       std::uint64_t salt) {
  constexpr std::array<std::uint16_t, 11> values = {
      0x0000, 0x2800, 0xa800, 0x3000, 0xb000, 0x3400,
      0xb400, 0x3800, 0xb800, 0x3c00, 0xbc00};
  constexpr std::array<std::uint16_t, 6> weights = {
      0x0000, 0x3000, 0x3400, 0x3800, 0x3a00, 0x3c00};
  const std::size_t pixels = std::size_t{width} * height;
  std::vector<std::uint8_t> result(pixels * 6u);
  for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
    const auto mixed = Mix(salt + pixel * 0xd1b54a32d192ed03ull);
    Store16(&result, pixel * 6u, values[mixed % values.size()]);
    Store16(&result, pixel * 6u + 2u,
            values[(mixed >> 10u) % values.size()]);
    Store16(&result, pixel * 6u + 4u,
            weights[(mixed >> 20u) % weights.size()]);
  }
  return result;
}

inline std::vector<std::uint8_t> Mask(const Resolution& resolution,
                                      bool generated) {
  const std::size_t pixels =
      std::size_t{resolution.working_width} * resolution.working_height;
  std::vector<std::uint8_t> result(pixels);
  const std::uint64_t phase = generated ? 0x13198a2e03707344ull
                                        : 0x243f6a8885a308d3ull;
  for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
    const auto mixed = Mix(phase + pixel * 0x94d049bb133111ebull);
    result[pixel] = (mixed & 7u) < 6u ? 1u : 0u;
  }
  return result;
}

inline std::vector<std::uint8_t> Module22Input(const Resolution& resolution,
                                               bool generated) {
  const std::uint32_t width = CeilDiv(resolution.working_width, 16);
  const std::uint32_t height = CeilDiv(resolution.working_height, 16);
  return Half3(width, height,
               generated ? 0xa4093822299f31d0ull : 0x082efa98ec4e6c89ull);
}

inline std::array<std::vector<std::uint8_t>, 4> Module22OutputInitial(
    const Resolution& resolution, std::uint8_t poison) {
  std::array<std::vector<std::uint8_t>, 4> result;
  std::uint32_t width = CeilDiv(resolution.working_width, 16);
  std::uint32_t height = CeilDiv(resolution.working_height, 16);
  for (std::size_t index = 0; index < result.size(); ++index) {
    width = CeilDiv(width, 2);
    height = CeilDiv(height, 2);
    result[index].assign(std::size_t{width} * height * 6u, poison);
  }
  return result;
}

inline std::array<std::vector<std::uint8_t>, 4> Module24Levels(
    const Resolution& resolution, bool generated) {
  std::array<std::vector<std::uint8_t>, 4> result;
  std::uint32_t width = resolution.working_width;
  std::uint32_t height = resolution.working_height;
  for (std::size_t index = 0; index < result.size(); ++index) {
    width = CeilDiv(width, 2);
    height = CeilDiv(height, 2);
    result[index] = Half3(
        width, height,
        (generated ? 0xbb67ae8584caa73bull : 0x3c6ef372fe94f82bull) +
            index * 0x100000001b3ull);
  }
  return result;
}

}  // namespace fg_resolution_l22_l24
