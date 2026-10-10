#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace fg_dynamic_module11 {

inline constexpr char kModuleSha256[] =
    "06eee29ec33202996f97c063a08ccdc767d0167252ad03b5ed457c941cb85206";
inline constexpr std::size_t kModuleBytes = 53920;
inline constexpr std::size_t kPackedParameterBytes = 168;
inline constexpr std::size_t kCanaryBytes = 4096;
inline constexpr std::uint8_t kCanaryValue = 0xc7;
inline constexpr char kGenerator[] = "fg-dynamic-module11-color-history-v1";

struct Resolution {
  const char* name;
  std::uint32_t source_width;
  std::uint32_t source_height;
  std::uint32_t output_width;
  std::uint32_t output_height;
  std::uint32_t mode;
  std::uint32_t grid_x;
  std::uint32_t grid_y;
  const char* reset_raw_parameter_sha256;
  const char* generated_raw_parameter_sha256;
  const char* address_free_parameter_sha256;
};

inline constexpr std::array<Resolution, 4> kResolutions = {{
    {"720p", 1280, 720, 640, 360, 2, 80, 45,
     "58995d9b2bc69e3743bcf791df448280b0cbc16cfb4fb9ebe5b94f3a285718ce",
     "56237412b763906f11db4264e2a60216744649099c5ad14dced982755e46beda",
     "dcf8bb3f221684c0722a26804bb2166a11435bb7c4541b43f112f7c2065aa78d"},
    {"1080p", 1920, 1080, 960, 540, 2, 120, 68,
     "825e9cfbe5e59b6a17d2038e5a20ef936988a8f9a1fca7f3bc8e01bc58bf3588",
     "a877d9529cdb9c0466c25128b89e2b3356393a69bcd2efb7ee9101a88ad03827",
     "ec699d800d69e85a722681d8ccd4d6efedc11ddc98ad06ca9e048cd168d3404a"},
    {"1440p", 2560, 1440, 1280, 720, 2, 160, 90,
     "686e6a7d29621ad92eabed03596b9d0b5216ee95f885607ad6b25e0253b0596e",
     "c466ec3869a77863539df2f5c34d8c3396efdb878bc1cd7e6f4da80b3ae07af3",
     "366e7de5f899c6479a59844e5a6d289341b209ff94aeccfd5db78b80886ff249"},
    {"4k", 3840, 2160, 960, 540, 4, 240, 135,
     "0f67762faac0fcc7aa4c8a3c320dad41e7121ef45d629bb851718d2abe0b2fe3",
     "566edf20adcd58991b22cbca47447770317e4c4046256ec336a34a876d359dcd",
     "973d1bf49c30b91b2044eb1ffbb8d8abf9f9b5a354f66669d9eeb1ffa89603f1"},
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

inline void Store16(std::vector<std::uint8_t>* bytes, std::size_t offset,
                    std::uint16_t value) {
  std::memcpy(bytes->data() + offset, &value, sizeof(value));
}

inline std::vector<std::uint8_t> Source(const Resolution& resolution,
                                        bool generated) {
  constexpr std::array<std::uint8_t, 16> anchors = {
      0, 1, 2, 3, 15, 31, 63, 64, 127, 128, 191, 223, 239, 253, 254, 255};
  const std::size_t pixels =
      std::size_t{resolution.source_width} * resolution.source_height;
  std::vector<std::uint8_t> result(pixels * 4u);
  const std::uint64_t salt = generated ? 0x13198a2e03707344ull
                                       : 0x243f6a8885a308d3ull;
  for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
    const auto mixed = Mix(salt ^ pixel * 0xd1b54a32d192ed03ull);
    for (std::size_t channel = 0; channel < 4u; ++channel)
      result[pixel * 4u + channel] = pixel < anchors.size()
          ? anchors[(pixel + channel * 3u + (generated ? 5u : 0u)) %
                    anchors.size()]
          : static_cast<std::uint8_t>(mixed >> (channel * 13u));
  }
  return result;
}

inline std::vector<std::uint8_t> History(const Resolution& resolution,
                                         bool generated) {
  const std::size_t pixels =
      std::size_t{resolution.source_width} * resolution.source_height;
  std::vector<std::uint8_t> result(pixels * 8u);
  if (!generated) return result;
  constexpr std::array<std::uint16_t, 16> values = {
      0x0000, 0x1400, 0x1800, 0x1c00, 0x2000, 0x2400, 0x2800, 0x2c00,
      0x3000, 0x3400, 0x3800, 0x3a00, 0x3b00, 0x3b80, 0x3c00, 0xbc00};
  for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
    const auto mixed = Mix(0xa4093822299f31d0ull ^
                           pixel * 0x94d049bb133111ebull);
    for (std::size_t channel = 0; channel < 4u; ++channel)
      Store16(&result, pixel * 8u + channel * 2u,
              values[(mixed >> (channel * 11u)) % values.size()]);
  }
  return result;
}

inline std::size_t SourceBytes(const Resolution& resolution) {
  return std::size_t{resolution.source_width} * resolution.source_height * 4u;
}

inline std::size_t HistoryBytes(const Resolution& resolution) {
  return std::size_t{resolution.source_width} * resolution.source_height * 8u;
}

inline std::size_t HalfColorBytes(const Resolution& resolution) {
  return std::size_t{resolution.output_width} * resolution.output_height * 8u;
}

inline std::size_t HalfMetricBytes(const Resolution& resolution) {
  return std::size_t{resolution.output_width} * resolution.output_height * 2u;
}

inline std::vector<std::uint8_t> Poison(std::size_t bytes,
                                        std::uint8_t poison) {
  return std::vector<std::uint8_t>(bytes, poison);
}

inline const char* HistoryReadBank(bool generated) {
  return generated ? "history-bank-b" : "history-bank-a";
}

inline const char* HistoryWriteBank(bool generated) {
  return generated ? "history-bank-a" : "history-bank-b";
}

}  // namespace fg_dynamic_module11
