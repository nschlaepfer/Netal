#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace fg_conv_3x3 {

struct Layer {
  const char* id;
  std::uint32_t launch;
  std::uint32_t module_token;
  std::size_t module_bytes;
  const char* module_sha256;
  std::uint32_t weight_token;
  std::size_t weight_bytes;
  const char* weight_sha256;
  std::uint32_t width;
  std::uint32_t height;
  std::uint32_t input_channels;
  std::uint32_t output_channels;
  std::array<std::uint32_t, 3> grid;
  std::array<std::uint32_t, 3> block;
  bool column_stage_schedule;
};

inline constexpr std::array<Layer, 4> kLayers = {{
    {"l20", 20, 28, 16512,
     "7f9c1f110e8ec9347aa968bc9019e5ffc8839057987a2e5676a0f64637c8df2d",
     63, 56320,
     "44595b02307a6b934a124abe476cfd327440e2acba7f5ed69f73195ca743e3d9",
     160, 96, 48, 32, {10, 8, 1}, {32, 4, 1}, false},
    {"l24", 24, 31, 11168,
     "e925f51165408fae4657d50247b10304802d46979b863712aeff844a217ade63",
     67, 299008,
     "4191924ff335d5018987f2526f933cb4cdbb9145411e3ce34f63c8206321be55",
     40, 24, 64, 128, {3, 24, 1}, {32, 1, 4}, false},
    {"l26", 26, 32, 11416,
     "c74961f6f71f4e7bac5b216cc704b10a0fdd6f0987352fdd315f371467ca7187",
     69, 593920,
     "c0c5648f9cb839e41d0e786c8034aa0bcc931b6ac84c58cfbde0502dc8a53404",
     20, 12, 128, 128, {2, 6, 1}, {32, 2, 4}, false},
    {"l28", 28, 33, 14752,
     "6f2ffbe7faacde8119719d92dacde40f21ddd562878381b81654dad7a66775dd",
     71, 1187840,
     "fe2c3057f2e894219601d3d7200eabfa97d990e08ccb23b43d52dfcba2d2fb93",
     10, 6, 128, 256, {1, 6, 2}, {32, 1, 8}, true},
}};

inline const Layer& FindLayer(std::string_view id) {
  for (const Layer& layer : kLayers) {
    if (id == layer.id) return layer;
  }
  throw std::runtime_error("unknown 3x3 convolution layer");
}

inline std::uint64_t SplitMix64(std::uint64_t value) {
  value += 0x9E3779B97F4A7C15ull;
  value = (value ^ (value >> 30u)) * 0xBF58476D1CE4E5B9ull;
  value = (value ^ (value >> 27u)) * 0x94D049BB133111EBull;
  return value ^ (value >> 31u);
}

inline std::vector<std::uint8_t> GenerateInput(const Layer& layer) {
  constexpr std::array<std::uint16_t, 11> values = {
      0x0000, 0x2800, 0xa800, 0x2c00, 0xac00, 0x3000,
      0xb000, 0x3400, 0xb400, 0x3800, 0xb800,
  };
  const std::size_t elements = std::size_t{layer.width} * layer.height *
                               layer.input_channels;
  std::vector<std::uint8_t> result(elements * 2);
  const std::uint64_t seed =
      0xD6E8FEB86659FD93ull ^
      (std::uint64_t{layer.launch} * 0xA5A3564E27F8862Bull);
  constexpr std::uint64_t step = 0x9E3779B97F4A7C15ull;
  for (std::size_t index = 0; index < elements; ++index) {
    const std::uint16_t bits =
        values[SplitMix64(seed + std::uint64_t{index} * step) % values.size()];
    result[index * 2] = static_cast<std::uint8_t>(bits);
    result[index * 2 + 1] = static_cast<std::uint8_t>(bits >> 8u);
  }
  return result;
}

inline std::size_t OutputBytes(const Layer& layer) {
  return std::size_t{layer.width} * layer.height * layer.output_channels * 2;
}

}  // namespace fg_conv_3x3
