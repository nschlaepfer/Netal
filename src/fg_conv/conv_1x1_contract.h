#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace fg_conv_1x1 {

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
  bool leaky_relu;
};

inline constexpr std::array<Layer, 13> kLayers = {{
    {"l22", 22, 30, 4952,
     "31bb08d04011db9a0def8f1a1bc3999efc08d5deb8d7cb9b5debf52ec489b785",
     65, 10240,
     "eb99328524fddc9f5f3fac12b8e3a4011e66a75e071cd07126a70c553458bc60",
     80, 48, 32, 64, {5, 24, 1}, {32, 2, 2}, true},
    {"l30", 30, 34, 11328,
     "2f490857c6f7aa22f2435057706ec10b28b6b8d3081bd967ebe1d61550ac6419",
     73, 540672,
     "e23316a86707c1d3d377aa0711df47d05976ad1a30f6428524d030dd17145de9",
     5, 3, 256, 512, {1, 1, 32}, {32, 4, 1}, true},
    {"l31", 31, 35, 19424,
     "331853771cae4aa041038d8e9c74dcb1f8d0d38b4d5603e2e8d079ad1e44fd04",
     75, 532480,
     "9f20d6589cc9269ad020233e68196d66eba0e8a76497374cf6d445385f94dbc4",
     5, 3, 512, 256, {1, 3, 16}, {32, 1, 1}, true},
    {"l34", 34, 38, 11320,
     "8e7abed27b7eae0dd274b2409a13d6dde98277d9e743f0e36a1042022a16ceda",
     77, 270336,
     "9e5da09b93650c74e89d5262aa8a9a6a3dcbd0d753dc8c9d9aed39731e39b2a0",
     10, 6, 256, 256, {1, 2, 16}, {32, 4, 1}, true},
    {"l35", 35, 39, 11328,
     "939d5d9cf7012fb3ed3ce4e4792ccf5c6befec165ae7f7a0d3704cb1103c3030",
     79, 135168,
     "3969384493583e98dcf357d3046a10a5b836585261f3e584605bf11e23fb8d46",
     10, 6, 256, 128, {1, 2, 8}, {32, 4, 1}, true},
    {"l38", 38, 42, 7368,
     "93f3bb1be638eaf8003ba04cc7977859dfb16ae7821801c3017e45546301ecbc",
     81, 69632,
     "a0c6690faee2838521c6ad0f9445425d639446c0ef1b7fed9b43d21e7a8ec650",
     20, 12, 128, 128, {2, 3, 8}, {32, 4, 1}, true},
    {"l39", 39, 42, 7368,
     "93f3bb1be638eaf8003ba04cc7977859dfb16ae7821801c3017e45546301ecbc",
     83, 69632,
     "3a13970b482f1f78ede1bfe531956c203f08f99077c81bda2d1e4d0f453cf9a0",
     20, 12, 128, 128, {2, 3, 8}, {32, 4, 1}, true},
    {"l42", 42, 43, 8440,
     "53fdf8d5404c93f2fe64582f271192df3b24c2ef08d9564ffeb9d7990485db0e",
     85, 69632,
     "47361a699fa6514332dc560cf2416872256e61d1b36b1ca29df323a53ee2be77",
     40, 24, 128, 128, {3, 24, 1}, {32, 1, 4}, true},
    {"l43", 43, 44, 8496,
     "7113dfaae4945541276b3e662cba366f9a3fd9666db7a338efa867cfd7f46e31",
     87, 34816,
     "327bd9b02da4aaac8b251320f60b4aeccb0df7df84bba44d7f1f6a9fdf62c934",
     40, 24, 128, 64, {3, 24, 2}, {32, 1, 1}, true},
    {"l46", 46, 47, 6088,
     "8a3efc1e9166ed3ee4a54aea535910c0562b6753b2d0daae0402e1ed36ff19dc",
     89, 18432,
     "e24f67110bab401ab7f70a861f3f09d2cd3cd6abf8d1aa158ba3300efb64797e",
     80, 48, 64, 64, {5, 24, 1}, {32, 2, 2}, true},
    {"l47", 47, 48, 5992,
     "11906664ac548c61262a12b0950bf135926cf269de2d7691d0edfbe4a95e63dd",
     91, 9216,
     "325d7dfac5620a89edc1d2315731414cf755c50506ea5758a8c3cf563649577c",
     80, 48, 64, 32, {5, 12, 1}, {32, 4, 1}, true},
    {"l50", 50, 51, 4840,
     "304d0c0e58958fa7eab6a82fd046ff689c51a986aff88a069f6f509c434a5b96",
     93, 5120,
     "bb98eb15be7d255a15a46994589705560e0691e273e286a36a68fd679a1890a9",
     160, 96, 32, 32, {10, 24, 1}, {32, 4, 1}, true},
    {"l51", 51, 52, 3912,
     "27486bcc1db8b04b0e18d72f33b06662da9c327eb15b41a09b3f909a3a52cad2",
     95, 2560,
     "477c68ebace499e2e1ac331f214fae3e9fb4df815a9e7d8142a0f1fb414b340e",
     160, 96, 32, 16, {10, 24, 1}, {32, 4, 1}, false},
}};

inline const Layer& FindLayer(std::string_view id) {
  for (const Layer& layer : kLayers) {
    if (id == layer.id) return layer;
  }
  throw std::runtime_error("unknown 1x1 convolution layer");
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
      0xA0761D6478BD642Full ^
      (std::uint64_t{layer.launch} * 0xE7037ED1A0B428DBull);
  constexpr std::uint64_t step = 0x8EBC6AF09C88C6E3ull;
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

}  // namespace fg_conv_1x1
