#pragma once
#include "../fg_early_l19/contract.h"
#include <bit>
#include <cstddef>
#include <cstdint>
#include <vector>
namespace fg_distortion_module9 {
inline constexpr std::uint32_t kWidth = 320, kHeight = 180;
inline constexpr std::size_t kPixels = std::size_t{kWidth} * kHeight,
                             kRgba16Bytes = kPixels * 8u,
                             kRg32Bytes = kPixels * 8u,
                             kRg16Bytes = kPixels * 4u,
                             kR32Bytes = kPixels * 4u;
inline constexpr char kModuleSha256[] =
    "6c507cc85455aeaed386e293c77aa65c861e0093eebd4e2ca220fe57636319ed";
inline constexpr std::size_t kModuleBytes = 6064;
inline constexpr char kRawParameterSha256[] =
    "b695e0e6d29050462620ad1f54b593dd57b12a616cb58e2367ffef10a2444fe2";
inline constexpr char kGeneratedRawParameterSha256[] =
    "723e2c4a348298753ace7172a6e1452e22faf4518c10952ab7857f7eaf8406c3";
inline constexpr char kFrozenAddressFreeSha256[] =
    "f68940013a394407574fc47f1a4399c85e0a4b00203d34c78079577aa2b50187";
inline constexpr char kFixtureAddressFreeSha256[] =
    "b8066c598cb5dd1b198b9a5284580c0e3ee7e8df0d84e39d4f2c1a4a277edb3c";
inline constexpr char kInputGenerator[] =
    "fg-distortion-module9-independent-nonzero-v1";
inline void S16(std::vector<std::uint8_t> &b, std::size_t o, std::uint16_t v) {
  b[o] = static_cast<std::uint8_t>(v);
  b[o + 1] = static_cast<std::uint8_t>(v >> 8);
}
inline void S32(std::vector<std::uint8_t> &b, std::size_t o, std::uint32_t v) {
  for (int i = 0; i < 4; i++)
    b[o + i] = static_cast<std::uint8_t>(v >> (i * 8));
}
struct Inputs {
  std::vector<std::uint8_t> a, b, flow32, flow16, mask;
};
inline Inputs Generate() {
  Inputs r{std::vector<std::uint8_t>(kRgba16Bytes),
           std::vector<std::uint8_t>(kRgba16Bytes),
           std::vector<std::uint8_t>(kRg32Bytes),
           std::vector<std::uint8_t>(kRg16Bytes),
           std::vector<std::uint8_t>(kR32Bytes)};
  for (std::size_t p = 0; p < kPixels; p++) {
    int x = static_cast<int>(p % kWidth), y = static_cast<int>(p / kWidth);
    float ax = float(x % 3 - 1) / 320.0f, ay = float(y % 3 - 1) / 180.0f,
          bx = float((x + 1) % 3 - 1) / 640.0f,
          by = float((y + 1) % 3 - 1) / 360.0f;
    S16(r.a, p * 8, fg_early_l19::FloatToHalf(ax));
    S16(r.a, p * 8 + 2, fg_early_l19::FloatToHalf(ay));
    S16(r.b, p * 8, fg_early_l19::FloatToHalf(bx));
    S16(r.b, p * 8 + 2, fg_early_l19::FloatToHalf(by));
    S32(r.flow32, p * 8, std::bit_cast<std::uint32_t>(float(x % 5 - 2) * .5f));
    S32(r.flow32, p * 8 + 4,
        std::bit_cast<std::uint32_t>(float(y % 5 - 2) * .5f));
    S16(r.flow16, p * 4, fg_early_l19::FloatToHalf(float(x % 3 - 1) / 512.0f));
    S16(r.flow16, p * 4 + 2,
        fg_early_l19::FloatToHalf(float(y % 3 - 1) / 512.0f));
    S32(r.mask, p * 4, std::uint32_t((x * 17 + y * 31) ^ 0x5a5a5a5a));
  }
  return r;
}
} // namespace fg_distortion_module9
