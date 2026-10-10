#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace fg_early_l11 {

constexpr std::uint32_t kTextureWidth = 640;
constexpr std::uint32_t kTextureHeight = 360;
constexpr std::uint32_t kInputOriginX = 17;
constexpr std::uint32_t kInputOriginY = 11;
constexpr std::uint32_t kOutputOriginX = 29;
constexpr std::uint32_t kOutputOriginY = 19;
constexpr std::uint32_t kCopyWidth = 587;
constexpr std::uint32_t kCopyHeight = 317;
constexpr std::size_t kPixelBytes = 4;
constexpr std::size_t kTextureBytes =
    std::size_t{kTextureWidth} * kTextureHeight * kPixelBytes;
constexpr std::size_t kRegionBytes =
    std::size_t{kCopyWidth} * kCopyHeight * kPixelBytes;
constexpr std::size_t kOutsidePixels =
    std::size_t{kTextureWidth} * kTextureHeight -
    std::size_t{kCopyWidth} * kCopyHeight;
constexpr std::size_t kPackedParameterBytes = 40;
constexpr std::size_t kModuleBytes = 4512;
constexpr char kModuleSha256[] =
    "c11cd7474562e20762d272162c94378a17f8f77c7de8206d695e6c469383e840";
constexpr char kFrozenRawParameterSha256[] =
    "b18c2cc81aac4bc673b19c101dcb331dfa86693659bf1dbfe1a77d4a52f3e3a7";
constexpr char kFrozenAddressFreeParameterSha256[] =
    "7e578a6c893a77c02fed71dcf1cc94624cb5e995cbfa77a4ad94b6a50bd48d71";
constexpr char kFixtureAddressFreeParameterSha256[] =
    "9b4595a88249e4192e7a66db345af68a8a53ec2e550bf86935ad5e974a48a723";
constexpr char kInputGenerator[] = "splitmix64-fg-early-l11-rgba8-v1";

static_assert(kInputOriginX + kCopyWidth <= kTextureWidth);
static_assert(kInputOriginY + kCopyHeight <= kTextureHeight);
static_assert(kOutputOriginX + kCopyWidth <= kTextureWidth);
static_assert(kOutputOriginY + kCopyHeight <= kTextureHeight);
static_assert(kRegionBytes == 744316);
static_assert(kOutsidePixels == 44321);

inline std::uint64_t SplitMix64(std::uint64_t value) {
  value += 0x9e3779b97f4a7c15ull;
  value = (value ^ (value >> 30u)) * 0xbf58476d1ce4e5b9ull;
  value = (value ^ (value >> 27u)) * 0x94d049bb133111ebull;
  return value ^ (value >> 31u);
}

inline std::vector<std::uint8_t> GenerateInput() {
  std::vector<std::uint8_t> result(kTextureBytes);
  for (std::uint32_t y = 0; y < kTextureHeight; ++y) {
    for (std::uint32_t x = 0; x < kTextureWidth; ++x) {
      const std::size_t pixel = std::size_t{y} * kTextureWidth + x;
      const auto value = SplitMix64(0x243f6a8885a308d3ull ^
                                    std::uint64_t{x} * 0x9e3779b97f4a7c15ull ^
                                    std::uint64_t{y} * 0xbf58476d1ce4e5b9ull);
      result[pixel * 4u] = static_cast<std::uint8_t>(value);
      result[pixel * 4u + 1u] = static_cast<std::uint8_t>(value >> 13u);
      result[pixel * 4u + 2u] = static_cast<std::uint8_t>(value >> 29u);
      result[pixel * 4u + 3u] = static_cast<std::uint8_t>(value >> 47u);
    }
  }
  return result;
}

inline std::vector<std::uint8_t> ExpectedOutput(
    const std::vector<std::uint8_t>& input, std::uint8_t poison) {
  std::vector<std::uint8_t> result(kTextureBytes, poison);
  for (std::uint32_t y = 0; y < kCopyHeight; ++y) {
    for (std::uint32_t x = 0; x < kCopyWidth; ++x) {
      const std::size_t source =
          (std::size_t{y + kInputOriginY} * kTextureWidth +
           x + kInputOriginX) * 4u;
      const std::size_t destination =
          (std::size_t{y + kOutputOriginY} * kTextureWidth +
           x + kOutputOriginX) * 4u;
      for (std::size_t channel = 0; channel < 4u; ++channel) {
        result[destination + channel] = input[source + channel];
      }
    }
  }
  return result;
}

inline std::vector<std::uint8_t> ExtractRegion(
    const std::vector<std::uint8_t>& output) {
  std::vector<std::uint8_t> result(kRegionBytes);
  for (std::uint32_t y = 0; y < kCopyHeight; ++y) {
    const std::size_t source =
        (std::size_t{y + kOutputOriginY} * kTextureWidth + kOutputOriginX) * 4u;
    const std::size_t destination = std::size_t{y} * kCopyWidth * 4u;
    for (std::size_t byte = 0; byte < std::size_t{kCopyWidth} * 4u; ++byte) {
      result[destination + byte] = output[source + byte];
    }
  }
  return result;
}

inline bool InOutputRegion(std::uint32_t x, std::uint32_t y) {
  return x >= kOutputOriginX && x < kOutputOriginX + kCopyWidth &&
         y >= kOutputOriginY && y < kOutputOriginY + kCopyHeight;
}

}  // namespace fg_early_l11
