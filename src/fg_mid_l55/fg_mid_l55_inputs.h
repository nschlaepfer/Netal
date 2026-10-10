#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "fg_mid_l55_contract.h"

namespace fg_mid_l55 {

inline std::uint64_t SplitMix64(std::uint64_t value) {
  value += 0x9e3779b97f4a7c15ull;
  value = (value ^ (value >> 30u)) * 0xbf58476d1ce4e5b9ull;
  value = (value ^ (value >> 27u)) * 0x94d049bb133111ebull;
  return value ^ (value >> 31u);
}

inline void Store16(std::vector<std::uint8_t>* bytes, std::size_t offset,
                    std::uint16_t value) {
  (*bytes)[offset] = static_cast<std::uint8_t>(value);
  (*bytes)[offset + 1u] = static_cast<std::uint8_t>(value >> 8u);
}

inline void Store64(std::vector<std::uint8_t>* bytes, std::size_t offset,
                    std::uint64_t value) {
  for (std::size_t index = 0; index < 8; ++index)
    (*bytes)[offset + index] =
        static_cast<std::uint8_t>(value >> (8u * index));
}

inline std::uint16_t SignedHalf(std::uint64_t identity, std::uint32_t channel,
                                std::uint32_t minimum_exponent,
                                std::uint32_t exponent_span) {
  const auto mixed = SplitMix64(
      0x6a09e667f3bcc909ull ^
      identity * 0x9e3779b97f4a7c15ull ^
      std::uint64_t{channel + 1u} * 0xbf58476d1ce4e5b9ull);
  const std::uint16_t sign = (mixed >> 63u) != 0u ? 0x8000u : 0u;
  const std::uint16_t exponent = static_cast<std::uint16_t>(
      minimum_exponent + ((mixed >> 17u) % exponent_span));
  return static_cast<std::uint16_t>(sign | exponent << 10u |
                                    (mixed & 0x03ffu));
}

inline std::vector<std::uint8_t> GenerateRgba(std::uint64_t salt) {
  std::vector<std::uint8_t> bytes(kRgbaTextureBytes);
  for (std::size_t pixel = 0; pixel < bytes.size() / 8u; ++pixel) {
    const auto identity = pixel ^ salt;
    for (std::uint32_t channel = 0; channel < 4; ++channel)
      Store16(&bytes, pixel * 8u + channel * 2u,
              SignedHalf(identity, channel, 11u, 4u));
  }
  return bytes;
}

inline std::vector<std::uint8_t> GenerateMotion() {
  std::vector<std::uint8_t> bytes(kRgTextureBytes);
  for (std::size_t pixel = 0; pixel < bytes.size() / 4u; ++pixel) {
    Store16(&bytes, pixel * 4u,
            SignedHalf(pixel ^ 0x243f6a8885a308d3ull, 0, 9u, 5u));
    Store16(&bytes, pixel * 4u + 2u,
            SignedHalf(pixel ^ 0x13198a2e03707344ull, 1, 9u, 5u));
  }
  return bytes;
}

inline std::vector<std::uint8_t> GenerateBlend() {
  constexpr std::array<std::uint16_t, 5> values = {
      0x0000u, 0x3400u, 0x3800u, 0x3a00u, 0x3c00u};
  std::vector<std::uint8_t> bytes(kRgTextureBytes);
  for (std::size_t pixel = 0; pixel < bytes.size() / 4u; ++pixel) {
    const auto mixed = SplitMix64(pixel ^ 0xa4093822299f31d0ull);
    Store16(&bytes, pixel * 4u, values[mixed % values.size()]);
    Store16(&bytes, pixel * 4u + 2u,
            values[(mixed >> 16u) % values.size()]);
  }
  return bytes;
}

inline std::vector<std::uint8_t> GenerateCandidates(bool first) {
  constexpr std::array<std::uint16_t, 7> offsets = {
      0x0000u, 0x1800u, 0x9800u, 0x1c00u,
      0x9c00u, 0x2000u, 0xa000u};
  std::vector<std::uint8_t> bytes(kCandidateBufferBytes);
  for (std::size_t pixel = 0; pixel < bytes.size() / 8u; ++pixel) {
    const bool invalid = first ? (pixel % 17u == 0u)
                               : (pixel % 19u == 0u);
    if (invalid) {
      Store64(&bytes, pixel * 8u, kInvalidCandidate);
      continue;
    }
    const auto mixed = SplitMix64(
        pixel ^ (first ? 0x082efa98ec4e6c89ull : 0x452821e638d01377ull));
    const std::uint16_t x = offsets[mixed % offsets.size()];
    const std::uint16_t y = offsets[(mixed >> 21u) % offsets.size()];
    Store64(&bytes, pixel * 8u,
            std::uint64_t{x} << 16u | std::uint64_t{y});
  }
  return bytes;
}

struct GeneratedInputs {
  std::vector<std::uint8_t> rgba_a =
      GenerateRgba(0x3bd39e10cb0ef593ull);
  std::vector<std::uint8_t> rgba_b =
      GenerateRgba(0xc0acf169b5f18a8cull);
  std::vector<std::uint8_t> motion = GenerateMotion();
  std::vector<std::uint8_t> blend = GenerateBlend();
  std::vector<std::uint8_t> candidate_a = GenerateCandidates(true);
  std::vector<std::uint8_t> candidate_b = GenerateCandidates(false);
};

}  // namespace fg_mid_l55
