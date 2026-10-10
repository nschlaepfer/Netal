#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace fg_mid_l54 {

inline constexpr std::uint32_t kWidth = 320;
inline constexpr std::uint32_t kHeight = 180;
inline constexpr std::size_t kPixelCount = std::size_t{kWidth} * kHeight;
inline constexpr std::size_t kTextureBytes = kPixelCount * 2u * 2u;
inline constexpr std::size_t kOffsetBufferBytes = kPixelCount * 8u;
inline constexpr std::size_t kOutputBytes = kTextureBytes;
inline constexpr std::size_t kModuleBytes = 7544;
inline constexpr char kModuleSha256[] =
    "d8d41b87895a3119f24db4a8d4cb1ec4f73613b507e3d1d4de5dea0f36fe5dc5";

inline std::uint64_t Mix(std::uint64_t value) {
  value += 0x9E3779B97F4A7C15ull;
  value = (value ^ (value >> 30u)) * 0xBF58476D1CE4E5B9ull;
  value = (value ^ (value >> 27u)) * 0x94D049BB133111EBull;
  return value ^ (value >> 31u);
}

inline void Store16(std::vector<std::uint8_t>* bytes, std::size_t offset,
                    std::uint16_t value) {
  (*bytes)[offset] = static_cast<std::uint8_t>(value);
  (*bytes)[offset + 1u] = static_cast<std::uint8_t>(value >> 8u);
}

inline std::vector<std::uint8_t> Texture(std::uint64_t seed) {
  constexpr std::array<std::uint16_t, 17> values = {
      0x0000, 0x8000, 0x2000, 0xa000, 0x2400, 0xa400,
      0x2800, 0xa800, 0x2c00, 0xac00, 0x3000, 0xb000,
      0x3400, 0xb400, 0x3800, 0xb800, 0x3c00,
  };
  std::vector<std::uint8_t> result(kTextureBytes);
  for (std::size_t index = 0; index < kPixelCount * 2u; ++index) {
    const auto bits = values[Mix(seed + index * 0xD1B54A32D192ED03ull) %
                             values.size()];
    Store16(&result, index * 2u, bits);
  }
  return result;
}

inline std::vector<std::uint8_t> TextureA() {
  return Texture(0x243F6A8885A308D3ull);
}

inline std::vector<std::uint8_t> TextureB() {
  return Texture(0x13198A2E03707344ull);
}

inline std::vector<std::uint8_t> OffsetBuffer(std::uint64_t seed,
                                               std::uint8_t poison) {
  constexpr std::array<std::uint16_t, 9> offsets = {
      0x0000, 0x8000, 0x1400, 0x9400, 0x1800,
      0x9800, 0x1c00, 0x9c00, 0x2000,
  };
  std::vector<std::uint8_t> result(kOffsetBufferBytes, poison);
  for (std::size_t pixel = 0; pixel < kPixelCount; ++pixel) {
    const auto mixed = Mix(seed + pixel * 0x9E3779B97F4A7C15ull);
    Store16(&result, pixel * 8u, offsets[mixed % offsets.size()]);
    Store16(&result, pixel * 8u + 2u,
            offsets[(mixed >> 8u) % offsets.size()]);
  }
  return result;
}

inline std::vector<std::uint8_t> OffsetA(std::uint8_t poison) {
  return OffsetBuffer(0xA4093822299F31D0ull, poison);
}

inline std::vector<std::uint8_t> OffsetB(std::uint8_t poison) {
  return OffsetBuffer(0x082EFA98EC4E6C89ull, poison);
}

}  // namespace fg_mid_l54
