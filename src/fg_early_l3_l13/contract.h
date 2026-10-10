#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace fg_early_l3_l13 {

inline constexpr std::uint32_t kWidth = 640;
inline constexpr std::uint32_t kHeight = 360;
inline constexpr std::uint32_t kHalfWidth = 320;
inline constexpr std::uint32_t kHalfHeight = 180;
inline constexpr std::size_t kSourceBytes =
    std::size_t{kWidth} * kHeight * 4u;
inline constexpr std::size_t kHistoryBytes =
    std::size_t{kWidth} * kHeight * 8u;
inline constexpr std::size_t kHalfColorBytes =
    std::size_t{kHalfWidth} * kHalfHeight * 8u;
inline constexpr std::size_t kHalfMetricBytes =
    std::size_t{kHalfWidth} * kHalfHeight * 2u;
inline constexpr std::size_t kCombinedOutputBytes =
    kHalfColorBytes + kHalfMetricBytes + kHistoryBytes;
inline constexpr std::size_t kPackedParameterBytes = 168;
inline constexpr std::size_t kModuleBytes = 53920;
inline constexpr std::size_t kCanaryBytes = 4096;
inline constexpr std::uint8_t kCanaryValue = 0xc7;

inline constexpr char kModuleSha256[] =
    "06eee29ec33202996f97c063a08ccdc767d0167252ad03b5ed457c941cb85206";
inline constexpr char kL3RawParameterSha256[] =
    "f9d25459eeb37610a50065102b61cd7f458fb1598ad754f3da241293aa15e435";
inline constexpr char kL13RawParameterSha256[] =
    "9eb930556ec08fd8db9ec361c917b8462707a6dc091b24227512a6a503115ffa";
inline constexpr char kL3AddressFreeParameterSha256[] =
    "631199cad3cff1b7e60a775d029de73644fcd82c085620aec984d6cdf6e292e5";
inline constexpr char kL13AddressFreeParameterSha256[] =
    "dfcd9a6cda167f7ef89c43dcb56997e35f58c6ec348730396e940d29a6560625";
inline constexpr char kGenerator[] =
    "splitmix64-fg-early-l3-l13-color-history-v1";

inline std::uint64_t Mix(std::uint64_t value) {
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

inline std::vector<std::uint8_t> GenerateSource() {
  constexpr std::array<std::uint8_t, 16> anchors = {
      0, 1, 2, 3, 15, 31, 63, 64, 127, 128, 191, 223, 239, 253, 254, 255};
  std::vector<std::uint8_t> result(kSourceBytes);
  for (std::size_t pixel = 0; pixel < kSourceBytes / 4u; ++pixel) {
    const auto mixed = Mix(0x243f6a8885a308d3ull ^
                           pixel * 0x9e3779b97f4a7c15ull);
    for (std::size_t channel = 0; channel < 4u; ++channel) {
      result[pixel * 4u + channel] = pixel < anchors.size()
          ? anchors[(pixel + channel * 3u) % anchors.size()]
          : static_cast<std::uint8_t>(mixed >> (channel * 13u));
    }
  }
  return result;
}

inline std::vector<std::uint8_t> GenerateHistory(const std::string& fixture) {
  std::vector<std::uint8_t> result(kHistoryBytes);
  if (fixture == "reset-history") return result;
  if (fixture != "generated-history")
    throw std::runtime_error("unsupported L3/L13 fixture");
  constexpr std::array<std::uint16_t, 16> values = {
      0x0000,0x1400,0x1800,0x1c00,0x2000,0x2400,0x2800,0x2c00,
      0x3000,0x3400,0x3800,0x3a00,0x3b00,0x3b80,0x3c00,0x3c00};
  for (std::size_t pixel = 0; pixel < kHistoryBytes / 8u; ++pixel) {
    const auto mixed = Mix(0x13198a2e03707344ull ^
                           pixel * 0xd1b54a32d192ed03ull);
    for (std::size_t channel = 0; channel < 4u; ++channel)
      Store16(&result, pixel * 8u + channel * 2u,
              values[(mixed >> (channel * 9u)) % values.size()]);
  }
  return result;
}

inline std::vector<std::uint8_t> Poison(std::size_t bytes,
                                        std::uint8_t poison) {
  return std::vector<std::uint8_t>(bytes, poison);
}

inline const char* HistoryReadBank(const std::string& phase) {
  if (phase == "l3-evaluate-0") return "history-bank-a";
  if (phase == "l13-evaluate-1") return "history-bank-b";
  throw std::runtime_error("unsupported L3/L13 phase");
}

inline const char* HistoryWriteBank(const std::string& phase) {
  if (phase == "l3-evaluate-0") return "history-bank-b";
  if (phase == "l13-evaluate-1") return "history-bank-a";
  throw std::runtime_error("unsupported L3/L13 phase");
}

inline std::uint32_t SelectedLaunch(const std::string& phase) {
  if (phase == "l3-evaluate-0") return 3;
  if (phase == "l13-evaluate-1") return 13;
  throw std::runtime_error("unsupported L3/L13 phase");
}

}  // namespace fg_early_l3_l13
