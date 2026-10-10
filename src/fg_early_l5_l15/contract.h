#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace fg_early_l5_l15 {

inline constexpr std::uint32_t kWidth = 320;
inline constexpr std::uint32_t kHeight = 180;
inline constexpr std::size_t kPixels = std::size_t{kWidth} * kHeight;
inline constexpr std::size_t kFlowBytes = kPixels * 2u * 4u;
inline constexpr std::size_t kDepthBytes = kPixels * 4u;
inline constexpr std::size_t kAtomicBytes = kPixels * 8u;
inline constexpr std::size_t kMaskBytes = kPixels;
inline constexpr std::size_t kModuleBytes = 13088;
inline constexpr char kModuleSha256[] =
    "fd0665969da887dd92540b673009e0361a6c1de74fe54badf48f300b1fccbbf1";
inline constexpr char kInputGenerator[] =
    "fg-early-l5-l15-collision-tie-v1";
inline constexpr std::uint64_t kAtomicSentinel =
    0x7fffffff7fffffffull;

enum class Phase { kReset, kGenerated };
enum class Scenario { kEqualHighTie, kPriorityCollision };

inline std::string PhaseName(Phase phase) {
  return phase == Phase::kReset ? "reset" : "generated";
}

inline std::string ScenarioName(Scenario scenario) {
  return scenario == Scenario::kEqualHighTie ? "equal-high-tie"
                                              : "priority-collision";
}

inline Phase ParsePhase(const std::string& value) {
  if (value == "reset") return Phase::kReset;
  if (value == "generated") return Phase::kGenerated;
  throw std::runtime_error("invalid phase: " + value);
}

inline Scenario ParseScenario(const std::string& value) {
  if (value == "equal-high-tie") return Scenario::kEqualHighTie;
  if (value == "priority-collision") return Scenario::kPriorityCollision;
  throw std::runtime_error("invalid scenario: " + value);
}

inline std::uint64_t Mix(std::uint64_t value) {
  value += 0x9e3779b97f4a7c15ull;
  value = (value ^ (value >> 30u)) * 0xbf58476d1ce4e5b9ull;
  value = (value ^ (value >> 27u)) * 0x94d049bb133111ebull;
  return value ^ (value >> 31u);
}

inline void Store32(std::vector<std::uint8_t>* bytes, std::size_t offset,
                    std::uint32_t value) {
  std::memcpy(bytes->data() + offset, &value, sizeof(value));
}

inline void Store64(std::vector<std::uint8_t>* bytes, std::size_t offset,
                    std::uint64_t value) {
  std::memcpy(bytes->data() + offset, &value, sizeof(value));
}

inline std::vector<std::uint8_t> Flow(Phase phase, Scenario scenario) {
  const std::array<float, 3> x_values = {
      -1.0f / float(kWidth), 0.0f, 1.0f / float(kWidth)};
  const std::array<float, 3> y_values = {
      -1.0f / float(kHeight), 0.0f, 1.0f / float(kHeight)};
  const std::uint32_t phase_shift = phase == Phase::kReset ? 0u : 1u;
  const std::uint32_t scenario_shift =
      scenario == Scenario::kEqualHighTie ? 0u : 1u;
  std::vector<std::uint8_t> result(kFlowBytes);
  for (std::uint32_t y = 0; y < kHeight; ++y) {
    for (std::uint32_t x = 0; x < kWidth; ++x) {
      const std::size_t pixel = std::size_t{y} * kWidth + x;
      const auto mixed = Mix(0x243f6a8885a308d3ull ^
                             std::uint64_t{x / 4u} * 0x9e3779b97f4a7c15ull ^
                             std::uint64_t{y / 4u} * 0xbf58476d1ce4e5b9ull);
      const std::size_t x_index =
          (x + 2u * y + phase_shift + (mixed & 1u)) % 3u;
      const std::size_t y_index =
          (2u * x + y + scenario_shift + ((mixed >> 7u) & 1u)) % 3u;
      Store32(&result, pixel * 8u,
              std::bit_cast<std::uint32_t>(x_values[x_index]));
      Store32(&result, pixel * 8u + 4u,
              std::bit_cast<std::uint32_t>(y_values[y_index]));
    }
  }
  return result;
}

inline std::vector<std::uint8_t> Depth(Phase phase, Scenario scenario) {
  constexpr std::array<std::uint32_t, 3> priorities = {
      0x3e800000u,  // 0.25
      0x3f000000u,  // 0.5
      0x3f400000u,  // 0.75
  };
  const std::uint32_t phase_shift = phase == Phase::kReset ? 0u : 1u;
  std::vector<std::uint8_t> result(kDepthBytes);
  for (std::uint32_t y = 0; y < kHeight; ++y) {
    for (std::uint32_t x = 0; x < kWidth; ++x) {
      const std::size_t pixel = std::size_t{y} * kWidth + x;
      const std::uint32_t value = scenario == Scenario::kEqualHighTie
          ? priorities[1]
          : priorities[((x / 3u) + 2u * (y / 3u) + phase_shift) % 3u];
      Store32(&result, pixel * 4u, value);
    }
  }
  return result;
}

inline std::vector<std::uint8_t> InitialAtomic() {
  std::vector<std::uint8_t> result(kAtomicBytes);
  for (std::size_t pixel = 0; pixel < kPixels; ++pixel) {
    Store64(&result, pixel * 8u, kAtomicSentinel);
  }
  return result;
}

}  // namespace fg_early_l5_l15
