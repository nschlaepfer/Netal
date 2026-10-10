#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace fg_dynamic_module13 {

inline constexpr char kModuleSha256[] =
    "fd0665969da887dd92540b673009e0361a6c1de74fe54badf48f300b1fccbbf1";
inline constexpr std::size_t kModuleBytes = 13088;
inline constexpr std::size_t kPackedParameterBytes = 144;
inline constexpr std::uint64_t kAtomicSentinel = 0x7fffffff7fffffffull;
inline constexpr char kGenerator[] = "fg-dynamic-module13-collision-tie-v1";

enum class Phase { kReset, kGenerated };
enum class Scenario { kEqualHighTie, kPriorityCollision };

struct Resolution {
  const char* name;
  std::uint32_t width;
  std::uint32_t height;
  std::uint32_t grid_x;
  std::uint32_t grid_y;
  std::uint32_t input_scale_x_bits;
  std::uint32_t input_scale_y_bits;
  std::uint32_t motion_divisor_bits;
  std::uint32_t neighbor_divisor_bits;
  const char* reset_raw_sha256;
  const char* generated_raw_sha256;
  const char* address_free_sha256;
};

inline constexpr std::array<Resolution, 4> kResolutions = {{
    {"720p", 640, 360, 40, 23, 0x3a4ccccd, 0x3ab60b61,
     0x44100000, 0x424f5c29,
     "e9ce58929f567348052851a720dbfb9eb3c318bfa36b40aa1132f5998cc4cb97",
     "d4370c5a4e5f51bcebe43b6ec48c43f256e64dc1ae757cbd62bd65fe50dcd45a",
     "e21e039797c67fa9b2c5791292e71a533f8079bea6f7f92dd421d97797e5d9c3"},
    {"1080p", 960, 540, 60, 34, 0x3a088889, 0x3a72b9d6,
     0x44a20000, 0x42e947ae,
     "e5cd86e489b3453380f42ef359b0730ec6c57c715d45085c566316f4c6e7d239",
     "56d6a52cb27f7e2431fefa88aaef767c173db5015302e13d2bcd04988db31bcc",
     "5b4f9354f8e86ff28baca22ffb4d726ab569f75ed9759ac3e03e1e35879b643b"},
    {"1440p", 1280, 720, 80, 45, 0x39cccccd, 0x3a360b61,
     0x45100000, 0x434f5c29,
     "e3859d5c8b3a6524e23b193c6acf9ca3c5d6ffe51b9febba7e222c502d3f6de7",
     "c34f09b9d394588b79c7561c0070caada99f48831e153beafe0fc6e82ebe594e",
     "5bf8a3a78f2b2362eadf6922ded51c7a9d9a3a734cd687037826169b45d563ca"},
    {"4k", 1920, 1080, 120, 68, 0x39888889, 0x39f2b9d6,
     0x45a20000, 0x43e947ae,
     "987bc5e293e59a374ed399493d08af7a3b0df237a941acd3511d9a5a2c00161e",
     "66395f312d9fe003b6f0310137d45227764b55c08f55de200110f2a563de2b48",
     "daab11d2a4e4d69ee0cfc8fa4c70b7daa13b30ec4447c2fb120b977cccb90431"},
}};

inline const Resolution& ParseResolution(const std::string& value) {
  for (const auto& resolution : kResolutions)
    if (value == resolution.name) return resolution;
  throw std::runtime_error("invalid resolution: " + value);
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

inline const char* PhaseName(Phase phase) {
  return phase == Phase::kReset ? "reset" : "generated";
}

inline const char* ScenarioName(Scenario scenario) {
  return scenario == Scenario::kEqualHighTie ? "equal-high-tie"
                                              : "priority-collision";
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

inline std::size_t Pixels(const Resolution& resolution) {
  return std::size_t{resolution.width} * resolution.height;
}

inline std::vector<std::uint8_t> Flow(
    const Resolution& resolution, Phase phase, Scenario scenario) {
  const std::array<float, 3> x_values = {
      -1.0f / float(resolution.width), 0.0f, 1.0f / float(resolution.width)};
  const std::array<float, 3> y_values = {
      -1.0f / float(resolution.height), 0.0f, 1.0f / float(resolution.height)};
  const std::uint32_t phase_shift = phase == Phase::kReset ? 0u : 1u;
  const std::uint32_t scenario_shift =
      scenario == Scenario::kEqualHighTie ? 0u : 1u;
  std::vector<std::uint8_t> result(Pixels(resolution) * 8u);
  for (std::uint32_t y = 0; y < resolution.height; ++y) {
    for (std::uint32_t x = 0; x < resolution.width; ++x) {
      const std::size_t pixel = std::size_t{y} * resolution.width + x;
      const auto mixed = Mix(0x243f6a8885a308d3ull ^
                             std::uint64_t{x / 4u} * 0x9e3779b97f4a7c15ull ^
                             std::uint64_t{y / 4u} * 0xbf58476d1ce4e5b9ull);
      const std::size_t xi =
          (x + 2u * y + phase_shift + (mixed & 1u)) % 3u;
      const std::size_t yi =
          (2u * x + y + scenario_shift + ((mixed >> 7u) & 1u)) % 3u;
      Store32(&result, pixel * 8u,
              std::bit_cast<std::uint32_t>(x_values[xi]));
      Store32(&result, pixel * 8u + 4u,
              std::bit_cast<std::uint32_t>(y_values[yi]));
    }
  }
  return result;
}

inline std::vector<std::uint8_t> Depth(
    const Resolution& resolution, Phase phase, Scenario scenario) {
  constexpr std::array<std::uint32_t, 3> priorities = {
      0x3e800000u, 0x3f000000u, 0x3f400000u};
  const std::uint32_t phase_shift = phase == Phase::kReset ? 0u : 1u;
  std::vector<std::uint8_t> result(Pixels(resolution) * 4u);
  for (std::uint32_t y = 0; y < resolution.height; ++y) {
    for (std::uint32_t x = 0; x < resolution.width; ++x) {
      const std::size_t pixel = std::size_t{y} * resolution.width + x;
      const auto value = scenario == Scenario::kEqualHighTie
          ? priorities[1]
          : priorities[((x / 3u) + 2u * (y / 3u) + phase_shift) % 3u];
      Store32(&result, pixel * 4u, value);
    }
  }
  return result;
}

inline std::vector<std::uint8_t> InitialAtomic(
    const Resolution& resolution) {
  std::vector<std::uint8_t> result(Pixels(resolution) * 8u);
  for (std::size_t pixel = 0; pixel < Pixels(resolution); ++pixel)
    Store64(&result, pixel * 8u, kAtomicSentinel);
  return result;
}

}  // namespace fg_dynamic_module13
