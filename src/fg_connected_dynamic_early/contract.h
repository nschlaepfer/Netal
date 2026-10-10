#pragma once

#include "../fg_dynamic_module11/contract.h"
#include "../fg_dynamic_module13/contract.h"
#include "../fg_dynamic_module15/contract.h"
#include "../fg_dynamic_module21/contract.h"
#include "../fg_dynamic_module23/contract.h"
#include "../fg_dynamic_module25/contract.h"
#include "../fg_dynamic_module5/contract.h"
#include "../fg_resolution_preprocess_l22_l24/contract.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace fg_connected_dynamic_early {

inline constexpr char kGenerator[] =
    "fg-connected-dynamic-early-live-handoff-v1";
inline constexpr std::uint8_t kCanaryValue = 0xc7;

struct Resolution {
  const char* name;
  std::uint32_t source_width;
  std::uint32_t source_height;
  std::uint32_t color_width;
  std::uint32_t color_height;
  std::uint32_t motion_width;
  std::uint32_t motion_height;
  std::uint32_t tensor_width;
  std::uint32_t tensor_height;
  std::uint32_t module11_mode;
  std::uint32_t module23_width;
  std::uint32_t module23_height;
  std::uint32_t selected_level;
  std::uint32_t extra_level_count;
  std::uint32_t origin_y;
};

inline constexpr std::array<Resolution, 4> kResolutions{{
    {"720p", 1280, 720, 640, 360, 640, 360, 320, 192,
     2, 320, 180, 10, 1, 6},
    {"1080p", 1920, 1080, 960, 540, 960, 540, 480, 288,
     2, 480, 270, 10, 1, 9},
    {"1440p", 2560, 1440, 1280, 720, 1280, 720, 640, 384,
     2, 320, 180, 11, 2, 12},
    {"4k", 3840, 2160, 960, 540, 1920, 1080, 480, 288,
     4, 480, 270, 11, 2, 9},
}};

inline const Resolution& Find(const std::string& name) {
  for (const auto& resolution : kResolutions)
    if (name == resolution.name) return resolution;
  throw std::runtime_error("unknown connected dynamic resolution");
}

inline std::uint32_t CeilDiv(std::uint32_t value, std::uint32_t divisor) {
  return (value + divisor - 1u) / divisor;
}

inline const fg_dynamic_module11::Resolution& Module11(
    const Resolution& resolution) {
  return fg_dynamic_module11::ParseResolution(resolution.name);
}

inline const fg_dynamic_module15::Resolution& Module15(
    const Resolution& resolution) {
  return fg_dynamic_module15::ParseResolution(resolution.name);
}

inline const fg_dynamic_module13::Resolution& Module13(
    const Resolution& resolution) {
  return fg_dynamic_module13::ParseResolution(resolution.name);
}

inline const fg_dynamic_module21::R& Module21(
    const Resolution& resolution) {
  return fg_dynamic_module21::Find(resolution.name);
}

inline const fg_dynamic_module23::R& Module23(
    const Resolution& resolution) {
  return fg_dynamic_module23::Find(resolution.name);
}

inline const fg_dynamic_module25::R& Module25(
    const Resolution& resolution) {
  return fg_dynamic_module25::Find(resolution.name);
}

inline const fg_dynamic_module5::Resolution& Module5(
    const Resolution& resolution) {
  return fg_dynamic_module5::FindResolution(resolution.name);
}

inline const fg_resolution_l22_l24::Resolution& ResolutionPreprocess(
    const Resolution& resolution) {
  return fg_resolution_l22_l24::ParseResolution(resolution.name);
}

struct ExternalInputs {
  std::vector<std::uint8_t> source_reset;
  std::vector<std::uint8_t> source_generated;
  std::vector<std::uint8_t> vector_reset;
  std::vector<std::uint8_t> vector_generated;
  std::vector<std::uint8_t> scalar_reset;
  std::vector<std::uint8_t> scalar_generated;
};

inline ExternalInputs GenerateExternal(const Resolution& resolution) {
  const auto& module11 = Module11(resolution);
  const auto& module15 = Module15(resolution);
  return {
      fg_dynamic_module11::Source(module11, false),
      fg_dynamic_module11::Source(module11, true),
      fg_dynamic_module15::VectorInput(module15, false),
      fg_dynamic_module15::VectorInput(module15, true),
      fg_dynamic_module15::ScalarInput(module15, false),
      fg_dynamic_module15::ScalarInput(module15, true),
  };
}

inline std::vector<std::uint8_t> History(
    const Resolution& resolution, std::uint8_t value) {
  return std::vector<std::uint8_t>(
      std::size_t{resolution.source_width} * resolution.source_height * 8u,
      value);
}

inline std::vector<std::uint8_t> InitialGeneratedDepth(
    const Resolution& resolution) {
  return fg_dynamic_module13::Depth(
      Module13(resolution), fg_dynamic_module13::Phase::kGenerated,
      fg_dynamic_module13::Scenario::kPriorityCollision);
}

inline std::vector<std::uint8_t> InitialGeneratedMask(
    const Resolution& resolution) {
  return std::vector<std::uint8_t>(
      std::size_t{resolution.motion_width} * resolution.motion_height, 1u);
}

inline std::vector<std::uint8_t> InitialAtomic(
    const Resolution& resolution) {
  return fg_dynamic_module13::InitialAtomic(Module13(resolution));
}

inline std::vector<std::uint8_t> InitialAllocation29(
    const Resolution& resolution) {
  const std::size_t pixels =
      std::size_t{resolution.color_width} * resolution.color_height;
  std::vector<std::uint8_t> result(pixels * 2u);
  constexpr std::uint16_t half = 0x3800u;
  for (std::size_t pixel = 0; pixel < pixels; ++pixel)
    std::memcpy(result.data() + pixel * 2u, &half, sizeof(half));
  return result;
}

inline std::array<std::size_t, 5> PyramidBytes(
    const Resolution& resolution) {
  return fg_dynamic_module21::Out(Module21(resolution));
}

inline std::array<std::size_t, 4> ExtraPyramidBytes(
    const Resolution& resolution) {
  std::array<std::size_t, 4> result{};
  std::uint32_t width = CeilDiv(resolution.motion_width, 16u);
  std::uint32_t height = CeilDiv(resolution.motion_height, 16u);
  for (auto& bytes : result) {
    width = CeilDiv(width, 2u);
    height = CeilDiv(height, 2u);
    bytes = std::size_t{width} * height * 6u;
  }
  return result;
}

inline std::size_t ColorBytes(const Resolution& resolution) {
  return std::size_t{resolution.color_width} * resolution.color_height * 8u;
}

inline std::size_t MetricBytes(const Resolution& resolution) {
  return std::size_t{resolution.color_width} * resolution.color_height * 2u;
}

inline std::size_t MotionPixels(const Resolution& resolution) {
  return std::size_t{resolution.motion_width} * resolution.motion_height;
}

inline std::size_t ModelInputBytes(const Resolution& resolution) {
  return std::size_t{resolution.tensor_width} * resolution.tensor_height *
         48u * 2u;
}

}  // namespace fg_connected_dynamic_early
