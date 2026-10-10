#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace neuralstudio::fg {

// A small, deterministic temporal scene used only to exercise the archived
// NVIDIA frame-generation runtime. It is deliberately synthetic so every
// input byte can be regenerated and hashed independently of a game capture.
struct SyntheticFrame {
  std::vector<std::uint8_t> backbuffer_rgba8;
  std::vector<std::uint8_t> hudless_rgba8;
  std::vector<std::uint8_t> ui_alpha_r8;
  std::vector<std::uint8_t> ui_color_alpha_rgba8;
  std::vector<std::uint16_t> motion_rg16f;
  std::vector<float> depth_r32f;
  std::vector<std::uint16_t> bidirectional_distortion_rgba16f;
};

inline std::uint16_t FloatToHalf(float value) {
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  const std::uint32_t sign = (bits >> 16u) & 0x8000u;
  const std::uint32_t exponent = (bits >> 23u) & 0xffu;
  std::uint32_t mantissa = bits & 0x7fffffu;

  if (exponent == 0xffu) {
    return static_cast<std::uint16_t>(
        sign | (mantissa == 0 ? 0x7c00u : 0x7e00u));
  }

  const int half_exponent = static_cast<int>(exponent) - 127 + 15;
  if (half_exponent >= 31) {
    return static_cast<std::uint16_t>(sign | 0x7c00u);
  }
  if (half_exponent <= 0) {
    if (half_exponent < -10) {
      return static_cast<std::uint16_t>(sign);
    }
    mantissa |= 0x800000u;
    const unsigned int shift = static_cast<unsigned int>(14 - half_exponent);
    std::uint32_t rounded = mantissa >> shift;
    const std::uint32_t remainder_mask = (1u << shift) - 1u;
    const std::uint32_t remainder = mantissa & remainder_mask;
    const std::uint32_t halfway = 1u << (shift - 1u);
    if (remainder > halfway || (remainder == halfway && (rounded & 1u))) {
      ++rounded;
    }
    return static_cast<std::uint16_t>(sign | rounded);
  }

  std::uint32_t rounded_mantissa = mantissa >> 13u;
  const std::uint32_t remainder = mantissa & 0x1fffu;
  if (remainder > 0x1000u ||
      (remainder == 0x1000u && (rounded_mantissa & 1u))) {
    ++rounded_mantissa;
    if (rounded_mantissa == 0x400u) {
      rounded_mantissa = 0;
      if (half_exponent + 1 >= 31) {
        return static_cast<std::uint16_t>(sign | 0x7c00u);
      }
      return static_cast<std::uint16_t>(
          sign | (static_cast<std::uint32_t>(half_exponent + 1) << 10u));
    }
  }
  return static_cast<std::uint16_t>(
      sign | (static_cast<std::uint32_t>(half_exponent) << 10u) |
      rounded_mantissa);
}

inline bool HalfIsFinite(std::uint16_t value) {
  return (value & 0x7c00u) != 0x7c00u;
}

inline std::uint64_t Fnv1a64(const void* data, std::size_t byte_count) {
  const auto* bytes = static_cast<const std::uint8_t*>(data);
  std::uint64_t hash = 1469598103934665603ull;
  for (std::size_t index = 0; index < byte_count; ++index) {
    hash ^= bytes[index];
    hash *= 1099511628211ull;
  }
  return hash;
}

template <typename T>
inline std::uint64_t Fnv1a64(const std::vector<T>& values) {
  return Fnv1a64(values.data(), values.size() * sizeof(T));
}

inline SyntheticFrame MakeSyntheticFrame(std::uint32_t width,
                                         std::uint32_t height,
                                         std::uint32_t rendered_frame_id) {
  if (width < 16 || height < 16 || width > 8192 || height > 8192) {
    throw std::invalid_argument("synthetic dimensions are outside 16..8192");
  }
  const std::size_t pixels = static_cast<std::size_t>(width) * height;
  if (pixels > std::numeric_limits<std::size_t>::max() / 8u) {
    throw std::overflow_error("synthetic frame byte count overflow");
  }

  SyntheticFrame frame;
  frame.backbuffer_rgba8.resize(pixels * 4u);
  frame.hudless_rgba8.resize(pixels * 4u);
  frame.ui_alpha_r8.resize(pixels);
  frame.ui_color_alpha_rgba8.resize(pixels * 4u);
  frame.motion_rg16f.resize(pixels * 2u);
  frame.depth_r32f.resize(pixels);
  frame.bidirectional_distortion_rgba16f.assign(pixels * 4u,
                                                FloatToHalf(0.0f));

  // The scene translates two pixels to the right per rendered frame. Motion
  // is expressed in pixel units and normalized at evaluation with
  // mvecScale=(1/width,1/height), matching the NGX DLSSG contract.
  const int translation = static_cast<int>(rendered_frame_id) * 2;
  const std::uint16_t motion_x = FloatToHalf(-2.0f);
  const std::uint16_t motion_y = FloatToHalf(0.0f);
  const std::uint16_t zero = FloatToHalf(0.0f);

  for (std::uint32_t y = 0; y < height; ++y) {
    for (std::uint32_t x = 0; x < width; ++x) {
      const std::size_t pixel = static_cast<std::size_t>(y) * width + x;
      int world_x = static_cast<int>(x) - translation;
      world_x %= static_cast<int>(width);
      if (world_x < 0) world_x += static_cast<int>(width);

      const float gradient_x = static_cast<float>(world_x) /
                               static_cast<float>(width - 1u);
      const float gradient_y =
          static_cast<float>(y) / static_cast<float>(height - 1u);
      const bool tile = ((world_x / 16) + (static_cast<int>(y) / 16)) % 2 != 0;
      const std::uint8_t scene[4] = {
          static_cast<std::uint8_t>(20.0f + 184.0f * gradient_x),
          static_cast<std::uint8_t>(15.0f + 174.0f * gradient_y),
          static_cast<std::uint8_t>(tile ? 184u : 46u),
          255u,
      };

      std::uint8_t ui_alpha = 0u;
      if (y < 6u || (x >= width / 2u - 1u && x <= width / 2u + 1u &&
                     y >= height / 2u - 8u && y <= height / 2u + 8u)) {
        ui_alpha = 192u;
      }
      const std::uint8_t ui_color[3] = {255u, 199u, 31u};

      for (std::size_t channel = 0; channel < 4u; ++channel) {
        const std::uint8_t hudless = scene[channel];
        const std::uint8_t composited =
            channel == 3u
                ? 255u
                : static_cast<std::uint8_t>(
                      (static_cast<std::uint32_t>(ui_color[channel]) *
                           ui_alpha +
                       static_cast<std::uint32_t>(hudless) *
                           (255u - ui_alpha) +
                       127u) /
                      255u);
        frame.hudless_rgba8[pixel * 4u + channel] = hudless;
        frame.backbuffer_rgba8[pixel * 4u + channel] = composited;
      }
      frame.ui_alpha_r8[pixel] = ui_alpha;
      for (std::size_t channel = 0; channel < 3u; ++channel) {
        frame.ui_color_alpha_rgba8[pixel * 4u + channel] =
            static_cast<std::uint8_t>(
                (static_cast<std::uint32_t>(ui_color[channel]) * ui_alpha +
                 127u) /
                255u);
      }
      frame.ui_color_alpha_rgba8[pixel * 4u + 3u] = ui_alpha;
      frame.motion_rg16f[pixel * 2u] = motion_x;
      frame.motion_rg16f[pixel * 2u + 1u] = motion_y;
      frame.depth_r32f[pixel] = 0.15f + 0.70f * gradient_y;
    }
  }

  // Make the first submitted frame a reset frame with no valid history.
  if (rendered_frame_id == 0) {
    for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
      frame.motion_rg16f[pixel * 2u] = zero;
      frame.motion_rg16f[pixel * 2u + 1u] = zero;
    }
  }

  return frame;
}

}  // namespace neuralstudio::fg
