#pragma once
#include <cstdint>

namespace netal {
struct SRRenderRatios { float x; float y; };
struct SRBoundaryLimits { float x; float y; };

// Dimensions are validated by the session before use. Preserve float32 reciprocal-then-
// multiply, rather than substituting a direct division with different rounding.
inline SRRenderRatios RenderRatios(std::uint32_t iw, std::uint32_t ih,
                                  std::uint32_t ow, std::uint32_t oh) {
  return {static_cast<float>(iw) * (1.0f / static_cast<float>(ow)),
          static_cast<float>(ih) * (1.0f / static_cast<float>(oh))};
}
inline SRBoundaryLimits BoundaryLimits(std::uint32_t iw, std::uint32_t ih,
                                      std::uint32_t ow, std::uint32_t oh) {
  return {static_cast<float>(-16.0 * static_cast<double>(ow) / iw),
          static_cast<float>(-16.0 * static_cast<double>(oh) / ih)};
}
}  // namespace netal
