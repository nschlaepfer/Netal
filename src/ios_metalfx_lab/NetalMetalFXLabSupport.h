#pragma once
#include "../ios_probe/NetalProbeMetrics.h"
#include <cstddef>
#include <cstdint>
#include <limits>

namespace netal_metalfx {
constexpr std::size_t kGuard = 256;
constexpr std::uint16_t kPoison = 0x7e00;
inline bool RowLayout(std::uint32_t width, std::uint32_t height, std::size_t pixelBytes,
                      std::size_t* row, std::size_t* bytes) {
  if (!width || !height || width > 4096 || height > 4096 ||
      (pixelBytes != 2 && pixelBytes != 4 && pixelBytes != 8)) return false;
  const auto payload = std::size_t(width) * pixelBytes;
  if (payload > std::numeric_limits<std::size_t>::max() - 255) return false;
  const auto stride = (payload + 255) & ~std::size_t(255);
  if (height > (std::numeric_limits<std::size_t>::max() - 2*kGuard) / stride) return false;
  *row = stride; *bytes = stride * height + 2*kGuard; return true;
}
inline bool ValidGeometry(netal_probe::Dimensions render, netal_probe::Dimensions output) {
  return render.width <= 4096 && render.height <= 4096 && output.width <= 4096 &&
         output.height <= 4096 && netal_probe::RenderFitsOutput(render, output);
}
inline bool WithinScaleRange(netal_probe::Dimensions render, netal_probe::Dimensions output,
                             double minimum, double maximum) {
  if (!ValidGeometry(render, output) || !std::isfinite(minimum) || !std::isfinite(maximum) ||
      minimum <= 0 || maximum < minimum) return false;
  const double x=double(output.width)/render.width, y=double(output.height)/render.height;
  return x >= minimum && x <= maximum && y >= minimum && y <= maximum;
}
} // namespace netal_metalfx
