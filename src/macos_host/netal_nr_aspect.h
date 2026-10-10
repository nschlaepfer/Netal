#pragma once
#include <cstdint>
#include <algorithm>
#include <numeric>

namespace netal_nr_adapter {
// Matches two uint4 fields in Metal. Pixel centers are used in both directions.
struct alignas(16) AspectLayout {
  uint32_t game_width=0, game_height=0, model_width=1920, model_height=1080;
  uint32_t x=0, y=0, width=0, height=0;
  bool identity() const { return game_width==model_width && game_height==model_height; }
};
static_assert(sizeof(AspectLayout)==32);
inline bool MakeAspectLayout(uint32_t w,uint32_t h,AspectLayout& out) {
  out=AspectLayout{};
  if(!w || !h || w>16384 || h>16384)return false;
  // Preserve the ratio EXACTLY for admitted integer-ratio profiles, rather than
  // independently rounding width and height and introducing an anisotropic scale.
  const uint32_t g=std::gcd(w,h), rw=w/g, rh=h/g;
  const uint32_t k=std::min(out.model_width/rw,out.model_height/rh);
  if(!k)return false;
  out.game_width=w;out.game_height=h;out.width=rw*k;out.height=rh*k;
  out.x=(out.model_width-out.width)/2;out.y=(out.model_height-out.height)/2;
  return true;
}
inline bool SupportedGameExtent(uint32_t w,uint32_t h) {
  // Initial measured profiles only. The layout math can be extended without
  // silently declaring additional game resolutions supported.
  return w==1920 && (h==1080 || h==1200);
}
} // namespace netal_nr_adapter
