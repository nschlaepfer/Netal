#pragma once
// CPU reference for the conditional input-pixel motion conversion. No Metal
// execution or temporal admission. Game convention and sampling policy require
// separate capture/reprojection evidence before this can feed live NR history.
#include "netal_nr_aspect.h"
#include <cmath>
#include <limits>

namespace netal_nr_adapter {
struct MotionGeometryInput {
  uint32_t game_width=0, game_height=0;
  uint32_t color_width=0, color_height=0, motion_width=0, motion_height=0;
  int motion_at_output_resolution=-1, motion_includes_jitter=-1;
  int64_t color_x=-1, color_y=-1, motion_x=-1, motion_y=-1, output_x=-1, output_y=-1;
  float scale_x=1, scale_y=1; // SDK: raw vector * scale = color-texture pixels.
};
enum class MotionGeometryError {
  None, UnsupportedGameExtent, InvalidInputExtent, UnsupportedFlags,
  UnsupportedSubrect, InvalidScale
};
struct MotionGeometry {
  AspectLayout aspect{};
  uint32_t input_width=0, input_height=0;
  double vector_scale_x=0, vector_scale_y=0;
  bool valid=false;
};
inline MotionGeometryError PlanInputPixelMotion(const MotionGeometryInput& in, MotionGeometry& out) {
  out=MotionGeometry{}; // Never leave an earlier valid plan after rejection.
  if(!SupportedGameExtent(in.game_width,in.game_height))return MotionGeometryError::UnsupportedGameExtent;
  if(!in.color_width || !in.color_height || in.color_width>16384 || in.color_height>16384 ||
     in.motion_width!=in.color_width || in.motion_height!=in.color_height)
    return MotionGeometryError::InvalidInputExtent;
  // Output-resolution motion is NOT assumed to use output-pixel units. Its
  // spatial-resolution flag and vector units are different questions.
  if(in.motion_at_output_resolution!=0 || in.motion_includes_jitter!=0)
    return MotionGeometryError::UnsupportedFlags;
  if(in.color_x!=0 || in.color_y!=0 || in.motion_x!=0 || in.motion_y!=0 || in.output_x!=0 || in.output_y!=0)
    return MotionGeometryError::UnsupportedSubrect;
  if(!std::isfinite(in.scale_x) || !std::isfinite(in.scale_y) || in.scale_x==0 || in.scale_y==0)
    return MotionGeometryError::InvalidScale;
  MotionGeometry p;
  if(!MakeAspectLayout(in.game_width,in.game_height,p.aspect))return MotionGeometryError::UnsupportedGameExtent;
  p.input_width=in.color_width;p.input_height=in.color_height;
  p.vector_scale_x=double(in.scale_x)*p.aspect.width/in.color_width;
  p.vector_scale_y=double(in.scale_y)*p.aspect.height/in.color_height;
  if(!std::isfinite(p.vector_scale_x) || !std::isfinite(p.vector_scale_y) ||
     std::abs(p.vector_scale_x)>std::numeric_limits<float>::max() ||
     std::abs(p.vector_scale_y)>std::numeric_limits<float>::max())return MotionGeometryError::InvalidScale;
  p.valid=true;out=p;return MotionGeometryError::None;
}
enum class MotionRegion { Invalid, Padding, Content };
struct MotionSampleLocation { MotionRegion region=MotionRegion::Invalid; double x=0,y=0; };
// Continuous texel coordinates, where integers denote input texel centers.
// This deliberately does not select nearest, bilinear or depth-aware sampling.
inline MotionSampleLocation MotionLocation(const MotionGeometry& p,uint32_t x,uint32_t y) {
  if(!p.valid || x>=p.aspect.model_width || y>=p.aspect.model_height)return {};
  const auto& a=p.aspect;
  if(x<a.x || y<a.y || x-a.x>=a.width || y-a.y>=a.height)
    return {MotionRegion::Padding,0,0};
  return {MotionRegion::Content,
    (double(x-a.x)+0.5)*p.input_width/a.width-0.5,
    (double(y-a.y)+0.5)*p.input_height/a.height-0.5};
}
struct ModelMotion { double x=0,y=0; };
// Output is model-canvas pixel displacement (current-to-previous, y down IF the
// game satisfies the SDK contract). Padding receives zero, never edge motion.
// RG16F output must not overflow; rejection requires fallback/reset, not clipping.
inline bool ConvertInputMotion(const MotionGeometry& p,MotionRegion region,double raw_x,double raw_y,ModelMotion& out) {
  out={};
  if(!p.valid || region==MotionRegion::Invalid)return false;
  if(region==MotionRegion::Padding)return true; // No input sample is consumed.
  if(region!=MotionRegion::Content)return false;
  if(!std::isfinite(raw_x) || !std::isfinite(raw_y))return false;
  const double x=raw_x*p.vector_scale_x,y=raw_y*p.vector_scale_y;
  if(!std::isfinite(x) || !std::isfinite(y) || std::abs(x)>65504 || std::abs(y)>65504)return false;
  out={x,y};return true;
}
} // namespace netal_nr_adapter
