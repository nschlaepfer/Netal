#include "netal_sr_boundary_geometry.h"
#include <cassert>
#include <iostream>

int main() {
  const auto base=netal::BoundaryLimits(1280,720,1920,1080);
  assert(base.x==-24.0f && base.y==-24.0f);
  const auto performance=netal::BoundaryLimits(960,540,1920,1080);
  assert(performance.x==-32.0f && performance.y==-32.0f);
  const auto asymmetric=netal::BoundaryLimits(854,534,1280,800);
  assert(asymmetric.x!=asymmetric.y);
  const auto native=netal::BoundaryLimits(1280,800,1280,800);
  assert(native.x==-16.0f && native.y==-16.0f);
  const auto ratio=netal::RenderRatios(1280,720,2560,1440);
  assert(ratio.x==0.5f && ratio.y==0.5f);
  std::cout << "5 boundary geometry cases passed\n";
}
