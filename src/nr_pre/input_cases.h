#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

// Original-image inputs only. No model output or intermediate is consulted.
// Fixed descriptor/reset experiments; not temporal or arbitrary-shape support.
inline void NrColorCase(std::vector<uint16_t>& color, unsigned width,
                        unsigned height, unsigned variant) {
  if (variant > 3 || width == 0 || height == 0)
    throw std::runtime_error("unsupported NR color case");
  constexpr uint16_t palette[] = {0x0000,0x3000,0x3400,0x3800,0x3a00,0x3c00};
  color.resize(static_cast<std::size_t>(width)*height*4);
  for (unsigned y=0;y<height;++y) for (unsigned x=0;x<width;++x) {
    auto* p=color.data()+(static_cast<std::size_t>(y)*width+x)*4;
    p[3]=0x3c00;
    if (variant==0) {
      p[0]=palette[(x/64)%6];p[1]=palette[(y/64)%6];p[2]=palette[((x+y)/64)%6];
    } else if (variant==1) {
      p[0]=p[1]=p[2]=0;
    } else if (variant==2) {
      p[0]=((x^y)&1)?0x3c00:0;
      p[1]=(x%17==0)?0x3c00:0x3000;
      p[2]=(y%31==0)?0x3c00:0x3400;
    } else {
      // Dense finite positive half bit patterns in [2^-7,1], not float math.
      for (unsigned c=0;c<3;++c) {
        const uint32_t mix=(x*(37u+c*18u))^(y*(83u+c*34u))^(c*0x159du);
        p[c]=static_cast<uint16_t>(0x2000u+mix%0x1c01u);
      }
    }
  }
}
