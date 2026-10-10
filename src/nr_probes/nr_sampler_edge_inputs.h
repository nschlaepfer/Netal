#pragma once
#include <array>
#include <cstdint>
#include <vector>
// Deterministic synthetic input, independent of model/expected bytes. Same
// dimensions and RGBA16F format as the NR history surface. Four tap words are
// rotated per channel to expose orientation/accumulation-order differences.
struct EdgeCase { const char* name; std::array<uint16_t,4> taps; };
inline const std::vector<EdgeCase>& EdgeCases() {
  static const std::vector<EdgeCase> c={
    {"positive-zero",{0,0,0,0}}, {"negative-zero",{0x8000,0x8000,0x8000,0x8000}},
    {"mixed-zero",{0,0x8000,0,0x8000}}, {"one",{0x3c00,0x3c00,0x3c00,0x3c00}},
    {"cancel-horizontal",{0x3c00,0xbc00,0x3c00,0xbc00}},
    {"cancel-vertical",{0x3c00,0x3c00,0xbc00,0xbc00}},
    {"cancel-checker",{0x3c00,0xbc00,0xbc00,0x3c00}},
    {"cancel-min-subnormal",{0x0001,0x8001,0x0001,0x8001}},
    {"cancel-min-normal",{0x0400,0x8400,0x0400,0x8400}},
    {"cancel-max-finite",{0x7bff,0xfbff,0xfbff,0x7bff}},
    {"positive-inf",{0x7c00,0x7c00,0x7c00,0x7c00}},
    {"negative-inf",{0xfc00,0xfc00,0xfc00,0xfc00}},
    {"opposite-inf",{0x7c00,0xfc00,0x7c00,0xfc00}},
    {"quiet-nan",{0x7e00,0x7e00,0x7e00,0x7e00}},
    {"signaling-nan",{0x7d00,0x7d00,0x7d00,0x7d00}},
    {"nan-payloads",{0x7e01,0x7fff,0xfe01,0xffff}},
    {"finite-next-to-nan",{0x3c00,0x7e00,0x3c00,0x7e00}},
    {"finite-next-to-inf",{0x3c00,0x7c00,0x3c00,0x7c00}},
    {"zero-next-to-nan",{0,0x7e00,0,0x7e00}},
    {"negative-zero-next-to-inf",{0x8000,0x7c00,0x8000,0x7c00}},
    {"negative-cancel-order",{0xbc00,0x3c00,0xbc00,0x3c00}},
    {"near-cancellation",{0x3c00,0xbc01,0x3c01,0xbc00}}
  }; return c;
}
constexpr int EDGE_W=1920,EDGE_H=1080;
inline std::vector<uint16_t> EdgeImage() {
  std::vector<uint16_t> image(size_t(EDGE_W)*EDGE_H*4,0);
  for(size_t n=0;n<EdgeCases().size();++n)for(int tap=0;tap<4;++tap)for(int ch=0;ch<4;++ch){
    size_t x=8*n+2+(tap%2),y=8+(tap/2);
    image[(y*EDGE_W+x)*4+ch]=EdgeCases()[n].taps[(tap+ch)%4];
  } return image;
}
inline std::vector<std::array<float,2>> EdgeQueries() {
  std::vector<std::array<float,2>> q;
  const std::array<float,7> f={0,1.f/512,127.f/256,.5f,129.f/256,255.f/256,1};
  for(size_t n=0;n<EdgeCases().size();++n)for(float y:f)for(float x:f)
    q.push_back({(float(8*n+2)+.5f+x)/float(EDGE_W),(8.5f+y)/float(EDGE_H)});
  return q;
}
