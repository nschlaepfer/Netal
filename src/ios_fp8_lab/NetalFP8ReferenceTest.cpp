#include "NetalFP8Reference.h"
#include <cassert>
#include <array>
using namespace netal_fp8_lab;
int main() {
    assert(Decode(0x7e,Format::E4M3)==448);assert(Decode(0xfe,Format::E4M3)==-448);
    assert(std::isnan(Decode(0x7f,Format::E4M3)));assert(std::isnan(Decode(0xff,Format::E4M3)));
    assert(Decode(0x7b,Format::E5M2)==57344);assert(std::isinf(Decode(0x7c,Format::E5M2)));
    assert(std::isnan(Decode(0x7d,Format::E5M2)));assert(std::signbit(Decode(0x80,Format::E4M3)));
    assert(Decode(1,Format::E4M3)==std::ldexp(1.0f,-9));assert(Decode(1,Format::E5M2)==std::ldexp(1.0f,-16));
    for(Format format:{Format::E4M3,Format::E5M2}) {
        for(unsigned code=0;code<256;++code) {
            const float value=Decode(std::uint8_t(code),format);
            if(std::isfinite(value))assert(EncodeFinite(value,format)==code);
        }
        assert(Decode(EncodeFinite(1e10f,format),format)==Maximum(format));
        assert(Decode(EncodeFinite(-1e10f,format),format)==-Maximum(format));
        assert(EncodeFinite(-0.0f,format)==128);
    }
    assert(EncodeFinite(1.0625f,Format::E4M3)==0x38); // tie rounds to even 1.0
    assert(EncodeFinite(1.1875f,Format::E4M3)==0x3a); // tie rounds to even 1.25
    assert(EncodeFinite(1.125f,Format::E5M2)==0x3c);
    std::array<std::uint8_t,16> guard;guard.fill(0xa5);assert(Guard(guard.data(),guard.size()));
    guard[9]=0;assert(!Guard(guard.data(),guard.size()));
    Error error;error.Add(2,1);error.Add(0,1);assert(error.maximum==1&&error.RMS()==1);
    error.Add(INFINITY,0);assert(error.nonfinite==1&&error.count==2);
    assert(std::isnan(Milliseconds(0,1,true)));assert(std::isnan(Milliseconds(4,3,true)));
    assert(std::isnan(Milliseconds(1,2,false)));assert(std::abs(Milliseconds(1,1.001,true)-1)<1e-9);
    assert(std::isnan(Median({1,NAN,3})));assert(Median({3,1,2})==2);assert(Median({4,1,2,3})==2.5);
    // Non-unit scale and nonzero bias catch applying bias in quantized units.
    const double accumulator=3,scaleA=0.5,scaleB=0.25,bias=0.75;
    assert(accumulator*scaleA*scaleB+bias==1.125);
    assert((accumulator+bias)*scaleA*scaleB!=1.125);
}
