#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

// Original procedural reference for OCP OFP8 formats. No model or vendor code.
namespace netal_fp8_lab {
enum class Format { E4M3, E5M2 };
inline float Decode(std::uint8_t code, Format format) {
    const unsigned magnitude=code&127u;
    const float sign=(code&128u)?-1.0f:1.0f;
    if(format==Format::E4M3) {
        if(magnitude==127u)return std::numeric_limits<float>::quiet_NaN();
        const int exponent=magnitude>>3u, mantissa=magnitude&7u;
        return sign*(exponent?std::ldexp(1.0f+mantissa/8.0f,exponent-7):std::ldexp(float(mantissa),-9));
    }
    const int exponent=magnitude>>2u,mantissa=magnitude&3u;
    if(exponent==31)return mantissa?std::numeric_limits<float>::quiet_NaN():sign*std::numeric_limits<float>::infinity();
    return sign*(exponent?std::ldexp(1.0f+mantissa/4.0f,exponent-15):std::ldexp(float(mantissa),-16));
}
inline float Maximum(Format format) { return format==Format::E4M3?448.0f:57344.0f; }
// Finite inputs only. Search monotonically ordered positive codes; choose the
// nearest endpoint, then the even code on a tie. Independent of host rounding.
inline std::uint8_t EncodeFinite(float value,Format format) {
    const unsigned sign=std::signbit(value)?128u:0u;
    const float magnitude=std::abs(value);
    const unsigned last=format==Format::E4M3?126u:123u;
    unsigned low=0,high=last;
    while(low<high){const unsigned mid=(low+high)/2;if(Decode(std::uint8_t(mid),format)<magnitude)low=mid+1;else high=mid;}
    unsigned best=low;
    if(low){const double upper=std::abs(double(magnitude)-Decode(std::uint8_t(low),format));
        const double lower=std::abs(double(magnitude)-Decode(std::uint8_t(low-1),format));
        if(lower<upper||(lower==upper&&((low-1)&1u)==0))best=low-1;}
    return std::uint8_t(sign|best);
}
inline bool Guard(const std::uint8_t* bytes,std::size_t size,std::uint8_t value=0xa5) {
    for(std::size_t i=0;i<size;++i)if(bytes[i]!=value)return false;
    return true;
}
struct Error {
    double maximum=0,squares=0; std::size_t count=0,nonfinite=0;
    void Add(double actual,double expected) {
        if(!std::isfinite(actual)||!std::isfinite(expected)){++nonfinite;return;}
        const double delta=std::abs(actual-expected);
        maximum=std::max(maximum,delta);squares+=delta*delta;++count;
    }
    double RMS() const {return count?std::sqrt(squares/count):std::numeric_limits<double>::quiet_NaN();}
};
inline double Milliseconds(double start,double end,bool completed) {
    if(!completed||!std::isfinite(start)||!std::isfinite(end)||start<=0||end<=start)
        return std::numeric_limits<double>::quiet_NaN();
    const double ms=(end-start)*1000.0;
    return std::isfinite(ms)&&ms>0?ms:std::numeric_limits<double>::quiet_NaN();
}
inline double Median(std::vector<double> samples) {
    if(samples.empty())return std::numeric_limits<double>::quiet_NaN();
    for(double x:samples)if(!std::isfinite(x)||x<=0)return std::numeric_limits<double>::quiet_NaN();
    std::sort(samples.begin(),samples.end());const auto middle=samples.size()/2;
    return samples.size()%2?samples[middle]:(samples[middle-1]+samples[middle])/2;
}
} // namespace netal_fp8_lab
