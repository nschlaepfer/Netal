#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string_view>
#include <vector>
#ifdef __OBJC__
#import <Foundation/Foundation.h>
#endif

namespace netal_probe {
struct Dimensions { std::uint32_t width=0,height=0; };
inline bool ParseSize(std::string_view text,Dimensions* result) {
    const auto separator=text.find('x');
    if(separator==std::string_view::npos || separator==0 || separator+1==text.size())return false;
    auto number=[](std::string_view part,std::uint32_t* value){
        if(part.empty() || part.front()<'1' || part.front()>'9')return false;
        std::uint32_t parsed=0;
        for(char digit:part){
            if(digit<'0'||digit>'9')return false;
            const auto next=static_cast<std::uint32_t>(digit-'0');
            if(parsed>(std::numeric_limits<std::uint32_t>::max()-next)/10)return false;
            parsed=parsed*10+next;
        }
        *value=parsed;return true;
    };
    Dimensions parsed;
    if(!number(text.substr(0,separator),&parsed.width) || !number(text.substr(separator+1),&parsed.height))return false;
    *result=parsed;return true;
}
inline bool RenderFitsOutput(Dimensions render,Dimensions output) {
    if(!render.width||!render.height||!output.width||!output.height||
       render.width>output.width||render.height>output.height)return false;
    const std::uint64_t inputAspect=std::uint64_t(render.width)*output.height;
    const std::uint64_t outputAspect=std::uint64_t(output.width)*render.height;
    const std::uint64_t difference=inputAspect>outputAspect?inputAspect-outputAspect:outputAspect-inputAspect;
    return static_cast<double>(difference)<=static_cast<double>(outputAspect)*0.01;
}
constexpr double kPlaneCyclesU=12.0,kPlaneCyclesV=7.0,kPlaneVelocityU=1.0/1280.0;
struct PlanePoint { double u,v; };
inline PlanePoint PlaneCoordinates(double x,double y,double jitterX,double jitterY,
                                    Dimensions render,std::uint64_t frame) {
    return {(x+0.5+jitterX)/render.width-static_cast<double>(frame)*kPlaneVelocityU,
            (y+0.5+jitterY)/render.height};
}
inline std::array<float,4> PlaneColor(double x,double y,double jitterX,double jitterY,
                                     Dimensions render,std::uint64_t frame) {
    const auto point=PlaneCoordinates(x,y,jitterX,jitterY,render,frame);
    constexpr double tau=6.28318530717958647692;
    const float wave=static_cast<float>(0.5+0.5*std::sin(tau*kPlaneCyclesU*point.u)*std::cos(tau*kPlaneCyclesV*point.v));
    return {wave*2.0f,0.25f+0.7f*wave,1.0f-wave*0.8f,1.0f};
}
#ifdef __OBJC__
// Persisting private output requires a JSON boolean, never a truthy number/string/null.
inline bool ReadOptionalBoolean(id value,bool fallback,bool* result) {
    if(!value){*result=fallback;return true;}
    if(CFGetTypeID((__bridge CFTypeRef)value)!=CFBooleanGetTypeID())return false;
    *result=[value boolValue];return true;
}
#endif
enum class StopReason { None, Cancelled, Inactive, CriticalThermal, Deadline };
// Evaluate only between completed frames: stopping must never free an in-flight GPU workload.
inline StopReason StopBeforeNextFrame(bool cancelled,bool active,bool criticalThermal,
                                      double elapsedMS,double deadlineMS) {
    if(cancelled)return StopReason::Cancelled;
    if(!active)return StopReason::Inactive;
    if(criticalThermal)return StopReason::CriticalThermal;
    if(!std::isfinite(elapsedMS)||!std::isfinite(deadlineMS)||deadlineMS<=0||elapsedMS>=deadlineMS)
        return StopReason::Deadline;
    return StopReason::None;
}
// A missing/invalid driver timestamp is unavailable, never a zero-cost inference.
inline double GPUMilliseconds(double start, double end, bool completed, bool error) {
    if (!completed || error || !std::isfinite(start) || !std::isfinite(end) ||
        start <= 0.0 || end <= start) return std::numeric_limits<double>::quiet_NaN();
    const double result = (end - start) * 1000.0;
    return std::isfinite(result) && result > 0.0 ? result : std::numeric_limits<double>::quiet_NaN();
}

// Nearest-rank percentiles. Reject invalid samples instead of silently dropping slow/failing frames.
inline double Percentile(std::vector<double> samples, double fraction) {
    if (samples.empty() || !std::isfinite(fraction) || fraction <= 0 || fraction > 1)
        return std::numeric_limits<double>::quiet_NaN();
    for (double sample : samples)
        if (!std::isfinite(sample) || sample < 0) return std::numeric_limits<double>::quiet_NaN();
    std::sort(samples.begin(), samples.end());
    const auto index = static_cast<std::size_t>(std::ceil(samples.size() * fraction)) - 1;
    return samples[index];
}
}  // namespace netal_probe
