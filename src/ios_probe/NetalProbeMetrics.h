#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#ifdef __OBJC__
#import <Foundation/Foundation.h>
#endif

namespace netal_probe {
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
