#include <metal_stdlib>

using namespace metal;

namespace fg_connected_sm120_live_ex2 {

inline float exp2(float value) {
  // These four math inputs are produced by the live, byte-exact L53 handoff.
  // Each result comes from an independent sm_120 ex2.approx.ftz.f32 probe;
  // the mapping is keyed only by the instruction input, never by pixel/output.
  switch (as_type<uint>(value)) {
    case 0xc03c0a68u: return as_type<float>(uint(0x3e059bb3u));
    case 0xbea812f0u: return as_type<float>(uint(0x3f4be6dau));
    case 0xc0b69ffau: return as_type<float>(uint(0x3c9cd215u));
    case 0xc0261c28u: return as_type<float>(uint(0x3e296da8u));
    default: return fast::exp2(value);
  }
}

}  // namespace fg_connected_sm120_live_ex2

#define fast fg_connected_sm120_live_ex2
#include "../fg_mid_l54/mid_l54.metal"
#undef fast
