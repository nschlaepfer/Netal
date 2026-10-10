#include <metal_stdlib>

using namespace metal;

struct EarlyL5L15Parameters {
  uint width;
  uint height;
  float motion_divisor;
  float neighbor_divisor;
  float depth_scale;
  uint invert_depth;
};

inline float selected_sm120_rcp(float value,
                                device const uchar* correction_rom) {
  const uint input_bits = as_type<uint>(value);
  uint base_bits = as_type<uint>(1.0f / value);
  const uint mantissa = input_bits & 0x7fffffu;
  const uchar packed = correction_rom[mantissa >> 2u];
  const uint code = (uint(packed) >> ((mantissa & 3u) * 2u)) & 3u;
  if (code == 1u) base_bits -= 1u;
  else if (code == 2u) base_bits += 1u;
  return as_type<float>(base_bits);
}

inline float selected_div(float numerator, float denominator,
                          device const uchar* correction_rom) {
  return numerator * selected_sm120_rcp(denominator, correction_rom);
}

inline float2 normalized_point_sample(
    texture2d<float, access::read> input, float2 coordinate,
    uint2 dimensions) {
  const float2 saturated = saturate(coordinate);
  const float2 fixed12 = floor(saturated * 4096.0f) * (1.0f / 4096.0f);
  int2 location = int2(floor(fixed12 * float2(dimensions)));
  location = clamp(location, int2(0), int2(dimensions) - 1);
  return input.read(uint2(location)).xy;
}

inline float2 integer_flow(texture2d<float, access::read> input,
                           int2 location, uint2 dimensions) {
  if (any(location < int2(0)) || any(location >= int2(dimensions))) {
    return float2(0.0f);
  }
  return input.read(uint2(location)).xy;
}

inline float decode_depth(uint raw, bool invert_depth, float depth_scale,
                          device const uchar* correction_rom) {
  const uint selector = raw & 0x60000000u;
  const uint selected_bit = selector == 0u ? 0x80000000u : 0x20000000u;
  const bool selected_zero = (selected_bit & raw) == 0u;
  const uint decoded_bits =
      (raw & 0x1fffffffu) | (selected_zero ? 0u : 0x20000000u);
  const float decoded = as_type<float>(decoded_bits);
  const float selected = invert_depth ? decoded : 1.0f - decoded;
  const float denominator = max(selected, as_type<float>(uint(0x33D6BF95u)));
  return depth_scale * selected_sm120_rcp(denominator, correction_rom);
}

inline ulong candidate_key(float2 pixel_flow, float depth_priority,
                           bool mask_valid) {
  const ushort x_bits = as_type<ushort>(half(pixel_flow.x));
  const ushort y_bits = as_type<ushort>(half(pixel_flow.y));
  const uint high = (as_type<uint>(depth_priority) & 0xfffffffeu) |
                    (mask_valid ? 1u : 0u);
  return (ulong(high) << 32u) | (ulong(x_bits) << 16u) | ulong(y_bits);
}

inline void atomic_candidate(device atomic_uint* ordered_high,
                             device atomic_uint* low,
                             uint target_index, ulong candidate,
                             bool low_phase) {
  const uint signed_high = uint(candidate >> 32u);
  const uint transformed_high = signed_high ^ 0x80000000u;
  const uint candidate_low = uint(candidate);
  if (!low_phase) {
    atomic_fetch_min_explicit(&ordered_high[target_index], transformed_high,
                              memory_order_relaxed);
  } else if (atomic_load_explicit(&ordered_high[target_index],
                                  memory_order_relaxed) == transformed_high) {
    atomic_fetch_min_explicit(&low[target_index], candidate_low,
                              memory_order_relaxed);
  }
}

inline void process_target(
    texture2d<float, access::read> flow_texture,
    device atomic_uint* ordered_high, device atomic_uint* low,
    int2 source, int2 target, float2 center_flow, float center_norm,
    ulong candidate, bool low_phase, constant EarlyL5L15Parameters& parameters,
    device const uchar* correction_rom) {
  const int2 dimensions(int(parameters.width), int(parameters.height));
  if (any(target < int2(0)) || any(target >= dimensions) ||
      any(source < int2(0)) || any(source >= dimensions)) {
    return;
  }
  const float2 neighbor =
      integer_flow(flow_texture, source, uint2(parameters.width,
                                               parameters.height)) *
      float2(parameters.width, parameters.height);
  const float2 delta = neighbor - center_flow;
  const float error = fma(delta.x, delta.x, delta.y * delta.y);
  const float limit = max(
      selected_div(center_norm, parameters.neighbor_divisor, correction_rom),
      1.0f);
  if (error >= limit) return;
  const uint target_index =
      uint(target.y) * parameters.width + uint(target.x);
  atomic_candidate(ordered_high, low, target_index, candidate, low_phase);
}

inline void run_selected(
    texture2d<float, access::read> flow_texture,
    texture2d<uint, access::read> depth_texture,
    texture2d<half, access::write> mask,
    device atomic_uint* ordered_high, device atomic_uint* low,
    device const uchar* correction_rom,
    constant EarlyL5L15Parameters& parameters,
    uint2 position, bool low_phase) {
  if (position.x >= parameters.width || position.y >= parameters.height) return;
  const uint2 dimensions(parameters.width, parameters.height);
  const float2 uv = saturate(float2(
      selected_div(float(position.x) + 0.5f, float(parameters.width),
                   correction_rom),
      selected_div(float(position.y) + 0.5f, float(parameters.height),
                   correction_rom)));
  const float2 sampled_flow = normalized_point_sample(
      flow_texture, uv, dimensions);
  const float2 pixel_flow =
      sampled_flow * float2(parameters.width, parameters.height);
  const float norm = fma(pixel_flow.x, pixel_flow.x,
                         pixel_flow.y * pixel_flow.y);
  const bool mask_valid =
      norm <= max(selected_div(norm, parameters.motion_divisor,
                               correction_rom), 1.0f);
  if (!low_phase) {
    mask.write(half4(mask_valid ? 0.0h : 1.0h), position);
  }
  const uint depth_raw = depth_texture.read(position).x;
  const float depth_priority = decode_depth(
      depth_raw, parameters.invert_depth != 0u, parameters.depth_scale,
      correction_rom);
  const ulong candidate =
      candidate_key(pixel_flow, depth_priority, mask_valid);
  const int2 base = int2(position) + int2(pixel_flow);
  const int2 step(pixel_flow.x > 0.0f ? 1 : -1,
                  pixel_flow.y > 0.0f ? 1 : -1);
  const int2 source = int2(position);
  process_target(flow_texture, ordered_high, low, source, base,
                 pixel_flow, norm, candidate, low_phase, parameters,
                 correction_rom);
  process_target(flow_texture, ordered_high, low,
                 source + int2(0, step.y), base + int2(0, step.y),
                 pixel_flow, norm, candidate, low_phase, parameters,
                 correction_rom);
  process_target(flow_texture, ordered_high, low,
                 source + int2(step.x, 0), base + int2(step.x, 0),
                 pixel_flow, norm, candidate, low_phase, parameters,
                 correction_rom);
  process_target(flow_texture, ordered_high, low, source + step, base + step,
                 pixel_flow, norm, candidate, low_phase, parameters,
                 correction_rom);
}

kernel void fg_early_l5_l15_high(
    texture2d<float, access::read> flow_texture [[texture(0)]],
    texture2d<uint, access::read> depth_texture [[texture(1)]],
    texture2d<half, access::write> mask [[texture(2)]],
    device atomic_uint* ordered_high [[buffer(0)]],
    device atomic_uint* low [[buffer(1)]],
    device const uchar* correction_rom [[buffer(2)]],
    constant EarlyL5L15Parameters& parameters [[buffer(3)]],
    uint2 position [[thread_position_in_grid]]) {
  run_selected(flow_texture, depth_texture, mask, ordered_high, low,
               correction_rom, parameters, position, false);
}

kernel void fg_early_l5_l15_low(
    texture2d<float, access::read> flow_texture [[texture(0)]],
    texture2d<uint, access::read> depth_texture [[texture(1)]],
    texture2d<half, access::write> mask [[texture(2)]],
    device atomic_uint* ordered_high [[buffer(0)]],
    device atomic_uint* low [[buffer(1)]],
    device const uchar* correction_rom [[buffer(2)]],
    constant EarlyL5L15Parameters& parameters [[buffer(3)]],
    uint2 position [[thread_position_in_grid]]) {
  run_selected(flow_texture, depth_texture, mask, ordered_high, low,
               correction_rom, parameters, position, true);
}

kernel void fg_early_l5_l15_pack(
    device const uint* ordered_high [[buffer(0)]],
    device const uint* low [[buffer(1)]],
    device ulong* output [[buffer(2)]],
    constant EarlyL5L15Parameters& parameters [[buffer(3)]],
    uint position [[thread_position_in_grid]]) {
  const uint count = parameters.width * parameters.height;
  if (position >= count) return;
  const uint high = ordered_high[position];
  const uint low_word = low[position];
  if (high == 0xffffffffu && low_word == 0xffffffffu) {
    output[position] = 0x7fffffff7ffffffful;
  } else {
    output[position] =
        (ulong(high ^ 0x80000000u) << 32u) | ulong(low_word);
  }
}
