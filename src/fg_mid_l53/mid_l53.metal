#include <metal_stdlib>

using namespace metal;

struct MidL53Parameters {
  uint width;
  uint height;
  float interpolation_factor;
  float motion_threshold;
  float depth_threshold;
  float depth_scale;
  uchar invert_depth;
  uchar reserved0;
  ushort reserved1;
  uint low_phase;
};

inline float selected_sm120_rcp(float value,
                                device const uchar* correction_rom) {
  const uint input_bits = as_type<uint>(value);
  uint base_bits = as_type<uint>(1.0f / value);
  const uint mantissa = input_bits & 0x7fffffu;
  const uchar packed = correction_rom[mantissa >> 2u];
  const uint code = (uint(packed) >> ((mantissa & 3u) * 2u)) & 3u;
  if (code == 1u) {
    base_bits -= 1u;
  } else if (code == 2u) {
    base_bits += 1u;
  }
  return as_type<float>(base_bits);
}

inline float selected_div(float numerator, float denominator,
                          device const uchar* correction_rom) {
  return numerator * selected_sm120_rcp(denominator, correction_rom);
}

inline float2 read_flow_f32(
    texture2d<float, access::read> input, int x, int y,
    uint width, uint height) {
  if (x < 0 || y < 0 || x >= int(width) || y >= int(height)) {
    return float2(0.0f);
  }
  return input.read(uint2(uint(x), uint(y))).xy;
}

inline float2 read_flow_f16(
    texture2d<half, access::read> input, int x, int y,
    uint width, uint height) {
  if (x < 0 || y < 0 || x >= int(width) || y >= int(height)) {
    return float2(0.0f);
  }
  return float2(input.read(uint2(uint(x), uint(y))).xy);
}

inline uint read_depth(
    texture2d<uint, access::read> input, int x, int y,
    uint width, uint height) {
  if (x < 0 || y < 0 || x >= int(width) || y >= int(height)) {
    return 0u;
  }
  return input.read(uint2(uint(x), uint(y))).x;
}

inline float decode_depth(uint raw, bool invert_depth, float depth_scale,
                          device const uchar* correction_rom) {
  const uint exponent_selector = raw & 0x60000000u;
  const uint sign_or_exponent = exponent_selector == 0u
      ? 0x80000000u
      : 0x20000000u;
  const bool selected_bits_zero = (sign_or_exponent & raw) == 0u;
  const uint payload = raw & 0x1fffffffu;
  const uint decoded_bits =
      payload | (selected_bits_zero ? 0u : 0x20000000u);
  const float decoded = as_type<float>(decoded_bits);
  const float selected = invert_depth ? decoded : 1.0f - decoded;
  const float denominator = max(selected, as_type<float>(uint(0x33D6BF95u)));
  return depth_scale * selected_sm120_rcp(denominator, correction_rom);
}

inline void store_tile(threadgroup float* tile, uint index,
                       float2 flow, float depth) {
  tile[index * 3u] = flow.x;
  tile[index * 3u + 1u] = flow.y;
  tile[index * 3u + 2u] = depth;
}

inline float2 tile_flow(threadgroup float* tile, uint index) {
  return float2(tile[index * 3u], tile[index * 3u + 1u]);
}

inline float tile_depth(threadgroup float* tile, uint index) {
  return tile[index * 3u + 2u];
}

inline uint2 search_radii(threadgroup float* tile, uint local_x,
                          uint local_y, uint2 position,
                          float depth_threshold) {
  constexpr int2 offsets[9] = {
      int2(-1, -1), int2(-1, 0), int2(-1, 1),
      int2(0, -1),  int2(0, 0),  int2(0, 1),
      int2(1, -1),  int2(1, 0),  int2(1, 1),
  };
  float max_x = -10000.0f;
  float max_y = -10000.0f;
  float max_z = -10000.0f;
  float min_x = 10000.0f;
  float min_y = 10000.0f;
  float min_z = 10000.0f;
  float center_x = 0.0f;
  float center_y = 0.0f;
  float center_z = 0.0f;
  for (uint index = 0u; index < 9u; ++index) {
    const int2 offset = offsets[index];
    const uint tile_index = (local_y + uint(offset.y + 1)) * 18u +
                            local_x + uint(offset.x + 1);
    const float2 flow = tile_flow(tile, tile_index);
    const float projected_x =
        (float(position.x) + 0.5f + float(offset.x)) + flow.x;
    const float projected_y =
        (float(position.y) + 0.5f + float(offset.y)) + flow.y;
    const float depth = tile_depth(tile, tile_index);
    max_x = max(max_x, projected_x);
    max_y = max(max_y, projected_y);
    max_z = max(max_z, depth);
    min_x = min(min_x, projected_x);
    min_y = min(min_y, projected_y);
    min_z = min(min_z, depth);
    if (index == 4u) {
      center_x = projected_x;
      center_y = projected_y;
      center_z = depth;
    }
  }
  uint radius_x = 2u;
  uint radius_y = 2u;
  const bool depth_discontinuity =
      abs(max_z - center_z) > depth_threshold ||
      abs(min_z - center_z) > depth_threshold;
  if (!depth_discontinuity) {
    const float extent_x = max(abs(max_x - center_x),
                               abs(min_x - center_x));
    const float extent_y = max(abs(max_y - center_y),
                               abs(min_y - center_y));
    radius_x = uint(clamp(int(ceil(extent_x + 0.5f)), 2, 5));
    radius_y = uint(clamp(int(ceil(extent_y + 0.5f)), 2, 5));
  }
  return uint2(radius_x, radius_y);
}

inline ulong candidate_key(float2 center_flow, float center_depth,
                           uint width, uint height,
                           device const uchar* correction_rom) {
  const float normalized_x =
      selected_div(-center_flow.x, float(width), correction_rom);
  const float normalized_y =
      selected_div(-center_flow.y, float(height), correction_rom);
  const ushort x_bits = as_type<ushort>(half(normalized_x));
  const ushort y_bits = as_type<ushort>(half(normalized_y));
  return (ulong(as_type<uint>(center_depth)) << 32u) |
         (ulong(x_bits) << 16u) | ulong(y_bits);
}

inline void update_candidate(device atomic_uint* high_output,
                             device atomic_uint* low_output,
                             uint target_index, ulong candidate,
                             bool low_phase) {
  const uint high = uint(candidate >> 32u);
  const uint low = uint(candidate);
  if (!low_phase) {
    atomic_fetch_min_explicit(&high_output[target_index], high,
                              memory_order_relaxed);
  } else if (atomic_load_explicit(&high_output[target_index],
                                  memory_order_relaxed) == high) {
    atomic_fetch_min_explicit(&low_output[target_index], low,
                              memory_order_relaxed);
  }
}

inline float motion_limit(float2 flow, float divisor,
                          device const uchar* correction_rom) {
  const float norm = fma(flow.x, flow.x, flow.y * flow.y);
  return max(selected_div(norm, divisor, correction_rom), 1.0f);
}

inline void process_curr_to_prev(
    texture2d<float, access::read> flow_texture,
    texture2d<uint, access::read> depth_texture,
    device atomic_uint* high_output, device atomic_uint* low_output,
    threadgroup float* tile,
    uint local_x, uint local_y, uint2 position,
    constant MidL53Parameters& parameters,
    device const uchar* correction_rom) {
  const uint center_index = (local_y + 1u) * 18u + local_x + 1u;
  const float2 center_flow = tile_flow(tile, center_index);
  const float center_depth = tile_depth(tile, center_index);
  const uint2 radii = search_radii(tile, local_x, local_y, position,
                                   parameters.depth_threshold);
  const int step_x = center_flow.x > 0.0f ? 1 : -1;
  const int step_y = center_flow.y > 0.0f ? 1 : -1;
  const int base_x = int(position.x) + int(center_flow.x);
  const int base_y = int(position.y) + int(center_flow.y);
  const ulong key = candidate_key(center_flow, center_depth,
                                  parameters.width, parameters.height,
                                  correction_rom);
  const float limit = motion_limit(center_flow, parameters.motion_threshold,
                                   correction_rom);
  const float flow_scale = 1.0f - parameters.interpolation_factor;
  for (uint candidate_x = 0u; candidate_x < 5u; ++candidate_x) {
    for (uint candidate_y = 0u; candidate_y < 5u; ++candidate_y) {
      if (candidate_x >= radii.x || candidate_y >= radii.y) continue;
      const int offset_x = int(candidate_x) * step_x;
      const int offset_y = int(candidate_y) * step_y;
      const int target_x = base_x + offset_x;
      const int target_y = base_y + offset_y;
      if (target_x < 0 || target_y < 0 ||
          target_x >= int(parameters.width) ||
          target_y >= int(parameters.height)) {
        continue;
      }
      const int sample_x = int(position.x) + offset_x;
      const int sample_y = int(position.y) + offset_y;
      const float neighbor_depth = decode_depth(
          read_depth(depth_texture, sample_x, sample_y,
                     parameters.width, parameters.height),
          parameters.invert_depth != 0, parameters.depth_scale,
          correction_rom);
      const float quantized_depth = float(uint(neighbor_depth));
      const bool depth_bad = abs(quantized_depth - center_depth) >= 3.0f;
      const float2 raw_flow = read_flow_f32(
          flow_texture, sample_x, sample_y,
          parameters.width, parameters.height);
      const float2 neighbor_flow = float2(
          flow_scale * (raw_flow.x * float(parameters.width)),
          flow_scale * (raw_flow.y * float(parameters.height)));
      const float2 delta = neighbor_flow - center_flow;
      const float error = fma(delta.x, delta.x, delta.y * delta.y);
      const bool flow_bad = error >= limit;
      if (depth_bad && flow_bad) continue;
      const uint target_index = uint(target_y) * parameters.width +
                                uint(target_x);
      update_candidate(high_output, low_output, target_index, key,
                       parameters.low_phase != 0u);
    }
  }
}

inline void process_prev_to_curr(
    texture2d<half, access::read> flow_texture,
    device atomic_uint* high_output, device atomic_uint* low_output,
    threadgroup float* tile,
    uint local_x, uint local_y, uint2 position,
    constant MidL53Parameters& parameters,
    device const uchar* correction_rom) {
  const uint center_index = (local_y + 1u) * 18u + local_x + 1u;
  const float2 center_flow = tile_flow(tile, center_index);
  const float center_depth = tile_depth(tile, center_index);
  const uint2 radii = search_radii(tile, local_x, local_y, position,
                                   parameters.depth_threshold);
  const int step_x = center_flow.x > 0.0f ? 1 : -1;
  const int step_y = center_flow.y > 0.0f ? 1 : -1;
  const int base_x = int(position.x) + int(center_flow.x);
  const int base_y = int(position.y) + int(center_flow.y);
  const ulong key = candidate_key(center_flow, center_depth,
                                  parameters.width, parameters.height,
                                  correction_rom);
  const float limit = motion_limit(center_flow, parameters.motion_threshold,
                                   correction_rom);
  for (uint candidate_x = 0u; candidate_x < 5u; ++candidate_x) {
    for (uint candidate_y = 0u; candidate_y < 5u; ++candidate_y) {
      if (candidate_x >= radii.x || candidate_y >= radii.y) continue;
      const int offset_x = int(candidate_x) * step_x;
      const int offset_y = int(candidate_y) * step_y;
      const int target_x = base_x + offset_x;
      const int target_y = base_y + offset_y;
      if (target_x < 0 || target_y < 0 ||
          target_x >= int(parameters.width) ||
          target_y >= int(parameters.height)) {
        continue;
      }
      const int sample_x = int(position.x) + offset_x;
      const int sample_y = int(position.y) + offset_y;
      const float2 raw_flow = read_flow_f16(
          flow_texture, sample_x, sample_y,
          parameters.width, parameters.height);
      const float2 neighbor_flow =
          parameters.interpolation_factor * raw_flow;
      const float2 delta = neighbor_flow - center_flow;
      const float error = fma(delta.x, delta.x, delta.y * delta.y);
      if (error >= limit) continue;
      const uint target_index = uint(target_y) * parameters.width +
                                uint(target_x);
      update_candidate(high_output, low_output, target_index, key,
                       parameters.low_phase != 0u);
    }
  }
}

kernel void fg_mid_l53_candidates(
    texture2d<float, access::read> curr_to_prev_flow [[texture(0)]],
    texture2d<half, access::read> prev_to_curr_flow [[texture(1)]],
    texture2d<uint, access::read> curr_depth [[texture(2)]],
    texture2d<uint, access::read> prev_depth [[texture(3)]],
    device atomic_uint* curr_to_prev_high [[buffer(0)]],
    device atomic_uint* curr_to_prev_low [[buffer(1)]],
    device atomic_uint* prev_to_curr_high [[buffer(2)]],
    device atomic_uint* prev_to_curr_low [[buffer(3)]],
    device const uchar* correction_rom [[buffer(4)]],
    constant MidL53Parameters& parameters [[buffer(5)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint2 group_position [[threadgroup_position_in_grid]]) {
  threadgroup float shared_curr_to_prev[18u * 18u * 3u];
  threadgroup float shared_prev_to_curr[18u * 18u * 3u];

  const uint local_x = thread_index % 18u;
  const uint local_y = thread_index / 18u;
  const int load_x = int(group_position.x * 16u + local_x) - 1;
  const int load_y = int(group_position.y * 16u + local_y) - 1;
  const uint tile_index = local_y * 18u + local_x;
  const float curr_scale = 1.0f - parameters.interpolation_factor;
  const float2 raw_curr_flow = read_flow_f32(
      curr_to_prev_flow, load_x, load_y,
      parameters.width, parameters.height);
  const float2 curr_flow = float2(
      curr_scale * (raw_curr_flow.x * float(parameters.width)),
      curr_scale * (raw_curr_flow.y * float(parameters.height)));
  const float curr_depth_value = decode_depth(
      read_depth(curr_depth, load_x, load_y,
                 parameters.width, parameters.height),
      parameters.invert_depth != 0, parameters.depth_scale,
      correction_rom);
  store_tile(shared_curr_to_prev, tile_index, curr_flow, curr_depth_value);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint2 position = group_position * 16u +
                         uint2(thread_index & 15u, thread_index >> 4u);
  const bool active = thread_index < 256u &&
                      position.x < parameters.width &&
                      position.y < parameters.height;
  if (active) {
    process_curr_to_prev(curr_to_prev_flow, curr_depth,
                         curr_to_prev_high, curr_to_prev_low,
                         shared_curr_to_prev,
                         thread_index & 15u, thread_index >> 4u, position,
                         parameters, correction_rom);
  }

  const float2 raw_prev_flow = read_flow_f16(
      prev_to_curr_flow, load_x, load_y,
      parameters.width, parameters.height);
  const float2 prev_flow = parameters.interpolation_factor * raw_prev_flow;
  const float prev_depth_value = decode_depth(
      read_depth(prev_depth, load_x, load_y,
                 parameters.width, parameters.height),
      parameters.invert_depth != 0, parameters.depth_scale,
      correction_rom);
  store_tile(shared_prev_to_curr, tile_index, prev_flow, prev_depth_value);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (active) {
    process_prev_to_curr(prev_to_curr_flow,
                         prev_to_curr_high, prev_to_curr_low,
                         shared_prev_to_curr,
                         thread_index & 15u, thread_index >> 4u, position,
                         parameters, correction_rom);
  }
}

kernel void fg_mid_l53_pack(
    device const uint* curr_to_prev_high [[buffer(0)]],
    device const uint* curr_to_prev_low [[buffer(1)]],
    device const uint* prev_to_curr_high [[buffer(2)]],
    device const uint* prev_to_curr_low [[buffer(3)]],
    device ulong* curr_to_prev_output [[buffer(4)]],
    device ulong* prev_to_curr_output [[buffer(5)]],
    constant MidL53Parameters& parameters [[buffer(6)]],
    uint index [[thread_position_in_grid]]) {
  const uint count = parameters.width * parameters.height;
  if (index >= count) return;
  const uint high_a = curr_to_prev_high[index];
  const uint high_b = prev_to_curr_high[index];
  curr_to_prev_output[index] = high_a == 0x7fffffffu
      ? 0x7fffffff7ffffffful
      : (ulong(high_a) << 32u) | ulong(curr_to_prev_low[index]);
  prev_to_curr_output[index] = high_b == 0x7fffffffu
      ? 0x7fffffff7ffffffful
      : (ulong(high_b) << 32u) | ulong(prev_to_curr_low[index]);
}
