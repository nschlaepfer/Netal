#include <metal_stdlib>
using namespace metal;

constexpr sampler kFlowLinearSampler(coord::normalized, address::clamp_to_edge, filter::linear);

struct NrFlowPyramidParams {
  uint full_width;
  uint full_height;
  uint half_width;
  uint half_height;
  uint quarter_width;
  uint quarter_height;
  uint has_previous;
  uint force_reset;
  uint num_groups;
  float scene_cut_threshold;
  ulong frame_index;
};

struct NrFlowCoarseParams {
  uint half_width;
  uint half_height;
  uint quarter_width;
  uint quarter_height;
  uint coarse_width;
  uint coarse_height;
  int max_radius;
};

struct NrFlowUpsampleParams {
  uint full_width;
  uint full_height;
  uint half_width;
  uint half_height;
  uint quarter_width;
  uint quarter_height;
  uint coarse_width;
  uint coarse_height;
};

struct NrFlowTelemetrySlot {
  float mean_luma_diff;
  uint scene_cut;
  ulong frame_index;
};

inline float nr_flow_perceptual_luma(float3 rgb) {
  float3 safe_rgb = select(float3(0.0f), max(rgb, float3(0.0f)), isfinite(rgb));
  float y = dot(safe_rgb, float3(0.2126f, 0.7152f, 0.0722f));
  return (y <= 1.0f) ? y : ((2.0f * y) / (1.0f + y));
}

kernel void nr_flow_luma_pyramid(texture2d<float, access::sample> current_color [[texture(0)]],
                                 texture2d<half, access::write> curr_half_luma [[texture(1)]],
                                 texture2d<half, access::write> curr_quarter_luma [[texture(2)]],
                                 texture2d<half, access::read> prev_quarter_luma [[texture(3)]],
                                 device float* partial_diffs [[buffer(0)]],
                                 constant NrFlowPyramidParams& params [[buffer(1)]],
                                 uint2 qxy [[thread_position_in_grid]],
                                 uint2 tid [[thread_position_in_threadgroup]],
                                 uint2 tgid [[threadgroup_position_in_grid]],
                                 uint2 tg_grid [[threadgroups_per_grid]],
                                 uint simd_lane [[thread_index_in_simdgroup]],
                                 uint simd_id [[simdgroup_index_in_threadgroup]]) {
  threadgroup float sg_sums[8];
  float local_diff = 0.0f;

  if (qxy.x < params.quarter_width && qxy.y < params.quarter_height) {
    float2 inv_half = 1.0f / float2(float(params.half_width), float(params.half_height));
    float q_accum = 0.0f;
    float q_weight = 0.0f;
    for (uint iy = 0u; iy < 2u; ++iy) {
      for (uint ix = 0u; ix < 2u; ++ix) {
        uint2 hxy = qxy * 2u + uint2(ix, iy);
        if (hxy.x < params.half_width && hxy.y < params.half_height) {
          float2 uv = (float2(hxy) + 0.5f) * inv_half;
          float luma = nr_flow_perceptual_luma(current_color.sample(kFlowLinearSampler, uv).rgb);
          curr_half_luma.write(half4(half(luma), 0.0h, 0.0h, 1.0h), hxy);
          q_accum += luma;
          q_weight += 1.0f;
        }
      }
    }
    float q_curr = (q_weight > 0.0f) ? (q_accum / q_weight) : 0.0f;
    curr_quarter_luma.write(half4(half(q_curr), 0.0h, 0.0h, 1.0h), qxy);
    if (params.has_previous != 0u) {
      float q_prev = float(prev_quarter_luma.read(qxy).r);
      local_diff = fabs(q_curr - q_prev);
    }
  }

  float sg_total = simd_sum(local_diff);
  if (simd_lane == 0u && simd_id < 8u) {
    sg_sums[simd_id] = sg_total;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid.x == 0u && tid.y == 0u) {
    float block_total = 0.0f;
    for (uint i = 0u; i < 8u; ++i) block_total += sg_sums[i];
    uint group_index = tgid.y * tg_grid.x + tgid.x;
    if (group_index < params.num_groups) {
      partial_diffs[group_index] = block_total;
    }
  }
}

kernel void nr_flow_scene_cut_reduce(device const float* partial_diffs [[buffer(0)]],
                                     device uint* scene_cut_flag [[buffer(1)]],
                                     device NrFlowTelemetrySlot* telemetry [[buffer(2)]],
                                     constant NrFlowPyramidParams& params [[buffer(3)]],
                                     uint tid [[thread_index_in_threadgroup]],
                                     uint simd_lane [[thread_index_in_simdgroup]],
                                     uint simd_id [[simdgroup_index_in_threadgroup]]) {
  threadgroup float sg_sums[8];
  float sum = 0.0f;
  for (uint i = tid; i < params.num_groups; i += 256u) {
    sum += partial_diffs[i];
  }
  float sg_total = simd_sum(sum);
  if (simd_lane == 0u && simd_id < 8u) {
    sg_sums[simd_id] = sg_total;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    float total = 0.0f;
    for (uint i = 0u; i < 8u; ++i) total += sg_sums[i];
    float denom = float(max(1u, params.quarter_width * params.quarter_height));
    float mean_diff = (params.has_previous != 0u) ? (total / denom) : 0.0f;
    uint is_cut = (params.force_reset != 0u || params.has_previous == 0u ||
                   mean_diff > params.scene_cut_threshold) ? 1u : 0u;
    scene_cut_flag[0] = is_cut;
    telemetry[0].mean_luma_diff = mean_diff;
    telemetry[0].scene_cut = is_cut;
    telemetry[0].frame_index = params.frame_index;
  }
}

inline float nr_flow_eval_coarse_patch16(thread const half* curr_patch,
                                         threadgroup const half* prev_sram,
                                         uint prev_stride,
                                         uint p_ox,
                                         uint p_oy,
                                         int dx,
                                         int dy,
                                         thread float& out_sad) {
  float sad = 0.0f;
  uint idx = 0u;
  uint base_row = uint(int(p_oy) + dy);
  uint base_col = uint(int(p_ox) + dx);
  for (uint py = 0u; py < 8u; py += 2u) {
    uint row_off = (base_row + py) * prev_stride + base_col;
    for (uint px = 0u; px < 8u; px += 2u) {
      sad += fabs(float(curr_patch[idx++]) - float(prev_sram[row_off + px]));
    }
  }
  sad *= (1.0f / 16.0f);
  out_sad = sad;
  return sad + 0.00010f * float(abs(dx) + abs(dy));
}

// 16-tap strided 8x8 half-res (= 16x16 full-res) patch SAD sampled on exact half-res texel centers (*.5f).
inline float nr_flow_half_patch_sad(thread const float* patch_curr,
                                    texture2d<half, access::sample> prev_half_luma,
                                    float2 base_hxy,
                                    float2 inv_half,
                                    float2 flow_half) {
  float sad = 0.0f;
  int k = 0;
  for (int py = -2; py <= 4; py += 2) {
    for (int px = -2; px <= 4; px += 2) {
      float2 puv = (base_hxy + float2(float(px), float(py)) + flow_half) * inv_half;
      sad += fabs(patch_curr[k++] - float(prev_half_luma.sample(kFlowLinearSampler, puv).r));
    }
  }
  return sad * (1.0f / 16.0f);
}

kernel void nr_flow_coarse_search(texture2d<half, access::read> curr_quarter_luma [[texture(0)]],
                                  texture2d<half, access::read> prev_quarter_luma [[texture(1)]],
                                  texture2d<half, access::sample> curr_half_luma [[texture(2)]],
                                  texture2d<half, access::sample> prev_half_luma [[texture(3)]],
                                  texture2d<half, access::write> coarse_flow [[texture(4)]],
                                  device const uint* scene_cut_flag [[buffer(0)]],
                                  constant NrFlowCoarseParams& params [[buffer(1)]],
                                  uint2 tile [[thread_position_in_grid]],
                                  uint2 tid [[thread_position_in_threadgroup]],
                                  uint2 tgid [[threadgroup_position_in_grid]]) {
  if (scene_cut_flag[0] != 0u) {
    if (tile.x < params.coarse_width && tile.y < params.coarse_height) {
      coarse_flow.write(half4(0.0h, 0.0h, 0.0h, 1.0h), tile);
    }
    return;
  }

  constexpr int kMaxRadius = 24;
  constexpr uint kCurrDim = 38u;
  constexpr uint kCurrStride = 40u;
  constexpr uint kPrevDim = 86u;
  constexpr uint kPrevStride = 88u;
  threadgroup half curr_sram[kCurrDim * kCurrStride];
  threadgroup half prev_sram[kPrevDim * kPrevStride];

  const uint linear_tid = tid.y * 8u + tid.x;
  const int2 base_q = int2(tgid) * 32;
  const int2 q_limit = int2(int(params.quarter_width) - 1, int(params.quarter_height) - 1);

  for (uint idx = linear_tid; idx < kCurrDim * kCurrDim; idx += 64u) {
    uint ly = idx / kCurrDim;
    uint lx = idx % kCurrDim;
    int2 qpos = clamp(base_q + int2(int(lx) - 2, int(ly) - 2), int2(0), q_limit);
    curr_sram[ly * kCurrStride + lx] = curr_quarter_luma.read(uint2(qpos)).r;
  }
  for (uint idx = linear_tid; idx < kPrevDim * kPrevDim; idx += 64u) {
    uint ly = idx / kPrevDim;
    uint lx = idx % kPrevDim;
    int2 qpos = clamp(base_q + int2(int(lx) - 2 - kMaxRadius, int(ly) - 2 - kMaxRadius), int2(0), q_limit);
    prev_sram[ly * kPrevStride + lx] = prev_quarter_luma.read(uint2(qpos)).r;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tile.x >= params.coarse_width || tile.y >= params.coarse_height) return;

  const int radius = clamp(params.max_radius, 4, kMaxRadius);
  const uint c_ox = tid.x * 4u;
  const uint c_oy = tid.y * 4u;
  const uint p_ox = tid.x * 4u + uint(kMaxRadius);
  const uint p_oy = tid.y * 4u + uint(kMaxRadius);

  half curr_patch[16];
  uint sample_idx = 0u;
  for (uint py = 0u; py < 8u; py += 2u) {
    for (uint px = 0u; px < 8u; px += 2u) {
      curr_patch[sample_idx++] = curr_sram[(c_oy + py) * kCurrStride + (c_ox + px)];
    }
  }

  float best_sad = 0.0f;
  float best_cost = nr_flow_eval_coarse_patch16(curr_patch, prev_sram, kPrevStride, p_ox, p_oy, 0, 0, best_sad);
  int2 best_vec = int2(0, 0);

  for (int dy = -3; dy <= 3; ++dy) {
    for (int dx = -3; dx <= 3; ++dx) {
      if (dx == 0 && dy == 0) continue;
      float sad = 0.0f;
      float cost = nr_flow_eval_coarse_patch16(curr_patch, prev_sram, kPrevStride, p_ox, p_oy, dx, dy, sad);
      if (cost < best_cost) {
        best_cost = cost;
        best_sad = sad;
        best_vec = int2(dx, dy);
      }
    }
  }

  int grid_limit = (radius / 4) * 4;
  int2 wide_vec = best_vec;
  float wide_cost = best_cost;
  float wide_sad = best_sad;
  for (int dy = -grid_limit; dy <= grid_limit; dy += 4) {
    for (int dx = -grid_limit; dx <= grid_limit; dx += 4) {
      if (abs(dx) <= 3 && abs(dy) <= 3) continue;
      float sad = 0.0f;
      float cost = nr_flow_eval_coarse_patch16(curr_patch, prev_sram, kPrevStride, p_ox, p_oy, dx, dy, sad);
      if (cost < wide_cost) {
        wide_cost = cost;
        wide_sad = sad;
        wide_vec = int2(dx, dy);
      }
    }
  }
  if (wide_cost < best_cost) {
    best_cost = wide_cost;
    best_sad = wide_sad;
    best_vec = wide_vec;
  }

  for (int step = 2; step >= 1; step >>= 1) {
    int2 center = best_vec;
    for (int oy = -1; oy <= 1; ++oy) {
      for (int ox = -1; ox <= 1; ++ox) {
        if (ox == 0 && oy == 0) continue;
        int2 cand = clamp(center + int2(ox * step, oy * step), int2(-radius), int2(radius));
        float sad = 0.0f;
        float cost = nr_flow_eval_coarse_patch16(curr_patch, prev_sram, kPrevStride, p_ox, p_oy, cand.x, cand.y, sad);
        if (cost < best_cost) {
          best_cost = cost;
          best_sad = sad;
          best_vec = cand;
        }
      }
    }
  }

  // Polish at half-res inside the tiny 8,160-thread coarse grid (eliminates non-multiple-of-4
  // quarter-res box-alignment bias before upsampling!). Store flow in half-res pixel units.
  const float2 inv_half = 1.0f / float2(float(params.half_width), float(params.half_height));
  const uint2 center_hp = min(tile * 8u + 3u, uint2(params.half_width - 1u, params.half_height - 1u));
  const float2 base_hxy = float2(center_hp) + 0.5f;
  float hpatch[16];
  int hpi = 0;
  for (int py = -4; py <= 8; py += 4) {
    for (int px = -4; px <= 8; px += 4) {
      hpatch[hpi++] = float(curr_half_luma.sample(kFlowLinearSampler, (base_hxy + float2(float(px), float(py))) * inv_half).r);
    }
  }

  float2 best_half = float2(best_vec) * 2.0f;
  float half_best_sad = 1e9f;
  float half_best_cost = 1e9f;
  float2 seed_half = best_half;
  for (int oy = -3; oy <= 3; ++oy) {
    for (int ox = -3; ox <= 3; ++ox) {
      float2 cand = seed_half + float2(float(ox), float(oy));
      float sad = 0.0f;
      int k = 0;
      for (int py = -4; py <= 8; py += 4) {
        for (int px = -4; px <= 8; px += 4) {
          float2 puv = (base_hxy + float2(float(px), float(py)) + cand) * inv_half;
          sad += fabs(hpatch[k++] - float(prev_half_luma.sample(kFlowLinearSampler, puv).r));
        }
      }
      sad *= (1.0f / 16.0f);
      float cost = sad + 0.00008f * (fabs(cand.x) + fabs(cand.y));
      if (cost < half_best_cost) {
        half_best_cost = cost;
        half_best_sad = sad;
        best_half = cand;
      }
    }
  }

  float confidence = clamp(1.0f - half_best_sad * 6.0f, 0.0f, 1.0f);
  coarse_flow.write(half4(half(best_half.x), half(best_half.y), half(confidence), half(half_best_sad)), tile);
}

inline void nr_flow_sort2(thread float& a, thread float& b) {
  float lo = min(a, b);
  float hi = max(a, b);
  a = lo;
  b = hi;
}

inline float nr_flow_median9(thread float v[9]) {
  nr_flow_sort2(v[1], v[2]); nr_flow_sort2(v[4], v[5]); nr_flow_sort2(v[7], v[8]);
  nr_flow_sort2(v[0], v[1]); nr_flow_sort2(v[3], v[4]); nr_flow_sort2(v[6], v[7]);
  nr_flow_sort2(v[1], v[2]); nr_flow_sort2(v[4], v[5]); nr_flow_sort2(v[7], v[8]);
  nr_flow_sort2(v[0], v[3]); nr_flow_sort2(v[5], v[8]); nr_flow_sort2(v[4], v[7]);
  nr_flow_sort2(v[3], v[6]); nr_flow_sort2(v[1], v[4]); nr_flow_sort2(v[2], v[5]);
  nr_flow_sort2(v[4], v[7]); nr_flow_sort2(v[4], v[2]); nr_flow_sort2(v[6], v[4]);
  nr_flow_sort2(v[4], v[2]);
  return v[4];
}

kernel void nr_flow_refine_upsample(texture2d<float, access::sample> current_color [[texture(0)]],
                                    texture2d<half, access::sample> curr_half_luma [[texture(1)]],
                                    texture2d<half, access::sample> prev_half_luma [[texture(2)]],
                                    texture2d<half, access::read> coarse_flow [[texture(3)]],
                                    texture2d<half, access::write> out_flow [[texture(4)]],
                                    device const uint* scene_cut_flag [[buffer(0)]],
                                    constant NrFlowUpsampleParams& params [[buffer(1)]],
                                    uint2 qxy [[thread_position_in_grid]],
                                    uint2 tid [[thread_position_in_threadgroup]]) {
  if (scene_cut_flag[0] != 0u) {
    if (qxy.x < params.quarter_width && qxy.y < params.quarter_height) {
      uint2 fbase = qxy * 4u;
      for (uint iy = 0u; iy < 4u; ++iy) {
        for (uint ix = 0u; ix < 4u; ++ix) {
          uint2 fxy = fbase + uint2(ix, iy);
          if (fxy.x < params.full_width && fxy.y < params.full_height) {
            out_flow.write(half4(0.0h, 0.0h, 0.0h, 0.0h), fxy);
          }
        }
      }
    }
    return;
  }

  threadgroup float2 tg_flow[16 * 16];
  threadgroup float tg_luma[16 * 16];

  const float2 inv_half = 1.0f / float2(float(params.half_width), float(params.half_height));
  const int2 c_limit = int2(int(params.coarse_width) - 1, int(params.coarse_height) - 1);

  float my_luma = 0.0f;
  float2 my_flow = float2(0.0f);

  if (qxy.x < params.quarter_width && qxy.y < params.quarter_height) {
    float2 base_hxy = float2(qxy * 2u) + 0.5f;
    my_luma = float(curr_half_luma.sample(kFlowLinearSampler, (base_hxy + 0.5f) * inv_half).r);

    // Gather 3x3 coarse tiles (coarse_flow.xy is already in half-res pixel units!).
    int2 c_center = clamp(int2(qxy) / 4, int2(0), c_limit);
    float mx[9], my[9];
    float2 bilat_sum = float2(0.0f);
    float bilat_w = 0.0f;
    int idx = 0;
    for (int cy = -1; cy <= 1; ++cy) {
      for (int cx = -1; cx <= 1; ++cx) {
        int2 cp = clamp(c_center + int2(cx, cy), int2(0), c_limit);
        half4 cf = coarse_flow.read(uint2(cp));
        float2 v_half = float2(cf.xy);
        mx[idx] = v_half.x;
        my[idx] = v_half.y;
        float spatial_w = (cx == 0 && cy == 0) ? 1.0f : ((cx == 0 || cy == 0) ? 0.65f : 0.4f);
        float conf_w = max(0.05f, float(cf.z));
        float w = spatial_w * conf_w;
        bilat_sum += v_half * w;
        bilat_w += w;
        ++idx;
      }
    }

    float2 med_half = rint(float2(nr_flow_median9(mx), nr_flow_median9(my)));
    float2 bilat_half = rint((bilat_w > 1e-5f) ? (bilat_sum / bilat_w) : med_half);

    float patch_curr[16];
    int pi = 0;
    for (int py = -2; py <= 4; py += 2) {
      for (int px = -2; px <= 4; px += 2) {
        patch_curr[pi++] = float(curr_half_luma.sample(kFlowLinearSampler, (base_hxy + float2(float(px), float(py))) * inv_half).r);
      }
    }

    float best_sad = nr_flow_half_patch_sad(patch_curr, prev_half_luma, base_hxy, inv_half, float2(0.0f));
    float best_cost = best_sad;
    float2 best_v = float2(0.0f);

    float2 seed = med_half;
    if (any(bilat_half != med_half)) {
      float bsad = nr_flow_half_patch_sad(patch_curr, prev_half_luma, base_hxy, inv_half, bilat_half);
      float msad = nr_flow_half_patch_sad(patch_curr, prev_half_luma, base_hxy, inv_half, med_half);
      if (bsad < msad) seed = bilat_half;
    }

    // Tight 3x3 integer refinement at half-res around polished coarse seed (only 9 points!).
    for (int oy = -1; oy <= 1; ++oy) {
      for (int ox = -1; ox <= 1; ++ox) {
        float2 cand = seed + float2(float(ox), float(oy));
        float sad = nr_flow_half_patch_sad(patch_curr, prev_half_luma, base_hxy, inv_half, cand);
        float cost = sad + 0.00008f * (fabs(cand.x) + fabs(cand.y));
        if (cost < best_cost) {
          best_cost = cost;
          best_sad = sad;
          best_v = cand;
        }
      }
    }

    // 3x3 0.5-half-res (= 1 full-res px) search (handles odd translations on both axes!) + parabolic subpixel fit.
    float2 half_step_center = best_v;
    for (int oy = -1; oy <= 1; ++oy) {
      for (int ox = -1; ox <= 1; ++ox) {
        if (ox == 0 && oy == 0) continue;
        float2 cand = half_step_center + 0.5f * float2(float(ox), float(oy));
        float sad = nr_flow_half_patch_sad(patch_curr, prev_half_luma, base_hxy, inv_half, cand);
        float cost = sad + 0.00008f * (fabs(cand.x) + fabs(cand.y));
        if (cost < best_cost) {
          best_cost = cost;
          best_sad = sad;
          best_v = cand;
        }
      }
    }

    float sub_x = 0.0f;
    float sub_y = 0.0f;
    if (best_sad > 1e-5f) {
      const float h = 0.5f;
      float cx_m = nr_flow_half_patch_sad(patch_curr, prev_half_luma, base_hxy, inv_half, best_v + float2(-h, 0.0f));
      float cx_p = nr_flow_half_patch_sad(patch_curr, prev_half_luma, base_hxy, inv_half, best_v + float2( h, 0.0f));
      float cy_m = nr_flow_half_patch_sad(patch_curr, prev_half_luma, base_hxy, inv_half, best_v + float2(0.0f, -h));
      float cy_p = nr_flow_half_patch_sad(patch_curr, prev_half_luma, base_hxy, inv_half, best_v + float2(0.0f,  h));
      float denom_x = cx_m - 2.0f * best_sad + cx_p;
      float denom_y = cy_m - 2.0f * best_sad + cy_p;
      if (denom_x > 1e-4f) sub_x = clamp(0.25f * (cx_m - cx_p) / denom_x, -0.22f, 0.22f);
      if (denom_y > 1e-4f) sub_y = clamp(0.25f * (cy_m - cy_p) / denom_y, -0.22f, 0.22f);
    }

    float2 scale_to_full = float2(float(params.full_width) / float(params.half_width),
                                  float(params.full_height) / float(params.half_height));
    my_flow = (best_v + float2(sub_x, sub_y)) * scale_to_full;
  }

  tg_flow[tid.y * 16u + tid.x] = my_flow;
  tg_luma[tid.y * 16u + tid.x] = my_luma;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (qxy.x >= params.quarter_width || qxy.y >= params.quarter_height) return;

  const float2 inv_full = 1.0f / float2(float(params.full_width), float(params.full_height));
  const uint nx_neg = (tid.x > 0u) ? (tid.x - 1u) : 0u;
  const uint nx_pos = (tid.x < 15u && (qxy.x + 1u) < params.quarter_width) ? (tid.x + 1u) : tid.x;
  const uint ny_neg = (tid.y > 0u) ? (tid.y - 1u) : 0u;
  const uint ny_pos = (tid.y < 15u && (qxy.y + 1u) < params.quarter_height) ? (tid.y + 1u) : tid.y;

  const uint2 fbase = qxy * 4u;
  for (uint iy = 0u; iy < 4u; ++iy) {
    uint ty1 = (iy < 2u) ? ny_neg : ny_pos;
    float wy_spatial = (iy == 0u || iy == 3u) ? 0.35f : 0.12f;
    for (uint ix = 0u; ix < 4u; ++ix) {
      uint2 fxy = fbase + uint2(ix, iy);
      if (fxy.x >= params.full_width || fxy.y >= params.full_height) continue;
      uint tx1 = (ix < 2u) ? nx_neg : nx_pos;
      float wx_spatial = (ix == 0u || ix == 3u) ? 0.35f : 0.12f;

      float2 fuv = (float2(fxy) + 0.5f) * inv_full;
      float fluma = nr_flow_perceptual_luma(current_color.sample(kFlowLinearSampler, fuv).rgb);

      uint idx00 = tid.y * 16u + tid.x;
      uint idx10 = tid.y * 16u + tx1;
      uint idx01 = ty1 * 16u + tid.x;
      uint idx11 = ty1 * 16u + tx1;

      float w00 = exp(-fabs(fluma - tg_luma[idx00]) * 14.0f) + 1e-4f;
      float w10 = wx_spatial * exp(-fabs(fluma - tg_luma[idx10]) * 14.0f);
      float w01 = wy_spatial * exp(-fabs(fluma - tg_luma[idx01]) * 14.0f);
      float w11 = (wx_spatial * wy_spatial) * exp(-fabs(fluma - tg_luma[idx11]) * 14.0f);

      float2 out_v = (tg_flow[idx00] * w00 + tg_flow[idx10] * w10 +
                      tg_flow[idx01] * w01 + tg_flow[idx11] * w11) / (w00 + w10 + w01 + w11);
      out_flow.write(half4(half(out_v.x), half(out_v.y), 0.0h, 0.0h), fxy);
    }
  }
}
