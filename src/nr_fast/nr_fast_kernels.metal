// SPDX-License-Identifier: MIT AND Apache-2.0
// Copyright (c) Netal Contributors.
//
// Native Metal compute kernels for DLSS 5 Neural Rendering (src/nr_fast/).
// Portions of the 1-head/multi-head Swin window blocks, streamed global ViT
// attention, bit-affine softmax, vendor cosine normalization, E4M3 magic-number
// rounder, and deterministic PCG/Catmull-Rom feature math are adapted from:
//   WaveCut / MLX-DLSS Contributors (https://github.com/iamwavecut/MLX-DLSS)
//   Licensed under the Apache License, Version 2.0.

#include <metal_stdlib>
#include <metal_simdgroup_matrix>

using namespace metal;

#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

// ============================================================================
// 1. Common Math, Bit-Exact E4M3 Rounder, Quadratic Gate, Softmax & Color
// ============================================================================

inline half nr_fast_e4m3_magnitude(ushort bits) {
  const half magic = as_type<half>(ushort(max(ushort(bits & 0x7C00u), ushort(0x2400u)) + 0x1C00u));
  return (as_type<half>(bits) + magic) - magic;
}

inline half nr_fast_e4m3(half value) {
  ushort b = as_type<ushort>(value);
  ushort m = b & 0x7FFFu;
  ushort s = m > 0x7C00u ? ushort(0u) : ushort(b ^ m);
  return as_type<half>(ushort(as_type<ushort>(nr_fast_e4m3_magnitude(min(m, ushort(0x5F00u)))) | s));
}

inline half nr_fast_e4m3_no_nan(half value) {
  const ushort b = as_type<ushort>(value);
  const ushort m = b & 0x7FFFu;
  return as_type<half>(ushort(as_type<ushort>(nr_fast_e4m3_magnitude(min(m, ushort(0x5F00u)))) | (b ^ m)));
}

inline half nr_fast_e4m3_nonnegative(half value) {
  return nr_fast_e4m3_magnitude(as_type<ushort>(value));
}

inline half2 nr_fast_e4m3_vec2(half2 v) {
  return half2(nr_fast_e4m3(v.x), nr_fast_e4m3(v.y));
}

inline half4 nr_fast_e4m3_vec4(half4 v) {
  return half4(nr_fast_e4m3(v.x), nr_fast_e4m3(v.y), nr_fast_e4m3(v.z), nr_fast_e4m3(v.w));
}

inline half nr_fast_quad_gate(half value) {
  half clamped = clamp(value, half(-4.0h), half(4.0h));
  half linear = fma(abs(clamped), half(-0.055908203125h), half(0.447265625h));
  half gate = fma(clamped, linear, half(0.89453125h));
  return value * gate;
}

inline half4 nr_fast_quad_gate_vec4(half4 value) {
  half4 clamped = clamp(value, half4(-4.0h), half4(4.0h));
  half4 linear = fma(abs(clamped), half4(-0.055908203125h), half4(0.447265625h));
  half4 gate = fma(clamped, linear, half4(0.89453125h));
  return value * gate;
}

inline half2 nr_fast_vendor_weights(half2 score) {
  half2 affine = fma(score, half2(0.044921875h), half2(1.30078125h));
  affine = clamp(affine, half2(1.03125h), half2(1.5693359375h));
  ushort2 bits = as_type<ushort2>(affine);
  uint packed = uint(bits.x) | (uint(bits.y) << 16);
  uint transformed = (packed << 5) + 0x7FF88000u;
  return as_type<half2>(ushort2(ushort(transformed), ushort(transformed >> 16)));
}

inline float nr_fast_linear_to_srgb_ch(float c) {
  float x = clamp(c, 0.0f, 1.0f);
  return x <= 0.0031308f ? (x * 12.92f) : (1.055f * precise::pow(x, 1.0f / 2.4f) - 0.055f);
}

inline float nr_fast_srgb_to_linear_ch(float c) {
  float x = clamp(c, 0.0f, 1.0f);
  return x <= 0.04045f ? (x / 12.92f) : precise::pow((x + 0.055f) / 1.055f, 2.4f);
}

inline float3 nr_fast_linear_to_srgb(float3 rgb) {
  return float3(
      nr_fast_linear_to_srgb_ch(rgb.x),
      nr_fast_linear_to_srgb_ch(rgb.y),
      nr_fast_linear_to_srgb_ch(rgb.z));
}

inline float3 nr_fast_srgb_to_linear(float3 rgb) {
  return float3(
      nr_fast_srgb_to_linear_ch(rgb.x),
      nr_fast_srgb_to_linear_ch(rgb.y),
      nr_fast_srgb_to_linear_ch(rgb.z));
}

// ============================================================================
// 2. Format Conversion & Elementwise Glue Kernels
// ============================================================================

struct CountParams {
  uint count;
  uint aux0;
  uint aux1;
  uint aux2;
};

kernel void nr_fast_f32_to_f16(
    const device float4* src [[buffer(0)]],
    device half4* dst [[buffer(1)]],
    constant CountParams& p [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid >= p.count) return;
  float4 v = src[tid];
  dst[tid] = half4(half(v.x), half(v.y), half(v.z), half(v.w));
}

kernel void nr_fast_f16_to_f32(
    const device half4* src [[buffer(0)]],
    device float4* dst [[buffer(1)]],
    constant CountParams& p [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid >= p.count) return;
  half4 v = src[tid];
  dst[tid] = float4(float(v.x), float(v.y), float(v.z), float(v.w));
}

struct Pool2PadParams {
  uint in_height;
  uint in_width;
  uint out_height;
  uint out_width;
  uint channels;
  uint publish;
};

// Fused optional spatial bottom/right zero-pad + 2x2 average pool in half
// (row-major sequential half sum * 0.25h) + optional E4M3 publication.
kernel void nr_fast_pool2_pad(
    const device half4* input4 [[buffer(0)]],
    device half4* pooled4 [[buffer(1)]],
    constant Pool2PadParams& p [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  const uint vec_per_px = p.channels >> 2;
  const uint total = p.out_height * p.out_width * vec_per_px;
  if (index >= total) return;

  const uint c4 = index % vec_per_px;
  const uint px_idx = index / vec_per_px;
  const uint py = px_idx / p.out_width;
  const uint px = px_idx % p.out_width;

  const uint sy0 = py * 2u;
  const uint sy1 = sy0 + 1u;
  const uint sx0 = px * 2u;
  const uint sx1 = sx0 + 1u;

  half4 v00 = (sy0 < p.in_height && sx0 < p.in_width)
                  ? input4[(sy0 * p.in_width + sx0) * vec_per_px + c4]
                  : half4(0.0h);
  half4 v01 = (sy0 < p.in_height && sx1 < p.in_width)
                  ? input4[(sy0 * p.in_width + sx1) * vec_per_px + c4]
                  : half4(0.0h);
  half4 v10 = (sy1 < p.in_height && sx0 < p.in_width)
                  ? input4[(sy1 * p.in_width + sx0) * vec_per_px + c4]
                  : half4(0.0h);
  half4 v11 = (sy1 < p.in_height && sx1 < p.in_width)
                  ? input4[(sy1 * p.in_width + sx1) * vec_per_px + c4]
                  : half4(0.0h);

  half4 sum = v00;
  sum = sum + v01;
  sum = sum + v10;
  sum = sum + v11;
  half4 mean = sum * half4(0.25h);
  if (p.publish != 0u) {
    mean = nr_fast_e4m3_vec4(mean);
  }
  pooled4[index] = mean;
}

struct UpsampleMergeParams {
  uint height;
  uint width;
  uint channels;
  uint low_width;
  uint has_low_scale;
  uint publish;
};

kernel void nr_fast_upsample_merge(
    const device half4* low4 [[buffer(0)]],
    const device half4* skip4 [[buffer(1)]],
    const device half4* low_scale4 [[buffer(2)]],
    const device half4* skip_scale4 [[buffer(3)]],
    device half4* out4 [[buffer(4)]],
    constant UpsampleMergeParams& p [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
  const uint vec_per_px = p.channels >> 2;
  const uint total = p.height * p.width * vec_per_px;
  if (index >= total) return;

  const uint c4 = index % vec_per_px;
  const uint px = index / vec_per_px;
  const uint y = px / p.width;
  const uint x = px % p.width;

  half4 upsampled = low4[((y >> 1) * p.low_width + (x >> 1)) * vec_per_px + c4];
  if (p.has_low_scale != 0u) {
    upsampled = upsampled * low_scale4[c4];
  }
  half4 scaled_skip = skip4[index] * skip_scale4[c4];
  half4 value = upsampled + scaled_skip;
  if (p.publish != 0u) {
    value = nr_fast_e4m3_vec4(value);
  }
  out4[index] = value;
}

// ============================================================================
// 3. Fused 1-Head 32-Channel Swin Window Block + Stem/Pool/Merge/Head Variants
// ============================================================================

struct Window1H32Params {
  uint height;
  uint width;
  uint pad_top;
  uint pad_left;
  uint publish;
  uint low_width;
  uint merge_flags;  // bit 0: has_low_scale, bit 1: publish_merged
  uint emit_full;
};

inline void nr_fast_window_1h32_core(
    thread simdgroup_matrix<half, 8, 8> (&xt)[4],
    thread simdgroup_matrix<half, 8, 8> (&out_tiles)[4],
    const device half* expansion,
    const device half* ffn_proj,
    const device half2* ffn_cos2,
    const device half* qkv_weight,
    const device half* attn_scale,
    const device half2* attn_bias2,
    const device half* attn_proj,
    const device half2* attn_cos2,
    threadgroup half* K,
    threadgroup half* V,
    threadgroup half* row_scratch,
    uint simd,
    ushort lane,
    ushort fragment_row,
    ushort fragment_col) {
  const half scale = attn_scale[0];
  simdgroup_matrix<half, 8, 8> ft[4];
  simdgroup_matrix<half, 8, 8> qt[4];
  simdgroup_matrix<half, 8, 8> at[4];

  const uint tid = simd * 32u + uint(lane);

  // Phase 1: FFN (32 -> 128 -> 32) in two 64-wide chunks, half accumulation.
  // Cooperatively stage each 4 KB expansion chunk + 4 KB projection chunk into idle K/V SRAM.
  simdgroup_matrix<half, 8, 8> pacc[4];
  _Pragma("clang loop unroll(full)")
  for (uint o = 0; o < 4; ++o) {
    pacc[o].thread_elements()[0] = 0.0h;
    pacc[o].thread_elements()[1] = 0.0h;
  }
  for (uint chunk = 0; chunk < 2; ++chunk) {
    threadgroup half4* exp_s4 = (threadgroup half4*)K;
    threadgroup half4* prj_s4 = (threadgroup half4*)V;
    _Pragma("clang loop unroll(full)")
    for (uint r4 = 0; r4 < 2u; ++r4) {
      const uint idx4 = tid + r4 * 256u;
      const uint erow = idx4 >> 4;
      const uint ecol4 = (idx4 & 15u) << 2;
      exp_s4[idx4] = *(const device half4*)(expansion + erow * 128u + chunk * 64u + ecol4);
      prj_s4[idx4] = *(const device half4*)(ffn_proj + chunk * 2048u + idx4 * 4u);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    _Pragma("clang loop unroll(full)")
    for (uint c = 0; c < 8; ++c) {
      simdgroup_matrix<half, 8, 8> hidden;
      hidden.thread_elements()[0] = 0.0h;
      hidden.thread_elements()[1] = 0.0h;
      _Pragma("clang loop unroll(full)")
      for (uint k = 0; k < 4; ++k) {
        simdgroup_matrix<half, 8, 8> right;
        simdgroup_load(right, K + k * 8u * 64u + c * 8u, 64, ulong2(0), false);
        simdgroup_multiply_accumulate(hidden, xt[k], right, hidden);
      }
      _Pragma("clang loop unroll(full)")
      for (uint e = 0; e < 2; ++e) {
        hidden.thread_elements()[e] = nr_fast_e4m3(nr_fast_quad_gate(hidden.thread_elements()[e]));
      }
      _Pragma("clang loop unroll(full)")
      for (uint o = 0; o < 4; ++o) {
        simdgroup_matrix<half, 8, 8> right;
        simdgroup_load(right, V + c * 8u * 32u + o * 8u, 32, ulong2(0), false);
        simdgroup_multiply_accumulate(pacc[o], hidden, right, pacc[o]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  _Pragma("clang loop unroll(full)")
  for (uint o = 0; o < 4; ++o) {
    const half2 cosine = ffn_cos2[(o * 8 + fragment_col) / 2];
    _Pragma("clang loop unroll(full)")
    for (uint e = 0; e < 2; ++e) {
      const half skip = xt[o].thread_elements()[e] * cosine[e];
      ft[o].thread_elements()[e] = pacc[o].thread_elements()[e] + skip;
    }
  }

  // Phase 2: QKV projection + vendor XOR cosine norm + E4M3 publish.
  threadgroup half* qkv_s = K + 4096u;
  for (uint part = 0; part < 3; ++part) {
    const uint qrow = tid >> 3;
    const uint qcol4 = (tid & 7u) << 2;
    *(threadgroup half4*)(qkv_s + tid * 4u) =
        *(const device half4*)(qkv_weight + qrow * 96u + part * 32u + qcol4);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_matrix<half, 8, 8> result[4];
    _Pragma("clang loop unroll(full)")
    for (uint o = 0; o < 4; ++o) {
      simdgroup_matrix<float, 8, 8> acc;
      acc.thread_elements()[0] = 0.0f;
      acc.thread_elements()[1] = 0.0f;
      _Pragma("clang loop unroll(full)")
      for (uint k = 0; k < 4; ++k) {
        simdgroup_matrix<half, 8, 8> right;
        simdgroup_load(right, qkv_s + k * 8u * 32u + o * 8u, 32, ulong2(0), false);
        simdgroup_multiply_accumulate(acc, ft[k], right, acc);
      }
      result[o].thread_elements()[0] = half(acc.thread_elements()[0]);
      result[o].thread_elements()[1] = half(acc.thread_elements()[1]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup half* destination = (part == 1 ? K : V) + simd * 8 * 32;
    if (part < 2) {
      half partial[2];
      _Pragma("clang loop unroll(full)")
      for (uint e = 0; e < 2; ++e) {
        const half v0 = result[0].thread_elements()[e];
        const half v1 = result[1].thread_elements()[e];
        const half v2 = result[2].thread_elements()[e];
        const half v3 = result[3].thread_elements()[e];
        half first = fma(v1, v1, v0 * v0);
        half second = fma(v3, v3, v2 * v2);
        partial[e] = first + second;
      }
      threadgroup half2* partials = (threadgroup half2*)(V + simd * 8 * 32) + fragment_row * 4;
      partials[((lane >> 2) & 2) | (lane & 1)] = half2(partial[0], partial[1]);
      simdgroup_barrier(mem_flags::mem_threadgroup);
      const half2 p0 = partials[0];
      const half2 p1 = partials[1];
      const half2 p2 = partials[2];
      const half2 p3 = partials[3];
      simdgroup_barrier(mem_flags::mem_threadgroup);
      half xor_one[2];
      _Pragma("clang loop unroll(full)")
      for (uint e = 0; e < 2; ++e) {
        xor_one[e] = (p0[e] + p2[e]) + (p1[e] + p3[e]);
      }
      half norm = max(xor_one[0] + xor_one[1], half(0.00006198883056640625h));
      const half inv_norm = half(metal::fast::rsqrt(float(norm)));
      _Pragma("clang loop unroll(full)")
      for (uint o = 0; o < 4; ++o) {
        _Pragma("clang loop unroll(full)")
        for (uint e = 0; e < 2; ++e) {
          half normalized = result[o].thread_elements()[e] * inv_norm;
          if (part == 0) {
            normalized *= scale;
          }
          result[o].thread_elements()[e] = nr_fast_e4m3(normalized);
        }
        if (part == 0) {
          qt[o] = result[o];
        } else {
          simdgroup_store(result[o], destination + o * 8, 32, ulong2(0), false);
        }
      }
    } else {
      _Pragma("clang loop unroll(full)")
      for (uint o = 0; o < 4; ++o) {
        _Pragma("clang loop unroll(full)")
        for (uint e = 0; e < 2; ++e) {
          result[o].thread_elements()[e] = nr_fast_e4m3(result[o].thread_elements()[e]);
        }
        simdgroup_store(result[o], destination + o * 8, 32, ulong2(0), false);
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Phase 3: Window attention scores, bit-affine softmax, E4M3(P * V).
  const uint token0 = simd * 8;
  simdgroup_matrix<half, 8, 8> weights[8];
  _Pragma("clang loop unroll(full)")
  for (uint j = 0; j < 8; ++j) {
    simdgroup_matrix<float, 8, 8> acc;
    acc.thread_elements()[0] = 0.0f;
    acc.thread_elements()[1] = 0.0f;
    _Pragma("clang loop unroll(full)")
    for (uint k = 0; k < 4; ++k) {
      simdgroup_matrix<half, 8, 8> right;
      simdgroup_load(right, K + j * 8 * 32 + k * 8, 32, ulong2(0), true);
      simdgroup_multiply_accumulate(acc, qt[k], right, acc);
    }
    half2 score = half2(half(acc.thread_elements()[0]), half(acc.thread_elements()[1])) +
                  attn_bias2[((token0 + fragment_row) * 64 + j * 8 + fragment_col) / 2];
    half2 affine = fma(score, half2(0.044921875h), half2(1.30078125h));
    affine = clamp(affine, half2(1.03125h), half2(1.5693359375h));
    half2 weight = as_type<half2>(ushort2((as_type<ushort2>(affine) << 5) ^ ushort2(0x8000u)));
    weights[j].thread_elements()[0] = weight.x;
    weights[j].thread_elements()[1] = weight.y;
  }

  half total = 0.0h;
  _Pragma("clang loop unroll(full)")
  for (uint pass = 0; pass < 2; ++pass) {
    _Pragma("clang loop unroll(full)")
    for (uint j = 0; j < 4; ++j) {
      simdgroup_store(weights[pass * 4 + j], row_scratch + j * 8, 32, ulong2(0), false);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    const threadgroup half4* row4 = (const threadgroup half4*)(row_scratch + fragment_row * 32);
    _Pragma("clang loop unroll(full)")
    for (uint i = 0; i < 8; ++i) {
      const half4 w = row4[i];
      total += w.x;
      total += w.y;
      total += w.z;
      total += w.w;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }

  const half reciprocal = half(1.0f / float(total));
  _Pragma("clang loop unroll(full)")
  for (uint j = 0; j < 8; ++j) {
    _Pragma("clang loop unroll(full)")
    for (uint e = 0; e < 2; ++e) {
      weights[j].thread_elements()[e] =
          nr_fast_e4m3_nonnegative(weights[j].thread_elements()[e] * reciprocal);
    }
  }

  _Pragma("clang loop unroll(full)")
  for (uint o = 0; o < 4; ++o) {
    simdgroup_matrix<float, 8, 8> acc;
    acc.thread_elements()[0] = 0.0f;
    acc.thread_elements()[1] = 0.0f;
    _Pragma("clang loop unroll(full)")
    for (uint k = 0; k < 8; ++k) {
      simdgroup_matrix<half, 8, 8> right;
      simdgroup_load(right, V + k * 8 * 32 + o * 8, 32, ulong2(0), false);
      simdgroup_multiply_accumulate(acc, weights[k], right, acc);
    }
    at[o].thread_elements()[0] = nr_fast_e4m3_no_nan(half(acc.thread_elements()[0]));
    at[o].thread_elements()[1] = nr_fast_e4m3_no_nan(half(acc.thread_elements()[1]));
  }

  // Phase 4: Output projection + cosine residual (stage 2 KB attn_proj into K SRAM).
  threadgroup_barrier(mem_flags::mem_threadgroup);
  *(threadgroup half4*)(K + tid * 4u) = *(const device half4*)(attn_proj + tid * 4u);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  _Pragma("clang loop unroll(full)")
  for (uint o = 0; o < 4; ++o) {
    simdgroup_matrix<float, 8, 8> acc;
    acc.thread_elements()[0] = 0.0f;
    acc.thread_elements()[1] = 0.0f;
    _Pragma("clang loop unroll(full)")
    for (uint k = 0; k < 4; ++k) {
      simdgroup_matrix<half, 8, 8> right;
      simdgroup_load(right, K + k * 8u * 32u + o * 8u, 32, ulong2(0), false);
      simdgroup_multiply_accumulate(acc, at[k], right, acc);
    }
    const half2 cosine = attn_cos2[(o * 8 + fragment_col) / 2];
    _Pragma("clang loop unroll(full)")
    for (uint e = 0; e < 2; ++e) {
      const half projected = half(acc.thread_elements()[e]);
      const half skip = ft[o].thread_elements()[e] * cosine[e];
      out_tiles[o].thread_elements()[e] = projected + skip;
    }
  }
}

// 3a. Standard 1-head 32ch Swin window block ([1, H, W, 32] -> [1, H, W, 32]).
kernel void nr_fast_fused_window_1h32(
    const device half* input [[buffer(0)]],
    const device half* expansion [[buffer(1)]],
    const device half* ffn_proj [[buffer(2)]],
    const device half* ffn_cos [[buffer(3)]],
    const device half* qkv_weight [[buffer(4)]],
    const device half* attn_scale [[buffer(5)]],
    const device half* attn_bias [[buffer(6)]],
    const device half* attn_proj [[buffer(7)]],
    const device half* attn_cos [[buffer(8)]],
    device half* output [[buffer(9)]],
    constant Window1H32Params& p [[buffer(10)]],
    uint window_index [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane_in [[thread_index_in_simdgroup]]) {
  const ushort lane = ushort(lane_in);
  threadgroup half4 arena4[6144 / 4];
  threadgroup half* arena = (threadgroup half*)arena4;
  threadgroup half* K = arena;
  threadgroup half* V = arena + 2048;
  threadgroup half* row_scratch = arena + 4096 + simd * 256;

  const ushort fragment_row = ((lane / 4) & 4) + ((lane / 2) % 4);
  const ushort fragment_col = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
  const uint windows_x = (p.width + p.pad_left + 7u) / 8u;
  const uint wy = window_index / windows_x;
  const uint wx = window_index % windows_x;
  const int row_y0 = int(wy * 8u) - int(p.pad_top);
  const int col_x0 = int(wx * 8u) - int(p.pad_left);
  const int y = row_y0 + int(simd);
  const int token_x = col_x0 + int(fragment_row);
  const bool inside = (y >= 0 && y < int(p.height) && token_x >= 0 && token_x < int(p.width));

  const device half2* input2 = (const device half2*)input;
  device half2* output2 = (device half2*)output;

  simdgroup_matrix<half, 8, 8> xt[4];
  const uint base = inside ? (uint(y) * p.width + uint(token_x)) * 16u + fragment_col / 2u : 0u;
  _Pragma("clang loop unroll(full)")
  for (uint k = 0; k < 4; ++k) {
    const half2 v = inside ? input2[base + k * 4u] : half2(0.0h);
    xt[k].thread_elements()[0] = v.x;
    xt[k].thread_elements()[1] = v.y;
  }

  simdgroup_matrix<half, 8, 8> out_tiles[4];
  nr_fast_window_1h32_core(
      xt, out_tiles, expansion, ffn_proj, (const device half2*)ffn_cos, qkv_weight, attn_scale,
      (const device half2*)attn_bias, attn_proj, (const device half2*)attn_cos, K, V, row_scratch,
      simd, lane, fragment_row, fragment_col);

  if (inside) {
    _Pragma("clang loop unroll(full)")
    for (uint o = 0; o < 4; ++o) {
      half2 val = half2(out_tiles[o].thread_elements()[0], out_tiles[o].thread_elements()[1]);
      if (p.publish != 0u) {
        val = nr_fast_e4m3_vec2(val);
      }
      output2[(uint(y) * p.width + uint(token_x)) * 16u + (o * 8u + fragment_col) / 2u] = val;
    }
  }
}

// 3b. Fused Stem (block 0): [H, W, 16] x adapter[16, 32] -> 1h32 -> fullSkip + 2x2 pool.
kernel void nr_fast_fused_window_1h32_adapter_pool(
    const device half* input16 [[buffer(0)]],
    const device half* adapter [[buffer(1)]],
    const device half* expansion [[buffer(2)]],
    const device half* ffn_proj [[buffer(3)]],
    const device half* ffn_cos [[buffer(4)]],
    const device half* qkv_weight [[buffer(5)]],
    const device half* attn_scale [[buffer(6)]],
    const device half* attn_bias [[buffer(7)]],
    const device half* attn_proj [[buffer(8)]],
    const device half* attn_cos [[buffer(9)]],
    device half* full_out [[buffer(10)]],
    device half* pooled_out [[buffer(11)]],
    constant Window1H32Params& p [[buffer(12)]],
    uint window_index [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane_in [[thread_index_in_simdgroup]]) {
  const ushort lane = ushort(lane_in);
  threadgroup half4 arena4[6144 / 4];
  threadgroup half* arena = (threadgroup half*)arena4;
  threadgroup half* K = arena;
  threadgroup half* V = arena + 2048;
  threadgroup half* row_scratch = arena + 4096 + simd * 256;

  const ushort fragment_row = ((lane / 4) & 4) + ((lane / 2) % 4);
  const ushort fragment_col = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
  const uint windows_x = (p.width + p.pad_left + 7u) / 8u;
  const uint wy = window_index / windows_x;
  const uint wx = window_index % windows_x;
  const int row_y0 = int(wy * 8u) - int(p.pad_top);
  const int col_x0 = int(wx * 8u) - int(p.pad_left);
  const int y = row_y0 + int(simd);
  const int token_x = col_x0 + int(fragment_row);
  const bool inside = (y >= 0 && y < int(p.height) && token_x >= 0 && token_x < int(p.width));

  const device half2* input2 = (const device half2*)input16;
  const uint base = inside ? (uint(y) * p.width + uint(token_x)) * 8u + fragment_col / 2u : 0u;
  simdgroup_matrix<half, 8, 8> raw[2];
  _Pragma("clang loop unroll(full)")
  for (uint k = 0; k < 2; ++k) {
    const half2 v = inside ? input2[base + k * 4u] : half2(0.0h);
    raw[k].thread_elements()[0] = v.x;
    raw[k].thread_elements()[1] = v.y;
  }

  simdgroup_matrix<half, 8, 8> xt[4];
  _Pragma("clang loop unroll(full)")
  for (uint o = 0; o < 4; ++o) {
    simdgroup_matrix<float, 8, 8> acc;
    acc.thread_elements()[0] = 0.0f;
    acc.thread_elements()[1] = 0.0f;
    _Pragma("clang loop unroll(full)")
    for (uint k = 0; k < 2; ++k) {
      simdgroup_matrix<half, 8, 8> right;
      simdgroup_load(right, adapter + k * 8 * 32 + o * 8, 32, ulong2(0), false);
      simdgroup_multiply_accumulate(acc, raw[k], right, acc);
    }
    xt[o].thread_elements()[0] = half(acc.thread_elements()[0]);
    xt[o].thread_elements()[1] = half(acc.thread_elements()[1]);
  }

  simdgroup_matrix<half, 8, 8> out_tiles[4];
  nr_fast_window_1h32_core(
      xt, out_tiles, expansion, ffn_proj, (const device half2*)ffn_cos, qkv_weight, attn_scale,
      (const device half2*)attn_bias, attn_proj, (const device half2*)attn_cos, K, V, row_scratch,
      simd, lane, fragment_row, fragment_col);

  device half2* full2 = (device half2*)full_out;
  if (p.emit_full != 0u && inside) {
    _Pragma("clang loop unroll(full)")
    for (uint o = 0; o < 4; ++o) {
      full2[(uint(y) * p.width + uint(token_x)) * 16u + (o * 8u + fragment_col) / 2u] =
          nr_fast_e4m3_vec2(half2(out_tiles[o].thread_elements()[0], out_tiles[o].thread_elements()[1]));
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  _Pragma("clang loop unroll(full)")
  for (uint o = 0; o < 4; ++o) {
    simdgroup_store(out_tiles[o], arena + simd * 256 + o * 8, 32, ulong2(0), false);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint pooled_token = simd * 2u + lane / 16u;
  const uint py = pooled_token / 4u;
  const uint px = pooled_token % 4u;
  const uint pair = lane % 16u;
  const int src_y = row_y0 + int(py * 2u);
  const int src_x = col_x0 + int(px * 2u);
  if (src_y >= 0 && src_y < int(p.height) && src_x >= 0 && src_x < int(p.width)) {
    const threadgroup half2* win2 = (const threadgroup half2*)arena;
    const uint src = (py * 2u * 8u + px * 2u) * 16u + pair;
    half2 sum = win2[src];
    sum = sum + win2[src + 16u];
    sum = sum + win2[src + 8u * 16u];
    sum = sum + win2[src + 8u * 16u + 16u];
    sum = sum * 0.25h;
    device half2* pooled2 = (device half2*)pooled_out;
    pooled2[(uint(src_y / 2) * (p.width / 2u) + uint(src_x / 2)) * 16u + pair] = nr_fast_e4m3_vec2(sum);
  }
}

// 3c. Fused Stage-0 Downsample Pool: [H, W, 32] -> 1h32 -> 2x2 pool + E4M3.
kernel void nr_fast_fused_window_1h32_pool(
    const device half* input [[buffer(0)]],
    const device half* expansion [[buffer(1)]],
    const device half* ffn_proj [[buffer(2)]],
    const device half* ffn_cos [[buffer(3)]],
    const device half* qkv_weight [[buffer(4)]],
    const device half* attn_scale [[buffer(5)]],
    const device half* attn_bias [[buffer(6)]],
    const device half* attn_proj [[buffer(7)]],
    const device half* attn_cos [[buffer(8)]],
    device half* pooled_out [[buffer(9)]],
    constant Window1H32Params& p [[buffer(10)]],
    uint window_index [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane_in [[thread_index_in_simdgroup]]) {
  const ushort lane = ushort(lane_in);
  threadgroup half4 arena4[6144 / 4];
  threadgroup half* arena = (threadgroup half*)arena4;
  threadgroup half* K = arena;
  threadgroup half* V = arena + 2048;
  threadgroup half* row_scratch = arena + 4096 + simd * 256;

  const ushort fragment_row = ((lane / 4) & 4) + ((lane / 2) % 4);
  const ushort fragment_col = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
  const uint windows_x = (p.width + p.pad_left + 7u) / 8u;
  const uint wy = window_index / windows_x;
  const uint wx = window_index % windows_x;
  const int row_y0 = int(wy * 8u) - int(p.pad_top);
  const int col_x0 = int(wx * 8u) - int(p.pad_left);
  const int y = row_y0 + int(simd);
  const int token_x = col_x0 + int(fragment_row);
  const bool inside = (y >= 0 && y < int(p.height) && token_x >= 0 && token_x < int(p.width));

  const device half2* input2 = (const device half2*)input;
  simdgroup_matrix<half, 8, 8> xt[4];
  const uint base = inside ? (uint(y) * p.width + uint(token_x)) * 16u + fragment_col / 2u : 0u;
  _Pragma("clang loop unroll(full)")
  for (uint k = 0; k < 4; ++k) {
    const half2 v = inside ? input2[base + k * 4u] : half2(0.0h);
    xt[k].thread_elements()[0] = v.x;
    xt[k].thread_elements()[1] = v.y;
  }

  simdgroup_matrix<half, 8, 8> out_tiles[4];
  nr_fast_window_1h32_core(
      xt, out_tiles, expansion, ffn_proj, (const device half2*)ffn_cos, qkv_weight, attn_scale,
      (const device half2*)attn_bias, attn_proj, (const device half2*)attn_cos, K, V, row_scratch,
      simd, lane, fragment_row, fragment_col);

  threadgroup_barrier(mem_flags::mem_threadgroup);
  _Pragma("clang loop unroll(full)")
  for (uint o = 0; o < 4; ++o) {
    simdgroup_store(out_tiles[o], arena + simd * 256 + o * 8, 32, ulong2(0), false);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint pooled_token = simd * 2u + lane / 16u;
  const uint py = pooled_token / 4u;
  const uint px = pooled_token % 4u;
  const uint pair = lane % 16u;
  const int src_y = row_y0 + int(py * 2u);
  const int src_x = col_x0 + int(px * 2u);
  if (src_y >= 0 && src_y < int(p.height) && src_x >= 0 && src_x < int(p.width)) {
    const threadgroup half2* win2 = (const threadgroup half2*)arena;
    const uint src = (py * 2u * 8u + px * 2u) * 16u + pair;
    half2 sum = win2[src];
    sum = sum + win2[src + 16u];
    sum = sum + win2[src + 8u * 16u];
    sum = sum + win2[src + 8u * 16u + 16u];
    sum = sum * 0.25h;
    device half2* pooled2 = (device half2*)pooled_out;
    pooled2[(uint(src_y / 2) * (p.width / 2u) + uint(src_x / 2)) * 16u + pair] = nr_fast_e4m3_vec2(sum);
  }
}

// 3d. Fused Decoder Stage-0 Upsample Merge + 1h32 Store.
kernel void nr_fast_fused_window_1h32_merge_store(
    const device half* low [[buffer(0)]],
    const device half* skip [[buffer(1)]],
    const device half* low_scale [[buffer(2)]],
    const device half* skip_scale [[buffer(3)]],
    const device half* expansion [[buffer(4)]],
    const device half* ffn_proj [[buffer(5)]],
    const device half* ffn_cos [[buffer(6)]],
    const device half* qkv_weight [[buffer(7)]],
    const device half* attn_scale [[buffer(8)]],
    const device half* attn_bias [[buffer(9)]],
    const device half* attn_proj [[buffer(10)]],
    const device half* attn_cos [[buffer(11)]],
    device half* output [[buffer(12)]],
    constant Window1H32Params& p [[buffer(13)]],
    uint window_index [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane_in [[thread_index_in_simdgroup]]) {
  const ushort lane = ushort(lane_in);
  threadgroup half4 arena4[6144 / 4];
  threadgroup half* arena = (threadgroup half*)arena4;
  threadgroup half* K = arena;
  threadgroup half* V = arena + 2048;
  threadgroup half* row_scratch = arena + 4096 + simd * 256;

  const ushort fragment_row = ((lane / 4) & 4) + ((lane / 2) % 4);
  const ushort fragment_col = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
  const uint windows_x = (p.width + p.pad_left + 7u) / 8u;
  const uint wy = window_index / windows_x;
  const uint wx = window_index % windows_x;
  const int row_y0 = int(wy * 8u) - int(p.pad_top);
  const int col_x0 = int(wx * 8u) - int(p.pad_left);
  const int y = row_y0 + int(simd);
  const int token_x = col_x0 + int(fragment_row);
  const bool inside = (y >= 0 && y < int(p.height) && token_x >= 0 && token_x < int(p.width));

  const device half2* low2 = (const device half2*)low;
  const device half2* skip2 = (const device half2*)skip;
  const device half2* low_scale2 = (const device half2*)low_scale;
  const device half2* skip_scale2 = (const device half2*)skip_scale;
  const bool has_low_scale = (p.merge_flags & 1u) != 0u;
  const bool publish_merged = (p.merge_flags & 2u) != 0u;

  simdgroup_matrix<half, 8, 8> xt[4];
  const uint skip_base = inside ? (uint(y) * p.width + uint(token_x)) * 16u + fragment_col / 2u : 0u;
  const uint low_base = inside ? (uint(y / 2) * p.low_width + uint(token_x / 2)) * 16u + fragment_col / 2u : 0u;
  _Pragma("clang loop unroll(full)")
  for (uint k = 0; k < 4; ++k) {
    half2 value = half2(0.0h);
    if (inside) {
      half2 upsampled = low2[low_base + k * 4u];
      if (has_low_scale) {
        upsampled = upsampled * low_scale2[(k * 8u + fragment_col) / 2u];
      }
      half2 scaled_skip = skip2[skip_base + k * 4u] * skip_scale2[(k * 8u + fragment_col) / 2u];
      value = upsampled + scaled_skip;
      if (publish_merged) {
        value = nr_fast_e4m3_vec2(value);
      }
    }
    xt[k].thread_elements()[0] = value.x;
    xt[k].thread_elements()[1] = value.y;
  }

  simdgroup_matrix<half, 8, 8> out_tiles[4];
  nr_fast_window_1h32_core(
      xt, out_tiles, expansion, ffn_proj, (const device half2*)ffn_cos, qkv_weight, attn_scale,
      (const device half2*)attn_bias, attn_proj, (const device half2*)attn_cos, K, V, row_scratch,
      simd, lane, fragment_row, fragment_col);

  if (inside) {
    device half2* output2 = (device half2*)output;
    _Pragma("clang loop unroll(full)")
    for (uint o = 0; o < 4; ++o) {
      half2 val = half2(out_tiles[o].thread_elements()[0], out_tiles[o].thread_elements()[1]);
      if (p.publish != 0u) {
        val = nr_fast_e4m3_vec2(val);
      }
      output2[(uint(y) * p.width + uint(token_x)) * 16u + (o * 8u + fragment_col) / 2u] = val;
    }
  }
}

// 3e. Fused Post Block: upsample merge + 1h32 + [32, 8] head -> [1, H, W, 4].
kernel void nr_fast_fused_window_1h32_merge_head(
    const device half* low [[buffer(0)]],
    const device half* skip [[buffer(1)]],
    const device half* low_scale [[buffer(2)]],
    const device half* skip_scale [[buffer(3)]],
    const device half* expansion [[buffer(4)]],
    const device half* ffn_proj [[buffer(5)]],
    const device half* ffn_cos [[buffer(6)]],
    const device half* qkv_weight [[buffer(7)]],
    const device half* attn_scale [[buffer(8)]],
    const device half* attn_bias [[buffer(9)]],
    const device half* attn_proj [[buffer(10)]],
    const device half* attn_cos [[buffer(11)]],
    const device half* head_weight [[buffer(12)]],
    device half* output4 [[buffer(13)]],
    constant Window1H32Params& p [[buffer(14)]],
    uint window_index [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane_in [[thread_index_in_simdgroup]]) {
  const ushort lane = ushort(lane_in);
  threadgroup half4 arena4[6144 / 4];
  threadgroup half* arena = (threadgroup half*)arena4;
  threadgroup half* K = arena;
  threadgroup half* V = arena + 2048;
  threadgroup half* row_scratch = arena + 4096 + simd * 256;

  const ushort fragment_row = ((lane / 4) & 4) + ((lane / 2) % 4);
  const ushort fragment_col = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
  const uint windows_x = (p.width + p.pad_left + 7u) / 8u;
  const uint wy = window_index / windows_x;
  const uint wx = window_index % windows_x;
  const int row_y0 = int(wy * 8u) - int(p.pad_top);
  const int col_x0 = int(wx * 8u) - int(p.pad_left);
  const int y = row_y0 + int(simd);
  const int token_x = col_x0 + int(fragment_row);
  const bool inside = (y >= 0 && y < int(p.height) && token_x >= 0 && token_x < int(p.width));

  const device half2* low2 = (const device half2*)low;
  const device half2* skip2 = (const device half2*)skip;
  const device half2* low_scale2 = (const device half2*)low_scale;
  const device half2* skip_scale2 = (const device half2*)skip_scale;
  const bool has_low_scale = (p.merge_flags & 1u) != 0u;
  const bool publish_merged = (p.merge_flags & 2u) != 0u;

  simdgroup_matrix<half, 8, 8> xt[4];
  const uint skip_base = inside ? (uint(y) * p.width + uint(token_x)) * 16u + fragment_col / 2u : 0u;
  const uint low_base = inside ? (uint(y / 2) * p.low_width + uint(token_x / 2)) * 16u + fragment_col / 2u : 0u;
  _Pragma("clang loop unroll(full)")
  for (uint k = 0; k < 4; ++k) {
    half2 value = half2(0.0h);
    if (inside) {
      half2 upsampled = low2[low_base + k * 4u];
      if (has_low_scale) {
        upsampled = upsampled * low_scale2[(k * 8u + fragment_col) / 2u];
      }
      half2 scaled_skip = skip2[skip_base + k * 4u] * skip_scale2[(k * 8u + fragment_col) / 2u];
      value = upsampled + scaled_skip;
      if (publish_merged) {
        value = nr_fast_e4m3_vec2(value);
      }
    }
    xt[k].thread_elements()[0] = value.x;
    xt[k].thread_elements()[1] = value.y;
  }

  simdgroup_matrix<half, 8, 8> out_tiles[4];
  nr_fast_window_1h32_core(
      xt, out_tiles, expansion, ffn_proj, (const device half2*)ffn_cos, qkv_weight, attn_scale,
      (const device half2*)attn_bias, attn_proj, (const device half2*)attn_cos, K, V, row_scratch,
      simd, lane, fragment_row, fragment_col);

  simdgroup_matrix<float, 8, 8> head_acc;
  head_acc.thread_elements()[0] = 0.0f;
  head_acc.thread_elements()[1] = 0.0f;
  _Pragma("clang loop unroll(full)")
  for (uint k = 0; k < 4; ++k) {
    simdgroup_matrix<half, 8, 8> right;
    simdgroup_load(right, head_weight + k * 8 * 8, 8, ulong2(0), false);
    simdgroup_multiply_accumulate(head_acc, out_tiles[k], right, head_acc);
  }
  const half2 projected = half2(half(head_acc.thread_elements()[0]), half(head_acc.thread_elements()[1]));
  const half2 upper = simd_shuffle_xor(projected, ushort(8));
  if ((lane & 8u) == 0u && inside) {
    device half2* out2 = (device half2*)output4;
    out2[(uint(y) * p.width + uint(token_x)) * 2u + fragment_col / 2u] = projected + upper;
  }
}

// ============================================================================
// 4. Multi-Head Fused Window Attention & Register-Resident Branched FFN (64/128)
// ============================================================================

struct WindowMHParams {
  uint height;
  uint width;
  uint pad_top;
  uint pad_left;
  uint head_count;
  uint publish;
};

// 8 simdgroups (256 threads) per (8x8 window, head) for head_count in {2, 4, 8, 16}.
kernel void nr_fast_window_attention_mh(
    const device half* qkv [[buffer(0)]],
    const device half* attn_scale [[buffer(1)]],
    const device half* attn_bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant WindowMHParams& p [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane_in [[thread_index_in_simdgroup]]) {
  const ushort lane = ushort(lane_in);
  threadgroup half4 arena4[6144 / 4];
  threadgroup half* arena = (threadgroup half*)arena4;
  threadgroup half* K = arena;
  threadgroup half* V = arena + 2048;
  threadgroup half* row_scratch = arena + 4096 + simd * 256;

  const ushort fragment_row = ((lane / 4) & 4) + ((lane / 2) % 4);
  const ushort fragment_col = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
  const uint channels = p.head_count * 32u;
  const uint row_stride = channels * 3u;
  const uint windows_x = (p.width + p.pad_left + 7u) / 8u;
  const uint head = tgid % p.head_count;
  const uint window_index = tgid / p.head_count;
  const uint wy = window_index / windows_x;
  const uint wx = window_index % windows_x;
  const int y = int(wy * 8u) - int(p.pad_top) + int(simd);
  const int x = int(wx * 8u) - int(p.pad_left) + int(fragment_row);
  const bool inside = (y >= 0 && y < int(p.height) && x >= 0 && x < int(p.width));
  const uint pixel = inside ? (uint(y) * p.width + uint(x)) : 0u;

  const device half2* qkv2 = (const device half2*)qkv;
  device half2* output2 = (device half2*)output;
  const device half2* bias2 = (const device half2*)attn_bias;
  const half scale = attn_scale[head];

  simdgroup_matrix<half, 8, 8> qt[4];
  _Pragma("clang loop unroll(full)")
  for (uint part = 0; part < 3; ++part) {
    simdgroup_matrix<half, 8, 8> tile[4];
    const uint base = (pixel * row_stride + part * channels + head * 32u + fragment_col) / 2u;
    _Pragma("clang loop unroll(full)")
    for (uint o = 0; o < 4; ++o) {
      const half2 value = inside ? qkv2[base + o * 4u] : half2(0.0h);
      tile[o].thread_elements()[0] = value.x;
      tile[o].thread_elements()[1] = value.y;
    }
    if (part < 2) {
      half partial[2];
      _Pragma("clang loop unroll(full)")
      for (uint e = 0; e < 2; ++e) {
        const half v0 = tile[0].thread_elements()[e];
        const half v1 = tile[1].thread_elements()[e];
        const half v2 = tile[2].thread_elements()[e];
        const half v3 = tile[3].thread_elements()[e];
        half first = fma(v1, v1, v0 * v0);
        half second = fma(v3, v3, v2 * v2);
        partial[e] = first + second;
      }
      threadgroup half2* partials = (threadgroup half2*)(V + simd * 8 * 32) + fragment_row * 4;
      partials[((lane >> 2) & 2) | (lane & 1)] = half2(partial[0], partial[1]);
      simdgroup_barrier(mem_flags::mem_threadgroup);
      const half2 p0 = partials[0];
      const half2 p1 = partials[1];
      const half2 p2 = partials[2];
      const half2 p3 = partials[3];
      simdgroup_barrier(mem_flags::mem_threadgroup);
      half xor_one[2];
      _Pragma("clang loop unroll(full)")
      for (uint e = 0; e < 2; ++e) {
        xor_one[e] = (p0[e] + p2[e]) + (p1[e] + p3[e]);
      }
      half norm = max(xor_one[0] + xor_one[1], half(0.00006198883056640625h));
      const half inv_norm = half(metal::fast::rsqrt(float(norm)));
      _Pragma("clang loop unroll(full)")
      for (uint o = 0; o < 4; ++o) {
        _Pragma("clang loop unroll(full)")
        for (uint e = 0; e < 2; ++e) {
          half normalized = tile[o].thread_elements()[e] * inv_norm;
          if (part == 0) {
            normalized *= scale;
          }
          tile[o].thread_elements()[e] = nr_fast_e4m3(normalized);
        }
      }
    } else {
      _Pragma("clang loop unroll(full)")
      for (uint o = 0; o < 4; ++o) {
        _Pragma("clang loop unroll(full)")
        for (uint e = 0; e < 2; ++e) {
          tile[o].thread_elements()[e] = nr_fast_e4m3(tile[o].thread_elements()[e]);
        }
      }
    }
    _Pragma("clang loop unroll(full)")
    for (uint o = 0; o < 4; ++o) {
      if (part == 0) {
        qt[o] = tile[o];
      } else {
        simdgroup_store(tile[o], (part == 1 ? K : V) + simd * 8 * 32 + o * 8, 32, ulong2(0), false);
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint token0 = simd * 8u;
  simdgroup_matrix<half, 8, 8> weights[8];
  _Pragma("clang loop unroll(full)")
  for (uint j = 0; j < 8; ++j) {
    simdgroup_matrix<float, 8, 8> acc;
    acc.thread_elements()[0] = 0.0f;
    acc.thread_elements()[1] = 0.0f;
    _Pragma("clang loop unroll(full)")
    for (uint k = 0; k < 4; ++k) {
      simdgroup_matrix<half, 8, 8> right;
      simdgroup_load(right, K + j * 8 * 32 + k * 8, 32, ulong2(0), true);
      simdgroup_multiply_accumulate(acc, qt[k], right, acc);
    }
    half2 score = half2(half(acc.thread_elements()[0]), half(acc.thread_elements()[1])) +
                  bias2[((head * 64u + token0 + fragment_row) * 64u + j * 8u + fragment_col) / 2u];
    half2 affine = fma(score, half2(0.044921875h), half2(1.30078125h));
    affine = clamp(affine, half2(1.03125h), half2(1.5693359375h));
    half2 weight = as_type<half2>(ushort2((as_type<ushort2>(affine) << 5) ^ ushort2(0x8000u)));
    weights[j].thread_elements()[0] = weight.x;
    weights[j].thread_elements()[1] = weight.y;
  }

  half total = 0.0h;
  _Pragma("clang loop unroll(full)")
  for (uint pass = 0; pass < 2; ++pass) {
    _Pragma("clang loop unroll(full)")
    for (uint j = 0; j < 4; ++j) {
      simdgroup_store(weights[pass * 4 + j], row_scratch + j * 8, 32, ulong2(0), false);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    const threadgroup half4* row4 = (const threadgroup half4*)(row_scratch + fragment_row * 32);
    _Pragma("clang loop unroll(full)")
    for (uint i = 0; i < 8; ++i) {
      const half4 w = row4[i];
      total += w.x;
      total += w.y;
      total += w.z;
      total += w.w;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }

  const half reciprocal = half(1.0f / float(total));
  _Pragma("clang loop unroll(full)")
  for (uint j = 0; j < 8; ++j) {
    _Pragma("clang loop unroll(full)")
    for (uint e = 0; e < 2; ++e) {
      weights[j].thread_elements()[e] =
          nr_fast_e4m3_nonnegative(weights[j].thread_elements()[e] * reciprocal);
    }
  }

  _Pragma("clang loop unroll(full)")
  for (uint o = 0; o < 4; ++o) {
    simdgroup_matrix<float, 8, 8> acc;
    acc.thread_elements()[0] = 0.0f;
    acc.thread_elements()[1] = 0.0f;
    _Pragma("clang loop unroll(full)")
    for (uint k = 0; k < 8; ++k) {
      simdgroup_matrix<half, 8, 8> right;
      simdgroup_load(right, V + k * 8 * 32 + o * 8, 32, ulong2(0), false);
      simdgroup_multiply_accumulate(acc, weights[k], right, acc);
    }
    if (inside) {
      output2[(pixel * channels + head * 32u + o * 8u + fragment_col) / 2u] = half2(
          nr_fast_e4m3_no_nan(half(acc.thread_elements()[0])),
          nr_fast_e4m3_no_nan(half(acc.thread_elements()[1])));
    }
  }
}

// Fused register-resident Branched FFN + cosine residual + E4M3 for 64ch & 128ch.
template <uint Channels, uint Simdgroups>
inline void nr_fast_branched_ffn_impl(
    const device half* input,
    const device half* expansion,
    const device half* projection,
    const device half* out_proj,
    const device half* cosine,
    device half* output,
    uint row_count,
    uint tgid,
    uint simd,
    ushort lane) {
  constexpr uint tiles = Channels / 8u;
  constexpr uint groups = Channels / 32u;
  constexpr uint hidden_stride = Channels * 4u;
  const ushort fragment_row = ((lane / 4) & 4) + ((lane / 2) % 4);
  const ushort fragment_col = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
  const uint row = (tgid * Simdgroups + simd) * 8u + fragment_row;
  const bool inside = row < row_count;
  const device half2* input2 = (const device half2*)input;
  device half2* output2 = (device half2*)output;
  const device half2* cosine2 = (const device half2*)cosine;

  simdgroup_matrix<half, 8, 8> xt[tiles];
  _Pragma("clang loop unroll(full)")
  for (uint k = 0; k < tiles; ++k) {
    const half2 v = inside ? input2[(row * Channels + k * 8u + fragment_col) / 2u] : half2(0.0h);
    xt[k].thread_elements()[0] = v.x;
    xt[k].thread_elements()[1] = v.y;
  }

  for (uint g = 0; g < groups; ++g) {
    simdgroup_matrix<float, 8, 8> branch[4];
    _Pragma("clang loop unroll(full)")
    for (uint o = 0; o < 4; ++o) {
      branch[o].thread_elements()[0] = 0.0f;
      branch[o].thread_elements()[1] = 0.0f;
    }
    for (uint b = 0; b < 4; ++b) {
      _Pragma("clang loop unroll(full)")
      for (uint n = 0; n < 4; ++n) {
        simdgroup_matrix<float, 8, 8> acc;
        acc.thread_elements()[0] = 0.0f;
        acc.thread_elements()[1] = 0.0f;
        const device half* col = expansion + (g * 4u + b) * 32u + n * 8u;
        _Pragma("clang loop unroll(full)")
        for (uint k = 0; k < tiles; ++k) {
          simdgroup_matrix<half, 8, 8> right;
          simdgroup_load(right, col + k * 8u * hidden_stride, hidden_stride, ulong2(0), false);
          simdgroup_multiply_accumulate(acc, xt[k], right, acc);
        }
        simdgroup_matrix<half, 8, 8> hidden;
        _Pragma("clang loop unroll(full)")
        for (uint e = 0; e < 2; ++e) {
          hidden.thread_elements()[e] = nr_fast_e4m3(nr_fast_quad_gate(half(acc.thread_elements()[e])));
        }
        const device half* rows = projection + g * 128u * 32u + (b * 32u + n * 8u) * 32u;
        _Pragma("clang loop unroll(full)")
        for (uint o = 0; o < 4; ++o) {
          simdgroup_matrix<half, 8, 8> right;
          simdgroup_load(right, rows + o * 8u, 32, ulong2(0), false);
          simdgroup_multiply_accumulate(branch[o], hidden, right, branch[o]);
        }
      }
    }
    _Pragma("clang loop unroll(full)")
    for (uint kk = 0; kk < 4; ++kk) {
      const half2 pub = half2(
          nr_fast_e4m3(half(branch[kk].thread_elements()[0])),
          nr_fast_e4m3(half(branch[kk].thread_elements()[1])));
      if (inside) {
        output2[(row * Channels + g * 32u + kk * 8u + fragment_col) / 2u] = pub;
      }
    }
  }

  constexpr uint chunk_tiles = 8u;
  _Pragma("clang loop unroll(full)")
  for (uint chunk = 0; chunk < tiles / chunk_tiles; ++chunk) {
    simdgroup_matrix<float, 8, 8> out[chunk_tiles];
    _Pragma("clang loop unroll(full)")
    for (uint o = 0; o < chunk_tiles; ++o) {
      out[o].thread_elements()[0] = 0.0f;
      out[o].thread_elements()[1] = 0.0f;
    }
    for (uint kk = 0; kk < tiles; ++kk) {
      const half2 parked = inside ? output2[(row * Channels + kk * 8u + fragment_col) / 2u] : half2(0.0h);
      simdgroup_matrix<half, 8, 8> pub;
      pub.thread_elements()[0] = parked.x;
      pub.thread_elements()[1] = parked.y;
      const device half* rows = out_proj + kk * 8u * Channels + chunk * chunk_tiles * 8u;
      _Pragma("clang loop unroll(full)")
      for (uint o = 0; o < chunk_tiles; ++o) {
        simdgroup_matrix<half, 8, 8> right;
        simdgroup_load(right, rows + o * 8u, Channels, ulong2(0), false);
        simdgroup_multiply_accumulate(out[o], pub, right, out[o]);
      }
    }
    _Pragma("clang loop unroll(full)")
    for (uint o = 0; o < chunk_tiles; ++o) {
      const uint tile = chunk * chunk_tiles + o;
      const half2 sc = cosine2[(tile * 8u + fragment_col) / 2u];
      _Pragma("clang loop unroll(full)")
      for (uint e = 0; e < 2; ++e) {
        const half bval = half(out[o].thread_elements()[e]);
        const half skip = xt[tile].thread_elements()[e] * sc[e];
        xt[tile].thread_elements()[e] = nr_fast_e4m3(bval + skip);
      }
    }
  }

  if (inside) {
    _Pragma("clang loop unroll(full)")
    for (uint o = 0; o < tiles; ++o) {
      output2[(row * Channels + o * 8u + fragment_col) / 2u] =
          half2(xt[o].thread_elements()[0], xt[o].thread_elements()[1]);
    }
  }
}

kernel void nr_fast_branched_ffn_64(
    const device half* input [[buffer(0)]],
    const device half* expansion [[buffer(1)]],
    const device half* projection [[buffer(2)]],
    const device half* out_proj [[buffer(3)]],
    const device half* cosine [[buffer(4)]],
    device half* output [[buffer(5)]],
    constant CountParams& p [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane_in [[thread_index_in_simdgroup]]) {
  const ushort lane = ushort(lane_in);
  nr_fast_branched_ffn_impl<64, 4>(input, expansion, projection, out_proj, cosine, output, p.count, tgid, simd, lane);
}

kernel void nr_fast_branched_ffn_128(
    const device half* input [[buffer(0)]],
    const device half* expansion [[buffer(1)]],
    const device half* projection [[buffer(2)]],
    const device half* out_proj [[buffer(3)]],
    const device half* cosine [[buffer(4)]],
    device half* output [[buffer(5)]],
    constant CountParams& p [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane_in [[thread_index_in_simdgroup]]) {
  const ushort lane = ushort(lane_in);
  nr_fast_branched_ffn_impl<128, 4>(input, expansion, projection, out_proj, cosine, output, p.count, tgid, simd, lane);
}

// Fused 64ch 2-head Window Attention Block (QKV + Attention + OutProj + Cosine Residual).
kernel void nr_fast_window_attn_block_64(
    const device half* input [[buffer(0)]],
    const device half* qkv_weight [[buffer(1)]],
    const device half* attn_scale [[buffer(2)]],
    const device half* attn_bias [[buffer(3)]],
    const device half* proj_weight [[buffer(4)]],
    const device half* cosine [[buffer(5)]],
    device half* output [[buffer(6)]],
    constant WindowMHParams& p [[buffer(7)]],
    uint window_index [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane_in [[thread_index_in_simdgroup]]) {
  const ushort lane = ushort(lane_in);
  constexpr uint channels = 64u;
  constexpr uint tiles = 8u;
  constexpr uint heads = 2u;
  constexpr uint qkv_stride = 192u;

  threadgroup half4 arena4[6144 / 4];
  threadgroup half* K = (threadgroup half*)arena4;
  threadgroup half* V = K + 2048;
  threadgroup half* row_scratch = K + 4096 + simd * 256;

  const ushort fragment_row = ((lane / 4) & 4) + ((lane / 2) % 4);
  const ushort fragment_col = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
  const uint windows_x = (p.width + p.pad_left + 7u) / 8u;
  const uint wy = window_index / windows_x;
  const uint wx = window_index % windows_x;
  const int y = int(wy * 8u) - int(p.pad_top) + int(simd);
  const int x = int(wx * 8u) - int(p.pad_left) + int(fragment_row);
  const bool inside = (y >= 0 && y < int(p.height) && x >= 0 && x < int(p.width));
  const uint slot = (inside ? (uint(y) * p.width + uint(x)) : 0u) * channels + fragment_col;

  const device half2* input2 = (const device half2*)input;
  device half2* output2 = (device half2*)output;
  const device half2* bias2 = (const device half2*)attn_bias;
  const device half2* cosine2 = (const device half2*)cosine;

  simdgroup_matrix<half, 8, 8> yt[tiles];
  _Pragma("clang loop unroll(full)")
  for (uint k = 0; k < tiles; ++k) {
    const half2 v = inside ? input2[(slot + k * 8u) / 2u] : half2(0.0h);
    yt[k].thread_elements()[0] = v.x;
    yt[k].thread_elements()[1] = v.y;
  }

  const uint token0 = simd * 8u;
  for (uint head = 0; head < heads; ++head) {
    const half scale = attn_scale[head];
    simdgroup_matrix<half, 8, 8> qt[4];
    for (uint part = 0; part < 3; ++part) {
      simdgroup_matrix<half, 8, 8> result[4];
      const device half* columns = qkv_weight + part * channels + head * 32u;
      _Pragma("clang loop unroll(full)")
      for (uint o = 0; o < 4; ++o) {
        simdgroup_matrix<float, 8, 8> acc;
        acc.thread_elements()[0] = 0.0f;
        acc.thread_elements()[1] = 0.0f;
        _Pragma("clang loop unroll(full)")
        for (uint k = 0; k < tiles; ++k) {
          simdgroup_matrix<half, 8, 8> right;
          simdgroup_load(right, columns + k * 8u * qkv_stride + o * 8u, qkv_stride, ulong2(0), false);
          simdgroup_multiply_accumulate(acc, yt[k], right, acc);
        }
        result[o].thread_elements()[0] = half(acc.thread_elements()[0]);
        result[o].thread_elements()[1] = half(acc.thread_elements()[1]);
      }
      if (part < 2) {
        half partial[2];
        _Pragma("clang loop unroll(full)")
        for (uint e = 0; e < 2; ++e) {
          const half v0 = result[0].thread_elements()[e];
          const half v1 = result[1].thread_elements()[e];
          const half v2 = result[2].thread_elements()[e];
          const half v3 = result[3].thread_elements()[e];
          partial[e] = fma(v1, v1, v0 * v0) + fma(v3, v3, v2 * v2);
        }
        threadgroup half2* partials = (threadgroup half2*)(V + simd * 8 * 32) + fragment_row * 4;
        partials[((lane >> 2) & 2) | (lane & 1)] = half2(partial[0], partial[1]);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        const half2 p0 = partials[0];
        const half2 p1 = partials[1];
        const half2 p2 = partials[2];
        const half2 p3 = partials[3];
        simdgroup_barrier(mem_flags::mem_threadgroup);
        half xor_one[2];
        _Pragma("clang loop unroll(full)")
        for (uint e = 0; e < 2; ++e) {
          xor_one[e] = (p0[e] + p2[e]) + (p1[e] + p3[e]);
        }
        half norm = max(xor_one[0] + xor_one[1], half(0.00006198883056640625h));
        const half inv_norm = half(metal::fast::rsqrt(float(norm)));
        _Pragma("clang loop unroll(full)")
        for (uint o = 0; o < 4; ++o) {
          _Pragma("clang loop unroll(full)")
          for (uint e = 0; e < 2; ++e) {
            half normalized = result[o].thread_elements()[e] * inv_norm;
            if (part == 0) {
              normalized *= scale;
            }
            result[o].thread_elements()[e] = nr_fast_e4m3(normalized);
          }
          if (part == 0) {
            qt[o] = result[o];
          } else {
            simdgroup_store(result[o], K + simd * 8 * 32 + o * 8, 32, ulong2(0), false);
          }
        }
      } else {
        _Pragma("clang loop unroll(full)")
        for (uint o = 0; o < 4; ++o) {
          _Pragma("clang loop unroll(full)")
          for (uint e = 0; e < 2; ++e) {
            result[o].thread_elements()[e] = nr_fast_e4m3(result[o].thread_elements()[e]);
          }
          simdgroup_store(result[o], V + simd * 8 * 32 + o * 8, 32, ulong2(0), false);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_matrix<half, 8, 8> weights[8];
    _Pragma("clang loop unroll(full)")
    for (uint j = 0; j < 8; ++j) {
      simdgroup_matrix<float, 8, 8> acc;
      acc.thread_elements()[0] = 0.0f;
      acc.thread_elements()[1] = 0.0f;
      _Pragma("clang loop unroll(full)")
      for (uint k = 0; k < 4; ++k) {
        simdgroup_matrix<half, 8, 8> right;
        simdgroup_load(right, K + j * 8 * 32 + k * 8, 32, ulong2(0), true);
        simdgroup_multiply_accumulate(acc, qt[k], right, acc);
      }
      half2 score = half2(half(acc.thread_elements()[0]), half(acc.thread_elements()[1])) +
                    bias2[((head * 64u + token0 + fragment_row) * 64u + j * 8u + fragment_col) / 2u];
      half2 affine = fma(score, half2(0.044921875h), half2(1.30078125h));
      affine = clamp(affine, half2(1.03125h), half2(1.5693359375h));
      half2 weight = as_type<half2>(ushort2((as_type<ushort2>(affine) << 5) ^ ushort2(0x8000u)));
      weights[j].thread_elements()[0] = weight.x;
      weights[j].thread_elements()[1] = weight.y;
    }

    half total = 0.0h;
    _Pragma("clang loop unroll(full)")
    for (uint pass = 0; pass < 2; ++pass) {
      _Pragma("clang loop unroll(full)")
      for (uint j = 0; j < 4; ++j) {
        simdgroup_store(weights[pass * 4 + j], row_scratch + j * 8, 32, ulong2(0), false);
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
      const threadgroup half4* row4 = (const threadgroup half4*)(row_scratch + fragment_row * 32);
      _Pragma("clang loop unroll(full)")
      for (uint i = 0; i < 8; ++i) {
        const half4 w = row4[i];
        total += w.x;
        total += w.y;
        total += w.z;
        total += w.w;
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    const half reciprocal = half(1.0f / float(total));
    _Pragma("clang loop unroll(full)")
    for (uint j = 0; j < 8; ++j) {
      _Pragma("clang loop unroll(full)")
      for (uint e = 0; e < 2; ++e) {
        weights[j].thread_elements()[e] =
            nr_fast_e4m3_nonnegative(weights[j].thread_elements()[e] * reciprocal);
      }
    }

    _Pragma("clang loop unroll(full)")
    for (uint o = 0; o < 4; ++o) {
      simdgroup_matrix<float, 8, 8> acc;
      acc.thread_elements()[0] = 0.0f;
      acc.thread_elements()[1] = 0.0f;
      _Pragma("clang loop unroll(full)")
      for (uint k = 0; k < 8; ++k) {
        simdgroup_matrix<half, 8, 8> right;
        simdgroup_load(right, V + k * 8 * 32 + o * 8, 32, ulong2(0), false);
        simdgroup_multiply_accumulate(acc, weights[k], right, acc);
      }
      if (inside) {
        output2[(slot + head * 32u + o * 8u) / 2u] = half2(
            nr_fast_e4m3_no_nan(half(acc.thread_elements()[0])),
            nr_fast_e4m3_no_nan(half(acc.thread_elements()[1])));
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  simdgroup_matrix<float, 8, 8> out[tiles];
  _Pragma("clang loop unroll(full)")
  for (uint o = 0; o < tiles; ++o) {
    out[o].thread_elements()[0] = 0.0f;
    out[o].thread_elements()[1] = 0.0f;
  }
  for (uint kk = 0; kk < tiles; ++kk) {
    const half2 parked = inside ? output2[(slot + kk * 8u) / 2u] : half2(0.0h);
    simdgroup_matrix<half, 8, 8> attended;
    attended.thread_elements()[0] = parked.x;
    attended.thread_elements()[1] = parked.y;
    const device half* rows = proj_weight + kk * 8u * channels;
    _Pragma("clang loop unroll(full)")
    for (uint o = 0; o < tiles; ++o) {
      simdgroup_matrix<half, 8, 8> right;
      simdgroup_load(right, rows + o * 8u, channels, ulong2(0), false);
      simdgroup_multiply_accumulate(out[o], attended, right, out[o]);
    }
  }
  if (inside) {
    _Pragma("clang loop unroll(full)")
    for (uint o = 0; o < tiles; ++o) {
      const half2 sc = cosine2[(o * 8u + fragment_col) / 2u];
      half2 val = half2(
          half(out[o].thread_elements()[0]) + yt[o].thread_elements()[0] * sc.x,
          half(out[o].thread_elements()[1]) + yt[o].thread_elements()[1] * sc.y);
      if (p.publish != 0u) {
        val = nr_fast_e4m3_vec2(val);
      }
      output2[(slot + o * 8u) / 2u] = val;
    }
  }
}

// ============================================================================
// 5. High-Occupancy Cooperative Tiled simdgroup_matrix GEMM & Grouped GEMM
// ============================================================================

enum GemmEpilogue : uint {
  GEMM_EPILOGUE_NONE = 0u,
  GEMM_EPILOGUE_E4M3 = 1u,
  GEMM_EPILOGUE_QUAD_GATE = 2u,
  GEMM_EPILOGUE_QUAD_GATE_E4M3 = 3u,
  GEMM_EPILOGUE_COSINE_RESIDUAL = 4u,
  GEMM_EPILOGUE_COSINE_RESIDUAL_E4M3 = 5u,
};

struct GemmParams {
  uint M;
  uint N;
  uint K;
  uint epilogue;
};

// BM=32, BN=64, BK=32, 128 threads (4 simdgroups = 2x2), ~7 KB threadgroup RAM.
kernel void nr_fast_gemm_f16(
    const device half* A [[buffer(0)]],
    const device half* B [[buffer(1)]],
    const device half* skip [[buffer(2)]],
    const device half* cosine [[buffer(3)]],
    device half* C [[buffer(4)]],
    constant GemmParams& p [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  alignas(16) threadgroup half As[32][40];
  alignas(16) threadgroup half Bs[32][72];

  const uint block_m = tgid.y * 32u;
  const uint block_n = tgid.x * 64u;
  if (block_m >= p.M || block_n >= p.N) return;

  const uint sg_m = simd >> 1;
  const uint sg_n = simd & 1u;

  simdgroup_float8x8 acc[2][4];
  _Pragma("clang loop unroll(full)")
  for (uint mi = 0; mi < 2; ++mi) {
    _Pragma("clang loop unroll(full)")
    for (uint ni = 0; ni < 4; ++ni) {
      acc[mi][ni] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
  }

  const uint ar = tid >> 3;           // 0..15
  const uint ac4 = (tid & 7u) << 2;   // 0..28
  const uint br = tid >> 4;           // 0..7
  const uint bc4 = (tid & 15u) << 2;  // 0..60

  const bool fast_tile = (block_m + 32u <= p.M) && (block_n + 64u <= p.N) && ((p.K & 31u) == 0u);

  if (fast_tile) {
    for (uint k0 = 0; k0 < p.K; k0 += 32u) {
      *(threadgroup half4*)(&As[ar][ac4]) =
          *(const device half4*)(A + (block_m + ar) * p.K + k0 + ac4);
      *(threadgroup half4*)(&As[ar + 16u][ac4]) =
          *(const device half4*)(A + (block_m + ar + 16u) * p.K + k0 + ac4);

      _Pragma("clang loop unroll(full)")
      for (uint s = 0; s < 32u; s += 8u) {
        *(threadgroup half4*)(&Bs[br + s][bc4]) =
            *(const device half4*)(B + (k0 + br + s) * p.N + block_n + bc4);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      _Pragma("clang loop unroll(full)")
      for (uint kk = 0; kk < 32u; kk += 8u) {
        simdgroup_half8x8 a_frag[2];
        simdgroup_half8x8 b_frag[4];
        simdgroup_load(a_frag[0], &As[sg_m * 16u + 0u][kk], 40, ulong2(0), false);
        simdgroup_load(a_frag[1], &As[sg_m * 16u + 8u][kk], 40, ulong2(0), false);
        simdgroup_load(b_frag[0], &Bs[kk][sg_n * 32u + 0u], 72, ulong2(0), false);
        simdgroup_load(b_frag[1], &Bs[kk][sg_n * 32u + 8u], 72, ulong2(0), false);
        simdgroup_load(b_frag[2], &Bs[kk][sg_n * 32u + 16u], 72, ulong2(0), false);
        simdgroup_load(b_frag[3], &Bs[kk][sg_n * 32u + 24u], 72, ulong2(0), false);
        _Pragma("clang loop unroll(full)")
        for (uint mi = 0; mi < 2; ++mi) {
          _Pragma("clang loop unroll(full)")
          for (uint ni = 0; ni < 4; ++ni) {
            simdgroup_multiply_accumulate(acc[mi][ni], a_frag[mi], b_frag[ni], acc[mi][ni]);
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  } else {
    for (uint k0 = 0; k0 < p.K; k0 += 32u) {
      _Pragma("clang loop unroll(full)")
      for (uint half_tile = 0; half_tile < 32u; half_tile += 16u) {
        const uint gm = block_m + ar + half_tile;
        half4 av = half4(0.0h);
        if (gm < p.M) {
          if (k0 + ac4 + 4u <= p.K) {
            av = *(const device half4*)(A + gm * p.K + k0 + ac4);
          } else {
            for (uint i = 0; i < 4; ++i) {
              if (k0 + ac4 + i < p.K) av[i] = A[gm * p.K + k0 + ac4 + i];
            }
          }
        }
        *(threadgroup half4*)(&As[ar + half_tile][ac4]) = av;
      }

      _Pragma("clang loop unroll(full)")
      for (uint s = 0; s < 32u; s += 8u) {
        const uint gk = k0 + br + s;
        const uint gn = block_n + bc4;
        half4 bv = half4(0.0h);
        if (gk < p.K) {
          if (gn + 4u <= p.N) {
            bv = *(const device half4*)(B + gk * p.N + gn);
          } else {
            for (uint i = 0; i < 4; ++i) {
              if (gn + i < p.N) bv[i] = B[gk * p.N + gn + i];
            }
          }
        }
        *(threadgroup half4*)(&Bs[br + s][bc4]) = bv;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      _Pragma("clang loop unroll(full)")
      for (uint kk = 0; kk < 32u; kk += 8u) {
        simdgroup_half8x8 a_frag[2];
        simdgroup_half8x8 b_frag[4];
        simdgroup_load(a_frag[0], &As[sg_m * 16u + 0u][kk], 40, ulong2(0), false);
        simdgroup_load(a_frag[1], &As[sg_m * 16u + 8u][kk], 40, ulong2(0), false);
        simdgroup_load(b_frag[0], &Bs[kk][sg_n * 32u + 0u], 72, ulong2(0), false);
        simdgroup_load(b_frag[1], &Bs[kk][sg_n * 32u + 8u], 72, ulong2(0), false);
        simdgroup_load(b_frag[2], &Bs[kk][sg_n * 32u + 16u], 72, ulong2(0), false);
        simdgroup_load(b_frag[3], &Bs[kk][sg_n * 32u + 24u], 72, ulong2(0), false);
        _Pragma("clang loop unroll(full)")
        for (uint mi = 0; mi < 2; ++mi) {
          _Pragma("clang loop unroll(full)")
          for (uint ni = 0; ni < 4; ++ni) {
            simdgroup_multiply_accumulate(acc[mi][ni], a_frag[mi], b_frag[ni], acc[mi][ni]);
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  // Stage rounded half results into Bs[32][72] for coalesced half4 epilogue writes.
  _Pragma("clang loop unroll(full)")
  for (uint mi = 0; mi < 2; ++mi) {
    _Pragma("clang loop unroll(full)")
    for (uint ni = 0; ni < 4; ++ni) {
      simdgroup_half8x8 out_frag;
      out_frag.thread_elements()[0] = half(acc[mi][ni].thread_elements()[0]);
      out_frag.thread_elements()[1] = half(acc[mi][ni].thread_elements()[1]);
      simdgroup_store(out_frag, &Bs[sg_m * 16u + mi * 8u][sg_n * 32u + ni * 8u], 72, ulong2(0), false);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint gn = block_n + bc4;
  _Pragma("clang loop unroll(full)")
  for (uint s = 0; s < 32u; s += 8u) {
    const uint gm = block_m + br + s;
    if (gm < p.M && gn + 4u <= p.N) {
      half4 val = *(const threadgroup half4*)(&Bs[br + s][bc4]);
      const uint idx = gm * p.N + gn;
      if (p.epilogue == GEMM_EPILOGUE_E4M3) {
        val = nr_fast_e4m3_vec4(val);
      } else if (p.epilogue == GEMM_EPILOGUE_QUAD_GATE) {
        val = nr_fast_quad_gate_vec4(val);
      } else if (p.epilogue == GEMM_EPILOGUE_QUAD_GATE_E4M3) {
        val = nr_fast_e4m3_vec4(nr_fast_quad_gate_vec4(val));
      } else if (p.epilogue == GEMM_EPILOGUE_COSINE_RESIDUAL) {
        half4 sk = *(const device half4*)(skip + idx);
        half4 cs = *(const device half4*)(cosine + gn);
        val = val + (sk * cs);
      } else if (p.epilogue == GEMM_EPILOGUE_COSINE_RESIDUAL_E4M3) {
        half4 sk = *(const device half4*)(skip + idx);
        half4 cs = *(const device half4*)(cosine + gn);
        val = nr_fast_e4m3_vec4(val + (sk * cs));
      }
      *(device half4*)(C + idx) = val;
    } else if (gm < p.M && gn < p.N) {
      for (uint i = 0; i < 4u && (gn + i) < p.N; ++i) {
        half v = Bs[br + s][bc4 + i];
        const uint idx = gm * p.N + gn + i;
        if (p.epilogue == GEMM_EPILOGUE_E4M3) {
          v = nr_fast_e4m3(v);
        } else if (p.epilogue == GEMM_EPILOGUE_QUAD_GATE) {
          v = nr_fast_quad_gate(v);
        } else if (p.epilogue == GEMM_EPILOGUE_QUAD_GATE_E4M3) {
          v = nr_fast_e4m3(nr_fast_quad_gate(v));
        } else if (p.epilogue == GEMM_EPILOGUE_COSINE_RESIDUAL) {
          v = v + (skip[idx] * cosine[gn + i]);
        } else if (p.epilogue == GEMM_EPILOGUE_COSINE_RESIDUAL_E4M3) {
          v = nr_fast_e4m3(v + (skip[idx] * cosine[gn + i]));
        }
        C[idx] = v;
      }
    }
  }
}

// BM=64, BN=64, BK=32, 128 threads (4 simdgroups = 2x2, each 32x32), ~9.7 KB threadgroup RAM.
// Halves B-matrix DRAM bandwidth for high-row-count stages (M >= 1024).
kernel void nr_fast_gemm_f16_bm64(
    const device half* A [[buffer(0)]],
    const device half* B [[buffer(1)]],
    const device half* skip [[buffer(2)]],
    const device half* cosine [[buffer(3)]],
    device half* C [[buffer(4)]],
    constant GemmParams& p [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  alignas(16) threadgroup half As[64][40];
  alignas(16) threadgroup half Bs[32][72];

  const uint block_m = tgid.y * 64u;
  const uint block_n = tgid.x * 64u;
  if (block_m >= p.M || block_n >= p.N) return;

  const uint sg_m = simd >> 1;
  const uint sg_n = simd & 1u;

  simdgroup_float8x8 acc[4][4];
  _Pragma("clang loop unroll(full)")
  for (uint mi = 0; mi < 4; ++mi) {
    _Pragma("clang loop unroll(full)")
    for (uint ni = 0; ni < 4; ++ni) {
      acc[mi][ni] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
  }

  const uint ar = tid >> 3;           // 0..15
  const uint ac4 = (tid & 7u) << 2;   // 0..28
  const uint br = tid >> 4;           // 0..7
  const uint bc4 = (tid & 15u) << 2;  // 0..60

  const bool fast_tile = (block_m + 64u <= p.M) && (block_n + 64u <= p.N) && ((p.K & 31u) == 0u);

  if (fast_tile) {
    for (uint k0 = 0; k0 < p.K; k0 += 32u) {
      _Pragma("clang loop unroll(full)")
      for (uint s = 0; s < 64u; s += 16u) {
        *(threadgroup half4*)(&As[ar + s][ac4]) =
            *(const device half4*)(A + (block_m + ar + s) * p.K + k0 + ac4);
      }
      _Pragma("clang loop unroll(full)")
      for (uint s = 0; s < 32u; s += 8u) {
        *(threadgroup half4*)(&Bs[br + s][bc4]) =
            *(const device half4*)(B + (k0 + br + s) * p.N + block_n + bc4);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      _Pragma("clang loop unroll(full)")
      for (uint kk = 0; kk < 32u; kk += 8u) {
        simdgroup_half8x8 a_frag[4];
        simdgroup_half8x8 b_frag[4];
        _Pragma("clang loop unroll(full)")
        for (uint mi = 0; mi < 4; ++mi) {
          simdgroup_load(a_frag[mi], &As[sg_m * 32u + mi * 8u][kk], 40, ulong2(0), false);
        }
        _Pragma("clang loop unroll(full)")
        for (uint ni = 0; ni < 4; ++ni) {
          simdgroup_load(b_frag[ni], &Bs[kk][sg_n * 32u + ni * 8u], 72, ulong2(0), false);
        }
        _Pragma("clang loop unroll(full)")
        for (uint mi = 0; mi < 4; ++mi) {
          _Pragma("clang loop unroll(full)")
          for (uint ni = 0; ni < 4; ++ni) {
            simdgroup_multiply_accumulate(acc[mi][ni], a_frag[mi], b_frag[ni], acc[mi][ni]);
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  } else {
    for (uint k0 = 0; k0 < p.K; k0 += 32u) {
      _Pragma("clang loop unroll(full)")
      for (uint s = 0; s < 64u; s += 16u) {
        const uint gm = block_m + ar + s;
        half4 av = half4(0.0h);
        if (gm < p.M) {
          if (k0 + ac4 + 4u <= p.K) {
            av = *(const device half4*)(A + gm * p.K + k0 + ac4);
          } else {
            for (uint i = 0; i < 4u; ++i) {
              if (k0 + ac4 + i < p.K) av[i] = A[gm * p.K + k0 + ac4 + i];
            }
          }
        }
        *(threadgroup half4*)(&As[ar + s][ac4]) = av;
      }
      _Pragma("clang loop unroll(full)")
      for (uint s = 0; s < 32u; s += 8u) {
        const uint gk = k0 + br + s;
        const uint gn = block_n + bc4;
        half4 bv = half4(0.0h);
        if (gk < p.K) {
          if (gn + 4u <= p.N) {
            bv = *(const device half4*)(B + gk * p.N + gn);
          } else {
            for (uint i = 0; i < 4u; ++i) {
              if (gn + i < p.N) bv[i] = B[gk * p.N + gn + i];
            }
          }
        }
        *(threadgroup half4*)(&Bs[br + s][bc4]) = bv;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      _Pragma("clang loop unroll(full)")
      for (uint kk = 0; kk < 32u; kk += 8u) {
        simdgroup_half8x8 a_frag[4];
        simdgroup_half8x8 b_frag[4];
        _Pragma("clang loop unroll(full)")
        for (uint mi = 0; mi < 4; ++mi) {
          simdgroup_load(a_frag[mi], &As[sg_m * 32u + mi * 8u][kk], 40, ulong2(0), false);
        }
        _Pragma("clang loop unroll(full)")
        for (uint ni = 0; ni < 4; ++ni) {
          simdgroup_load(b_frag[ni], &Bs[kk][sg_n * 32u + ni * 8u], 72, ulong2(0), false);
        }
        _Pragma("clang loop unroll(full)")
        for (uint mi = 0; mi < 4; ++mi) {
          _Pragma("clang loop unroll(full)")
          for (uint ni = 0; ni < 4; ++ni) {
            simdgroup_multiply_accumulate(acc[mi][ni], a_frag[mi], b_frag[ni], acc[mi][ni]);
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  const uint gn = block_n + bc4;
  for (uint half_m = 0; half_m < 2u; ++half_m) {
    if (sg_m == half_m) {
      _Pragma("clang loop unroll(full)")
      for (uint mi = 0; mi < 4; ++mi) {
        _Pragma("clang loop unroll(full)")
        for (uint ni = 0; ni < 4; ++ni) {
          simdgroup_half8x8 out_frag;
          out_frag.thread_elements()[0] = half(acc[mi][ni].thread_elements()[0]);
          out_frag.thread_elements()[1] = half(acc[mi][ni].thread_elements()[1]);
          simdgroup_store(out_frag, &Bs[mi * 8u][sg_n * 32u + ni * 8u], 72, ulong2(0), false);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    _Pragma("clang loop unroll(full)")
    for (uint s = 0; s < 32u; s += 8u) {
      const uint gm = block_m + half_m * 32u + br + s;
      if (gm < p.M && gn + 4u <= p.N) {
        half4 val = *(const threadgroup half4*)(&Bs[br + s][bc4]);
        const uint idx = gm * p.N + gn;
        if (p.epilogue == GEMM_EPILOGUE_E4M3) {
          val = nr_fast_e4m3_vec4(val);
        } else if (p.epilogue == GEMM_EPILOGUE_QUAD_GATE) {
          val = nr_fast_quad_gate_vec4(val);
        } else if (p.epilogue == GEMM_EPILOGUE_QUAD_GATE_E4M3) {
          val = nr_fast_e4m3_vec4(nr_fast_quad_gate_vec4(val));
        } else if (p.epilogue == GEMM_EPILOGUE_COSINE_RESIDUAL) {
          half4 sk = *(const device half4*)(skip + idx);
          half4 cs = *(const device half4*)(cosine + gn);
          val = val + (sk * cs);
        } else if (p.epilogue == GEMM_EPILOGUE_COSINE_RESIDUAL_E4M3) {
          half4 sk = *(const device half4*)(skip + idx);
          half4 cs = *(const device half4*)(cosine + gn);
          val = nr_fast_e4m3_vec4(val + (sk * cs));
        }
        *(device half4*)(C + idx) = val;
      } else if (gm < p.M && gn < p.N) {
        for (uint i = 0; i < 4u && (gn + i) < p.N; ++i) {
          half v = Bs[br + s][bc4 + i];
          const uint idx = gm * p.N + gn + i;
          if (p.epilogue == GEMM_EPILOGUE_E4M3) {
            v = nr_fast_e4m3(v);
          } else if (p.epilogue == GEMM_EPILOGUE_QUAD_GATE) {
            v = nr_fast_quad_gate(v);
          } else if (p.epilogue == GEMM_EPILOGUE_QUAD_GATE_E4M3) {
            v = nr_fast_e4m3(nr_fast_quad_gate(v));
          } else if (p.epilogue == GEMM_EPILOGUE_COSINE_RESIDUAL) {
            v = v + (skip[idx] * cosine[gn + i]);
          } else if (p.epilogue == GEMM_EPILOGUE_COSINE_RESIDUAL_E4M3) {
            v = nr_fast_e4m3(v + (skip[idx] * cosine[gn + i]));
          }
          C[idx] = v;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

struct GroupedGemmParams {
  uint M;
  uint N;
  uint K;
  uint groups;
  uint epilogue;
};

// Zero-transpose strided grouped GEMM: A [M, G, K] x B [G, K, N] -> C [M, G, N]
// BM=32, BN=32, BK=32, 128 threads (4 simdgroups = 2x2), ~5 KB threadgroup RAM.
kernel void nr_fast_grouped_gemm_f16(
    const device half* A [[buffer(0)]],
    const device half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant GroupedGemmParams& p [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  alignas(16) threadgroup half As[32][40];
  alignas(16) threadgroup half Bs[32][40];

  const uint g = tgid.z;
  const uint block_m = tgid.y * 32u;
  const uint block_n = tgid.x * 32u;
  if (g >= p.groups || block_m >= p.M || block_n >= p.N) return;

  const uint lda = p.groups * p.K;
  const uint ldc = p.groups * p.N;
  const device half* Ag = A + g * p.K;
  const device half* Bg = B + g * p.K * p.N;
  device half* Cg = C + g * p.N;

  const uint sg_m = simd >> 1;
  const uint sg_n = simd & 1u;

  simdgroup_float8x8 acc[2][2];
  acc[0][0] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  acc[0][1] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  acc[1][0] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  acc[1][1] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

  const uint r = tid >> 3;          // 0..15
  const uint c4 = (tid & 7u) << 2;  // 0..28
  const bool fast_tile = (block_m + 32u <= p.M) && (block_n + 32u <= p.N) && ((p.K & 31u) == 0u);

  for (uint k0 = 0; k0 < p.K; k0 += 32u) {
    if (fast_tile) {
      *(threadgroup half4*)(&As[r][c4]) = *(const device half4*)(Ag + (block_m + r) * lda + k0 + c4);
      *(threadgroup half4*)(&As[r + 16u][c4]) =
          *(const device half4*)(Ag + (block_m + r + 16u) * lda + k0 + c4);
      *(threadgroup half4*)(&Bs[r][c4]) = *(const device half4*)(Bg + (k0 + r) * p.N + block_n + c4);
      *(threadgroup half4*)(&Bs[r + 16u][c4]) =
          *(const device half4*)(Bg + (k0 + r + 16u) * p.N + block_n + c4);
    } else {
      _Pragma("clang loop unroll(full)")
      for (uint ht = 0; ht < 32u; ht += 16u) {
        const uint gm = block_m + r + ht;
        half4 av = (gm < p.M && (k0 + c4 + 4u <= p.K))
                       ? *(const device half4*)(Ag + gm * lda + k0 + c4)
                       : half4(0.0h);
        *(threadgroup half4*)(&As[r + ht][c4]) = av;

        const uint gk = k0 + r + ht;
        half4 bv = (gk < p.K && (block_n + c4 + 4u <= p.N))
                       ? *(const device half4*)(Bg + gk * p.N + block_n + c4)
                       : half4(0.0h);
        *(threadgroup half4*)(&Bs[r + ht][c4]) = bv;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    _Pragma("clang loop unroll(full)")
    for (uint kk = 0; kk < 32u; kk += 8u) {
      simdgroup_half8x8 a_frag[2];
      simdgroup_half8x8 b_frag[2];
      simdgroup_load(a_frag[0], &As[sg_m * 16u + 0u][kk], 40, ulong2(0), false);
      simdgroup_load(a_frag[1], &As[sg_m * 16u + 8u][kk], 40, ulong2(0), false);
      simdgroup_load(b_frag[0], &Bs[kk][sg_n * 16u + 0u], 40, ulong2(0), false);
      simdgroup_load(b_frag[1], &Bs[kk][sg_n * 16u + 8u], 40, ulong2(0), false);
      simdgroup_multiply_accumulate(acc[0][0], a_frag[0], b_frag[0], acc[0][0]);
      simdgroup_multiply_accumulate(acc[0][1], a_frag[0], b_frag[1], acc[0][1]);
      simdgroup_multiply_accumulate(acc[1][0], a_frag[1], b_frag[0], acc[1][0]);
      simdgroup_multiply_accumulate(acc[1][1], a_frag[1], b_frag[1], acc[1][1]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  _Pragma("clang loop unroll(full)")
  for (uint mi = 0; mi < 2; ++mi) {
    _Pragma("clang loop unroll(full)")
    for (uint ni = 0; ni < 2; ++ni) {
      simdgroup_half8x8 out_frag;
      out_frag.thread_elements()[0] = half(acc[mi][ni].thread_elements()[0]);
      out_frag.thread_elements()[1] = half(acc[mi][ni].thread_elements()[1]);
      simdgroup_store(out_frag, &Bs[sg_m * 16u + mi * 8u][sg_n * 16u + ni * 8u], 40, ulong2(0), false);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint gn = block_n + c4;
  _Pragma("clang loop unroll(full)")
  for (uint ht = 0; ht < 32u; ht += 16u) {
    const uint gm = block_m + r + ht;
    if (gm < p.M && gn + 4u <= p.N) {
      half4 val = *(const threadgroup half4*)(&Bs[r + ht][c4]);
      if (p.epilogue == GEMM_EPILOGUE_E4M3) {
        val = nr_fast_e4m3_vec4(val);
      } else if (p.epilogue == GEMM_EPILOGUE_QUAD_GATE) {
        val = nr_fast_quad_gate_vec4(val);
      } else if (p.epilogue == GEMM_EPILOGUE_QUAD_GATE_E4M3) {
        val = nr_fast_e4m3_vec4(nr_fast_quad_gate_vec4(val));
      }
      *(device half4*)(Cg + gm * ldc + gn) = val;
    }
  }
}

// ============================================================================
// 6. Streamed & On-Chip Resident Global ViT Attention (32 Heads, 1024ch)
// ============================================================================

struct GlobalAttnParams {
  uint tokens;
  uint pitch;   // (tokens + 7) / 8 * 8
  uint tiles;   // pitch / 8
  uint groups;  // (tiles + 3) / 4
  uint heads;   // 32
};

// Publishes QKV [tokens, 3072] into lane-major layouts for resident v2 global attention.
kernel void nr_fast_global_publish_qkv(
    const device half* projected [[buffer(0)]],
    const device half* scale [[buffer(1)]],
    device half* query [[buffer(2)]],
    device half* key [[buffer(3)]],
    device half* value [[buffer(4)]],
    constant GlobalAttnParams& p [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]]) {
  const uint g = tgid.y;
  const uint blocks = p.heads / 8u;
  const uint part = tgid.z / blocks;
  const uint block = tgid.z % blocks;

  threadgroup half T[8 * 32 * 32];
  {
    const uint local = tid / 32u;
    const uint r = tid % 32u;
    const uint head = block * 8u + local;
    const uint token = g * 32u + r;
    threadgroup half* mine = T + tid * 32u;
    if (token >= p.tokens) {
      for (uint c = 0; c < 32u; ++c) mine[c] = 0.0h;
    } else {
      const device half4* in4 =
          (const device half4*)(projected + (token * 3u + part) * p.heads * 32u + head * 32u);
      half val[32];
      for (uint i = 0; i < 8u; ++i) {
        const half4 v = in4[i];
        val[4u * i] = v.x;
        val[4u * i + 1u] = v.y;
        val[4u * i + 2u] = v.z;
        val[4u * i + 3u] = v.w;
      }
      if (part == 2u) {
        for (uint c = 0; c < 32u; ++c) mine[c] = nr_fast_e4m3(val[c]);
      } else {
        half partial[4][2];
        for (uint lane = 0; lane < 4u; ++lane) {
          for (uint parity = 0; parity < 2u; ++parity) {
            uint ch = lane * 2u + parity;
            half first = fma(val[ch + 8u], val[ch + 8u], val[ch] * val[ch]);
            half second = fma(val[ch + 24u], val[ch + 24u], val[ch + 16u] * val[ch + 16u]);
            partial[lane][parity] = first + second;
          }
        }
        half xor_two[4][2];
        for (uint lane = 0; lane < 4u; ++lane) {
          for (uint parity = 0; parity < 2u; ++parity) {
            xor_two[lane][parity] = partial[lane][parity] + partial[lane ^ 2u][parity];
          }
        }
        half xor_one[4][2];
        for (uint lane = 0; lane < 4u; ++lane) {
          for (uint parity = 0; parity < 2u; ++parity) {
            xor_one[lane][parity] = xor_two[lane][parity] + xor_two[lane ^ 1u][parity];
          }
        }
        half norm = max(xor_one[0][0] + xor_one[0][1], half(0.00006198883056640625h));
        half inv = half(metal::fast::rsqrt(float(norm)));
        const half sc = scale[head];
        for (uint c = 0; c < 32u; ++c) {
          half norm_val = val[c] * inv;
          if (part == 0u) norm_val *= sc;
          mine[c] = nr_fast_e4m3(norm_val);
        }
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint n = 0; n < 4u; ++n) {
    const uint chunk = tid * 4u + n;
    const uint local = chunk / 128u;
    const uint within = chunk % 128u;
    const uint head = block * 8u + local;
    const threadgroup half* source = T + local * 1024u;
    const uint l = within % 32u;
    const uint fm = ((l >> 4) << 2) + ((l >> 1) & 3u);
    const uint fn = ((l >> 2) & 2u) * 2u + (l & 1u) * 2u;
    half v[8];
    device half4* target = nullptr;
    if (part == 0u) {
      const uint r = within / 4u;
      const uint quarter = within % 4u;
      if (g * 32u + r >= p.pitch) continue;
      for (uint k = 0; k < 8u; ++k) v[k] = source[r * 32u + quarter * 8u + k];
      target = (device half4*)(query + (head * p.pitch + g * 32u + r) * 32u + quarter * 8u);
    } else if (part == 1u) {
      const uint i = within / 32u;
      if (g * 4u + i >= p.tiles) continue;
      for (uint c = 0; c < 4u; ++c) {
        v[2u * c] = source[(i * 8u + fn) * 32u + c * 8u + fm];
        v[2u * c + 1u] = source[(i * 8u + fn + 1u) * 32u + c * 8u + fm];
      }
      target = (device half4*)(key + ((head * p.tiles + g * 4u + i) * 32u + l) * 8u);
    } else {
      const uint d = within / 32u;
      for (uint i = 0; i < 4u; ++i) {
        v[2u * i] = source[(i * 8u + fm) * 32u + d * 8u + fn];
        v[2u * i + 1u] = source[(i * 8u + fm) * 32u + d * 8u + fn + 1u];
      }
      target = (device half4*)(value + ((head * p.groups + g) * 4u + d) * 256u + l * 8u);
    }
    target[0] = half4(v[0], v[1], v[2], v[3]);
    target[1] = half4(v[4], v[5], v[6], v[7]);
  }
}

// 1-pass on-chip resident global attention for tokens <= 1024 (256 threads = 8 simdgroups).
kernel void nr_fast_global_attn_resident(
    const device half* query [[buffer(0)]],
    const device half* key [[buffer(1)]],
    const device half* value [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant GlobalAttnParams& p [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 tgid [[threadgroup_position_in_grid]]) {
  threadgroup half S[32 * 256];  // up to 32 groups = 1024 tokens (16 KB)
  threadgroup half R[8];

  const uint z = tgid.z;
  const uint head_base = z * p.pitch * 32u;
  const uint row0 = tgid.y * 8u;

  simdgroup_matrix<half, 8, 8> q[4];
  for (uint c = 0; c < 4u; ++c) {
    simdgroup_load(q[c], query + head_base + row0 * 32u + c * 8u, 32, ulong2(0), false);
  }

  for (uint g = simd; g < p.groups; g += 8u) {
    half weights[8];
    for (uint i = 0; i < 4u; ++i) {
      const uint t = g * 4u + i;
      simdgroup_matrix<float, 8, 8> acc;
      acc.thread_elements()[0] = 0.0f;
      acc.thread_elements()[1] = 0.0f;
      if (t < p.tiles) {
        const device half4* chunk = (const device half4*)(key + head_base + (t * 32u + lane) * 8u);
        const half4 a = chunk[0];
        const half4 b = chunk[1];
        simdgroup_matrix<half, 8, 8> k;
        k.thread_elements()[0] = a.x;
        k.thread_elements()[1] = a.y;
        simdgroup_multiply_accumulate(acc, q[0], k, acc);
        k.thread_elements()[0] = a.z;
        k.thread_elements()[1] = a.w;
        simdgroup_multiply_accumulate(acc, q[1], k, acc);
        k.thread_elements()[0] = b.x;
        k.thread_elements()[1] = b.y;
        simdgroup_multiply_accumulate(acc, q[2], k, acc);
        k.thread_elements()[0] = b.z;
        k.thread_elements()[1] = b.w;
        simdgroup_multiply_accumulate(acc, q[3], k, acc);
      }
      const half2 w = nr_fast_vendor_weights(
          clamp(half2(half(acc.thread_elements()[0]), half(acc.thread_elements()[1])), half2(-3.0h), half2(3.0h)));
      weights[2u * i] = w.x;
      weights[2u * i + 1u] = w.y;
    }
    threadgroup half4* stored = (threadgroup half4*)(S + (g * 32u + lane) * 8u);
    stored[0] = half4(weights[0], weights[1], weights[2], weights[3]);
    stored[1] = half4(weights[4], weights[5], weights[6], weights[7]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const threadgroup half4* S4 = (const threadgroup half4*)S;
  if (tid < 8u) {
    uint L[4];
    for (uint pr = 0; pr < 4u; ++pr) {
      L[pr] = ((tid >> 2) << 4) | ((pr >> 1) << 3) | ((tid & 3u) << 1) | (pr & 1u);
    }
    half total = 0.0h;
    const uint full_groups = p.tokens / 32u;
    for (uint g = 0; g < full_groups; ++g) {
      half4 c[4][2];
      for (uint pr = 0; pr < 4u; ++pr) {
        c[pr][0] = S4[(g * 32u + L[pr]) * 2u];
        c[pr][1] = S4[(g * 32u + L[pr]) * 2u + 1u];
      }
      for (uint pr = 0; pr < 4u; ++pr) { total += c[pr][0].x; total += c[pr][0].y; }
      for (uint pr = 0; pr < 4u; ++pr) { total += c[pr][0].z; total += c[pr][0].w; }
      for (uint pr = 0; pr < 4u; ++pr) { total += c[pr][1].x; total += c[pr][1].y; }
      for (uint pr = 0; pr < 4u; ++pr) { total += c[pr][1].z; total += c[pr][1].w; }
    }
    if ((p.tokens % 32u) != 0u) {
      const uint g = full_groups;
      half c[4][8];
      for (uint pr = 0; pr < 4u; ++pr) {
        const half4 a = S4[(g * 32u + L[pr]) * 2u];
        const half4 b = S4[(g * 32u + L[pr]) * 2u + 1u];
        c[pr][0] = a.x; c[pr][1] = a.y; c[pr][2] = a.z; c[pr][3] = a.w;
        c[pr][4] = b.x; c[pr][5] = b.y; c[pr][6] = b.z; c[pr][7] = b.w;
      }
      for (uint i = 0; i < 4u; ++i) {
        for (uint pr = 0; pr < 4u; ++pr) {
          if (g * 32u + i * 8u + 2u * pr < p.tokens) {
            total += c[pr][2u * i];
            total += c[pr][2u * i + 1u];
          }
        }
      }
    }
    R[tid] = half(1.0f / float(total));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint chunk = tid; chunk < p.groups * 32u; chunk += 256u) {
    const uint g = chunk / 32u;
    const uint l = chunk % 32u;
    const uint fm = ((l >> 4) << 2) + ((l >> 1) & 3u);
    const uint fn = ((l >> 2) & 2u) * 2u + (l & 1u) * 2u;
    const half r = R[fm];
    threadgroup half4* cell = (threadgroup half4*)(S + chunk * 8u);
    for (uint h = 0; h < 2u; ++h) {
      const half4 pr = cell[h] * r;
      const uint j0 = g * 32u + (2u * h) * 8u + fn;
      const uint j1 = j0 + 8u;
      cell[h] = half4(
          j0 < p.tokens ? nr_fast_e4m3_nonnegative(pr.x) : 0.0h,
          j0 + 1u < p.tokens ? nr_fast_e4m3_nonnegative(pr.y) : 0.0h,
          j1 < p.tokens ? nr_fast_e4m3_nonnegative(pr.z) : 0.0h,
          j1 + 1u < p.tokens ? nr_fast_e4m3_nonnegative(pr.w) : 0.0h);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd >= 4u) return;
  simdgroup_matrix<float, 8, 8> acc;
  acc.thread_elements()[0] = 0.0f;
  acc.thread_elements()[1] = 0.0f;
  const device half4* values = (const device half4*)(value + z * p.groups * 1024u);
  for (uint g = 0; g < p.groups; ++g) {
    const half4 pa = S4[(g * 32u + lane) * 2u];
    const half4 pb = S4[(g * 32u + lane) * 2u + 1u];
    const half4 va = values[((g * 4u + simd) * 32u + lane) * 2u];
    const half4 vb = values[((g * 4u + simd) * 32u + lane) * 2u + 1u];
    simdgroup_matrix<half, 8, 8> pm, vm;
    pm.thread_elements()[0] = pa.x; pm.thread_elements()[1] = pa.y;
    vm.thread_elements()[0] = va.x; vm.thread_elements()[1] = va.y;
    simdgroup_multiply_accumulate(acc, pm, vm, acc);
    if (g * 4u + 1u < p.tiles) {
      pm.thread_elements()[0] = pa.z; pm.thread_elements()[1] = pa.w;
      vm.thread_elements()[0] = va.z; vm.thread_elements()[1] = va.w;
      simdgroup_multiply_accumulate(acc, pm, vm, acc);
    }
    if (g * 4u + 2u < p.tiles) {
      pm.thread_elements()[0] = pb.x; pm.thread_elements()[1] = pb.y;
      vm.thread_elements()[0] = vb.x; vm.thread_elements()[1] = vb.y;
      simdgroup_multiply_accumulate(acc, pm, vm, acc);
    }
    if (g * 4u + 3u < p.tiles) {
      pm.thread_elements()[0] = pb.z; pm.thread_elements()[1] = pb.w;
      vm.thread_elements()[0] = vb.z; vm.thread_elements()[1] = vb.w;
      simdgroup_multiply_accumulate(acc, pm, vm, acc);
    }
  }

  const uint fragment_row = ((lane >> 4) << 2) + ((lane >> 1) & 3u);
  const uint fragment_col = ((lane >> 2) & 2u) * 2u + (lane & 1u) * 2u;
  const uint row = row0 + fragment_row;
  if (row < p.tokens) {
    *(device half2*)(output + row * p.heads * 32u + z * 32u + simd * 8u + fragment_col) = half2(
        nr_fast_e4m3_no_nan(half(acc.thread_elements()[0])),
        nr_fast_e4m3_no_nan(half(acc.thread_elements()[1])));
  }
}

// 2-pass streamed global attention for large sequences (tokens > 1024, zero N*N buffer).
kernel void nr_fast_global_qkv_split_norm(
    const device half* projected [[buffer(0)]],
    const device half* scale [[buffer(1)]],
    device half* q_out [[buffer(2)]],
    device half* k_out [[buffer(3)]],
    device half* v_out [[buffer(4)]],
    constant GlobalAttnParams& p [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
  const uint total = p.tokens * p.heads * 3u;
  if (tid >= total) return;
  const uint head = tid % p.heads;
  const uint rem = tid / p.heads;
  const uint part = rem % 3u;
  const uint token = rem / 3u;

  const device half4* in4 =
      (const device half4*)(projected + (token * 3u + part) * p.heads * 32u + head * 32u);
  half val[32];
  for (uint i = 0; i < 8u; ++i) {
    half4 v = in4[i];
    val[4u * i] = v.x; val[4u * i + 1u] = v.y; val[4u * i + 2u] = v.z; val[4u * i + 3u] = v.w;
  }

  device half4* dst4 = (device half4*)((part == 0u ? q_out : (part == 1u ? k_out : v_out)) +
                                       (head * p.tokens + token) * 32u);
  if (part == 2u) {
    for (uint i = 0; i < 8u; ++i) {
      dst4[i] = nr_fast_e4m3_vec4(half4(val[4u * i], val[4u * i + 1u], val[4u * i + 2u], val[4u * i + 3u]));
    }
    return;
  }

  half partial[4][2];
  for (uint lane = 0; lane < 4u; ++lane) {
    for (uint parity = 0; parity < 2u; ++parity) {
      uint ch = lane * 2u + parity;
      partial[lane][parity] =
          fma(val[ch + 8u], val[ch + 8u], val[ch] * val[ch]) +
          fma(val[ch + 24u], val[ch + 24u], val[ch + 16u] * val[ch + 16u]);
    }
  }
  half xor_two[4][2];
  for (uint lane = 0; lane < 4u; ++lane) {
    for (uint parity = 0; parity < 2u; ++parity) {
      xor_two[lane][parity] = partial[lane][parity] + partial[lane ^ 2u][parity];
    }
  }
  half xor_one[4][2];
  for (uint lane = 0; lane < 4u; ++lane) {
    for (uint parity = 0; parity < 2u; ++parity) {
      xor_one[lane][parity] = xor_two[lane][parity] + xor_two[lane ^ 1u][parity];
    }
  }
  half norm = max(xor_one[0][0] + xor_one[0][1], half(0.00006198883056640625h));
  half inv = half(metal::fast::rsqrt(float(norm)));
  if (part == 0u) inv *= scale[head];
  for (uint i = 0; i < 8u; ++i) {
    dst4[i] = nr_fast_e4m3_vec4(
        half4(val[4u * i] * inv, val[4u * i + 1u] * inv, val[4u * i + 2u] * inv, val[4u * i + 3u] * inv));
  }
}

inline void nr_fast_score_tile_v2(
    threadgroup half* Q,
    threadgroup half* K,
    threadgroup half* S,
    const device half* key,
    uint head_base,
    uint token0,
    uint tokens,
    uint tid,
    uint simd) {
  for (uint i = tid; i < 32u * 32u; i += 128u) {
    uint token = token0 + i / 32u;
    K[i] = token < tokens ? key[head_base + token * 32u + (i % 32u)] : 0.0h;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  simdgroup_matrix<float, 8, 8> acc;
  acc.thread_elements()[0] = 0.0f;
  acc.thread_elements()[1] = 0.0f;
  for (uint c = 0; c < 32u; c += 8u) {
    simdgroup_matrix<half, 8, 8> q, k;
    simdgroup_load(q, Q + c, 32, ulong2(0), false);
    simdgroup_load(k, K + simd * 8u * 32u + c, 32, ulong2(0), true);
    simdgroup_multiply_accumulate(acc, q, k, acc);
  }
  simdgroup_matrix<half, 8, 8> scores;
  scores.thread_elements()[0] = clamp(half(acc.thread_elements()[0]), -3.0h, 3.0h);
  scores.thread_elements()[1] = clamp(half(acc.thread_elements()[1]), -3.0h, 3.0h);
  simdgroup_store(scores, S + simd * 8u, 32, ulong2(0), false);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint valid = min(32u, tokens - token0);
  for (uint i = tid; i < 8u * 16u; i += 128u) {
    const uint row = i / 16u;
    const uint j = (i % 16u) * 2u;
    if (j < valid) {
      half2 w = nr_fast_vendor_weights(half2(S[row * 32u + j], S[row * 32u + j + 1u]));
      S[row * 32u + j] = w.x;
      S[row * 32u + j + 1u] = w.y;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

kernel void nr_fast_global_attn_denom(
    const device half* query [[buffer(0)]],
    const device half* key [[buffer(1)]],
    device half* reciprocal [[buffer(2)]],
    constant GlobalAttnParams& p [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]]) {
  const uint head = tgid.z;
  const uint row0 = tgid.y * 8u;
  const uint head_base = head * p.tokens * 32u;
  threadgroup half Q[8 * 32];
  threadgroup half K[32 * 32];
  threadgroup half S[8 * 32];
  for (uint i = tid; i < 8u * 32u; i += 128u) {
    uint row = row0 + i / 32u;
    Q[i] = row < p.tokens ? query[head_base + row * 32u + (i % 32u)] : 0.0h;
  }
  half total = 0.0h;
  for (uint token0 = 0; token0 < p.tokens; token0 += 32u) {
    nr_fast_score_tile_v2(Q, K, S, key, head_base, token0, p.tokens, tid, simd);
    if (tid < 8u) {
      const uint valid = min(32u, p.tokens - token0);
      for (uint j = 0; j < valid; ++j) total += S[tid * 32u + j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid < 8u && row0 + tid < p.tokens) {
    reciprocal[head * p.tokens + row0 + tid] = half(1.0f / float(total));
  }
}

kernel void nr_fast_global_attn_values(
    const device half* query [[buffer(0)]],
    const device half* key [[buffer(1)]],
    const device half* value [[buffer(2)]],
    const device half* reciprocal [[buffer(3)]],
    device half* output [[buffer(4)]],
    constant GlobalAttnParams& p [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]]) {
  const uint head = tgid.z;
  const uint row0 = tgid.y * 8u;
  const uint head_base = head * p.tokens * 32u;
  threadgroup half Q[8 * 32];
  threadgroup half K[32 * 32];
  threadgroup half S[8 * 32];
  for (uint i = tid; i < 8u * 32u; i += 128u) {
    uint row = row0 + i / 32u;
    Q[i] = row < p.tokens ? query[head_base + row * 32u + (i % 32u)] : 0.0h;
  }
  simdgroup_matrix<float, 8, 8> attended;
  attended.thread_elements()[0] = 0.0f;
  attended.thread_elements()[1] = 0.0f;
  for (uint token0 = 0; token0 < p.tokens; token0 += 32u) {
    nr_fast_score_tile_v2(Q, K, S, key, head_base, token0, p.tokens, tid, simd);
    for (uint i = tid; i < 8u * 32u; i += 128u) {
      const uint row = i / 32u;
      const uint j = i % 32u;
      const half rr = (row0 + row < p.tokens) ? reciprocal[head * p.tokens + row0 + row] : 0.0h;
      S[i] = (token0 + j < p.tokens) ? nr_fast_e4m3(S[i] * rr) : 0.0h;
    }
    for (uint i = tid; i < 32u * 32u; i += 128u) {
      uint token = token0 + i / 32u;
      K[i] = token < p.tokens ? value[head_base + token * 32u + (i % 32u)] : 0.0h;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint c = 0; c < 32u; c += 8u) {
      simdgroup_matrix<half, 8, 8> pm, vm;
      simdgroup_load(pm, S + c, 32, ulong2(0), false);
      simdgroup_load(vm, K + c * 32u + simd * 8u, 32, ulong2(0), false);
      simdgroup_multiply_accumulate(attended, pm, vm, attended);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  simdgroup_matrix<half, 8, 8> result;
  result.thread_elements()[0] = nr_fast_e4m3_no_nan(half(attended.thread_elements()[0]));
  result.thread_elements()[1] = nr_fast_e4m3_no_nan(half(attended.thread_elements()[1]));
  simdgroup_store(result, S + simd * 8u, 32, ulong2(0), false);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < 8u * 32u; i += 128u) {
    uint row = row0 + i / 32u;
    if (row < p.tokens) {
      output[(row * p.heads + head) * 32u + (i % 32u)] = S[i];
    }
  }
}

// ============================================================================
// 7. GPU Feature Preprocessor, Catmull-Rom Temporal Reprojection & Postprocessor
// ============================================================================

inline uint nr_fast_pcg_mix(uint input) {
  uint shift = (input >> 28) + 4u;
  return (input ^ (input >> shift)) * 0x108ef2d9u;
}

inline float nr_fast_uniform24(uint input) {
  uint multiplied = nr_fast_pcg_mix(input);
  uint bits = (multiplied >> 30) ^ (multiplied >> 8);
  return float(bits + 1u) * 5.960464477539063e-8f;
}

inline float nr_fast_log(float x) {
  uint bits = as_type<uint>(x);
  float e = float(int((bits >> 23) & 0xffu) - 126);
  float m = as_type<float>((bits & 0x807fffffu) | 0x3f000000u);
  if (m < 0.707106781186547524f) {
    e -= 1.0f;
    m = m + m - 1.0f;
  } else {
    m = m - 1.0f;
  }
  float z = m * m;
  float y = 7.0376836292e-2f * m - 1.1514610310e-1f;
  y = y * m + 1.1676998740e-1f;
  y = y * m - 1.2420140846e-1f;
  y = y * m + 1.4249322787e-1f;
  y = y * m - 1.6668057665e-1f;
  y = y * m + 2.0000714765e-1f;
  y = y * m - 2.4999993993e-1f;
  y = y * m + 3.3333331174e-1f;
  y = y * m * z;
  y = y + -2.12194440e-4f * e;
  y = y + -0.5f * z;
  float result = m + y;
  result = result + 0.693359375f * e;
  return result;
}

inline float nr_fast_sin_poly(float z, float x) {
  float y = -1.9515295891e-4f * z + 8.3321608736e-3f;
  y = y * z - 1.6666654611e-1f;
  y = y * z * x;
  return y + x;
}

inline float nr_fast_cos_poly(float z) {
  float y = 2.443315711809948e-5f * z - 1.388731625493765e-3f;
  y = y * z + 4.166664568298827e-2f;
  y = y * z * z;
  y = y - 0.5f * z;
  return y + 1.0f;
}

inline float nr_fast_sin(float input) {
  float sign = 1.0f;
  float x = input;
  if (x < 0.0f) { x = -x; sign = -1.0f; }
  int j = int(1.27323954473516f * x);
  float y = float(j);
  if (j & 1) { j += 1; y += 1.0f; }
  j &= 7;
  if (j > 3) { sign = -sign; j -= 4; }
  x = ((x - y * 0.78515625f) - y * 2.4187564849853515625e-4f) - y * 3.77489497744594108e-8f;
  float z = x * x;
  float result = (j == 1 || j == 2) ? nr_fast_cos_poly(z) : nr_fast_sin_poly(z, x);
  return sign < 0.0f ? -result : result;
}

inline float nr_fast_cos(float input) {
  float x = input < 0.0f ? -input : input;
  int j = int(1.27323954473516f * x);
  float y = float(j);
  if (j & 1) { j += 1; y += 1.0f; }
  j &= 7;
  float sign = 1.0f;
  if (j > 3) { j -= 4; sign = -sign; }
  if (j > 1) { sign = -sign; }
  x = ((x - y * 0.78515625f) - y * 2.4187564849853515625e-4f) - y * 3.77489497744594108e-8f;
  float z = x * x;
  float result = (j == 1 || j == 2) ? nr_fast_sin_poly(z, x) : nr_fast_cos_poly(z);
  return sign < 0.0f ? -result : result;
}

inline half nr_fast_scaled_color(float value) {
  float sampled = float(half(value));
  float centered = float(half(sampled - 0.5f));
  return half(centered * 0.125f);
}

struct CatmullCoords {
  float outer0;
  float middle;
  float outer3;
  float w0;
  float w3;
  float g;
};

inline CatmullCoords nr_fast_catmull_coords(float normalized, uint dimension) {
  float pixel = normalized * float(dimension) - 0.5f;
  float base_index = floor(pixel);
  float t = clamp(pixel - base_index, 0.0f, 1.0f);
  float sq = t * t;
  float cb = sq * t;
  float w0 = -0.5f * t + sq - 0.5f * cb;
  float w1 = 1.0f - 2.5f * sq + 1.5f * cb;
  float w2 = 0.5f * t + 2.0f * sq - 1.5f * cb;
  float w3 = -0.5f * sq + 0.5f * cb;
  float g = w1 + w2;
  float base = base_index + 0.5f;
  float upper = float(dimension) - 0.5f;
  return {
      clamp(base - 1.0f, 0.5f, upper),
      clamp(base + w2 / g, 0.5f, upper),
      clamp(base + 2.0f, 0.5f, upper),
      w0,
      w3,
      g};
}

inline float3 nr_fast_sample_linear_tex(texture2d<float, access::read> tex, uint w, uint h, float x, float y) {
  float px = x - 0.5f;
  float py = y - 0.5f;
  int x0 = clamp(int(floor(px)), 0, int(w) - 1);
  int y0 = clamp(int(floor(py)), 0, int(h) - 1);
  int x1 = min(x0 + 1, int(w) - 1);
  int y1 = min(y0 + 1, int(h) - 1);
  float tx = clamp(px - float(x0), 0.0f, 1.0f);
  float ty = clamp(py - float(y0), 0.0f, 1.0f);
  float3 top = tex.read(uint2(x0, y0)).rgb * (1.0f - tx) + tex.read(uint2(x1, y0)).rgb * tx;
  float3 bot = tex.read(uint2(x0, y1)).rgb * (1.0f - tx) + tex.read(uint2(x1, y1)).rgb * tx;
  return top * (1.0f - ty) + bot * ty;
}

inline float3 nr_fast_sample_catmull5_tex(
    texture2d<float, access::read> tex,
    uint w,
    uint h,
    float u,
    float v) {
  CatmullCoords cx = nr_fast_catmull_coords(u, w);
  CatmullCoords cy = nr_fast_catmull_coords(v, h);
  float lw = cx.w0 * cy.g;
  float tw = cx.g * cy.w0;
  float mw = cx.g * cy.g;
  float bw = cx.g * cy.w3;
  float rw = cx.w3 * cy.g;
  float3 sum =
      nr_fast_sample_linear_tex(tex, w, h, cx.outer0, cy.middle) * lw +
      nr_fast_sample_linear_tex(tex, w, h, cx.middle, cy.outer0) * tw +
      nr_fast_sample_linear_tex(tex, w, h, cx.middle, cy.middle) * mw +
      nr_fast_sample_linear_tex(tex, w, h, cx.middle, cy.outer3) * bw +
      nr_fast_sample_linear_tex(tex, w, h, cx.outer3, cy.middle) * rw;
  return sum / (lw + tw + mw + bw + rw);
}

struct PreFeatureParams {
  uint in_width;
  uint in_height;
  uint net_width;
  uint net_height;
  uint logical_width;
  uint logical_height;
  uint frame_index;
  uint scaled_network;     // 0 = reflect-pad logical->net, 1 = resample input->net
  uint has_history;        // 1 on temporal frames (!reset)
  uint has_motion;         // 1 if motion texture provided
  uint has_depth;          // 1 if depth texture provided & use_closest_depth
  uint depth_inverted;
  uint srgb_transfer;      // 1 = convert linear input -> sRGB
  float normalized_style;
  float local_tone;
  float local_structure;
  float motion_scale_x;
  float motion_scale_y;
};

inline uint nr_fast_reflect_coord(uint c, uint extent) {
  if (c < extent) return c;
  int r = 2 * int(extent) - 2 - int(c);
  return uint(max(0, r));
}

inline float3 nr_fast_read_input_srgb(
    texture2d<float, access::read> in_tex,
    uint x,
    uint y,
    bool srgb_transfer) {
  float3 rgb = in_tex.read(uint2(x, y)).rgb;
  return srgb_transfer ? nr_fast_linear_to_srgb(rgb) : rgb;
}

// Generates [1, H_net, W_net, 16] half features + [H_net, W_net, 4] base RGB.
kernel void nr_fast_pre_features(
    texture2d<float, access::read> color_tex [[texture(0)]],
    texture2d<float, access::read> history_tex [[texture(1)]],
    texture2d<float, access::read> motion_tex [[texture(2)]],
    texture2d<float, access::read> depth_tex [[texture(3)]],
    device half* features_f16 [[buffer(0)]],
    device float4* base_net_rgb [[buffer(1)]],
    constant PreFeatureParams& p [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= p.net_width || gid.y >= p.net_height) return;

  // 1. PCG RXS-M-XS 0x108ef2d9 Gaussian noise at network coordinate (gid.x, gid.y).
  uint seed = gid.y * 0xd8163841u;
  seed ^= gid.x * 0x8da6b343u;
  seed ^= p.frame_index * 0x9e3779b9u;
  seed ^= 0x243f6a88u;
  uint mult = nr_fast_pcg_mix(seed);
  uint mixed = mult ^ (mult >> 22);

  float rad_a_u = nr_fast_uniform24(mixed * 0xcaa5b80du + 0x21dd796bu);
  float ang_b_u = nr_fast_uniform24(mixed * 0x83232c31u + 0x3463e0acu);
  float rad_b_u = nr_fast_uniform24(mixed * 0x2c9277b5u + 0xac564b05u);
  float ang_a_u = nr_fast_uniform24(mixed * 0xfa6dc5f9u + 0x4712a88eu);

  float rad_a = precise::sqrt(-2.0f * nr_fast_log(rad_a_u));
  float rad_b = precise::sqrt(-2.0f * nr_fast_log(rad_b_u));
  const float tau = 6.2831854820251465f;
  float ang_a = tau * ang_a_u;
  float ang_b = tau * ang_b_u;

  half n0 = half(rad_b * nr_fast_cos(ang_a));
  half n1 = half(rad_b * nr_fast_sin(ang_a));
  half n2 = half(rad_a * nr_fast_cos(ang_b));

  // 2. Current frame RGB at network coordinate.
  float3 cur_rgb;
  uint logical_x = gid.x;
  uint logical_y = gid.y;
  if (p.scaled_network == 0u) {
    logical_x = nr_fast_reflect_coord(gid.x, p.logical_width);
    logical_y = nr_fast_reflect_coord(gid.y, p.logical_height);
    cur_rgb = nr_fast_read_input_srgb(color_tex, logical_x, logical_y, p.srgb_transfer != 0u);
  } else {
    // Integer box average if exact integer downscale, else bilinear sample.
    if (p.in_width % p.net_width == 0u && p.in_height % p.net_height == 0u &&
        (p.in_width / p.net_width) == (p.in_height / p.net_height) && (p.in_width / p.net_width) <= 4u) {
      uint factor = p.in_width / p.net_width;
      float3 sum = float3(0.0f);
      for (uint dy = 0; dy < factor; ++dy) {
        for (uint dx = 0; dx < factor; ++dx) {
          sum += nr_fast_read_input_srgb(
              color_tex, gid.x * factor + dx, gid.y * factor + dy, p.srgb_transfer != 0u);
        }
      }
      cur_rgb = sum * (1.0f / float(factor * factor));
    } else {
      float fx = (float(gid.x) + 0.5f) * float(p.in_width) / float(p.net_width);
      float fy = (float(gid.y) + 0.5f) * float(p.in_height) / float(p.net_height);
      cur_rgb = nr_fast_sample_linear_tex(color_tex, p.in_width, p.in_height, fx, fy);
      if (p.srgb_transfer != 0u) {
        cur_rgb = nr_fast_linear_to_srgb(cur_rgb);
      }
    }
  }

  base_net_rgb[gid.y * p.net_width + gid.x] = float4(cur_rgb, 1.0f);

  half r_cur = nr_fast_scaled_color(cur_rgb.x);
  half g_cur = nr_fast_scaled_color(cur_rgb.y);
  half b_cur = nr_fast_scaled_color(cur_rgb.z);

  // 3. Temporal history reprojection (5-tap Catmull-Rom) or reset = current RGB.
  half r_hist = r_cur;
  half g_hist = g_cur;
  half b_hist = b_cur;
  if (p.has_history != 0u) {
    uint hist_w = (p.scaled_network == 0u) ? p.logical_width : p.net_width;
    uint hist_h = (p.scaled_network == 0u) ? p.logical_height : p.net_height;
    uint qx = (p.scaled_network == 0u) ? logical_x : gid.x;
    uint qy = (p.scaled_network == 0u) ? logical_y : gid.y;

    int sel_x = int(qx);
    int sel_y = int(qy);
    if (p.has_depth != 0u) {
      uint dw = depth_tex.get_width();
      uint dh = depth_tex.get_height();
      uint dx0 = clamp(uint((float(sel_x) + 0.5f) * float(dw) / float(hist_w)), 0u, dw - 1u);
      uint dy0 = clamp(uint((float(sel_y) + 0.5f) * float(dh) / float(hist_h)), 0u, dh - 1u);
      float best_d = depth_tex.read(uint2(dx0, dy0)).x;
      const int2 diags[4] = {int2(-1, -1), int2(1, -1), int2(-1, 1), int2(1, 1)};
      for (uint i = 0; i < 4u; ++i) {
        int cx = clamp(int(qx) + diags[i].x, 0, int(hist_w) - 1);
        int cy = clamp(int(qy) + diags[i].y, 0, int(hist_h) - 1);
        uint cdx = clamp(uint((float(cx) + 0.5f) * float(dw) / float(hist_w)), 0u, dw - 1u);
        uint cdy = clamp(uint((float(cy) + 0.5f) * float(dh) / float(hist_h)), 0u, dh - 1u);
        float cd = depth_tex.read(uint2(cdx, cdy)).x;
        bool closer = (p.depth_inverted != 0u) ? (cd > best_d) : (cd < best_d);
        if (closer) {
          sel_x = cx;
          sel_y = cy;
          best_d = cd;
        }
      }
    }

    float2 mv = float2(0.0f);
    if (p.has_motion != 0u) {
      uint mw = motion_tex.get_width();
      uint mh = motion_tex.get_height();
      uint mx = clamp(uint((float(sel_x) + 0.5f) * float(mw) / float(hist_w)), 0u, mw - 1u);
      uint my = clamp(uint((float(sel_y) + 0.5f) * float(mh) / float(hist_h)), 0u, mh - 1u);
      mv = motion_tex.read(uint2(mx, my)).xy * float2(p.motion_scale_x, p.motion_scale_y);
    }

    float u = (float(qx) + 0.5f) / float(hist_w) + mv.x;
    float v = (float(qy) + 0.5f) / float(hist_h) + mv.y;
    float3 sampled_hist = nr_fast_sample_catmull5_tex(history_tex, hist_w, hist_h, u, v);
    r_hist = nr_fast_scaled_color(sampled_hist.x);
    g_hist = nr_fast_scaled_color(sampled_hist.y);
    b_hist = nr_fast_scaled_color(sampled_hist.z);
  }

  const uint feat_off = (gid.y * p.net_width + gid.x) * 16u;
  features_f16[feat_off + 0u] = n0;
  features_f16[feat_off + 1u] = n1;
  features_f16[feat_off + 2u] = n2;
  features_f16[feat_off + 3u] = 1.0h;
  features_f16[feat_off + 4u] = r_cur;
  features_f16[feat_off + 5u] = g_cur;
  features_f16[feat_off + 6u] = b_cur;
  features_f16[feat_off + 7u] = r_hist;
  features_f16[feat_off + 8u] = g_hist;
  features_f16[feat_off + 9u] = b_hist;
  features_f16[feat_off + 10u] = half(p.normalized_style);
  features_f16[feat_off + 11u] = half(p.local_tone);
  features_f16[feat_off + 12u] = half(p.local_structure);
  features_f16[feat_off + 13u] = -1.0h;
  features_f16[feat_off + 14u] = -1.0h;
  features_f16[feat_off + 15u] = 0.0h;
}

struct PostTemporalParams {
  uint net_width;
  uint net_height;
  uint hist_width;
  uint hist_height;
  uint out_width;
  uint out_height;
  uint scaled_network;
  uint has_history;
  uint direct_write_output;  // 1 when 1:1 and detail_strength==1 && colour_strength==1
  uint srgb_transfer;
  float blend_scale;
  float intensity;
};

// Evaluates predicted RGB + temporal blend, updates persistent history_tex, and
// either writes directly to output_tex (fast 1:1 path) or emits delta_net_rgb.
kernel void nr_fast_post_temporal(
    const device half4* head_f16 [[buffer(0)]],
    const device half* features_f16 [[buffer(1)]],
    const device float4* base_net_rgb [[buffer(2)]],
    device float4* delta_net_rgb [[buffer(3)]],
    texture2d<float, access::write> history_out_tex [[texture(0)]],
    texture2d<float, access::write> output_tex [[texture(1)]],
    constant PostTemporalParams& p [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= p.hist_width || gid.y >= p.hist_height) return;

  const uint net_idx = gid.y * p.net_width + gid.x;
  const half4 h = head_f16[net_idx];
  const float3 cur = base_net_rgb[net_idx].rgb;

  float3 residual = float3(float(h.x), float(h.y), float(h.z)) * 0.25f;
  float3 predicted = clamp(cur + residual, 0.0f, 1.0f);
  float3 temporal = predicted;

  if (p.has_history != 0u) {
    float logit = float(h.w);
    float bscale = float(half(p.blend_scale));
    float alpha = clamp((1.0f / (1.0f + precise::exp(-logit))) * bscale, 0.0f, 1.0f);
    const uint f_off = net_idx * 16u;
    float3 hist_rgb = float3(
        float(features_f16[f_off + 7u]) * 8.0f + 0.5f,
        float(features_f16[f_off + 8u]) * 8.0f + 0.5f,
        float(features_f16[f_off + 9u]) * 8.0f + 0.5f);
    temporal = predicted + alpha * (hist_rgb - predicted);
  }

  float3 nr_rgb = temporal;
  if (p.scaled_network == 0u && p.intensity != 1.0f) {
    float b = clamp(p.intensity, 0.0f, 1.0f);
    nr_rgb = clamp(cur + b * (temporal - cur), 0.0f, 1.0f);
  }

  history_out_tex.write(float4(nr_rgb, 1.0f), gid);

  if (p.direct_write_output != 0u) {
    if (gid.x < p.out_width && gid.y < p.out_height) {
      float3 out_rgb = (p.srgb_transfer != 0u) ? nr_fast_srgb_to_linear(nr_rgb) : nr_rgb;
      output_tex.write(float4(out_rgb, 1.0f), gid);
    }
  } else {
    delta_net_rgb[gid.y * p.hist_width + gid.x] = float4(nr_rgb - cur, 0.0f);
  }
}

struct DetailComposeParams {
  uint net_width;
  uint net_height;
  uint out_width;
  uint out_height;
  uint radius;  // Gaussian half-width (taps = 2 * radius + 1, max radius = 32)
  uint scaled_network;
  uint srgb_transfer;
  float detail_strength;
  float colour_strength;
  float intensity;
};

inline float3 nr_fast_sample_delta_bilinear(
    const device float4* delta,
    uint w,
    uint h,
    float u,
    float v) {
  float px = u * float(w) - 0.5f;
  float py = v * float(h) - 0.5f;
  int x0 = clamp(int(floor(px)), 0, int(w) - 1);
  int y0 = clamp(int(floor(py)), 0, int(h) - 1);
  int x1 = min(x0 + 1, int(w) - 1);
  int y1 = min(y0 + 1, int(h) - 1);
  float tx = clamp(px - float(x0), 0.0f, 1.0f);
  float ty = clamp(py - float(y0), 0.0f, 1.0f);
  float3 top = delta[uint(y0) * w + uint(x0)].rgb * (1.0f - tx) + delta[uint(y0) * w + uint(x1)].rgb * tx;
  float3 bot = delta[uint(y1) * w + uint(x0)].rgb * (1.0f - tx) + delta[uint(y1) * w + uint(x1)].rgb * tx;
  return top * (1.0f - ty) + bot * ty;
}

inline float3 nr_fast_read_output_base_srgb(
    texture2d<float, access::read> color_tex,
    uint x,
    uint y,
    uint out_w,
    uint out_h,
    bool srgb_transfer) {
  const uint in_w = color_tex.get_width();
  const uint in_h = color_tex.get_height();
  if (in_w == out_w && in_h == out_h) {
    return nr_fast_read_input_srgb(color_tex, x, y, srgb_transfer);
  }
  const float fx = (float(x) + 0.5f) * float(in_w) / float(out_w);
  const float fy = (float(y) + 0.5f) * float(in_h) / float(out_h);
  float3 rgb = nr_fast_sample_linear_tex(color_tex, in_w, in_h, fx, fy);
  return srgb_transfer ? nr_fast_linear_to_srgb(rgb) : rgb;
}

// Fast upsample-and-compose when detail_strength == colour_strength (no blur needed).
kernel void nr_fast_upsample_compose(
    texture2d<float, access::read> color_tex [[texture(0)]],
    const device float4* delta_net [[buffer(0)]],
    texture2d<float, access::write> output_tex [[texture(1)]],
    constant DetailComposeParams& p [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= p.out_width || gid.y >= p.out_height) return;

  float3 src = nr_fast_read_output_base_srgb(
      color_tex, gid.x, gid.y, p.out_width, p.out_height, p.srgb_transfer != 0u);
  float3 delta;
  if (p.scaled_network == 0u) {
    delta = delta_net[gid.y * p.net_width + gid.x].rgb;
    float3 composed = clamp(src + p.detail_strength * delta, 0.0f, 1.0f);
    if (p.srgb_transfer != 0u) composed = nr_fast_srgb_to_linear(composed);
    output_tex.write(float4(composed, 1.0f), gid);
  } else {
    float u = (float(gid.x) + 0.5f) / float(p.out_width);
    float v = (float(gid.y) + 0.5f) / float(p.out_height);
    delta = nr_fast_sample_delta_bilinear(delta_net, p.net_width, p.net_height, u, v);
    float3 composed = clamp(src + p.intensity * p.detail_strength * delta, 0.0f, 1.0f);
    if (p.srgb_transfer != 0u) composed = nr_fast_srgb_to_linear(composed);
    output_tex.write(float4(composed, 1.0f), gid);
  }
}

inline float3 nr_fast_fetch_delta_for_blur(
    const device float4* delta_net,
    constant DetailComposeParams& p,
    int x,
    int y) {
  int cx = clamp(x, 0, int(p.out_width) - 1);
  int cy = clamp(y, 0, int(p.out_height) - 1);
  if (p.scaled_network == 0u) {
    return delta_net[uint(cy) * p.net_width + uint(cx)].rgb;
  }
  float u = (float(cx) + 0.5f) / float(p.out_width);
  float v = (float(cy) + 0.5f) / float(p.out_height);
  return nr_fast_sample_delta_bilinear(delta_net, p.net_width, p.net_height, u, v);
}

// Pass 1 of separable Gaussian detail split: upsample delta (if scaled) + horizontal blur.
kernel void nr_fast_detail_blur_h(
    const device float4* delta_net [[buffer(0)]],
    device float4* delta_full [[buffer(1)]],
    device float4* blur_h [[buffer(2)]],
    constant DetailComposeParams& p [[buffer(3)]],
    constant float* weights [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= p.out_width || gid.y >= p.out_height) return;

  float3 center = nr_fast_fetch_delta_for_blur(delta_net, p, int(gid.x), int(gid.y));
  delta_full[gid.y * p.out_width + gid.x] = float4(center, 0.0f);

  int rad = int(p.radius);
  float3 acc = float3(0.0f);
  for (int dx = -rad; dx <= rad; ++dx) {
    acc += nr_fast_fetch_delta_for_blur(delta_net, p, int(gid.x) + dx, int(gid.y)) * weights[dx + rad];
  }
  blur_h[gid.y * p.out_width + gid.x] = float4(acc, 0.0f);
}

// Pass 2 of separable Gaussian detail split: vertical blur + low/high frequency recompose.
kernel void nr_fast_detail_blur_v_compose(
    texture2d<float, access::read> color_tex [[texture(0)]],
    const device float4* delta_full [[buffer(0)]],
    const device float4* blur_h [[buffer(1)]],
    texture2d<float, access::write> output_tex [[texture(1)]],
    constant DetailComposeParams& p [[buffer(2)]],
    constant float* weights [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= p.out_width || gid.y >= p.out_height) return;

  int rad = int(p.radius);
  float3 low_pass = float3(0.0f);
  for (int dy = -rad; dy <= rad; ++dy) {
    int cy = clamp(int(gid.y) + dy, 0, int(p.out_height) - 1);
    low_pass += blur_h[uint(cy) * p.out_width + gid.x].rgb * weights[dy + rad];
  }

  float3 change = delta_full[gid.y * p.out_width + gid.x].rgb;
  float3 high_pass = change - low_pass;
  float3 src = nr_fast_read_output_base_srgb(
      color_tex, gid.x, gid.y, p.out_width, p.out_height, p.srgb_transfer != 0u);
  float scale = (p.scaled_network == 0u) ? 1.0f : p.intensity;
  float3 composed = clamp(
      src + scale * (p.colour_strength * low_pass + p.detail_strength * high_pass),
      0.0f,
      1.0f);
  if (p.srgb_transfer != 0u) {
    composed = nr_fast_srgb_to_linear(composed);
  }
  output_tex.write(float4(composed, 1.0f), gid);
}

// ============================================================================
// RenderTarget Fullscreen Blit Shaders (for CAMetalDrawable / sRGB outputs
// that carry MTLTextureUsageRenderTarget without MTLTextureUsageShaderWrite)
// ============================================================================

struct NRBridgeBlitVSOut {
  float4 position [[position]];
};

vertex NRBridgeBlitVSOut nr_bridge_blit_vs(uint vid [[vertex_id]]) {
  NRBridgeBlitVSOut out;
  float2 uv = float2((vid << 1u) & 2u, vid & 2u);
  out.position = float4(uv * float2(2.0f, -2.0f) + float2(-1.0f, 1.0f), 0.0f, 1.0f);
  return out;
}

fragment float4 nr_bridge_blit_fs(
    NRBridgeBlitVSOut in [[stage_in]],
    texture2d<float, access::read> src_tex [[texture(0)]]) {
  uint2 coord = uint2(in.position.xy);
  return float4(src_tex.read(coord).rgb, 1.0f);
}
