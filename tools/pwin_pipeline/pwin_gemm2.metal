// Batched per-token GEMM, version 2: the weight matrix (K x N halves, up to 16 KB) and the fp32 C tile
// are copied once per threadgroup into threadgroup memory; each SIMD group owns 16 tokens, keeps its A
// tiles in registers across the N tiles (K <= 64) and reads B from threadgroup memory. Same numerics as
// pwin_gemm_tokens (hw3 accumulation order, fp16 rounding points, split-K quirk, epilogues).
#include <metal_stdlib>
using namespace metal;

constant uint FC_K [[function_constant(0)]];   // reduction length (16..128)
constant uint FC_N [[function_constant(1)]];   // output columns (16..128)
constant uint FC_XS [[function_constant(2)]];  // optional compile-time x_stride (A row stride)
constant uint FC_CS [[function_constant(3)]];  // optional compile-time c_stride (C tile / Cin row stride)
constant uint FC_GDBG [[function_constant(10)]];  // timing diagnostics only: 1 no A loads, 2 no B loads, 3 no MMAs, 4 minimal store

struct Gemm2Params {
  uint tokens, k, n, x_stride, y_stride, c_stride, epilogue, c_mode;
  uint split_k16, width, shift_y, w_stride, store_cols, r_stride, row_filter, pad4;
  // split_k16: 1 = round N tile 0 after K tile 0 for all rows, 2 = same for window blocks 0..2 only (block 3 fused),
  //            3 = round every N tile after K tile 1 (K = 64 as two fp16-rounded halves, all rows)
  // epilogue: 1 relu, 2 GELU approximation, 3 residual add h = fp16(h + Cin[row * r_stride + col])
  // store_cols: 0 = all N columns; row_filter: 1 = store only rows whose window block ((y + shift_y) & 7) >> 1 is 0 or 2
};

static inline half gemm2_gelu(half h) {
  const half t = min(max(h, half(-2.0h)), half(2.0h));
  const half u = fma(-half(0x1.4c1bacp-4f), abs(t), half(0x1.a60dd6p-2f));
  const half s = fma(t, u, half(0.5h));
  return h * s;
}

kernel void pwin_gemm2_tokens(device const half* X [[buffer(0)]],
                              device const half* W [[buffer(1)]],
                              device const float* C [[buffer(2)]],
                              device half* Y [[buffer(3)]],
                              constant Gemm2Params& p [[buffer(4)]],
                              device const half* Cin [[buffer(5)]],
                              device const half* Cbias [[buffer(6)]],
                              uint simd_index [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint lid [[thread_index_in_threadgroup]],
                              uint3 tg [[threadgroup_position_in_grid]],
                              uint simdgroups [[simdgroups_per_threadgroup]],
                              uint3 threads3 [[threads_per_threadgroup]]) {
  const uint threads = threads3.x;
  threadgroup half Wt[8192];          // K x N halves (K*N <= 8192; larger weights are read from device memory)
  const bool w_in_tg = FC_K * FC_N <= 8192u;
  threadgroup float stage[8 * 256];   // per SIMD group fp32 tile
  threadgroup float* mine = stage + simd_index * 256u;
  const uint ws = p.w_stride == 0u ? FC_N : p.w_stride;
  const uint xs = is_function_constant_defined(FC_XS) ? FC_XS : p.x_stride;
  const uint cs = is_function_constant_defined(FC_CS) ? FC_CS : p.c_stride;
  // cooperative copy of W into threadgroup memory (row-major K x N)
  if (w_in_tg) {
    for (uint e = lid; e < FC_K * FC_N; e += threads) {
      const uint kk = e / FC_N, nn = e - kk * FC_N;
      Wt[e] = W[kk * ws + nn];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint row0 = (tg.x * simdgroups + simd_index) * 16u;
  if (row0 >= p.tokens) return;
  const bool full = row0 + 16u <= p.tokens;
  device const half* x = X + row0 * xs;
  // A tiles in registers (K <= 64: 4 K tiles x 4 matrices); all indices compile-time
  simdgroup_float8x8 A[4][2][2];
  const uint ktiles = FC_K / 16u;
#pragma clang loop unroll(full)
  for (uint kt = 0u; kt < 4u; ++kt) {
    if (kt < ktiles) {
#pragma clang loop unroll(full)
      for (uint i = 0u; i < 2u; ++i)
#pragma clang loop unroll(full)
        for (uint j = 0u; j < 2u; ++j) {
          simdgroup_half8x8 hm;
          if (is_function_constant_defined(FC_GDBG) && FC_GDBG == 1u) { A[kt][i][j] = simdgroup_float8x8(float(lane) * 1e-3f); continue; }
          simdgroup_load(hm, x + kt * 16u + j * 8u, xs, ulong2(0, i * 8u));
          A[kt][i][j].thread_elements()[0] = float(hm.thread_elements()[0]);
          A[kt][i][j].thread_elements()[1] = float(hm.thread_elements()[1]);
        }
    }
  }
  if (FC_K > 64u) {
    // long K (output projection): per N tile, A streamed from device memory per K tile
    for (uint n0 = 0u; n0 < FC_N; n0 += 16u) {
      simdgroup_float8x8 acc[2][2];
      #pragma clang loop unroll(full)
      for (uint i = 0u; i < 2u; ++i)
      #pragma clang loop unroll(full)
        for (uint j = 0u; j < 2u; ++j)
        simdgroup_load(acc[i][j], C + n0 + j * 8u, cs, ulong2(0, i * 8u));
      for (uint kt = 0u; kt < ktiles; ++kt) {
        simdgroup_float8x8 a[2][2], b[2][2];
        #pragma clang loop unroll(full)
        for (uint i = 0u; i < 2u; ++i)
        #pragma clang loop unroll(full)
          for (uint j = 0u; j < 2u; ++j) {
          simdgroup_half8x8 hm;
          simdgroup_load(hm, x + kt * 16u + j * 8u, xs, ulong2(0, i * 8u));
          a[i][j].thread_elements()[0] = float(hm.thread_elements()[0]);
          a[i][j].thread_elements()[1] = float(hm.thread_elements()[1]);
          if (w_in_tg) simdgroup_load(hm, Wt + (kt * 16u + i * 8u) * FC_N + n0 + j * 8u, FC_N);
          else simdgroup_load(hm, W + (kt * 16u + i * 8u) * ws + n0 + j * 8u, ws);
          b[i][j].thread_elements()[0] = float(hm.thread_elements()[0]);
          b[i][j].thread_elements()[1] = float(hm.thread_elements()[1]);
        }
        #pragma clang loop unroll(full)
        for (uint i = 0u; i < 2u; ++i)
        #pragma clang loop unroll(full)
          for (uint j = 0u; j < 2u; ++j) {
          simdgroup_multiply_accumulate(acc[i][j], a[i][0], b[0][j], acc[i][j]);
          simdgroup_multiply_accumulate(acc[i][j], a[i][1], b[1][j], acc[i][j]);
        }
      }
      #pragma clang loop unroll(full)
      for (uint i = 0u; i < 2u; ++i)
      #pragma clang loop unroll(full)
        for (uint j = 0u; j < 2u; ++j) simdgroup_store(acc[i][j], mine, 16u, ulong2(j * 8u, i * 8u));
      simdgroup_barrier(mem_flags::mem_threadgroup);
      const uint rr = lane >> 1u, c0 = (lane & 1u) * 8u;
      if (full || row0 + rr < p.tokens) {
        device half* y = Y + (row0 + rr) * p.y_stride + n0 + c0;
        half v[8];
        #pragma clang loop unroll(full)
        for (uint c = 0u; c < 8u; ++c) v[c] = half(mine[rr * 16u + c0 + c]);
        *reinterpret_cast<device half4*>(y) = half4(v[0], v[1], v[2], v[3]);
        *reinterpret_cast<device half4*>(y + 4) = half4(v[4], v[5], v[6], v[7]);
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    return;
  }
  // N tiles in pairs (static inner indices); A stays in registers for K <= 64
  for (uint nc = 0u; nc < FC_N; nc += 32u) {
    const bool two = nc + 16u < FC_N;
    simdgroup_float8x8 acc[2][2][2];
#pragma clang loop unroll(full)
    for (uint t = 0u; t < 2u; ++t) {
      if (t == 1u && !two) break;
      const uint n0 = nc + t * 16u;
      if (p.c_mode == 0u) {
        #pragma clang loop unroll(full)
        for (uint i = 0u; i < 2u; ++i)
        #pragma clang loop unroll(full)
          for (uint j = 0u; j < 2u; ++j)
          simdgroup_load(acc[t][i][j], C + n0 + j * 8u, cs, ulong2(0, i * 8u));
      } else {
        for (uint e = lane; e < 256u; e += 32u) {
          const uint rr = e >> 4u, c = e & 15u;
          const uint row = min(row0 + rr, p.tokens - 1u);
          mine[e] = float(half(Cin[row * cs + n0 + c] + Cbias[n0 + c]));
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (uint i = 0u; i < 2u; ++i)
        #pragma clang loop unroll(full)
          for (uint j = 0u; j < 2u; ++j) simdgroup_load(acc[t][i][j], mine, 16u, ulong2(j * 8u, i * 8u));
        simdgroup_barrier(mem_flags::mem_threadgroup);
      }
    }
#pragma clang loop unroll(full)
    for (uint kt = 0u; kt < 8u; ++kt) {
      if (kt >= ktiles) break;
      simdgroup_float8x8 a[2][2];
      if (kt < 4u) {
#pragma clang loop unroll(full)
        for (uint i = 0u; i < 2u; ++i)
#pragma clang loop unroll(full)
          for (uint j = 0u; j < 2u; ++j) a[i][j] = A[kt][i][j];
      } else {
        #pragma clang loop unroll(full)
        for (uint i = 0u; i < 2u; ++i)
        #pragma clang loop unroll(full)
          for (uint j = 0u; j < 2u; ++j) {
          simdgroup_half8x8 hm;
          simdgroup_load(hm, x + kt * 16u + j * 8u, xs, ulong2(0, i * 8u));
          a[i][j].thread_elements()[0] = float(hm.thread_elements()[0]);
          a[i][j].thread_elements()[1] = float(hm.thread_elements()[1]);
        }
      }
#pragma clang loop unroll(full)
      for (uint t = 0u; t < 2u; ++t) {
        if (t == 1u && !two) break;
        const uint n0 = nc + t * 16u;
        simdgroup_float8x8 b[2][2];
        #pragma clang loop unroll(full)
        for (uint i = 0u; i < 2u; ++i)
        #pragma clang loop unroll(full)
          for (uint j = 0u; j < 2u; ++j) {
          simdgroup_half8x8 hm;
          if (is_function_constant_defined(FC_GDBG) && FC_GDBG == 2u) { b[i][j] = a[i][j]; continue; }
          if (w_in_tg) simdgroup_load(hm, Wt + (kt * 16u + i * 8u) * FC_N + n0 + j * 8u, FC_N);
          else simdgroup_load(hm, W + (kt * 16u + i * 8u) * ws + n0 + j * 8u, ws);
          b[i][j].thread_elements()[0] = float(hm.thread_elements()[0]);
          b[i][j].thread_elements()[1] = float(hm.thread_elements()[1]);
        }
        #pragma clang loop unroll(full)
        for (uint i = 0u; i < 2u; ++i)
        #pragma clang loop unroll(full)
          for (uint j = 0u; j < 2u; ++j) {
          if (is_function_constant_defined(FC_GDBG) && FC_GDBG == 3u) {
            acc[t][i][j].thread_elements()[0] += a[i][0].thread_elements()[0] * b[0][j].thread_elements()[0];
            acc[t][i][j].thread_elements()[1] += a[i][1].thread_elements()[1] * b[1][j].thread_elements()[1];
          } else {
          simdgroup_multiply_accumulate(acc[t][i][j], a[i][0], b[0][j], acc[t][i][j]);
          simdgroup_multiply_accumulate(acc[t][i][j], a[i][1], b[1][j], acc[t][i][j]);
          }
        }
        const bool split_here = ((p.split_k16 == 1u || p.split_k16 == 2u) && kt == 0u && n0 == 0u && ktiles > 1u) ||
                                (p.split_k16 == 3u && kt == 1u);
        if (split_here) {
          #pragma clang loop unroll(full)
          for (uint i = 0u; i < 2u; ++i)
          #pragma clang loop unroll(full)
            for (uint j = 0u; j < 2u; ++j) simdgroup_store(acc[t][i][j], mine, 16u, ulong2(j * 8u, i * 8u));
          simdgroup_barrier(mem_flags::mem_threadgroup);
          for (uint e = lane; e < 256u; e += 32u) {
            const uint row = min(row0 + (e >> 4u), p.tokens - 1u);
            const uint block = (((row / p.width) + p.shift_y) & 7u) >> 1u;
            if (p.split_k16 != 2u || block != 3u) mine[e] = float(half(mine[e]));
          }
          simdgroup_barrier(mem_flags::mem_threadgroup);
          #pragma clang loop unroll(full)
          for (uint i = 0u; i < 2u; ++i)
          #pragma clang loop unroll(full)
            for (uint j = 0u; j < 2u; ++j) simdgroup_load(acc[t][i][j], mine, 16u, ulong2(j * 8u, i * 8u));
          simdgroup_barrier(mem_flags::mem_threadgroup);
        }
      }
    }
#pragma clang loop unroll(full)
    for (uint t = 0u; t < 2u; ++t) {
      if (t == 1u && !two) break;
      const uint n0 = nc + t * 16u;
      if (is_function_constant_defined(FC_GDBG) && FC_GDBG == 4u) {
        float keep = 0.0f;
#pragma clang loop unroll(full)
        for (uint i = 0u; i < 2u; ++i)
#pragma clang loop unroll(full)
          for (uint j = 0u; j < 2u; ++j) keep += acc[t][i][j].thread_elements()[0] + acc[t][i][j].thread_elements()[1];
        if (lane == 0u) Y[row0 * p.y_stride + n0] = half(keep);
        continue;
      }
      #pragma clang loop unroll(full)
      for (uint i = 0u; i < 2u; ++i)
      #pragma clang loop unroll(full)
        for (uint j = 0u; j < 2u; ++j) simdgroup_store(acc[t][i][j], mine, 16u, ulong2(j * 8u, i * 8u));
      simdgroup_barrier(mem_flags::mem_threadgroup);
      const uint rr = lane >> 1u, c0 = (lane & 1u) * 8u;
      const uint limit = p.store_cols == 0u ? FC_N : p.store_cols;
      const uint row = row0 + rr;
      const bool row_ok = p.row_filter == 0u || (((((row / p.width) + p.shift_y) & 7u) >> 1u) & 1u) == 0u;
      if ((full || row < p.tokens) && n0 + c0 < limit && row_ok) {
        device half* y = Y + row * p.y_stride + n0 + c0;
        half v[8];
        #pragma clang loop unroll(full)
        for (uint c = 0u; c < 8u; ++c) {
          half h = half(mine[rr * 16u + c0 + c]);
          if (p.epilogue == 1u) h = max(h, half(0.0h));
          else if (p.epilogue == 2u) h = gemm2_gelu(h);
          else if (p.epilogue == 3u) h = h + Cin[row * p.r_stride + n0 + c0 + c];
          v[c] = h;
        }
        *reinterpret_cast<device half4*>(y) = half4(v[0], v[1], v[2], v[3]);
        *reinterpret_cast<device half4*>(y + 4) = half4(v[4], v[5], v[6], v[7]);
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
}
