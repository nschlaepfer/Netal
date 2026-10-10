// Batched per-token GEMM, version 3: same numerics as pwin_gemm2_tokens (hw3 accumulation order, fp16
// rounding points, split-K modes, epilogues) with register-only epilogues (no threadgroup staging,
// no barriers after the weight copy), compile-time shapes and strides, weights in threadgroup memory
// when K*N <= 4096 halves (else read from device memory), and elementwise half2 stores that handle
// the partial last block, store_cols and the row filter uniformly.
#include <metal_stdlib>
using namespace metal;

// FC_K (0), FC_N (1), FC_XS (2) and FC_CS (3) are declared in pwin_gemm2.metal (the kernels are built as one
// concatenated translation unit by build-all.sh).
constant uint FC_YS [[function_constant(7)]];   // Y row stride
constant uint FC_WS [[function_constant(8)]];   // W row stride (0 = N)
constant uint FC_RS [[function_constant(9)]];   // residual row stride (epilogue 3)
constant uint FC_NSPLIT [[function_constant(16)]];   // optional: SIMD groups per 16-token block, each owning FC_N / FC_NSPLIT columns (small layers)

struct Gemm3Params {
  uint tokens, epilogue, c_mode, split_k16, width, shift_y, store_cols, row_filter;
};

static inline half gemm3_gelu(half h) {
  const half t = min(max(h, half(-2.0h)), half(2.0h));
  const half u = fma(-half(0x1.4c1bacp-4f), abs(t), half(0x1.a60dd6p-2f));
  const half s = fma(t, u, half(0.5h));
  return h * s;
}

kernel void pwin_gemm3_tokens(device const half* X [[buffer(0)]],
                              device const half* W [[buffer(1)]],
                              device const float* C [[buffer(2)]],
                              device half* Y [[buffer(3)]],
                              constant Gemm3Params& p [[buffer(4)]],
                              device const half* Cin [[buffer(5)]],
                              device const half* Cbias [[buffer(6)]],
                              uint simd_index [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint lid [[thread_index_in_threadgroup]],
                              uint3 tg [[threadgroup_position_in_grid]],
                              uint simdgroups [[simdgroups_per_threadgroup]],
                              uint3 threads3 [[threads_per_threadgroup]]) {
  const uint ws = FC_WS == 0u ? FC_N : FC_WS;
  const bool w_in_tg = FC_K * FC_N <= 4096u;
  threadgroup half Wt[4096];
  if (w_in_tg) {
    const uint chunks = FC_K * FC_N / 4u;
    for (uint e = lid; e < chunks; e += threads3.x) {
      const uint h = e * 4u;
      const uint kk = h / FC_N, nn = h - kk * FC_N;
      *reinterpret_cast<threadgroup half4*>(Wt + h) = *reinterpret_cast<device const half4*>(W + kk * ws + nn);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const uint nsplit = (is_function_constant_defined(FC_NSPLIT) && FC_NSPLIT != 0u) ? FC_NSPLIT : 1u;
  const uint group = tg.x * simdgroups + simd_index;
  const uint row0 = (group / nsplit) * 16u;
  const uint ncols = FC_N / nsplit;
  const uint nbase = (group % nsplit) * ncols;
  if (row0 >= p.tokens) return;
  const uint er = ((lane >> 1u) & 3u) + 4u * ((lane >> 4u) & 1u);
  const uint ec = 2u * (lane & 1u) + 4u * ((lane >> 3u) & 1u);
  device const half* x = X + row0 * FC_XS;
  const uint ktiles = FC_K / 16u;
  simdgroup_float8x8 A[4][2][2];
#pragma clang loop unroll(full)
  for (uint kt = 0u; kt < 4u; ++kt) {
    if (kt < ktiles) {
#pragma clang loop unroll(full)
      for (uint i = 0u; i < 2u; ++i)
#pragma clang loop unroll(full)
        for (uint j = 0u; j < 2u; ++j) {
          simdgroup_half8x8 hm;
          simdgroup_load(hm, x + kt * 16u + j * 8u, FC_XS, ulong2(0, i * 8u));
          A[kt][i][j].thread_elements()[0] = float(hm.thread_elements()[0]);
          A[kt][i][j].thread_elements()[1] = float(hm.thread_elements()[1]);
        }
    }
  }
  uint rowv[2]; bool rok[2]; uint blk[2];
#pragma clang loop unroll(full)
  for (uint i = 0u; i < 2u; ++i) {
    rowv[i] = row0 + 8u * i + er;
    rok[i] = rowv[i] < p.tokens;
    const uint rr = min(rowv[i], p.tokens - 1u);
    blk[i] = (((rr / p.width) + p.shift_y) & 7u) >> 1u;
  }
  const uint limit = p.store_cols == 0u ? FC_N : p.store_cols;
  for (uint n0 = nbase; n0 < nbase + ncols; n0 += 16u) {
    simdgroup_float8x8 acc[2][2];
    if (p.c_mode == 0u) {
#pragma clang loop unroll(full)
      for (uint i = 0u; i < 2u; ++i)
#pragma clang loop unroll(full)
        for (uint j = 0u; j < 2u; ++j) simdgroup_load(acc[i][j], C + n0 + j * 8u, FC_CS, ulong2(0, i * 8u));
    } else {
#pragma clang loop unroll(full)
      for (uint i = 0u; i < 2u; ++i) {
        const uint rr = min(rowv[i], p.tokens - 1u);
#pragma clang loop unroll(full)
        for (uint j = 0u; j < 2u; ++j) {
          const half2 cv = *reinterpret_cast<device const half2*>(Cin + rr * FC_CS + n0 + 8u * j + ec);
          const half2 cb = *reinterpret_cast<device const half2*>(Cbias + n0 + 8u * j + ec);
          acc[i][j].thread_elements()[0] = float(cv.x + cb.x);
          acc[i][j].thread_elements()[1] = float(cv.y + cb.y);
        }
      }
    }
#pragma clang loop unroll(full)
    for (uint kt = 0u; kt < ktiles; ++kt) {
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
            simdgroup_load(hm, x + kt * 16u + j * 8u, FC_XS, ulong2(0, i * 8u));
            a[i][j].thread_elements()[0] = float(hm.thread_elements()[0]);
            a[i][j].thread_elements()[1] = float(hm.thread_elements()[1]);
          }
      }
      simdgroup_float8x8 b[2][2];
#pragma clang loop unroll(full)
      for (uint i = 0u; i < 2u; ++i)
#pragma clang loop unroll(full)
        for (uint j = 0u; j < 2u; ++j) {
          simdgroup_half8x8 hm;
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
      // split modes (low nibble of split_k16; the high bits carry the round count for 5/6):
      // 1/2 round N tile 0 after K tile 0 (2: window blocks 0..2 only); 3 round every tile after K tile 1
      // (K = 64 as two halves); 4 round after every second K tile (chained groups of 32 with fp16 handoffs);
      // 5/6 round the first 16 output columns after each of the leading `srounds` K tiles, matching the
      // fused MLP's expand split (5 = each warp's first token block only, 6 = every token block).
      const uint smode = p.split_k16 & 15u;
      const uint srounds = (p.split_k16 >> 4) != 0u ? (p.split_k16 >> 4) : (ktiles - 1u);
      const bool split_here = ((smode == 1u || smode == 2u) && kt == 0u && n0 == 0u && ktiles > 1u) ||
                              (smode == 3u && kt == 1u) ||
                              (smode == 4u && (kt & 1u) == 1u && kt + 1u < ktiles) ||
                              ((smode == 5u || smode == 6u) && nbase + n0 == 0u && kt < srounds);
      if (split_here) {
#pragma clang loop unroll(full)
        for (uint i = 0u; i < 2u; ++i) {
          if (smode == 2u && blk[i] == 3u) continue;
          if (smode == 5u && (blk[i] & 1u) != 0u) continue;
#pragma clang loop unroll(full)
          for (uint j = 0u; j < 2u; ++j) {
            acc[i][j].thread_elements()[0] = float(half(acc[i][j].thread_elements()[0]));
            acc[i][j].thread_elements()[1] = float(half(acc[i][j].thread_elements()[1]));
          }
        }
      }
    }
#pragma clang loop unroll(full)
    for (uint i = 0u; i < 2u; ++i) {
      if (!rok[i]) continue;
      if (p.row_filter != 0u && (blk[i] & 1u) != 0u) continue;
#pragma clang loop unroll(full)
      for (uint j = 0u; j < 2u; ++j) {
        const uint col = n0 + 8u * j + ec;
        if (col >= limit) continue;
        half h0 = half(acc[i][j].thread_elements()[0]);
        half h1 = half(acc[i][j].thread_elements()[1]);
        if (p.epilogue == 1u) { h0 = max(h0, half(0.0h)); h1 = max(h1, half(0.0h)); }
        else if (p.epilogue == 2u) { h0 = gemm3_gelu(h0); h1 = gemm3_gelu(h1); }
        else if (p.epilogue == 3u) {
          const half2 rv = *reinterpret_cast<device const half2*>(Cin + rowv[i] * FC_RS + col);
          h0 = h0 + rv.x; h1 = h1 + rv.y;
        }
        *reinterpret_cast<device half2*>(Y + rowv[i] * FC_YS + col) = half2(h0, h1);
      }
    }
  }
}
