#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// Three lowerings of the same NVIDIA HMMA semantics: exact sum of K=16 fp16 products plus the
// prior fp16 accumulator, rounded once to fp16.
//
//   oracle   - integer bit decomposition, the shape FG's fg_hmma_f16_bits_16 uses. Correct, slow.
//   twofloat - exact sum via a double-float accumulator. Exact for K=16 with no MMA and no fallback.
//   checked  - simdgroup MMA plus a provable error bound; falls back to twofloat when the bound
//              cannot decide the fp16 result.
//
// M=8 rows, N=8 cols, K=16, one 8x8 tile per simdgroup, matching mma.sync.m16n8k16's shape.

constant uint TILES [[function_constant(0)]];

// ---------- oracle: align 16 exact products at the largest exponent, sum, round once -------------
static inline float exact_sum_bits(const device half* a, const device half* b,
                                   uint arow, uint bcol, uint bstride) {
    // Each fp16 product is exact in 22 bits; accumulate the aligned integer significands in int64.
    int emin = 127;
    float p[16];
    for (uint k = 0; k < 16u; ++k) {
        p[k] = float(a[arow * 16u + k]) * float(b[k * bstride + bcol]);
        if (p[k] != 0.0f) { int e; frexp(p[k], e); emin = min(emin, e - 24); }
    }
    long acc = 0;
    for (uint k = 0; k < 16u; ++k) {
        if (p[k] == 0.0f) continue;
        int e; float m = frexp(p[k], e);            // m in [0.5,1), p = m * 2^e
        long sig = long(m * 16777216.0f);           // 24-bit significand
        int shift = (e - 24) - emin;
        acc += (shift >= 0) ? (sig << shift) : (sig >> (-shift));
    }
    return float(acc) * exp2(float(emin));
}

kernel void lower_oracle(device const half* A [[buffer(0)]], device const half* B [[buffer(1)]],
                         device half* C [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    uint tile = gid / 64u, e = gid % 64u, i = e / 8u, j = e % 8u;
    if (tile >= TILES) return;
    C[tile * 64u + e] = half(exact_sum_bits(A + tile * 128u, B + tile * 128u, i, j, 8u));
}

// ---------- twofloat: exact sum with a double-float accumulator ----------------------------------
static inline void two_sum(thread float& s, thread float& e, float a, float b) {
    s = a + b; float bb = s - a; e = (a - (s - bb)) + (b - bb);
}
// (hi,lo) stays a normalised double-float, so hi+lo is the correctly rounded fp32 sum. fp32 carries
// 24 significand bits and fp16 needs 11, and 24 >= 2*11+2, so rounding through it to half is
// innocuous - the double-rounding theorem holds exactly at this pairing.
static inline void df_add(thread float& hi, thread float& lo, float x) {
    float s, e; two_sum(s, e, hi, x);
    e += lo;
    float s2, e2; two_sum(s2, e2, s, e);
    hi = s2; lo = e2;
}
kernel void lower_twofloat(device const half* A [[buffer(0)]], device const half* B [[buffer(1)]],
                           device half* C [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    uint tile = gid / 64u, e = gid % 64u, i = e / 8u, j = e % 8u;
    if (tile >= TILES) return;
    const device half* a = A + tile * 128u; const device half* b = B + tile * 128u;
    float hi = 0.0f, lo = 0.0f;
    for (uint k = 0; k < 16u; ++k)
        df_add(hi, lo, float(a[i * 16u + k]) * float(b[k * 8u + j]));   // each product exact in fp32
    C[tile * 64u + e] = half(hi + lo);
}

// ---------- checked: MMA plus a provable bound, twofloat fallback --------------------------------
kernel void lower_checked(device const half* A [[buffer(0)]], device const half* B [[buffer(1)]],
                          device half* C [[buffer(2)]], device atomic_uint* fallbacks [[buffer(3)]],
                          uint tile [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]]) {
    if (tile >= TILES) return;
    const device half* a = A + tile * 128u; const device half* b = B + tile * 128u;
    threadgroup half absA[128], absB[128];
    for (uint t = lane; t < 128u; t += 32u) { absA[t] = abs(a[t]); absB[t] = abs(b[t]); }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_half8x8 a0, a1, b0, b1;
    simdgroup_float8x8 v = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 P0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 P  = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_load(a0, a, 16); simdgroup_load(a1, a + 8, 16);
    simdgroup_load(b0, b, 8);  simdgroup_load(b1, b + 64, 8);
    simdgroup_multiply_accumulate(v, a0, b0, v);
    simdgroup_multiply_accumulate(v, a1, b1, v);
    simdgroup_half8x8 qa0, qa1, qb0, qb1;
    simdgroup_load(qa0, absA, 16); simdgroup_load(qa1, absA + 8, 16);
    simdgroup_load(qb0, absB, 8);  simdgroup_load(qb1, absB + 64, 8);
    simdgroup_multiply_accumulate(P0, qa0, qb0, P0);
    P = P0;
    simdgroup_multiply_accumulate(P, qa1, qb1, P);

    threadgroup float vt[64], p0t[64], pt[64];
    simdgroup_store(v, vt, 8); simdgroup_store(P0, p0t, 8); simdgroup_store(P, pt, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = lane; e < 64u; e += 32u) {
        float val = vt[e];
        // 0.5 * 8 * ulp32(P0) + 0.5 * 8 * ulp32(P) + 0.5 * ulp32(|v|)
        int e0, e1, e2; frexp(p0t[e], e0); frexp(pt[e], e1); frexp(fabs(val), e2);
        float bound = 4.0f * exp2(float(e0 - 24)) + 4.0f * exp2(float(e1 - 24))
                    + 0.5f * exp2(float(e2 - 24));
        half lo = half(val - bound), hi = half(val + bound);
        if (as_type<ushort>(lo) == as_type<ushort>(hi)) {
            C[tile * 64u + e] = half(val);
        } else {
            atomic_fetch_add_explicit(fallbacks, 1u, memory_order_relaxed);
            uint i = e / 8u, j = e % 8u;
            float h = 0.0f, l = 0.0f;
            for (uint k = 0; k < 16u; ++k)
                df_add(h, l, float(a[i * 16u + k]) * float(b[k * 8u + j]));
            C[tile * 64u + e] = half(h + l);
        }
    }
}
