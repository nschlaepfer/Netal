#include <metal_stdlib>
using namespace metal;

// Block-ordered compaction: each 256-element threadgroup ranks its flagged elements with a prefix sum,
// reserves one contiguous span with a single atomic, and writes them in index order. Blocks land in
// arbitrary order but every block's flagged elements are contiguous and sorted, so adjacent repair
// lanes share a pixel's input taps and neighbouring weights instead of hitting unrelated pixels.
kernel void fg_compact_suspects_ordered(device const uchar* suspect [[buffer(0)]],
                                        device uint* list [[buffer(1)]],
                                        device atomic_uint* counter [[buffer(2)]],
                                        constant uint& total [[buffer(3)]],
                                        uint gid [[thread_position_in_grid]],
                                        uint lid [[thread_index_in_threadgroup]],
                                        uint simd_id [[simdgroup_index_in_threadgroup]],
                                        uint lane [[thread_index_in_simdgroup]]) {
  threadgroup uint simd_totals[8];
  threadgroup uint block_base;
  const uint flag = (gid < total && suspect[gid] != 0u) ? 1u : 0u;
  const uint rank_in_simd = simd_prefix_exclusive_sum(flag);
  const uint simd_total = simd_sum(flag);
  if (lane == 0u) simd_totals[simd_id] = simd_total;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid == 0u) {
    uint run = 0u;
    for (uint s = 0u; s < 8u; ++s) { const uint t = simd_totals[s]; simd_totals[s] = run; run += t; }
    block_base = run != 0u ? atomic_fetch_add_explicit(counter, run, memory_order_relaxed) : 0u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (flag != 0u) list[block_base + simd_totals[simd_id] + rank_in_simd] = gid;
}
