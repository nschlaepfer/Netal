#include <metal_stdlib>
using namespace metal;

// Block-ordered compaction (as fg_compact_suspects_ordered) that also writes its own indirect-dispatch
// arguments, so the tiny fg_indirect_from_counter dispatch that used to follow it disappears.
// Every threadgroup, after its single span-reserving atomic on `counter`, bumps a second "groups done"
// counter; the group that sees it reach total_groups-1 is the last one, every other group's counter add
// is complete, and it writes indirect = {ceil(counter/group_size), 1, 1} from the final counter value.
// Ordering: release side = threadgroup_barrier(mem_device) + device-scope seq_cst fence between the
// counter add and the done add; acquire side = the same fence after the done add, before the final read.
// counter, done and indirect are zeroed on the CPU before each run. Needs -std=metal3.2 (atomic_thread_fence).
kernel void fg_compact_suspects_ordered_ind(device const uchar* suspect [[buffer(0)]],
                                            device uint* list [[buffer(1)]],
                                            device atomic_uint* counter [[buffer(2)]],
                                            constant uint& total [[buffer(3)]],
                                            device atomic_uint* done [[buffer(4)]],
                                            device uint* indirect [[buffer(5)]],
                                            constant uint& group_size [[buffer(6)]],
                                            uint gid [[thread_position_in_grid]],
                                            uint lid [[thread_index_in_threadgroup]],
                                            uint simd_id [[simdgroup_index_in_threadgroup]],
                                            uint lane [[thread_index_in_simdgroup]],
                                            uint total_groups [[threadgroups_per_grid]]) {
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
  // release: this group's counter add (and list writes) are ordered before its done add
  threadgroup_barrier(mem_flags::mem_device);
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  if (lid == 0u) {
    const uint prev = atomic_fetch_add_explicit(done, 1u, memory_order_relaxed);
    if (prev == total_groups - 1u) {
      // acquire: every other group's done add, hence its counter add, is ordered before this read
      atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
      const uint c = atomic_load_explicit(counter, memory_order_relaxed);
      indirect[0] = (c + group_size - 1u) / group_size;
      indirect[1] = 1u;
      indirect[2] = 1u;
    }
  }
}
