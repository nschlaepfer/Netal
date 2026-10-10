#include <metal_stdlib>
using namespace metal;

// Compaction, because a mask over a scattered third of a dispatch is not a third of a dispatch.
// The measured two-pass was 0.91x precisely because (1-0.3322)^32 is about one in 170,000, so every
// threadgroup still contained live work. Making the flagged set dense is what turns a 33% mask into
// a 33% dispatch.
kernel void fg_compact_suspects(device const uchar* suspect [[buffer(0)]],
                                device uint* list [[buffer(1)]],
                                device atomic_uint* counter [[buffer(2)]],
                                constant uint& total [[buffer(3)]],
                                uint gid [[thread_position_in_grid]]) {
  if (gid >= total) return;
  if (suspect[gid] != 0u) {
    list[atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed)] = gid;
  }
}

// Threadgroup count for an indirect dispatch, so the repair pass launches only what it needs.
kernel void fg_indirect_from_counter(device const uint* counter [[buffer(0)]],
                                     device uint* indirect [[buffer(1)]],
                                     constant uint& group_size [[buffer(2)]],
                                     uint gid [[thread_position_in_grid]]) {
  if (gid != 0u) return;
  indirect[0] = (counter[0] + group_size - 1u) / group_size;
  indirect[1] = 1u;
  indirect[2] = 1u;
}
