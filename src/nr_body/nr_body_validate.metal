// nr_body_validate.metal — resident-body GPU evidence kernels (contract v2, docs/NR_RESIDENT_ENCODE_CONTRACT.md).
// v3 (2026-09-06, Codex review finding 3): the first-failure identity is no longer a packed (element << 8 | count) key (it overflowed above 2^24
// elements; valid pool indices reach 53,129,983). Each validator record now carries first_elem (full 32-bit index, atomic min over faulting
// threads) and first_obs (its observed count, written by nr_cov_first_obs from the still-intact counters right after the validator). The global
// sticky record keeps fault_flag / fault_count / validators_run only.
// v2 (2026-09-06): the per-validator fault total lives in the completion record itself (atomic field), so no second writer kernel is needed and
// records[] alone proves faults-per-validator; modes: 0 = every element exactly once, 1 = expected map (1 = exactly once, 0 = padding, never written),
// 2 = at most once (decoder pool words). Added nr_digest64 (deterministic, order-independent 64-bit digest of a word buffer into a checkpoint table)
// and nr_apply_pad_mask (GPU replacement of the driver's padding-token zeroing before repack 58).
//
// nr_cov_validate checks one launch's coverage slice BEFORE it is cleared for reuse. counters[i] is the write count of element i. Any deviation
// - missing, duplicate, or padding write - sets the sticky fault_flag, counts faults, and records the FIRST failing element of the validator (lowest index, atomic min,
// full 32-bit) in its completion record (its observed count is copied by nr_cov_first_obs) and increments the per-frame fault count. Every validator, faulting or not, atomically increments
// `validators_run` and writes its completion record (launch_id, elements checked, faults seen, done) at records[launch_slot], so the CPU can prove at
// completion that the expected number of validators ran and that none was skipped or partial (the CPU poisons records[] before the frame).
#include <metal_stdlib>
using namespace metal;
struct NrValidateArgs { uint count; uint launch_id; uint launch_slot; uint mode; };
struct NrStickyRecord { atomic_uint fault_flag; atomic_uint reserved; atomic_uint fault_count; atomic_uint validators_run; };
struct NrValidatorRecord { uint launch_id; uint elements; atomic_uint faults; uint done; atomic_uint first_elem; uint first_obs; uint pad0; uint pad1; };   // first_elem: lowest faulting element (init 0xffffffff by the CPU); first_obs: its counter, filled by nr_cov_first_obs
kernel void nr_cov_validate(device const uint* counters [[buffer(0)]], device const uchar* expect [[buffer(1)]], device NrStickyRecord* sticky [[buffer(2)]], device NrValidatorRecord* records [[buffer(3)]], constant NrValidateArgs& a [[buffer(4)]], uint gid [[thread_position_in_grid]], uint tpg [[threads_per_grid]]) {
  uint local = 0u;
  for (uint i = gid; i < a.count; i += tpg) {
    const uint c = counters[i]; bool bad;
    if (a.mode == 1u) bad = c != uint(expect[i]); else if (a.mode == 2u) bad = c > 1u; else bad = c != 1u;
    if (bad) { ++local; atomic_fetch_or_explicit(&sticky->fault_flag, 1u, memory_order_relaxed); atomic_fetch_min_explicit(&records[a.launch_slot].first_elem, i, memory_order_relaxed); }
  }
  if (local) { atomic_fetch_add_explicit(&sticky->fault_count, local, memory_order_relaxed); atomic_fetch_add_explicit(&records[a.launch_slot].faults, local, memory_order_relaxed); }
  if (gid == 0u) {   // completion record: written by exactly one thread; faults accumulate atomically from every thread
    records[a.launch_slot].launch_id = a.launch_id; records[a.launch_slot].elements = a.count; records[a.launch_slot].done = 0xC0DE0001u;
    atomic_fetch_add_explicit(&sticky->validators_run, 1u, memory_order_relaxed);
  }
}
// second pass (one thread, encoded right after nr_cov_validate while the counters are still intact): observed count of the first faulting element
kernel void nr_cov_first_obs(device const uint* counters [[buffer(0)]], device NrValidatorRecord* records [[buffer(1)]], constant NrValidateArgs& a [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid != 0u) return; const uint e = atomic_load_explicit(&records[a.launch_slot].first_elem, memory_order_relaxed); records[a.launch_slot].first_obs = (e < a.count) ? counters[e] : 0xffffffffu;
}
// Deterministic 64-bit digest of `words` 32-bit words: the buffer is cut into 16-word chunks, each chunk is hashed together with its index, and the
// chunk hashes are summed (low and high 32-bit halves separately, modulo 2^32). The sum is commutative, so the result does not depend on thread
// scheduling; the CPU mirror (Digest64 in nr_body_runtime_v12.mm) computes the same value from the same bytes.
struct NrDigestArgs { uint words; uint launch_id; uint slot; uint reserved; };
struct NrDigestRecord { uint launch_id; uint words; atomic_uint lo; atomic_uint hi; uint done; uint pad0; uint pad1; uint pad2; };
kernel void nr_digest64(device const uint* data [[buffer(0)]], device NrDigestRecord* rec [[buffer(1)]], constant NrDigestArgs& a [[buffer(2)]], uint gid [[thread_position_in_grid]], uint tpg [[threads_per_grid]]) {
  const uint chunks = (a.words + 15u) / 16u; uint lo = 0u, hi = 0u;
  for (uint c = gid; c < chunks; c += tpg) {
    ulong h = 0x9E3779B97F4A7C15ul ^ (ulong(c) * 0xD1B54A32D192ED03ul);
    const uint base = c * 16u, end = min(base + 16u, a.words);
    for (uint i = base; i < end; ++i) { h = (h ^ ulong(data[i])) * 0xBF58476D1CE4E5B9ul; h ^= h >> 31; }
    h *= 0x94D049BB133111EBul; h ^= h >> 29;
    lo += uint(h); hi += uint(h >> 32);
  }
  atomic_fetch_add_explicit(&rec[a.slot].lo, lo, memory_order_relaxed); atomic_fetch_add_explicit(&rec[a.slot].hi, hi, memory_order_relaxed);
  if (gid == 0u) { rec[a.slot].launch_id = a.launch_id; rec[a.slot].words = a.words; rec[a.slot].done = 0xD16E5701u; }
}
// data[i] = 0 wherever mask[i] != 0 (byte granularity): the accepted driver rule "padding tokens' bytes are zero before repack 58", moved onto the GPU
kernel void nr_apply_pad_mask(device uchar* data [[buffer(0)]], device const uchar* mask [[buffer(1)]], constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) { if (gid < n && mask[gid] != 0) data[gid] = 0; }
