// nr_body_digest.h — CPU mirror of the GPU checkpoint digest (nr_digest64 in nr_body_validate.metal). The buffer is cut into 16-word chunks; each
// chunk is hashed with its index; the chunk hashes are summed in two independent 32-bit halves (modulo 2^32). The sum is commutative, so the GPU
// result is independent of thread scheduling and equals this function bit for bit. Shared by the runtime (v12+) and the validator probe.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
struct NrDigest64 { uint32_t lo = 0, hi = 0; bool operator==(const NrDigest64& o) const { return lo == o.lo && hi == o.hi; } };
inline NrDigest64 nr_digest64_cpu(const uint8_t* p, size_t bytes) {
  NrDigest64 d; if (bytes % 4 != 0) return NrDigest64{0xffffffffu, 0xffffffffu};   // callers reject non-word sizes before use
  const size_t words = bytes / 4, chunks = (words + 15) / 16;
  for (size_t c = 0; c < chunks; ++c) {
    uint64_t h = 0x9E3779B97F4A7C15ull ^ (uint64_t(c) * 0xD1B54A32D192ED03ull);
    const size_t base = c * 16, end = base + 16 < words ? base + 16 : words;
    for (size_t i = base; i < end; ++i) { uint32_t w; std::memcpy(&w, p + 4 * i, 4); h = (h ^ uint64_t(w)) * 0xBF58476D1CE4E5B9ull; h ^= h >> 31; }
    h *= 0x94D049BB133111EBull; h ^= h >> 29;
    d.lo += uint32_t(h); d.hi += uint32_t(h >> 32);
  }
  return d;
}
