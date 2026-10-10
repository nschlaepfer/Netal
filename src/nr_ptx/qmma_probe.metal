// Primitive test of the QMMA instruction helper: one warp per case, lane registers a[4] b[2] c[2] -> d[2] (the RTX operand-probe layout).
#include "nr_ptx_rt.metal"
kernel void qmma_probe(device const uint* in [[buffer(0)]], device uint* out [[buffer(1)]], uint3 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup NrTpScratch nr_tp_scratch; nr_tp_scratch.enabled = true; (void)tg;
  const uint base = (tg.x * 32u + lane) * 8u; uint d0 = 0u, d1 = 0u;
  nr_mma_e4m3(in[base], in[base + 1u], in[base + 2u], in[base + 3u], in[base + 4u], in[base + 5u], in[base + 6u], in[base + 7u], d0, d1, lane, nr_tp_scratch);
  out[(tg.x * 32u + lane) * 2u] = d0; out[(tg.x * 32u + lane) * 2u + 1u] = d1;
}

kernel void qmma_probe_reg(device const uint* in [[buffer(0)]], device uint* out [[buffer(1)]], uint3 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup NrTpScratch nr_tp_scratch; nr_tp_scratch.enabled = false; (void)tg;   // forces the multi-warp path (register-only when NR_QMMA_TP_REG=1)
  const uint base = (tg.x * 32u + lane) * 8u; uint d0 = 0u, d1 = 0u;
  nr_mma_e4m3(in[base], in[base + 1u], in[base + 2u], in[base + 3u], in[base + 4u], in[base + 5u], in[base + 6u], in[base + 7u], d0, d1, lane, nr_tp_scratch);
  out[(tg.x * 32u + lane) * 2u] = d0; out[(tg.x * 32u + lane) * 2u + 1u] = d1;
}
