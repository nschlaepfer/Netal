# Pinned reference sources (read-only in this worktree)

Verbatim copies of the accepted exact-lane sources the experimental runtime is derived from. They are here so the
derivation is auditable inside this worktree; **nothing in this directory is compiled**, and nothing here may be
edited. The authoritative copies live in the exact lane and are unchanged by this worktree.

| Copy | Source of truth |
|---|---|
| `encoder-zero-v3-source/` | `NeuralStudio-nr-first-inference/build/nr_encoder_telemetry/zero-v3/source` (encoder object `942541e1…`) |
| `nr_body_runtime_v17.mm`, `nr_body_runtime.h`, `nr_body_digest.h` | `NeuralStudio-nr-fable/src/nr_body/` |
| `nr_temporal/` | `NeuralStudio-nr-fable/src/nr_temporal/` |

The experimental runtime in `../runtime/` re-emits the same dispatches with the same weights as encode-only work on
a caller-owned command buffer. It removes research-only validation (per-dispatch commit and wait, coverage
buffers and their CPU scans, SHA-256 of activations, diagnostic dumps and stdout). It changes no weights.
