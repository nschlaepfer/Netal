#!/usr/bin/env bash
# Build the FG custom block (launches 56-83) with the guarded pipelines and verify every launch checkpoint
# against the RTX 5090 hashes recorded in the Mac-fg-research receipt.
#
#   final/build_and_verify.sh <Mac-fg-research checkout> [model dir] [reps] [formal 0|1]
#
# The kernels here are drop-ins for that checkout's src/fg_conv (new file names only; nothing there is
# overwritten). The scalar oracles, the connected bottom kernels, the block contract and the probe helpers
# come from the checkout. Requires Xcode command line tools (metal, metallib, clang++).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
fg="${1:?Mac-fg-research checkout path}"; reps="${3:-15}"; formal="${4:-1}"
sub="build/fg_default_readback/eval0-default-all-a5-a/runtime-default-buffers-post-evaluate-0"
model="${2:-}"; if [[ -z "$model" ]]; then for c in "$fg/$sub" "$fg/../NeuralStudio-fg-research/$sub"; do [[ -f "$c/default-token-000106.bin" ]] && { model="$c"; break; }; done; fi
[[ -n "$model" && -f "$model/default-token-000106.bin" ]] || { echo "model tokens not found; pass the runtime-default-buffers-post-evaluate-0 directory as the second argument" >&2; exit 2; }
[[ -d "$fg/src/fg_conv" && -f "$fg/src/fg_connected_custom_l56_l83/contract_full.h" ]] || { echo "not a Mac-fg-research checkout: $fg" >&2; exit 2; }
# stage the pipeline kernels and headers (they include hmma_exact.metalh from the checkout)
for f in "$here"/*.metal "$here"/*.metalh; do
  dst="$fg/src/fg_conv/$(basename "$f")"
  if [[ -e "$dst" ]] && ! cmp -s "$f" "$dst"; then echo "note: $(basename "$f") differs from the checkout copy; using the final/ version"; fi
  cp "$f" "$dst"
done
mkdir -p "$fg/src/fg_custom_upsample" "$fg/src/fg_connected_custom_l56_l83"
cp "$here/connected/custom_upsample_planes3.metal" "$fg/src/fg_custom_upsample/"
cp "$here/connected/macos_connected_pipelines.mm" "$fg/src/fg_connected_custom_l56_l83/"
cp "$here/connected/build_connected_pipelines.sh" "$fg/build_connected_pipelines.sh"; chmod +x "$fg/build_connected_pipelines.sh"
( cd "$fg" && ./build_connected_pipelines.sh src/fg_conv/custom_block1_bot0_pipeline_wq.metal src/fg_conv/custom_block0_bot1_pipeline_wq.metal src/fg_conv/custom_block1_bot1_pipeline_wq.metal )
"$fg/build_cp/host" "$fg/build_cp/libs" "$model" "$fg/evidence/specs/fg-connected-custom-l56-l83-v1/verification.json" "$reps" "$formal"
