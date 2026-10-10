#!/bin/zsh
# Generate the batched pipeline set for an output resolution the bundle has no profile for.
#
#   scripts/netal-make-pipelines.sh 1920x1200 [renderWxH]
#
# Without a set, a synthesized geometry silently runs the transpiled kernels: correct, but about
# twice as slow (16.4 ms against 8.1 ms for one game measured at 1920x1200). Every gain from the
# batched pipelines is simply unused.
#
# Sets are keyed by OUTPUT resolution only, and that is not an approximation: the layer token grids
# were compared for render 1728x1080 and 960x600 at the same output and are identical. So one set
# serves every MetalFX or DLSS quality mode at that output, and only a window or display change
# needs a new one. The render size still has to be supplied for the export step, because the
# session synthesizes a complete geometry; any plausible value gives the same pipelines.
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"; repository="$(cd "$script_dir/.." && pwd)"
bundle="$repository/work/re9-sr-runtime-bundle"

output="${1:-}"
[[ -n "$output" ]] || { print -u2 "usage: $0 <outputWxH> [renderWxH]"; exit 64; }
[[ "$output" == <->x<-> ]] || { print -u2 "output must look like 1920x1200"; exit 64; }

# Default render: two thirds of output, roughly DLSS Quality. Only the export needs it.
if [[ -n "${2:-}" ]]; then
  render="$2"
else
  ow="${output%x*}"; oh="${output#*x}"
  render="$(( ow * 2 / 3 ))x$(( oh * 2 / 3 ))"
fi

# Refuse early, and say why, for a resolution the bundle ships a real profile for: synthesis only
# runs when there is no profile, so the export below would produce nothing and fail obscurely.
if [[ -d "$bundle/profiles/$output/pipeline" ]]; then
  print "$output ships with the bundle ($(ls "$bundle/profiles/$output/pipeline"/*.json 2>/dev/null | wc -l | tr -d ' ') pipelines); nothing to generate"
  exit 0
fi

target="$bundle/pipeline-synth/$output"
if [[ -d "$target" && -n "$(print -rn -- "$target"/*.json(N))" ]]; then
  print "already present: $target ($(ls "$target"/*.json | wc -l | tr -d ' ') pipelines)"
  exit 0
fi

export_dir="$(mktemp -d)"
trap 'rm -rf "$export_dir"' EXIT

print "exporting synthesized profile  render=$render output=$output"
# The session owns the synthesis rules, so the artefacts come from it rather than being recomputed.
# It exits non-zero on the capture (the frame geometry will not match); only the export matters.
env NEURALSTUDIO_SR_PROFILE="$output" NEURALSTUDIO_SR_RENDER="$render" \
    NEURALSTUDIO_SR_EXPORT_SYNTH_DIR="$export_dir" \
    NEURALSTUDIO_SR_PWIN_SUFFIX="${PWIN_SUFFIX:-_hw7}" \
    NEURALSTUDIO_SR_TEMPORAL_FRAME_COUNT=1 \
    "$repository/build/macos_host/dlss-sr-session-temporal-probe" \
    "$bundle" "$repository/work/rtx-dlss-sr-temporal-4frame-20260901-v1/capture" \
    "$export_dir/receipt.txt" >/dev/null 2>&1 || true

[[ -d "$export_dir/pwin" ]] || { print -u2 "export produced no pwin/ - is the runtime bundle intact?"; exit 1; }

print "building pipelines"
python3 "$repository/tools/pwin_pipeline/build-profile-pipelines.py" --synth "$export_dir" "$output" >/dev/null

count="$(ls "$target"/*.json 2>/dev/null | wc -l | tr -d ' ')"
[[ "$count" -gt 0 ]] || { print -u2 "no pipelines were written to $target"; exit 1; }
print "built $count pipelines -> pipeline-synth/$output"
