#!/bin/zsh
# Standalone SR frame-time benchmark (run with the game closed for clean numbers).
# usage: scripts/bench-sr-frame.sh [suffix=_hw] [frames=8]
set -euo pipefail
script_dir="$(cd "$(dirname "$0")" && pwd)"; repository="$(cd "$script_dir/.." && pwd)"
suffix="${1:-_hw}"; frames="${2:-8}"
bundle="$repository/work/re9-sr-runtime-bundle"; out="$repository/work/perf/bench"; mkdir -p "$out"
probe="$repository/build/macos_host/dlss-sr-session-temporal-probe"
run() {
  local name="$1" capture="$2"; shift 2
  rm -f "$out/receipt-$name.txt"
  env "$@" NEURALSTUDIO_SR_PWIN_SUFFIX="$suffix" NEURALSTUDIO_SR_DEPTH_INPUT_LIBRARY=hiluma_input_depthinv \
      NEURALSTUDIO_SR_PWIN_PIPELINE="${PWIN_PIPELINE:-1}" \
      NEURALSTUDIO_SR_TEMPORAL_FRAME_COUNT="$frames" "$probe" "$bundle" "$capture" "$out/receipt-$name.txt" >/dev/null 2>&1 || true
  # a failed probe used to leave the previous receipt in place and the numbers below were silently stale
  [[ -f "$out/receipt-$name.txt" ]] || { print "$name: probe failed (no receipt; the 4-frame captures cap frames at 4)"; return; }
  local ms; ms=$(grep gpu_ms "$out/receipt-$name.txt" | cut -d= -f2 | sort -n | awk '{a[NR]=$1} END {printf "min=%.1f median=%.1f max=%.1f", a[1], a[int((NR+1)/2)], a[NR]}')
  print "$name: $ms (frames=$frames suffix=$suffix)"
}
run 1080p-from-1280x720 "$repository/work/rtx-dlss-sr-temporal-4frame-20260901-v1/capture"
run 1440p "$repository/work/rtx-dlss-sr-temporal-k1440p-4frame-20260901-v1/capture" NEURALSTUDIO_SR_PROFILE=2560x1440
run 4k "$repository/work/rtx-dlss-sr-temporal-k2160p-4frame-20260901-v1/capture" NEURALSTUDIO_SR_PROFILE=3840x2160
run 720p "$repository/work/rtx-dlss-sr-temporal-k720p-4frame-20260901-v1/capture" NEURALSTUDIO_SR_PROFILE=1280x720
