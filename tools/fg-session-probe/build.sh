#!/bin/zsh
# Offline gate for the FG session: interpolator-shaped textures through MTL4 command buffers.
#   tools/fg-session-probe/build.sh && build/fg-session-probe/probe work/re9-fg-runtime-bundle 8
# Geometry via PROBE_W/H (textures), PROBE_CW/CH (content), PROBE_RW/RH (render); PROBE_DUMP=<dir>
# writes each output; NEURALSTUDIO_FG_FULLRES=0 selects the 640x360 upsample path.
set -euo pipefail
repo="$(cd "$(dirname "$0")/../.." && pwd)"; out="$repo/build/fg-session-probe"; mkdir -p "$out"
xcrun --sdk macosx clang++ -fobjc-arc -std=c++20 -O2 -mmacosx-version-min=26.0 \
  -I "$repo/work/re9-fg-runtime-bundle/include" -I "$repo/src/macos_host" \
  "$repo/tools/fg-session-probe/fg_session_probe.mm" "$repo/src/macos_host/dlss_fg_session.mm" \
  "$repo/work/re9-fg-runtime-bundle/lib/libnetal_fg_runtime.a" -framework Foundation -framework Metal -o "$out/probe"
print "built=$out/probe"
