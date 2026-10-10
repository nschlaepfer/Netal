#!/bin/zsh
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
out="${1:?new output directory}"
[[ ! -e "$out" ]] || exit 2
/opt/homebrew/bin/python3.13 - "$repo/../NeuralStudio-nr-fable/scripts/nr-run-recipe.py" <<'PY'
import importlib.util,sys
s=importlib.util.spec_from_file_location('guard',sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
ok,why=m.held_reservation('Codex');print(why)
if not ok:raise SystemExit('Codex GPU reservation required')
PY
mkdir -p "$out/source"
cp -p "$repo/src/macos_host/netal_nr_color_transfer.metal" "$repo/src/macos_host/netal_nr_color_transfer_audit.mm" "$out/source/"
cp -p "$0" "$out/source/test-as-run.sh"
xcrun --sdk macosx metal -c -O2 -fno-fast-math -std=metal3.1 "$out/source/netal_nr_color_transfer.metal" -o "$out/color.air"
xcrun --sdk macosx metallib "$out/color.air" -o "$out/color.metallib"
xcrun --sdk macosx clang++ -arch arm64 -std=c++20 -fobjc-arc -O2 -Wall -Wextra -Werror -framework Foundation -framework Metal "$out/source/netal_nr_color_transfer_audit.mm" -o "$out/color-audit"
shasum -a 256 "$out/source"/* "$out/color-audit" "$out/color.metallib" > "$out/pins.txt"
MTL_DEBUG_LAYER=1 "$out/color-audit" "$out/color.metallib" \
  "$repo/build/nr-game-input-replay-v1/frames/frame-0-stream-7/model-input.rgba16f" \
  "$repo/build/nr-bounded-color-control-v1/input-color.rgba16le" "$out/gpu-bounded.rgba16le" > "$out/audit.log" 2>&1
cat "$out/audit.log"
shasum -a 256 "$out/gpu-bounded.rgba16le" "$out/audit.log" > "$out/result-sha256.txt"
