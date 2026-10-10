#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
attempt="${1:-$repo/build/fg_connected_l3_l19/macos/attempt-$(date -u +%Y%m%dT%H%M%SZ)}"
model="${2:-$repo/build/fg_default_readback/eval0-default-all-a5-a/runtime-default-buffers-post-evaluate-0}"
rom="${3:-$repo/build/fg_connected_tail/macos-l56-l87/attempt-exact-v1/sm120-rcp-correction-rom.bin}"
if [[ -e "$attempt" ]]; then
  echo "Attempt exists; preserve it: $attempt" >&2
  exit 2
fi
mkdir -p "$attempt"
required=(
  src/fg_early_l3_l13/early_l3_l13.metal
  src/fg_early_l4_l14/early_l4_l14.metal
  src/fg_early_l5_l15/early_l5_l15.metal
  src/fg_early_l6_l16/early_l6_l16.metal
  src/fg_early_l7_l17/early_l7_l17.metal
  src/fg_early_l8_l18/early_l8_l18.metal
  src/fg_early_l9/early_l9.metal
  src/fg_early_status_l10_l12/status_write.metal
  src/fg_early_l11/early_l11.metal
  src/fg_early_l19/early_l19.metal
  src/fg_conv/conv_3x3.metal
)
missing=()
for path in "${required[@]}"; do
  if ! git -C "$repo" cat-file -e "HEAD:$path" 2>/dev/null; then
    missing+=("$path")
  fi
done
if ((${#missing[@]})); then
  python3 - "$attempt/dependency-blocker.json" "${missing[@]}" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
result = {
    "schema": "neuralstudio.fg.connected-l3-l19-dependency-blocker.v1",
    "missing_committed_paths": sys.argv[2:],
    "uncommitted_files_ignored": True,
    "result": "blocked-on-commits",
}
path.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
PY
  echo "Connected L3-L19 dependencies are not committed" >&2
  exit 4
fi

libs="$attempt/libs"
mkdir -p "$libs"
compile_metal() {
  local source="$1" name="$2"
  xcrun -sdk macosx metal -std=metal3.1 -fno-fast-math \
    -ffp-contract=off -I "$repo/src/fg_conv" -c "$repo/$source" \
    -o "$libs/$name.air"
  xcrun -sdk macosx metallib "$libs/$name.air" -o "$libs/$name.metallib"
}
compile_metal src/fg_early_l3_l13/early_l3_l13.metal l3_l13
compile_metal src/fg_early_l4_l14/early_l4_l14.metal l4_l14
compile_metal src/fg_early_l5_l15/early_l5_l15.metal l5_l15
compile_metal src/fg_early_l6_l16/early_l6_l16.metal l6_l16
compile_metal src/fg_early_l7_l17/early_l7_l17.metal l7_l17
compile_metal src/fg_early_l8_l18/early_l8_l18.metal l8_l18
compile_metal src/fg_early_l9/early_l9.metal l9
compile_metal src/fg_early_status_l10_l12/status_write.metal l10_l12
compile_metal src/fg_early_l11/early_l11.metal l11
compile_metal src/fg_early_l19/early_l19.metal l19
compile_metal src/fg_conv/conv_3x3.metal l20
xcrun -sdk macosx clang++ -std=c++20 -fobjc-arc -Wall -Wextra -Werror \
  -Wno-unused-function "$repo/src/fg_connected_l3_l19/macos_connected.mm" \
  -framework Foundation -framework Metal -o "$attempt/host"
for poison in a5 5a; do
  for ordinal in a b; do
    run_id="connected-l3-l19-poison-$poison-process-$ordinal"
    run="$attempt/$run_id"
    mkdir -p "$run"
    "$attempt/host" "$libs" "$model" "$rom" "$run" "$run_id" "$poison" \
      >"$run/stdout.log" 2>"$run/stderr.log"
  done
done
echo "attempt_root=$attempt"
echo "result=execution-complete-verification-required"
