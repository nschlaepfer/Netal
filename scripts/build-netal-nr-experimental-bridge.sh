#!/bin/zsh
# Isolated NR bridge. Default build is observation-only and makes no NR claim.
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
out="$repo/build/nr-experimental-bridge"
delayed="${NR_BRIDGE_DELAYED:-0}"
[[ "$delayed" == 0 || "$delayed" == 1 ]] || exit 2
[[ "$delayed" == 1 ]] && out="$repo/build/nr-experimental-delayed-bridge"
if [[ "$delayed" == 1 ]]; then
  runtime_headers="${NR_BRIDGE_RUNTIME_HEADERS:?matching frozen runtime headers required}"
  [[ -f "$runtime_headers/nr_experimental_api.h" && -f "$runtime_headers/nr_experimental_reset_worker.h" &&
     -f "$runtime_headers/runtime/nr_encoder_scope.h" ]] || exit 2
fi
mkdir -p "$out"
# Reuse Netal's existing pixel codecs only, not its SR network. Retain build files.
xcrun --sdk macosx metal -c -O2 -std=metal3.1 "$repo/src/macos_host/netal_texture_import_ext.metal" -o "$out/formats.air"
xcrun --sdk macosx metal -c -O2 -std=metal3.1 "$repo/src/macos_host/netal_nr_publication_guard.metal" -o "$out/guard.air"
xcrun --sdk macosx metal -c -O2 -std=metal3.1 "$repo/src/macos_host/netal_nr_aspect.metal" -o "$out/aspect.air"
xcrun --sdk macosx metal -c -O2 -fno-fast-math -std=metal3.1 "$repo/src/macos_host/netal_nr_color_transfer.metal" -o "$out/color.air"
xcrun --sdk macosx metallib "$out/formats.air" "$out/guard.air" "$out/aspect.air" "$out/color.air" -o "$out/netal_nr_formats.metallib"
archive="${1:-}"
args=()
[[ "$delayed" == 1 ]] && args+=(-DNETAL_NR_DELAYED_ONCE=1)
if [[ -z "$archive" ]]; then
  [[ "$delayed" == 0 ]] || { print -u2 'delayed build requires an explicit worker archive'; exit 2; }
  args+=(-DNETAL_NR_OBSERVE_ONLY=1)
  product="$out/libNetalNRObserve.dylib"
  mode=observe_only
else
  [[ -f "$archive" ]] || { print -u2 "runtime archive missing: $archive"; exit 2; }
  args+=("$archive")
  product="$out/libNetalNRExperimental.dylib"
  mode=experimental_runtime_linked
fi
bridge_source="$repo/src/macos_host/netal_nr_experimental_bridge.mm"
if [[ "$delayed" == 1 ]]; then
  # Compile the reviewed bridge against the SAME public types as the frozen
  # archive, never moving repository headers. C++ return-by-value layouts matter.
  mkdir -p "$out/source/src/macos_host" "$out/source/src/nr_experimental"
  for nr_name in netal_nr_experimental_bridge.mm netal_nr_delayed_once_bridge.h netal_nr_delayed_once_state.h \
      netal_nr_frame_capture.h netal_nr_present_observer.h netal_nr_aspect.h netal_nr_motion_pair_capture.h; do
    cp -p "$repo/src/macos_host/$nr_name" "$out/source/src/macos_host/$nr_name"
  done
  for nr_name in nr_experimental_api.h nr_experimental_reset_worker.h; do
    cp -p "$runtime_headers/$nr_name" "$out/source/src/nr_experimental/$nr_name"
  done
  cp -p "$runtime_headers/runtime/nr_encoder_scope.h" "$out/source/src/nr_experimental/nr_encoder_scope.h"
  if rg -q 'reset_worker_submissions' "$out/source/src/nr_experimental/nr_experimental_reset_worker.h"; then
    args+=(-DNR_DELAYED_HAS_SUBMISSION_LABELS=1)
  fi
  bridge_source="$out/source/src/macos_host/netal_nr_experimental_bridge.mm"
fi
xcrun --sdk macosx clang++ -arch arm64 -dynamiclib -fobjc-arc -fobjc-arc-exceptions -std=c++20 -O2 \
  -Wall -Wextra -Werror -mmacosx-version-min=26.0 \
  -framework Foundation -framework Metal -framework MetalFX \
  "$bridge_source" "${args[@]}" -o "$product"
codesign --force --sign - "$product"
codesign --verify --strict "$product"
print "mode=$mode product=$product"
shasum -a 256 "$product"
