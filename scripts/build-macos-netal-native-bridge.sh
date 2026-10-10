#!/bin/zsh
# Build the arm64 bridge for native macOS games.
#
# The CrossOver bridge is x86_64 because it loads into a Wine process. A native Apple-silicon game
# is arm64, and dyld will not load a dylib of the wrong architecture, so the same sources are built
# again for arm64 here. One source tree, two slices - never two implementations.
#
#   scripts/build-macos-netal-native-bridge.sh
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
repository="$(cd "$script_dir/.." && pwd)"
build_dir="$repository/build/netal-native"
dylib="$build_dir/libNetalNative.dylib"

fg_bundle="${NEURALSTUDIO_FG_BUNDLE:-$repository/work/re9-fg-runtime-bundle}"
[[ -f "$fg_bundle/lib/libnetal_fg_runtime.a" ]] || { print -u2 "FG runtime bundle missing: $fg_bundle"; exit 2; }

mkdir -p "$build_dir"
xcrun --sdk macosx clang++ -arch arm64 -dynamiclib -Wl,-install_name,@rpath/libNetalNative.dylib -fobjc-arc -std=c++20 \
  -O2 -Wall -Wextra -Werror -mmacosx-version-min=26.0 \
  -framework Foundation -framework Metal -framework MetalFX -framework QuartzCore -framework CoreGraphics \
  -I "$fg_bundle/include" \
  "$repository/src/macos_host/dlss_fg_session.mm" \
  "$repository/src/macos_host/dlss_sr_session.mm" \
  "$repository/src/macos_host/dlss_sr_texture_adapter.mm" \
  "$repository/src/macos_host/re9_metalfx_noop_bridge.mm" \
  "$fg_bundle/lib/libnetal_fg_runtime.a" -o "$dylib"

# Ad-hoc signature. A sandboxed App Store game will still load this because it has no hardened
# runtime, or carries disable-library-validation; the pre-flight is what decides that per game.
codesign --force --sign - "$dylib" >/dev/null

print "built=$dylib"
print "arch=$(lipo -archs "$dylib")"
shasum -a 256 "$dylib" | cut -c1-24 | sed 's/^/sha=/'
print "result=pass"
