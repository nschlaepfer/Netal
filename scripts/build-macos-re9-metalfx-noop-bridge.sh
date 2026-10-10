#!/bin/zsh
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
repository="$(cd "$script_dir/.." && pwd)"
build_dir="$repository/build/re9_crossover"
dylib="$build_dir/libNeuralStudioRe9MetalFXNoop.dylib"
host="$build_dir/re9.exe"
# The FG runtime (exact DLSS-G port, Metal 4 shim) is linked from its bundle: a universal static
# archive plus the metallibs, model tokens and tables, built by
# scripts/build-macos-fg-runtime-bundle.sh in the Mac-fg-research worktree.
fg_bundle="${NEURALSTUDIO_FG_BUNDLE:-$repository/work/re9-fg-runtime-bundle}"
[[ -f "$fg_bundle/lib/libnetal_fg_runtime.a" ]] || { print -u2 "FG runtime bundle missing: $fg_bundle"; exit 2; }

mkdir -p "$build_dir"
xcrun --sdk macosx clang++ -arch x86_64 -dynamiclib -Wl,-install_name,@rpath/libNeuralStudioRe9MetalFXNoop.dylib -fobjc-arc -std=c++20 \
  -O2 -Wall -Wextra -Werror -mmacosx-version-min=26.0 \
  -framework Foundation -framework Metal -framework MetalFX -framework QuartzCore -framework CoreGraphics \
  -I "$fg_bundle/include" \
  "$repository/src/macos_host/dlss_sr_session.mm" \
  "$repository/src/macos_host/dlss_sr_texture_adapter.mm" \
  "$repository/src/macos_host/dlss_fg_session.mm" \
  "$repository/src/macos_host/re9_metalfx_noop_bridge.mm" \
  "$fg_bundle/lib/libnetal_fg_runtime.a" -o "$dylib"
xcrun --sdk macosx clang++ -arch x86_64 -fobjc-arc -std=c++20 \
  -O2 -Wall -Wextra -Werror -mmacosx-version-min=26.0 \
  -framework Foundation -framework Metal -framework MetalFX \
  "$repository/src/macos_host/re9_metalfx_noop_test.mm" -o "$host"
codesign --force --sign - "$dylib" >/dev/null
codesign --force --sign - "$host" >/dev/null
file "$dylib" "$host"
shasum -a 256 "$dylib" "$host"
print "result=pass"
