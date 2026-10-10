#!/bin/zsh
# Build the NR split-Swin Metal library and the launch probe into a fresh attempt directory.
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
attempt="${1:?Provide a fresh output directory}"
[[ ! -e "$attempt" ]] || { echo "Preserve existing attempt: $attempt" >&2; exit 2; }
mkdir -p "$attempt/source/src/nr_split" "$attempt/source/scripts"
for name in nr_split_common.metal nr_split_kernels.metal nr_split_all.metal split_probe.mm; do cp "$repo/src/nr_split/$name" "$attempt/source/src/nr_split/$name"; done
cp "$repo/src/nr_vit/nr_vit_common.metal" "$attempt/source/src/nr_split/nr_vit_common.metal"
cp "$repo/scripts/build-nr-split.sh" "$attempt/source/scripts/"
tcbin="${NR_TCBIN:?set NR_TCBIN to the directory holding the Metal toolchain binaries (metal, metallib); the previous session-scratchpad shim is gone (2026-09-07)}"
export PATH="$tcbin:$PATH"
# SDK: the accepted libraries were built with SDKROOT set to the RESOLVED SDK path; the symlink spelling changes the metallib hash. Override with NR_SDKROOT.
sdkroot="${NR_SDKROOT:-$(cd "$(xcrun --show-sdk-path)" && pwd -P)}"
export SDKROOT="$sdkroot"   # the SDKROOT VALUE (resolved, non-symlink spelling) is embedded by the Metal frontend: with the MacOSX.sdk symlink the metallib hash differs from the accepted libraries (2026-09-07 finding)
metal_path="$(xcrun -f metal)"; metallib_path="$(xcrun -f metallib)"
"$metal_path" --version > "$attempt/metal-version.txt" 2>&1 || true
xcrun clang++ --version > "$attempt/clang-version.txt"
"$metal_path" -isysroot "$sdkroot" -std=metal3.1 -fno-fast-math -ffp-contract=off ${=NR_METAL_DEFINES:-} \
  -I "$attempt/source/src/nr_split" -c "$attempt/source/src/nr_split/nr_split_all.metal" -o "$attempt/split.air"
"$metallib_path" "$attempt/split.air" -o "$attempt/split.metallib"
xcrun -sdk macosx clang++ -std=c++20 -O2 -fobjc-arc -Wall -Wextra -Werror -Wno-deprecated-declarations -Wno-unused-but-set-variable \
  "$attempt/source/src/nr_split/split_probe.mm" -framework Foundation -framework Metal -o "$attempt/split_probe"
shasum -a 256 "$attempt/split.metallib" "$attempt/split_probe" "$attempt"/source/src/nr_split/* > "$attempt/build-sha256.txt"
echo "built $attempt"
