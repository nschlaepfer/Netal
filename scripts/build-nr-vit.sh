#!/bin/zsh
# Build the NR ViT-1D Metal library and the block probe into a fresh attempt directory.
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
attempt="${1:?Provide a fresh output directory}"
[[ ! -e "$attempt" ]] || { echo "Preserve existing attempt: $attempt" >&2; exit 2; }
mkdir -p "$attempt/source/src/nr_vit" "$attempt/source/scripts"
for name in nr_vit_common.metal nr_vit_ffn.metal nr_vit_attention.metal nr_vit_repack.metal nr_vit_all.metal vit_probe.mm repack_probe.mm; do
  [[ -f "$repo/src/nr_vit/$name" ]] && cp "$repo/src/nr_vit/$name" "$attempt/source/src/nr_vit/$name"
done
cp "$repo/scripts/build-nr-vit.sh" "$attempt/source/scripts/"
# the toolchain: xcrun's metal, or the mounted Metal toolchain image used throughout this project
tcbin="${NR_TCBIN:?set NR_TCBIN to the directory holding the Metal toolchain binaries (metal, metallib); the previous session-scratchpad shim is gone (2026-09-07)}"
export PATH="$tcbin:$PATH"
# SDK: the accepted libraries were built with SDKROOT set to the RESOLVED SDK path; the symlink spelling changes the metallib hash. Override with NR_SDKROOT.
sdkroot="${NR_SDKROOT:-$(cd "$(xcrun --show-sdk-path)" && pwd -P)}"
export SDKROOT="$sdkroot"   # the SDKROOT VALUE (resolved, non-symlink spelling) is embedded by the Metal frontend: with the MacOSX.sdk symlink the metallib hash differs from the accepted libraries (2026-09-07 finding)
metal_path="$(xcrun -f metal)"; metallib_path="$(xcrun -f metallib)"
"$metal_path" --version > "$attempt/metal-version.txt" 2>&1 || true
xcrun clang++ --version > "$attempt/clang-version.txt"
"$metal_path" -isysroot "$sdkroot" -std=metal3.1 -fno-fast-math -ffp-contract=off ${=NR_METAL_DEFINES:-} \
  -I "$attempt/source/src/nr_vit" -c "$attempt/source/src/nr_vit/nr_vit_all.metal" -o "$attempt/vit.air"
"$metallib_path" "$attempt/vit.air" -o "$attempt/vit.metallib"
xcrun -sdk macosx clang++ -std=c++20 -O2 -fobjc-arc -Wall -Wextra -Werror -Wno-deprecated-declarations \
  "$attempt/source/src/nr_vit/vit_probe.mm" -framework Foundation -framework Metal -o "$attempt/vit_probe"
xcrun -sdk macosx clang++ -std=c++20 -O2 -fobjc-arc -Wall -Wextra -Werror -Wno-deprecated-declarations \
  "$attempt/source/src/nr_vit/repack_probe.mm" -framework Foundation -framework Metal -o "$attempt/repack_probe"
shasum -a 256 "$attempt/vit.metallib" "$attempt/vit_probe" "$attempt/repack_probe" "$attempt"/source/src/nr_vit/* > "$attempt/build-sha256.txt"
echo "built $attempt"
