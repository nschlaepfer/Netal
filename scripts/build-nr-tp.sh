#!/bin/zsh
# Build a three-product (tp) Metal library into a fresh directory, reproducing the historical ad-hoc invocation:
#   build-nr-tp.sh split|vit OUTDIR [SOURCE_METAL] [PROBE_MM]
# Sources default to the tracked ones; pass explicit paths (e.g. a `git show` extract) to rebuild a historical variant.
# Reproduction contract: docs/NR_BUILD_REPRODUCTION.md (NR_TCBIN toolchain, exported resolved SDKROOT, -std=metal3.1 -fno-fast-math -ffp-contract=off).
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"; cd "$repo"; fam="${1:?split|vit}"; out="${2:?fresh output directory}"
[[ ! -e "$out" ]] || { echo "Preserve existing build: $out" >&2; exit 2; }
case "$fam" in
  split) srcdir="src/nr_split -I src/nr_vit"; met="${3:-src/nr_split/nr_split_tp.metal}"; probe="${4:-src/nr_split/attend_tp_probe.mm}"; mname=nr_split_tp.metal; pname=attend_tp_probe;;
  vit)   srcdir=src/nr_vit;   met="${3:-src/nr_vit/nr_vit_tp.metal}";   probe="${4:-src/nr_vit/vit_tp_probe.mm}";   mname=nr_vit_tp.metal;   pname=vit_tp_probe;;
  *) echo "family must be split or vit" >&2; exit 2;;
esac
tcbin="${NR_TCBIN:?set NR_TCBIN to the Metal toolchain bin directory (see docs/NR_BUILD_REPRODUCTION.md)}"; export PATH="$tcbin:$PATH"
sdkroot="${NR_SDKROOT:-$(cd "$(xcrun --show-sdk-path)" && pwd -P)}"; export SDKROOT="$sdkroot"
srcdir_flags="-I ${srcdir}"   # split_common includes the ViT common header, so both source dirs are on the include path
mkdir -p "$out"; cp "$met" "$out/$mname"; cp "$probe" "$out/$pname.mm"
"$(xcrun -f metal)" -isysroot "$sdkroot" -std=metal3.1 -fno-fast-math -ffp-contract=off ${=srcdir_flags} -c "$out/$mname" -o "$out/tp.air"
"$(xcrun -f metallib)" "$out/tp.air" -o "$out/tp.metallib"
xcrun -sdk macosx clang++ -std=c++20 -O2 -fobjc-arc -Wall -Wextra -Werror -Wno-deprecated-declarations "$out/$pname.mm" -framework Foundation -framework Metal -o "$out/$pname"
shasum -a 256 "$out/tp.metallib" "$out/$pname" "$out"/*.metal "$out"/*.mm > "$out/build-sha256.txt"; echo "built $out"
