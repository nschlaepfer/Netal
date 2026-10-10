#!/bin/zsh
# Fresh isolated build of the Metal texture-sampler probe (nr_tex_probe.metal + nr_tex_sample.metal + driver): build-nr-tex-probe.sh OUTDIR
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"; out="${1:?fresh output directory}"; [[ ! -e "$out" ]] || { echo "Preserve existing build: $out" >&2; exit 2; }
mkdir -p "$out/source"; cp "$repo"/src/nr_temporal/nr_tex_probe.metal "$repo"/src/nr_temporal/nr_tex_sample.metal "$repo"/src/nr_temporal/nr_tex_probe.mm "$out/source/"
tcbin="${NR_TCBIN:?set NR_TCBIN to the directory holding the Metal toolchain binaries (metal, metallib); the previous session-scratchpad shim is gone (2026-09-07)}"; export PATH="$tcbin:$PATH"
# SDK: the accepted libraries were built with SDKROOT set to the RESOLVED SDK path; the symlink spelling changes the metallib hash. Override with NR_SDKROOT.
sdkroot="${NR_SDKROOT:-$(cd "$(xcrun --show-sdk-path)" && pwd -P)}"
export SDKROOT="$sdkroot"   # the SDKROOT VALUE (resolved, non-symlink spelling) is embedded by the Metal frontend: with the MacOSX.sdk symlink the metallib hash differs from the accepted libraries (2026-09-07 finding)
"$(xcrun -f metal)" -isysroot "$sdkroot" -std=metal3.1 -fno-fast-math -ffp-contract=off -I "$out/source" -c "$out/source/nr_tex_probe.metal" -o "$out/probe.air"
"$(xcrun -f metallib)" "$out/probe.air" -o "$out/probe.metallib"
xcrun -sdk macosx clang++ -std=c++20 -O2 -fobjc-arc -Wall -Wextra -Werror -Wno-deprecated-declarations "$out/source/nr_tex_probe.mm" -framework Foundation -framework Metal -o "$out/nr_tex_probe"
shasum -a 256 "$out/probe.metallib" "$out/nr_tex_probe" "$out"/source/* > "$out/build-sha256.txt"; echo "built $out"
