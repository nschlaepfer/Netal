#!/bin/zsh
# Build the experimental NR runtime and its tools into a fresh directory.
#   build-nr-experimental-runtime.sh OUTDIR
# EXPERIMENTAL LANE: original weights and kernels with research-only validation removed. Not exact parity.
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"; out="${1:?fresh output directory}"
[[ ! -e "$out" ]] || { echo "Preserve existing build: $out" >&2; exit 2; }
src="$repo/src/nr_experimental"
mkdir -p "$out" "$out/source/runtime" "$out/source/runtime/nr_temporal" "$out/source/tools"
cp "$src"/runtime/*.mm "$src"/runtime/*.h "$out/source/runtime/"
cp "$src"/runtime/nr_temporal/* "$out/source/runtime/nr_temporal/" 2>/dev/null || true
cp "$src"/tools/*.mm "$out/source/tools/"
cp "$src/nr_experimental_api.h" "$out/source/"
cp "$src/nr_experimental_reset_worker.h" "$out/source/"
# The two experimental flags: the frame is encoded into a CALLER-owned command buffer, and the research-only
# coverage validators, checkpoint digests and snapshots are not encoded. Both are off in the accepted lane.
# Deployment target is explicit so the archive links against the bridge without a version-mismatch warning.
# Codex's bridge targets macOS 26; the SDK here is 27, and an unset target silently takes the SDK's. Setting it
# does NOT by itself establish that the runtime works on macOS 26 - it makes the mismatch visible and makes
# availability problems compile errors instead of link-time surprises. Override with NR_EXP_DEPLOYMENT_TARGET.
: "${NR_EXP_DEPLOYMENT_TARGET:=26.0}"
echo "deployment target: macOS $NR_EXP_DEPLOYMENT_TARGET (SDK $(xcrun -sdk macosx --show-sdk-version))" | tee "$out/deployment-target.txt"
CFLAGS=(-std=c++20 -fobjc-arc -O2 -Wall -Wextra -Werror -Wno-deprecated-declarations -Wno-unused-variable -arch arm64
        -mmacosx-version-min="$NR_EXP_DEPLOYMENT_TARGET"
        -DNR_EXPERIMENTAL_CALLER_COMMAND -DNR_EXPERIMENTAL_CHUNKED -DNR_EXPERIMENTAL_BODY_OVERRIDE -DNR_EXPERIMENTAL_ENTRY_ORDERED_CHECK
        -include nr_experimental_gpu_profile.h -include nr_experimental_encoder_scope_adoption.h
        -I "$out/source/runtime" -I "$out/source/runtime/nr_temporal" -I "$out/source")
LDFLAGS=(-framework Foundation -framework Metal)
xcrun -sdk macosx clang++ "${CFLAGS[@]}" -c "$out/source/runtime/nr_experimental_assets.mm"    -o "$out/assets.o"
xcrun -sdk macosx clang++ "${CFLAGS[@]}" -c "$out/source/runtime/nr_experimental_asset_pins.mm" -o "$out/pins.o"
xcrun -sdk macosx clang++ "${CFLAGS[@]}" -c "$out/source/runtime/nr_experimental_block0.mm"    -o "$out/block0.o"
xcrun -sdk macosx clang++ "${CFLAGS[@]}" -c "$out/source/runtime/nr_experimental_encoder.mm"   -o "$out/encoder.o"
xcrun -sdk macosx clang++ "${CFLAGS[@]}" -c "$out/source/runtime/nr_experimental_gpu_profile.mm" -o "$out/gpuprof.o"
xcrun -sdk macosx clang++ "${CFLAGS[@]}" -c "$out/source/runtime/nr_experimental_body.mm"      -o "$out/body.o"
xcrun -sdk macosx clang++ "${CFLAGS[@]}" -c "$out/source/runtime/nr_experimental_runtime.mm"   -o "$out/runtime.o"
xcrun -sdk macosx clang++ "${CFLAGS[@]}" -c "$out/source/runtime/nr_experimental_reset_worker.mm" -o "$out/resetworker.o"
# CHECKED-DIAG build only: nr_exp_encode_profile is defined inside the body's NR_EXPERIMENTAL_NO_VALIDATION
# region, which this build does not compile. Supplying the symbol from a separate TU keeps the flagged
# source byte-for-byte identical to the shipping copy - this build must not change what is measured.
cat > "$out/source/runtime/nr_exp_profile_shim.mm" <<'SHIM'
#define NR_EXPERIMENTAL_NO_VALIDATION 1     /* header only: pulls in the NrExpEncodeProfile declaration */
#include "nr_body_runtime.h"
NrExpEncodeProfile nr_exp_encode_profile;
SHIM
xcrun -sdk macosx clang++ "${CFLAGS[@]}" -c "$out/source/runtime/nr_exp_profile_shim.mm" -o "$out/profshim.o"
libtool -static -o "$out/libnr_experimental_runtime.a" "$out/profshim.o" "$out/assets.o" "$out/pins.o" "$out/block0.o" "$out/encoder.o" "$out/gpuprof.o" "$out/body.o" "$out/runtime.o" "$out/resetworker.o"
for tool in nr_experimental_frame; do
  xcrun -sdk macosx clang++ "${CFLAGS[@]}" "${LDFLAGS[@]}" "$out/source/tools/$tool.mm" "$out/libnr_experimental_runtime.a" -o "$out/$tool"
done
# Package manifest: every source file, every object, the library and the tools, plus the toolchain identity.
{ xcrun -sdk macosx clang++ --version | head -2; echo "arch: arm64"; echo "sdk: $(xcrun -sdk macosx --show-sdk-path)"; } > "$out/toolchain.txt"
shasum -a 256 "$out/libnr_experimental_runtime.a" "$out"/nr_experimental_frame "$out"/*.o \
  "$out"/source/runtime/*.mm "$out"/source/runtime/*.h "$out"/source/runtime/nr_temporal/* "$out"/source/tools/* "$out"/source/*.h > "$out/build-sha256.txt"
for f in "$out"/source/runtime/*.mm "$out"/source/runtime/*.h "$out"/source/runtime/nr_temporal/* "$out"/source/tools/* "$out"/source/*.h; do
  grep -q "  $f\$" "$out/build-sha256.txt" || { echo "package manifest does not cover $f" >&2; exit 5; }
done
echo "built $out"
