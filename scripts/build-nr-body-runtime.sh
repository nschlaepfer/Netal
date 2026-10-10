#!/bin/zsh
# Build the in-memory body runtime (versioned core + file-loading CLI adapter) into a fresh directory: build-nr-body-runtime.sh VERSION OUTDIR
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"; ver="${1:?runtime version, e.g. v2}"; out="${2:?fresh output directory}"
[[ ! -e "$out" ]] || { echo "Preserve existing build: $out" >&2; exit 2; }
mkdir -p "$out/source/src/nr_body" "$out/source/src/nr_temporal"; cp "$repo"/src/nr_temporal/* "$out/source/src/nr_temporal/" 2>/dev/null || true; cp "$repo/src/nr_body/nr_body_runtime.h" "$repo/src/nr_body/nr_body_digest.h" "$repo/src/nr_body/nr_body_runtime_$ver.mm" "$repo/src/nr_body/nr_body_cli.mm" "$out/source/src/nr_body/"
xcrun -sdk macosx clang++ -std=c++20 -O2 -fobjc-arc -Wall -Wextra -Werror -Wno-deprecated-declarations ${=NR_BODY_DEFINES:-} -I "$out/source/src/nr_body" -I "$out/source/src" \
  "$out/source/src/nr_body/nr_body_cli.mm" "$out/source/src/nr_body/nr_body_runtime_$ver.mm" -framework Foundation -framework Metal -o "$out/nr_body_cli"
echo "defines: ${NR_BODY_DEFINES:-none}" > "$out/build-defines.txt"
# Immutable build manifest: every copied source file, including the shared src/nr_temporal headers the body includes
# transitively (Codex review 2026-09-08 06:17). A copied file that is not hashed fails the build.
shasum -a 256 "$out/nr_body_cli" "$out"/source/src/nr_body/* "$out"/source/src/nr_temporal/* > "$out/build-sha256.txt"
for f in "$out"/source/src/nr_body/* "$out"/source/src/nr_temporal/*; do grep -q "  $f\$" "$out/build-sha256.txt" || { echo "build manifest does not cover $f" >&2; exit 5; }; done
echo "built $out"
# negative-control program (runtime-level fail-closed checks)
cp "$repo/src/nr_body/nr_body_negctl.mm" "$out/source/src/nr_body/"
xcrun -sdk macosx clang++ -std=c++20 -O2 -fobjc-arc -Wall -Wextra -Werror -Wno-deprecated-declarations -I "$out/source/src/nr_body" -I "$out/source/src" \
  "$out/source/src/nr_body/nr_body_negctl.mm" "$out/source/src/nr_body/nr_body_runtime_$ver.mm" -framework Foundation -framework Metal -o "$out/nr_body_negctl"
shasum -a 256 "$out/nr_body_negctl" >> "$out/build-sha256.txt"; echo "built $out/nr_body_negctl"
# gate-2 API misuse tests (only when the runtime version exports the public API, i.e. the source declares nr_body_encode)
if grep -q "int nr_body_encode" "$repo/src/nr_body/nr_body_runtime_$ver.mm"; then cp "$repo/src/nr_body/nr_body_apitest.mm" "$out/source/src/nr_body/"
  xcrun -sdk macosx clang++ -std=c++20 -O2 -fobjc-arc -Wall -Wextra -Werror -Wno-deprecated-declarations ${=NR_BODY_DEFINES:-} -I "$out/source/src/nr_body" -I "$out/source/src" \
    "$out/source/src/nr_body/nr_body_apitest.mm" "$out/source/src/nr_body/nr_body_runtime_$ver.mm" -framework Foundation -framework Metal -o "$out/nr_body_apitest"
  shasum -a 256 "$out/nr_body_apitest" >> "$out/build-sha256.txt"; echo "built $out/nr_body_apitest"; fi
