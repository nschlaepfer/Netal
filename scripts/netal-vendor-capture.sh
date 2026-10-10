#!/bin/zsh
# Refresh the vendored capture shims from a research checkout.
#
# Only needed when the recorder itself changes. Vendored binaries are what make an extraction
# reproducible, so this is deliberately a separate, explicit step rather than something the ingest
# does for itself.
set -euo pipefail
repository="$(cd "$(dirname "$0")/.." && pwd)"
harness="${1:-$(cd "$repository/.." && pwd)/NeuralStudio-sr-research}"
build="$harness/build/dlss_sr_capture"
[[ -f "$build/dlss-sr-cuda-capture.exe" ]] \
  || { print -u2 "capture tool not built: run $harness/scripts/build-dlss-sr-capture-tools.sh"; exit 1 }
for f in dlss-sr-cuda-capture.exe nvcuda.dll nvapi64.dll nvngx.dll nvcuda-forward-trace.dll; do
  cp "$build/$f" "$repository/vendor/dlss-capture/$f"
done
cd "$repository/vendor/dlss-capture"
shasum -a 256 dlss-sr-cuda-capture.exe nvcuda.dll nvapi64.dll nvngx.dll nvcuda-forward-trace.dll > SHA256SUMS
print "vendored from $harness"
cat SHA256SUMS
