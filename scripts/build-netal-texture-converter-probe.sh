#!/bin/zsh
# Builds the synthetic conformance probe only. Run under the shared GPU reservation workflow.
set -euo pipefail
root="$(cd "${0:h}/.." && pwd)"
out="$root/build/netal-tools/texture-converter-probe"
mkdir -p "${out:h}"
xcrun clang++ -std=c++20 -fobjc-arc -Wall -Wextra -Werror -framework Foundation -framework Metal \
  "$root/src/macos_host/netal_texture_converter_probe.mm" -o "$out"
codesign --force --sign - "$out" >/dev/null
print -r -- "built=$out"
print 'usage: texture-converter-probe <prepared reference metallib> <candidate metallib>'
