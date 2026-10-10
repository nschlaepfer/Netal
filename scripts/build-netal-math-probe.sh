#!/bin/zsh
set -euo pipefail
root="$(cd "${0:h}/.." && pwd)"
out="$root/build/math-calibration-tools"
mkdir -p "$out"
compiler="$(xcrun --find metal)"
"$compiler" -fmetal-math-mode=safe -fmetal-math-fp32-functions=precise -c "$root/tools/math-calibration/m3-baseline.metal" -o "$out/baseline.air"
"${compiler:h}/metallib" "$out/baseline.air" -o "$out/baseline.metallib"
xcrun clang++ -std=c++20 -fobjc-arc -Wall -Wextra -Werror -framework Foundation -framework Metal "$root/tools/math-calibration/m3-baseline.mm" -o "$out/baseline"
codesign --force --sign - "$out/baseline" >/dev/null
print 'result=pass (built only; no GPU execution)'
