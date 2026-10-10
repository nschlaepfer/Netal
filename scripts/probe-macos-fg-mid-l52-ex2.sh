#!/usr/bin/env bash
set -euo pipefail
repository="$(cd "$(dirname "$0")/.." && pwd)"
attempt_root="${1:-$repository/build/fg_mid_l52/ex2/macos-$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ -e "$attempt_root" ]]; then
  echo "Attempt exists: $attempt_root" >&2
  exit 2
fi
source_root="$repository/src/fg_mid_l52"
build_root="$attempt_root/build"
archive="$repository/evidence/checkpoints/2026-08-31-sm120-rcp-rom/external-evidence-sm120-rcp-rom.tar.gz"
rom="$build_root/sm120-rcp-correction-rom.bin"
mkdir -p "$build_root"
tar -xOf "$archive" work/g23-rcp-architecture/m3-rcp-rom/correction-rom.bin >"$rom"
[[ "$(shasum -a 256 "$rom" | awk '{print $1}')" == "7cf4f8cece6b18b07605edc9cb601a153346a79b10bf3b93c7ec058d57d29750" ]]
xcrun -sdk macosx metal -std=metal3.1 -fno-fast-math -ffp-contract=off \
  -c "$source_root/ex2_probe.metal" -o "$build_root/ex2_probe.air" \
  >"$build_root/metal-compile.log" 2>&1
xcrun -sdk macosx metallib "$build_root/ex2_probe.air" \
  -o "$build_root/ex2_probe.metallib" >"$build_root/metallib-link.log" 2>&1
xcrun -sdk macosx clang++ -std=c++20 -fobjc-arc -Wall -Wextra -Werror \
  "$source_root/macos_ex2_probe.mm" -framework Foundation -framework Metal \
  -o "$build_root/fg-mid-l52-mac-ex2" >"$build_root/host-compile.log" 2>&1
"$build_root/fg-mid-l52-mac-ex2" "$build_root/ex2_probe.metallib" "$rom" \
  "$attempt_root/output" >"$attempt_root/stdout.log" 2>"$attempt_root/stderr.log"
cat "$attempt_root/stdout.log"
