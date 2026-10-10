#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)";root="${1:-$repo/build/fg_early_l7_l17/macos/attempt-$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ -e "$root" ]]; then echo "Attempt exists: $root" >&2;exit 2;fi
src="$repo/src/fg_early_l7_l17";build="$root/build";mkdir -p "$build"
archive="$repo/evidence/checkpoints/2026-08-31-sm120-rcp-rom/external-evidence-sm120-rcp-rom.tar.gz";rom="$build/rcp.bin";tar -xOf "$archive" work/g23-rcp-architecture/m3-rcp-rom/correction-rom.bin >"$rom";[[ "$(shasum -a 256 "$rom"|awk '{print $1}')" == 7cf4f8cece6b18b07605edc9cb601a153346a79b10bf3b93c7ec058d57d29750 ]]
xcrun -sdk macosx metal -std=metal3.1 -fno-fast-math -ffp-contract=off -Wno-unused-variable -Wno-unused-function -c "$src/early_l7_l17.metal" -o "$build/kernel.air" >"$build/metal.log" 2>&1
xcrun -sdk macosx metallib "$build/kernel.air" -o "$build/kernel.metallib" >"$build/link.log" 2>&1
xcrun -sdk macosx clang++ -std=c++20 -fobjc-arc -Wall -Wextra -Werror -Wno-unused-function "$src/macos_probe.mm" -framework Foundation -framework Metal -o "$build/probe" >"$build/host.log" 2>&1
for phase in reset generated;do launch=l7;[[ "$phase" == generated ]]&&launch=l17;for fixture in dense sparse;do for poison in a5 5a;do for ordinal in a b;do id="$launch-$phase-$fixture-poison-$poison-process-$ordinal";dir="$root/$id";mkdir -p "$dir";"$build/probe" "$build/kernel.metallib" "$rom" "$dir" "$id" "$poison" "$phase" "$fixture" >"$dir/stdout.log" 2>"$dir/stderr.log";done;done;done;done
echo "attempt_root=$root";echo result=pass
