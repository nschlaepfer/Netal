#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "$0")/.." && pwd)"
output_root="${1:?output root required}"
if [[ -e "$output_root" ]]; then
  echo "output root already exists: $output_root" >&2
  exit 2
fi
build_root="$output_root/build"
mkdir -p "$build_root"
archive="$repository_root/evidence/checkpoints/2026-08-31-sm120-rcp-rom/external-evidence-sm120-rcp-rom.tar.gz"
rom="$build_root/sm120-rcp-correction-rom.bin"
tar -xOf "$archive" work/g23-rcp-architecture/m3-rcp-rom/correction-rom.bin >"$rom"
[[ "$(shasum -a 256 "$rom" | awk '{print $1}')" == \
  "7cf4f8cece6b18b07605edc9cb601a153346a79b10bf3b93c7ec058d57d29750" ]]
for source in module10 module9; do
  xcrun -sdk macosx metal -std=metal3.1 -fno-fast-math -ffp-contract=off \
    -c "$repository_root/src/fg_distortion_path/$source.metal" \
    -o "$build_root/$source.air"
done
xcrun -sdk macosx metal -std=metal3.1 -fno-fast-math -ffp-contract=off \
  -c "$repository_root/src/fg_early_l8_l18/early_l8_l18.metal" \
  -o "$build_root/module25.air"
xcrun -sdk macosx metallib "$build_root/module10.air" \
  "$build_root/module9.air" "$build_root/module25.air" \
  -o "$build_root/connected.metallib"
xcrun -sdk macosx clang++ -std=c++20 -fobjc-arc \
  "$repository_root/src/fg_distortion_path/connected_mac.mm" \
  -framework Foundation -framework Metal -o "$build_root/connected"
for poison in a5 5a; do
  for process in a b; do
    run_id="distortion-six-poison-$poison-$process"
    run_root="$output_root/$run_id"
    mkdir -p "$run_root"
    "$build_root/connected" "$build_root/connected.metallib" "$rom" \
      "$run_root" "$run_id" "$poison"
  done
done
