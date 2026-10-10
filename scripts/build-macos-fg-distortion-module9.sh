#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "$0")/.." && pwd)"
output_root="${1:?output root required}"
if [[ -e "$output_root" ]]; then
  echo "output root already exists: $output_root" >&2
  exit 2
fi
mkdir -p "$output_root/build"

xcrun -sdk macosx metal -std=metal3.1 -fno-fast-math -ffp-contract=off \
  -c "$repository_root/src/fg_distortion_path/module9.metal" \
  -o "$output_root/build/m9.air"
xcrun -sdk macosx metallib "$output_root/build/m9.air" \
  -o "$output_root/build/m9.metallib"
xcrun -sdk macosx clang++ -std=c++20 -fobjc-arc \
  "$repository_root/src/fg_distortion_path/module9_mac.mm" \
  -framework Foundation -framework Metal \
  -o "$output_root/build/m9"

for poison in a5 5a; do
  for process in a b; do
    run_id="m9-poison-$poison-$process"
    run_root="$output_root/$run_id"
    mkdir -p "$run_root"
    "$output_root/build/m9" "$output_root/build/m9.metallib" \
      "$run_root" "$run_id" "$poison"
  done
done
