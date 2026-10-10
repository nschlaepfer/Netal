#!/usr/bin/env bash
set -euo pipefail

repository="$(cd "$(dirname "$0")/.." && pwd)"
attempt_root="${1:-}"
if [[ -z "$attempt_root" ]]; then
  attempt_root="$repository/build/fg_custom_upsample/macos/attempt-$(date -u +%Y%m%dT%H%M%SZ)"
fi
if [[ -e "$attempt_root" ]]; then
  echo "Attempt directory already exists; preserve it and choose a new path: $attempt_root" >&2
  exit 2
fi

source_root="$repository/src/fg_custom_upsample"
build_root="$attempt_root/build"
mkdir -p "$build_root"

xcrun -sdk macosx metal -std=metal3.1 -fno-fast-math -ffp-contract=off \
  -c "$source_root/custom_upsample.metal" \
  -o "$build_root/custom_upsample.air" \
  >"$build_root/metal-compile.log" 2>&1
xcrun -sdk macosx metallib \
  "$build_root/custom_upsample.air" \
  -o "$build_root/custom_upsample.metallib" \
  >"$build_root/metallib-link.log" 2>&1
xcrun -sdk macosx clang++ \
  -std=c++20 -fobjc-arc -Wall -Wextra -Werror \
  "$source_root/macos_probe.mm" \
  -framework Foundation -framework Metal \
  -o "$build_root/fg-custom-upsample-mac" \
  >"$build_root/host-compile.log" 2>&1

run_case() {
  local run_id="$1"
  local poison="$2"
  local run_root="$attempt_root/$run_id"
  mkdir -p "$run_root"
  "$build_root/fg-custom-upsample-mac" \
    "$build_root/custom_upsample.metallib" \
    "$run_root" "$run_id" "$poison" \
    >"$run_root/stdout.log" 2>"$run_root/stderr.log"
}

run_case poison-a5-process-a a5
run_case poison-a5-process-b a5
run_case poison-5a-process-a 5a
run_case poison-5a-process-b 5a

PYTHONDONTWRITEBYTECODE=1 python3 \
  "$repository/scripts/verify-fg-custom-upsample.py" backend \
  --backend mac \
  --root "$attempt_root" \
  --output "$attempt_root/backend-receipt.json" \
  >"$attempt_root/backend-verification.log"

echo "attempt_root=$attempt_root"
echo "backend_receipt_sha256=$(shasum -a 256 "$attempt_root/backend-receipt.json" | awk '{print $1}')"
echo "result=pass"
