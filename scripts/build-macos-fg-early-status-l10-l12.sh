#!/usr/bin/env bash
set -euo pipefail
repository="$(cd "$(dirname "$0")/.." && pwd)"
attempt_root="${1:-$repository/build/fg_early_status_l10_l12/macos/attempt-$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ -e "$attempt_root" ]]; then echo "Attempt exists: $attempt_root" >&2; exit 2; fi
source_root="$repository/src/fg_early_status_l10_l12"
build_root="$attempt_root/build"
mkdir -p "$build_root"
xcrun -sdk macosx metal -std=metal3.1 -fno-fast-math -ffp-contract=off -c "$source_root/status_write.metal" -o "$build_root/status_write.air" >"$build_root/metal-compile.log" 2>&1
xcrun -sdk macosx metallib "$build_root/status_write.air" -o "$build_root/status_write.metallib" >"$build_root/metallib-link.log" 2>&1
xcrun -sdk macosx clang++ -std=c++20 -fobjc-arc -Wall -Wextra -Werror "$source_root/macos_probe.mm" -framework Foundation -framework Metal -o "$build_root/fg-early-status-l10-l12-mac" >"$build_root/host-compile.log" 2>&1
for mode in reset generated; do
  for poison in a5 5a; do
    for ordinal in a b; do
      run_id="status-$mode-poison-$poison-process-$ordinal"
      run_root="$attempt_root/$run_id"
      mkdir -p "$run_root"
      "$build_root/fg-early-status-l10-l12-mac" "$build_root/status_write.metallib" "$run_root" "$run_id" "$poison" "$mode" >"$run_root/stdout.log" 2>"$run_root/stderr.log"
    done
  done
done
negative_root="$attempt_root/unsupported-mode-negative"
mkdir -p "$negative_root"
set +e
"$build_root/fg-early-status-l10-l12-mac" "$build_root/status_write.metallib" "$negative_root" unsupported-mode-negative a5 unsupported >"$negative_root/stdout.log" 2>"$negative_root/stderr.log"
negative_exit=$?
set -e
if [[ "$negative_exit" -ne 1 || -e "$negative_root/run-receipt.json" ]]; then echo "Unsupported mode did not fail closed" >&2; exit 3; fi
PYTHONDONTWRITEBYTECODE=1 python3 "$repository/scripts/verify-fg-early-status-l10-l12.py" backend --backend mac --root "$attempt_root" --output "$attempt_root/backend-receipt.json" >"$attempt_root/backend-verification.log"
shasum -a 256 "$attempt_root/backend-receipt.json" >"$attempt_root/backend-receipt.sha256"
echo "attempt_root=$attempt_root"
echo "backend_receipt_sha256=$(awk '{print $1}' "$attempt_root/backend-receipt.sha256")"
echo 'result=pass'
