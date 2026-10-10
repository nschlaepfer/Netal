#!/usr/bin/env bash
set -euo pipefail
repository="$(cd "$(dirname "$0")/.." && pwd)"
attempt_root="${1:-$repository/build/fg_early_l4_l14/macos/attempt-$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ -e "$attempt_root" ]]; then echo "Attempt exists: $attempt_root" >&2; exit 2; fi
source_root="$repository/src/fg_early_l4_l14"
build_root="$attempt_root/build"
mkdir -p "$build_root"
xcrun -sdk macosx metal -std=metal3.1 -fno-fast-math -ffp-contract=off -c "$source_root/early_l4_l14.metal" -o "$build_root/early_l4_l14.air" >"$build_root/metal-compile.log" 2>&1
xcrun -sdk macosx metallib "$build_root/early_l4_l14.air" -o "$build_root/early_l4_l14.metallib" >"$build_root/metallib-link.log" 2>&1
xcrun -sdk macosx clang++ -std=c++20 -fobjc-arc -Wall -Wextra -Werror "$source_root/macos_probe.mm" -framework Foundation -framework Metal -o "$build_root/fg-early-l4-l14-mac" >"$build_root/host-compile.log" 2>&1
for fixture in branch-grid center-tie; do
  for phase in reset generated; do
    if [[ "$phase" == reset ]]; then launch=4; else launch=14; fi
    for poison in a5 5a; do
      for ordinal in a b; do
        run_id="l$launch-$fixture-poison-$poison-process-$ordinal"
        run_root="$attempt_root/$run_id"
        mkdir -p "$run_root"
        "$build_root/fg-early-l4-l14-mac" "$build_root/early_l4_l14.metallib" "$run_root" "$run_id" "$poison" "$phase" "$fixture" >"$run_root/stdout.log" 2>"$run_root/stderr.log"
      done
    done
  done
done
negative_root="$attempt_root/unsupported-negative"
mkdir -p "$negative_root"
set +e
"$build_root/fg-early-l4-l14-mac" "$build_root/early_l4_l14.metallib" "$negative_root" unsupported-phase a5 unsupported branch-grid >"$negative_root/phase.stdout.log" 2>"$negative_root/phase.stderr.log"
negative_phase=$?
"$build_root/fg-early-l4-l14-mac" "$build_root/early_l4_l14.metallib" "$negative_root" unsupported-fixture a5 reset unsupported >"$negative_root/fixture.stdout.log" 2>"$negative_root/fixture.stderr.log"
negative_fixture=$?
set -e
if [[ "$negative_phase" -ne 1 || "$negative_fixture" -ne 1 || -e "$negative_root/run-receipt.json" ]]; then echo "Unsupported L4/L14 domain did not fail closed" >&2; exit 3; fi
PYTHONDONTWRITEBYTECODE=1 python3 "$repository/scripts/verify-fg-early-l4-l14.py" backend --backend mac --root "$attempt_root" --output "$attempt_root/backend-receipt.json" >"$attempt_root/backend-verification.log"
shasum -a 256 "$attempt_root/backend-receipt.json" >"$attempt_root/backend-receipt.sha256"
echo "attempt_root=$attempt_root"
echo "backend_receipt_sha256=$(awk '{print $1}' "$attempt_root/backend-receipt.sha256")"
echo 'result=pass'
