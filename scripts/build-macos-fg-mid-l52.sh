#!/usr/bin/env bash
set -euo pipefail

repository="$(cd "$(dirname "$0")/.." && pwd)"
attempt_root="${1:-$repository/build/fg_mid_l52/macos/attempt-$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ -e "$attempt_root" ]]; then
  echo "Attempt exists; preserve it: $attempt_root" >&2
  exit 2
fi
source_root="$repository/src/fg_mid_l52"
build_root="$attempt_root/build"
archive="$repository/evidence/checkpoints/2026-08-31-sm120-rcp-rom/external-evidence-sm120-rcp-rom.tar.gz"
rcp_rom="$build_root/sm120-rcp-correction-rom.bin"
mkdir -p "$build_root"
tar -xOf "$archive" work/g23-rcp-architecture/m3-rcp-rom/correction-rom.bin >"$rcp_rom"
rcp_hash="$(shasum -a 256 "$rcp_rom" | awk '{print $1}')"
if [[ "$rcp_hash" != "7cf4f8cece6b18b07605edc9cb601a153346a79b10bf3b93c7ec058d57d29750" ]]; then
  echo "RCP ROM identity mismatch: $rcp_hash" >&2
  exit 3
fi
xcrun -sdk macosx metal -std=metal3.1 -fno-fast-math -ffp-contract=off \
  -c "$source_root/mid_l52.metal" -o "$build_root/mid_l52.air" \
  >"$build_root/metal-compile.log" 2>&1
xcrun -sdk macosx metallib "$build_root/mid_l52.air" \
  -o "$build_root/mid_l52.metallib" >"$build_root/metallib-link.log" 2>&1
xcrun -sdk macosx clang++ -std=c++20 -fobjc-arc -Wall -Wextra -Werror \
  "$source_root/macos_probe.mm" -framework Foundation -framework Metal \
  -o "$build_root/fg-mid-l52-mac" >"$build_root/host-compile.log" 2>&1

run_case() {
  local run_id="$1" fixture="$2" poison="$3"
  local run_root="$attempt_root/$run_id"
  mkdir -p "$run_root"
  "$build_root/fg-mid-l52-mac" "$build_root/mid_l52.metallib" "$rcp_rom" \
    "$run_root" "$run_id" "$fixture" "$poison" \
    >"$run_root/stdout.log" 2>"$run_root/stderr.log"
}
run_case binary-depth-poison-a5 binary-depth a5
run_case binary-depth-poison-5a binary-depth 5a
run_case ramp-depth-poison-a5 ramp-depth a5
run_case ramp-depth-poison-5a ramp-depth 5a

negative_root="$attempt_root/unsupported-fixture-negative"
mkdir -p "$negative_root"
set +e
"$build_root/fg-mid-l52-mac" "$build_root/mid_l52.metallib" "$rcp_rom" \
  "$negative_root" unsupported-fixture-negative unsupported a5 \
  >"$negative_root/stdout.log" 2>"$negative_root/stderr.log"
negative_exit=$?
set -e
if [[ "$negative_exit" -ne 1 || -e "$negative_root/run-receipt.json" ]]; then
  echo "Unsupported fixture did not fail closed: exit=$negative_exit" >&2
  exit 4
fi
PYTHONDONTWRITEBYTECODE=1 python3 "$repository/scripts/verify-fg-mid-l52.py" \
  backend --backend mac --root "$attempt_root" \
  --output "$attempt_root/backend-receipt.json" \
  >"$attempt_root/backend-verification.log"
shasum -a 256 "$attempt_root/backend-receipt.json" \
  >"$attempt_root/backend-receipt.sha256"
echo "attempt_root=$attempt_root"
echo "rcp_rom_sha256=$rcp_hash"
echo "backend_receipt_sha256=$(awk '{print $1}' "$attempt_root/backend-receipt.sha256")"
echo "result=pass"
