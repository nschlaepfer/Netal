#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
attempt="${1:-$repo/build/fg_early_l3_l13/macos/attempt-$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ -e "$attempt" ]]; then
  echo "Attempt exists; preserve it: $attempt" >&2
  exit 2
fi
mkdir -p "$attempt/build"
source_root="$repo/src/fg_early_l3_l13"
xcrun -sdk macosx metal -std=metal3.1 -fno-fast-math \
  -ffp-contract=off -c "$source_root/early_l3_l13.metal" \
  -o "$attempt/build/early_l3_l13.air"
xcrun -sdk macosx metallib "$attempt/build/early_l3_l13.air" \
  -o "$attempt/build/early_l3_l13.metallib"
xcrun -sdk macosx clang++ -std=c++20 -fobjc-arc -Wall -Wextra -Werror \
  -Wno-unused-function "$source_root/macos_probe.mm" \
  -framework Foundation -framework Metal -o "$attempt/build/host"

for phase in l3-evaluate-0 l13-evaluate-1; do
  for fixture in reset-history generated-history; do
    for poison in a5 5a; do
      for ordinal in a b; do
        run_id="$phase-$fixture-poison-$poison-process-$ordinal"
        run="$attempt/$run_id"
        mkdir -p "$run"
        "$attempt/build/host" "$attempt/build/early_l3_l13.metallib" \
          "$run" "$run_id" "$poison" "$phase" "$fixture" \
          >"$run/stdout.log" 2>"$run/stderr.log"
      done
    done
  done
done

python3 - "$attempt" "$source_root" "$repo/scripts/build-macos-fg-early-l3-l13.sh" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
source = pathlib.Path(sys.argv[2])
builder = pathlib.Path(sys.argv[3])
receipts = [json.loads(path.read_text()) for path in sorted(
    root.glob("*/run-receipt.json"))]
if len(receipts) != 16 or len({r["process_id"] for r in receipts}) != 16:
    raise SystemExit("fresh process gate failed")
outputs = {}
for fixture in ("reset-history", "generated-history"):
    hashes = {r["combined_output_sha256"] for r in receipts
              if r["fixture"] == fixture}
    if len(hashes) != 1:
        raise SystemExit(f"phase/poison invariance failed: {fixture}")
    outputs[fixture] = next(iter(hashes))
if outputs["reset-history"] == outputs["generated-history"]:
    raise SystemExit("history fixture sensitivity failed")
if any(r["result"] != "pass" for r in receipts):
    raise SystemExit("run receipt failed")
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
result = {
    "schema": "neuralstudio.fg.early-l3-l13-mac-backend.v1",
    "fresh_process_count": 16,
    "phases": ["l3-evaluate-0", "l13-evaluate-1"],
    "fixtures": ["reset-history", "generated-history"],
    "poison_bytes": [90, 165],
    "fixture_output_sha256": outputs,
    "source_sha256": digest(source / "macos_probe.mm"),
    "metal_sha256": digest(source / "early_l3_l13.metal"),
    "contract_sha256": digest(source / "contract.h"),
    "builder_sha256": digest(builder),
    "all_phase_poison_process_invariant": True,
    "full_write_coverage_from_two_poison_invariance": True,
    "typed_resource_backend": True,
    "expected_or_archived_tensor_ingress": False,
    "address_fields_recorded": False,
    "result": "pass",
}
(root / "backend-receipt.json").write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n")
PY

echo "attempt_root=$attempt"
echo "result=pass"
