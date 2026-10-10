#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
attempt="${1:-$repo/build/fg_connected_l20_l87/macos/attempt-$(date -u +%Y%m%dT%H%M%SZ)}"
model="${2:-$repo/build/fg_default_readback/eval0-default-all-a5-a/runtime-default-buffers-post-evaluate-0}"
rom="${3:-$repo/build/fg_connected_tail/macos-l56-l87/attempt-exact-v1/sm120-rcp-correction-rom.bin}"
sigmoid="${4:-$repo/build/fg_postprocess_l84/rtx/oracle-006/sigmoid-process-a/sm120-half-logit-sigmoid-lut.bin}"
if [[ -e "$attempt" ]]; then
  echo "Attempt exists; preserve it: $attempt" >&2
  exit 2
fi
libs="$attempt/libs"
mkdir -p "$libs"

compile_metal() {
  local source="$1"
  local name="$2"
  xcrun -sdk macosx metal -std=metal3.1 -fno-fast-math \
    -ffp-contract=off -I "$repo/src/fg_conv" -c "$repo/$source" \
    -o "$libs/$name.air"
  xcrun -sdk macosx metallib "$libs/$name.air" \
    -o "$libs/$name.metallib"
}

compile_metal src/fg_conv/conv_3x3.metal conv_3x3
compile_metal src/fg_conv/conv_1x1.metal conv_1x1
compile_metal src/fg_pooling/pooling.metal pooling
compile_metal src/fg_upscale/upscale.metal upscale-prefix
compile_metal src/fg_elementwise/elementwise_family.metal elementwise
compile_metal src/fg_mid_l52/mid_l52.metal mid_l52
compile_metal src/fg_mid_l53/mid_l53.metal mid_l53
compile_metal src/fg_mid_l54/mid_l54.metal mid_l54
compile_metal src/fg_mid_l55/fg_mid_l55.metal mid_l55
compile_metal src/fg_initial_merge/initial_merge.metal initial_merge
compile_metal src/fg_conv/custom_conv_pre.metal conv_pre
compile_metal src/fg_conv/custom_conv0_c8.metal conv0
compile_metal src/fg_conv/custom_conv1.metal conv1
compile_metal src/fg_conv/custom_conv2.metal conv2
compile_metal src/fg_connected_custom_l56_l83/connected_block0_bot0.metal bot0
compile_metal src/fg_connected_custom_l56_l83/connected_block0_bot1.metal bot1
compile_metal src/fg_custom_upsample/custom_upsample.metal upsample
compile_metal src/fg_central_block/central_block.metal central
compile_metal src/fg_conv/custom_block1_conv0.metal block1_conv0
compile_metal src/fg_conv/custom_block1_conv1.metal block1_conv1
compile_metal src/fg_conv/custom_block1_conv2.metal block1_conv2
compile_metal src/fg_connected_custom_l56_l83/connected_block1_bot0.metal block1_bot0
compile_metal src/fg_connected_custom_l56_l83/connected_block1_bot1.metal block1_bot1
compile_metal src/fg_postprocess_l84/postprocess_l84.metal postprocess_l84
compile_metal src/fg_postprocess_l85/postprocess_l85.metal postprocess_l85
compile_metal src/fg_postprocess_l86/postprocess_l86_mode0.metal postprocess_l86
compile_metal src/fg_compositor_l87/compositor_l87.metal compositor_l87

xcrun -sdk macosx clang++ -std=c++20 -fobjc-arc -Wall -Wextra -Werror \
  -Wno-unused-function -Wno-deprecated-declarations \
  "$repo/src/fg_connected_l20_l87/macos_connected_l20_l87.mm" \
  -framework Foundation -framework Metal -o "$attempt/host"

for poison in a5 5a; do
  for ordinal in a b; do
    run_id="connected-l20-l87-poison-$poison-process-$ordinal"
    run="$attempt/$run_id"
    mkdir -p "$run"
    "$attempt/host" "$libs" "$model" "$rom" "$sigmoid" "$run" \
      "$run_id" "$poison" >"$run/stdout.log" 2>"$run/stderr.log"
  done
done

python3 - "$attempt" "$repo" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
repo = pathlib.Path(sys.argv[2])
receipts = [json.loads(p.read_text()) for p in sorted(root.glob(
    "connected-l20-l87-*/run-receipt.json"))]
if len(receipts) != 4:
    raise SystemExit("fresh process receipt count failed")
if len({r["process_id"] for r in receipts}) != 4:
    raise SystemExit("fresh process id gate failed")
if len({json.dumps(r["checkpoint_sha256"], sort_keys=True)
        for r in receipts}) != 1:
    raise SystemExit("two-poison checkpoint invariance failed")
if any(r["result"] != "pass" for r in receipts):
    raise SystemExit("run receipt failed")
for key in ("input_model_mismatched_bytes", "canary_mismatched_bytes",
            "readonly_tail_mismatched_bytes"):
    if any(r.get(key) != 0 for r in receipts):
        raise SystemExit(f"connected integrity gate failed: {key}")
if any(r.get("final_output_bytes") != 921600 for r in receipts):
    raise SystemExit("final output byte count failed")
source = repo / "src/fg_connected_l20_l87/macos_connected_l20_l87.mm"
script = repo / "scripts/build-macos-fg-connected-l20-l87.sh"
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
summary = {
    "schema": "neuralstudio.fg.connected-l20-l87-mac-backend.v1",
    "fresh_process_count": 4,
    "poison_bytes": [90, 165],
    "checkpoint_sha256": receipts[0]["checkpoint_sha256"],
    "invariant_output_sha256": receipts[0]["final_output_sha256"],
    "source_sha256": digest(source),
    "builder_sha256": digest(script),
    "all_checkpoint_two_poison_invariant": True,
    "final_full_write_coverage_from_two_poison_invariance": True,
    "input_model_immutable": True,
    "canaries_clean": True,
    "readonly_tail_immutable": True,
    "host_intermediate_ingress": False,
    "expected_or_archived_tensor_ingress": False,
    "address_fields_recorded": False,
    "result": "pass",
}
(root / "backend-receipt.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n")
PY

echo "attempt_root=$attempt"
echo "result=pass"
