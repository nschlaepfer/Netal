#!/usr/bin/env bash
set -euo pipefail

repository="$(cd "$(dirname "$0")/.." && pwd)"
attempt="${1:-$repository/build/fg_connected_subrect_l3_l83/macos/tail-$(date -u +%Y%m%dT%H%M%SZ)}"
model="${2:-$repository/build/fg_default_readback/eval0-default-all-a5-a/runtime-default-buffers-post-evaluate-0}"
rom="${3:-$repository/build/fg_connected_tail/macos-l56-l87/attempt-exact-v1/sm120-rcp-correction-rom.bin}"
sigmoid="${4:-$repository/build/fg_postprocess_l84/rtx/oracle-006/sigmoid-process-a/sm120-half-logit-sigmoid-lut.bin}"
ex2_unit="${5:-$repository/build/fg_math_ex2/unit-v1/unit-minus1-to-zero-u32.bin}"
ex2_fraction="${6:-$repository/build/fg_math_ex2/rtx-v3/fraction-minus2-to-minus1-u32.bin}"
if [[ -e "$attempt" ]]; then
  echo "Attempt exists; preserve it: $attempt" >&2
  exit 2
fi
build="$attempt/build"
libs="$attempt/libs"
mkdir -p "$build" "$libs"

compile_metal() {
  local source="$1" name="$2"
  xcrun -sdk macosx metal -std=metal3.1 -fno-fast-math -ffp-contract=off \
    -I "$repository/src/fg_conv" \
    -I "$repository/src/fg_connected_custom_l56_l83" \
    -I "$repository/src/fg_production_contract_l3_l87" \
    -I "$repository/src/fg_math_ex2" \
    -c "$source" -o "$libs/$name.air"
  xcrun -sdk macosx metallib "$libs/$name.air" -o "$libs/$name.metallib"
}

compile_metal "$repository/src/fg_conv/conv_3x3.metal" conv_3x3
compile_metal "$repository/src/fg_production_contract_l3_l87/conv_1x1_temporal.metal" conv_1x1
compile_metal "$repository/src/fg_pooling/pooling.metal" pooling
compile_metal "$repository/src/fg_upscale/upscale.metal" upscale-prefix
compile_metal "$repository/src/fg_elementwise/elementwise_family.metal" elementwise
compile_metal "$repository/src/fg_production_contract_l3_l87/l52_production_exact.metal" mid_l52
compile_metal "$repository/src/fg_mid_l53/mid_l53.metal" mid_l53
compile_metal "$repository/src/fg_production_contract_l3_l87/l54_production_exact.metal" mid_l54
compile_metal "$repository/src/fg_initial_merge/initial_merge.metal" initial_merge
compile_metal "$repository/src/fg_custom_upsample/custom_upsample.metal" upsample
compile_metal "$repository/src/fg_central_block/central_block.metal" central
compile_metal "$repository/src/fg_connected_subrect_l3_l83/subrect_postprocess_l84.metal" postprocess_l84_subrect

python3 -B "$repository/scripts/generate-fg-connected-subrect-l3-l83-mid-l55.py" \
  "$repository/src/fg_mid_l55/fg_mid_l55.metal" \
  "$build/fg_mid_l55_subrect.metal"
compile_metal "$build/fg_mid_l55_subrect.metal" mid_l55

python3 -B "$repository/scripts/generate-fg-connected-subrect-l3-l83-module19.py" \
  "$repository/src/fg_postprocess_l86/postprocess_l86_mode0.metal" \
  "$build/postprocess_l86_mode1_subrect.metal"
compile_metal "$build/postprocess_l86_mode1_subrect.metal" postprocess_l86_subrect

custom="$build/custom"
python3 -B "$repository/scripts/generate-fg-connected-subrect-l3-l83-custom-shaders.py" \
  "$repository" "$custom"
compile_metal "$custom/custom_conv_pre.metal" conv_pre
compile_metal "$custom/custom_conv0_c8.metal" conv0
compile_metal "$custom/custom_conv1.metal" conv1
compile_metal "$custom/custom_conv2.metal" conv2
compile_metal "$custom/connected_block0_bot0.metal" bot0
compile_metal "$custom/connected_block0_bot1_production.metal" bot1
compile_metal "$custom/custom_block1_conv0.metal" block1_conv0
compile_metal "$custom/custom_block1_conv1.metal" block1_conv1
compile_metal "$custom/custom_block1_conv2.metal" block1_conv2
compile_metal "$custom/connected_block1_bot0.metal" block1_bot0
compile_metal "$custom/connected_block1_bot1.metal" block1_bot1

python3 -B "$repository/scripts/generate-fg-connected-subrect-l3-l83-mac-tail.py" \
  "$repository/src/fg_connected_l20_l87/macos_connected_l20_l87.mm" \
  "$build/macos_subrect_tail.mm"
xcrun -sdk macosx clang++ -std=c++20 -fobjc-arc -Wall -Wextra -Werror \
  -Wno-unused-function -Wno-deprecated-declarations \
  -I "$repository/src/fg_connected_subrect_l3_l83" \
  -I "$repository/src/fg_connected_l20_l87" \
  "$build/macos_subrect_tail.mm" -framework Foundation -framework Metal \
  -o "$build/subrect-tail"

for poison in a5 5a; do
  for process in a b; do
    run_id="subrect-tail-poison-$poison-process-$process"
    run="$attempt/$run_id"
    mkdir -p "$run"
    "$build/subrect-tail" "$libs" "$model" "$rom" "$sigmoid" \
      "$ex2_unit" "$ex2_fraction" "$run" "$run_id" "$poison" \
      >"$run/stdout.log" 2>"$run/stderr.log"
  done
done

python3 -B - "$attempt" "$repository" <<'PY'
import hashlib,json,pathlib,platform,subprocess,sys
root,repository=map(pathlib.Path,sys.argv[1:])
runs=sorted(path for path in root.glob('subrect-tail-poison-*-process-*')
            if path.is_dir())
if len(runs)!=4:raise SystemExit('four run directories required')
receipts=[json.loads((run/'run-receipt.json').read_text()) for run in runs]
if len({row['process_id'] for row in receipts})!=4:
    raise SystemExit('fresh process gate')
if sorted(row['poison_byte'] for row in receipts)!=[90,90,165,165]:
    raise SystemExit('poison coverage')
if any(row['result']!='pass' for row in receipts):raise SystemExit('run failed')
maps={json.dumps(row['checkpoint_sha256'],sort_keys=True,separators=(',',':'))
      for row in receipts}
regions={row['output_region_sha256'] for row in receipts}
if len(maps)!=1 or len(regions)!=1:raise SystemExit('region invariance')
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
system=subprocess.run(['system_profiler','SPDisplaysDataType'],check=True,
                      capture_output=True,text=True).stdout.splitlines()
gpu=next((line.split(':',1)[1].strip() for line in system
          if 'Chipset Model:' in line),'unknown')
backend={
 'schema':'neuralstudio.fg.connected-subrect-l18-l83-mac-backend.v1',
 'host':{'machine':platform.machine(),'macos':platform.mac_ver()[0],'gpu':gpu},
 'source_sha256':sha(root/'build/macos_subrect_tail.mm'),
 'binary_sha256':sha(root/'build/subrect-tail'),
 'processes':[{'run_id':run.name,'process_id':row['process_id'],
   'poison_byte':row['poison_byte'],'output_full_sha256':row['output_full_sha256'],
   'output_region_sha256':row['output_region_sha256']}
   for run,row in zip(runs,receipts)],
 'fresh_process_count':4,'poison_bytes':[90,165],
 'launch_range':[18,83],'launch_count':66,
 'checkpoint_sha256':receipts[0]['checkpoint_sha256'],
 'output_region_sha256':receipts[0]['output_region_sha256'],
 'full_surface_poison_invariant':False,'region_scoped_poison_invariant':True,
 'input_model_table_immutability':True,'canary_integrity':True,
 'live_device_handoffs':True,'expected_or_archived_tensor_ingress':False,
 'address_fields_recorded':False,'result':'pass'}
(root/'backend-receipt.json').write_text(
    json.dumps(backend,indent=2,sort_keys=True)+'\n')
print(f"output_region_sha256={backend['output_region_sha256']}")
print('result=pass')
PY

echo "attempt_root=$attempt"
echo "result=pass"
