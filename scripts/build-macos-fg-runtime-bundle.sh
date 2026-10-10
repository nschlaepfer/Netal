#!/bin/zsh
# Package an accepted FG runtime attempt for the Netal game bridge:
#   scripts/build-macos-fg-runtime-bundle.sh <attempt-dir> <bundle-dir> [model] [rom] [sigmoid] [ex2-unit] [ex2-fraction]
# Produces <bundle-dir>/{libs/*.metallib, model/<tokens>, rom.bin, sigmoid.bin, ex2-unit.bin, ex2-fraction.bin,
# lib/libnetal_fg_runtime.a (x86_64 + arm64), include/fg_runtime.h, manifest.json}. The archive is the
# runtime host (fg_runtime.mm, Metal 4 shim, span recorder, tail bridge, live handoff, the attempt's
# generated runtime_tail.mm) built -O2 -ffp-contract=off, exactly as the receipted hosts are.
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
attempt="$1"; out="$2"
model="${3:-$repo/build/fg_default_readback/eval0-default-all-a5-a/runtime-default-buffers-post-evaluate-0}"
rom="${4:-$repo/build/fg_connected_tail/macos-l56-l87/attempt-exact-v1/sm120-rcp-correction-rom.bin}"
sigmoid="${5:-$repo/build/fg_postprocess_l84/rtx/oracle-006/sigmoid-process-a/sm120-half-logit-sigmoid-lut.bin}"
unit="${6:-$repo/build/fg_math_ex2/compact/attempt-v4/packed-a/unit-corrections.bin}"
fraction="${7:-$repo/build/fg_math_ex2/compact/attempt-v4/packed-a/fraction-corrections.bin}"
[[ -f "$attempt/build/runtime_tail.mm" && -d "$attempt/libs" ]] || { echo "not a runtime attempt: $attempt" >&2; exit 2; }
# Preserve all existing bundles and use a staging directory owned by this invocation.
final="${out:A}"
[[ ! -e "$final" && ! -L "$final" ]] || { print -u2 "Output exists; use a new bundle directory: $final"; exit 2; }
mkdir -p "$(dirname "$final")"
out="$(mktemp -d "${final}.staging.XXXXXX")"
trap 'rm -rf -- "$out"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
mkdir -p "$out/libs" "$out/model" "$out/lib" "$out/include" "$out/obj"
cp "$attempt"/libs/*.metallib "$out/libs/"
cp "$model"/* "$out/model/"
cp "$rom" "$out/rom.bin"; cp "$sigmoid" "$out/sigmoid.bin"; cp "$unit" "$out/ex2-unit.bin"; cp "$fraction" "$out/ex2-fraction.bin"
cp "$repo/src/fg_runtime/fg_runtime.h" "$out/include/"
sources=("$repo/src/fg_runtime/fg_runtime.mm" "$repo/src/fg_runtime/metal4_shim.mm" "$repo/src/fg_runtime/span_recorder.mm"
         "$repo/src/fg_runtime/tail_output_bridge.mm" "$repo/src/fg_production_contract_l3_l87/macos_live_handoff.mm"
         "$attempt/build/runtime_tail.mm")
for arch in x86_64 arm64; do
  objs=()
  for src in "${sources[@]}"; do
    obj="$out/obj/$arch-$(basename "$src" .mm).o"
    xcrun -sdk macosx clang++ -arch "$arch" -std=c++20 -fobjc-arc -ffp-contract=off -O2 -mmacosx-version-min=26.0 \
      -Wall -Wextra -Wno-unused-function -Wno-deprecated-declarations \
      -I "$repo/src/fg_runtime" -I "$repo/src/fg_production_contract_l3_l87" -I "$repo/src/fg_connected_l20_l87" \
      -c "$src" -o "$obj"
    objs+=("$obj")
  done
  xcrun -sdk macosx libtool -static -o "$out/lib/libnetal_fg_runtime-$arch.a" "${objs[@]}"
done
xcrun -sdk macosx lipo -create "$out/lib/libnetal_fg_runtime-x86_64.a" "$out/lib/libnetal_fg_runtime-arm64.a" -o "$out/lib/libnetal_fg_runtime.a"
rm -rf "$out/obj"
python3 - "$out" "$attempt" <<'PY'
import hashlib, json, pathlib, sys
out, attempt = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
files = {}
for p in sorted(out.rglob('*')):
    if p.is_file() and p.name != 'manifest.json':
        files[str(p.relative_to(out))] = {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
receipt = attempt / 'backend-receipt.json'
manifest = {'schema': 'netal.fg.runtime-bundle.v1', 'profile': '640x360', 'source_attempt': str(attempt),
            'backend_receipt': json.loads(receipt.read_text()) if receipt.exists() else None, 'files': files}
(out / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
print(f'bundle={out} files={len(files)}')
PY
lipo -info "$out/lib/libnetal_fg_runtime.a"
python3 - "$out" "$final" <<'PY_END'
from pathlib import Path
import sys
stage, destination = map(Path, sys.argv[1:])
if destination.exists() or destination.is_symlink():
    raise SystemExit('Output appeared during build; preserved')
stage.rename(destination)
PY_END
print "bundle=$final"
print "result=pass"
