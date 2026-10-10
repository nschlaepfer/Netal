#!/bin/zsh
# Build the colour import/export extension metallib into the runtime bundle.
#
# This library covers the texture formats games hand MetalFX that the verified RE9 converter does
# not: RE9 supplies RGB9E5, Lies of P RGBA16Float and RG11B10Float, Death Stranding BGR10A2Unorm.
# It is deliberately separate from dlss_sr_live_texture_convert.metallib, which belongs to the
# runtime bundle whose output is checked against RTX receipts and must stay byte-for-byte unchanged.
#
# Its absence is not fatal - the adapter treats the library as optional and simply supports fewer
# games - which is why this was easy to leave unscripted. It is scripted now so the bundle can be
# rebuilt from source rather than from a shell command someone once ran.
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"; repository="$(cd "$script_dir/.." && pwd)"
source="$repository/src/macos_host/netal_texture_import_ext.metal"
out_dir="$repository/work/re9-sr-runtime-bundle/metallib"
air="$(mktemp -t netal_ext_air).air"
trap 'rm -f "$air"' EXIT

xcrun -sdk macosx metal -c -O2 -std=metal3.1 "$source" -o "$air"
xcrun -sdk macosx metallib "$air" -o "$out_dir/netal_texture_import_ext.metallib"

print "built=$out_dir/netal_texture_import_ext.metallib"
print "bytes=$(stat -f%z "$out_dir/netal_texture_import_ext.metallib")"
print "functions=$(xcrun -sdk macosx metal-nm "$out_dir/netal_texture_import_ext.metallib" 2>/dev/null | grep -c neuralstudio_sr || true)"
print "result=pass"
