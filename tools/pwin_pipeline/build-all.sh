#!/bin/zsh
# Concatenate the pipeline kernels into one metallib (build/pwin_pipeline/pwin_all.metallib).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"; repository="$(cd "$here/../.." && pwd)"
out="$repository/build/pwin_pipeline"; mkdir -p "$out"
srcs=(pwin_gemm.metal pwin_gemm2.metal pwin_gemm3.metal pwin_norm.metal pwin_mix.metal pwin_mix2.metal pwin_mix3.metal pwin_mix4.metal pwin_mix5.metal pwin_mixp.metal pwin_mixp2.metal pwin_mixm.metal pwin_mixm2.metal pwin_merge.metal pwin_d2s.metal pwin_pad.metal pwin_pad2.metal pwin_mlp.metal pwin_mlp2.metal pwin_attn.metal pwin_dbg.metal)
: > "$out/pwin_all.metal"
for s in $srcs; do print "// ==== $s" >> "$out/pwin_all.metal"; cat "$here/$s" >> "$out/pwin_all.metal"; done
xcrun metal -std=metal4.0 -fno-fast-math -c "$out/pwin_all.metal" -o "$out/pwin_all.air"
xcrun metallib "$out/pwin_all.air" -o "$out/pwin_all.metallib"
print "built=$out/pwin_all.metallib"
