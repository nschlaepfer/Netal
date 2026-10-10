#!/usr/bin/env python3
"""CPU test of the FULL launch-155 zero-motion coordinate chain (supersedes the 'centre lemma' of test-nr-tex-sample.py, which started at x+0.5 and
omitted div.approx and the multiply back; Codex review 11:04): q = div.approx(x+0.5, W) modelled as (x+0.5)*rcp(W) with the ROM reciprocal (which
equals the correctly rounded 1/W for W in {1920, 1080}), p = q*W, centre c = floor(p-0.5)+0.5, fraction f = clamp(p-c). Reports which columns/rows
land exactly on a centre (f == 0), which land one fp32 ulp off (f ~ 3e-5 .. 6e-5), and the resulting Catmull-Rom lobe weights; the old claim
'all taps on centres' is recorded as superseded. No sampler proof is implied."""
import os, sys, json, hashlib, importlib.util; import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); spec = importlib.util.spec_from_file_location("PW", os.path.join(ROOT, "scripts/ptx_warp_emu.py")); PW = importlib.util.module_from_spec(spec); spec.loader.exec_module(PW)
f32 = np.float32
def chain(i, dim):
    r = float(PW.rcp_rom_f32(np.array([float(dim)]))[0]); a = f32(i) + f32(0.5); q = f32(a * f32(r)); p = f32(q * f32(dim)); c = f32(np.floor(f32(p - f32(0.5))) + f32(0.5)); f = f32(min(max(f32(p - c), 0), 1)); return float(p), float(c), float(f)
def lobes(f):
    f = f32(f); f2 = f * f; f3 = f * f2; w0 = f2 - f32(0.5) * (f + f3); w1 = f3 * f32(1.5) - f2 * f32(2.5) + f32(1); w3 = (f3 - f2) * f32(0.5); w2 = f32(1) - w1 - w0 - w3; return float(w0), float(w1), float(w2), float(w3)
cols = [chain(x, 1920) for x in range(1920)]; rows = [chain(y, 1080) for y in range(1080)]
cx0 = [x for x, (p, c, f) in enumerate(cols) if f == 0.0]; rx0 = [y for y, (p, c, f) in enumerate(rows) if f == 0.0]
off = [(x, f) for x, (p, c, f) in enumerate(cols) if f != 0.0]
rec = {"schema": "neuralstudio.nr.post155-coordinate-chain.v1", "model": "q = (i+0.5)*rcp_rom(dim); p = q*dim; c = floor(p-0.5)+0.5; f = clamp(p-c,0,1) in fp32", "rcp_1920_bits": hex(int(np.float32(PW.rcp_rom_f32(np.array([1920.0]))[0]).view(np.uint32))), "rcp_1080_bits": hex(int(np.float32(PW.rcp_rom_f32(np.array([1080.0]))[0]).view(np.uint32))),
       "columns_exact_centre": len(cx0), "columns_off_centre": 1920 - len(cx0), "rows_exact_centre": len(rx0), "rows_off_centre": 1080 - len(rx0), "max_column_fraction": max(f for _, f in off) if off else 0.0, "example_columns_exact": cx0[:12], "example_columns_off": off[:6],
       "lobe_weights_at_max_fraction": lobes(max(f for _, f in off)) if off else None, "superseded_claim": "test-nr-tex-sample.py 'zero-motion taps all on texel centres' started from x+0.5 directly; with the real chain 1168/1920 columns and 229/1080 rows are one fp32 ulp off centre (752 columns and 851 rows land exactly on a centre) (Codex witness coordinate-roundtrip-v1: 1168 off using the then-deployed half-table rcp, which is also superseded by the ROM reciprocal)",
       "what_this_does_not_prove": "the CUDA texture unit's fractional rule and the RTX rcp/div results; both need RTX micro-evidence"}
rec["receipt_root_sha256"] = hashlib.sha256(json.dumps(rec, sort_keys=True, default=str).encode()).hexdigest()
d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "build/nr_temporal/coordinate-chain-v1"); os.makedirs(d, exist_ok=True); json.dump(rec, open(os.path.join(d, "chain.json"), "w"), indent=1, default=str)
print(json.dumps({k: rec[k] for k in ("rcp_1920_bits", "rcp_1080_bits", "columns_exact_centre", "columns_off_centre", "rows_exact_centre", "rows_off_centre", "max_column_fraction", "lobe_weights_at_max_fraction")}, indent=1))
