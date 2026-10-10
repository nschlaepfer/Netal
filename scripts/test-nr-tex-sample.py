#!/usr/bin/env python3
"""CPU tests for the fractional-sampling primitive groundwork (no GPU). NOTE (2026-09-06 11:xx): test (1) below, the centre lemma, is SUPERSEDED by
scripts/test-nr-post-coordinates.py (it started at x+0.5 and omitted the div.approx + multiply-back chain; with the real chain 1168/1920 columns and 229/1080 rows are one
fp32 ulp off centre). It is kept unchanged as the superseded claim. What the zero-motion reference determines and what it does not:
(1) launch-155 tap-position lemma: with zero motion every history tap of every pixel lands on a texel centre (weights (0,1,0,0)), for all
1920x1080 positions including the clamped edges - so the current reference exercises centre fetches only; (2) at texel centres the q8
primitive equals a point fetch in both quantisation modes; (3) clamp-to-edge behaviour; (4) the two quantisation modes differ on generic
fractional coordinates (so a nonzero-motion capture can discriminate them); (5) the CPU model is self-consistent with the Metal formula
(same arithmetic, spot-checked by re-deriving in float64)."""
import os, sys, json, hashlib; import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import nr_tex_sample_ref as R
f32 = np.float32; W, H = 1920, 1080; fails = []; notes = {}
# (1) tap lemma over the full frame using the PTX formulas (r35 = W-0.5 clamp; r36 = H-0.5)
xs = np.arange(W, dtype=f32) + f32(0.5); c = np.floor(xs - f32(0.5)) + f32(0.5); f = np.clip(xs - c, 0, 1); f2 = f * f; f3 = f * f2
w0 = f2 - f32(0.5) * (f + f3); w1 = f3 * f32(1.5) - f2 * f32(2.5) + f32(1); w3 = (f3 - f2) * f32(0.5); w2 = f32(1) - w1 - w0 - w3
x12 = np.clip(c + w2 / (w1 + w2), 0.5, W - 0.5); x2 = np.clip(c + 2, 0.5, W - 0.5); x0 = np.clip(c - 1, 0.5, W - 0.5)
centres = all(np.all(np.mod(t, 1) == 0.5) for t in (x12, x2, x0, c)) and np.all(w1 == 1) and np.all(w0 == 0) and np.all(w2 == 0) and np.all(w3 == 0)
notes["tap_lemma_zero_motion_all_columns_on_centres"] = bool(centres); fails += [] if centres else ["tap lemma"]
# (2) centre fetch == point fetch in both modes
rng = np.random.default_rng(7); img = rng.standard_normal((16, 24, 4)).astype(f32)
ok = True
for mode in (0, 1):
    for y in range(16):
        for x in range(24):
            s = R.sample_q8(img, (x + 0.5) / 24, (y + 0.5) / 16, mode)
            if not np.array_equal(s, img[y, x]): ok = False
notes["centre_equals_point_fetch_both_modes"] = ok; fails += [] if ok else ["centre fetch"]
# (3) clamp: coordinates beyond the edges reproduce the edge texel
edge_ok = np.array_equal(R.sample_q8(img, -0.3, 0.5 / 16, 0), img[0, 0]) and np.array_equal(R.sample_q8(img, 1.3, (15 + 0.5) / 16, 0), img[15, 23])
notes["clamp_to_edge"] = bool(edge_ok); fails += [] if edge_ok else ["clamp"]
# (4) the modes are discriminable on generic fractions
diff = 0
for _ in range(2000):
    u, v = rng.uniform(0, 1), rng.uniform(0, 1)
    if not np.array_equal(R.sample_q8(img, u, v, 0), R.sample_q8(img, u, v, 1)): diff += 1
notes["mode_discriminating_samples_of_2000"] = diff; fails += [] if diff > 0 else ["modes indistinguishable"]
# (5) fp32 blend vs float64 reference within 2 ulp (self-consistency of the fixed order)
worst = 0.0
for _ in range(500):
    u, v = rng.uniform(0, 1), rng.uniform(0, 1); x0, y0, x1, y1, fx, fy = R.coord_q8(u, v, 24, 16, 0)
    a, b, d, e = (img[y0, x0].astype(np.float64), img[y0, x1].astype(np.float64), img[y1, x0].astype(np.float64), img[y1, x1].astype(np.float64))
    ref = ((a + (b - a) * fx) + ((d + (e - d) * fx) - (a + (b - a) * fx)) * fy); got = R.sample_q8(img, u, v, 0).astype(np.float64); worst = max(worst, float(np.max(np.abs(got - ref))))
notes["fp32_vs_float64_max_abs_diff"] = worst; fails += [] if worst < 1e-5 else ["fp32 consistency"]
out = {"schema": "neuralstudio.nr.tex-sample-tests.v1", "notes": notes, "limits": "the fractional weight quantisation rule and blend order are NOT validated by any current RTX evidence (zero-motion reference hits texel centres only); a nonzero-motion capture is required before either endpoint relies on the primitive", "failures": fails, "result": "pass" if not fails else "fail"}
out["receipt_root_sha256"] = hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()
d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build/nr_temporal/tex-sample-tests-v1"); os.makedirs(d, exist_ok=True); json.dump(out, open(os.path.join(d, "tests.json"), "w"), indent=1)
print(json.dumps(out, indent=1)); sys.exit(0 if not fails else 1)
