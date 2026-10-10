#!/usr/bin/env python3
"""Sampler edge-case oracle comparison (CPU only): regenerates Codex's 22-case RGBA16F atlas and 1,078 queries in RAM from the case table
(src/nr_probes/nr_sampler_edge_inputs.h in Codex's checkout; hashes must equal the fixture), runs the CPU RTX sampler rule (ptx_warp_emu.tex_bilinear_rtx,
the single implementation) and compares every returned FP32 bit pattern with the RTX capture (four fresh processes, must be identical).
Usage: nr-sampler-edge-compare.py FIXTURE.json RTX_RUN_DIR [RTX_RUN_DIR...] OUT.json"""
import hashlib, json, os, struct, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from ptx_warp_emu import tex_bilinear_rtx
fix = json.load(open(sys.argv[1])); runs = sys.argv[2:-1]; out = sys.argv[-1]; W, H = fix["width"], fix["height"]; cases = fix["cases"]
img16 = np.zeros((H, W, 4), dtype=np.uint16)
for n, c in enumerate(cases):
    for tap in range(4):
        for ch in range(4): img16[8 + tap // 2, 8 * n + 2 + tap % 2, ch] = c["taps"][(tap + ch) % 4]
assert hashlib.sha256(img16.tobytes()).hexdigest() == fix["input_sha256"], "regenerated atlas hash differs from the fixture"
F = [0.0, 1.0 / 512, 127.0 / 256, 0.5, 129.0 / 256, 255.0 / 256, 1.0]; q = []
for n in range(len(cases)):
    for y in F:
        for x in F: q.append((np.float32((np.float32(8 * n + 2) + np.float32(0.5) + np.float32(x)) / np.float32(W)), np.float32((np.float32(8.5) + np.float32(y)) / np.float32(H))))
qb = b"".join(struct.pack("<ff", float(a), float(b)) for a, b in q); assert hashlib.sha256(qb).hexdigest() == fix["queries_sha256"], "regenerated queries differ from the fixture"
rtx = None
for r in runs:
    s = open(os.path.join(r, "samples.bin"), "rb").read(); assert len(s) == len(q) * 16, r
    if rtx is None: rtx = s
    assert s == rtx, f"RTX processes disagree: {r}"
rtx_bits = np.frombuffer(rtx, dtype=np.uint32).reshape(len(q), 4)
img = img16.view(np.float16).astype(np.float32)
u = np.array([a for a, _ in q], dtype=np.float64); v = np.array([b for _, b in q], dtype=np.float64)
cpu = tex_bilinear_rtx(img, W, H, u, v)   # list of 4 channel arrays (float64 carrying float32 values)
cpu_bits = np.stack([np.asarray(ch, dtype=np.float64).astype(np.float32).view(np.uint32) for ch in cpu], axis=1)
mism = cpu_bits != rtx_bits; per_case = []; cex = []
for n, c in enumerate(cases):
    sl = slice(49 * n, 49 * (n + 1)); m = int(mism[sl].sum()); per_case.append({"case": c["name"], "values": 196, "mismatching_values": m})
    for qi in range(49 * n, 49 * (n + 1)):
        for ch in range(4):
            if mism[qi, ch]: cex.append({"query": qi, "channel": ch, "case": c["name"], "cpu_bits": "%08x" % cpu_bits[qi, ch], "rtx_bits": "%08x" % rtx_bits[qi, ch]})
R = {"schema": "neuralstudio.nr.sampler-edge-cpu-comparison.v2", "scope": "CPU rule (ptx_warp_emu.tex_bilinear_rtx) vs the repeated RTX primitive on the regenerated atlas; NOT Metal execution or model acceptance", "emulator_source_sha256": hashlib.sha256(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ptx_warp_emu.py"), "rb").read()).hexdigest(), "input_sha256": fix["input_sha256"], "queries_sha256": fix["queries_sha256"], "rtx_runs": runs, "rtx_samples_sha256": hashlib.sha256(rtx).hexdigest(), "values": int(mism.size), "mismatching_values": int(mism.sum()), "cases": per_case, "counterexamples": cex[:400], "counterexamples_total": len(cex)}
json.dump(R, open(out, "w"), indent=1); print("values", mism.size, "mismatching", int(mism.sum())); [print(f"  {c['case']:28s} {c['mismatching_values']}") for c in per_case if c["mismatching_values"]]
sys.exit(0 if not mism.any() else 1)
