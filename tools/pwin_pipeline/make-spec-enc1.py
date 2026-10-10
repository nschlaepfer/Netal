"""Weight spec for enc1 (blob = weights.bin + 67264; offsets from work/native/enc1/phase-report.md section 0)."""
import json, sys
B = 67264
def W1():   # K = 64 (4 k tiles), N = 64 (4 nn tiles)
    t = [[0] * 4 for _ in range(4)]
    for k in range(4):
        for nn in range(4):
            base = {(0, 0): 4224, (1, 0): 10368, (0, 1): 16512, (1, 1): 22656}[(k >> 1, nn >> 1)]
            t[k][nn] = B + base + 512 * (k & 1) + 1024 * (nn & 1)
    return t
spec = {"matrices": [
    {"name": "W1", "k": 64, "n": 64, "tiles": W1()},
    {"name": "M1a", "k": 64, "n": 64, "fragment": "A", "tiles": [[B + 24704 + 512 * r + 2048 * k for k in range(4)] for r in range(4)]},
    {"name": "M1b", "k": 64, "n": 64, "fragment": "A", "tiles": [[B + 32896 + 512 * r + 2048 * k for k in range(4)] for r in range(4)]},
    {"name": "W2", "k": 64, "n": 64, "tiles": [[B + 41216 + 4096 * (k >> 1) + 1024 * n + 512 * (k & 1) for n in range(4)] for k in range(4)]},
    {"name": "We", "k": 64, "n": 256, "tiles": [[B + 49664 + 4096 * (j >> 1) + 2048 * (j & 1) + 512 * k for j in range(16)] for k in range(4)]},
    {"name": "Wc", "k": 256, "n": 64, "tiles": [[B + 82944 + 4096 * (kk >> 1) + 1024 * n + 512 * (kk & 1) for n in range(4)] for kk in range(16)]},
    {"name": "W5", "k": 256, "n": 64, "tiles": [[B + 115712 + 8192 * nn + 512 * s for nn in range(4)] for s in range(16)]},
  ],
  "biases": [   # fp32 [16 x N] broadcast tiles (MMA C operands)
    {"name": "C2", "n": 64, "offsets": [B + 41088 + 32 * j for j in range(4)]},
    {"name": "Ce", "n": 256, "offsets": [B + 82432 + 32 * j for j in range(16)]},
    {"name": "C5", "n": 64, "offsets": [B + 148480 + 32 * j for j in range(4)]},
    {"name": "C0z", "n": 64, "offsets": []},
  ],
  "vectors": [  # fp16 per-channel vectors
    {"name": "G1", "n": 64, "offsets": [B + 0 + 32 * j for j in range(4)]},
    {"name": "S1", "n": 64, "offsets": [B + 49408 + 32 * j for j in range(4)]},
    {"name": "Bc", "n": 64, "offsets": [B + 49536 + 32 * j for j in range(4)]},
    {"name": "Zero64", "n": 64, "offsets": []},
  ]}
out = sys.argv[1] if len(sys.argv) > 1 else "tools/pwin_pipeline/spec-enc1.json"
json.dump(spec, open(out, "w"), indent=1)
print("wrote", out, "W1 tiles", spec["matrices"][0]["tiles"][0][:2], spec["matrices"][0]["tiles"][2][2:])
