"""Weight spec for enc2 (blob = weights.bin + 215872; work/native/enc2/phase-report.md section 0)."""
import json, sys
B = 215872
spec = {"matrices": [
    {"name": "Wqkv", "k": 64, "n": 192, "tiles": [[B + 128 + 12288 * (j // 6) + 6144 * (k >> 1) + 1024 * (j % 6) + 512 * (k & 1) for j in range(12)] for k in range(4)]},
    *[{"name": f"Bpos{h}", "k": 64, "n": 64, "fragment": "A", "tiles": [[B + 24704 + 8192 * h + 512 * b + 2048 * bp for bp in range(4)] for b in range(4)]} for h in range(2)],
    {"name": "Wo", "k": 64, "n": 64, "tiles": [[B + 41216 + 4096 * (k >> 1) + 1024 * n + 512 * (k & 1) for n in range(4)] for k in range(4)]},
    {"name": "We", "k": 64, "n": 256, "tiles": [[B + 49664 + 4096 * (j >> 1) + 2048 * (j & 1) + 512 * k for j in range(16)] for k in range(4)]},
    {"name": "Wc", "k": 256, "n": 64, "tiles": [[B + 82944 + 4096 * (kk >> 1) + 1024 * n + 512 * (kk & 1) for n in range(4)] for kk in range(16)]},
    {"name": "W5", "k": 256, "n": 96, "tiles": [[B + 115712 + 8192 * nn + 512 * s for nn in range(6)] for s in range(16)]},
  ],
  "biases": [
    {"name": "Cbo", "n": 64, "offsets": [B + 41088 + 32 * j for j in range(4)]},
    {"name": "Cbe", "n": 256, "offsets": [B + 82432 + 32 * j for j in range(16)]},
    {"name": "Cb5", "n": 96, "offsets": [B + 164864 + 32 * j for j in range(6)]},
    {"name": "C0z", "n": 192, "offsets": []},
  ],
  "vectors": [
    {"name": "G1", "n": 64, "offsets": [B + 0 + 32 * j for j in range(4)]},
    {"name": "S1", "n": 64, "offsets": [B + 49408 + 32 * j for j in range(4)]},
    {"name": "Bc", "n": 64, "offsets": [B + 49536 + 32 * j for j in range(4)]},
    {"name": "Zero64", "n": 64, "offsets": []},
  ]}
out = sys.argv[1] if len(sys.argv) > 1 else "tools/pwin_pipeline/spec-enc2.json"
json.dump(spec, open(out, "w"), indent=1); print("wrote", out)
