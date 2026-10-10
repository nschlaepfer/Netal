"""Weight spec for dec1 (blob = weights.bin + 3327360; work/native/dec1/phase-report.md section 0)."""
import json, sys
B = 3327360
spec = {"matrices": [
    {"name": "W0", "k": 64, "n": 256, "tiles": [[B + 512 * s + 2048 * N for N in range(16)] for s in range(4)]},
    {"name": "Wqkv", "k": 64, "n": 192, "tiles": [[B + 33408 + 12288 * (j // 6) + 6144 * (k >> 1) + 1024 * (j % 6) + 512 * (k & 1) for j in range(12)] for k in range(4)]},
    *[{"name": f"Bpos{h}", "k": 64, "n": 64, "fragment": "A", "tiles": [[B + 57984 + 8192 * h + 512 * b + 2048 * bp for bp in range(4)] for b in range(4)]} for h in range(2)],
    {"name": "Wo", "k": 64, "n": 64, "tiles": [[B + 74496 + 4096 * (k >> 1) + 1024 * n + 512 * (k & 1) for n in range(4)] for k in range(4)]},
    {"name": "We", "k": 64, "n": 256, "tiles": [[B + 82944 + 4096 * (j >> 1) + 2048 * (j & 1) + 512 * k for j in range(16)] for k in range(4)]},
    {"name": "Wc", "k": 256, "n": 64, "tiles": [[B + 116224 + 4096 * (kk >> 1) + 1024 * n + 512 * (kk & 1) for n in range(4)] for kk in range(16)]},
  ],
  "biases": [
    {"name": "Cb0", "n": 256, "offsets": [B + 32768 + 32 * j for j in range(16)]},
    {"name": "Cbo", "n": 64, "offsets": [B + 74368 + 32 * j for j in range(4)]},
    {"name": "Cbe", "n": 256, "offsets": [B + 115712 + 32 * j for j in range(16)]},
    {"name": "C0z", "n": 192, "offsets": []},
  ],
  "vectors": [
    {"name": "G1", "n": 64, "offsets": [B + 33280 + 32 * j for j in range(4)]},
    {"name": "S1", "n": 64, "offsets": [B + 82688 + 32 * j for j in range(4)]},
    {"name": "Bc", "n": 64, "offsets": [B + 82816 + 32 * j for j in range(4)]},
    {"name": "Zero64", "n": 64, "offsets": []},
  ]}
out = sys.argv[1] if len(sys.argv) > 1 else "tools/pwin_pipeline/spec-dec1.json"
json.dump(spec, open(out, "w"), indent=1); print("wrote", out)
