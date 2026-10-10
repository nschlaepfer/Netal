"""Weight spec for dec2 (blob = weights.bin + 3161984; work/native/dec2/phase-report.md section 0)."""
import json, sys
B = 3161984
spec = {"matrices": [
    {"name": "W0", "k": 96, "n": 256, "tiles": [[B + 512 * s + 3072 * N for N in range(16)] for s in range(6)]},
    {"name": "Wqkv", "k": 64, "n": 192, "tiles": [[B + 49792 + 12288 * (j // 6) + 6144 * (k >> 1) + 1024 * (j % 6) + 512 * (k & 1) for j in range(12)] for k in range(4)]},
    *[{"name": f"Bpos{h}", "k": 64, "n": 64, "fragment": "A", "tiles": [[B + 74368 + 8192 * h + 512 * b + 2048 * bp for bp in range(4)] for b in range(4)]} for h in range(2)],
    {"name": "Wo", "k": 64, "n": 64, "tiles": [[B + 90880 + 4096 * (k >> 1) + 1024 * n + 512 * (k & 1) for n in range(4)] for k in range(4)]},
    {"name": "We", "k": 64, "n": 256, "tiles": [[B + 99328 + 4096 * (j >> 1) + 2048 * (j & 1) + 512 * k for j in range(16)] for k in range(4)]},
    {"name": "Wc", "k": 256, "n": 64, "tiles": [[B + 132608 + 4096 * (kk >> 1) + 1024 * n + 512 * (kk & 1) for n in range(4)] for kk in range(16)]},
  ],
  "biases": [
    {"name": "Cb0", "n": 256, "offsets": [B + 49152 + 32 * j for j in range(16)]},
    {"name": "Cbo", "n": 64, "offsets": [B + 90752 + 32 * j for j in range(4)]},
    {"name": "Cbe", "n": 256, "offsets": [B + 132096 + 32 * j for j in range(16)]},
    {"name": "C0z", "n": 192, "offsets": []},
  ],
  "vectors": [
    {"name": "G1", "n": 64, "offsets": [B + 49664 + 32 * j for j in range(4)]},
    {"name": "S1", "n": 64, "offsets": [B + 99072 + 32 * j for j in range(4)]},
    {"name": "Bc", "n": 64, "offsets": [B + 99200 + 32 * j for j in range(4)]},
    {"name": "Zero64", "n": 64, "offsets": []},
  ]}
out = sys.argv[1] if len(sys.argv) > 1 else "tools/pwin_pipeline/spec-dec2.json"
json.dump(spec, open(out, "w"), indent=1); print("wrote", out)
