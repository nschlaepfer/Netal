"""Weight spec for dec3 (blob = weights.bin + 2782848; work/native/dec3/phase-report.md section 0)."""
import json, sys
B = 2782848
spec = {"matrices": [
    {"name": "W0", "k": 128, "n": 384, "tiles": [[B + 4096 * N + 512 * s for N in range(24)] for s in range(8)]},
    {"name": "Wqkv", "k": 96, "n": 384, "tiles": [[B + 99264 + 18432 * (j // 6) + 6144 * (k >> 1) + 1024 * (j % 6) + 512 * (k & 1) for j in range(24)] for k in range(6)]},
    *[{"name": f"Bpos{h}", "k": 64, "n": 64, "fragment": "A", "tiles": [[B + 172992 + 8192 * h + 512 * b + 2048 * bp for bp in range(4)] for b in range(4)]} for h in range(4)],
    {"name": "Wo", "k": 128, "n": 96, "tiles": [[B + 205952 + 6144 * (k >> 1) + 1024 * n + 512 * (k & 1) for n in range(6)] for k in range(8)]},
    {"name": "We", "k": 96, "n": 384, "tiles": [[B + 230912 + 6144 * (j >> 1) + 3072 * (j & 1) + 512 * k for j in range(24)] for k in range(6)]},
    {"name": "Wc", "k": 384, "n": 96, "tiles": [[B + 305408 + 6144 * (kk >> 1) + 1024 * n + 512 * (kk & 1) for n in range(6)] for kk in range(24)]},
  ],
  "biases": [
    {"name": "Cb0", "n": 384, "offsets": [B + 98304 + 32 * j for j in range(24)]},
    {"name": "Cbo", "n": 96, "offsets": [B + 205760 + 32 * j for j in range(6)]},
    {"name": "Cbe", "n": 384, "offsets": [B + 304640 + 32 * j for j in range(24)]},
    {"name": "C0z", "n": 384, "offsets": []},
  ],
  "vectors": [
    {"name": "G1", "n": 96, "offsets": [B + 99072 + 32 * j for j in range(6)]},
    {"name": "S1", "n": 96, "offsets": [B + 230528 + 32 * j for j in range(6)]},
    {"name": "Bc", "n": 96, "offsets": [B + 230720 + 32 * j for j in range(6)]},
    {"name": "Zero96", "n": 96, "offsets": []},
  ]}
out = sys.argv[1] if len(sys.argv) > 1 else "tools/pwin_pipeline/spec-dec3.json"
json.dump(spec, open(out, "w"), indent=1); print("wrote", out)
