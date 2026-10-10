"""Weight spec for enc3 (blob = weights.bin + 380928; work/native/enc3/phase-report.md section 0)."""
import json, sys
B = 380928
spec = {"matrices": [
    {"name": "Wqkv", "k": 96, "n": 384, "tiles": [[B + 192 + 18432 * (j // 6) + 6144 * (k >> 1) + 1024 * (j % 6) + 512 * (k & 1) for j in range(24)] for k in range(6)]},
    *[{"name": f"Bpos{h}", "k": 64, "n": 64, "fragment": "A", "tiles": [[B + 73920 + 8192 * h + 512 * b + 2048 * bp for bp in range(4)] for b in range(4)]} for h in range(4)],
    {"name": "Wo", "k": 128, "n": 96, "tiles": [[B + 106880 + 6144 * (k >> 1) + 1024 * n + 512 * (k & 1) for n in range(6)] for k in range(8)]},
    {"name": "We", "k": 96, "n": 384, "tiles": [[B + 131840 + 6144 * (j >> 1) + 3072 * (j & 1) + 512 * k for j in range(24)] for k in range(6)]},
    {"name": "Wc", "k": 384, "n": 96, "tiles": [[B + 206336 + 6144 * (kk >> 1) + 1024 * n + 512 * (kk & 1) for n in range(6)] for kk in range(24)]},
    {"name": "W5", "k": 384, "n": 128, "tiles": [[B + 280064 + 12288 * nn + 512 * s for nn in range(8)] for s in range(24)]},
  ],
  "biases": [
    {"name": "Cbo", "n": 96, "offsets": [B + 106688 + 32 * j for j in range(6)]},
    {"name": "Cbe", "n": 384, "offsets": [B + 205568 + 32 * j for j in range(24)]},
    {"name": "Cb5", "n": 128, "offsets": [B + 378368 + 32 * j for j in range(8)]},
    {"name": "C0z", "n": 384, "offsets": []},
  ],
  "vectors": [
    {"name": "G1", "n": 96, "offsets": [B + 0 + 32 * j for j in range(6)]},
    {"name": "S1", "n": 96, "offsets": [B + 131456 + 32 * j for j in range(6)]},
    {"name": "Bc", "n": 96, "offsets": [B + 131648 + 32 * j for j in range(6)]},
    {"name": "Zero96", "n": 96, "offsets": []},
  ]}
out = sys.argv[1] if len(sys.argv) > 1 else "tools/pwin_pipeline/spec-enc3.json"
json.dump(spec, open(out, "w"), indent=1); print("wrote", out)
