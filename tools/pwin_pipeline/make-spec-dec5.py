"""Weight spec for dec5 (blob = weights.bin + 1384576; work/native/dec5/phase-report.md section 0)."""
import json, sys
B = 1384576
spec = {"matrices": [
    {"name": "Wqkv", "k": 160, "n": 768, "tiles": [[B + 320 + 30720 * (j // 6) + 6144 * (k >> 1) + 1024 * (j % 6) + 512 * (k & 1) for j in range(48)] for k in range(10)]},
    *[{"name": f"Bpos{h}", "k": 64, "n": 64, "fragment": "A", "tiles": [[B + 246080 + 8192 * h + 512 * b + 2048 * bp for bp in range(4)] for b in range(4)]} for h in range(8)],
    {"name": "Wo", "k": 256, "n": 160, "tiles": [[B + 311936 + 10240 * (k >> 1) + 1024 * n + 512 * (k & 1) for n in range(10)] for k in range(16)]},
    {"name": "We", "k": 160, "n": 640, "tiles": [[B + 394496 + 10240 * (j >> 1) + 5120 * (j & 1) + 512 * k for j in range(40)] for k in range(10)]},
    {"name": "Wc", "k": 640, "n": 160, "tiles": [[B + 600576 + 10240 * (kk >> 1) + 1024 * n + 512 * (kk & 1) for n in range(10)] for kk in range(40)]},
  ],
  "biases": [
    {"name": "Cbo", "n": 160, "offsets": [B + 311616 + 32 * j for j in range(10)]},
    {"name": "Cbe", "n": 640, "offsets": [B + 599296 + 32 * j for j in range(40)]},
    {"name": "C0z", "n": 768, "offsets": []},
  ],
  "vectors": [
    {"name": "G1", "n": 160, "offsets": [B + 0 + 32 * j for j in range(10)]},
    {"name": "S1", "n": 160, "offsets": [B + 393856 + 32 * j for j in range(10)]},
    {"name": "Bc", "n": 160, "offsets": [B + 394176 + 32 * j for j in range(10)]},
    {"name": "Zero160", "n": 160, "offsets": []},
  ]}
out = sys.argv[1] if len(sys.argv) > 1 else "tools/pwin_pipeline/spec-dec5.json"
json.dump(spec, open(out, "w"), indent=1); print("wrote", out)
