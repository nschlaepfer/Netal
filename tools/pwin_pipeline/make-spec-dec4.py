"""Weight spec for dec4 (blob = weights.bin + 2189952; work/native/dec4/phase-report.md section 0)."""
import json, sys
B = 2189952
spec = {"matrices": [
    {"name": "W0", "k": 160, "n": 512, "tiles": [[B + 512 * s + 5120 * N for N in range(32)] for s in range(10)]},
    {"name": "Wqkv", "k": 128, "n": 384, "tiles": [[B + 165120 + 24576 * (j // 6) + 6144 * (k >> 1) + 1024 * (j % 6) + 512 * (k & 1) for j in range(24)] for k in range(8)]},
    *[{"name": f"Bpos{h}", "k": 64, "n": 64, "fragment": "A", "tiles": [[B + 263424 + 8192 * h + 512 * b + 2048 * bp for bp in range(4)] for b in range(4)]} for h in range(4)],
    {"name": "Wo", "k": 128, "n": 128, "tiles": [[B + 296448 + 8192 * (k >> 1) + 1024 * n + 512 * (k & 1) for n in range(8)] for k in range(8)]},
    {"name": "We", "k": 128, "n": 512, "tiles": [[B + 329728 + 8192 * (j >> 1) + 4096 * (j & 1) + 512 * k for j in range(32)] for k in range(8)]},
    {"name": "Wc", "k": 512, "n": 128, "tiles": [[B + 461824 + 8192 * (kk >> 1) + 1024 * n + 512 * (kk & 1) for n in range(8)] for kk in range(32)]},
  ],
  "biases": [
    {"name": "Cb0", "n": 512, "offsets": [B + 163840 + 32 * j for j in range(32)]},
    {"name": "Cbo", "n": 128, "offsets": [B + 296192 + 32 * j for j in range(8)]},
    {"name": "Cbe", "n": 512, "offsets": [B + 460800 + 32 * j for j in range(32)]},
    {"name": "C0z", "n": 384, "offsets": []},
  ],
  "vectors": [
    {"name": "G1", "n": 128, "offsets": [B + 164864 + 32 * j for j in range(8)]},
    {"name": "S1", "n": 128, "offsets": [B + 329216 + 32 * j for j in range(8)]},
    {"name": "Bc", "n": 128, "offsets": [B + 329472 + 32 * j for j in range(8)]},
    {"name": "Zero128", "n": 128, "offsets": []},
  ]}
out = sys.argv[1] if len(sys.argv) > 1 else "tools/pwin_pipeline/spec-dec4.json"
json.dump(spec, open(out, "w"), indent=1); print("wrote", out)
