"""Weight spec for enc4 (blob = weights.bin + 759552; work/native/enc4/phase-report.md section 0)."""
import json, sys
B = 759552
spec = {"matrices": [
    {"name": "Wqkv", "k": 128, "n": 384, "tiles": [[B + 256 + 24576 * (j // 6) + 6144 * (k >> 1) + 1024 * (j % 6) + 512 * (k & 1) for j in range(24)] for k in range(8)]},
    *[{"name": f"Bpos{h}", "k": 64, "n": 64, "fragment": "A", "tiles": [[B + 98560 + 8192 * h + 512 * b + 2048 * bp for bp in range(4)] for b in range(4)]} for h in range(4)],
    {"name": "Wo", "k": 128, "n": 128, "tiles": [[B + 131584 + 8192 * (k >> 1) + 1024 * n + 512 * (k & 1) for n in range(8)] for k in range(8)]},
    {"name": "We", "k": 128, "n": 512, "tiles": [[B + 164864 + 8192 * (j >> 1) + 4096 * (j & 1) + 512 * k for j in range(32)] for k in range(8)]},
    {"name": "Wc", "k": 512, "n": 128, "tiles": [[B + 296960 + 8192 * (kk >> 1) + 1024 * n + 512 * (kk & 1) for n in range(8)] for kk in range(32)]},
    {"name": "W5p", "k": 512, "n": 192, "tiles": [[B + 428032 + 16384 * nn + 512 * s for nn in range(12)] for s in range(32)]},
  ],
  "biases": [
    {"name": "Cbo", "n": 128, "offsets": [B + 131328 + 32 * j for j in range(8)]},
    {"name": "Cbe", "n": 512, "offsets": [B + 295936 + 32 * j for j in range(32)]},
    {"name": "Cb5p", "n": 192, "offsets": [B + 624640 + 32 * j for j in range(12)]},
    {"name": "C0z", "n": 384, "offsets": []},
  ],
  "vectors": [
    {"name": "G1", "n": 128, "offsets": [B + 0 + 32 * j for j in range(8)]},
    {"name": "S1", "n": 128, "offsets": [B + 164352 + 32 * j for j in range(8)]},
    {"name": "Bc", "n": 128, "offsets": [B + 164608 + 32 * j for j in range(8)]},
    {"name": "Zero128", "n": 128, "offsets": []},
  ]}
# Four 48-channel groups contain 40 live projection channels each. The slim
# projection and bias keep those channels in group order; do not simply crop to 160.
spec["derived"] = [
    {"name":"W5s", "source":"W5p", "operation":"compact_groups", "group_width":48, "keep":40},
    {"name":"Cb5s", "source":"Cb5p", "operation":"compact_groups", "group_width":48, "keep":40},
]
out = sys.argv[1] if len(sys.argv) > 1 else "tools/pwin_pipeline/spec-enc4.json"
json.dump(spec, open(out, "w"), indent=1); print("wrote", out)
