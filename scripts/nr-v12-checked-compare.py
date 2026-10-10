#!/usr/bin/env python3
"""Per-launch SHA-256 identity: v12 checked mode (in-buffer snapshots hashed at completion) vs v11 checked mode (CPU hash after each launch) on the
SAME descriptor/inputs. Usage: nr-v12-checked-compare.py V11_CHAIN_RUN.json V12_CHAIN_RUN.json OUT.json
Compares every middle/repack/ViT output (role by role), the boundary identities (middle view, decoder pre-image, final pool, launch-155 output,
final surface) and lists what v12 does NOT compare (per-launch decoder pool SHA-256: digest-only in v12). Exit 0 only if every compared hash matches."""
import json, sys
v11 = json.load(open(sys.argv[1])); v12 = json.load(open(sys.argv[2]))
s12 = {}
for l in v12["launches"]:
    for k, v in l["digests"].items():
        if k.endswith("_sha256"): s12[(int(l["launch"]), k[:-7])] = v
matched, mismatched, uncompared = [], [], []
for l in v11["launches"]:
    n = int(l["launch"])
    for role, h in (l.get("outputs") or {}).items():
        key = (n, role)
        if key in s12: (matched if s12[key] == h else mismatched).append({"launch": n, "role": role, "v11": h, "v12": s12[key]})
        else: uncompared.append({"launch": n, "role": role, "v11": h, "why": "no v12 snapshot for this role"})
    if l.get("kind") == "ptx":
        key = (n, "pool")
        if key in s12: (matched if s12[key] == l.get("pool_post_sha256") else mismatched).append({"launch": n, "role": "pool_post_sha256", "v11": l.get("pool_post_sha256"), "v12": s12[key]})
        else: uncompared.append({"launch": n, "role": "pool_post_sha256", "v11": l.get("pool_post_sha256"), "why": "no per-launch pool snapshot (v12 digest-only; v13 checked mode snapshots)"})
        t = l.get("texture") or {}
        if n == 155 and t.get("output_sha256") is not None: (matched if v12.get("tex155_output_sha256") == t["output_sha256"] else mismatched).append({"launch": 155, "role": "texture_out", "v11": t["output_sha256"], "v12": v12.get("tex155_output_sha256")})
d11 = v11.get("decoder") or {}
for name, a, b in (("decoder_pre_image", d11.get("pool_pre_sha256"), v12.get("decoder_pre_image_sha256")), ("pool_final", d11.get("pool_final_sha256"), v12.get("pool_final_sha256"))):
    if a is None or b is None: uncompared.append({"role": name, "v11": a, "v12": b, "why": "field absent"})
    else: (matched if a == b else mismatched).append({"role": name, "v11": a, "v12": b})
v12_only = [k for k in s12 if k not in {(int(l["launch"]), r) for l in v11["launches"] for r in (l.get("outputs") or {})}]
R = {"schema": "neuralstudio.nr.v12-checked-compare.v1", "v11": sys.argv[1], "v12": sys.argv[2], "matched": len(matched), "mismatched": mismatched, "uncompared": uncompared, "v12_only_snapshots": [{"launch": k[0], "role": k[1]} for k in sorted(v12_only)], "all_compared_match": not mismatched and len(matched) > 0, "matched_list": matched}
json.dump(R, open(sys.argv[3], "w"), indent=1)
print(f"matched {len(matched)} mismatched {len(mismatched)} uncompared {len(uncompared)} v12-only {len(v12_only)} -> {'ALL COMPARED MATCH' if R['all_compared_match'] else 'MISMATCH'}")
sys.exit(0 if R["all_compared_match"] else 1)
