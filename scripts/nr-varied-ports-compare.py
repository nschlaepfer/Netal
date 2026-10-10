#!/usr/bin/env python3
"""Compare the six producer-port hashes recorded per frame in a connected-sequence receipt (entry, skip_c256/128/64/32, block0_skip) with the
per-evaluation boundary ports of Codex's RTX varied matrix (block22-middle, block22/14/8/4-native, block0-skip). Bit-level hash equality only."""
import json, sys
MAP = {"entry": "block22-middle", "skip_c256": "block22-native", "skip_c128": "block14-native", "skip_c64": "block8-native", "skip_c32": "block4-native", "block0_skip": "block0-skip"}
def compare(receipt, matrix):
    R = json.load(open(receipt)); M = json.load(open(matrix)); bp = M["boundary_ports"]; out = []; ok = True
    for f in R["frames"]:
        if "regions" not in f: out.append({"frame": f["frame"], "error": f.get("error")}); ok = False; continue
        ports = bp[2 * f["frame"] + 1]; row = {"frame": f["frame"]}
        for mine, theirs in MAP.items():
            a = f["regions"][mine]["sha256"]; b = ports.get(theirs); row[theirs] = {"mac": a, "rtx": b, "equal": a == b}; ok = ok and a == b
        out.append(row)
    return ok, out
if __name__ == "__main__":
    ok, rows = compare(sys.argv[1], sys.argv[2]); print("ports", "EQUAL" if ok else "DIFFER"); [print(" frame", r.get("frame"), [(k, v["equal"]) for k, v in r.items() if isinstance(v, dict)]) for r in rows]; sys.exit(0 if ok else 1)
