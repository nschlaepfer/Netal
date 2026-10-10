#!/usr/bin/env python3
"""Check the SplitAttention model against the oracle for whole windows (all 16 heads) of qkv launches, including
out-of-bounds and shifted windows. Reports per (launch, CTA, head) match counts and which quad orders / q modes survive."""
import sys, os, json, argparse, itertools, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nr_split_ref import *
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = json.load(open(emu.MANIFEST)); WB = {w["destination"]["byte_offset"]: w for w in M["base_graph_weight_bindings"]}
GRID_W, GRID_H, TILES_X = 60, 36, 15
def launch_info(n):
    L = next(l for l in M["launches"] if l["launch_id"] == f"launch_{n:03d}"); b = {}
    for bd in L["arguments"]["bindings"]:
        t = bd["target"]; b[bd["argument_offset"]] = ("P", t["byte_offset"]) if t["id"] == "allocation_003" else ("W", WB[t["byte_offset"]]["tensor_name"])
    c = bytes.fromhex(L["arguments"]["constant_bytes_hex"]); sx, sy = np.frombuffer(c[32:40], dtype=np.int32)
    return b, int(sx), int(sy)
def window_tiles(x, y, sx, sy):
    """m-tile order of the 4 spatial 4x4 tiles of window (x, y) with token-space shift (sx, sy); None = out of bounds."""
    ox, oy = 8 * x + sx, 8 * y + sy; out = []
    for ty in (oy // 4, oy // 4 + 1):
        for tx in (ox // 4, ox // 4 + 1):
            out.append(None if (tx < 0 or ty < 0 or tx >= TILES_X or ty >= 9) else ty * TILES_X + tx)
    return out
def main():
    p = argparse.ArgumentParser(); p.add_argument("--launch", type=int, default=27); p.add_argument("--ctas", default="0,0;7,0;0,4;7,4;3,2"); p.add_argument("--heads", default="all")
    p.add_argument("--search", action="store_true", help="on mismatch, search quad orders and q modes"); a = p.parse_args()
    b, sx, sy = launch_info(a.launch); pre = pool_state(a.launch - 1); post = pool_state(a.launch); blob = tensor(b[16][1]); tables = load_tables(ROOT)
    xin = b[0][1]; xout = b[8][1]; print(f"launch {a.launch} shift=({sx},{sy}) in={xin} out={xout} weights={b[16][1]}")
    t16 = np.arange(16)[:, None]; ia = frag_a_index_n(t16, np.arange(512)[None, :], 512)
    heads = range(16) if a.heads == "all" else [int(h) for h in a.heads.split(",")]
    models = {h: SplitAttention(blob, tables, h) for h in heads}
    total_ok = total = 0
    for cta in a.ctas.split(";"):
        x, y = [int(v) for v in cta.split(",")]; tiles = window_tiles(x, y, sx, sy)
        X = np.concatenate([tile_tokens(pre[xin:xin + 2160 * 512], T) if T is not None else np.zeros((16, 512), np.float32) for T in tiles], axis=0)
        orc = np.concatenate([np.frombuffer(bytes(post[xout + T * 8192: xout + (T + 1) * 8192]), dtype=np.uint8)[ia] if T is not None else np.full((16, 512), 255, np.uint8) for T in tiles], axis=0)
        valid = np.concatenate([np.full(16, T is not None) for T in tiles])
        for h in heads:
            out = models[h](X); mine = np.zeros((64, 32), dtype=np.uint8); mine[:, CFD[:32]] = out
            eq = int((mine[valid] == orc[valid, 32 * h:32 * h + 32]).sum()); n = int(valid.sum()) * 32; total_ok += eq; total += n
            flag = "" if eq == n else "  <-- MISMATCH"
            print(f"  cta ({x},{y}) tiles {tiles} head {h:2d}: {eq}/{n}{flag}", flush=True)
            if eq != n and a.search:
                for qm in ("inv_then_scale", "scale_then_inv"):
                    for perm in itertools.permutations(range(4)):
                        o2 = models[h](X, quad_order=perm, q_mode=qm); m2 = np.zeros((64, 32), dtype=np.uint8); m2[:, CFD[:32]] = o2
                        e2 = int((m2[valid] == orc[valid, 32 * h:32 * h + 32]).sum())
                        if e2 == n: print(f"      exact with q_mode={qm} quad_order={perm}")
    print(f"TOTAL {total_ok}/{total}")
if __name__ == "__main__": main()
