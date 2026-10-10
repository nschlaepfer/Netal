#!/usr/bin/env python3
"""Fit/check split-Swin kernel models against the oracle, one tile at a time."""
import sys, os, json, argparse, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nr_split_ref import *
M = json.load(open(emu.MANIFEST)); KN = {k["selection_id"]: k["name"] for k in M["selected_kernels"]}
WB = {w["destination"]["byte_offset"]: w for w in M["base_graph_weight_bindings"]}
def launch(n):
    L = next(l for l in M["launches"] if l["launch_id"] == f"launch_{n:03d}"); b = {}
    for bd in L["arguments"]["bindings"]:
        t = bd["target"]; b[bd["argument_offset"]] = ("P", t["byte_offset"]) if t["id"] == "allocation_003" else ("W", WB[t["byte_offset"]]["tensor_name"])
    return KN[L["selection_id"]], b
def main():
    p = argparse.ArgumentParser(); p.add_argument("--launch", type=int, default=29); p.add_argument("--tiles", default="0,1,14,120,134"); a = p.parse_args()
    kname, b = launch(a.launch); print(kname, b)
    pre = pool_state(a.launch - 1); post = pool_state(a.launch)
    blob = tensor(b[16][1]); m = SplitFfwd(blob)
    xin = b[0][1]; xout = b[8][1]
    for tile in [int(x) for x in a.tiles.split(",")]:
        X = tile_tokens(pre[xin:xin + 2160 * 512], tile)
        out = m(X)                                                  # (16, 512) e4m3 codes, columns = W3 order
        orc = np.frombuffer(bytes(post[xout + tile * 8192: xout + (tile + 1) * 8192]), dtype=np.uint8)
        t = np.arange(16)[:, None]; c = np.arange(512)[None, :]
        for name, idx in (("A-identity", frag_a_index_n(t, c, 512)), ("D-layout", frag_d_index_n(t, c, 512))):
            mine = np.zeros(8192, dtype=np.uint8); mine[idx] = out
            eq = int((mine == orc).sum()); print(f"tile {tile} layout {name}: {eq}/8192 bytes equal")
        # slot-permutation search: for each 16-byte oracle slot, does some model column-set match?
if __name__ == "__main__": main()
