#!/usr/bin/env python3
"""Race hypothesis for a single mismatching CTA of a chained launch: emulate the CTA with one of its input tiles replaced by
that tile's content BEFORE the producing launch (a stale read), and report whether the stored bytes then match the oracle."""
import argparse, json, os, struct, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import importlib.util
spec = importlib.util.spec_from_file_location("emu_launch", os.path.join(HERE, "nr-vit-emulate-launch.py")); E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)
import ptx_warp_emu as PW
from nr_vit_ref import tensor
def main():
    p = argparse.ArgumentParser(); p.add_argument("--launch", type=int, required=True); p.add_argument("--cta", required=True); p.add_argument("--stale-from", type=int, required=True, help="pool state index whose bytes replace an input tile (e.g. launch-2 for 'before the previous launch wrote it')")
    p.add_argument("--region", required=True, help="base,size of the input region whose tiles are candidates"); p.add_argument("--tile", type=int, default=1024); a = p.parse_args()
    m = json.load(open(E.MANIFEST)); L = next(l for l in m["launches"] if l["launch_id"] == f"launch_{a.launch:03d}")
    kname = {k["selection_id"]: k["name"] for k in m["selected_kernels"]}[L["selection_id"]]; weights = {w["destination"]["byte_offset"]: w for w in m["base_graph_weight_bindings"]}
    ptx = open(E.ptx_path(kname)).read(); before = bytearray(E.pool_state(a.launch - 1)); after = np.frombuffer(bytes(E.pool_state(a.launch)), dtype=np.uint8); stale = E.pool_state(a.stale_from)
    params = bytearray(L["arguments"]["bytes"]); params[:] = bytes.fromhex(L["arguments"]["constant_bytes_hex"]); wbytes = b"\0" * 16
    for bnd in L["arguments"]["bindings"]:
        t = bnd["target"]
        if t["id"] == "allocation_003": struct.pack_into("<Q", params, bnd["argument_offset"], E.POOL + t["byte_offset"])
        elif t["id"] == "allocation_001": struct.pack_into("<Q", params, bnd["argument_offset"], E.WBASE); wbytes = tensor(weights[t["byte_offset"]]["tensor_name"]).tobytes() + b"\0" * 256
    grid = tuple(L["grid"]); block = tuple(L["block"]); cta = tuple(int(v) for v in a.cta.split(",")) + (0,); tables = PW.load_tables(os.path.join(E.ROOT, "build/nr_vit/tables"))
    rb, rs = [int(v) for v in a.region.split(",")]
    def run(pool_bytes, track=False):
        os.environ["EMU_TRACK_READS"] = "1" if track else ""
        gm = {E.POOL: bytearray(pool_bytes), E.WBASE: bytearray(wbytes)}
        c = PW.run_cta(ptx, kname, bytes(params), cta, block, grid, gm, tables)
        cur = np.frombuffer(gm[E.POOL], dtype=np.uint8)
        idx = np.unique(np.concatenate([np.arange(x - E.POOL, x - E.POOL + n) for x, n in c.written if E.POOL <= x < E.POOL + len(pool_bytes)]))
        return int((cur[idx] != after[idx]).sum()), c
    base_mism, c = run(bytes(before), track=True)
    tiles = sorted({(x - E.POOL) // a.tile for x, n in c.reads if E.POOL + rb <= x < E.POOL + rb + rs})
    print(f"baseline mismatching bytes {base_mism}; input tiles read in region: {len(tiles)}")
    for t in tiles:
        lo = t * a.tile; hi = lo + a.tile
        if bytes(before[lo:hi]) == bytes(stale[lo:hi]): print(f"tile {t} (offset {lo}): identical before/stale, skipped"); continue
        mod = bytearray(before); mod[lo:hi] = stale[lo:hi]; mism, _ = run(bytes(mod))
        print(f"tile {t} (offset {lo}) stale -> mismatching bytes {mism}" + ("   <== EXACT" if mism == 0 else ""))
if __name__ == "__main__": main()
