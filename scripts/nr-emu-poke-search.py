#!/usr/bin/env python3
"""Perturbation search on one CTA of a recorded NR launch: re-run the CTA once per candidate register poke (one
e4m3 byte or one half of a packed register moved by +-1 unit right after its defining statement) and count how many
of the CTA's stored bytes disagree with the oracle. A poke that drives the count to zero names the single
intermediate value whose rounding differs from RTX; partial improvements point at the stage."""
import argparse, json, os, re, struct, sys, time
import multiprocessing as mp
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import importlib.util
spec = importlib.util.spec_from_file_location("emu_launch", os.path.join(HERE, "nr-vit-emulate-launch.py")); E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)
import ptx_warp_emu as PW
from ptx_int_emu import Kernel
from nr_vit_ref import tensor

G = {}
def init(ptx, kname, params, block, grid, pool_bytes, weights_bytes, tables, after_bytes, cta):
    G.update(ptx=ptx, kname=kname, params=params, block=block, grid=grid, tables=tables, cta=cta)
    G["gmem"] = {E.POOL: bytearray(pool_bytes), E.WBASE: bytearray(weights_bytes)}
    G["pre"] = np.frombuffer(pool_bytes, dtype=np.uint8); G["after"] = np.frombuffer(after_bytes, dtype=np.uint8)

def run(poke):
    PW.POKES = {poke[0]: [tuple(poke[1:])]} if poke else {}
    gm = G["gmem"]; pool = gm[E.POOL]
    cta = PW.run_cta(G["ptx"], G["kname"], G["params"], G["cta"], G["block"], G["grid"], gm, G["tables"])
    idx = np.unique(np.concatenate([np.arange(a - E.POOL, a - E.POOL + n) for a, n in cta.written if E.POOL <= a < E.POOL + len(pool)]))
    cur = np.frombuffer(pool, dtype=np.uint8); vals = cur[idx].copy(); mism = int((vals != G["after"][idx]).sum()); del cur
    for i in idx.tolist(): pool[i] = int(G["pre"][i])
    return poke, mism

def operand_defs(k):
    defs = {}
    for i, s in enumerate(k.instrs):
        if s.startswith(("@", "bra", "ret", "st.", "red.", "bar", "cp.")): continue
        m = re.match(r"\S+\s+(\{[^}]*\}|%\S+?),", s)
        if m:
            for r in re.findall(r"%r\d+|%rs\d+|%rd\d+", m.group(1)): defs.setdefault(r, []).append(i)
    return defs

def candidates(k, ranges, operand, warps):
    defs = operand_defs(k); seen = set(); out = []
    for lo, hi in ranges:
        for i in range(lo, hi + 1):
            s = k.instrs[i]
            if not s.startswith("mma"): continue
            parts = [p.strip() for p in re.findall(r"\{([^}]*)\}", s)]
            regs = [x.strip() for x in parts[{"D": 0, "A": 1, "B": 2, "C": 3}[operand]].split(",")]
            for r in regs:
                pc = i if operand == "D" else max(d for d in defs[r] if d < i)
                if (pc, r) in seen: continue
                seen.add((pc, r))
                kinds = ("lo", "hi") if operand == "D" else ("b0", "b1", "b2", "b3")
                for w in warps:
                    for lane in range(32):
                        for kind in kinds:
                            for delta in (1, -1): out.append((pc, r, w, lane, kind, delta))
    return out

def main():
    p = argparse.ArgumentParser(); p.add_argument("--launch", type=int, required=True); p.add_argument("--cta", required=True)
    p.add_argument("--custom", default="", help="pc:reg:kind[|kind] entries (lanes 0-31, +-1) instead of mma operands"); p.add_argument("--mma-range", default="", help="pc ranges of the mma statements, e.g. 4125-4140,4882-4897"); p.add_argument("--operand", default="A", choices=["A", "B", "C", "D"])
    p.add_argument("--warps", default="0"); p.add_argument("--procs", type=int, default=8); p.add_argument("--output", required=True); p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    if os.path.exists(a.output): raise SystemExit("preserve previous run: " + a.output)
    m = json.load(open(E.MANIFEST)); L = next(l for l in m["launches"] if l["launch_id"] == f"launch_{a.launch:03d}")
    kname = {k["selection_id"]: k["name"] for k in m["selected_kernels"]}[L["selection_id"]]
    weights = {w["destination"]["byte_offset"]: w for w in m["base_graph_weight_bindings"]}
    ptx = open(E.ptx_path(kname)).read(); before = E.pool_state(a.launch - 1); after = E.pool_state(a.launch)
    params = bytearray(L["arguments"]["bytes"]); params[:] = bytes.fromhex(L["arguments"]["constant_bytes_hex"]); wbytes = b"\0" * 16
    for bnd in L["arguments"]["bindings"]:
        t = bnd["target"]
        if t["id"] == "allocation_003": struct.pack_into("<Q", params, bnd["argument_offset"], E.POOL + t["byte_offset"])
        elif t["id"] == "allocation_001": struct.pack_into("<Q", params, bnd["argument_offset"], E.WBASE); wbytes = tensor(weights[t["byte_offset"]]["tensor_name"]).tobytes() + b"\0" * 256
    grid = tuple(L["grid"]); block = tuple(L["block"]); cta = tuple(int(v) for v in a.cta.split(",")) + (0,)
    tables = PW.load_tables(os.path.join(E.ROOT, "build/nr_vit/tables"))
    k = Kernel(ptx, kname); ranges = [tuple(int(v) for v in r.split("-")) for r in a.mma_range.split(",") if r]
    cands = candidates(k, ranges, a.operand, [int(w) for w in a.warps.split(",")]) if a.mma_range else []
    for item in filter(None, a.custom.split(",")):
        pc, reg, kinds = item.split(":")
        for w in [int(x) for x in a.warps.split(",")]:
            for lane in range(32):
                for kind in kinds.split("|"):
                    for delta in (1, -1): cands.append((int(pc), reg, w, lane, kind, delta))
    if a.limit: cands = cands[:a.limit]
    os.makedirs(a.output); t0 = time.time(); results = []
    ctx = mp.get_context("fork")
    with ctx.Pool(a.procs, initializer=init, initargs=(ptx, kname, bytes(params), block, grid, bytes(before), wbytes, tables, bytes(after), cta)) as pool:
        for n, (poke, mism) in enumerate(pool.imap_unordered(run, [None] + cands, chunksize=4)):
            results.append((mism, poke))
            if n % 500 == 0: print(f"{n}/{len(cands)} done, {time.time() - t0:.0f}s, best so far {min(r[0] for r in results)}", flush=True)
    results.sort(key=lambda r: (r[0], str(r[1])))
    with open(os.path.join(a.output, "results.txt"), "w") as f:
        for mism, poke in results: f.write(f"{mism} {poke}\n")
    print("baseline", [r for r in results if r[1] is None][0][0], "best", results[:10], f"{time.time() - t0:.0f}s")
if __name__ == "__main__": main()
