#!/usr/bin/env python3
"""Per-site contraction search for a kernel with texture output (post block): evaluate the output-image agreement of a
CTA sample under chosen sets of mul.f16x2->add.f16x2 fusion sites. Modes: list (print candidate sites), single (each
site alone, from the unfused baseline), set (one explicit site set)."""
import argparse, json, os, struct, sys, time
import multiprocessing as mp
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import importlib.util
spec = importlib.util.spec_from_file_location("emu_launch", os.path.join(HERE, "nr-vit-emulate-launch.py")); E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)
import ptx_warp_emu as PW
from ptx_int_emu import Kernel
from nr_vit_ref import tensor

G = {}
def init(ptx, kname, params, block, grid, pool_bytes, weights_bytes, tables, ref):
    G.update(ptx=ptx, kname=kname, params=params, block=block, grid=grid, tables=tables, ref=ref)
    G["gmem"] = {E.POOL: bytearray(pool_bytes), E.WBASE: bytearray(weights_bytes)}; G["pre"] = np.frombuffer(pool_bytes, dtype=np.uint8)

def run(task):
    sites, cta = task
    os.environ["EMU_FUSE_SITES"] = sites
    gm = G["gmem"]; pool = gm[E.POOL]
    c = PW.run_cta(G["ptx"], G["kname"], G["params"], cta, G["block"], G["grid"], gm, G["tables"])
    for a, n in c.written:
        if E.POOL <= a < E.POOL + len(pool):
            o = a - E.POOL; pool[o:o + n] = bytes(G["pre"][o:o + n])
    mism = 0; total = 0
    for rec in PW.OUT_IMAGES.values():
        w = np.nonzero(rec["written"].ravel())[0]
        if w.size:
            idx = (w[:, None] * 4 + np.arange(4)[None, :]).ravel(); mine = rec["img"].reshape(-1)[idx]; mism += int((mine != G["ref"][idx]).sum()); total += int(idx.size)
            rec["img"].reshape(-1, 4)[w] = 0xa5a5; rec["written"].ravel()[w] = False
    return sites, mism, total

def main():
    p = argparse.ArgumentParser(); p.add_argument("--launch", type=int, default=155); p.add_argument("--sample", type=int, default=40); p.add_argument("--procs", type=int, default=8)
    p.add_argument("--mode", default="single", choices=["list", "single", "set", "flip"]); p.add_argument("--sites", default=""); p.add_argument("--base", default="", help="site set always included (single mode adds each candidate to it)"); p.add_argument("--output", default="")
    a = p.parse_args()
    m = json.load(open(E.MANIFEST)); L = next(l for l in m["launches"] if l["launch_id"] == f"launch_{a.launch:03d}")
    kname = {k["selection_id"]: k["name"] for k in m["selected_kernels"]}[L["selection_id"]]; weights = {w["destination"]["byte_offset"]: w for w in m["base_graph_weight_bindings"]}
    ptx = open(E.ptx_path(kname)).read(); k = Kernel(ptx, kname); PW.fuse_mul_add(k); sites = k.fusion_sites
    if a.mode == "list":
        print(len(sites), "candidate sites"); [print(i, kind, k.instrs[i]) for i, kind in sites]; return
    before = E.pool_state(a.launch - 1); params = bytearray(L["arguments"]["bytes"]); params[:] = bytes.fromhex(L["arguments"]["constant_bytes_hex"]); wbinds = []; textures = []; wbytes = b"\0" * 16
    for bnd in L["arguments"]["bindings"]:
        t = bnd["target"]
        if t["id"] == "allocation_003": struct.pack_into("<Q", params, bnd["argument_offset"], E.POOL + t["byte_offset"])
        elif t["id"] == "allocation_001": wbinds.append((bnd["argument_offset"], t["byte_offset"]))
        elif t["kind"] == "texture": struct.pack_into("<Q", params, bnd["argument_offset"], 0x300000000 + int(t["id"][-3:])); textures.append(t["id"])
    base = min(o for _, o in wbinds); blobs = {o: tensor(weights[o]["tensor_name"]).tobytes() for _, o in wbinds}; wb = bytearray(max(o - base + len(b) for o, b in blobs.items()) + 256)
    for o, b in blobs.items(): wb[o - base:o - base + len(b)] = b
    for ao, o in wbinds: struct.pack_into("<Q", params, ao, E.WBASE + (o - base))
    img = np.fromfile(os.path.join(E.ORACLE, "capture/input-color.rgba16le"), dtype=np.uint16).reshape(1080, 1920, 4).view(np.float16).astype(np.float32)
    PW.TEXTURES[0x300000003] = (img, 1920, 1080); PW.OUT_IMAGES[0x300000004] = {"img": np.full((1080, 1920, 4), 0xa5a5, dtype=np.uint16), "written": np.zeros((1080, 1920), dtype=bool)}
    ref = np.fromfile(os.path.join(E.ORACLE, "capture/output.rgba16le"), dtype=np.uint16)
    grid = tuple(L["grid"]); block = tuple(L["block"]); tables = PW.load_tables(os.path.join(E.ROOT, "build/nr_vit/tables"))
    rng = np.random.default_rng(1); allc = [(x, y, 0) for y in range(grid[1]) for x in range(grid[0])]; ctas = [allc[i] for i in sorted(rng.choice(len(allc), a.sample, replace=False))]
    basesites = [x for x in a.base.split(",") if x]
    key = lambda toks: ",".join(sorted(set(toks), key=lambda t: int(t.split(":")[0])))
    if a.mode == "set": sets = [key(basesites + [x for x in a.sites.split(",") if x])]
    elif a.mode == "flip":       # from the base set: for every candidate site try first / second / unfused
        base_map = {int(t.split(":")[0]): t for t in basesites}; sets = [key(basesites)]
        for i, _ in sites:
            for alt in (f"{i}:f", f"{i}:s", None):
                toks = [t for j, t in base_map.items() if j != i] + ([alt] if alt else [])
                k2 = key(toks)
                if k2 not in sets: sets.append(k2)
    else: sets = [key(basesites)] + [key(basesites + [str(i)]) for i, _ in sites]
    tasks = [(s, c) for s in sets for c in ctas]; t0 = time.time(); agg = {}
    ctx = mp.get_context("fork")
    with ctx.Pool(a.procs, initializer=init, initargs=(ptx, kname, bytes(params), block, grid, bytes(before), bytes(wb), tables, ref)) as pool:
        for s, mism, total in pool.imap_unordered(run, tasks, chunksize=2):
            r = agg.setdefault(s, [0, 0]); r[0] += mism; r[1] += total
    base_key = sets[0]; print(f"baseline sites [{base_key}]: {agg[base_key][0]} mismatching of {agg[base_key][1]} ({time.time() - t0:.0f}s)")
    rows = sorted(((agg[s][0], s) for s in sets[1:]), key=lambda r: r[0])
    bset = set(basesites)
    for mism, s in rows: print(f"{mism:7d}  changed {sorted(set(s.split(',')) - bset)} removed {sorted(bset - set(s.split(',')))}")
    if a.output:
        os.makedirs(os.path.dirname(a.output) or ".", exist_ok=True); json.dump({"baseline": {"sites": base_key, "mismatching": agg[base_key][0], "values": agg[base_key][1]}, "results": [{"sites": s, "mismatching": mism} for mism, s in rows], "sample": a.sample}, open(a.output, "w"), indent=1)
if __name__ == "__main__": main()
