#!/usr/bin/env python3
"""Dump the exact operands (A, B, C fragments, all 32 lanes) of one mma.sync statement in one CTA of a recorded NR
launch, plus the emulator's D under two accumulation associations, as a self-contained JSON probe request that can
be replayed on RTX hardware with the same instruction."""
import argparse, json, os, re, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import importlib.util
spec = importlib.util.spec_from_file_location("emu_launch", os.path.join(HERE, "nr-vit-emulate-launch.py")); E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)
from ptx_int_emu import Kernel

def watch(launch, cta, items, env_extra, tmp):
    env = dict(os.environ, EMU_WATCH=",".join(items), EMU_WATCH_LANES=",".join(str(i) for i in range(32)), **env_extra)
    subprocess.run(["rm", "-rf", tmp]); r = subprocess.run([sys.executable, os.path.join(HERE, "nr-vit-emulate-launch.py"), "--launch", str(launch), "--procs", "1", "--only", cta, "--output", tmp], env=env, capture_output=True, text=True)
    if r.returncode: raise SystemExit(r.stdout[-2000:] + r.stderr[-4000:])
    regs = {}
    for line in r.stdout.splitlines():
        m = re.match(r"\[watch\] cta \((\d+), (\d+), 0\) warp (\d+) pc (\d+) (\S+) lanes\[.*?\]: \[(.*)\]", line)
        if m: regs[(int(m.group(4)), m.group(5))] = [int(x.strip().strip("'"), 16) for x in m.group(6).split(",")]
    return regs

def main():
    p = argparse.ArgumentParser(); p.add_argument("--launch", type=int, required=True); p.add_argument("--cta", required=True); p.add_argument("--pc", type=int, required=True)
    p.add_argument("--warp", type=int, default=0); p.add_argument("--output", required=True); a = p.parse_args()
    m = json.load(open(E.MANIFEST)); L = next(l for l in m["launches"] if l["launch_id"] == f"launch_{a.launch:03d}")
    kname = {k["selection_id"]: k["name"] for k in m["selected_kernels"]}[L["selection_id"]]
    k = Kernel(open(E.ptx_path(kname)).read(), kname); s = k.instrs[a.pc]; assert s.startswith("mma"), s
    parts = [q.strip() for q in re.findall(r"\{([^}]*)\}", s)]; D, A, B, C = [[x.strip() for x in q.split(",")] for q in parts]
    items = [f"{a.pc - 1}:{r}:{a.warp}" for r in A + B + C] + [f"{a.pc}:{r}:{a.warp}" for r in D]
    tmp = os.path.join(E.ROOT, "build/nr_dec/watch-tmp-operands")
    seq = watch(a.launch, a.cta, items, {"EMU_MMA_ORDER": "seq"}, tmp); clast = watch(a.launch, a.cta, items, {"EMU_MMA_ORDER": "c_last"}, tmp)
    get = lambda R, pc, r: [f"{v:08x}" for v in R[(pc, r)]]
    lanes = []
    for l in range(32):
        lanes.append({"lane": l, "a": [get(seq, a.pc - 1, r)[l] for r in A], "b": [get(seq, a.pc - 1, r)[l] for r in B], "c": [get(seq, a.pc - 1, r)[l] for r in C],
                      "d_seq": [get(seq, a.pc, r)[l] for r in D], "d_c_last": [get(clast, a.pc, r)[l] for r in D]})
    diff = [(x["lane"], i) for x in lanes for i in range(2) if x["d_seq"][i] != x["d_c_last"][i]]
    out = {"schema": "neuralstudio.nr.mma-operand-probe.v1", "launch": a.launch, "kernel": kname, "cta": [int(v) for v in a.cta.split(",")] + [0], "warp": a.warp, "pc": a.pc, "statement": s,
           "fragment_layout": "PTX mma.sync m16n8k32 row.col e4m3 x e4m3 -> f16 (A: 4 x b32 of 4 e4m3 each, B: 2 x b32, C/D: 2 x f16x2); values are the raw 32-bit register contents per lane",
           "d_seq": "emulator, fp32 FMA chain seeded with C then the 32 products in k order, one RNE to half", "d_c_last": "emulator, fp32 chain over the 32 products from zero, C added last, one RNE to half",
           "lanes_where_orders_differ": diff, "lanes": lanes}
    os.makedirs(os.path.dirname(a.output), exist_ok=True); json.dump(out, open(a.output, "w"), indent=1); print("wrote", a.output, "differences:", diff)
if __name__ == "__main__": main()
