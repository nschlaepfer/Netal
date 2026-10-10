#!/usr/bin/env python3
"""Emit an observation-tap request for one CTA of a recorded launch: the emulator's values of chosen registers right after
chosen statements (all 32 lanes, raw 32-bit), so an RTX run of the original kernel with an observation-only tap can be
compared statement by statement and the first divergence located."""
import argparse, json, os, re, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import importlib.util
spec = importlib.util.spec_from_file_location("emu_launch", os.path.join(HERE, "nr-vit-emulate-launch.py")); E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)
from ptx_int_emu import Kernel
def main():
    p = argparse.ArgumentParser(); p.add_argument("--launch", type=int, required=True); p.add_argument("--cta", required=True); p.add_argument("--warp", type=int, default=0)
    p.add_argument("--mma-range", action="append", default=[], help="pc range LO-HI of mma statements: tap their D registers (after) and named operands (before), e.g. 3511-3526:D,A,B")
    p.add_argument("--output", required=True); a = p.parse_args()
    m = json.load(open(E.MANIFEST)); L = next(l for l in m["launches"] if l["launch_id"] == f"launch_{a.launch:03d}")
    kname = {k["selection_id"]: k["name"] for k in m["selected_kernels"]}[L["selection_id"]]; k = Kernel(open(E.ptx_path(kname)).read(), kname); ins = k.instrs
    items = []      # (pc_after_which_value_is_read, reg, role, mma_pc)
    for rg in a.mma_range:
        rng, roles = rg.split(":"); lo, hi = [int(v) for v in rng.split("-")]
        for i in range(lo, hi + 1):
            if not ins[i].startswith("mma"): continue
            parts = [q.strip() for q in re.findall(r"\{([^}]*)\}", ins[i])]; D, A, B, C = [[x.strip() for x in q.split(",")] for q in parts]
            for role, regs in (("D", D), ("A", A), ("B", B), ("C", C)):
                if role not in roles.split(","): continue
                for r in regs: items.append((i if role == "D" else i - 1, r, role, i))
    seen = set(); items = [it for it in items if not (it[:2] in seen or seen.add(it[:2]))]
    env = dict(os.environ, EMU_WATCH=",".join(f"{pc}:{r}:{a.warp}" for pc, r, _, _ in items), EMU_WATCH_LANES=",".join(str(i) for i in range(32)))
    tmp = os.path.join(E.ROOT, "build/nr_dec/watch-tmp-tap"); subprocess.run(["rm", "-rf", tmp])
    r = subprocess.run([sys.executable, os.path.join(HERE, "nr-vit-emulate-launch.py"), "--launch", str(a.launch), "--procs", "1", "--only", a.cta, "--output", tmp], env=env, capture_output=True, text=True)
    if r.returncode: raise SystemExit(r.stdout[-1500:] + r.stderr[-3000:])
    vals = {}
    for line in r.stdout.splitlines():
        mm = re.match(r"\[watch\] cta \((\d+), (\d+), 0\) warp (\d+) pc (\d+) (\S+) lanes\[.*?\]: \[(.*)\]", line)
        if mm: vals[(int(mm.group(4)), mm.group(5))] = [f"{int(x.strip().strip(chr(39)), 16):08x}" for x in mm.group(6).split(",")]
    taps = [{"statement": pc, "when": "after this statement executes", "register": reg, "role": role, "of_mma_statement": mpc, "ptx": ins[mpc][:140], "emulator_lanes": vals.get((pc, reg))} for pc, reg, role, mpc in items]
    out = {"schema": "neuralstudio.nr.observation-tap-request.v1", "launch": a.launch, "kernel": kname, "cta": [int(v) for v in a.cta.split(",")] + [0], "warp": a.warp,
           "purpose": "locate the first statement where RTX differs from the emulator in this CTA (the only mismatching CTA of the launch: 65 output bytes)",
           "statement_indexing": "0-based index over the kernel's PTX statements as parsed by scripts/ptx_int_emu.py (labels/directives excluded), same as the emulator watch pcs",
           "taps": taps}
    os.makedirs(os.path.dirname(a.output), exist_ok=True); json.dump(out, open(a.output, "w"), indent=1); print("wrote", a.output, len(taps), "taps", sum(1 for t in taps if t["emulator_lanes"] is None), "without emulator values")
if __name__ == "__main__": main()
