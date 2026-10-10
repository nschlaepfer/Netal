#!/usr/bin/env python3
"""Read the mul.f16x2 -> add.f16x2 contraction choice per PTX site from the driver-JIT SASS. For the q/k sum-of-squares sites
(x_a*x_a + x_b*x_b, both operands direct MMA outputs) the SASS HFMA2's fused operand register is a QMMA destination; matching
(MMA program index, D slot) pairs between PTX and SASS names the fused PTX operand. Prints an EMU_FUSE_SITES string."""
import argparse, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from ptx_int_emu import Kernel
def main():
    p = argparse.ArgumentParser(); p.add_argument("--ptx", required=True); p.add_argument("--entry", required=True); p.add_argument("--sass", required=True); a = p.parse_args()
    k = Kernel(open(a.ptx).read(), a.entry); ins = k.instrs
    defs = {}
    for i, s in enumerate(ins):
        if s.startswith(("st.", "@", "bra")): continue
        m = re.match(r"\S+\s+(%\w+),", s)
        if m: defs.setdefault(m.group(1), []).append(i)
        m2 = re.match(r"\S+\s+\{([^}]*)\},", s)
        if m2:
            for r in [x.strip() for x in m2.group(1).split(",")]: defs.setdefault(r, []).append(i)
    def prod(r, before):
        ds = [d for d in defs.get(r, []) if d < before]; return ds[-1] if ds else None
    # PTX e4m3 MMAs in program order: D register -> (index, slot)
    mmas = [i for i, s in enumerate(ins) if s.startswith("mma") and "e4m3" in s]; dmap = {}
    for idx, i in enumerate(mmas):
        d = [x.strip() for x in re.findall(r"\{([^}]*)\}", ins[i])[0].split(",")]
        for slot, r in enumerate(d): dmap[(r, i)] = (idx, slot)
    def dslot(r, before):
        pc = prod(r, before)
        if pc is None or not ins[pc].startswith("mma"): return None
        return dmap.get((r, pc))
    sites = []
    for i, s in enumerate(ins):
        if not s.startswith("add.f16x2"): continue
        d, pa, pb = [x.strip() for x in s.split(" ", 1)[1].split(",")]; ia, ib = prod(pa, i), prod(pb, i)
        if ia is None or ib is None or not ins[ia].startswith("mul.f16x2") or not ins[ib].startswith("mul.f16x2"): continue
        ma = [x.strip() for x in ins[ia].split(" ", 1)[1].split(",")]; mb = [x.strip() for x in ins[ib].split(" ", 1)[1].split(",")]
        if ma[1] != ma[2] or mb[1] != mb[2]: continue                       # squares only
        ta, tb = dslot(ma[1], ia), dslot(mb[1], ib)
        if ta is None or tb is None: continue
        sites.append((i, ta, tb))
    # SASS: QMMA in program order, D base register -> (index, slot)
    sass = [l for l in open(a.sass)]
    qm = [(j, re.search(r"QMMA\S*\s+R(\d+),", l).group(1)) for j, l in enumerate(sass) if "QMMA" in l]
    smap = {}
    for idx, (j, r) in enumerate(qm): smap[(int(r), j)] = (idx, 0); smap[(int(r) + 1, j)] = (idx, 1)
    def sass_writer(reg, before):
        for j in range(before - 1, -1, -1):
            m = re.search(r"\*/\s+(\S+)\s+R(\d+)", sass[j])
            if m and int(m.group(2)) == reg and not m.group(1).startswith(("ST", "@", "BRA")): return j, m.group(1)
        return None
    def sass_dslot(reg, before):
        """(QMMA index, slot) that produced register reg: a QMMA names only the base register of its D pair, so the odd
        register of a pair is found as base+1; a MOV copy is followed one level"""
        for j in range(before - 1, -1, -1):
            m = re.search(r"\*/\s+(\S+)\s+R(\d+)", sass[j])
            if not m or m.group(1).startswith(("ST", "@", "BRA")): continue
            op, dst = m.group(1), int(m.group(2))
            if "QMMA" in op and dst in (reg, reg - 1): return smap.get((reg, j))
            if dst == reg:
                if op.startswith("MOV"):
                    mm = re.search(r"MOV\S*\s+R\d+, R(\d+)", sass[j]); return sass_dslot(int(mm.group(1)), j) if mm else None
                return None
        return None
    pairs = {}
    for j, l in enumerate(sass):
        m = re.search(r"HFMA2 R(\d+), R(\d+)(?:\.reuse)?, R\2(?:\.reuse)?, R(\d+)", l)
        if not m: continue
        rx, ry = int(m.group(2)), int(m.group(3)); fx = sass_dslot(rx, j)
        w = sass_writer(ry, j)
        if w is None or w[1] != "HMUL2": continue
        mm = re.search(r"HMUL2 R\d+, R(\d+)(?:\.reuse)?, R(\d+)", sass[w[0]]); 
        if not mm or mm.group(1) != mm.group(2): continue
        fy = sass_dslot(int(mm.group(1)), w[0])
        if fx is None or fy is None: print(f"  unresolved SASS site at line {j}: fused R{rx} -> {fx}, explicit R{mm.group(1)} -> {fy}"); continue
        pairs[frozenset([fx, fy])] = fx
    out = []; unmatched = 0
    for i, ta, tb in sites:
        key = frozenset([ta, tb]); fx = pairs.get(key)
        if fx is None: unmatched += 1; print(f"  PTX site {i}: MMA pair {ta},{tb} not found in SASS"); continue
        kind = "f" if fx == ta else "s"; out.append(f"{i}:{kind}"); print(f"  PTX site {i}: first operand = mma {ta}, second = mma {tb}; SASS fused mma {fx} -> {'FIRST' if kind == 'f' else 'SECOND'}")
    print(f"{len(sites)} square sites, {len(pairs)} SASS square contractions, {unmatched} unmatched")
    print("EMU_FUSE_SITES_SQUARES=" + ",".join(out))
if __name__ == "__main__": main()
