#!/usr/bin/env python3
"""Transliterate an integer-only PTX kernel (loads, stores, integer ALU, predicates, branches) into a
Metal compute kernel. Control flow becomes a `switch` state machine over basic blocks inside a loop, so
no CFG restructuring is needed. Register widths follow PTX (b16 -> ushort, b32 -> uint, b64 -> ulong);
signed semantics come from the instruction types. Pointers from the parameter block are replaced by fake
integer bases so loads and stores can be routed to the right Metal buffer by range.

Used for the NR ViT repack kernels (2D tin layout <-> 1D token layout); the PTX integer emulator
(scripts/ptx_int_emu.py) is the reference the generated code is checked against.
"""
import argparse, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ptx_int_emu import Kernel

WIDTH = {"8": 8, "16": 16, "32": 32, "64": 64}
def bits_of(t): return int(re.search(r"(8|16|32|64)$", t).group(1))
def ctype(bits): return {8: "ushort", 16: "ushort", 32: "uint", 64: "ulong"}[bits]
def signed(v, bits): return f"int(short({v}))" if bits == 16 else (f"int({v})" if bits == 32 else f"long({v})")
def sext8(v): return f"int(char(uchar({v})))"

def translate(ptx, entry, msl_name, params):
    k = Kernel(ptx, entry)
    ins = k.instrs; n = len(ins)
    starts = {0} | set(k.labels.values())
    for i, s in enumerate(ins):
        if s.startswith("@") or s.startswith("bra") or s.startswith("ret"): starts.add(i + 1)
    starts = sorted(x for x in starts if x < n); blocks = list(zip(starts, starts[1:] + [n])); bid = {st: b for b, (st, en) in enumerate(blocks)}
    regs = {}; preds = set()
    def reg(name):
        w = 32 if name.startswith("%r") and not name.startswith(("%rs", "%rd")) else 16 if name.startswith("%rs") else 64 if name.startswith("%rd") else 32
        regs[name] = w; return name.replace("%", "R_")
    def rd(x, bits):
        x = x.strip()
        if x.startswith("%"):
            reg(x); return f"{ctype(bits)}({x.replace('%', 'R_')})"
        if x in ("%ctaid.x",): return "uint(ctaid)"
        v = int(x, 0) if not x.startswith("0f") else int(x[2:], 16)
        return f"{ctype(bits)}({v & ((1 << bits) - 1)}u)" if bits < 64 else f"ulong({v & ((1 << 64) - 1)}ul)"
    body = []
    def emit(s): body.append("        " + s)
    sregs = {"%ctaid.x": "ctaid", "%ntid.x": "ntid", "%tid.x": "tid", "%nctaid.x": "nctaid"}
    for b, (st, en) in enumerate(blocks):
        body.append(f"      case {b}: {{")
        for i in range(st, en):
            s = ins[i]; pred = None
            if s.startswith("@"):
                pred, s = s.split(" ", 1); neg = pred.startswith("@!"); pv = pred.lstrip("@!").replace("%", "P_")
                cond = f"{'!' if neg else ''}{pv}"
            op, _, rest = s.partition(" "); args = [a.strip() for a in re.split(r",(?![^{]*\})", rest)] if rest else []
            parts = op.split("."); name = parts[0]
            def guard(stmt): emit((f"if ({cond}) " if pred else "") + stmt)
            if name == "ret": guard("return;"); continue
            if name == "bra": guard(f"{{ pc = {bid[k.labels[args[0]]]}; continue; }}"); continue
            if name == "ld" and "param" in parts:
                bits = bits_of(parts[-1]); m = re.match(r"\[%s(?:\+(\d+))?\]" % k.param_name, args[-1]); off = int(m.group(1) or 0)
                names = [x.strip() for x in args[0].strip("{}").split(",")]
                for j, nm in enumerate(names):
                    p = params[off + j * bits // 8]; emit(f"{reg(nm).replace('%','R_')} = {ctype(bits)}({p});")
                continue
            if name == "mov":
                src = args[1]; bits = bits_of(parts[-1])
                val = sregs.get(src) or rd(src, bits)
                if src in sregs: val = f"{ctype(bits)}({val})"
                emit(f"{reg(args[0]).replace('%','R_')} = {val};"); continue
            if name == "cvta": emit(f"{reg(args[0]).replace('%','R_')} = ulong({rd(args[1], 64)});"); continue
            if name in ("add", "sub", "mul", "mad", "and", "or", "xor", "shl", "shr", "div", "rem", "min", "max"):
                t = parts[-1]; bits = bits_of(t); sg = t.startswith("s"); d = reg(args[0]).replace("%", "R_")
                if name == "mul" and len(parts) > 2 and parts[1] == "wide":
                    a = signed(rd(args[1], bits), bits) if sg else rd(args[1], bits); bb = signed(rd(args[2], bits), bits) if sg else rd(args[2], bits)
                    emit(f"{d} = {ctype(bits * 2)}({'long' if sg else 'ulong'}({a}) * {'long' if sg else 'ulong'}({bb}));"); continue
                a = signed(rd(args[1], bits), bits) if sg else rd(args[1], bits); bb = signed(rd(args[2], bits), bits) if sg else rd(args[2], bits)
                if name == "add": e = f"{a} + {bb}"
                elif name == "sub": e = f"{a} - {bb}"
                elif name == "mul": e = f"{a} * {bb}"
                elif name == "mad": c = signed(rd(args[3], bits), bits) if sg else rd(args[3], bits); e = f"{a} * {bb} + {c}"
                elif name == "and": e = f"{rd(args[1], bits)} & {rd(args[2], bits)}"
                elif name == "or": e = f"{rd(args[1], bits)} | {rd(args[2], bits)}"
                elif name == "xor": e = f"{rd(args[1], bits)} ^ {rd(args[2], bits)}"
                elif name == "shl": e = f"{rd(args[1], bits)} << {rd(args[2], 32)}"
                elif name == "shr": e = f"{a} >> {rd(args[2], 32)}"          # arithmetic when signed type
                elif name == "div": e = f"{a} / {bb}"
                elif name == "rem": e = f"{a} % {bb}"
                elif name == "min": e = f"min({a}, {bb})"
                elif name == "max": e = f"max({a}, {bb})"
                emit(f"{d} = {ctype(bits)}({e});"); continue
            if name == "cvt":
                dst_t, src_t = parts[-2], parts[-1]; db, sb = bits_of(dst_t), bits_of(src_t); d = reg(args[0]).replace("%", "R_")
                v = rd(args[1], sb)
                if src_t.startswith("s"): v = signed(v, sb) if sb > 8 else sext8(v)
                if db == 8: v = sext8(v) if dst_t.startswith("s") else f"uint(uchar({v}))"       # 8-bit results live in 16-bit registers
                elif db < sb: v = signed(f"{ctype(db)}({v})", db) if dst_t.startswith("s") else f"{ctype(db)}({v})"
                emit(f"{d} = {ctype(16 if db == 8 else db)}({v});"); continue
            if name == "setp":
                cmp = parts[1]; t = parts[-1]; bits = bits_of(t); sg = t.startswith("s")
                a = signed(rd(args[1], bits), bits) if sg else rd(args[1], bits); bb = signed(rd(args[2], bits), bits) if sg else rd(args[2], bits)
                opx = {"eq": "==", "ne": "!=", "lt": "<", "le": "<=", "gt": ">", "ge": ">=", "lo": "<", "ls": "<=", "hi": ">", "hs": ">="}[cmp]
                preds.add(args[0]); emit(f"{args[0].replace('%','P_')} = ({a}) {opx} ({bb});"); continue
            if name == "selp":
                bits = bits_of(parts[-1]); emit(f"{reg(args[0]).replace('%','R_')} = {args[3].replace('%','P_')} ? {rd(args[1], bits)} : {rd(args[2], bits)};"); continue
            if name == "ld" and "global" in parts:
                bits = bits_of(parts[-1]); addr = rd(args[1].strip("[]"), 64)
                emit(f"{reg(args[0]).replace('%','R_')} = nr_load32({addr}, in_buf, out_buf);"); continue
            if name == "st" and "global" in parts:
                bits = bits_of(parts[-1]); addr = rd(args[0].strip("[]"), 64)
                guard(f"nr_store32({addr}, {rd(args[1], 32)}, out_buf, coverage);"); continue
            raise NotImplementedError(s)
        # fall through to the next block
        if b + 1 < len(blocks): body.append(f"        pc = {b + 1}; continue;")
        else: body.append("        return;")
        body.append("      }")
    decls = "\n".join(f"  {ctype(w)} {nm.replace('%','R_')} = 0;" for nm, w in sorted(regs.items())) + "\n" + "\n".join(f"  bool {nm.replace('%','P_')} = false;" for nm in sorted(preds))
    return f"""
kernel void {msl_name}(device const uint* in_buf [[buffer(0)]], device uint* out_buf [[buffer(1)]], device atomic_uint* coverage [[buffer(2)]],
                       constant NrRepackParams& prm [[buffer(3)]],
                       uint tid [[thread_position_in_threadgroup]], uint ctaid [[threadgroup_position_in_grid]],
                       uint ntid [[threads_per_threadgroup]], uint nctaid [[threadgroups_per_grid]]) {{
{decls}
  uint pc = 0u;
  for (uint guard_iter = 0u; guard_iter < 1000000u; ++guard_iter) {{
    switch (pc) {{
{chr(10).join(body)}
      default: return;
    }}
  }}
}}
"""

HEADER = """#include "nr_vit_common.metal"
// Generated by scripts/ptx_int_to_msl.py from the recovered PTX of the NR ViT repack kernels. Do not edit by hand.
struct NrRepackParams { int width; int height; uint in_dwords; uint out_dwords; };
constant ulong NR_IN_BASE = 0x10000000ul;
constant ulong NR_OUT_BASE = 0x20000000ul;
inline uint nr_load32(ulong addr, device const uint* in_buf, device uint* out_buf) {
  if (addr >= NR_OUT_BASE) return out_buf[(addr - NR_OUT_BASE) >> 2];
  return in_buf[(addr - NR_IN_BASE) >> 2];
}
inline void nr_store32(ulong addr, uint value, device uint* out_buf, device atomic_uint* coverage) {
  const uint i = uint((addr - NR_OUT_BASE) >> 2);
  out_buf[i] = value;
  atomic_fetch_add_explicit(&coverage[i], 1u, memory_order_relaxed);
}
"""

def main():
    p = argparse.ArgumentParser(); p.add_argument("--ptxdir", required=True); p.add_argument("--output", required=True); a = p.parse_args()
    out = [HEADER]
    for entry, msl in (("cc_vit_1d_repack_2d_to_1d_fp8", "nr_vit_repack_2d_to_1d"), ("cc_vit_1d_repack_1d_to_2d_fp8", "nr_vit_repack_1d_to_2d")):
        ptx = open(os.path.join(a.ptxdir, entry + ".ptx")).read()
        params = {0: "NR_IN_BASE", 8: "NR_OUT_BASE", 16: "uint(prm.width)", 20: "uint(prm.height)"}
        out.append(translate(ptx, entry, msl, params))
    open(a.output, "w").write("\n".join(out)); print("wrote", a.output)

if __name__ == "__main__": main()
