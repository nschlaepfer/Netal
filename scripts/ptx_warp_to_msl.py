#!/usr/bin/env python3
"""Transliterate a recorded NR PTX kernel into a Metal compute kernel, one Metal thread per PTX lane.

Semantics mirror scripts/ptx_warp_emu.py (the emulator that reproduces the RTX launches byte for byte):
registers become per-thread integer containers, f16x2/f32 ops act through as_type views, mma.sync /
shfl / movmatrix use SIMD-group shuffles, shared memory is threadgroup memory, bar.sync is a
threadgroup-wide wait protocol run at the top of the block-dispatch loop, and lanes reconverge at every
block that contains a warp-synchronous op (all 32 lanes must be present before it runs). Control flow is a
per-thread state machine over basic blocks. The same ptxas mul+add contraction pre-pass is applied.
"""
import argparse, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ptx_int_emu import Kernel
from ptx_warp_emu import fuse_mul_add, ex2_half_sites

SREGS = {"%tid.x": "tid.x", "%tid.y": "tid.y", "%tid.z": "tid.z", "%ctaid.x": "ctaid.x", "%ctaid.y": "ctaid.y", "%ctaid.z": "ctaid.z",
         "%ntid.x": "ntid.x", "%ntid.y": "ntid.y", "%ntid.z": "ntid.z", "%nctaid.x": "nctaid.x", "%nctaid.y": "nctaid.y", "%nctaid.z": "nctaid.z",
         "%laneid": "lane", "%warpid": "warp", "WARP_SZ": "32u"}
NAMED = {"hl": 16, "hu": 16, "low": 16, "fl": 32, "fu": 32, "smem": 64, "P_OUT": 1, "v": 128}
SYNC_OPS = ("mma", "shfl", "movmatrix", "bar", "elect")
def bits_of(t):
    m = re.search(r"(128|64|32|16|8)$", t); return int(m.group(1)) if m else 32
def ctype(b): return {1: "bool", 8: "ushort", 16: "ushort", 32: "uint", 64: "ulong", 128: "uint4"}[b]
def sgn(v, b): return {16: f"short({v})", 32: f"int({v})", 64: f"long({v})"}[b]

class Emitter:
    def __init__(self, ptx, entry, tables=True, watches=(), watch_cta=(0, 0, 0), watch_warp=0, coverage=True):
        self.k = Kernel(ptx, entry); fuse_mul_add(self.k); self.entry = entry; self.coverage = coverage
        self.watches = {}; self.watch_cta = watch_cta; self.watch_warp = watch_warp
        for i, w in enumerate(watches): pc, reg = w.split(":"); self.watches.setdefault(int(pc), []).append((reg, i))
        self.ins = self.k.instrs; self.regs = {}; self.preds = set(); self.unsupported = {}; self.approx = {}
        self.shared_size = 0
        for m in re.finditer(r"\.shared\s+\.align\s+(\d+)\s+\.b8\s+(\S+)\[(\d+)\]", ptx):
            self.shared_size = max(self.shared_size, self.k.shared_syms.get(m.group(2), 0) + int(m.group(3)))
        self.body = []
    # ---- register/operand helpers ----
    def reg(self, name, bits=None):
        name = name.strip()
        if name in NAMED: self.regs[name] = NAMED[name]; return name
        if name.startswith("%p"): self.preds.add(name); return "P_" + name[2:]
        w = 64 if name.startswith("%rd") else 16 if name.startswith("%rs") else 32
        self.regs[name] = w; return "R_" + name[1:]
    def rd(self, x, bits):
        x = x.strip()
        if x in SREGS: return f"{ctype(bits)}({SREGS[x]})"
        if x.startswith("%") or x in NAMED: return f"{ctype(bits)}({self.reg(x)})"
        if x.startswith("0f"): return f"{ctype(bits)}({int(x[2:], 16)}u)"
        if x.startswith("0d"): return f"{ctype(bits)}({int(x[2:], 16)}ul)"
        if x in self.k.shared_syms: return f"{ctype(bits)}({self.k.shared_syms[x]}u)"
        if x == self.k.param_name or x.startswith("cc_") or x.startswith("_ZZ"): return f"{ctype(bits)}(0u)"
        v = int(x, 0)
        if bits == 64: return f"ulong({v & ((1 << 64) - 1)}ul)"
        return f"{ctype(bits)}({v & ((1 << bits) - 1)}u)"
    def rd_f32(self, x): return f"nr_f32({self.rd(x, 32)})"
    def rd_h2(self, x): return f"nr_h2({self.rd(x, 32)})"
    def wr(self, d, expr, bits):
        d = d.strip()
        if d == "_": return ""
        r = self.reg(d); return f"{r} = {ctype(bits if bits != 8 else 16)}({expr});"
    def addr(self, tok, bits=64):
        inner = tok.strip().strip("[]")
        if "+" in inner:
            base, off = inner.split("+", 1); return f"({self.rd(base, bits)} + {ctype(bits)}({int(off, 0)}u))"
        return self.rd(inner, bits)
    # ---- one instruction -> MSL statements ----
    def emit(self, s, out):
        s = s.strip()
        while s.startswith("{"): s = s[1:].strip()
        if s.endswith("}") and s.count("{") < s.count("}"): s = s[:-1].strip()
        if not s or s.startswith(".reg") or s.startswith(".pragma"): return
        pred = None
        if s.startswith("@"):
            p, s = s.split(" ", 1); neg = p.startswith("@!"); pred = ("!" if neg else "") + self.reg(p.lstrip("@!"))
        op, _, rest = s.partition(" "); args = [a.strip() for a in re.split(r",(?![^{]*\})", rest.rstrip(";"))] if rest else []
        parts = op.split("."); name = parts[0]
        def put(stmt):
            if stmt: out.append((f"if ({pred}) {{ {stmt} }}" if pred else stmt))
        if name == "ret": put("done = true; break;" if getattr(self, "single_warp", False) else "atomic_fetch_add_explicit(&exit_count, 1u, memory_order_relaxed); done = true; break;"); return
        if name == "bra": put(f"pc = {self.block_of[self.k.labels[args[0]]]}u; break;"); return
        if name == "ld" and "param" in parts:
            b = bits_of(parts[-1]); inner = args[-1].strip().strip("[]")
            m = re.match(r"%s(?:\+(\d+))?$" % re.escape(self.k.param_name), inner)
            if m: base = f"{int(m.group(1) or 0)}u"
            else:
                reg, _, off = inner.partition("+"); base = f"(uint({self.rd(reg, 64)}) + {int(off or 0, 0)}u)"   # register-based: the register holds the param base (0)
            names = [x.strip() for x in args[0].strip("{}").split(",")]
            for i, nm in enumerate(names): put(self.wr(nm, f"nr_ld_param{b}(prm, {base} + {i * b // 8}u)", b))
            return
        if name == "mov":
            t = parts[-1]
            if t == "pred":
                src = args[1]; v = self.reg(src) if src.startswith("%") else ("true" if int(src, 0) else "false"); put(f"{self.reg(args[0])} = {v};"); return
            b = bits_of(t)
            if args[1].startswith("{"):
                srcs = [x.strip() for x in args[1].strip("{}").split(",")]; part = b // len(srcs)
                if b == 128: put(f"{self.reg(args[0])} = uint4({', '.join(self.rd(x, 32) for x in srcs)});"); return
                expr = " | ".join(f"({ctype(b)}({self.rd(x, part)}) << {part * i}u)" for i, x in enumerate(srcs)); put(self.wr(args[0], expr, b)); return
            if args[0].startswith("{"):
                dsts = [x.strip() for x in args[0].strip("{}").split(",")]; part = b // len(dsts); src = self.rd(args[1], b)
                for i, d in enumerate(dsts):
                    if d != "_": put(self.wr(d, f"({src} >> {part * i}u)", part))
                return
            put(self.wr(args[0], self.rd(args[1], b) if args[1] not in SREGS else SREGS[args[1]], b)); return
        if name == "cvta": put(self.wr(args[0], self.rd(args[1], 64), 64)); return
        if name in ("bar", "barrier"): put("atomic_fetch_add_explicit(&bar_count, 1u, memory_order_relaxed); bar_wait = true; pc = pc_next; break;"); return
        if name == "nanosleep" or name == "fence" or name == "membar": return
        if name == "elect":
            m = re.search(r"\|(\S+)", args[0]); 
            if m: put(f"{self.reg(m.group(1))} = true;")
            return
        if name == "setp":
            cmp_ = parts[1]; t = parts[-1]; b = bits_of(t)
            if t.startswith("f") or "ftz" in parts:
                a, c = self.rd_f32(args[1]), self.rd_f32(args[2]); opx = {"eq": "==", "ne": "!=", "lt": "<", "le": "<=", "gt": ">", "ge": ">=", "equ": "==", "neu": "!=", "ltu": "<", "leu": "<=", "gtu": ">", "geu": ">="}[cmp_]
                e = f"({a}) {opx} ({c})"
                if cmp_ in ("equ", "neu", "ltu", "leu", "gtu", "geu"): e = f"(isnan({a}) || isnan({c}) || ({e}))"
                put(f"{self.reg(args[0])} = {e};"); return
            sg = t.startswith("s"); a = sgn(self.rd(args[1], b), b) if sg else self.rd(args[1], b); c = sgn(self.rd(args[2], b), b) if sg else self.rd(args[2], b)
            opx = {"eq": "==", "ne": "!=", "lt": "<", "le": "<=", "gt": ">", "ge": ">=", "lo": "<", "ls": "<=", "hi": ">", "hs": ">="}[cmp_]
            put(f"{self.reg(args[0])} = ({a}) {opx} ({c});"); return
        if name == "selp":
            b = 32 if parts[-1] == "f32" else bits_of(parts[-1]); put(self.wr(args[0], f"{self.reg(args[3])} ? {self.rd(args[1], b)} : {self.rd(args[2], b)}", b)); return
        if parts[-1] == "pred" and name in ("and", "or", "xor", "not"):
            def pv(x): return ("!" if x.startswith("!") else "") + self.reg(x.lstrip("!"))
            if name == "not": put(f"{self.reg(args[0])} = !{pv(args[1])};"); return
            put(f"{self.reg(args[0])} = {pv(args[1])} {'&&' if name == 'and' else '||' if name == 'or' else '!='} {pv(args[2])};"); return
        if name == "mma":
            d = [x.strip() for x in args[0].strip("{}").split(",")]; a = [x.strip() for x in args[1].strip("{}").split(",")]; bb = [x.strip() for x in args[2].strip("{}").split(",")]; c = [x.strip() for x in args[3].strip("{}").split(",")]
            fn = "nr_mma_e4m3" if "e4m3" in op else "nr_mma_f16k16"
            if fn == "nr_mma_f16k16": self.unsupported["mma.f16"] = self.unsupported.get("mma.f16", 0) + 1
            extra = ", nr_tp_scratch" if (fn == "nr_mma_e4m3" and getattr(self, "tp_scratch", False)) else ""
            put(f"{fn}({', '.join(self.rd(x, 32) for x in a)}, {', '.join(self.rd(x, 32) for x in bb)}, {self.rd(c[0], 32)}, {self.rd(c[1], 32)}, {self.reg(d[0])}, {self.reg(d[1])}, lane{extra});"); return
        if name == "shfl":
            d = args[0].split("|"); dst = d[0]; kind = "nr_shfl_bfly" if "bfly" in op else "nr_shfl_idx"
            if "bfly" not in op and "idx" not in op: raise NotImplementedError(s)
            put(self.wr(dst, f"{kind}({self.rd(args[1], 32)}, {self.rd(args[2], 32)})", 32))
            if len(d) > 1: put(f"{self.reg(d[1])} = true;")
            return
        if name == "movmatrix": put(self.wr(args[0], f"nr_movmatrix({self.rd(args[1], 32)}, lane)", 32)); return
        if name == "prmt": put(self.wr(args[0], f"nr_prmt({self.rd(args[1], 32)}, {self.rd(args[2], 32)}, {int(args[3].rstrip('U'), 0)}u)", 32)); return
        if name == "cvt":
            dst_t, src_t = parts[-2], parts[-1]
            if dst_t == "e4m3x2": put(self.wr(args[0], f"nr_cvt_e4m3x2_f16x2({self.rd(args[1], 32)})", 16)); return
            if src_t == "e4m3x2": put(self.wr(args[0], f"nr_cvt_f16x2_e4m3x2({self.rd(args[1], 16)})", 32)); return
            if dst_t == "f16" and src_t == "f32": put(self.wr(args[0], f"nr_b16(half({self.rd_f32(args[1])}))", 16)); return
            if dst_t == "f32" and src_t == "f16": put(self.wr(args[0], f"nr_b32f(float(nr_h({self.rd(args[1], 16)})))", 32)); return
            if dst_t == "f32" and src_t == "f32": put(self.wr(args[0], f"nr_b32f(floor({self.rd_f32(args[1])}))" if "rmi" in parts else f"nr_b32f({self.rd_f32(args[1])})", 32)); return
            if dst_t == "f32" and src_t in ("s32", "u32"): put(self.wr(args[0], f"nr_b32f(float({'int' if src_t == 's32' else 'uint'}({self.rd(args[1], 32)})))", 32)); return
            db, sb = bits_of(dst_t), bits_of(src_t); v = self.rd(args[1], sb if sb != 8 else 16)
            if src_t.startswith("s"): v = sgn(v, sb) if sb > 8 else f"int(char(uchar({v})))"
            if db == 8: v = f"int(char(uchar({v})))" if dst_t.startswith("s") else f"uint(uchar({v}))"
            elif db < sb: v = sgn(f"{ctype(db)}({v})", db) if dst_t.startswith("s") else f"{ctype(db)}({v})"
            put(self.wr(args[0], v, 16 if db == 8 else db)); return
        if name in ("add", "sub", "mul", "fma", "max", "min", "abs", "neg") and (parts[-1] == "f16x2" or "f16x2" in parts):
            srcs = [self.rd_h2(a) for a in args[1:]]
            e = {"add": lambda: f"{srcs[0]} + {srcs[1]}", "sub": lambda: f"{srcs[0]} - {srcs[1]}", "mul": lambda: f"{srcs[0]} * {srcs[1]}", "fma": lambda: f"fma({srcs[0]}, {srcs[1]}, {srcs[2]})",
                 "max": lambda: f"max({srcs[0]}, {srcs[1]})", "min": lambda: f"min({srcs[0]}, {srcs[1]})", "abs": lambda: f"abs({srcs[0]})", "neg": lambda: f"-({srcs[0]})"}[name]()
            put(self.wr(args[0], f"nr_b32(half2({e}))", 32)); return
        if parts[-1] == "f32" and name in ("add", "sub", "mul", "fma", "max", "min", "abs", "neg", "rsqrt", "rcp", "ex2", "div", "sqrt"):
            srcs = [(f"(-{self.rd_f32(a.strip()[1:])})" if a.strip().startswith("-") else self.rd_f32(a)) for a in args[1:]]   # "-%r": negated operand (fp32 contraction sites)
            e = {"add": lambda: f"{srcs[0]} + {srcs[1]}", "sub": lambda: f"{srcs[0]} - {srcs[1]}", "mul": lambda: f"{srcs[0]} * {srcs[1]}", "fma": lambda: f"fma({srcs[0]}, {srcs[1]}, {srcs[2]})",
                 "max": lambda: f"max({srcs[0]}, {srcs[1]})", "min": lambda: f"min({srcs[0]}, {srcs[1]})", "abs": lambda: f"abs({srcs[0]})", "neg": lambda: f"-({srcs[0]})",
                 "rsqrt": lambda: f"nr_rsqrt_approx(rsqrt_table, {srcs[0]})", "rcp": lambda: f"nr_rcp_approx2(rcp_table, rcp_rom, {srcs[0]})", "ex2": lambda: self.ex2_expr(srcs[0]), "div": lambda: f"nr_div_approx({srcs[0]}, {srcs[1]}, rcp_rom)", "sqrt": lambda: f"sqrt({srcs[0]})"}[name]()
            if name in ("ex2", "div"): self.approx[name + ".approx"] = self.approx.get(name + ".approx", 0) + 1
            put(self.wr(args[0], f"nr_b32f({e})", 32)); return
        if name in ("add", "sub", "mul", "mad", "and", "or", "xor", "shl", "shr", "min", "max", "not", "neg", "div", "rem"):
            t = parts[-1]; b = bits_of(t); sg = t.startswith("s")
            if name == "mul" and "wide" in parts:
                a, c = self.rd(args[1], b), self.rd(args[2], b)
                if sg: e = f"long({sgn(a, b)}) * long({sgn(c, b)})"
                else: e = f"ulong({a}) * ulong({c})"
                put(self.wr(args[0], e, 2 * b)); return
            a = self.rd(args[1], b); c = self.rd(args[2], b) if len(args) > 2 else None
            sa = sgn(a, b) if sg else a; sc = sgn(c, b) if (sg and c) else c
            e = {"add": lambda: f"{sa} + {sc}", "sub": lambda: f"{sa} - {sc}", "mul": lambda: f"{sa} * {sc}", "mad": lambda: f"{sa} * {sc} + {sgn(self.rd(args[3], b), b) if sg else self.rd(args[3], b)}",
                 "and": lambda: f"{a} & {c}", "or": lambda: f"{a} | {c}", "xor": lambda: f"{a} ^ {c}", "shl": lambda: f"({a}) << ({self.rd(args[2], 32)} & {b - 1}u)",
                 "shr": lambda: f"({sa}) >> ({self.rd(args[2], 32)} & {b - 1}u)", "min": lambda: f"min({sa}, {sc})", "max": lambda: f"max({sa}, {sc})", "not": lambda: f"~({a})",
                 "neg": lambda: f"-({sa})", "div": lambda: f"({sa}) / ({sc})", "rem": lambda: f"({sa}) % ({sc})"}[name]()
            put(self.wr(args[0], e, b)); return
        if name in ("ld", "st", "red", "atom"):
            shared = any(p.startswith("shared") for p in parts); glob = "global" in parts or (not shared and name in ("ld", "st"))
            b = bits_of(parts[-1]); vec = 4 if ".v4" in op else 2 if ".v2" in op else 1
            if name == "ld":
                addr = self.addr(args[-1], 32 if shared else 64)
                base = f"(threadgroup uchar*)smem + ({addr})" if shared else (f"nr_gptr(pool, wbuf, {addr})" if "global" in parts else f"nr_gptr(pool, wbuf, {addr})")
                names = [x.strip() for x in args[0].strip("{}").split(",")]
                put(f"{{ {'threadgroup' if shared else 'device'} const {ctype(b)}* p_ = ({'threadgroup' if shared else 'device'} const {ctype(b)}*)({base}); " + " ".join(self.wr(nm, f"p_[{i}]", b) for i, nm in enumerate(names)) + " }")
                return
            addr = self.addr(args[0], 32 if shared else 64); base = f"(threadgroup uchar*)smem + ({addr})" if shared else f"nr_gptr(pool, wbuf, {addr})"
            src = args[1].strip()
            if b == 128: vals = [f"{self.reg(src)}.x", f"{self.reg(src)}.y", f"{self.reg(src)}.z", f"{self.reg(src)}.w"]; b = 32
            else: vals = [self.rd(x, b) for x in src.strip("{}").split(",")]
            if name in ("red", "atom"): raise NotImplementedError(s)
            cov = "" if shared or not self.coverage else f" nr_cover(cov, {addr}, {len(vals) * b // 8}u);"
            put(f"{{ {'threadgroup' if shared else 'device'} {ctype(b)}* p_ = ({'threadgroup' if shared else 'device'} {ctype(b)}*)({base}); " + " ".join(f"p_[{i}] = {v};" for i, v in enumerate(vals)) + cov + " }")
            return
        if name == "tex":
            m = re.match(r"\{([^}]*)\}\s*,\s*\[(\S+)\s*,\s*\{([^}]*)\}\]", rest.rstrip(";").strip()); dsts = [x.strip() for x in m.group(1).split(",")]; tref = m.group(2).strip(); xy = [x.strip() for x in m.group(3).split(",")]
            texname = "tex_in"
            if getattr(self, "texref_map", None):
                off = self.texref_param_offset(tref)
                if off is None or off not in self.texref_map: raise NotImplementedError(f"texref {tref} (param offset {off}) has no entry in --texref-map: {s}")
                texname = self.texref_map[off]
            fn = "nr_tex2d" if texname == "tex_in" else "nr_tex2d_mapped"   # mapped textures: sampler variant selected by NR_TEX_MODE at build time (hw / q8 truncate / q8 round)
            put(f"{{ float4 t_ = {fn}({texname}, {self.rd(tref, 64)}, {self.rd_f32(xy[0])}, {self.rd_f32(xy[1])}); " + " ".join(self.wr(d, f"nr_b32f(t_[{i}])", 32) for i, d in enumerate(dsts)) + " }"); return
        if name == "sust":
            m = re.match(r"\[(\S+)\s*,\s*\{([^}]*)\}\]\s*,\s*\{([^}]*)\}", rest.rstrip(";").strip()); sref = m.group(1).strip(); xy = [x.strip() for x in m.group(2).split(",")]; vals = [x.strip() for x in m.group(3).split(",")]
            put(f"nr_sust2d(tex_out, texcov, {self.rd(sref, 64)}, {self.rd(xy[0], 32)}, {self.rd(xy[1], 32)}, float4({', '.join(self.rd_f32(v) for v in vals)}));"); return
        if name == "suld":
            self.unsupported[name] = self.unsupported.get(name, 0) + 1; put(f"/* UNSUPPORTED: {s} */"); return
        raise NotImplementedError(s)
    def ex2_expr(self, x):
        """ex2.approx: the sm120 range-reduction model, or (--ex2-half-table) the audited half-domain table indexed by the ORIGINAL half operand found by static
        def-use (unique cvt.f32.f16 -> unique mul by 0fBFB8AA3B); the lookup checks the exact fp32 argument and counts faults (fail closed). A site without a
        resolvable half source under --ex2-half-table is a generation error."""
        if not getattr(self, "ex2_half", False): return f"nr_ex2_sm120({x}, ex2_unit, ex2_frac)"
        site = self.ex2_sites.get(self.cur_index)
        if site is None: raise NotImplementedError(f"--ex2-half-table: statement {self.cur_index} has no unique half source: {self.ins[self.cur_index]}")
        self.ex2_half_used += 1; return f"nr_ex2_half_table(ex2_half, fault, ushort({self.rd(site, 16)}), {x})"
    def texref_param_offset(self, reg):
        """the ld.param offset that defines texref register `reg` (static; fails closed when a register has several different offsets or none)"""
        offs = set()
        for s in self.ins:
            m = re.match(r"\s*ld\.param\.b64\s+(%rd\d+)\s*,\s*\[([^\]]*)\]", s.strip())
            if m and m.group(1) == reg:
                src = m.group(2); mm = re.search(r"\+(\d+)\s*$", src); offs.add(int(mm.group(1)) if mm else 0)
        return next(iter(offs)) if len(offs) == 1 else None
    # ---- whole kernel ----
    def translate(self, msl_name):
        ins = self.ins; n = len(ins)
        starts = {0} | set(self.k.labels.values())
        for i, s in enumerate(ins):
            t = s.strip()
            body = t.split(" ", 1)[1] if t.startswith("@") and " " in t else t
            while body.startswith("{"): body = body[1:].strip()
            if t.startswith("@") or t.startswith("bra") or t.startswith("ret") or body.startswith("bar."): starts.add(i + 1)
        starts = sorted(x for x in starts if x < n); blocks = list(zip(starts, starts[1:] + [n])); self.block_of = {st: b for b, (st, en) in enumerate(blocks)}
        cases = []
        for b, (st, en) in enumerate(blocks):
            stmts = []; has_sync = False
            for i in range(st, en):
                t = ins[i].strip(); body = t.split(" ", 1)[1] if t.startswith("@") and " " in t else t
                while body.startswith("{"): body = body[1:].strip()
                if body.split(".")[0].split(" ")[0] in SYNC_OPS: has_sync = True
                if t.startswith("@") and (" bra" in t and body.startswith("bra")):
                    p, rest = t.split(" ", 1); neg = p.startswith("@!"); pr = ("!" if neg else "") + self.reg(p.lstrip("@!")); target = self.block_of[self.k.labels[rest.split()[1]]]
                    stmts.append(f"pc = ({pr}) ? {target}u : {b + 1}u; break;"); continue
                self.cur_index = i; self.emit(t, stmts)
                for reg, slot in self.watches.get(i, []):
                    cx, cy, cz = self.watch_cta
                    stmts.append(f"if (ctaid.x == {cx}u && ctaid.y == {cy}u && ctaid.z == {cz}u && warp == {self.watch_warp}u) dbg[{slot}u * 32u + lane] = uint({self.reg(reg)});")
            head = "        if (nr_active_lanes() != 32u) break;   // warp-synchronous block: wait for all lanes\n" if has_sync else ""
            tail = "" if stmts and (stmts[-1].endswith("break;") and not stmts[-1].startswith("if (")) else f"        pc = {b + 1}u; break;\n"
            cases.append(f"      case {b}u: {{\n{head}" + "".join("        " + x + "\n" for x in stmts) + tail + "      }\n")
        decls = "\n".join(f"  {ctype(w)} {('R_' + nm[1:]) if nm.startswith('%') else nm} = {ctype(w)}(0);" if w != 1 else f"  bool {nm} = false;" for nm, w in sorted(self.regs.items()))
        decls += "\n" + "\n".join(f"  bool P_{p[2:]} = false;" for p in sorted(self.preds))
        shared_words = max(1, (self.shared_size + 15) // 16)
        tp_warps = 1
        if getattr(self, "tp_scratch", False) and getattr(self, "tp_warps", 1) > 1:
            budget = 30 * 1024 - shared_words * 16 - 256          # threadgroup memory left for scratches (32 KB limit with margin)
            tp_warps = max(1, min(int(getattr(self, "tp_warps", 1)), budget // 4608))   # sizeof(NrTpScratch) (slim) ~4.5 KB
        tp_decl = f"\n  threadgroup NrTpScratch nr_tp_scratch_arr[{tp_warps}]; threadgroup NrTpScratch& nr_tp_scratch = nr_tp_scratch_arr[warp % {tp_warps}u];" if getattr(self, "tp_scratch", False) else ""
        tp_init = f" if (tlin < {tp_warps}u) nr_tp_scratch_arr[tlin].enabled = (nthreads <= 32u * {tp_warps}u);   // one scratch per warp; kernels with more warps than scratches take the scalar rule" if getattr(self, "tp_scratch", False) else ""
        extra_tex = ""; extra_buf = "\n                       device const uint2* ex2_half [[buffer(11)]], device atomic_uint* fault [[buffer(12)]]," if getattr(self, "ex2_half", False) else ""
        if getattr(self, "texref_map", None):   # experimental temporal routing: extra sampled textures, indices 2.. in map order (tex_in stays texture(0))
            names = [n for n in getattr(self, "texref_order", []) if n != "tex_in"]
            extra_tex = "".join(f"\n                       texture2d<float, access::sample> {n} [[texture({2 + i})]]," for i, n in enumerate(names))
        sw = bool(getattr(self, "single_warp", False))
        if sw:   # opt-in single-SIMD-group dispatcher: eligibility proven from the PARSED instruction list, fail closed
            found = []
            for ins in self.ins:
                t = str(ins).strip()
                while t.startswith("@"):
                    t = t.split(None, 1)[1].strip() if " " in t else ""
                while t.startswith("{"): t = t[1:].strip()
                op = t.split()[0].split(".")[0] if t.split() else ""
                if op in ("bar", "barrier", "membar", "fence"): found.append(t[:60])
            if found: raise SystemExit("--single-warp refused: %d synchronisation instruction(s) in the parsed kernel, e.g. %r" % (len(found), found[0]))
            if "cg2r_copy" in str(self.entry) or "copy" in msl_name: raise SystemExit("--single-warp refused: the copy kernel (launch 156) is excluded by review")
            if getattr(self, "shared_size", 0) > 0: raise SystemExit("--single-warp refused: kernel uses %d bytes of emulated shared memory" % self.shared_size)
        sw_decl = "" if sw else "\n  threadgroup atomic_uint bar_count, exit_count;"
        # NOTE (v2): this text is followed on the SAME line by {tp_init}, so it MUST NOT end inside a // comment -
        # in v1 it did, which commented out the threadgroup scratch initialisation. It now ends with a newline + indent.
        sw_init = ("// single-SIMD-group dispatcher (opt-in): no CTA-barrier protocol. HOST CONTRACT: reject any dispatch that is not\n  // 32x1x1 with pipeline threadExecutionWidth 32 before encoding (docs/NR_LAUNCH155_CENSUS.md).\n "
                   if sw else "if (tlin == 0u) { atomic_store_explicit(&bar_count, 0u, memory_order_relaxed); atomic_store_explicit(&exit_count, 0u, memory_order_relaxed); }")
        sw_barwait = "" if sw else ", bar_wait = false"
        sw_default = "" if sw else " atomic_fetch_add_explicit(&exit_count, 1u, memory_order_relaxed);"
        # v2: ALL THREE dispatcher fences are retained at their original sites with their original flags;
        # only the shared-counter EXIT BOOKKEEPING is normalised (simd_all(done) replaces exit_count == nthreads).
        # The bar_count/release/bar_wait arm is absent because eligibility above proves the kernel has no bar.sync arrivals.
        sw_loop = ("    simdgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);   // fence 1 of 3 - original site, original flags\n"
                   "    const bool all_exited = simd_all(done);                                  // exit bookkeeping only: uniform vote replaces exit_count == nthreads\n"
                   "    simdgroup_barrier(mem_flags::mem_threadgroup);                           // fence 2 of 3 - original site, original flags\n"
                   "    if (all_exited) break;\n"
                   "    simdgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);   // fence 3 of 3 - original site, original flags\n"
                   "    if (done) continue;"
                   if sw else
                   "    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);\n"
                   "    const uint waiting = atomic_load_explicit(&bar_count, memory_order_relaxed), exited = atomic_load_explicit(&exit_count, memory_order_relaxed);\n"
                   "    const bool all_exited = exited == nthreads, release = (waiting + exited == nthreads) && waiting > 0u;\n"
                   "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
                   "    if (all_exited) break;\n"
                   "    if (release) { if (tlin == 0u) atomic_store_explicit(&bar_count, 0u, memory_order_relaxed); bar_wait = false; }\n"
                   "    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);\n"
                   "    if (done || bar_wait) continue;")
        return f"""// Generated by scripts/ptx_warp_to_msl.py from {self.entry}. Do not edit by hand.
#include "nr_ptx_rt.metal"
kernel void {msl_name}(device uchar* pool [[buffer(0)]], device uchar* wbuf [[buffer(1)]], constant uchar* prm [[buffer(2)]],
                       device const ushort* rsqrt_table [[buffer(3)]], device const ushort* rcp_table [[buffer(4)]], device uint* dbg [[buffer(5)]], device atomic_uint* cov [[buffer(6)]],
                       device const uchar* rcp_rom [[buffer(7)]], device const uchar* ex2_unit [[buffer(8)]], device const uchar* ex2_frac [[buffer(9)]], device atomic_uint* texcov [[buffer(10)]],{extra_buf}
                       texture2d<float, access::sample> tex_in [[texture(0)]], texture2d<half, access::write> tex_out [[texture(1)]],{extra_tex}
                       uint3 tid [[thread_position_in_threadgroup]], uint3 ctaid [[threadgroup_position_in_grid]],
                       uint3 ntid [[threads_per_threadgroup]], uint3 nctaid [[threadgroups_per_grid]],
                       uint lane [[thread_index_in_simdgroup]], uint warp [[simdgroup_index_in_threadgroup]], uint tlin [[thread_index_in_threadgroup]]) {{
  threadgroup uint4 smem[{shared_words}];{tp_decl}{sw_decl}
  const uint nthreads = ntid.x * ntid.y * ntid.z;
  {sw_init}{tp_init}
  threadgroup_barrier(mem_flags::mem_threadgroup);
{decls}
  uint pc = 0u, pc_next = 0u; bool done = false{sw_barwait};
  for (uint iter_ = 0u; iter_ < 4000000u; ++iter_) {{
{sw_loop}
    pc_next = pc + 1u;
    switch (pc) {{
{''.join(cases)}      default: done = true;{sw_default} break;
    }}
  }}
}}
"""

def main():
    p = argparse.ArgumentParser(); p.add_argument("--ptx", required=True); p.add_argument("--entry", required=True); p.add_argument("--name", required=True); p.add_argument("--output", required=True)
    p.add_argument("--ex2-half-table", action="store_true", help="experimental temporal builds: ex2.approx sites whose operand is a half promoted and scaled by 0fBFB8AA3B take the audited half-domain table (buffer 11) with an exact argument check and fault counters (buffer 12); other ex2 sites are a generation error"); p.add_argument("--texref-map", default="", help="experimental temporal routing (v1): PARAM_OFFSET:TEXTURE_NAME,... selects the Metal texture for each tex.2d site by the ld.param offset its texref register was loaded from (e.g. 56:tex_in,96:tex_motion,88:tex_history); off by default (all sites use tex_in)"); p.add_argument("--tp-warps", type=int, default=1, help="max warps per threadgroup given their own NrTpScratch (limited by the 32 KB threadgroup budget after emulated shared memory)"); p.add_argument("--single-warp", action="store_true", help="EXPERIMENTAL opt-in (docs/NR_LAUNCH155_CENSUS.md): for a kernel PROVEN free of synchronisation instructions, emit the single-SIMD-group dispatcher (simdgroup fence + uniform simd_all(done) vote) instead of the CTA-barrier protocol; refused otherwise and never for the copy kernel. The HOST must reject any dispatch that is not 32x1x1 with pipeline threadExecutionWidth 32."); p.add_argument("--tp-scratch", action="store_true", help="declare a kernel-scope threadgroup NrTpScratch and pass it to nr_mma_e4m3 (three-product QMMA path, NR_QMMA_TP)"); p.add_argument("--fuse-sites-file", default="", help="contraction-sites JSON (spec/nr/contraction/*.json): contract exactly these add.f16x2 statements with the recorded operand choice"); p.add_argument("--watch", action="append", default=[], help="STMT:REG (statement index as in the emulator); values land in the dbg buffer"); p.add_argument("--watch-cta", default="0,0,0"); p.add_argument("--watch-warp", type=int, default=0); a = p.parse_args()
    if a.fuse_sites_file:
        import json as _json; _spec = _json.load(open(a.fuse_sites_file)); os.environ["EMU_FUSE_SITES"] = ",".join(f"{x['statement']}:{x['fused']}" for x in _spec["sites"])
    e = Emitter(open(a.ptx).read(), a.entry, watches=a.watch, watch_cta=tuple(int(v) for v in a.watch_cta.split(",")), watch_warp=a.watch_warp); e.tp_scratch = bool(a.tp_scratch); e.single_warp = bool(a.single_warp)
    if a.texref_map:
        e.texref_map = {}; e.texref_order = []
        for item in a.texref_map.split(","):
            off, name = item.split(":"); e.texref_map[int(off)] = name; e.texref_order.append(name)
    e.tp_warps = a.tp_warps; e.ex2_half = bool(a.ex2_half_table); e.ex2_sites = ex2_half_sites(e.k) if e.ex2_half else {}; e.ex2_half_used = 0; code = e.translate(a.name)
    if e.ex2_half: print(f"ex2 half-table sites: {e.ex2_half_used} of {sum(1 for s in e.ins if s.strip().startswith('ex2.'))} ex2 statements")
    open(a.output, "w").write(code); print(f"wrote {a.output}: {len(e.ins)} statements, {len(e.regs)} registers, shared {e.shared_size} bytes, unsupported {e.unsupported}, approximate {e.approx}")
if __name__ == "__main__": main()
