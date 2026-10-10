#!/usr/bin/env python3
"""Minimal PTX interpreter for the integer/address subset used by NR's repack and copy kernels.

Executes one thread at a time (no shared memory, no barriers, no floating point). Loads read from a
dict of {base_address: bytearray}; stores are recorded into an output bytearray keyed the same way.
Everything unsupported raises, so a silent misinterpretation is impossible.
"""
import re, struct

MASK = {8: 0xff, 16: 0xffff, 32: 0xffffffff, 64: 0xffffffffffffffff}

def to_signed(v, bits):
    v &= MASK[bits]
    return v - (1 << bits) if v & (1 << (bits - 1)) else v

class Kernel:
    def __init__(self, ptx_text, entry):
        m = re.search(r"\.visible \.entry %s\((.*?)\)\s*(\.maxnreg \d+\s*)?\{(.*)" % re.escape(entry), ptx_text, re.S)
        if not m: raise ValueError("entry not found: " + entry)
        body = m.group(3)
        # cut at the matching closing brace of the entry body
        depth = 1; end = 0
        for i, ch in enumerate(body):
            if ch == "{": depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0: end = i; break
        body = body[:end]
        self.param_name = re.search(r"\.param\s+(?:\.align\s+\d+\s+)?\.b8\s+(\w+)\[(\d+)\]", m.group(1)).group(1)
        # shared-memory symbols: laid out sequentially with their declared alignment
        self.shared_syms = {}; off = 0
        for align, name, size in re.findall(r"\.shared\s+\.align\s+(\d+)\s+\.b8\s+(\w+)\[(\d+)\]", body):
            a = int(align); off = (off + a - 1) // a * a; self.shared_syms[name] = off; off += int(size)
        self.shared_bytes = off
        self.instrs = []; self.labels = {}
        pending = ""
        def flush():
            nonlocal pending
            stmt = pending.rstrip(";").strip(); pending = ""
            if stmt: self.instrs.append(stmt)
        for raw in body.split("\n"):
            line = raw.strip()
            if not line or line.startswith("//"): continue
            # braces on their own lines (or glued to the front of a statement) are block boundaries when no
            # statement is pending; inside a pending multi-line statement they are operand-list braces.
            if not pending:
                while line.startswith(("{", "}")):
                    line = line[1:].strip()
                if line.endswith("}") and not line.startswith(".reg") and not line.endswith("};"):
                    line = line[:-1].strip()
            while not pending and line.startswith("$"):          # one or more labels, possibly followed by a statement
                lab, _, rest = line.partition(":")
                self.labels[lab] = len(self.instrs); line = rest.strip()
            if not line: continue
            if not pending and line.startswith(".reg") and ";" in line:
                # single-line compound block: "{ .reg .b128 v; mov.b128 v, {...}; st.global... [%rd], v; }"
                for piece in line.split(";"):
                    piece = piece.strip()
                    while piece.startswith("{") and not piece.startswith("{%"): piece = piece[1:].strip()
                    while piece.endswith("}") and piece.count("}") > piece.count("{"): piece = piece[:-1].strip()
                    if piece and not piece.startswith(".reg"): self.instrs.append(piece)
                continue
            if not pending and line.startswith((".reg", ".shared", ".local", ".const")): continue
            pending = (pending + " " + line).strip() if pending else line
            if pending.endswith(";"): flush()
        flush()

    def run_thread(self, params: bytes, ctaid, ntid, tid, nctaid, mem, out):
        R = {}; P = {}
        sregs = {"%ctaid.x": ctaid[0], "%ctaid.y": ctaid[1], "%ctaid.z": ctaid[2], "%ntid.x": ntid[0], "%ntid.y": ntid[1], "%ntid.z": ntid[2],
                 "%tid.x": tid[0], "%tid.y": tid[1], "%tid.z": tid[2], "%nctaid.x": nctaid[0], "%nctaid.y": nctaid[1], "%nctaid.z": nctaid[2]}
        def rd(x, bits):
            x = x.strip()
            if x in sregs: return sregs[x] & MASK[bits]
            if x.startswith("%"): return R.get(x, 0) & MASK[bits]
            if x.startswith("0x"): return int(x, 16) & MASK[bits]
            if x.startswith("0f"): return int(x[2:], 16) & MASK[bits]
            return int(x) & MASK[bits]
        def bits_of(t):
            return int(re.search(r"(8|16|32|64)$", t).group(1))
        def load(addr, bits):
            for base, buf in mem.items():
                if base <= addr < base + len(buf):
                    o = addr - base; return int.from_bytes(buf[o:o + bits // 8], "little")
            raise KeyError("load outside mapped memory: 0x%x" % addr)
        def store(addr, bits, val):
            for base, buf in out.items():
                if base <= addr < base + len(buf):
                    o = addr - base; buf[o:o + bits // 8] = (val & MASK[bits]).to_bytes(bits // 8, "little"); return
            raise KeyError("store outside mapped memory: 0x%x" % addr)
        pc = 0; steps = 0
        while pc < len(self.instrs):
            ins = self.instrs[pc]; pc += 1; steps += 1
            if steps > 200000: raise RuntimeError("runaway thread")
            pred = None
            if ins.startswith("@"):
                pred, ins = ins.split(" ", 1); neg = pred.startswith("@!"); pv = P[pred.lstrip("@!")]
                if pv == neg: continue
            op, _, rest = ins.partition(" ")
            args = [a.strip() for a in re.split(r",(?![^{]*\})", rest)] if rest else []
            parts = op.split(".")
            name = parts[0]
            if name == "ret": return
            if name == "bra": pc = self.labels[args[0]]; continue
            if name == "ld" and "param" in parts:
                bits = bits_of(parts[-1]); m = re.match(r"\[%s(?:\+(\d+))?\]" % self.param_name, args[-1]); off = int(m.group(1) or 0)
                if ".v2" in op:
                    names = [n.strip() for n in args[0].strip("{}").split(",")]
                    for i, n in enumerate(names): R[n] = int.from_bytes(params[off + i * bits // 8: off + (i + 1) * bits // 8], "little")
                else: R[args[0]] = int.from_bytes(params[off:off + bits // 8], "little")
                continue
            if name == "mov":
                bits = bits_of(parts[-1]); R[args[0]] = rd(args[1], bits); continue
            if name == "cvta": R[args[0]] = rd(args[1], 64); continue
            if name in ("add", "sub", "mul", "mad", "and", "or", "xor", "shl", "shr", "div", "rem", "min", "max"):
                t = parts[-1]; bits = bits_of(t); signed = t.startswith("s")
                if name == "mul" and parts[1] == "wide":
                    # the type names the source width; the destination is twice as wide
                    a, b = rd(args[1], bits), rd(args[2], bits)
                    if signed: a, b = to_signed(a, bits), to_signed(b, bits)
                    R[args[0]] = (a * b) & MASK[bits * 2]; continue
                a = rd(args[1], bits); b = rd(args[2], bits)
                if signed: a, b = to_signed(a, bits), to_signed(b, bits)
                if name == "add": v = a + b
                elif name == "sub": v = a - b
                elif name == "mul": v = a * b            # .lo
                elif name == "mad": v = a * b + (to_signed(rd(args[3], bits), bits) if signed else rd(args[3], bits))
                elif name == "and": v = a & b
                elif name == "or": v = a | b
                elif name == "xor": v = a ^ b
                elif name == "shl": v = (a << min(b & 0xffff, bits)) if (b & 0xffff) < bits else 0
                elif name == "shr":
                    sh = rd(args[2], 32)
                    if signed: v = a >> min(sh, bits - 1)
                    else: v = (a & MASK[bits]) >> sh if sh < bits else 0
                elif name == "div":
                    if b == 0: raise ZeroDivisionError
                    v = abs(a) // abs(b); v = -v if (a < 0) != (b < 0) else v
                elif name == "rem":
                    q = abs(a) // abs(b); q = -q if (a < 0) != (b < 0) else q; v = a - q * b
                elif name == "min": v = min(a, b)
                elif name == "max": v = max(a, b)
                R[args[0]] = v & MASK[bits]; continue
            if name == "cvt":
                dst_t, src_t = parts[-2], parts[-1]
                if not re.match(r"[usb](8|16|32|64)$", dst_t) or not re.match(r"[usb](8|16|32|64)$", src_t):
                    raise NotImplementedError(ins)
                db, sb = bits_of(dst_t), bits_of(src_t); v = rd(args[1], sb)
                if src_t.startswith("s"): v = to_signed(v, sb)
                if db < sb:
                    v &= MASK[db]                                        # truncate
                    if dst_t.startswith("s"): v = to_signed(v, db)
                # destination register width: 8-bit types live in 16-bit registers, sign/zero-extended per dst type
                regbits = 16 if db == 8 else db
                R[args[0]] = v & MASK[regbits]; continue
            if name == "setp":
                cmp = parts[1]; t = parts[-1]; bits = bits_of(t); signed = t.startswith("s")
                a, b = rd(args[1], bits), rd(args[2], bits)
                if signed: a, b = to_signed(a, bits), to_signed(b, bits)
                P[args[0]] = {"eq": a == b, "ne": a != b, "lt": a < b, "le": a <= b, "gt": a > b, "ge": a >= b,
                              "lo": a < b, "ls": a <= b, "hi": a > b, "hs": a >= b}[cmp]; continue
            if name == "selp":
                bits = bits_of(parts[-1]); R[args[0]] = rd(args[1], bits) if P[args[3]] else rd(args[2], bits); continue
            if name == "ld":
                bits = bits_of(parts[-1]); addr = rd(args[1].strip("[]"), 64)
                if "global" not in parts: raise NotImplementedError(ins)
                R[args[0]] = load(addr, bits); continue
            if name == "st":
                bits = bits_of(parts[-1]); addr = rd(args[0].strip("[]"), 64)
                if "global" not in parts: raise NotImplementedError(ins)
                store(addr, bits, rd(args[1], bits)); continue
            raise NotImplementedError(ins)

def run_grid(kernel, params, grid, block, mem, out, thread_filter=None):
    n = 0
    for bz in range(grid[2]):
        for by in range(grid[1]):
            for bx in range(grid[0]):
                for tz in range(block[2]):
                    for ty in range(block[1]):
                        for tx in range(block[0]):
                            if thread_filter and not thread_filter(bx, by, bz, tx, ty, tz): continue
                            kernel.run_thread(params, (bx, by, bz), block, (tx, ty, tz), grid, mem, out); n += 1
    return n
