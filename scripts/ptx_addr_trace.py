#!/usr/bin/env python3
"""Address tracer for NR PTX kernels: executes the integer/address subset per thread and records every
global/shared memory access and bulk copy, while floating-point, MMA and shuffle ops are stubbed (their
destination registers become 0). The recovered addresses define the operand layouts; the arithmetic is
read separately from the PTX text. Nothing here is a numerical reference."""
import re
from ptx_int_emu import Kernel, MASK, to_signed

FLOAT_TYPES = ("f16", "f16x2", "f32", "f64", "bf16", "e4m3x2", "e5m2x2")

class Tracer(Kernel):
    def run_thread(self, params, ctaid, ntid, tid, nctaid, mem, events, shared_size=1 << 20):
        R = {}; P = {}
        lane = tid[0] % 32
        sregs = {"%ctaid.x": ctaid[0], "%ctaid.y": ctaid[1], "%ctaid.z": ctaid[2], "%ntid.x": ntid[0], "%ntid.y": ntid[1], "%ntid.z": ntid[2],
                 "%tid.x": tid[0], "%tid.y": tid[1], "%tid.z": tid[2], "%nctaid.x": nctaid[0], "%nctaid.y": nctaid[1], "%nctaid.z": nctaid[2],
                 "%laneid": lane, "%warpid": (tid[0] + ntid[0] * (tid[1] + ntid[1] * tid[2])) // 32}
        def rd(x, bits):
            x = x.strip()
            if x in sregs: return sregs[x] & MASK[bits]
            if x.startswith("%"): return R.get(x, 0) & MASK[bits]
            if x.startswith("0x"): return int(x, 16) & MASK[bits]
            if x.startswith("0f") or x.startswith("0d"): return int(x[2:], 16) & MASK[bits]
            if x[0] == "_": return self.shared_syms.get(x, 0)   # shared-memory symbol -> its offset
            if x[0].isalpha(): return 0
            return int(x) & MASK[bits]
        def bits_of(t):
            m = re.search(r"(8|16|32|64|128)$", t); return int(m.group(1)) if m else 32
        def load(addr, bits):
            for base, buf in mem.items():
                if base <= addr < base + len(buf):
                    o = addr - base; return int.from_bytes(buf[o:o + bits // 8], "little")
            return 0
        pc = 0; steps = 0; self._recent = []
        while pc < len(self.instrs):
            ins = self.instrs[pc]; pc += 1; steps += 1
            if steps > 400000:
                import collections; hist = collections.Counter(self._recent)
                raise RuntimeError("runaway thread; hot statements:\n" + "\n".join(f"{n:6d}x  {self.instrs[i]}" for i, n in hist.most_common(12)))
            if steps > 399000: self._recent.append(pc - 1)
            if ins.startswith("@"):
                pred, ins = ins.split(" ", 1); neg = pred.startswith("@!"); pv = P.get(pred.lstrip("@!"), False)
                if pv == neg: continue
            ins = ins.strip("{} ").strip()
            if not ins: continue
            # inline braces blocks like "{ .reg .b128 v; mov.b128 v, {...}; st.global...b128 [%rd], v; }"
            if ins.startswith(".reg .b128") or ins.startswith(".reg .pred") or ins.startswith(".reg .f16"):
                # split compound statement
                stmts = [s.strip() for s in ins.split(";") if s.strip()]
                for s in stmts:
                    if s.startswith(".reg"): continue
                    self._exec(s, R, P, rd, bits_of, load, mem, events, ctaid, tid, lane)
                continue
            r = self._exec(ins, R, P, rd, bits_of, load, mem, events, ctaid, tid, lane)
            if r == "ret": return
            if isinstance(r, tuple) and r[0] == "bra": pc = self.labels[r[1]]

    def _exec(self, ins, R, P, rd, bits_of, load, mem, events, ctaid, tid, lane):
        op, _, rest = ins.partition(" ")
        args = [a.strip() for a in re.split(r",(?![^{]*\})", rest)] if rest else []
        parts = op.split("."); name = parts[0]
        if name == "ret": return "ret"
        if name == "bra": return ("bra", args[0])
        if name in ("bar", "barrier", "mbarrier", "nanosleep", "fence", "membar", "cvta"):
            if name == "cvta": R[args[0]] = rd(args[1], 64)
            if name == "mbarrier" and ("try_wait" in op or "test_wait" in op): P[args[0]] = True   # the copy has landed
            return
        if name == "ld" and "param" in parts:
            bits = bits_of(parts[-1]); m = re.match(r"\[%s(?:\+(\d+))?\]" % self.param_name, args[-1]); off = int(m.group(1) or 0)
            params = self._params
            if ".v2" in op or ".v4" in op:
                names = [n.strip() for n in args[0].strip("{}").split(",")]
                for i, n in enumerate(names): R[n] = int.from_bytes(params[off + i * bits // 8: off + (i + 1) * bits // 8], "little")
            else: R[args[0]] = int.from_bytes(params[off:off + bits // 8], "little")
            return
        if name == "mov":
            t = parts[-1]
            if args[1].startswith("{"):  # pack: unknown float content -> 0 (but preserve 64-bit packing of ints if simple)
                R[args[0]] = 0; return
            if args[0].startswith("{"):  # unpack {lo,hi} <- reg
                v = rd(args[1], 32); names = [n.strip() for n in args[0].strip("{}").split(",")]
                R[names[0]] = v & 0xffff; R[names[1]] = (v >> 16) & 0xffff; return
            R[args[0]] = rd(args[1], bits_of(t)); return
        if name == "elect":
            P["P_OUT"] = lane == 0; return
        if name == "selp":
            bits = bits_of(parts[-1]); R[args[0]] = rd(args[1], bits) if P.get(args[3], False) else rd(args[2], bits); return
        if name == "setp":
            t = parts[-1]
            if t in FLOAT_TYPES: P[args[0].split("|")[0]] = False; return
            cmp = parts[1]; bits = bits_of(t); signed = t.startswith("s")
            a, b = rd(args[1], bits), rd(args[2], bits)
            if signed: a, b = to_signed(a, bits), to_signed(b, bits)
            v = {"eq": a == b, "ne": a != b, "lt": a < b, "le": a <= b, "gt": a > b, "ge": a >= b, "lo": a < b, "ls": a <= b, "hi": a > b, "hs": a >= b}[cmp]
            if len(args) > 3:  # setp with .and/.or predicate combination
                q = P.get(args[3].lstrip("!"), False); q = (not q) if args[3].startswith("!") else q
                comb = parts[2] if len(parts) > 3 else "and"
                v = (v and q) if comb == "and" else (v or q)
            P[args[0]] = v; return
        if name in ("and", "or", "xor", "not") and parts[-1] == "pred":
            if name == "not": P[args[0]] = not P.get(args[1], False); return
            a = P.get(args[1].lstrip("!"), False); b = P.get(args[2].lstrip("!"), False)
            if args[1].startswith("!"): a = not a
            if args[2].startswith("!"): b = not b
            P[args[0]] = (a and b) if name == "and" else (a or b) if name == "or" else (a != b); return
        if parts[-1] in FLOAT_TYPES or name in ("mma", "shfl", "prmt", "rsqrt", "rcp", "ex2", "lg2", "sqrt", "abs", "neg", "fma") and not parts[-1].startswith(("s", "u", "b")):
            # floating point / warp ops: stub destination
            if args: R[args[0].split("|")[0]] = 0
            return
        if name in ("shfl", "prmt", "mma", "bfe", "bfi", "popc", "clz", "brev", "movmatrix", "ldmatrix", "stmatrix"):
            if name == "bfe":
                bits = bits_of(parts[-1]); a = rd(args[1], bits); b = rd(args[2], 32); c = rd(args[3], 32)
                v = (a >> b) & ((1 << c) - 1) if c < bits else a >> b
                if parts[-1].startswith("s") and c and (v >> (c - 1)) & 1: v -= 1 << c
                R[args[0]] = v & MASK[bits]; return
            if args: R[args[0].split("|")[0]] = 0
            return
        if name == "not":
            bits = bits_of(parts[-1]); R[args[0]] = (~rd(args[1], bits)) & MASK[bits]; return
        if name in ("add", "sub", "mul", "mad", "and", "or", "xor", "shl", "shr", "div", "rem", "min", "max", "neg", "abs"):
            t = parts[-1]; bits = bits_of(t); signed = t.startswith("s")
            if name == "mul" and len(parts) > 2 and parts[1] == "wide":
                a, b = rd(args[1], bits), rd(args[2], bits)
                if signed: a, b = to_signed(a, bits), to_signed(b, bits)
                R[args[0]] = (a * b) & MASK[bits * 2]; return
            if name == "mad" and len(parts) > 2 and parts[1] == "wide":
                a, b = rd(args[1], bits), rd(args[2], bits); c = rd(args[3], bits * 2)
                if signed: a, b, c = to_signed(a, bits), to_signed(b, bits), to_signed(c, bits * 2)
                R[args[0]] = (a * b + c) & MASK[bits * 2]; return
            if name in ("neg", "abs"):
                a = to_signed(rd(args[1], bits), bits); R[args[0]] = (-a if name == "neg" else abs(a)) & MASK[bits]; return
            a = rd(args[1], bits); b = rd(args[2], bits)
            if signed: a, b = to_signed(a, bits), to_signed(b, bits)
            if name == "add": v = a + b
            elif name == "sub": v = a - b
            elif name == "mul":
                v = a * b
                if len(parts) > 2 and parts[1] == "hi": v = v >> bits
            elif name == "mad":
                c = rd(args[3], bits); c = to_signed(c, bits) if signed else c
                v = a * b
                if len(parts) > 2 and parts[1] == "hi": v = v >> bits
                v += c
            elif name == "and": v = a & b
            elif name == "or": v = a | b
            elif name == "xor": v = a ^ b
            elif name == "shl": sh = rd(args[2], 32); v = (a << sh) if sh < bits else 0
            elif name == "shr":
                sh = rd(args[2], 32)
                v = (a >> min(sh, bits - 1)) if signed else ((a & MASK[bits]) >> sh if sh < bits else 0)
            elif name == "div": v = 0 if b == 0 else (abs(a) // abs(b)) * (-1 if (a < 0) != (b < 0) else 1)
            elif name == "rem":
                if b == 0: v = 0
                else: q = (abs(a) // abs(b)) * (-1 if (a < 0) != (b < 0) else 1); v = a - q * b
            elif name == "min": v = min(a, b)
            elif name == "max": v = max(a, b)
            R[args[0]] = v & MASK[bits]; return
        if name == "cvt":
            dst_t, src_t = parts[-2], parts[-1]
            if dst_t in FLOAT_TYPES or src_t in FLOAT_TYPES or not re.match(r"[usb](8|16|32|64)$", dst_t) or not re.match(r"[usb](8|16|32|64)$", src_t):
                R[args[0]] = 0; return
            db, sb = bits_of(dst_t), bits_of(src_t); v = rd(args[1], sb)
            if src_t.startswith("s"): v = to_signed(v, sb)
            if db < sb:
                v &= MASK[db]
                if dst_t.startswith("s"): v = to_signed(v, db)
            R[args[0]] = v & MASK[16 if db == 8 else db]; return
        if name == "cp" and "bulk" not in parts:   # cp.async.ca.shared.global [dst], [src], n  |  commit_group / wait_group
            if parts[2] in ("commit_group", "wait_group", "wait_all"): return
            dst = rd(args[0].strip("[]"), 32); src = rd(args[1].strip("[]"), 64); nbytes = int(args[2], 0)
            events.append(("cp.async", ctaid, tid, src, dst, nbytes)); return
        if name == "cp":  # cp.async.bulk.shared::cta.global.mbarrier... [dst_shared], [src_global], bytes, [mbar]
            dst = rd(args[0].strip("[]"), 32); src = rd(args[1].strip("[]"), 64); nbytes = rd(args[2], 32)
            events.append(("cp.async", ctaid, tid, src, dst, nbytes)); return
        if ins.lstrip().startswith(".pragma"): return
        if name in ("ld", "st", "red", "atom"):
            space = "shared" if any(p.startswith("shared") for p in parts) else "global" if "global" in parts else "?"
            bits = bits_of(parts[-1])
            vec = 4 if ".v4" in op else 2 if ".v2" in op else 1
            if name == "ld":
                inner = args[-1].strip("[]"); addr = (rd(inner.split("+")[0], 64 if space == "global" else 32) + (int(inner.split("+")[1], 0) if "+" in inner else 0))
                events.append(("ld." + space, ctaid, tid, addr, bits * vec // 8))
                dests = [n.strip() for n in args[0].strip("{}").split(",")]
                for i, d in enumerate(dests): R[d] = load(addr + i * bits // 8, bits) if space == "global" else 0
                return
            inner = args[0].strip("[]"); addr = (rd(inner.split("+")[0], 64 if space == "global" else 32) + (int(inner.split("+")[1], 0) if "+" in inner else 0))
            events.append((name + "." + space, ctaid, tid, addr, bits * vec // 8, args[1] if len(args) > 1 else None)); return
        raise NotImplementedError(ins)

def trace_grid(ptx_text, entry, params, grid, block, mem=None, cta_filter=None, thread_filter=None):
    k = Tracer(ptx_text, entry); k._params = params; events = []
    for bz in range(grid[2]):
        for by in range(grid[1]):
            for bx in range(grid[0]):
                if cta_filter and not cta_filter(bx, by, bz): continue
                for tz in range(block[2]):
                    for ty in range(block[1]):
                        for tx in range(block[0]):
                            if thread_filter and not thread_filter(tx, ty, tz): continue
                            k.run_thread(params, (bx, by, bz), block, (tx, ty, tz), grid, mem or {}, events)
    return events
