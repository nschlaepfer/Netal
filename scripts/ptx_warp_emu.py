#!/usr/bin/env python3
"""Numeric CTA-level emulator for the NR ViT PTX kernels (sm_120 subset).

Executes every warp of one CTA in lockstep-per-warp with a barrier scheduler, over real byte buffers:
  - mma.sync.m16n8k32.row.col.f16.e4m3.e4m3.f16 with the accepted QMMA rule (exact E4M3 products,
    fp32 FMA chain in k order starting from C, one round to half),
  - f16x2 arithmetic rounded per operation, E4M3<->f16 conversions,
  - rsqrt/rcp.approx.ftz.f32 through the RTX-captured half tables when the input is a half,
  - shfl.sync idx/bfly, prmt, selp, packing moves, shared memory, cp.async.bulk (immediate),
  - chained-kernel counters (memory-backed, pre-satisfied).
Branches must be warp-uniform (elect.sync is made true on every lane so copy-issue paths do not
diverge); a divergent branch raises. Anything unimplemented raises. This is a diagnostic reference,
not a performance path; it exists to pin the exact semantics the Metal kernels must reproduce.
"""
import re, struct, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from ptx_int_emu import Kernel
from nr_fp8 import E4M3, encode_e4m3

M32 = 0xffffffff; M16 = 0xffff; M64 = 0xffffffffffffffff
LANES = np.arange(32)

def load_tables(root):
    rsq = np.frombuffer(open(os.path.join(root, "rtx-rsqrt-half.bin"), "rb").read(), dtype=np.uint16)
    rcp = np.frombuffer(open(os.path.join(root, "rtx-rcp-half.bin"), "rb").read(), dtype=np.uint16)
    return rsq, rcp

def f16_bits_to_f64(b): return b.astype(np.uint16).view(np.float16).astype(np.float64)
def f64_to_f16_bits(x): return x.astype(np.float16).view(np.uint16).astype(np.uint64)
def f32_bits_to_f64(b): return b.astype(np.uint32).view(np.float32).astype(np.float64)
def f64_to_f32_bits(x): return x.astype(np.float32).view(np.uint32).astype(np.uint64)

class Warp:
    def __init__(self, cta, warp_index, block, ctaid, nctaid):
        self.cta = cta; self.w = warp_index; self.pc = 0; self.done = False
        self.R = {}; self.P = {}
        tid_lin = warp_index * 32 + LANES
        bx, by, bz = block
        self.tid = (tid_lin % bx, (tid_lin // bx) % by, tid_lin // (bx * by))
        self.ctaid = ctaid; self.nctaid = nctaid; self.block = block
        self.waiting = False; self.bar_wait = False; self.exec_count = 0
        self.mask = np.ones(32, dtype=bool); self.div_stack = []   # (reconvergence pc, mask to restore)

def compute_ipdom(kernel):
    """immediate post-dominator (as a statement index) for every branch statement; EXIT is len(instrs)."""
    ins = kernel.instrs; n = len(ins); EXIT = n
    starts = {0} | set(kernel.labels.values())
    for i, s in enumerate(ins):
        if s.startswith("@") or s.startswith("bra") or s == "ret" or s.startswith("ret"):
            starts.add(i + 1)
    starts = sorted(x for x in starts if x < n); blocks = list(zip(starts, starts[1:] + [n]))
    bid = {st: b for b, (st, en) in enumerate(blocks)}
    def succ(b):
        st, en = blocks[b]
        last = ins[en - 1]; nxt = bid.get(en, EXIT)
        body = last.split(" ", 1)[1] if last.startswith("@") else last
        if body.startswith("ret"): return [EXIT]
        if body.startswith("bra"):
            tgt = bid[kernel.labels[body.split()[1]]]
            return [tgt] if not last.startswith("@") else [tgt, nxt]
        return [nxt]
    nb = len(blocks); S = {b: succ(b) for b in range(nb)}; S[EXIT] = []
    nodes = list(range(nb)) + [EXIT]
    pdom = {b: set(nodes) for b in nodes}; pdom[EXIT] = {EXIT}
    changed = True
    while changed:
        changed = False
        for b in range(nb):
            ss = S[b]
            new = ({b} | set.intersection(*[pdom[x] for x in ss])) if ss else {b, EXIT}
            if new != pdom[b]: pdom[b] = new; changed = True
    def ipd(b):
        cands = pdom[b] - {b}
        # the immediate one is the candidate post-dominated by all other candidates
        for c in cands:
            if all(d in pdom[c] for d in cands): return c      # every other post-dominator also post-dominates c
        return EXIT
    result = {}
    for b, (st, en) in enumerate(blocks):
        last = ins[en - 1] if en > st else ""
        if last.startswith("@") and "bra" in last:
            c = ipd(b); result[en - 1] = n if c == EXIT else blocks[c][0]
    return result

PARAM_BASE = 0x700000000            # generic address of the kernel parameter block (mov.b64 %rd, <param symbol>)
TEXTURES = {}       # texture handle -> (float32 array H x W x 4, W, H): sampled by tex.2d (nearest texel, normalized coords, clamp: hypothesis)
OUT_IMAGES = {}     # surface handle -> {"img": uint16 array H x W x 4, "written": bool array H x W}: written by sust.p (f32 -> half round toward zero)
MUFU = {}           # lazily loaded sm_120 MUFU model data (Mac-fg-research evidence): rcp correction ROM, fast::exp2 grids + 2-bit corrections
def mufu_data():
    if not MUFU:
        root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build/nr_ptx/tables")
        MUFU["rcp_rom"] = np.fromfile(os.path.join(root, "sm120-rcp-correction-rom.bin"), dtype=np.uint8)
        MUFU["ex2_unit_fast"] = np.fromfile(os.path.join(root, "fast-exp2-unit.u32"), dtype=np.uint32); MUFU["ex2_frac_fast"] = np.fromfile(os.path.join(root, "fast-exp2-frac.u32"), dtype=np.uint32)
        MUFU["ex2_unit_corr"] = np.fromfile(os.path.join(root, "unit-corrections.bin"), dtype=np.uint8); MUFU["ex2_frac_corr"] = np.fromfile(os.path.join(root, "fraction-corrections.bin"), dtype=np.uint8)
    return MUFU
def ftz32(v):
    """flush f32-valued float64 array denormals to (signed) zero"""
    return np.where(np.abs(v) < 1.1754943508222875e-38, np.copysign(0.0, v), v)
def rcp_rom_f32(x):
    """rcp.approx.ftz.f32 model: correctly rounded reciprocal +-1 ulp per the 2-bit ROM indexed by the input mantissa (FG evidence)"""
    d = mufu_data(); x32 = x.astype(np.float32); base = (np.float32(1) / x32); bb = base.view(np.uint32).astype(np.int64); ib = x32.view(np.uint32).astype(np.int64)
    ie = (ib >> 23) & 0xff; oe = (bb >> 23) & 0xff; mant = ib & 0x7fffff; code = (d["rcp_rom"][mant >> 2].astype(np.int64) >> ((mant & 3) * 2)) & 3
    adj = np.where(code == 1, -1, np.where(code == 2, 1, 0)); ok = ((ib >> 31) == 0) & (ie != 0) & (ie != 0xff) & (oe != 0) & (oe != 0xff)
    out = np.where(ok, bb + adj, bb).astype(np.uint32).view(np.float32).astype(np.float64); return out
def corrected_bits(fast, corr, index):
    delta = ((corr[index >> 2].astype(np.int64) >> ((index & 3) * 2)) & 3) - 1; return (fast[index].astype(np.int64) - delta)
def ex2_sm120_f32(x, pos_unit=True):
    """ex2.approx.ftz.f32 model: Metal fast::exp2 on the FG grids corrected by the FG 2-bit streams (exact for x <= 0 in FG); x > 0 by the same reduction (hypothesis)"""
    d = mufu_data(); x = x.astype(np.float32).astype(np.float64); out = np.ones_like(x); bits = x.astype(np.float32).view(np.uint32).astype(np.int64); expo = (bits >> 23) & 0xff
    neg_unit = (x < 0) & (expo <= 126) & (expo > 103); pos_unit_m = (x > 0) & (expo <= 126) & (expo > 103) & pos_unit
    if neg_unit.any():
        idx = np.minimum(((-x[neg_unit]).astype(np.float32) * np.float32(16777216.0)).astype(np.float32).astype(np.int64), 16777216)
        out[neg_unit] = corrected_bits(d["ex2_unit_fast"], d["ex2_unit_corr"], idx).astype(np.uint32).view(np.float32).astype(np.float64)
    if pos_unit_m.any():
        idx = 16777216 - np.minimum((x[pos_unit_m].astype(np.float32) * np.float32(16777216.0)).astype(np.float32).astype(np.int64), 16777216)
        out[pos_unit_m] = 2.0 * corrected_bits(d["ex2_unit_fast"], d["ex2_unit_corr"], idx).astype(np.uint32).view(np.float32).astype(np.float64)
    frac_m = (expo > 126) & ~(x < -126.0) & ~(x >= 128.0) & (expo != 0xff)
    if frac_m.any():
        v = x[frac_m]; integral = np.floor(v); scale = integral.astype(np.int64) + 2; normalized = (v.astype(np.float32) - scale.astype(np.float32)).astype(np.float32)
        idx = np.minimum(((normalized + np.float32(2.0)).astype(np.float32) * np.float32(8388608.0)).astype(np.float32).astype(np.int64), 8388608)
        base = corrected_bits(d["ex2_frac_fast"], d["ex2_frac_corr"], idx); out[frac_m] = (base + scale * 0x00800000).astype(np.uint32).view(np.float32).astype(np.float64)
    out = np.where((x < -126.0) & (expo != 0xff), 0.0, out); out = np.where((x >= 128.0), np.inf, out); out = np.where(np.isnan(x), np.nan, out)
    return out
def f32_to_f16_rz_bits(v):
    """f32 -> half bit pattern, round toward zero (formatted surface store of f32 data into a half surface: FG module24 evidence)"""
    b = v.astype(np.float32).view(np.uint32).astype(np.int64); sign = (b >> 16) & 0x8000; e = (b >> 23) & 0xff; m = b & 0x7fffff; he = e - 127 + 15
    out = np.where(e == 0xff, sign | np.where(m == 0, 0x7c00, 0x7e00), 0)
    normal = (e != 0xff) & (he >= 1) & (he <= 30); out = np.where(normal, sign | (he << 10) | (m >> 13), out)
    out = np.where((e != 0xff) & (he >= 31), sign | 0x7bff, out)
    sub = (e != 0xff) & (he <= 0) & (he >= -10); sh = np.clip(14 - he, 0, 63); out = np.where(sub, sign | ((m | 0x800000) >> sh), out)
    out = np.where((e != 0xff) & (he < -10), sign, out); return out.astype(np.uint64)

def hmma16_element(a_bits, b_bits, c_bits):
    """one D element of mma.m16n8k16 f16: two guard bits, per-term truncation onto the 2^(maxexp-25) grid, exact integer sum, one RNE (FG HMMA rule)"""
    maxexp = -21; terms = []
    for i in range(16):
        af = (a_bits[i] >> 10) & 31; bf = (b_bits[i] >> 10) & 31; am = a_bits[i] & 1023; bm = b_bits[i] & 1023
        if af: am += 1024
        if bf: bm += 1024
        psig = am * bm * 8; pexp = (-14 if af == 0 else af - 15) + (-14 if bf == 0 else bf - 15); pneg = ((a_bits[i] ^ b_bits[i]) & 0x8000) != 0
        if psig: maxexp = max(maxexp, pexp)
        terms.append((psig, pexp, pneg))
    cf = (c_bits >> 10) & 31; cfr = c_bits & 1023; csig = 0; cexp = -1024
    if cf: csig = (1024 + cfr) << 13; cexp = cf - 15
    elif cfr:
        lead = cfr.bit_length() - 1; csig = cfr << (23 - lead); cexp = lead - 24
    if csig: maxexp = max(maxexp, cexp)
    total = 0
    for psig, pexp, pneg in terms:
        sh = maxexp - pexp; al = 0 if (psig == 0 or sh > 31) else ((psig << 2) >> sh) & 0xffffffff; total += -al if pneg else al
    if csig:
        sh = maxexp - cexp; al = 0 if sh > 31 else ((csig << 2) >> sh) & 0xffffffff; total += -al if (c_bits & 0x8000) else al
    return fixed_to_half_rne(total, maxexp - 25)
def round_shift_rne(value, shift):
    if shift == 0: return value
    if shift >= 64: return 0
    q = value >> shift; r = value & ((1 << shift) - 1); h = 1 << (shift - 1); return q + (1 if (r > h or (r == h and (q & 1))) else 0)
def fixed_to_half_rne(value, scale):
    if value == 0: return 0
    negative = value < 0; magnitude = -value if negative else value; leading = magnitude.bit_length() - 1; exponent = leading + scale
    if exponent >= -14:
        shift = leading - 10; significand = round_shift_rne(magnitude, shift) if shift > 0 else magnitude << (-shift)
        if significand >= 2048: significand >>= 1; exponent += 1
        result = 0x7c00 if exponent > 15 else (((exponent + 15) << 10) | (significand - 1024))
    else:
        shift = -(scale + 24); significand = round_shift_rne(magnitude, shift) if shift > 0 else magnitude << (-shift)
        result = 0x0400 if significand >= 1024 else significand
    return result | (0x8000 if negative else 0)

def qmma_fixed_point(A, B, C, precision=25):
    """QMMA e4m3 rule matching all 4096 RTX registers of the 64-case discriminator (Codex audit 2026-09-05, FG HMMA-style):
    per element, max_exp = max over nonzero products of (floor(log2|a|) + floor(log2|b|)) and floor(log2|C|); every term's
    magnitude is truncated to units 2^(max_exp - precision); the signed integers are summed exactly; one RNE to half.
    A (16x32), B (32x8), C (16x8) as float64 exact values; returns float64 half-rounded values."""
    terms = np.stack([np.outer(A[:, k], B[k, :]) for k in range(32)] + [C])                    # exact in float64
    ae = np.where(A == 0, -1024, np.frexp(np.abs(A))[1] - 1); be = np.where(B == 0, -1024, np.frexp(np.abs(B))[1] - 1)
    pexp = np.stack([ae[:, k, None] + be[k, None, :] for k in range(32)])
    pexp = np.where(terms[:32] == 0, -1024, pexp)
    cexp = np.where(C == 0, -1024, np.frexp(np.abs(C))[1] - 1)
    maxexp = np.maximum(pexp.max(axis=0), cexp)
    scale = maxexp - precision
    mags = np.floor(np.ldexp(np.abs(terms), -scale[None, :, :])).astype(np.int64)
    total = np.where(terms < 0, -mags, mags).sum(axis=0, dtype=np.int64)
    out = np.ldexp(total.astype(np.float64), scale).astype(np.float16).astype(np.float64)
    return np.where(total == 0, 0.0, out)

POKES = {}      # pc -> [(reg, warp, lane, kind, delta)]: register perturbations applied after the statement at pc executes
def mma_accumulate(P, C, order):
    """fp32 accumulation of the 32 exact products P[k] (16x8 float32) and the half C operand under one association."""
    if order == "seq":
        acc = C.copy()
        for k in range(32): acc = (P[k] + acc).astype(np.float32)
    elif order == "c_last":
        acc = np.zeros_like(C)
        for k in range(32): acc = (P[k] + acc).astype(np.float32)
        acc = (acc + C).astype(np.float32)
    elif order == "rev":
        acc = C.copy()
        for k in range(31, -1, -1): acc = (P[k] + acc).astype(np.float32)
    elif order == "halves":          # two chains of 16 from C and 0, then combined
        a0 = C.copy(); a1 = np.zeros_like(C)
        for k in range(16): a0 = (P[k] + a0).astype(np.float32); a1 = (P[16 + k] + a1).astype(np.float32)
        acc = (a0 + a1).astype(np.float32)
    elif order == "quads":           # four chains of 8 (k%4 interleaved? no: contiguous), C into chain 0
        ch = [C.copy(), np.zeros_like(C), np.zeros_like(C), np.zeros_like(C)]
        for k in range(32): ch[k // 8] = (P[k] + ch[k // 8]).astype(np.float32)
        acc = (((ch[0] + ch[1]).astype(np.float32) + ch[2]).astype(np.float32) + ch[3]).astype(np.float32)
    elif order == "tree":            # pairwise tree over the 32 products, C added last
        lvl = P
        while len(lvl) > 1: lvl = [(lvl[i] + lvl[i + 1]).astype(np.float32) for i in range(0, len(lvl), 2)]
        acc = (lvl[0] + C).astype(np.float32)
    elif order == "tree_c_first":    # C + p0 first, then pairwise tree
        lvl = [(C + P[0]).astype(np.float32)] + P[1:]
        lvl = lvl + [np.zeros_like(C)]              # 33 -> pad
        while len(lvl) > 1:
            if len(lvl) % 2: lvl.append(np.zeros_like(C))
            lvl = [(lvl[i] + lvl[i + 1]).astype(np.float32) for i in range(0, len(lvl), 2)]
        acc = lvl[0]
    else: raise NotImplementedError(order)
    return acc

class CTA:
    def __init__(self, kernel, params, ctaid, block, grid, gmem, shared_bytes, tables, trace_ops=False):
        self.k = kernel; self.params = params; self.gmem = dict(gmem); self.gmem[PARAM_BASE] = bytearray(params)
        if getattr(kernel, "ipdom", None) is None: kernel.ipdom = compute_ipdom(kernel)
        self.ipdom = kernel.ipdom
        self.written = []                       # (absolute address, byte count) of every global store / reduction, per lane
        self.reads = []; self.track_reads = bool(os.environ.get("EMU_TRACK_READS"))          # global loads, when requested
        self.pokes = {pc: list(v) for pc, v in POKES.items()}
        for item in filter(None, os.environ.get("EMU_POKE", "").split(",")):     # pc:reg:warp:lane:kind:delta
            f = item.split(":"); self.pokes.setdefault(int(f[0]), []).append((f[1], int(f[2]), int(f[3]), f[4], int(f[5], 0)))
        self.sense = open(os.environ["EMU_MMA_SENSE"], "a") if os.environ.get("EMU_MMA_SENSE") else None
        self.pcensus = open(os.environ["EMU_P_CENSUS"], "a") if os.environ.get("EMU_P_CENSUS") else None       # count A-operand bytes != 0x08 (uniform P) at the given mma pcs
        self.pcensus_pcs = set()
        for rg in filter(None, os.environ.get("EMU_P_CENSUS_PCS", "").split(",")):
            lo, hi = rg.split("-"); self.pcensus_pcs.update(range(int(lo), int(hi) + 1))
        self.shared = bytearray(shared_bytes); self.rsq, self.rcp = tables
        nw = (block[0] * block[1] * block[2]) // 32
        self.warps = [Warp(self, w, block, ctaid, grid) for w in range(nw)]
        self.trace_ops = trace_ops; self.unimpl = {}
        self.cplog = [] if os.environ.get("EMU_LOG_CP") else None
        self.watch = {}
        for item in filter(None, os.environ.get("EMU_WATCH", "").split(",")):
            f = item.split(":"); pc, reg = f[0], f[1]; self.watch.setdefault(int(pc), []).append((reg, int(f[2]) if len(f) > 2 else 0))

    # ---------------- memory helpers ----------------
    def gfind(self, addr):
        for base, buf in self.gmem.items():
            if base <= addr < base + len(buf): return buf, addr - base
        raise KeyError("global address 0x%x unmapped" % addr)
    def gload(self, addrs, nbytes, mask=None):
        out = np.zeros((32, nbytes), dtype=np.uint8)
        for l in range(32):
            if mask is not None and not mask[l]: continue
            a = int(addrs[l]); buf, o = self.gfind(a); out[l] = np.frombuffer(bytes(buf[o:o + nbytes]), dtype=np.uint8)
            if self.track_reads: self.reads.append((a, nbytes))
        return out
    def gstore(self, addrs, data, mask):
        for l in range(32):
            if not mask[l]: continue
            a = int(addrs[l]); buf, o = self.gfind(a); buf[o:o + len(data[l])] = bytes(data[l]); self.written.append((a, len(data[l])))
    def sload(self, addrs, nbytes, mask=None):
        out = np.zeros((32, nbytes), dtype=np.uint8)
        for l in range(32):
            if mask is not None and not mask[l]: continue
            o = int(addrs[l]); out[l] = np.frombuffer(bytes(self.shared[o:o + nbytes]), dtype=np.uint8)
        return out
    def sstore(self, addrs, data, mask):
        for l in range(32):
            if not mask[l]: continue
            o = int(addrs[l]); self.shared[o:o + len(data[l])] = bytes(data[l])

    # ---------------- register helpers ----------------
    @staticmethod
    def width_of(t):
        m = re.search(r"(128|64|32|16|8)$", t); return int(m.group(1)) if m else 32
    def rd(self, wp, x, bits=32):
        x = x.strip()
        mask = (1 << bits) - 1
        if x == self.k.param_name: return np.full(32, PARAM_BASE, dtype=np.uint64)
        if x.startswith("%") or (x[0].isalpha() and not x.startswith("_Z") and x not in ("WARP_SZ",)):
            if x in wp.R:
                v = wp.R[x]
                if isinstance(v, tuple): return v
                return v & mask
            sp = {"%laneid": LANES, "%tid.x": wp.tid[0], "%tid.y": wp.tid[1], "%tid.z": wp.tid[2],
                  "%ctaid.x": wp.ctaid[0], "%ctaid.y": wp.ctaid[1], "%ctaid.z": wp.ctaid[2],
                  "%ntid.x": wp.block[0], "%ntid.y": wp.block[1], "%ntid.z": wp.block[2],
                  "%nctaid.x": wp.nctaid[0], "%nctaid.y": wp.nctaid[1], "%nctaid.z": wp.nctaid[2], "WARP_SZ": 32}
            if x in sp: return (np.zeros(32, dtype=np.uint64) + np.uint64(0) + np.asarray(sp[x], dtype=np.uint64)) & mask
            return np.zeros(32, dtype=np.uint64)     # undefined register reads as 0
        if x.startswith("0f") or x.startswith("0d"): v = int(x[2:], 16)
        elif x.startswith("0x") or x.startswith("-0x"): v = int(x, 16)
        elif x.startswith("0F"): v = int(x[2:], 16)
        elif x.startswith("_"): v = self.k.shared_syms.get(x, 0)         # mangled shared symbol -> its shared offset
        elif x == "WARP_SZ": v = 32
        else: v = int(x)
        return np.full(32, v & mask, dtype=np.uint64)
    def wr(self, wp, x, val, bits=32):
        v = np.asarray(val, dtype=np.uint64) & ((1 << bits) - 1)
        if not wp.mask.all():
            old = wp.R.get(x.strip()); 
            if old is None or isinstance(old, tuple): old = np.zeros(32, dtype=np.uint64)
            v = np.where(wp.mask, v, old)
        wp.R[x.strip()] = v
    def signed(self, v, bits):
        v = v.astype(np.int64) if bits < 64 else v.view(np.int64)
        if bits < 64:
            sign = 1 << (bits - 1); v = np.where(v & sign, v - (1 << bits), v)
        return v

    def addr(self, wp, operand, bits):
        inner = operand.strip().strip("[]")
        if "+" in inner:
            base, off = inner.split("+", 1); return (self.rd(wp, base.strip(), bits) + int(off.strip(), 0)) & ((1 << bits) - 1)
        return self.rd(wp, inner, bits)

    # ---------------- f16 helpers ----------------
    def h2(self, bits32):   # -> (lo, hi) float64 arrays
        b = bits32.astype(np.uint64); return f16_bits_to_f64(b & M16), f16_bits_to_f64((b >> 16) & M16)
    def pack_h2(self, lo, hi):
        if os.environ.get("EMU_F16_FTZ"):                                                               # hypothesis: f16x2 results flushed
            lo = np.where(np.abs(lo) < 6.103515625e-05, 0.0, lo); hi = np.where(np.abs(hi) < 6.103515625e-05, 0.0, hi)
        return f64_to_f16_bits(lo) | (f64_to_f16_bits(hi) << 16)
    def e4m3_pair_to_f16x2(self, bits16):
        b = bits16.astype(np.uint64); lo = E4M3[(b & 0xff).astype(np.int64)].astype(np.float64); hi = E4M3[((b >> 8) & 0xff).astype(np.int64)].astype(np.float64)
        return self.pack_h2(lo, hi)
    def f16x2_to_e4m3_pair(self, bits32):
        lo, hi = self.h2(bits32)
        if os.environ.get("EMU_E4M3_CENSUS"):          # rare-value classes of the conversion inputs, per CTA
            c = self.__dict__.setdefault("e4m3_census", {"values": 0, "sat_gt448": 0, "in_448_464": 0, "subnormal": 0, "sub_tie": 0, "below_half_min": 0, "nan_inf": 0})
            for v in (lo, hi):
                a = np.abs(v); c["values"] += int(a.size); c["sat_gt448"] += int((a > 448).sum()); c["in_448_464"] += int(((a > 448) & (a < 464)).sum())
                c["subnormal"] += int(((a > 0) & (a < 2.0 ** -6)).sum()); c["below_half_min"] += int(((a > 0) & (a < 2.0 ** -10)).sum()); c["nan_inf"] += int((~np.isfinite(v)).sum())
                sub = a[(a > 0) & (a < 2.0 ** -6)]; c["sub_tie"] += int((np.abs(sub / 2.0 ** -9 - np.round(sub / 2.0 ** -9)) == 0.5).sum())
        if os.environ.get("EMU_TIE_CENSUS"):
            for v in (lo, hi):
                a = np.abs(v); fin = a[(a > 0) & (a < 448)]
                if fin.size:
                    m, e = np.frexp(fin)          # fin = m * 2**e, m in [0.5, 1)
                    sig = m * 2                    # [1, 2)
                    ties = np.abs(sig * 8 - np.round(sig * 8)) == 0.5          # exactly halfway between two 3-bit mantissas
                    binade = np.abs(sig - 1.9375) < 1e-12
                    self.tie_count = getattr(self, "tie_count", 0) + int(ties.sum()); self.binade_tie_count = getattr(self, "binade_tie_count", 0) + int((ties & binade).sum())
        return encode_e4m3(lo.astype(np.float32)).astype(np.uint64) | (encode_e4m3(hi.astype(np.float32)).astype(np.uint64) << 8)

    # ---------------- MMA ----------------
    def mma(self, wp, dregs, aregs, bregs, cregs):
        g = LANES // 4; tig = LANES % 4
        A = np.zeros((16, 32), dtype=np.float32); B = np.zeros((32, 8), dtype=np.float32); C = np.zeros((16, 8), dtype=np.float32)
        a = [self.rd(wp, r) for r in aregs]; b = [self.rd(wp, r) for r in bregs]; c = [self.rd(wp, r) for r in cregs]
        if self.pcensus is not None and (wp.pc - 1) in self.pcensus_pcs:
            n = sum(int(((np.asarray(a[i], dtype=np.uint64) >> np.uint64(8 * j)) & np.uint64(0xff) != 8).sum()) for i in range(4) for j in range(4))
            self.pcensus.write(f"{wp.ctaid[0]} {wp.ctaid[1]} {wp.pc - 1} {n}\n"); self.pcensus.flush()
        for l in range(32):
            for j in range(4):
                A[g[l], 4 * tig[l] + j] = E4M3[int(a[0][l] >> (8 * j)) & 0xff]
                A[g[l] + 8, 4 * tig[l] + j] = E4M3[int(a[1][l] >> (8 * j)) & 0xff]
                A[g[l], 16 + 4 * tig[l] + j] = E4M3[int(a[2][l] >> (8 * j)) & 0xff]
                A[g[l] + 8, 16 + 4 * tig[l] + j] = E4M3[int(a[3][l] >> (8 * j)) & 0xff]
                B[4 * tig[l] + j, g[l]] = E4M3[int(b[0][l] >> (8 * j)) & 0xff]
                B[16 + 4 * tig[l] + j, g[l]] = E4M3[int(b[1][l] >> (8 * j)) & 0xff]
            lo, hi = self.h2(c[0][l:l + 1]); C[g[l], 2 * tig[l]] = lo[0]; C[g[l], 2 * tig[l] + 1] = hi[0]
            lo, hi = self.h2(c[1][l:l + 1]); C[g[l] + 8, 2 * tig[l]] = lo[0]; C[g[l] + 8, 2 * tig[l] + 1] = hi[0]
        if os.environ.get("EMU_MMA_DEBUG") and wp.w == 0 and getattr(wp, "mma_count", 0) < 3:
            wp.mma_count = getattr(wp, "mma_count", 0) + 1
            print(f"[mma debug] warp0 #{wp.mma_count}: A nonzero={int((A!=0).sum())}/512 B nonzero={int((B!=0).sum())}/256 C absmax={float(np.abs(C).max()):.4g} a0 lane0-3={[hex(int(x)) for x in a[0][:4]]} b0 lane0-3={[hex(int(x)) for x in b[0][:4]]}")
        if os.environ.get("EMU_MMA_EXACT"):        # hypothesis: exact sum of the 32 products and C, one rounding to half
            acc64 = C.astype(np.float64)
            for k in range(32): acc64 = acc64 + np.outer(A[:, k].astype(np.float64), B[k, :].astype(np.float64))
            D = acc64.astype(np.float16).astype(np.float64)
        else:
            order = os.environ.get("EMU_MMA_ORDER", "fixed")       # default "fixed" = the RTX-matched fixed-point rule (qmma_fixed_point); "c_last" products-first fp32 chain; "seq" the old C-first chain
            P = [np.outer(A[:, k], B[k, :]).astype(np.float32) for k in range(32)]      # exact products
            if order == "fixed": D = qmma_fixed_point(A.astype(np.float64), B.astype(np.float64), C.astype(np.float64)); acc = D.astype(np.float32)
            if os.environ.get("EMU_QMMA_SPAN_CENSUS"):      # guard hit rate: exponent span of the nonzero terms of each output element
                A64 = A.astype(np.float64); B64 = B.astype(np.float64); C64 = C.astype(np.float64)
                ae = np.where(A64 == 0, -100000, np.frexp(np.abs(A64))[1] - 1); be = np.where(B64 == 0, -100000, np.frexp(np.abs(B64))[1] - 1)
                pe = np.stack([ae[:, k, None] + be[k, None, :] for k in range(32)]); pe = np.where(pe < -50000, -100000, pe)
                ce = np.where(C64 == 0, -100000, np.frexp(np.abs(C64))[1] - 1); allx = np.concatenate([pe, ce[None]], axis=0)
                mx = allx.max(axis=0); mn = np.where(allx > -50000, allx, 100000).min(axis=0); span = np.where(mx > -50000, mx - mn, 0)
                # q-based criterion: terms as integers in units 2^qmin; the fp32 chain is exact when maxexp - qmin <= 17 (33 terms below 2^24 units)
                qa = np.where(A64 == 0, 100000, np.where(np.abs(A64) < 2.0 ** -6, -9, ae - 3)); qb = np.where(B64 == 0, 100000, np.where(np.abs(B64) < 2.0 ** -6, -9, be - 3))
                pq = np.stack([qa[:, k, None] + qb[k, None, :] for k in range(32)]); pq = np.where(terms_zero := (pe < -50000), 100000, pq)
                cq = np.where(C64 == 0, 100000, np.where(np.abs(C64) < 2.0 ** -14, -24, ce - 10)); qmin = np.minimum(pq.min(axis=0), cq)
                qspan = np.where(mx > -50000, mx - qmin, 0)
                c = self.__dict__.setdefault("span_census", {"elements": 0, "span_le7": 0, "span_le11": 0, "span_le18": 0, "qspan_le17": 0})
                c["elements"] += int(span.size); c["span_le7"] += int((span <= 7).sum()); c["span_le11"] += int((span <= 11).sum()); c["span_le18"] += int((span <= 18).sum()); c["qspan_le17"] += int((qspan <= 17).sum())
            elif order != "fixed":   # (2026-09-06: the accepted fixed-point rule was computed above but then overwritten by the fp32-chain fallback, which raised for "fixed")
                acc = mma_accumulate(P, C, order)
                D = acc.astype(np.float16).astype(np.float64)
            if self.sense is not None:                     # log D elements whose half result depends on the association
                alt = {o: mma_accumulate(P, C, o).astype(np.float16).view(np.uint16) for o in ("seq", "c_last", "tree", "halves")}
                base = alt[order] if order in alt else acc.astype(np.float16).view(np.uint16)
                diff = np.zeros(C.shape, dtype=bool)
                for v in alt.values(): diff |= (v != base)
                for r, ci in zip(*np.nonzero(diff)):
                    self.sense.write(f"{wp.ctaid[0]} {wp.ctaid[1]} pc {wp.pc - 1} warp {wp.w} lane {4 * (r % 8) + ci // 2} dreg {r // 8} half {ci % 2} " + " ".join(f"{o}={int(alt[o][r, ci]):04x}" for o in alt) + "\n")
                self.sense.flush()
        if os.environ.get("EMU_MMA_FTZ"): D = np.where(np.abs(D) < 6.103515625e-05, 0.0, D)          # hypothesis: half-subnormal D flushed
        d0 = self.pack_h2(D[g, 2 * tig], D[g, 2 * tig + 1]); d1 = self.pack_h2(D[g + 8, 2 * tig], D[g + 8, 2 * tig + 1])
        self.wr(wp, dregs[0], d0); self.wr(wp, dregs[1], d1)

    def mma_f16(self, wp, dregs, aregs, bregs, cregs):
        """mma.sync.m16n8k16.row.col.f16.f16.f16.f16 with the HMMA rule (see hmma16_element). Fragments: a0 row g k 2t..; a1 row g+8;
        a2 row g k 8+2t..; a3 row g+8 k 8+2t..; b0 k 2t.. col g; b1 k 8+2t.. col g; c0/d0 row g cols 2t,2t+1; c1/d1 row g+8."""
        a = [self.rd(wp, r) for r in aregs]; b = [self.rd(wp, r) for r in bregs]; c = [self.rd(wp, r) for r in cregs]
        A = [[0] * 16 for _ in range(16)]; B = [[0] * 8 for _ in range(16)]; C = [[0] * 8 for _ in range(16)]
        for l in range(32):
            gl, tl = l // 4, l % 4
            for j in range(2):
                A[gl][2 * tl + j] = (int(a[0][l]) >> (16 * j)) & 0xffff; A[gl + 8][2 * tl + j] = (int(a[1][l]) >> (16 * j)) & 0xffff
                A[gl][8 + 2 * tl + j] = (int(a[2][l]) >> (16 * j)) & 0xffff; A[gl + 8][8 + 2 * tl + j] = (int(a[3][l]) >> (16 * j)) & 0xffff
                B[2 * tl + j][gl] = (int(b[0][l]) >> (16 * j)) & 0xffff; B[8 + 2 * tl + j][gl] = (int(b[1][l]) >> (16 * j)) & 0xffff
                C[gl][2 * tl + j] = (int(c[0][l]) >> (16 * j)) & 0xffff; C[gl + 8][2 * tl + j] = (int(c[1][l]) >> (16 * j)) & 0xffff
        Bt = [[B[k][n] for k in range(16)] for n in range(8)]
        mode = os.environ.get("EMU_HMMA_MODE", "rule")
        if mode == "rule": D = [[hmma16_element(A[r], Bt[n], C[r][n]) for n in range(8)] for r in range(16)]
        else:            # alternative hypotheses: fp32 FMA chain from C in k order, or exact sum, one RNE to half
            h = lambda bits: float(np.array([bits], dtype=np.uint16).view(np.float16)[0])
            D = [[0] * 8 for _ in range(16)]
            for r in range(16):
                for n in range(8):
                    if mode == "fp32":
                        acc = np.float32(h(C[r][n]))
                        for k in range(16): acc = np.float32(np.float32(h(A[r][k])) * np.float32(h(Bt[n][k])) + acc)
                        D[r][n] = int(np.array([acc], dtype=np.float32).astype(np.float16).view(np.uint16)[0])
                    else:
                        from fractions import Fraction
                        tot = Fraction(h(C[r][n])) + sum(Fraction(h(A[r][k])) * Fraction(h(Bt[n][k])) for k in range(16))
                        D[r][n] = int(np.array([float(tot)], dtype=np.float64).astype(np.float16).view(np.uint16)[0])
        d0 = np.zeros(32, dtype=np.uint64); d1 = np.zeros(32, dtype=np.uint64)
        for l in range(32):
            gl, tl = l // 4, l % 4
            d0[l] = D[gl][2 * tl] | (D[gl][2 * tl + 1] << 16); d1[l] = D[gl + 8][2 * tl] | (D[gl + 8][2 * tl + 1] << 16)
        self.wr(wp, dregs[0], d0); self.wr(wp, dregs[1], d1)

    # ---------------- execution ----------------
    def run(self, max_steps=5_000_000):
        pending = set(range(len(self.warps))); steps = 0; idle_rounds = 0
        self.mbar = {}          # shared address -> {"count", "pending", "phase"}
        while pending:
            executed = 0
            live = [wp for wp in self.warps if not wp.done]
            if live and all(wp.bar_wait for wp in live):            # every live warp sits at bar.sync: release them all
                for wp in live: wp.bar_wait = False
            for wp in self.warps:
                if wp.done or wp.bar_wait: continue
                n0 = wp.exec_count; self.run_warp(wp); executed += wp.exec_count - n0
                if wp.done: pending.discard(wp.w)
            steps += 1
            if steps > 400000: raise RuntimeError("scheduler stuck")
            idle_rounds = idle_rounds + 1 if executed == 0 else 0
            if idle_rounds > 4: raise RuntimeError("deadlock: no warp progressed; bar_wait=" + str([wp.bar_wait for wp in self.warps]) + " pcs=" + str([wp.pc for wp in self.warps]) + " mbar=" + str(self.mbar))
    def run_warp(self, wp):
        """run until a yield point (barrier / mbarrier wait) or ret."""
        k = self.k; n = 0
        while wp.pc < len(k.instrs):
            while wp.div_stack and wp.pc == wp.div_stack[-1][0]:
                ipd, pend_pc, pend_mask, saved = wp.div_stack[-1]
                if pend_pc is not None:                               # run the other path now
                    wp.div_stack[-1] = (ipd, None, None, saved); wp.pc = pend_pc; wp.mask = pend_mask
                    wp.masklog = getattr(wp, "masklog", []); wp.masklog.append((wp.pc, "path2", int(pend_mask.sum()), len(wp.div_stack), ""))
                    if wp.pc != ipd: break
                    continue
                wp.div_stack.pop(); wp.mask = saved                   # both paths reached the post-dominator
                wp.masklog = getattr(wp, "masklog", []); wp.masklog.append((wp.pc, "join", int(saved.sum()), len(wp.div_stack), ""))
            ins = k.instrs[wp.pc]; wp.pc += 1; n += 1; wp.exec_count += 1
            if n > 2_000_000: raise RuntimeError("runaway warp")
            wp.hist = getattr(wp, "hist", []); wp.hist.append((wp.pc - 1, ins)); 
            if len(wp.hist) > 40: wp.hist.pop(0)
            try:
                r = self.exec(wp, ins)
                if self.pokes and (wp.pc - 1) in self.pokes:
                    for reg, w, lane, kind, delta in self.pokes[wp.pc - 1]:
                        if w != wp.w: continue
                        v = wp.R[reg].copy(); x = int(v[lane])
                        if kind == "lo": x = (x & 0xffff0000) | (((x & 0xffff) + delta) & 0xffff)
                        elif kind == "hi": x = (x & 0xffff) | (((((x >> 16) & 0xffff) + delta) & 0xffff) << 16)
                        elif kind[0] == "b": sh = 8 * int(kind[1]); x = (x & ~(0xff << sh) & 0xffffffff) | (((((x >> sh) & 0xff) + delta) & 0xff) << sh)
                        else: x = (x + delta) & 0xffffffff
                        v[lane] = x; wp.R[reg] = v
                if self.watch and (wp.pc - 1) in self.watch:
                    for reg, ww in self.watch[wp.pc - 1]:
                        if ww != wp.w: continue
                        v = wp.R.get(reg); lanes = [int(x) for x in os.environ.get("EMU_WATCH_LANES", "0,1,2,3").split(",") if x.strip()]
                        print(f"[watch] cta {wp.ctaid} warp {wp.w} pc {wp.pc-1} {reg} lanes{lanes}:", [hex(int(v[x])) for x in lanes] if v is not None and not isinstance(v, tuple) else v, "P" if reg in wp.P else "", flush=True)
            except Exception as e:
                regs = sorted(set(re.findall(r"%r[ds]?\d+", " ".join(h[1] for h in wp.hist[-12:]))))
                dump = {rg: [hex(int(x)) for x in wp.R[rg][:4]] for rg in regs if rg in wp.R and not isinstance(wp.R[rg], tuple)}
                raise RuntimeError(f"warp {wp.w} pc {wp.pc-1}: {ins}\n  mask={wp.mask.astype(int).tolist()}\n  masklog(first)={getattr(wp,'masklog',[])[:10]}\n  masklog(last)={getattr(wp,'masklog',[])[-6:]}\n  recent:\n    " + "\n    ".join(f"{pc}: {s}" for pc, s in wp.hist[-14:]) + f"\n  regs(lanes0-3): {dump}") from e
            if r == "ret": wp.done = True; return
            if r == "yield": return
            if isinstance(r, tuple): wp.pc = k.labels[r[1]]
        wp.done = True

    def exec(self, wp, ins):
        ins = ins.strip()
        while ins.startswith("{ ") or ins.startswith("{."): ins = ins[1:].strip()     # compound-block remnants only
        if ins.endswith("}") and ins.count("{") < ins.count("}"): ins = ins[:-1].strip()
        if not ins or ins.startswith(".reg"): return
        pred = None
        if ins.startswith("@"):
            p, ins = ins.split(" ", 1); neg = p.startswith("@!"); pv = wp.P.get(p.lstrip("@!"), np.zeros(32, dtype=bool))
            act = ((~pv) if neg else pv) & wp.mask
            if not act.any(): return
            if ins.startswith("bra"):
                if (act == wp.mask).all(): return ("bra", ins.split()[1])      # uniform for the active group
                # divergent branch: run the fall-through lanes first, then the taken lanes, merge at the ipdom
                target = self.k.labels[ins.split()[1]]; ipd = self.ipdom[wp.pc - 1]
                wp.div_stack.append((ipd, target, act.copy(), wp.mask.copy())); wp.mask = wp.mask & ~act
                wp.masklog = getattr(wp, "masklog", []); wp.masklog.append((wp.pc - 1, "div", int(act.sum()), len(wp.div_stack), ins)); return
            pred = act
        elif not wp.mask.all():
            if ins.startswith("bra"): return ("bra", ins.split()[1])            # the active group follows its own jumps
            if ins.startswith("mma") or ins.startswith("shfl"): raise RuntimeError("non-uniform " + ins + f" mask={wp.mask.astype(int).tolist()}")
            pred = wp.mask.copy()
        op, _, rest = ins.partition(" ")
        args = [a.strip() for a in re.split(r",(?![^{]*\})", rest)] if rest else []
        parts = op.split("."); name = parts[0]
        if name == "ret": return "ret"
        if ins.lstrip().startswith(".pragma"): return
        if name == "bra": return ("bra", args[0])
        if name in ("bar", "barrier"): wp.bar_wait = True; return "yield"
        if name == "mbarrier":
            # phase-accurate mbarrier: init sets the arrival count; each arriving lane subtracts its count; the phase
            # advances when the count reaches zero; try_wait succeeds once the phase has moved past the token's phase.
            # Bulk copies are performed at issue, so tx accounting is not needed for correctness.
            if "init" in parts:
                a = int(self.addr(wp, args[0], 32)[int(np.argmax(wp.mask))]); cnt = int(self.rd(wp, args[1], 32)[int(np.argmax(wp.mask))])
                self.mbar[a] = {"count": cnt, "pending": cnt, "phase": 0}; return
            if "expect_tx" in parts: return
            if "arrive" in parts:
                mask = pred if pred is not None else wp.mask
                a = int(self.addr(wp, args[1], 32)[int(np.argmax(mask))]); st = self.mbar[a]
                cnt = self.rd(wp, args[2], 32) if len(args) > 2 else np.ones(32, dtype=np.uint64)
                st["pending"] -= int(cnt[mask].sum())
                token = np.full(32, st["phase"], dtype=np.uint64)
                if st["pending"] <= 0: st["phase"] += 1; st["pending"] = st["count"]
                self.wr(wp, args[0], token, 64); return
            if "try_wait" in parts or "test_wait" in parts:
                a = int(self.addr(wp, args[1], 32)[int(np.argmax(wp.mask))]); st = self.mbar[a]; tok = int(self.rd(wp, args[2], 64)[int(np.argmax(wp.mask))])
                ok = st["phase"] > tok
                wp.P[args[0]] = np.full(32, ok, dtype=bool)
                return None if ok else "yield"
            raise NotImplementedError(ins)
        if name in ("nanosleep", "fence", "membar", "elect"):
            if name == "elect": wp.P["P_OUT"] = np.ones(32, dtype=bool)
            return
        if name == "cvta": self.wr(wp, args[0], self.rd(wp, args[1], 64), 64); return
        if name == "ld" and "param" in parts:
            bits = self.width_of(parts[-1]); m = re.match(r"\[%s(?:\+(\d+))?\]" % self.k.param_name, args[-1])
            off = int(m.group(1) or 0) if m else int(self.addr(wp, args[-1], 64)[int(np.argmax(wp.mask))]) - PARAM_BASE      # register-based parameter address
            names = [x.strip() for x in args[0].strip("{}").split(",")]
            for i, nm in enumerate(names):
                v = int.from_bytes(self.params[off + i * bits // 8: off + (i + 1) * bits // 8], "little"); self.wr(wp, nm, np.full(32, v, dtype=np.uint64), bits)
            return
        if name == "mov" and parts[-1] == "pred":
            src = args[1].strip()
            v = wp.P[src].copy() if src.startswith("%") else np.full(32, int(src, 0) != 0, dtype=bool)
            mask = pred if pred is not None else wp.mask
            wp.P[args[0]] = np.where(mask, v, wp.P.get(args[0], np.zeros(32, dtype=bool))); return
        if name == "mov":
            bits = 128 if parts[-1] == "b128" else self.width_of(parts[-1])
            if args[1].startswith("{"):
                srcs = [x.strip() for x in args[1].strip("{}").split(",")]; part = bits // len(srcs)
                if bits == 128: wp.R[args[0]] = ("b128", [self.rd(wp, s, part) for s in srcs]); return
                v = np.zeros(32, dtype=np.uint64)
                for i, s in enumerate(srcs): v |= self.rd(wp, s, part) << (part * i)
                self.wr(wp, args[0], v, bits); return
            if args[0].startswith("{"):
                names = [x.strip() for x in args[0].strip("{}").split(",")]; part = bits // len(names)
                v = self.rd(wp, args[1], bits)
                for i, nm in enumerate(names): self.wr(wp, nm, (v >> (part * i)) & ((1 << part) - 1), part)
                return
            if bits == 128: wp.R[args[0]] = wp.R[args[1]]; return
            self.wr(wp, args[0], self.rd(wp, args[1], bits), bits); return
        if name == "selp":
            bits = self.width_of(parts[-1]); p = wp.P[args[3]]
            self.wr(wp, args[0], np.where(p, self.rd(wp, args[1], bits), self.rd(wp, args[2], bits)), bits); return
        if name == "setp":
            t = parts[-1]; cmp = parts[1]
            if t in ("f16", "f32", "f16x2"):
                a = f32_bits_to_f64(self.rd(wp, args[1])) if t == "f32" else f16_bits_to_f64(self.rd(wp, args[1], 16)); b = f32_bits_to_f64(self.rd(wp, args[2])) if t == "f32" else f16_bits_to_f64(self.rd(wp, args[2], 16))
            else:
                bits = self.width_of(t); a = self.rd(wp, args[1], bits); b = self.rd(wp, args[2], bits)
                if t.startswith("s"): a = self.signed(a, bits); b = self.signed(b, bits)
                else: a = a.astype(np.int64) if bits < 64 else a; b = b.astype(np.int64) if bits < 64 else b
            if t in ("f16", "f32", "f16x2") and "ftz" in parts: a = ftz32(a); b = ftz32(b)
            un = np.isnan(a) | np.isnan(b) if t in ("f16", "f32", "f16x2") else np.zeros(32, dtype=bool)
            v = {"eq": a == b, "ne": a != b, "lt": a < b, "le": a <= b, "gt": a > b, "ge": a >= b, "lo": a < b, "ls": a <= b, "hi": a > b, "hs": a >= b,
                 "equ": (a == b) | un, "neu": (a != b) | un, "ltu": (a < b) | un, "leu": (a <= b) | un, "gtu": (a > b) | un, "geu": (a >= b) | un, "num": ~un, "nan": un}[cmp]
            if len(args) > 3:
                q = wp.P.get(args[3].lstrip("!"), np.zeros(32, dtype=bool)); q = ~q if args[3].startswith("!") else q
                v = (v & q) if (len(parts) < 4 or parts[2] == "and") else (v | q)
            nm = args[0].split("|")[0]; wp.P[nm] = np.where(wp.mask, v, wp.P.get(nm, np.zeros(32, dtype=bool))); return
        if name in ("and", "or", "xor", "not") and parts[-1] == "pred":
            if name == "not": wp.P[args[0]] = ~wp.P.get(args[1], np.zeros(32, dtype=bool)); return
            a = wp.P.get(args[1].lstrip("!"), np.zeros(32, dtype=bool)); b = wp.P.get(args[2].lstrip("!"), np.zeros(32, dtype=bool))
            if args[1].startswith("!"): a = ~a
            if args[2].startswith("!"): b = ~b
            wp.P[args[0]] = (a & b) if name == "and" else (a | b) if name == "or" else (a ^ b); return
        if name == "mma":
            d = [x.strip() for x in args[0].strip("{}").split(",")]; a = [x.strip() for x in args[1].strip("{}").split(",")]
            b = [x.strip() for x in args[2].strip("{}").split(",")]; c = [x.strip() for x in args[3].strip("{}").split(",")]
            assert ("m16n8k32" in op and "e4m3.e4m3" in op) or "m16n8k16.row.col.f16.f16.f16.f16" in op, op
            (self.mma_f16 if ".f16.f16.f16.f16" in op else self.mma)(wp, d, a, b, c); return
        if name == "shfl":
            d = args[0].split("|")[0]; a = self.rd(wp, args[1]); bidx = self.rd(wp, args[2]).astype(np.int64)
            if "bfly" in op: src = LANES ^ (bidx & 31)
            elif "idx" in op: src = bidx & 31
            elif "up" in op: src = np.maximum(LANES - (bidx & 31), 0)
            elif "down" in op: src = np.minimum(LANES + (bidx & 31), 31)
            else: raise NotImplementedError(ins)
            self.wr(wp, d, a[src]);
            if "|" in args[0]: wp.P[args[0].split("|")[1]] = np.ones(32, dtype=bool)
            return
        if name == "prmt":
            a = self.rd(wp, args[1]); b = self.rd(wp, args[2]); sel = int(args[3].rstrip("U"), 16)
            src = (b.astype(np.uint64) << 32) | a.astype(np.uint64); out = np.zeros(32, dtype=np.uint64)
            for i in range(4):
                s = (sel >> (4 * i)) & 0xf; byte = (src >> (8 * (s & 7))) & 0xff
                if s & 8: byte = np.where(byte & 0x80, 0xff, 0)
                out |= byte << (8 * i)
            self.wr(wp, args[0], out); return
        if name == "movmatrix":   # .sync.trans.aligned.m8n8.b16: transpose an 8x8 b16 matrix held as lane l: row l/4, cols 2(l%4),2(l%4)+1
            a = self.rd(wp, args[1]); g = LANES // 4; tig = LANES % 4
            Mx = np.zeros((8, 8), dtype=np.uint64)
            for l in range(32): Mx[g[l], 2 * tig[l]] = a[l] & M16; Mx[g[l], 2 * tig[l] + 1] = (a[l] >> 16) & M16
            T = Mx.T; self.wr(wp, args[0], T[g, 2 * tig] | (T[g, 2 * tig + 1] << 16)); return
        # ---------- f16x2 arithmetic ----------
        if name != "cvt" and (parts[-1] == "f16x2" or (len(parts) > 1 and parts[1] == "f16x2")):
            srcs = [self.h2(self.rd(wp, a)) for a in args[1:]]
            f = {"add": lambda x, y: x + y, "sub": lambda x, y: x - y, "mul": lambda x, y: x * y, "max": np.maximum, "min": np.minimum}
            if name in f: lo = f[name](srcs[0][0], srcs[1][0]); hi = f[name](srcs[0][1], srcs[1][1])
            elif name == "fma": lo = srcs[0][0] * srcs[1][0] + srcs[2][0]; hi = srcs[0][1] * srcs[1][1] + srcs[2][1]
            elif name == "abs": lo = np.abs(srcs[0][0]); hi = np.abs(srcs[0][1])
            elif name == "neg": lo = -srcs[0][0]; hi = -srcs[0][1]
            else: raise NotImplementedError(ins)
            self.wr(wp, args[0], self.pack_h2(lo, hi)); return
        # ---------- conversions ----------
        if name == "cvt":
            dst_t, src_t = parts[-2], parts[-1]
            if dst_t == "e4m3x2" and src_t == "f16x2": self.wr(wp, args[0], self.f16x2_to_e4m3_pair(self.rd(wp, args[1])), 16); return
            if dst_t == "f16x2" and src_t == "e4m3x2": self.wr(wp, args[0], self.e4m3_pair_to_f16x2(self.rd(wp, args[1], 16))); return
            if dst_t == "f16" and src_t == "f32": self.wr(wp, args[0], f64_to_f16_bits(f32_bits_to_f64(self.rd(wp, args[1]))), 16); return
            if dst_t == "f32" and src_t == "f16": self.wr(wp, args[0], f64_to_f32_bits(f16_bits_to_f64(self.rd(wp, args[1], 16)))); return
            if dst_t in ("s32", "u32") and src_t == "f32":          # f32 -> integer with the PTX rounding modifier (rzi trunc, rni rint, rmi floor, rpi ceil), saturating
                v = f32_bits_to_f64(self.rd(wp, args[1]))
                if "ftz" in parts: v = ftz32(v)
                r = np.trunc(v) if "rzi" in parts else np.floor(v) if "rmi" in parts else np.ceil(v) if "rpi" in parts else np.rint(v)
                r = np.where(np.isnan(r), 0.0, r)
                if dst_t == "s32": r = np.clip(r, -2147483648.0, 2147483647.0).astype(np.int64).astype(np.uint64) & 0xffffffff
                else: r = np.clip(r, 0.0, 4294967295.0).astype(np.int64).astype(np.uint64)
                self.wr(wp, args[0], r); return
            if dst_t == "f32" and src_t in ("u32", "s32"):
                v = self.rd(wp, args[1]); v = self.signed(v, 32) if src_t == "s32" else v.astype(np.int64)
                self.wr(wp, args[0], f64_to_f32_bits(v.astype(np.float64))); return
            if dst_t == "f32" and src_t == "f32":
                v = f32_bits_to_f64(self.rd(wp, args[1]))
                if "ftz" in parts: v = ftz32(v)
                if "rmi" in parts: v = np.floor(v)
                elif "rpi" in parts: v = np.ceil(v)
                elif "rzi" in parts: v = np.trunc(v)
                elif "rni" in parts: v = np.rint(v)
                self.wr(wp, args[0], f64_to_f32_bits(v)); return
            if re.match(r"[usb](8|16|32|64)$", dst_t) and re.match(r"[usb](8|16|32|64)$", src_t):
                db, sb = self.width_of(dst_t), self.width_of(src_t); v = self.rd(wp, args[1], sb)
                if src_t.startswith("s"): v = self.signed(v, sb)
                else: v = v.astype(np.int64) if sb < 64 else v
                if db < sb:
                    v = v & ((1 << db) - 1)
                    if dst_t.startswith("s"): v = np.where(v & (1 << (db - 1)), v - (1 << db), v)
                regbits = 16 if db == 8 else db
                self.wr(wp, args[0], np.asarray(v).astype(np.int64).astype(np.uint64) & ((1 << regbits) - 1), regbits); return
            raise NotImplementedError(ins)
        # ---------- f32 scalar ops ----------
        if parts[-1] == "f32" and name not in ("tex", "sust", "suld"):
            srcs = [(-f32_bits_to_f64(self.rd(wp, a.strip()[1:])) if a.strip().startswith("-") else f32_bits_to_f64(self.rd(wp, a))) for a in args[1:]]   # "-%r" operands: negated (fp32 contraction sites, see fuse_mul_add)
            if "ftz" in parts: srcs = [ftz32(x) for x in srcs]
            if name == "min": v = np.fmin(srcs[0], srcs[1])
            elif name == "max": v = np.fmax(srcs[0], srcs[1])
            elif name == "abs": v = np.abs(srcs[0])
            elif name == "neg": v = -srcs[0]
            elif name == "ex2" and os.environ.get("EMU_EX2_MODE") == "halftable":   # 2026-09-06: audited half-domain table (Codex), exact argument check, finite halves only; any failure raises (fail closed)
                site = getattr(self.k, "ex2_half_sites", None); site = None if site is None else site.get(wp.pc - 1)
                if site is None: raise RuntimeError(f"ex2 half-table mode: statement {wp.pc - 1} has no unique half source ({ins})")
                act = wp.mask if pred is None else (wp.mask & pred); targs, tres = ex2_half_table(); h = (self.rd(wp, site) & 0xffff).astype(np.int64)
                if np.any(act & ((h & 0x7c00) == 0x7c00)): raise RuntimeError(f"ex2 half-table domain fault (nonfinite half operand) at statement {wp.pc - 1}")
                xb = f64_to_f32_bits(srcs[0]).astype(np.uint64)
                if np.any(act & (targs[h] != xb)): raise RuntimeError(f"ex2 half-table argument mismatch at statement {wp.pc - 1}: half {[hex(int(q)) for q in h[act][:4]]} arg {[hex(int(q)) for q in xb[act][:4]]}")
                v = f32_bits_to_f64(tres[h])
            elif name == "ex2": v = np.exp2(srcs[0].astype(np.float32)).astype(np.float32).astype(np.float64) if os.environ.get("EMU_EX2_MODE") == "exact" else ex2_sm120_f32(srcs[0], pos_unit=os.environ.get("EMU_EX2_POS", "unit") == "unit")
            elif name == "div": v = (srcs[0].astype(np.float32) / srcs[1].astype(np.float32)).astype(np.float64) if os.environ.get("EMU_DIV_MODE") == "exact" else (srcs[0].astype(np.float32) * rcp_rom_f32(srcs[1]).astype(np.float32)).astype(np.float64)
            elif name == "mul": v = (srcs[0].astype(np.float32) * srcs[1].astype(np.float32)).astype(np.float64)
            elif name == "add": v = (srcs[0].astype(np.float32) + srcs[1].astype(np.float32)).astype(np.float64)
            elif name == "sub": v = (srcs[0].astype(np.float32) - srcs[1].astype(np.float32)).astype(np.float64)
            elif name == "fma": v = (srcs[0] * srcs[1] + srcs[2]).astype(np.float32).astype(np.float64)
            elif name in ("rsqrt", "rcp"):
                x = srcs[0]; hb = x.astype(np.float16); exact = hb.astype(np.float64) == x
                table = self.rsq if name == "rsqrt" else self.rcp
                idx = np.clip(hb.view(np.uint16).astype(np.int64) - 1, 0, len(table) - 1)
                tv = table[idx].view(np.float16).astype(np.float64)
                fallback = (1.0 / np.sqrt(x)) if name == "rsqrt" else rcp_rom_f32(x)          # general f32 rcp: the sm_120 correction-ROM model
                if name == "rsqrt" and not exact.all(): self.unimpl[name + ".nonhalf"] = self.unimpl.get(name + ".nonhalf", 0) + int((~exact).sum())
                if name == "rcp" and os.environ.get("EMU_RCP_MODE") == "rom32": v = fallback.astype(np.float32).astype(np.float64)   # 2026-09-06: fp32 consumers (post block r37/r38, weight-sum and sigmoid reciprocals) must not take the half-rounded table
                elif name == "rcp" and os.environ.get("EMU_RCP_MODE") == "exact": v = (np.float32(1) / x.astype(np.float32)).astype(np.float64)
                else: v = np.where(exact, tv, fallback.astype(np.float32).astype(np.float64))
            elif name == "sqrt": v = np.sqrt(srcs[0].astype(np.float32)).astype(np.float64)
            else: raise NotImplementedError(ins)
            if "ftz" in parts: v = ftz32(v)
            self.wr(wp, args[0], f64_to_f32_bits(v)); return
        # ---------- memory ----------
        if name == "cp" and "bulk" not in parts:   # cp.async.ca/cg.shared.global [dst], [src], cp-size  (per-lane, completes at wait_group)
            if parts[1] == "async" and parts[2] in ("commit_group", "wait_group", "wait_all"): return   # copies are performed synchronously here
            mask = pred if pred is not None else wp.mask
            nb = int(args[2].strip(), 0)
            dst = self.addr(wp, args[0], 32); src = self.addr(wp, args[1], 64)
            for l in np.nonzero(mask)[0].tolist():
                buf, o = self.gfind(int(src[l])); self.shared[int(dst[l]):int(dst[l]) + nb] = buf[o:o + nb]
                if self.cplog is not None: self.cplog.append((wp.w, l, int(src[l]), int(dst[l]), nb))
            return
        if name == "cp":   # cp.async.bulk.shared::cta.global... [dst], [src], bytes, [mbar]
            l0 = int(np.argmax(wp.mask))      # the issuing lane: first active lane
            dst = int(self.addr(wp, args[0], 32)[l0]); src = int(self.addr(wp, args[1], 64)[l0]); nb = int(self.rd(wp, args[2])[l0])
            buf, o = self.gfind(src); self.shared[dst:dst + nb] = buf[o:o + nb]
            if self.cplog is not None: self.cplog.append((wp.w, l0, src, dst, nb))
            return
        if name == "tex":         # tex.2d.v4.f32.f32 {d0..d3}, [texref, {x, y}]: nearest texel of normalized coordinates, clamp (hypothesis)
            dsts = [x.strip() for x in args[0].strip("{}").split(",")]
            m = re.match(r"\[\s*(\S+?)\s*,\s*\{\s*(\S+?)\s*,\s*(\S+?)\s*\}\s*\]", ",".join(args[1:])); assert m, ins
            mask = pred if pred is not None else wp.mask; l0 = int(np.argmax(mask)); handle = int(self.rd(wp, m.group(1), 64)[l0])
            if handle not in TEXTURES: raise NotImplementedError(f"tex on unbound texture handle {handle:#x}: {ins}")
            img, W, H = TEXTURES[handle]; u = f32_bits_to_f64(self.rd(wp, m.group(2))); vv = f32_bits_to_f64(self.rd(wp, m.group(3)))
            if os.environ.get("EMU_TEX_LOG") and wp.w == 0:
                _W, _H = TEXTURES[handle][1], TEXTURES[handle][2]; _tx = u.astype(np.float32) * np.float32(_W) - np.float32(0.5); _ty = vv.astype(np.float32) * np.float32(_H) - np.float32(0.5)
                with open(os.environ["EMU_TEX_LOG"], "a") as f:
                    for l in range(4): f.write(f"pc {wp.pc-1} handle {handle:#x} lane {l} u {float(u[l]):.9f} v {float(vv[l]):.9f} tx {float(_tx[l]):.7f} ty {float(_ty[l]):.7f}\n")
            filt = os.environ.get("EMU_TEX_FILTER", "nearest")      # nearest | linear | q8t | q8r | q8rc | rtx  (2026-09-06 temporal-path hypotheses; rtx = fitted micro-probe rule); EMU_TEX_BLEND = f64 (exact then one f32 rounding) | lerp32
            if filt == "rtx":
                rs = tex_bilinear_rtx(img, W, H, u, vv)
                for ci, nm in enumerate(dsts): self.wr(wp, nm, np.where(mask, f64_to_f32_bits(rs[ci]), self.rd(wp, nm)))
                return
            if filt == "nearest":
                ix = np.clip(np.floor(u * W), 0, W - 1).astype(np.int64); iy = np.clip(np.floor(vv * H), 0, H - 1).astype(np.int64)
                ix = np.where(np.isnan(u), 0, ix); iy = np.where(np.isnan(vv), 0, iy)
                for ci, nm in enumerate(dsts):
                    val = f64_to_f32_bits(img[iy, ix, ci].astype(np.float64))
                    self.wr(wp, nm, np.where(mask, val, self.rd(wp, nm)))
                return
            # bilinear on the texel grid: t = u*W - 0.5 (computed in f32 like the hardware's coordinate path is assumed to), fraction quantised per mode
            f32 = np.float32; tx = (u.astype(f32) * f32(W) - f32(0.5)).astype(np.float64); ty = (vv.astype(f32) * f32(H) - f32(0.5)).astype(np.float64)
            tx = np.where(np.isnan(tx), 0.0, tx); ty = np.where(np.isnan(ty), 0.0, ty); fx0 = np.floor(tx); fy0 = np.floor(ty); fx = tx - fx0; fy = ty - fy0
            if filt == "q8t": fx = np.floor(fx * 256) / 256; fy = np.floor(fy * 256) / 256
            elif filt == "q8r": fx = np.rint(fx * 256) / 256; fy = np.rint(fy * 256) / 256
            elif filt == "q8rc": fx = np.minimum(np.rint(fx * 256), 255) / 256; fy = np.minimum(np.rint(fy * 256), 255) / 256   # hypothesis: rounded 8-bit fraction without carry into the texel index (saturates at 255/256)
            x0 = np.clip(fx0, 0, W - 1).astype(np.int64); x1 = np.clip(fx0 + 1, 0, W - 1).astype(np.int64); y0 = np.clip(fy0, 0, H - 1).astype(np.int64); y1 = np.clip(fy0 + 1, 0, H - 1).astype(np.int64)
            blend = os.environ.get("EMU_TEX_BLEND", "f64"); taplog = []
            for ci, nm in enumerate(dsts):
                a = img[y0, x0, ci].astype(np.float64); b = img[y0, x1, ci].astype(np.float64); d = img[y1, x0, ci].astype(np.float64); e = img[y1, x1, ci].astype(np.float64)
                if blend == "lerp32":
                    A, B, Dd, E, FX, FY = (v.astype(f32) for v in (a, b, d, e, fx, fy)); top = (A + (B - A) * FX).astype(f32); bot = (Dd + (E - Dd) * FX).astype(f32); r = (top + (bot - top) * FY).astype(f32).astype(np.float64)
                else: r = (a + (b - a) * fx) + ((d + (e - d) * fx) - (a + (b - a) * fx)) * fy
                if os.environ.get("EMU_TEX_HALF"): r = r.astype(np.float16).astype(np.float64)   # hypothesis: the texture unit returns the filtered half-format sample rounded to half (RNE)
                self.wr(wp, nm, np.where(mask, f64_to_f32_bits(r), self.rd(wp, nm))); taplog.append((a, b, d, e, r))
            if os.environ.get("EMU_TEX_LOG_ALL"):      # diagnostic: every lane's bilinear fetch (coordinates, quantised fraction, corner texels and result per channel)
                with open(os.environ["EMU_TEX_LOG_ALL"], "a") as f:
                    for l in np.nonzero(mask)[0]:
                        corners = " ".join(f"c{ci}[{float(a[l]):.9g},{float(b[l]):.9g},{float(d[l]):.9g},{float(e[l]):.9g}]->{float(r[l]):.9g}" for ci, (a, b, d, e, r) in enumerate(taplog))
                        f.write(f"tex pc {wp.pc-1} handle {handle:#x} warp {wp.w} lane {l} u {float(u[l]):.9f} v {float(vv[l]):.9f} tx {float(tx[l]):.7f} ty {float(ty[l]):.7f} x0 {int(x0[l])} y0 {int(y0[l])} fx {float(fx[l]):.7f} fy {float(fy[l]):.7f} {corners}\n")
            return
        if name == "sust":        # sust.p.2d.v4.b32.zero [surfref, {x, y}], {r, g, b, a}: formatted store into a half-float surface, round toward zero; out-of-range dropped
            m = re.match(r"\[\s*(\S+?)\s*,\s*\{\s*(\S+?)\s*,\s*(\S+?)\s*\}\s*\]\s*,\s*\{([^}]*)\}", ",".join(args)); assert m, ins
            mask = pred if pred is not None else wp.mask; l0 = int(np.argmax(mask)); handle = int(self.rd(wp, m.group(1), 64)[l0])
            if handle not in OUT_IMAGES: raise NotImplementedError(f"sust on unbound surface handle {handle:#x}: {ins}")
            rec = OUT_IMAGES[handle]; img = rec["img"]; H, W = img.shape[0], img.shape[1]
            x = self.signed(self.rd(wp, m.group(2)), 32); y = self.signed(self.rd(wp, m.group(3)), 32)
            conv = f64_to_f16_bits if os.environ.get("EMU_SUST_RNE") else f32_to_f16_rz_bits          # hypothesis switch: RNE instead of RZ at the surface store
            vals = [conv(f32_bits_to_f64(self.rd(wp, r.strip()))) for r in m.group(4).split(",")]
            ok = mask & (x >= 0) & (x < W) & (y >= 0) & (y < H)
            if os.environ.get("EMU_TEX_LOG_ALL"):
                raw = [self.rd(wp, r.strip()) for r in m.group(4).split(",")]
                with open(os.environ["EMU_TEX_LOG_ALL"], "a") as f:
                    for l in np.nonzero(ok)[0]: f.write(f"sust pc {wp.pc-1} warp {wp.w} lane {l} x {int(x[l])} y {int(y[l])} f32 {' '.join(f'{int(v[l]):#010x}' for v in raw)} f16 {' '.join(f'{int(v[l]) & 0xffff:#06x}' for v in vals)}\n")
            for l in np.nonzero(ok)[0]:
                for ci in range(4): img[int(y[l]), int(x[l]), ci] = int(vals[ci][l]) & 0xffff
                rec["written"][int(y[l]), int(x[l])] = True
            return
        if name in ("ld", "st", "red"):
            shared = any(p.startswith("shared") for p in parts); bits = self.width_of(parts[-1]); vec = 4 if ".v4" in op else 2 if ".v2" in op else 1
            mask = pred if pred is not None else np.ones(32, dtype=bool)
            if name == "ld":
                addr = self.addr(wp, args[-1], 32 if shared else 64)
                data = self.sload(addr, bits * vec // 8, mask) if shared else self.gload(addr, bits * vec // 8, mask)
                names = [x.strip() for x in args[0].strip("{}").split(",")]
                for i, nm in enumerate(names):
                    chunk = data[:, i * bits // 8:(i + 1) * bits // 8]
                    v = np.zeros(32, dtype=np.uint64)
                    for bi in range(bits // 8): v |= chunk[:, bi].astype(np.uint64) << (8 * bi)
                    if pred is not None: v = np.where(mask, v, self.rd(wp, nm, bits))
                    self.wr(wp, nm, v, bits)
                return
            addr = self.addr(wp, args[0], 32 if shared else 64)
            names = [x.strip() for x in args[1].strip("{}").split(",")]
            if bits == 128:
                tag, partsv = wp.R[names[0]]; vals = partsv; bits = 32; vec = 4
            else: vals = [self.rd(wp, nm, bits) for nm in names]
            data = np.zeros((32, bits * vec // 8), dtype=np.uint8)
            for i, v in enumerate(vals):
                for bi in range(bits // 8): data[:, i * bits // 8 + bi] = (v >> (8 * bi)) & 0xff
            if name == "red":
                assert "f16x2" in op and "add" in op, ins
                old = self.gload(addr, bits * vec // 8, mask)
                for i in range(vec):
                    o16 = old[:, 4 * i:4 * i + 4]; ov = o16[:, 0].astype(np.uint64) | (o16[:, 1].astype(np.uint64) << 8) | (o16[:, 2].astype(np.uint64) << 16) | (o16[:, 3].astype(np.uint64) << 24)
                    lo_o, hi_o = self.h2(ov); lo_n, hi_n = self.h2(vals[i]); nv = self.pack_h2(lo_o + lo_n, hi_o + hi_n)
                    for bi in range(4): data[:, 4 * i + bi] = (nv >> (8 * bi)) & 0xff
            if shared: self.sstore(addr, data, mask)
            else: self.gstore(addr, data, mask)
            return
        # ---------- integer ALU ----------
        if name in ("add", "sub", "mul", "mad", "and", "or", "xor", "shl", "shr", "div", "rem", "min", "max", "neg", "abs", "not"):
            t = parts[-1]; bits = self.width_of(t); sgn = t.startswith("s")
            if name == "not": self.wr(wp, args[0], ~self.rd(wp, args[1], bits), bits); return
            if name in ("mul", "mad") and len(parts) > 2 and parts[1] == "wide":
                a = self.rd(wp, args[1], bits); b = self.rd(wp, args[2], bits)
                a = self.signed(a, bits) if sgn else a.astype(np.int64); b = self.signed(b, bits) if sgn else b.astype(np.int64)
                v = a * b
                if name == "mad":
                    c = self.rd(wp, args[3], bits * 2); c = self.signed(c, bits * 2) if sgn else c.astype(np.int64); v = v + c
                self.wr(wp, args[0], v.astype(np.int64).astype(np.uint64), bits * 2); return
            a = self.rd(wp, args[1], bits)
            if name in ("neg", "abs"):
                a = self.signed(a, bits); v = -a if name == "neg" else np.abs(a); self.wr(wp, args[0], v.astype(np.uint64), bits); return
            b = self.rd(wp, args[2], bits)
            if name in ("shl", "shr"):
                sh = b.astype(np.int64)
                if name == "shl": v = np.where(sh < bits, a.astype(np.uint64) << sh.astype(np.uint64), 0)
                elif sgn: v = (self.signed(a, bits) >> np.minimum(sh, bits - 1)).astype(np.int64).astype(np.uint64)
                else: v = np.where(sh < bits, a >> sh.astype(np.uint64), 0)
                self.wr(wp, args[0], v, bits); return
            a2 = self.signed(a, bits) if sgn else a.astype(np.int64); b2 = self.signed(b, bits) if sgn else b.astype(np.int64)
            if name == "add": v = a2 + b2
            elif name == "sub": v = a2 - b2
            elif name == "mul": v = (a2 * b2) >> (bits if (len(parts) > 2 and parts[1] == "hi") else 0)
            elif name == "mad":
                c = self.rd(wp, args[3], bits); c2 = self.signed(c, bits) if sgn else c.astype(np.int64)
                v = ((a2 * b2) >> (bits if (len(parts) > 2 and parts[1] == "hi") else 0)) + c2
            elif name == "and": v = a2 & b2
            elif name == "or": v = a2 | b2
            elif name == "xor": v = a2 ^ b2
            elif name == "min": v = np.minimum(a2, b2)
            elif name == "max": v = np.maximum(a2, b2)
            elif name == "div":
                q = np.abs(a2) // np.where(b2 == 0, 1, np.abs(b2)); v = np.where((a2 < 0) != (b2 < 0), -q, q)
            elif name == "rem":
                q = np.abs(a2) // np.where(b2 == 0, 1, np.abs(b2)); q = np.where((a2 < 0) != (b2 < 0), -q, q); v = a2 - q * b2
            self.wr(wp, args[0], np.asarray(v).astype(np.int64).astype(np.uint64), bits); return
        raise NotImplementedError(ins)

def fuse_mul_add(kernel):
    """ptxas contracts a single-use `mul.f16x2` product feeding an `add.f16x2` into `fma.rn.f16x2` (product
    unrounded). When both add operands are products the FIRST one is contracted (ViT QKV sum-of-squares: unfused
    7/20480 and 16/20480 q/k mismatches, first-operand 0/0, second-operand 12/13); when only the second operand is
    a product it is contracted (dec_input_upsample: first-only 4211 mismatches, first-else-second 0). EMU_FUSE=
    none|first|second|both (default both = first if possible, else second)."""
    ins = kernel.instrs; uses = {}
    for s in ins:
        body = s.split(" ", 1)[1] if " " in s else ""
        ops = [x.strip() for x in re.split(r",(?![^{]*\})", body)]
        if s.startswith(("st.", "red.", "cp.")): srcs = ops                      # stores: every operand is a source
        elif s.startswith("mov") and ops and ops[0].startswith("{"): srcs = ops[1:]
        else: srcs = ops[1:]
        for o in srcs:
            for r in re.findall(r"%r\d+", o): uses[r] = uses.get(r, 0) + 1
    defs = {}
    for i, s in enumerate(ins):
        if s.startswith("mul.f16x2"):
            d, a, b = [x.strip() for x in s.split(" ", 1)[1].split(",")]; defs[d] = (i, a, b)
    fused = 0; sites_env = os.environ.get("EMU_FUSE_SITES")          # "i" (mode rule), "i:f" (contract the first operand), "i:s" (second); unlisted sites stay unfused
    allowed = None if sites_env is None else {int(x.split(":")[0]): (x.split(":")[1] if ":" in x else None) for x in sites_env.split(",") if x}
    kernel.fusion_sites = []
    for i, s in enumerate(ins):
        if s.startswith("add.f16x2"):
            d, p, c = [x.strip() for x in s.split(" ", 1)[1].split(",")]
            mode = os.environ.get("EMU_FUSE", "both")
            if (p in defs and uses.get(p, 0) == 1) or (c in defs and uses.get(c, 0) == 1): kernel.fusion_sites.append((i, "first" if (p in defs and uses.get(p, 0) == 1) else "second"))
            if allowed is not None and i not in allowed: continue          # EMU_FUSE_SITES: contract only the listed add statements
            if allowed is not None and allowed[i] is not None: mode = {"f": "first", "s": "second"}[allowed[i]]
            if mode in ("first", "both") and p in defs and uses.get(p, 0) == 1:
                j, a, b = defs[p]; ins[i] = f"fma.rn.f16x2 {d}, {a}, {b}, {c}"; ins[j] = f"mov.b32 {p}, {p}"; fused += 1
            elif mode in ("second", "both") and c in defs and uses.get(c, 0) == 1:
                j, a, b = defs[c]; ins[i] = f"fma.rn.f16x2 {d}, {a}, {b}, {p}"; ins[j] = f"mov.b32 {c}, {c}"; fused += 1
    # fp32 contraction sites (2026-09-06, temporal post block): ONLY when explicitly listed in EMU_FUSE_SITES as "i:f"/"i:s"/"i:d"; unlisted sites are never
    # touched, so accepted kernels are unchanged. Statement i is an add/sub.ftz.f32. f/s: the first/second operand is a mul.ftz.f32 product contracted into
    # the add (a*b + c -> fma(a,b,c); a*b - c -> fma(a,b,-c); c - a*b -> fma(-a,b,c)); the mul statement is KEPT because the product may have other consumers
    # (the post block's dimension*uv feeds both the floor and the centre subtraction: SASS a290/a2e0), an unused product is dead code. d: the first/second
    # operand is a single-use div.approx quotient, contracted as MUFU.RCP + FFMA (SASS a4f0/a530): q = a/b; d = q + c -> rcp q, b; fma d, a, q, c.
    # (v1 of this pass replaced the mul with a neg of the other operand, which read that operand BEFORE its definition when the mul preceded it -
    # the weights-f census result of 1,165,452 mismatches was that defect, not evidence against the contraction.)
    if allowed:
        defs32 = {}; divs32 = {}
        for i, s_ in enumerate(ins):
            m = re.match(r"mul\.ftz\.f32\s+(%r\d+),\s*(\S+),\s*(\S+)$", s_.strip().rstrip(";"))
            if m: defs32[m.group(1)] = (i, m.group(2), m.group(3))
            m = re.match(r"div\.approx\.ftz\.f32\s+(%r\d+),\s*(\S+),\s*(\S+)$", s_.strip().rstrip(";"))
            if m: divs32[m.group(1)] = (i, m.group(2), m.group(3))
        for i, mode in allowed.items():
            if i >= len(ins): continue
            m = re.match(r"(add|sub)\.ftz\.f32\s+(%r\d+),\s*(\S+),\s*(\S+)$", ins[i].strip().rstrip(";"))
            if not m or mode not in ("f", "s", "d"): continue
            op, d, x, y = m.groups()
            if mode == "d":
                if op != "add": continue
                q, other = (x, y) if x in divs32 else (y, x)
                if q not in divs32 or uses.get(q, 0) != 1: continue
                j, a, b = divs32[q]; ins[j] = f"rcp.approx.ftz.f32 {q}, {b}"; ins[i] = f"fma.rn.ftz.f32 {d}, {a}, {q}, {other}"
            else:
                prod = x if mode == "f" else y; other = y if mode == "f" else x
                if prod not in defs32: continue
                j, a, b = defs32[prod]
                if op == "add": ins[i] = f"fma.rn.ftz.f32 {d}, {a}, {b}, {other}"
                elif mode == "f": ins[i] = f"fma.rn.ftz.f32 {d}, {a}, {b}, -{other}"          # a*b - c
                else: ins[i] = f"fma.rn.ftz.f32 {d}, -{a}, {b}, {other}"                    # c - a*b
            fused += 1; kernel.fusion_sites.append((i, "f32:" + mode))
    kernel.fused_mul_add = fused
    return kernel

EX2_HALF_TABLE_SHA256 = "f9ae941e60868e72489db582fe725bdb401e033fce8694ecd35dd0c3329ba3cf"   # Codex build/nr_ex2_half/table-v1 (efc8df3): 65536 x {exact scaled-f32 argument bits, RTX ex2.approx.ftz.f32 result bits}, indexed by the ORIGINAL half bits
EX2_HALF_SCALE = "0fBFB8AA3B"
_EX2_HALF = {}
def ex2_half_table():
    """the audited half-domain EX2 table (EMU_EX2_HALF_TABLE=path); size and sha pinned; returns (args_u32[65536], results_u32[65536])"""
    path = os.environ["EMU_EX2_HALF_TABLE"]
    if path not in _EX2_HALF:
        import hashlib; blob = open(path, "rb").read()
        if len(blob) != 524288 or hashlib.sha256(blob).hexdigest() != EX2_HALF_TABLE_SHA256: raise RuntimeError("ex2 half-domain table: size/sha mismatch (pinned %s)" % EX2_HALF_TABLE_SHA256)
        t = np.frombuffer(blob, dtype="<u4").reshape(65536, 2); _EX2_HALF[path] = (t[:, 0].astype(np.uint64), t[:, 1].astype(np.uint64))
    return _EX2_HALF[path]
def ex2_half_sites(kernel):
    """static def-use: for every ex2.approx.ftz.f32 statement whose argument is the UNIQUE definition `mul.ftz.f32 x, y, 0fBFB8AA3B` (either operand
    order) of a UNIQUE `cvt.f32.f16 y, %rsN`, map statement index -> the half register %rsN. Sites that do not match are absent (the consumer fails closed)."""
    ins = kernel.instrs; defs = {}
    for i, s in enumerate(ins):
        m = re.match(r"(?:@!?%p\d+\s+)?([a-z][\w.]*)\s+(%\w+)\s*,", s.strip())
        if m and not m.group(1).startswith(("st", "bra", "setp", "bar", "red", "atom")): defs.setdefault(m.group(2), []).append(i)
    sites = {}
    for i, s in enumerate(ins):
        m = re.match(r"ex2\.approx\.ftz\.f32\s+(%r\d+),\s*(%r\d+)$", s.strip().rstrip(";"))
        if not m: continue
        x = m.group(2); dx = defs.get(x, [])
        if len(dx) != 1: continue
        mm = re.match(r"mul\.ftz\.f32\s+%r\d+,\s*(\S+),\s*(\S+)$", ins[dx[0]].strip().rstrip(";"))
        if not mm: continue
        ops = [mm.group(1), mm.group(2)]
        if EX2_HALF_SCALE not in ops: continue
        y = ops[0] if ops[1] == EX2_HALF_SCALE else ops[1]; dy = defs.get(y, [])
        if len(dy) != 1: continue
        mc = re.match(r"cvt\.f32\.f16\s+%r\d+,\s*(%rs\d+)$", ins[dy[0]].strip().rstrip(";"))
        if not mc: continue
        sites[i] = mc.group(1)
    return sites

RTX_TEX_NAN_BITS = 0x7fffe000   # canonical NaN returned by the RTX texture unit (Codex's sampler edge oracle, 2026-09-07 00:01 UTC, 4 fresh processes)
def _rtx_nonfinite(*wt):
    """RTX texture-unit rule for exceptional operands over the NONZERO-WEIGHT taps (Codex's edge oracle, 22 cases x 49 queries, 4 processes identical):
    any NaN tap -> canonical NaN 0x7fffe000 (payload/sign/quiet/signaling irrelevant); +inf and -inf together -> the same canonical NaN; a lone infinity
    (any multiplicity) -> that infinity; zero-weight taps never contaminate the result. Returns (override_mask, override_values) in float64 carrying f32 values."""
    n = wt[0][1].shape[0]; anynan = np.zeros(n, dtype=bool); pinf = np.zeros(n, dtype=bool); ninf = np.zeros(n, dtype=bool)
    for w, t in wt:
        nz = w != 0; anynan |= nz & np.isnan(t); pinf |= nz & (t == np.inf); ninf |= nz & (t == -np.inf)
    nanv = float(np.array([RTX_TEX_NAN_BITS], dtype=np.uint32).view(np.float32)[0])
    mask = anynan | pinf | ninf
    val = np.where(anynan | (pinf & ninf), nanv, np.where(pinf, np.inf, np.where(ninf, -np.inf, 0.0)))
    return mask, val
def _rtx_accumulate(*wt):
    """fixed-point bilinear accumulation of up to four (weight, texel) lane-arrays (weights integer 0..256 summing to 256), see tex_bilinear_rtx;
    exceptional operands (NaN/Inf among the nonzero-weight taps) take the _rtx_nonfinite rule first - the finite path below is unchanged"""
    xmask, xval = _rtx_nonfinite(*wt)
    wt = tuple((w, np.where(xmask | ~np.isfinite(t), 0.0, t)) for w, t in wt)   # exceptional lanes run the finite path on zeros (overridden below); zero-weight nonfinite taps never contribute
    n = wt[0][1].shape[0]; mx = np.zeros(n); allneg0 = np.ones(n, dtype=bool); anynz = np.zeros(n, dtype=bool)
    for w, t in wt:
        nz = w != 0; mx = np.where(nz, np.maximum(mx, np.abs(t)), mx); allneg0 = allneg0 & (~nz | ((t == 0) & (np.signbit(t)))); anynz = anynz | nz
    E = np.where(mx > 0, np.floor(np.log2(np.where(mx > 0, mx, 1.0))), 0.0); grid = np.exp2(E - 14.0)
    acc = np.zeros(n)
    for w, t in wt:
        nz = w != 0; q = np.trunc(t / grid) * grid; acc = np.where(nz, acc + w * q, acc)     # exact in float64 (|w*q| < 2^24 * grid, sums < 2^26 * grid)
    val = acc / 256.0; sign = np.where(val < 0, -1.0, 1.0); a = np.abs(val)
    ea = np.where(a > 0, np.floor(np.log2(np.where(a > 0, a, 1.0))), 0.0); hg = np.exp2(np.maximum(ea - 10.0, -24.0))   # half grid (subnormals at 2^-24)
    r = np.floor(a / hg + 0.5) * hg                                                     # ties away from zero
    res = np.where(a > 0, sign * r, np.where(allneg0 & anynz, -0.0, 0.0))
    res = np.where(xmask, xval, res)
    return res.astype(np.float32).astype(np.float64)
def tex_bilinear_rtx(img, W, H, u, v):
    """RTX texture-unit bilinear rule fitted 2026-09-06 to Codex's 4 x 303-query step-texture micro-probe (build/nr_tex_micro, 4848/4848 values):
    normalised coordinate truncated to 21 fractional bits, texel coordinate = U*dim/2^21 - 0.5 in exact integer arithmetic, 8-bit fraction rounded
    half up (0..256), clamp-to-edge; weights: wd = (fx*fy + 128) >> 8, wb = fx - wd, wc = fy - wd, wa = 256 - fx - fy + wd (sum 256, one rounding);
    accumulation wa*a, then fma(wb, b), fma(wc, c), fma(wd, d), scaled by 1/256 - the accumulation order/precision is a HYPOTHESIS (the 0/1 probe
    texture cannot see it); img is float32 (H, W, C). u, v: float arrays (32 lanes)."""
    u = np.where(np.isnan(u), 0.0, np.clip(u.astype(np.float64), -1.0, 2.0)); v = np.where(np.isnan(v), 0.0, np.clip(v.astype(np.float64), -1.0, 2.0))
    U = np.floor(u * 2097152.0).astype(np.int64); V = np.floor(v * 2097152.0).astype(np.int64)
    TX = U * np.int64(W) - np.int64(1 << 20); TY = V * np.int64(H) - np.int64(1 << 20)
    ix = TX >> 21; iy = TY >> 21; fx = ((TX - (ix << 21)) * 256 + (1 << 20)) >> 21; fy = ((TY - (iy << 21)) * 256 + (1 << 20)) >> 21
    x0 = np.clip(ix, 0, W - 1); x1 = np.clip(ix + 1, 0, W - 1); y0 = np.clip(iy, 0, H - 1); y1 = np.clip(iy + 1, 0, H - 1)
    wd = (fx * fy + 128) >> 8; wb = fx - wd; wc = fy - wd; wa = 256 - fx - fy + wd
    out = []
    for ci in range(img.shape[2]):
        a = img[y0, x0, ci].astype(np.float64); b = img[y0, x1, ci].astype(np.float64); c = img[y1, x0, ci].astype(np.float64); d = img[y1, x1, ci].astype(np.float64)
        # accumulation fitted to Codex's finite-half capture (build/nr_tex_finite_half, 4 runs x 5524 values, 0 unequal): the nonzero-weight texels are
        # aligned to the exponent of the largest one on a 15-significant-bit grid (2^(E-14)) and truncated toward zero, the weighted sum is exact, and the
        # result is rounded to half with ties away from zero (then returned as f32). Zero-weight taps do not contribute; an all-negative-zero set returns -0
        # (history copy 156 evidence); cancellation to exactly zero returns +0 (hypothesis).
        out.append(_rtx_accumulate((wa, a), (wb, b), (wc, c), (wd, d)))
    return out

_KCACHE = {}
def prepared_kernel(ptx_text, entry, fuse=True):
    """parse + contraction pre-pass + post-dominator analysis once per (PTX, entry, fusion policy) per process"""
    key = (hash(ptx_text), len(ptx_text), entry, os.environ.get("EMU_FUSE", "both") if fuse else "none", os.environ.get("EMU_FUSE_SITES"))
    k = _KCACHE.get(key)
    if k is None:
        k = Kernel(ptx_text, entry)
        if fuse and os.environ.get("EMU_FUSE", "both") != "none": fuse_mul_add(k)
        k.ex2_half_sites = ex2_half_sites(k); k.ipdom = compute_ipdom(k); _KCACHE[key] = k
    return k

def run_cta(ptx_text, entry, params, ctaid, block, grid, gmem, tables, shared_bytes=1 << 16, fuse=True):
    k = prepared_kernel(ptx_text, entry, fuse)
    cta = CTA(k, params, ctaid, block, grid, gmem, shared_bytes, tables)
    cta.run()
    if os.environ.get("EMU_TIE_CENSUS"):
        with open(os.environ["EMU_TIE_CENSUS"], "a") as f: f.write(f"{ctaid} ties {getattr(cta, 'tie_count', 0)} binade_ties {getattr(cta, 'binade_tie_count', 0)}\n")
    if os.environ.get("EMU_E4M3_CENSUS"):
        with open(os.environ["EMU_E4M3_CENSUS"], "a") as f: f.write(f"{ctaid} {getattr(cta, 'e4m3_census', {})}\n")
    if os.environ.get("EMU_QMMA_SPAN_CENSUS"):
        with open(os.environ["EMU_QMMA_SPAN_CENSUS"], "a") as f: f.write(f"{ctaid} {getattr(cta, 'span_census', {})}\n")
    if cta.cplog is not None:
        with open(os.environ["EMU_LOG_CP"], "a") as f:
            for w, l, src, dst, nb in cta.cplog: f.write(f"{ctaid[0]} {ctaid[1]} {ctaid[2]} {w} {l} {src} {dst} {nb}\n")
    return cta
