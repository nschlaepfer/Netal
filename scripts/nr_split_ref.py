#!/usr/bin/env python3
"""Numpy reference for the NR split-Swin 16-head family (blocks 23-30, 40-47). Every formula here is checked against
the RTX oracle (see nr-split-fit.py); nothing is assumed from the ViT family without a test."""
import os, sys, struct, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nr_fp8 import E4M3, weight_index, encode_e4m3
from nr_vit_ref import tensor, frag_a_index_n, frag_d_index_n, qmma_chain, cubic_silu_e4m3
import importlib.util
_spec = importlib.util.spec_from_file_location("emu", os.path.join(os.path.dirname(os.path.abspath(__file__)), "nr-vit-emulate-launch.py"))
emu = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(emu)
pool_state = emu.pool_state
C = 512
def slot_channel(j):
    """channel held in k-slot j when a D-fragment tensor (4 n-tiles) is re-read as an A operand."""
    j = np.asarray(j); return 2 * ((j % 16) // 4) + j % 2 + 8 * ((j % 4) // 2) + 16 * (j // 16)
def tile_tokens(buf, tile, channels=C):
    """(16, channels) float32 of one 16-token tile in the A-identity 2D tin layout (tile = 16*channels bytes)."""
    base = tile * 16 * channels; t = np.arange(16)[:, None]; c = np.arange(channels)[None, :]
    return E4M3[np.frombuffer(bytes(buf), dtype=np.uint8)[base + frag_a_index_n(t, c, channels)]].astype(np.float32)
def w_group(blob, base, K, N, group_bytes=2048, group_cols=64):
    """dense (K, N) E4M3 weights from [k-step][N/group_cols groups of group_bytes][weight_index(kk, n%group_cols)]."""
    Wm = np.zeros((K, N), dtype=np.float32); kk = np.arange(32)[:, None]; idx = weight_index(kk, np.arange(group_cols)[None, :])
    per_step = (N // group_cols) * group_bytes
    for s in range(K // 32):
        for g in range(N // group_cols):
            o = base + s * per_step + g * group_bytes; Wm[32 * s:32 * s + 32, group_cols * g:group_cols * g + group_cols] = E4M3[blob[o:o + group_bytes][idx]]
    return Wm
class SplitFfwd:
    """cc_split_swin_16h_ffwd_*: X(512) -> H1 = X W1 (512, per 64-col group g) -> H2_g = e4m3(H1_g) W2_g (256)
    -> a = e4m3(cubic_silu(H2_g)) -> OUT_g = a W3_g (64). Blob: W1 [16 steps][8 x 2048], then per group g:
    W2_g at 262144 + 16384 g ([2 steps][8 x 1024 (32 cols)]), W3_g at 393216 + 16384 g ([8 steps][1 x 2048])."""
    def __init__(self, blob):
        self.W1 = w_group(blob, 0, 512, 512)
        self.W2 = [w_group(blob, 262144 + 16384 * g, 64, 256, group_bytes=1024, group_cols=32) for g in range(8)]
        self.W3 = [w_group(blob, 393216 + 16384 * g, 256, 64) for g in range(8)]
    def __call__(self, X):
        """X: (T, 512) float32 E4M3 values -> (T, 512) uint8 E4M3 output, column n = W3 column (group g*64 + n3)."""
        H1 = qmma_chain(X, self.W1)                       # (T, 512) half
        out = np.zeros((X.shape[0], 512), dtype=np.uint8)
        for g in range(8):
            h1 = E4M3[encode_e4m3(H1[:, 64 * g:64 * g + 64].astype(np.float32))].astype(np.float32)      # e4m3 of the half D fragments
            # A slots: k-step s holds h1 column 32 s + slot_channel(j) in slot j
            A2 = np.concatenate([h1[:, 32 * s + slot_channel(np.arange(32))] for s in range(2)], axis=1)
            H2 = qmma_chain(A2, self.W2[g])              # (T, 256) half
            a = E4M3[cubic_silu_e4m3(H2)].astype(np.float32)
            A3 = np.concatenate([a[:, 32 * s + slot_channel(np.arange(32))] for s in range(8)], axis=1)
            O = qmma_chain(A3, self.W3[g])
            out[:, 64 * g:64 * g + 64] = encode_e4m3(O.astype(np.float32))
        return out

CFD = (lambda n: (n & 1) + ((n & 6) << 1) + ((n & 8) >> 2) + (n & 16) + (n & ~31))(np.arange(512))
def store_cfd(cols):
    """(T, 512) values indexed by MMA column n -> (T, 512) values indexed by channel position (position cfd(n) holds column n)."""
    out = np.empty_like(cols); out[:, CFD] = cols; return out
class SplitConv:
    """Conv2d1x1 512->512 with residual folding (ffwd_proj / proj): Y[t, n] = e4m3(chain(W[:, n] . X[t, :], C0 = half(res_pos[t, cfd(n)] * scale[n])))
    stored at position cfd(n). X and res are read at channel positions (no permutation). Blob: [16 steps][8 x 2048] then 512 half scales."""
    def __init__(self, blob, ksplit=1):
        self.W = w_group(blob, 0, 512, 512); self.scale = np.frombuffer(bytes(blob[262144:262144 + 1024]), dtype=np.float16).astype(np.float32); self.ksplit = ksplit
    def __call__(self, Xpos, Rpos):
        C0 = (Rpos[:, CFD] * self.scale[None, :]).astype(np.float16)     # half(res * scale), res already E4M3 values (exact in half)
        if self.ksplit == 1: Y = qmma_chain(Xpos, self.W, C0=C0)
        else:
            parts = [qmma_chain(Xpos, self.W, C0=(C0 if p == 0 else None), steps=range(p * 16 // self.ksplit, (p + 1) * 16 // self.ksplit)) for p in range(self.ksplit)]
            acc = parts[0].astype(np.float32)
            for p in parts[1:]: acc = (acc + p.astype(np.float32)).astype(np.float16).astype(np.float32)
            Y = acc.astype(np.float16)
        return store_cfd(encode_e4m3(Y.astype(np.float32)))

# ---------------------------------------------------------------- attention (cc_split_swin_16h_qkv_512_chained)
def h16(x): return np.asarray(x, dtype=np.float32).astype(np.float16)
def f16_bits(x): return np.asarray(x, dtype=np.float16).view(np.uint16)
def bits_f16(b): return np.asarray(b, dtype=np.uint16).view(np.float16)
def fma16(a, b, c):
    """half fma: exact product+sum in float64 then one rounding to half (a, b, c are halves as float32 values)."""
    return (np.asarray(a, dtype=np.float64) * np.asarray(b, dtype=np.float64) + np.asarray(c, dtype=np.float64)).astype(np.float16)
def load_tables(root):
    rs = np.fromfile(os.path.join(root, "build/nr_vit/tables/rtx-rsqrt-half.bin"), dtype=np.uint16); rc = np.fromfile(os.path.join(root, "build/nr_vit/tables/rtx-rcp-half.bin"), dtype=np.uint16)
    return rs, rc
def table_lookup(tab, hval):
    """RTX rsqrt/rcp.approx on a positive half: table index = half bits - 1 (one entry per positive finite half)."""
    b = f16_bits(hval).astype(np.int64); return bits_f16(tab[b - 1])
EPS_H = bits_f16(np.array([0x0410], dtype=np.uint16))[0]     # 6.198883e-05, the max() floor before rsqrt/rcp
def sumsq32_k(t):
    """Sum of squares over 32 dims for K rows as the kernel does it. t: (T, 32) halves as float32 (dims in n-tile order
    d = 8*nt + 2*m + e; lane m holds dims (2m, 2m+1) of each n-tile). Tree per lane m and parity e:
    P0 = fma(x[nt4? see code]) -- pairs (nt0+nt2), (nt1+nt3) with first-operand fusion, then (nt1nt3)+(nt0nt2), quad xor2, xor1, then e0+e1."""
    T = t.shape[0]; x = t.reshape(T, 4, 4, 2)          # [nt][m][e]
    x64 = x.astype(np.float64)
    a = (x64[:, 0] * x64[:, 0] + h16(x[:, 2] * x[:, 2]).astype(np.float64)).astype(np.float16).astype(np.float32)   # fma(nt0,nt0, half(nt2^2)), one rounding
    b = (x64[:, 1] * x64[:, 1] + h16(x[:, 3] * x[:, 3]).astype(np.float64)).astype(np.float16).astype(np.float32)   # fma(nt1,nt1, half(nt3^2))
    s = h16(b + a).astype(np.float32)                   # add %r801 = %r796(nt1+nt3) + %r797(nt0+nt2): (T, m, e)
    s = h16(s + s[:, [2, 3, 0, 1], :]).astype(np.float32)   # bfly xor 2
    s = h16(s + s[:, [1, 0, 3, 2], :]).astype(np.float32)   # bfly xor 1
    s = h16(s[:, :, 0] + s[:, :, 1])                    # lo + hi (all lanes now equal)
    return np.maximum(s[:, 0], EPS_H)
class SplitAttention:
    """One window (64 tokens as 4 m-tiles of 16 slots), one head. Weights: QKV blob [16 steps][24 groups x 2048] (head h: cols 96h + [q 0-31 | k 32-63 | v 64-95]),
    bias [16][8192 B] at 786432 in D-fragment chunk order, f32 head scales at 917504."""
    def __init__(self, blob, tables, head):
        self.h = head; Wall = w_group(blob, 0, 512, 1536); self.W = Wall[:, 96 * head: 96 * head + 96]
        self.bias = self.bias_table(blob, head); self.rs, self.rc = tables
        self.scale = np.frombuffer(bytes(blob[917504 + 4 * head: 917508 + 4 * head]), dtype=np.float32)[0]
    @staticmethod
    def bias_table(blob, head):
        raw = np.frombuffer(bytes(blob[786432 + 8192 * head: 786432 + 8192 * (head + 1)]), dtype=np.float16)
        B = np.zeros((64, 64), dtype=np.float32)
        for q in range(64):
            mt, g = q // 16, q % 16
            for k in range(64):
                nt, m, e = k // 8, (k % 8) // 2, k % 2
                chunk = 4 * mt + nt // 2; lane = 4 * (g % 8) + m; half = (nt % 2) * 4 + (g // 8) * 2 + e
                B[q, k] = raw[chunk * 256 + lane * 8 + half]
        return B
    def __call__(self, X, quad_order=(0, 1, 2, 3), q_mode="inv_then_scale"):
        """X: (64, 512) E4M3 values of the window tokens (m-tile-major). Returns (64, 32) uint8 E4M3 output columns d."""
        t = qmma_chain(X, self.W).astype(np.float32)            # (64, 96) half
        tq, tk, tv = t[:, 0:32], t[:, 32:64], t[:, 64:96]
        inv_k = table_lookup(self.rs, sumsq32_k(tk)).astype(np.float32)
        inv_q = table_lookup(self.rs, sumsq32_k(tq)).astype(np.float32)
        k = E4M3[encode_e4m3(h16(tk * inv_k[:, None]).astype(np.float32))]
        hs = h16(self.scale).astype(np.float32)
        if q_mode == "inv_then_scale": q = E4M3[encode_e4m3(h16(h16(tq * inv_q[:, None]).astype(np.float32) * hs).astype(np.float32))]
        else: q = E4M3[encode_e4m3(h16(tq * h16(inv_q * hs).astype(np.float32)[:, None]).astype(np.float32))]
        v = E4M3[encode_e4m3(tv)].astype(np.float32)
        # scores: fp32 chain over dims in slot order slot_channel(j), starting from the bias (as f32), one rounding to half
        sl = slot_channel(np.arange(32)); qs = q[:, sl].astype(np.float32); ks = k[:, sl].astype(np.float32)
        acc = self.bias.astype(np.float32).copy()
        for j in range(32): acc = acc + np.outer(qs[:, j], ks[:, j])
        s = acc.astype(np.float16).astype(np.float32)
        c29, c30, c31, c32 = [h16(np.float32(np.frombuffer(np.array([v], dtype=np.uint32).tobytes(), dtype=np.float32)[0])).astype(np.float32) for v in (1027077105, 1067877303, 1065615360, 1070129152)]
        vv = np.minimum(np.maximum(fma16(s, c29, c30).astype(np.float32), c31), c32)
        e = bits_f16(((f16_bits(vv).astype(np.uint32) << 5) + 0x8000) & 0xffff).astype(np.float32)   # (64 q, 64 k)
        # row sums: per lane m (keys 8nt+2m+e): tree8 over nt in the kernel's order, then quad lanes, then e
        E = e.reshape(64, 8, 4, 2)                                    # [q][nt][m][e]
        def add(a, b): return h16(a + b).astype(np.float32)
        p = add(add(add(add(E[:, 0], E[:, 1]), add(E[:, 2], E[:, 3])), add(E[:, 4], E[:, 5])), add(E[:, 6], E[:, 7]))   # (64, m, e)
        o = quad_order; qsum = add(add(add(p[:, o[0]], p[:, o[1]]), p[:, o[2]]), p[:, o[3]])                                # (64, e)
        rowsum = np.maximum(add(qsum[:, 0], qsum[:, 1]), EPS_H)
        inv = table_lookup(self.rc, rowsum).astype(np.float32)
        P = E4M3[encode_e4m3(h16(e * inv[:, None]).astype(np.float32))].astype(np.float32)                                  # (64 q, 64 k)
        # PV: two 32-key steps, keys within a step in slot order, C0 = 0
        out = np.zeros((64, 32), dtype=np.float16)
        for step in range(2):
            acc = out.astype(np.float32); keys = 32 * step + sl
            for j in range(32): acc = acc + np.outer(P[:, keys[j]], v[keys[j], :])
            out = acc.astype(np.float16)
        return encode_e4m3(out.astype(np.float32))

# ---------------------------------------------------------------- the "view" layout (block-23 input, block-47 output)
VIEW_BYTES = 16 * 69120
def view_index(r, c, ch):
    """Byte offset of token (row r, col c) channel ch in the 2x2 space-to-depth view tensor used at the family's
    boundaries: [cg = ch/32][R = r/2 + 18*((ch%32)/16)][C = c/2 + 30*(r%2)][byte = ch%16 + 16*(c%2)] (derived from the
    emulator's per-lane copy log of launch 25, verified against the oracle on launches 25, 26, 132)."""
    r = np.asarray(r); c = np.asarray(c); ch = np.asarray(ch)
    return (ch // 32) * 69120 + (r // 2 + 18 * ((ch % 32) // 16)) * 1920 + (c // 2 + 30 * (r % 2)) * 32 + (ch % 16) + 16 * (c % 2)
def view_tokens(view, tile):
    """(16, 512) E4M3 values of spatial tile `tile` (15 tiles per row, row-major 4x4 slots) gathered from a view tensor."""
    tr, tc = tile // 15, tile % 15; g = np.arange(16)[:, None]; ch = np.arange(512)[None, :]
    return E4M3[np.asarray(view)[view_index(4 * tr + g // 4, 4 * tc + g % 4, ch)]].astype(np.float32)
