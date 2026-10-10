"""Pipeline description for enc1 on the unpadded token grid (240 x 144, 64 channels).
enc1 has one token-mixing sub-layer and uniform rounding rules per token, so the reflected (out-of-image)
window tokens equal their in-image sources: the mixing gathers with reflection and everything else is
per token on the T = W*H grid. Math: work/native/enc1/phase-report.md.
usage: make-enc1.py [out.json]"""
import json, os, struct, sys
OUT = sys.argv[1] if len(sys.argv) > 1 else "work/native/enc1/pipeline/enc1.json"
W, H = int(os.environ.get("PWIN_W", "240")), int(os.environ.get("PWIN_H", "144"))   # enc1 token grid of the profile (arguments template +0/+4)
T = W * H
lay = json.load(open(os.environ.get("PWIN_LAYOUT_FILE", "work/native/enc1/pipeline/layout-enc1.json")))
def off(name): return lay[name]["offset"]
sc = lambda halves: dict(name="scratch", offset=halves * 2)
pw = lambda name, halves=0: dict(name="pweights", offset=off(name) + halves * 2)

def relocations(path):
    """{param_offset: (arena name, byte offset)} from a replay-config.bin (16-byte relocation entries)."""
    b = open(path, "rb").read(); n = struct.unpack_from("<I", b, 164)[0]; out = {}
    for k in range(n):
        param, buf, off = struct.unpack_from("<IIQ", b, 168 + 16 * k)
        out[param] = ({6: "arena6", 7: "arena7", 8: "arena8", 9: "weights"}[buf], off)
    return out
PROFILE_DIR = os.environ.get("PWIN_PROFILE_DIR", "work/re9-sr-runtime-bundle/profiles/1920x1080")
REL = relocations(f"{PROFILE_DIR}/pwin/pwin-02/replay-config.bin")
IN = dict(name=REL[8][0], offset=REL[8][1])            # enc0 half-res output [H][W][64]
OOUT = dict(name=REL[24][0], offset=REL[24][1])        # o [H/2][W/2][64]
ZOUT = dict(name=REL[32][0], offset=REL[32][1])        # z [H][W][64] (profile-dependent offset)
# scratch layout in halves per token
N1, H1, M1, U, S, Hh, E0, E1, E2, ZA, ZB, MER = (T*0, T*64, T*128, T*192, T*256, T*320, T*576, T*592, T*608, T*624, T*688, T*752)
scratch_bytes = T * 816 * 2
GEMM = os.environ.get("PWIN_GEMM", "pwin_gemm3_tokens")
def gemm(x, w, c, y, tokens, k, n, xs, ys, cs, epi=0, cmode=0, cin=None, cbias=None, split=0, wstride=0, store_cols=0, width=W, shift_y=4, r_stride=0, row_filter=0, nsplit=1):
    bufs = [dict(index=0, **x), dict(index=1, **w), dict(index=2, **c), dict(index=3, **y),
            dict(index=5, **(cin or pw("Zero64"))), dict(index=6, **(cbias or pw("Zero64")))]
    if GEMM == "pwin_gemm3_tokens":
        prm = struct.pack("<8I", tokens, epi, cmode, split, width, shift_y, store_cols, row_filter)
        groups = ((tokens + 15) // 16) * nsplit
        return {"function": GEMM, "grid": [(groups + 7) // 8, 1, 1], "block": [256, 1, 1], "buffers": bufs,
                "bytes": [{"index": 4, "u32": list(struct.unpack("<8I", prm))}],
                "constants": {"0": k, "1": n, "2": xs, "3": cs, "7": ys, "8": wstride, "9": r_stride, "16": nsplit}}
    prm = struct.pack("<16I", tokens, k, n, xs, ys, cs, epi, cmode, split, width, shift_y, wstride, store_cols, r_stride, row_filter, 0)
    return {"function": "pwin_gemm2_tokens", "grid": [(tokens + 127) // 128, 1, 1], "block": [256, 1, 1], "buffers": bufs,
            "bytes": [{"index": 4, "u32": list(struct.unpack("<16I", prm))}], "constants": ({"0": k, "1": n, "2": xs, "3": cs, "10": int(os.environ["PWIN_GEMM_DBG"])} if os.environ.get("PWIN_GEMM_DBG") else {"0": k, "1": n, "2": xs, "3": cs})}
def norm(inp, out, gamma, mode, in_stride=64):
    prm = struct.pack("<4I", T, in_stride, 64, mode)
    return {"function": "pwin_norm64", "grid": [(T + 255) // 256, 1, 1], "block": [256, 1, 1],
            "buffers": [dict(index=0, **inp), dict(index=1, **out), dict(index=2, **gamma)], "bytes": [{"index": 3, "u32": list(struct.unpack("<4I", prm))}]}
MIXM_TGS = int(os.environ.get("PWIN_MIXM_TGS", "160"))
def mixm(zin, ma, mb, out):   # one persistent kernel: branch-free interior path + reflected-gather border path
    bufs = [dict(index=0, **zin), dict(index=1, **ma), dict(index=2, **mb), dict(index=3, **out)]
    wx, wy = W // 8 + 1, H // 8 + 1
    prm2 = struct.pack("<2I2i4I", W, H, 4, 4, 64, 64, wx, wy)
    return [{"function": "pwin_mixm2_interior", "grid": [MIXM_TGS, 1, 1], "block": [128, 1, 1], "buffers": bufs,
             "bytes": [{"index": 4, "u32": list(struct.unpack("<8I", prm2))}], "constants": {"4": 64, "5": 64}}]
def merge(x, a):
    prm = struct.pack("<6I", W, H, 64, 256, 64, 0)
    return {"function": "pwin_merge2x2c", "grid": [(T + 255) // 256, 1, 1], "block": [256, 1, 1],
            "buffers": [dict(index=0, **x), dict(index=1, **a)], "bytes": [{"index": 2, "u32": list(struct.unpack("<6I", prm))}]}
d = [
  norm(IN, sc(N1), pw("G1"), 0),                                                          # N = norm(x)
  gemm(sc(N1), pw("W1"), pw("C0z"), sc(H1), T, 64, 64, 64, 64, 64, split=3),              # h1 = f16(f16(N W1[0:32]) + N W1[32:64])
  *mixm(sc(H1), pw("M1a"), pw("M1b"), sc(M1)),                                            # m1 = M1_z . h1 per channel half
  gemm(sc(M1), pw("W2"), pw("C2"), sc(U), T, 64, 64, 64, 64, 64, split=3, epi=3, cin=IN, r_stride=64),   # u = f16(f16(f16(b2 + m1 W2[0:32]) + m1 W2[32:64]) + x)
  norm(sc(U), sc(S), pw("S1"), 1),                                                        # us = f16(u * s)
]
MLP = os.environ.get("PWIN_MLP", "fused")
def mlp2(x, out, k, hd, n, xs, cs, ys, exp_split, x_scale=None):
    prm = struct.pack("<8I", T, exp_split, W, 4, 1 if x_scale else 0, 0, 0, 0)
    return {"function": "pwin_mlp2_tokens", "grid": [(T + 127) // 128, 1, 1], "block": [256, 1, 1],
            "buffers": [dict(index=0, **x), dict(index=1, **pw("We")), dict(index=2, **pw("Ce")), dict(index=3, **pw("Wc")),
                        dict(index=4, **sc(U)), dict(index=5, **pw("Bc")), dict(index=6, **out), dict(index=8, **(x_scale or pw("Zero64")))],
            "bytes": [{"index": 7, "u32": list(struct.unpack("<8I", prm))}],
            "constants": {"0": k, "1": n, "2": xs, "3": cs, "7": ys, "14": hd}}
if MLP == "fused":
    d.pop()   # the separate scale pass: folded into the MLP's A loads
    d.append(mlp2(sc(U), ZOUT, 64, 256, 64, 64, 64, 64, 5, x_scale=pw("S1")))
elif MLP == "sep":
    # one expand dispatch: gemm3 split mode 5 reproduces the fused expand's per-token-block
    # rounding of hidden units 0..15 directly, so the four chained 16-wide GEMMs below are not needed
    d.append(gemm(sc(S), pw("We"), pw("Ce"), sc(Hh), T, 64, 256, 64, 256, 256, epi=2, split=5,
                  wstride=256, nsplit=int(os.environ.get("PWIN_NSPLIT", "4"))))
else:
  d += [
  # e = act(f16(be + us We)) (fused form), as two N = 128 halves so the weights fit threadgroup memory
  gemm(sc(S), pw("We"), pw("Ce"), sc(Hh), T, 64, 128, 64, 256, 256, epi=2, wstride=256),
  gemm(sc(S), pw("We", 128), pw("Ce", 2 * 128), sc(Hh + 128), T, 64, 128, 64, 256, 256, epi=2, wstride=256),
  # 4-way split of hidden units 0..15 for tokens in window blocks 0 and 2 (chains 40-43): one K tile per rounding
  gemm(sc(S), pw("We"), pw("Ce"), sc(E0), T, 16, 16, 64, 16, 256, wstride=256),
  gemm(sc(S + 16), pw("We", 16 * 256), pw("C0z"), sc(E1), T, 16, 16, 64, 16, 16, cmode=1, cin=sc(E0), cbias=pw("Zero64"), wstride=256),
  gemm(sc(S + 32), pw("We", 32 * 256), pw("C0z"), sc(E2), T, 16, 16, 64, 16, 16, cmode=1, cin=sc(E1), cbias=pw("Zero64"), wstride=256),
  gemm(sc(S + 48), pw("We", 48 * 256), pw("C0z"), sc(Hh), T, 16, 16, 64, 256, 16, cmode=1, cin=sc(E2), cbias=pw("Zero64"), wstride=256, epi=2, row_filter=1),
]
# contraction: 8 groups of 32 hidden units, fp16 handoff after every group, C = f16(bc + u) for group 0
if MLP == "fused":
    pass
elif MLP == "sep" or (os.environ.get("PWIN_CONTRACT", "fused") == "fused" and GEMM == "pwin_gemm3_tokens"):
    d.append(gemm(sc(Hh), pw("Wc"), pw("C0z"), ZOUT, T, 256, 64, 256, 64, 64, cmode=1, cin=sc(U), cbias=pw("Bc"), wstride=64, split=4))
else:
    prev = None
    for g in range(8):
        out = ZOUT if g == 7 else sc(ZA if g % 2 == 0 else ZB)
        d.append(gemm(sc(Hh + 32 * g), pw("Wc", 32 * g * 64), pw("C0z"), out, T, 32, 64, 256, 64, 64, cmode=1,
                      cin=(sc(U) if g == 0 else prev), cbias=(pw("Bc") if g == 0 else pw("Zero64")), wstride=64))
        prev = out
d += [merge(ZOUT, sc(MER)),                                                                # s2d(z): [T/4][256]
      gemm(sc(MER), pw("W5"), pw("C5"), OOUT, T // 4, 256, 64, 256, 64, 64, width=W // 2)]  # o = f16(b5 + s2d(z) W5)
pipe = {"library": "build/pwin_pipeline/pwin_all.metallib", "scratch_bytes": scratch_bytes,
        "files": [{"name": "pweights", "path": "work/native/enc1/pipeline/weights-enc1.bin"}], "dispatches": d}
json.dump(pipe, open(OUT, "w"), indent=1)
print("enc1 dispatches", len(d), "scratch MB", scratch_bytes / 1e6, "->", OUT)
