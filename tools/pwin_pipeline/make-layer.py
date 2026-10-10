"""Pipeline description for a PWin layer on the padded token grid (see pwin_pad.metal).
usage: make-layer.py enc0|dec0 [out.json]   (env: PWIN_GEMM=pwin_gemm2_tokens|pwin_gemm_mpp)"""
import json, os, struct, sys
layer = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else f"work/native/{layer}/pipeline/{layer}-pad.json"
GEMM = os.environ.get("PWIN_GEMM", "pwin_gemm3_tokens")
CONTRACT = os.environ.get("PWIN_CONTRACT", "fused")
W, H = int(os.environ.get("PWIN_W", "480")), int(os.environ.get("PWIN_H", "288"))   # enc0/dec0 token grid of the profile
Wp, Hp = W + 8, H + 8
T, Tp = W * H, Wp * Hp
lay = json.load(open(os.environ.get("PWIN_LAYOUT_FILE", f"work/native/{layer}/pipeline/layout-{layer}.json")))

def relocations(path):
    """{param_offset: (arena name, byte offset)} from a replay-config.bin (16-byte relocation entries)."""
    b = open(path, "rb").read(); n = struct.unpack_from("<I", b, 164)[0]; out = {}
    for k in range(n):
        param, buf, off = struct.unpack_from("<IIQ", b, 168 + 16 * k)
        out[param] = ({6: "arena6", 7: "arena7", 8: "arena8", 9: "weights"}[buf], off)
    return out
PROFILE_DIR = os.environ.get("PWIN_PROFILE_DIR", "work/re9-sr-runtime-bundle/profiles/1920x1080")
REL = relocations(f"{PROFILE_DIR}/pwin/{'enc0' if layer == 'enc0' else 'pwin-11'}/replay-config.bin")
AR = lambda param: dict(name=REL[param][0], offset=REL[param][1])
def off(name): return lay[name]["offset"]
sc = lambda halves: dict(name="scratch", offset=halves * 2)
pw = lambda name, halves=0: dict(name="pweights", offset=off(name) + halves * 2)
# scratch layout in halves (padded-grid tensors)
XIN, Y1, N1, Z, X2, X3, S, Hh, X4a, X4b, H0, MER, OUTP = (0, Tp*16, Tp*48, Tp*80, Tp*144, Tp*176, Tp*208, Tp*240, Tp*368, Tp*400, Tp*432, Tp*464, Tp*496)
scratch_bytes = Tp * 560 * 2
def gemm(x, w, c, y, tokens, k, n, xs, ys, cs, epi=0, cmode=0, cin=None, cbias=None, split=0, wstride=0, store_cols=0, width=Wp, shift_y=0):
    bufs = [dict(index=0, **x), dict(index=1, **w), dict(index=2, **c), dict(index=3, **y),
            dict(index=5, **(cin or pw("Zero32"))), dict(index=6, **(cbias or pw("Zero32")))]
    if GEMM == "pwin_gemm3_tokens":
        prm = struct.pack("<8I", tokens, epi, cmode, split, width, shift_y, store_cols, 0)
        return {"function": GEMM, "grid": [(tokens + 127) // 128, 1, 1], "block": [256, 1, 1], "buffers": bufs,
                "bytes": [{"index": 4, "u32": list(struct.unpack("<8I", prm))}],
                "constants": {"0": k, "1": n, "2": xs, "3": cs, "7": ys, "8": wstride, "9": 0}}
    prm = struct.pack("<16I", tokens, k, n, xs, ys, cs, epi, cmode, split, width, shift_y, wstride, store_cols, 0, 0, 0)
    grid = [(tokens + 63) // 64, 1, 1] if GEMM == "pwin_gemm_mpp" else [(tokens + 127) // 128, 1, 1]
    block = [128, 1, 1] if GEMM == "pwin_gemm_mpp" else [256, 1, 1]
    return {"function": GEMM, "grid": grid, "block": block, "buffers": bufs,
            "bytes": [{"index": 4, "u32": list(struct.unpack("<16I", prm))}], "constants": {"0": k, "1": n}}
def norm(inp, out, gamma, mode, tokens=Tp):
    prm = struct.pack("<4I", tokens, 32, 32, mode)
    return {"function": "pwin_norm32", "grid": [(tokens + 255) // 256, 1, 1], "block": [256, 1, 1],
            "buffers": [dict(index=0, **inp), dict(index=1, **out), dict(index=2, **gamma)], "bytes": [{"index": 3, "u32": list(struct.unpack("<4I", prm))}]}
MIXP = os.environ.get("PWIN_MIXP", "persistent")
MIXP_TGS = int(os.environ.get("PWIN_MIXP_TGS", "320"))
def mixp(zin, w3, w4, r, out, rmode, rbias, zs, out_padded=1):
    bufs = [dict(index=0, **zin), dict(index=1, **w3), dict(index=2, **w4), dict(index=3, **r), dict(index=4, **out), dict(index=6, **rbias)]
    if MIXP == "persistent":
        prm = struct.pack("<2I2i8I", W, H, 4, 4, zs, 32, 32, rmode, out_padded, W // 8 + 1, H // 8 + 1, 0)
        return {"function": "pwin_mixp2_persistent", "grid": [MIXP_TGS, 1, 1], "block": [128, 1, 1], "buffers": bufs,
                "bytes": [{"index": 5, "u32": list(struct.unpack("<12I", prm))}], "constants": {"11": zs, "12": 32, "13": 32}}
    prm = struct.pack("<2I2i8I", W, H, 4, 4, zs, 32, 32, rmode, out_padded, 0, 0, 0)
    return {"function": "pwin_mixp_window", "grid": [W // 8 + 1, H // 8 + 1, 1], "block": [128, 1, 1], "buffers": bufs,
            "bytes": [{"index": 5, "u32": list(struct.unpack("<12I", prm))}]}
def pad(fn, bufs, in_stride, out_stride, channels, h_stride=128, s_stride=32, grid_items=None):
    prm = struct.pack("<8I", W, H, in_stride, out_stride, channels, h_stride, s_stride, 0)
    n = grid_items
    return {"function": fn, "grid": [(n + 255) // 256, 1, 1], "block": [256, 1, 1], "buffers": [dict(index=i, **b) for i, b in enumerate(bufs)],
            "bytes": [{"index": len(bufs), "u32": list(struct.unpack("<8I", prm))}]}
d = []
if layer == "enc0":
    d += [pad("pwin_pad_gather", [AR(8), sc(XIN)], 16, 16, 16, grid_items=Tp * 2),
          gemm(sc(XIN), pw("W1"), pw("C1"), sc(Y1), Tp, 16, 32, 16, 32, 32),
          norm(sc(Y1), sc(N1), pw("G1"), 0)]
    rmode_a = 1
else:
    d += [gemm(AR(8), pw("W0"), pw("C0"), sc(H0), T // 4, 64, 128, 64, 128, 128, width=W // 2),
          pad("pwin_pad_d2s_add", [sc(H0), AR(16), sc(Y1)], 32, 32, 32, grid_items=Tp * 4),
          norm(sc(Y1), sc(N1), pw("G1"), 2)]
    rmode_a = 2
d += [gemm(sc(N1), pw("W2"), pw("C0z"), sc(Z), Tp, 32, 64, 32, 64, 64, split=2),
      mixp(sc(Z), pw("W3"), pw("W4"), sc(Y1), sc(X2), rmode_a, pw("Rb1"), 64),
      mixp(sc(Z + 32), pw("W3b"), pw("W4b"), sc(X2), sc(X3), 0, pw("Rb1"), 64),
      norm(sc(X3), sc(S), pw("G2"), 1)]
MLP = os.environ.get("PWIN_MLP", "fused")
def mlp2(x, out, exp_split, x_scale=None):
    prm = struct.pack("<8I", Tp, exp_split, Wp, 0, 1 if x_scale else 0, 0, 0, 0)
    return {"function": "pwin_mlp2_tokens", "grid": [(Tp + 127) // 128, 1, 1], "block": [256, 1, 1],
            "buffers": [dict(index=0, **x), dict(index=1, **pw("W5")), dict(index=2, **pw("C5")), dict(index=3, **pw("W6")),
                        dict(index=4, **sc(X3)), dict(index=5, **pw("B6")), dict(index=6, **out), dict(index=8, **(x_scale or pw("Zero32")))],
            "bytes": [{"index": 7, "u32": list(struct.unpack("<8I", prm))}],
            "constants": {"0": 32, "1": 32, "2": 32, "3": 32, "7": 32, "14": 128}}
if MLP == "fused":
    d.pop()   # the separate scale pass: folded into the MLP's A loads
    d += [mlp2(sc(X3), sc(X4b), 2, x_scale=pw("G2"))]
else:
  d += [
      gemm(sc(S), pw("W5"), pw("C5"), sc(Hh), Tp, 32, 128, 32, 128, 128, epi=2, split=2),
      *([gemm(sc(Hh), pw("W6"), pw("C0z"), sc(X4b), Tp, 128, 32, 128, 32, 32, cmode=1, cin=sc(X3), cbias=pw("B6"), wstride=32, split=4)]
        if CONTRACT == "fused" and GEMM == "pwin_gemm3_tokens" else
        [gemm(sc(Hh), pw("W6"), pw("C0z"), sc(X4a), Tp, 32, 32, 128, 32, 32, cmode=1, cin=sc(X3), cbias=pw("B6"), wstride=32),
         gemm(sc(Hh + 32), pw("W6", 32 * 32), pw("C0z"), sc(X4b), Tp, 32, 32, 128, 32, 32, cmode=1, cin=sc(X4a), cbias=pw("Zero32"), wstride=32),
         gemm(sc(Hh + 64), pw("W6", 64 * 32), pw("C0z"), sc(X4a), Tp, 32, 32, 128, 32, 32, cmode=1, cin=sc(X4b), cbias=pw("Zero32"), wstride=32),
         gemm(sc(Hh + 96), pw("W6", 96 * 32), pw("C0z"), sc(X4b), Tp, 32, 32, 128, 32, 32, cmode=1, cin=sc(X4a), cbias=pw("Zero32"), wstride=32)])]
if layer == "enc0":
    d += [pad("pwin_pad_store", [sc(X4b), AR(32)], 32, 32, 32, grid_items=T * 4),
          pad("pwin_pad_merge2x2", [sc(X4b), sc(MER)], 32, 128, 32, grid_items=T),
          gemm(sc(MER), pw("W7"), pw("C7"), AR(24), T // 4, 128, 64, 128, 64, 64, width=W // 2)]
else:
    d += [gemm(sc(X4b), pw("W7"), pw("C7"), sc(OUTP), Tp, 32, 48, 32, 40, 48, store_cols=40),
          pad("pwin_pad_store", [sc(OUTP), AR(24)], 40, 40, 40, grid_items=T * 5)]
pipe = {"library": "build/pwin_pipeline/pwin_all.metallib" if GEMM != "pwin_gemm_mpp" else "build/pwin_pipeline/pwin_all2.metallib", "scratch_bytes": scratch_bytes,
        "files": [{"name": "pweights", "path": f"work/native/{layer}/pipeline/weights-{layer}.bin"}], "dispatches": d}
json.dump(pipe, open(OUT, "w"), indent=1)
print(layer, "dispatches", len(d), "scratch MB", scratch_bytes / 1e6, "->", OUT)
