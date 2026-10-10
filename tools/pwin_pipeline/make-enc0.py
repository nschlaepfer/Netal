import json, struct, sys, os
GEMM = os.environ.get("PWIN_GEMM", "pwin_gemm2_tokens")
MLP = os.environ.get("PWIN_MLP", "separate")
MIX = os.environ.get("PWIN_MIX", "pwin_mix_window")
MIXDBG = int(os.environ.get("PWIN_MIXDBG", "0"))
MIX4_TGS = int(os.environ.get("PWIN_MIX4_TGS", "160"))
OUT = os.environ.get("PWIN_OUT", "work/native/enc0/pipeline/enc0.json")
W, H = 480, 288
T = W * H
import os
lay = json.load(open(os.environ.get("PWIN_LAYOUT_FILE", "work/native/enc0/pipeline/layout-enc0.json")))
def off(name): return lay[name]["offset"]
# scratch layout in halves
Y1, N1, Z, X2, X3, S, Hh, X4a, X4b, MER, ZALT = 0, T*32, T*64, T*128, T*160, T*192, T*224, T*352, T*384, T*416, T*448
scratch_bytes = T * 464 * 2
def gemm(x, w, c, y, tokens, k, n, xs, ys, cs, epi=0, cmode=0, cin=None, cbias=None, split=0, wstride=0):
    prm = struct.pack("<16I", tokens, k, n, xs, ys, cs, epi, cmode, split, W, 4, wstride, 0, 0, 0, 0)
    bufs = [dict(index=0, **x), dict(index=1, **w), dict(index=2, **c), dict(index=3, **y),
            dict(index=5, **(cin or dict(name="pweights", offset=off("Zero32")))), dict(index=6, **(cbias or dict(name="pweights", offset=off("Zero32"))))]
    grid = [(tokens + 63) // 64, 1, 1] if GEMM == "pwin_gemm_mpp" else [(tokens + 127) // 128, 1, 1]
    block = [128, 1, 1] if GEMM == "pwin_gemm_mpp" else [256, 1, 1]
    return {"function": GEMM, "grid": grid, "block": block, "buffers": bufs,
            "bytes": [{"index": 4, "u32": list(struct.unpack("<16I", prm))}], "constants": {"0": k, "1": n}}
def norm(inp, out, gamma, mode):
    prm = struct.pack("<4I", T, 32, 32, mode)
    return {"function": "pwin_norm32", "grid": [(T + 255) // 256, 1, 1], "block": [256, 1, 1],
            "buffers": [dict(index=0, **inp), dict(index=1, **out), dict(index=2, **gamma)], "bytes": [{"index": 3, "u32": list(struct.unpack("<4I", prm))}]}
def mix(zin, w3, w4, r, out, rmode, rbias, zs, zalt=None):
    bufs = [dict(index=0, **zin), dict(index=1, **w3), dict(index=2, **w4), dict(index=3, **r), dict(index=4, **out), dict(index=6, **rbias), dict(index=7, **(zalt or zin))]
    if MIX == "split":
        # interior windows: streaming kernel (persistent SIMD groups, 4 per row block); border windows: gather kernel
        prm4 = struct.pack("<2I2i8I", W, H, 4, 4, zs, 32, 32, rmode, 1 if zalt else 0, 16, 61, 37)
        prm1 = struct.pack("<2I2i8I", W, H, 4, 4, zs, 32, 32, rmode, 1 if zalt else 0, 16, 1, 0)
        return [{"function": "pwin_mix4_interior", "grid": [MIX4_TGS, 1, 1], "block": [128, 1, 1], "buffers": bufs,
                 "bytes": [{"index": 5, "u32": list(struct.unpack("<12I", prm4))}]},
                {"function": "pwin_mix_window", "grid": [61, 37, 1], "block": [128, 1, 1], "buffers": bufs,
                 "bytes": [{"index": 5, "u32": list(struct.unpack("<12I", prm1))}]}]
    prm = struct.pack("<2I2i8I", W, H, 4, 4, zs, 32, 32, rmode, 1 if zalt else 0, 16, MIXDBG, 0)
    return [{"function": MIX, "grid": [61, 37, 1], "block": [128 if MIX == "pwin_mix_window" else 256, 1, 1], "buffers": bufs,
             "bytes": [{"index": 5, "u32": list(struct.unpack("<12I", prm))}]}]
def mlp(u, out):
    prm = struct.pack("<8I", T, W, 4, 32, 32, 0, 0, 0)
    return {"function": "pwin_mlp_tile", "grid": [(T + 63) // 64, 1, 1], "block": [128, 1, 1],
            "buffers": [dict(index=0, **u), dict(index=1, **pw("G2")), dict(index=2, **pw("W5")), dict(index=3, **pw("B5v")),
                        dict(index=4, **pw("W6")), dict(index=5, **pw("B6")), dict(index=6, **out)],
            "bytes": [{"index": 7, "u32": list(struct.unpack("<8I", prm))}]}
def merge(x, a):
    prm = struct.pack("<4I", W, H, 32, 128)
    return {"function": "pwin_merge2x2", "grid": [(T + 255) // 256, 1, 1], "block": [256, 1, 1],
            "buffers": [dict(index=0, **x), dict(index=1, **a)], "bytes": [{"index": 2, "u32": list(struct.unpack("<4I", prm))}]}
sc = lambda halves: dict(name="scratch", offset=halves * 2)
pw = lambda name, halves=0: dict(name="pweights", offset=off(name) + halves * 2)
d = [
  gemm(dict(name="arena6", offset=0), pw("W1"), pw("C1"), sc(Y1), T, 16, 32, 16, 32, 32),
  norm(sc(Y1), sc(N1), pw("G1"), 0),
  gemm(sc(N1), pw("W2"), pw("C0z"), sc(Z), T, 32, 64, 32, 64, 64, split=1),
  gemm(sc(N1), pw("W2"), pw("C0z"), sc(ZALT), T, 32, 16, 32, 16, 64, split=0, wstride=64),
  *mix(sc(Z), pw("W3"), pw("W4"), sc(Y1), sc(X2), 1, pw("Rb1"), 64, zalt=sc(ZALT)),
  *mix(sc(Z + 32), pw("W3b"), pw("W4b"), sc(X2), sc(X3), 0, pw("Rb1"), 64),
  *([mlp(sc(X3), dict(name="arena8", offset=43200))] if MLP == "fused" else [
    norm(sc(X3), sc(S), pw("G2"), 1),
    gemm(sc(S), pw("W5"), pw("C5"), sc(Hh), T, 32, 128, 32, 128, 128, epi=2, split=2),
    gemm(sc(Hh), pw("W6"), pw("C0z"), sc(X4a), T, 32, 32, 128, 32, 32, cmode=1, cin=sc(X3), cbias=pw("B6"), wstride=32),
    gemm(sc(Hh + 32), pw("W6", 32 * 32), pw("C0z"), sc(X4b), T, 32, 32, 128, 32, 32, cmode=1, cin=sc(X4a), cbias=pw("Zero32"), wstride=32),
    gemm(sc(Hh + 64), pw("W6", 64 * 32), pw("C0z"), sc(X4a), T, 32, 32, 128, 32, 32, cmode=1, cin=sc(X4b), cbias=pw("Zero32"), wstride=32),
    gemm(sc(Hh + 96), pw("W6", 96 * 32), pw("C0z"), dict(name="arena8", offset=43200), T, 32, 32, 128, 32, 32, cmode=1, cin=sc(X4a), cbias=pw("Zero32"), wstride=32)]),
  merge(dict(name="arena8", offset=43200), sc(MER)),
  gemm(sc(MER), pw("W7"), pw("C7"), dict(name="arena7", offset=0), T // 4, 128, 64, 128, 64, 64),
]
pipe = {"library": "build/pwin_pipeline/pwin_all.metallib", "scratch_bytes": scratch_bytes,
        "files": [{"name": "pweights", "path": "work/native/enc0/pipeline/weights-enc0.bin"}], "dispatches": d}
json.dump(pipe, open(OUT, "w"), indent=1)
print("dispatches", len(d), "scratch MB", scratch_bytes / 1e6)
