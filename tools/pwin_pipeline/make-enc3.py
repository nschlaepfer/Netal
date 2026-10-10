"""Pipeline description for enc3 (four-head windowed attention, model width 96 with attention inner width
128, MLP 96->384->96 in 12 groups, 2x2 space-to-depth output projection 384->128) on the unpadded 60x36 grid.
Math: work/native/enc3/phase-report.md.  usage: make-enc3.py [out.json]"""
import json, os, struct, sys
OUT = sys.argv[1] if len(sys.argv) > 1 else "work/native/enc3/pipeline/enc3.json"
PROFILE_DIR = os.environ.get("PWIN_PROFILE_DIR", "work/re9-sr-runtime-bundle/profiles/1920x1080")
_args = open(f"{PROFILE_DIR}/pwin/pwin-04/arguments-template.bin", "rb").read()
W, H = struct.unpack_from("<2I", _args, 0)
PADX, PADY = struct.unpack_from("<2I", _args, 56)
PADX = int(os.environ.get("PWIN_PADX", PADX)); PADY = int(os.environ.get("PWIN_PADY", PADY))
T = W * H
lay = json.load(open(os.environ.get("PWIN_LAYOUT_FILE", "work/native/enc3/pipeline/layout-enc3.json")))
off = lambda name: lay[name]["offset"]
sc = lambda halves: dict(name="scratch", offset=halves * 2)
pw = lambda name, halves=0: dict(name="pweights", offset=off(name) + halves * 2)
def relocations(path):
    b = open(path, "rb").read(); n = struct.unpack_from("<I", b, 164)[0]; out = {}
    for k in range(n):
        param, buf, o = struct.unpack_from("<IIQ", b, 168 + 16 * k)
        out[param] = ({6: "arena6", 7: "arena7", 8: "arena8", 9: "weights"}[buf], o)
    return out
REL = relocations(f"{PROFILE_DIR}/pwin/pwin-04/replay-config.bin")
IN = dict(name=REL[8][0], offset=REL[8][1])        # enc1's merged output, [H][W][64]
OOUT = dict(name=REL[24][0], offset=REL[24][1])    # o [H/2][W/2][96]
ZOUT = dict(name=REL[32][0], offset=REL[32][1])    # z [H][W][64]
N1, QKV, O, U, MER = T * 0, T * 96, T * 480, T * 608, T * 704
SCL, HID = T * 704 + (T // 4) * 384, T * 800 + (T // 4) * 384
scratch_bytes = (T * 1184 + (T // 4) * 384) * 2
def gemm(x, w, c, y, tokens, k, n, xs, ys, cs, epi=0, cmode=0, cin=None, cbias=None, split=0, wstride=0,
         width=W, shift_y=PADY, r_stride=0, nsplit=1):
    bufs = [dict(index=0, **x), dict(index=1, **w), dict(index=2, **c), dict(index=3, **y),
            dict(index=5, **(cin or pw("Zero96"))), dict(index=6, **(cbias or pw("Zero96")))]
    prm = struct.pack("<8I", tokens, epi, cmode, split, width, shift_y, 0, 0)
    groups = ((tokens + 15) // 16) * nsplit
    return {"function": "pwin_gemm3_tokens", "grid": [(groups + 7) // 8, 1, 1], "block": [256, 1, 1], "buffers": bufs,
            "bytes": [{"index": 4, "u32": list(struct.unpack("<8I", prm))}],
            "constants": {"0": k, "1": n, "2": xs, "3": cs, "7": ys, "8": wstride, "9": r_stride, "16": nsplit}}
def norm(inp, out):
    prm = struct.pack("<4I", T, 96, 96, 0)
    return {"function": "pwin_norm96", "grid": [(T + 255) // 256, 1, 1], "block": [256, 1, 1],
            "buffers": [dict(index=0, **inp), dict(index=1, **out), dict(index=2, **pw("G1"))],
            "bytes": [{"index": 3, "u32": list(struct.unpack("<4I", prm))}]}
def attn(qkv, out, heads, split_table):
    wx, wy = (W + PADX + 7) // 8, (H + PADY + 7) // 8
    tasks = wx * wy * heads * 4
    prm = struct.pack("<2I2i4I8I4I", W, H, PADX, PADY, 96 * heads, 32 * heads, heads, wx, *split_table, 0, 0, 0, 0)
    return {"function": "pwin_attn_window", "grid": [(tasks + 3) // 4, 1, 1], "block": [128, 1, 1],
            "buffers": [dict(index=0, **qkv), dict(index=1, **pw("Bpos0")), dict(index=2, **out)],
            "bytes": [{"index": 3, "u32": list(struct.unpack("<20I", prm))}]}
def mlp2(x, out, n, c0):
    # the fused MLP holds its output tiles in registers (four 16-wide tiles), so a 96-channel
    # contraction runs as two column slices; Wc keeps its full row stride
    prm = struct.pack("<8I", T, 6, W, PADY, 1, 4, 0, 0)   # expand split 1+1+1+1+2 for window blocks 0 and 2
    return {"function": "pwin_mlp2_tokens", "grid": [(T + 127) // 128, 1, 1], "block": [256, 1, 1],
            "buffers": [dict(index=0, **x), dict(index=1, **pw("We")), dict(index=2, **pw("Cbe")), dict(index=3, **pw("Wc", c0)),
                        dict(index=4, **sc(U + c0)), dict(index=5, **pw("Bc", c0)), dict(index=6, **out), dict(index=8, **pw("S1"))],
            "bytes": [{"index": 7, "u32": list(struct.unpack("<8I", prm))}],
            "constants": {"0": 96, "1": n, "2": 96, "3": 96, "7": 96, "14": 384, "15": 96}}
def merge(x, a):
    prm = struct.pack("<6I", W, H, 96, 384, 96, 0)
    return {"function": "pwin_merge2x2c", "grid": [(T + 255) // 256, 1, 1], "block": [256, 1, 1],
            "buffers": [dict(index=0, **x), dict(index=1, **a)], "bytes": [{"index": 2, "u32": list(struct.unpack("<6I", prm))}]}
SPLIT_ENC3 = [7, 4, 7, 0, 7, 0, 0, 0]      # bit b' set: round O(b, n) after key block b'; index 2b + n
d = [
  norm(IN, sc(N1)),
  gemm(sc(N1), pw("Wqkv"), pw("C0z"), sc(QKV), T, 96, 384, 96, 384, 384, split=4, nsplit=4),
  attn(sc(QKV), sc(O), 4, SPLIT_ENC3),
  gemm(sc(O), pw("Wo"), pw("Cbo"), sc(U), T, 128, 96, 128, 96, 96, split=4, epi=3, cin=IN, r_stride=96, nsplit=2),
  # unfused MLP: the expand runs once instead of once per output slice (see make-dec5.py).
  # split 6 | rounds 4 in the high bits reproduces the fused expand's 1+1+1+1+2 rounding.
  {"function": "pwin_norm96", "grid": [(T + 255) // 256, 1, 1], "block": [256, 1, 1],
   "buffers": [dict(index=0, **sc(U)), dict(index=1, **sc(SCL)), dict(index=2, **pw("S1"))],
   "bytes": [{"index": 3, "u32": list(struct.unpack("<4I", struct.pack("<4I", T, 96, 96, 1)))}]},
  gemm(sc(SCL), pw("We"), pw("Cbe"), sc(HID), T, 96, 384, 96, 384, 384, epi=2, split=6 + 16 * 4, nsplit=6),
  gemm(sc(HID), pw("Wc"), pw("C0z"), ZOUT, T, 384, 96, 384, 96, 96,
       cmode=1, cin=sc(U), cbias=pw("Bc"), split=4, nsplit=6),
  merge(ZOUT, sc(MER)),
  gemm(sc(MER), pw("W5"), pw("Cb5"), OOUT, T // 4, 384, 128, 384, 128, 128, width=W // 2, nsplit=2),
]
pipe = {"library": "build/pwin_pipeline/pwin_all.metallib", "scratch_bytes": scratch_bytes,
        "files": [{"name": "pweights", "path": "work/native/enc3/pipeline/weights-enc3.bin"}], "dispatches": d}
json.dump(pipe, open(OUT, "w"), indent=1)
print(f"enc3 dispatches {len(d)} W,H {W},{H} pad {PADX},{PADY} scratch MB {scratch_bytes/1e6:.1f} -> {OUT}")
