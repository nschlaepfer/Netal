"""Pipeline description for dec1 (decoder attention block on the PADDED window grid: the half-resolution parent of a
padded position is floor(y/2) of the padded coordinate, so reflected tokens are not equal to their in-image sources).
P0 64->256 from the half-res input, 2x2 depth-to-space + enc4 skip -> 128 ch, RMS norm, 4-head attention, Wo + residual,
MLP 64->256->64 in 8 groups, clipped store. Math: work/native/dec1/phase-report.md.  usage: make-dec1.py [out.json]"""
import json, os, struct, sys
OUT = sys.argv[1] if len(sys.argv) > 1 else "work/native/dec1/pipeline/dec1.json"
PROFILE_DIR = os.environ.get("PWIN_PROFILE_DIR", "work/re9-sr-runtime-bundle/profiles/1920x1080")
_args = open(f"{PROFILE_DIR}/pwin/pwin-10/arguments-template.bin", "rb").read()
W, H = struct.unpack_from("<2I", _args, 0)
PADX, PADY = struct.unpack_from("<2I", _args, 56)
# the five deep layers cycle through 8 window-shift phases per frame (kPwinShift* in dlss_sr_session.mm);
# PWIN_PADX/PWIN_PADY select the phase, and every window count is derived from them
PADX = int(os.environ.get("PWIN_PADX", PADX)); PADY = int(os.environ.get("PWIN_PADY", PADY))
GX, GY = (W + PADX + 7) // 8, (H + PADY + 7) // 8      # the session's per-phase grid formula
WP, HP = 8 * GX, 8 * GY                       # the windows tile the padded grid exactly
T, TP, TH = W * H, WP * HP, (W // 2) * (H // 2)
lay = json.load(open(os.environ.get("PWIN_LAYOUT_FILE", "work/native/dec1/pipeline/layout-dec1.json")))
def off(name): return lay[name]["offset"]
sc = lambda halves: dict(name="scratch", offset=halves * 2)
pw = lambda name, halves=0: dict(name="pweights", offset=off(name) + halves * 2)
def relocations(path):
    b = open(path, "rb").read(); n = struct.unpack_from("<I", b, 164)[0]; out = {}
    for k in range(n):
        param, buf, o = struct.unpack_from("<IIQ", b, 168 + 16 * k)
        out[param] = ({6: "arena6", 7: "arena7", 8: "arena8", 9: "weights"}[buf], o)
    return out
REL = relocations(f"{PROFILE_DIR}/pwin/pwin-10/replay-config.bin")
IN = dict(name=REL[8][0], offset=REL[8][1])       # dec5's output, half-res [H/2][W/2][160]
SKIP = dict(name=REL[16][0], offset=REL[16][1])   # enc4's z, [H][W][128]
OUTB = dict(name=REL[24][0], offset=REL[24][1])   # output [H][W][128]
P0 = 0
XP, N1, QKV, O, U = TH * 256, TH * 256 + TP * 64, TH * 256 + TP * 128, TH * 256 + TP * 320, TH * 256 + TP * 384
scratch_bytes = (TH * 256 + TP * 448) * 2
def gemm(x, w, c, y, tokens, k, n, xs, ys, cs, epi=0, cmode=0, cin=None, cbias=None, split=0, wstride=0, store_cols=0, width=W, shift_y=0, r_stride=0, row_filter=0, nsplit=1):
    bufs = [dict(index=0, **x), dict(index=1, **w), dict(index=2, **c), dict(index=3, **y),
            dict(index=5, **(cin or pw("Zero64"))), dict(index=6, **(cbias or pw("Zero64")))]
    prm = struct.pack("<8I", tokens, epi, cmode, split, width, shift_y, store_cols, row_filter)
    groups = ((tokens + 15) // 16) * nsplit
    return {"function": "pwin_gemm3_tokens", "grid": [(groups + 7) // 8, 1, 1], "block": [256, 1, 1], "buffers": bufs,
            "bytes": [{"index": 4, "u32": list(struct.unpack("<8I", prm))}],
            "constants": {"0": k, "1": n, "2": xs, "3": cs, "7": ys, "8": wstride, "9": r_stride, "16": nsplit}}
def pad2(fn, bufs, channels, in_stride, skip_stride, out_stride, items):
    prm = struct.pack("<4I2i5I", W, H, WP, HP, PADX, PADY, channels, in_stride, skip_stride, out_stride, 0)
    return {"function": fn, "grid": [(items + 255) // 256, 1, 1], "block": [256, 1, 1],
            "buffers": [dict(index=i, **b) for i, b in enumerate(bufs)], "bytes": [{"index": len(bufs), "u32": list(struct.unpack("<11I", prm)) + [0]}]}
def norm(inp, out):
    prm = struct.pack("<4I", TP, 64, 64, 0)
    return {"function": "pwin_norm64", "grid": [(TP + 255) // 256, 1, 1], "block": [256, 1, 1],
            "buffers": [dict(index=0, **inp), dict(index=1, **out), dict(index=2, **pw("G1"))], "bytes": [{"index": 3, "u32": list(struct.unpack("<4I", prm))}]}
def attn(qkv, out, heads, split_table):
    tasks = GX * GY * heads * 4
    prm = struct.pack("<2I2i4I8I4I", W, H, PADX, PADY, 96 * heads, 32 * heads, heads, GX, *split_table, 1, WP, 0, 0)
    return {"function": "pwin_attn_window", "grid": [(tasks + 3) // 4, 1, 1], "block": [128, 1, 1],
            "buffers": [dict(index=0, **qkv), dict(index=1, **pw("Bpos0")), dict(index=2, **out)], "bytes": [{"index": 3, "u32": list(struct.unpack("<20I", prm))}]}
def mlp2(x, out, n, c0):
    prm = struct.pack("<8I", TP, 5, WP, PADY, 1, 0, 0, 0)   # 4-way expand split, each warp's first token block
    return {"function": "pwin_mlp2_tokens", "grid": [(TP + 127) // 128, 1, 1], "block": [256, 1, 1],
            "buffers": [dict(index=0, **x), dict(index=1, **pw("We")), dict(index=2, **pw("Cbe")), dict(index=3, **pw("Wc", c0)),
                        dict(index=4, **sc(U + c0)), dict(index=5, **pw("Bc", c0)), dict(index=6, **out), dict(index=8, **pw("S1"))],
            "bytes": [{"index": 7, "u32": list(struct.unpack("<8I", prm))}],
            "constants": {"0": 64, "1": n, "2": 64, "3": 64, "7": 64, "14": 256, "15": 64}}
SPLIT_DEC1 = [7, 4, 7, 4, 7, 0, 0, 0]
d = [
  gemm(IN, pw("W0"), pw("Cb0"), sc(P0), TH, 64, 256, 64, 256, 256, nsplit=4),
  pad2("pwin_padg_d2s_skip", [sc(P0), SKIP, sc(XP)], 64, 256, 64, 64, TP * 8),
  norm(sc(XP), sc(N1)),
  gemm(sc(N1), pw("Wqkv"), pw("C0z"), sc(QKV), TP, 64, 192, 64, 192, 192, split=4, width=WP, nsplit=2),
  attn(sc(QKV), sc(O), 2, SPLIT_DEC1),
  gemm(sc(O), pw("Wo"), pw("Cbo"), sc(U), TP, 64, 64, 64, 64, 64, split=4, epi=3, cin=sc(XP), r_stride=64, width=WP),
  mlp2(sc(U), sc(N1), 64, 0),
  pad2("pwin_padg_store", [sc(N1), OUTB], 64, 64, 64, 64, T * 8),
]
pipe = {"library": "build/pwin_pipeline/pwin_all.metallib", "scratch_bytes": scratch_bytes,
        "files": [{"name": "pweights", "path": "work/native/dec1/pipeline/weights-dec1.bin"}], "dispatches": d}
json.dump(pipe, open(OUT, "w"), indent=1)
print(f"dec1 dispatches {len(d)} W,H {W},{H} pad {PADX},{PADY} padded {WP}x{HP} scratch MB {scratch_bytes/1e6:.1f} -> {OUT}")
