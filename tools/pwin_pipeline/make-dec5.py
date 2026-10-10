"""Pipeline description for dec5 (bottleneck attention block: width 160, 8 heads x 32, MLP 160->640->160 in 20 groups,
output = the MLP result). Math: work/native/dec5/phase-report.md.  usage: make-dec5.py [out.json]"""
import json, os, struct, sys
OUT = sys.argv[1] if len(sys.argv) > 1 else "work/native/dec5/pipeline/dec5.json"
PROFILE_DIR = os.environ.get("PWIN_PROFILE_DIR", "work/re9-sr-runtime-bundle/profiles/1920x1080")
_args = open(f"{PROFILE_DIR}/pwin/pwin-06/arguments-template.bin", "rb").read()
W, H = struct.unpack_from("<2I", _args, 0)
PADX, PADY = struct.unpack_from("<2I", _args, 56)
# the five deep layers cycle through 8 window-shift phases per frame (kPwinShift* in dlss_sr_session.mm);
# PWIN_PADX/PWIN_PADY select the phase, and every window count is derived from them
PADX = int(os.environ.get("PWIN_PADX", PADX)); PADY = int(os.environ.get("PWIN_PADY", PADY))
T = W * H
lay = json.load(open(os.environ.get("PWIN_LAYOUT_FILE", "work/native/dec5/pipeline/layout-dec5.json")))
def off(name): return lay[name]["offset"]
sc = lambda halves: dict(name="scratch", offset=halves * 2)
pw = lambda name, halves=0: dict(name="pweights", offset=off(name) + halves * 2)
def relocations(path):
    b = open(path, "rb").read(); n = struct.unpack_from("<I", b, 164)[0]; out = {}
    for k in range(n):
        param, buf, o = struct.unpack_from("<IIQ", b, 168 + 16 * k)
        out[param] = ({6: "arena6", 7: "arena7", 8: "arena8", 9: "weights"}[buf], o)
    return out
REL = relocations(f"{PROFILE_DIR}/pwin/pwin-06/replay-config.bin")
IN = dict(name=REL[8][0], offset=REL[8][1])       # [H][W][160] (enc4's o)
ZOUT = dict(name=REL[24][0], offset=REL[24][1])   # [H][W][160] output
N1, QKV, O, U, SCL, HID = T * 0, T * 160, T * 928, T * 1184, T * 1344, T * 1504
scratch_bytes = T * 2144 * 2
for h in range(1, 8): assert off(f"Bpos{h}") == off("Bpos0") + 8192 * h
def gemm(x, w, c, y, tokens, k, n, xs, ys, cs, epi=0, cmode=0, cin=None, cbias=None, split=0, wstride=0, store_cols=0, width=W, shift_y=0, r_stride=0, row_filter=0, nsplit=1):
    bufs = [dict(index=0, **x), dict(index=1, **w), dict(index=2, **c), dict(index=3, **y),
            dict(index=5, **(cin or pw("Zero160"))), dict(index=6, **(cbias or pw("Zero160")))]
    prm = struct.pack("<8I", tokens, epi, cmode, split, width, shift_y, store_cols, row_filter)
    groups = ((tokens + 15) // 16) * nsplit
    return {"function": "pwin_gemm3_tokens", "grid": [(groups + 7) // 8, 1, 1], "block": [256, 1, 1], "buffers": bufs,
            "bytes": [{"index": 4, "u32": list(struct.unpack("<8I", prm))}],
            "constants": {"0": k, "1": n, "2": xs, "3": cs, "7": ys, "8": wstride, "9": r_stride, "16": nsplit}}
def norm(inp, out, gamma):
    prm = struct.pack("<4I", T, 160, 160, 0)
    return {"function": "pwin_norm160", "grid": [(T + 255) // 256, 1, 1], "block": [256, 1, 1],
            "buffers": [dict(index=0, **inp), dict(index=1, **out), dict(index=2, **gamma)], "bytes": [{"index": 3, "u32": list(struct.unpack("<4I", prm))}]}
def attn(qkv, bpos, out, heads, split_table, shift_x, shift_y):
    wx, wy = (W + shift_x + 7) // 8, (H + shift_y + 7) // 8
    tasks = wx * wy * heads * 4
    prm = struct.pack("<2I2i4I8I4I", W, H, shift_x, shift_y, 96 * heads, 32 * heads, heads, wx, *split_table, 0, 0, 0, 0)
    return {"function": "pwin_attn_window", "grid": [(tasks + 3) // 4, 1, 1], "block": [128, 1, 1],
            "buffers": [dict(index=0, **qkv), dict(index=1, **bpos), dict(index=2, **out)], "bytes": [{"index": 3, "u32": list(struct.unpack("<20I", prm))}]}
def mlp2(x, out, k, hd, n, xs, cs, ys, x_scale, cin, cbias, we, ce, wc, wcs):
    prm = struct.pack("<8I", T, 0, W, 0, 1 if x_scale else 0, 0, 0, 0)
    return {"function": "pwin_mlp2_tokens", "grid": [(T + 127) // 128, 1, 1], "block": [256, 1, 1],
            "buffers": [dict(index=0, **x), dict(index=1, **we), dict(index=2, **ce), dict(index=3, **wc), dict(index=4, **cin), dict(index=5, **cbias), dict(index=6, **out), dict(index=8, **(x_scale or pw("Zero160")))],
            "bytes": [{"index": 7, "u32": list(struct.unpack("<8I", prm))}],
            "constants": {"0": k, "1": n, "2": xs, "3": cs, "7": ys, "14": hd, "15": wcs}}
SPLIT_DEC5 = [7, 4, 7, 0, 7, 0, 0, 0]
d = [
  norm(IN, sc(N1), pw("G1")),
  gemm(sc(N1), pw("Wqkv"), pw("C0z"), sc(QKV), T, 160, 768, 160, 768, 768, split=4, nsplit=8),
  attn(sc(QKV), pw("Bpos0"), sc(O), 8, SPLIT_DEC5, PADX, PADY),
  gemm(sc(O), pw("Wo"), pw("Cbo"), sc(U), T, 256, 160, 256, 160, 160, split=4, epi=3, cin=IN, r_stride=160, nsplit=5),
]
# The fused MLP caps at four 16-wide output tiles, so 160 channels would need three column
# slices and each slice recomputes the whole 640-wide expand. Unfused, the expand runs once
# and nsplit gives 135 tokens a parallel axis: 0.79 ms -> 0.16, bit-identical.
d += [{"function": "pwin_norm160", "grid": [(T + 255) // 256, 1, 1], "block": [256, 1, 1],
       "buffers": [dict(index=0, **sc(U)), dict(index=1, **sc(SCL)), dict(index=2, **pw("S1"))],
       "bytes": [{"index": 3, "u32": list(struct.unpack("<4I", struct.pack("<4I", T, 160, 160, 1)))}]},
      gemm(sc(SCL), pw("We"), pw("Cbe"), sc(HID), T, 160, 640, 160, 640, 640, epi=2, nsplit=10),
      gemm(sc(HID), pw("Wc"), pw("C0z"), ZOUT, T, 640, 160, 640, 160, 160,
           cmode=1, cin=sc(U), cbias=pw("Bc"), split=4, nsplit=10)]
pipe = {"library": "build/pwin_pipeline/pwin_all.metallib", "scratch_bytes": scratch_bytes,
        "files": [{"name": "pweights", "path": "work/native/dec5/pipeline/weights-dec5.bin"}], "dispatches": d}
json.dump(pipe, open(OUT, "w"), indent=1)
print("dec5 dispatches", len(d), "W,H", W, H, "pad", PADX, PADY, "->", OUT)
