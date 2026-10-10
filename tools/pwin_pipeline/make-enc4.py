"""Pipeline description for enc4 (attention block, width 128, 4 heads, MLP 128->512->128, output projection 512->160)
on the unpadded 30x18 token grid. Math: work/native/enc4/phase-report.md.  usage: make-enc4.py [out.json]"""
import json, os, struct, sys
OUT = sys.argv[1] if len(sys.argv) > 1 else "work/native/enc4/pipeline/enc4.json"
PROFILE_DIR = os.environ.get("PWIN_PROFILE_DIR", "work/re9-sr-runtime-bundle/profiles/1920x1080")
_args = open(f"{PROFILE_DIR}/pwin/pwin-05/arguments-template.bin", "rb").read()
W, H = struct.unpack_from("<2I", _args, 0)
PADX, PADY = struct.unpack_from("<2I", _args, 56)
# the five deep layers cycle through 8 window-shift phases per frame (kPwinShift* in dlss_sr_session.mm);
# PWIN_PADX/PWIN_PADY select the phase, and every window count is derived from them
PADX = int(os.environ.get("PWIN_PADX", PADX)); PADY = int(os.environ.get("PWIN_PADY", PADY))
T = W * H
lay = json.load(open(os.environ.get("PWIN_LAYOUT_FILE", "work/native/enc4/pipeline/layout-enc4.json")))
def off(name): return lay[name]["offset"]
sc = lambda halves: dict(name="scratch", offset=halves * 2)
pw = lambda name, halves=0: dict(name="pweights", offset=off(name) + halves * 2)
def relocations(path):
    b = open(path, "rb").read(); n = struct.unpack_from("<I", b, 164)[0]; out = {}
    for k in range(n):
        param, buf, o = struct.unpack_from("<IIQ", b, 168 + 16 * k)
        out[param] = ({6: "arena6", 7: "arena7", 8: "arena8", 9: "weights"}[buf], o)
    return out
REL = relocations(f"{PROFILE_DIR}/pwin/pwin-05/replay-config.bin")
IN = dict(name=REL[8][0], offset=REL[8][1])       # [H][W][128] raw input (enc3's o)
OOUT = dict(name=REL[24][0], offset=REL[24][1])   # o [H/2][W/2][160]
ZOUT = dict(name=REL[32][0], offset=REL[32][1])   # z [H][W][128]
N1, QKV, O, U, MER = T * 0, T * 128, T * 512, T * 640, T * 768
SCL, HID = T * 896, T * 1024
scratch_bytes = T * 1536 * 2
assert off("Bpos1") == off("Bpos0") + 8192 and off("Bpos3") == off("Bpos0") + 3 * 8192
def gemm(x, w, c, y, tokens, k, n, xs, ys, cs, epi=0, cmode=0, cin=None, cbias=None, split=0, wstride=0, store_cols=0, width=W, shift_y=0, r_stride=0, row_filter=0, nsplit=1):
    bufs = [dict(index=0, **x), dict(index=1, **w), dict(index=2, **c), dict(index=3, **y),
            dict(index=5, **(cin or pw("Zero128"))), dict(index=6, **(cbias or pw("Zero128")))]
    prm = struct.pack("<8I", tokens, epi, cmode, split, width, shift_y, store_cols, row_filter)
    groups = ((tokens + 15) // 16) * nsplit
    return {"function": "pwin_gemm3_tokens", "grid": [(groups + 7) // 8, 1, 1], "block": [256, 1, 1], "buffers": bufs,
            "bytes": [{"index": 4, "u32": list(struct.unpack("<8I", prm))}],
            "constants": {"0": k, "1": n, "2": xs, "3": cs, "7": ys, "8": wstride, "9": r_stride, "16": nsplit}}
def norm(inp, out, gamma, in_stride):
    prm = struct.pack("<4I", T, in_stride, 128, 0)
    return {"function": "pwin_norm128", "grid": [(T + 255) // 256, 1, 1], "block": [256, 1, 1],
            "buffers": [dict(index=0, **inp), dict(index=1, **out), dict(index=2, **gamma)], "bytes": [{"index": 3, "u32": list(struct.unpack("<4I", prm))}]}
def attn(qkv, bpos, out, heads, split_table, shift_x, shift_y):
    wx, wy = (W + shift_x + 7) // 8, (H + shift_y + 7) // 8
    tasks = wx * wy * heads * 4
    prm = struct.pack("<2I2i4I8I4I", W, H, shift_x, shift_y, 96 * heads, 32 * heads, heads, wx, *split_table, 0, 0, 0, 0)
    return {"function": "pwin_attn_window", "grid": [(tasks + 3) // 4, 1, 1], "block": [128, 1, 1],
            "buffers": [dict(index=0, **qkv), dict(index=1, **bpos), dict(index=2, **out)], "bytes": [{"index": 3, "u32": list(struct.unpack("<20I", prm))}]}
def mlp2(x, out, k, hd, n, xs, cs, ys, x_scale, wc_col, cin, cbias, we, ce, wc, wcs):
    prm = struct.pack("<8I", T, 0, W, 0, 1 if x_scale else 0, 0, 0, 0)
    return {"function": "pwin_mlp2_tokens", "grid": [(T + 127) // 128, 1, 1], "block": [256, 1, 1],
            "buffers": [dict(index=0, **x), dict(index=1, **we), dict(index=2, **ce), dict(index=3, **wc), dict(index=4, **cin), dict(index=5, **cbias), dict(index=6, **out), dict(index=8, **(x_scale or pw("Zero128")))],
            "bytes": [{"index": 7, "u32": list(struct.unpack("<8I", prm))}],
            "constants": {"0": k, "1": n, "2": xs, "3": cs, "7": ys, "14": hd, "15": wcs}}
def merge(x, a, channels, xs, a_stride):
    prm = struct.pack("<6I", W, H, xs, a_stride, channels, 0)
    return {"function": "pwin_merge2x2c", "grid": [(T + 255) // 256, 1, 1], "block": [256, 1, 1],
            "buffers": [dict(index=0, **x), dict(index=1, **a)], "bytes": [{"index": 2, "u32": list(struct.unpack("<6I", prm))}]}
# P.V fp16 handoff table (bit b': round after key block b'), index 2b + n; enc4: (0,0) 3 handoffs, (0,1) after b'=2, (1,0) 3, (2,0) 3, rest fused
SPLIT_ENC4 = [7, 4, 7, 0, 7, 0, 0, 0]
d = [
  norm(IN, sc(N1), pw("G1"), 128),
  gemm(sc(N1), pw("Wqkv"), pw("C0z"), sc(QKV), T, 128, 384, 128, 384, 384, split=4, nsplit=4),
  attn(sc(QKV), pw("Bpos0"), sc(O), 4, SPLIT_ENC4, PADX, PADY),
  gemm(sc(O), pw("Wo"), pw("Cbo"), sc(U), T, 128, 128, 128, 128, 128, split=4, epi=3, cin=IN, r_stride=128, nsplit=2),
]
# unfused MLP: the expand runs once instead of once per output slice (see make-dec5.py)
d += [
  {"function": "pwin_norm128", "grid": [(T + 255) // 256, 1, 1], "block": [256, 1, 1],
   "buffers": [dict(index=0, **sc(U)), dict(index=1, **sc(SCL)), dict(index=2, **pw("S1"))],
   "bytes": [{"index": 3, "u32": list(struct.unpack("<4I", struct.pack("<4I", T, 128, 128, 1)))}]},
  gemm(sc(SCL), pw("We"), pw("Cbe"), sc(HID), T, 128, 512, 128, 512, 512, epi=2, nsplit=8),
  gemm(sc(HID), pw("Wc"), pw("C0z"), ZOUT, T, 512, 128, 512, 128, 128,
       cmode=1, cin=sc(U), cbias=pw("Bc"), split=4, nsplit=8)]
d += [merge(ZOUT, sc(MER), 128, 128, 512),
      gemm(sc(MER), pw("W5s"), pw("Cb5s"), OOUT, T // 4, 512, 160, 512, 160, 160, width=W // 2, nsplit=5)]
pipe = {"library": "build/pwin_pipeline/pwin_all.metallib", "scratch_bytes": scratch_bytes,
        "files": [{"name": "pweights", "path": "work/native/enc4/pipeline/weights-enc4.bin"}], "dispatches": d}
json.dump(pipe, open(OUT, "w"), indent=1)
print("enc4 dispatches", len(d), "scratch MB", scratch_bytes / 1e6, "->", OUT)
