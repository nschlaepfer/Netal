#!/usr/bin/env python3
"""Inspect a live FG dump (NEURALSTUDIO_FG_DUMP_DIR): motion/depth statistics and PNGs of the
runtime's inputs and generated output next to the game's frame."""
import glob, os, re, sys
import numpy as np
from PIL import Image

d = sys.argv[1]
def load(name):
    files = sorted(glob.glob(os.path.join(d, f"*-{name}-*.bin")))
    out = []
    for f in files:
        m = re.search(r"f(\d+)-%s-(\d+)x(\d+)-f(\d+)\.bin" % name, os.path.basename(f))
        frame, w, h, fmt = map(int, m.groups())
        out.append((frame, w, h, fmt, np.fromfile(f, dtype=np.uint8)))
    return out

for frame, w, h, fmt, raw in load("motion"):
    v = raw.view(np.float16).reshape(h, w, 2).astype(np.float32)
    mag = np.hypot(v[..., 0], v[..., 1])
    print(f"frame {frame} motion(640px units): mean x={v[...,0].mean():+.3f} y={v[...,1].mean():+.3f} "
          f"|v| p50={np.percentile(mag,50):.3f} p90={np.percentile(mag,90):.3f} max={mag.max():.2f} nonfinite={(~np.isfinite(v)).sum()}")
for frame, w, h, fmt, raw in load("depth"):
    z = raw.view(np.float32).reshape(h, w)
    print(f"frame {frame} depth: min={z.min():.5f} p50={np.percentile(z,50):.5f} max={z.max():.5f} zeros={(z==0).mean():.3f}")
def rgba8_png(name):
    for frame, w, h, fmt, raw in load(name):
        img = raw.reshape(h, w, 4)[..., :3]
        Image.fromarray(img).save(os.path.join(d, f"f{frame}-{name}.png"))
for n in ("color", "prev", "out"): rgba8_png(n)
def bgr10a2_png(name):
    for frame, w, h, fmt, raw in load(name):
        p = raw.view(np.uint32).reshape(h, w)
        b = (p & 1023) >> 2; g = ((p >> 10) & 1023) >> 2; r = ((p >> 20) & 1023) >> 2
        Image.fromarray(np.stack([r, g, b], -1).astype(np.uint8)).save(os.path.join(d, f"f{frame}-{name}.png"))
for n in ("gamecolor", "gameout"): bgr10a2_png(n)
# generated vs mean-of-neighbours error at 640x360
cols = {f: raw.reshape(h, w, 4)[..., :3].astype(np.float32) for f, w, h, fmt, raw in load("color")}
prevs = {f: raw.reshape(h, w, 4)[..., :3].astype(np.float32) for f, w, h, fmt, raw in load("prev")}
outs = {f: raw.reshape(h, w, 4)[..., :3].astype(np.float32) for f, w, h, fmt, raw in load("out")}
for f in sorted(outs):
    if f in cols and f in prevs:
        mean = 0.5 * (cols[f] + prevs[f])
        print(f"frame {f} out vs mean(prev,cur): mae={np.abs(outs[f]-mean).mean():.2f}  out vs cur: {np.abs(outs[f]-cols[f]).mean():.2f}  prev vs cur: {np.abs(prevs[f]-cols[f]).mean():.2f}  out range {outs[f].min():.0f}..{outs[f].max():.0f}")
print("pngs written to", d)
