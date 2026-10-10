#!/usr/bin/env python3
"""Characterise the interpolator's UI texture from a live FG dump: alpha levels, premultiplication,
and how UI pixels relate to the composited game colour. BGR10A2: B[0:10) G[10:20) R[20:30) A[30:32)."""
import glob, os, sys
import numpy as np
from PIL import Image
d = sys.argv[1]
def bgr10a2(path):
    p = np.fromfile(path, dtype=np.uint32)
    b = (p & 1023) / 1023.0; g = ((p >> 10) & 1023) / 1023.0; r = ((p >> 20) & 1023) / 1023.0; a = (p >> 30) / 3.0
    return r, g, b, a
for ui_path in sorted(glob.glob(os.path.join(d, "f6*-gameui-*.bin")))[:2]:
    frame = os.path.basename(ui_path).split("-")[0]
    w, h = map(int, os.path.basename(ui_path).split("-")[2].split("x"))
    r, g, b, a = [x.reshape(h, w) for x in bgr10a2(ui_path)]
    color_path = glob.glob(os.path.join(d, f"{frame}-gamecolor-*.bin"))[0]
    cr, cg, cb, _ = [x.reshape(h, w) for x in bgr10a2(color_path)]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    print(f"{frame}: alpha levels", {lvl: int((a == lvl / 3).sum()) for lvl in range(4)},
          f"| rgb>0 where a==0: {int(((lum > 0.002) & (a == 0)).sum())} px",
          f"| rgb>0 anywhere: {int((lum > 0.002).sum())} px")
    for lvl in (1, 2, 3):
        m = a == lvl / 3
        if m.sum() == 0: continue
        # premultiplied would keep rgb <= alpha; straight would not
        print(f"   a={lvl}/3: n={int(m.sum())} max lum {lum[m].max():.3f} mean lum {lum[m].mean():.3f}"
              f" | game colour at these px: mean {((0.2126*cr+0.7152*cg+0.0722*cb)[m]).mean():.3f}"
              f" | mean |ui-color| {np.abs(np.stack([r-cr,g-cg,b-cb],-1)[m]).mean():.3f}")
    ys, xs = np.nonzero(a > 0)
    if len(xs): print(f"   UI bbox x {xs.min()}..{xs.max()} y {ys.min()}..{ys.max()}")
    # PNG: UI rgb, alpha mask, and rgb*alpha over grey
    rgb = (np.stack([r, g, b], -1) * 255).astype(np.uint8)
    Image.fromarray(rgb).save(os.path.join(d, f"{frame}-gameui-rgb.png"))
    Image.fromarray((a * 255).astype(np.uint8)).save(os.path.join(d, f"{frame}-gameui-alpha.png"))
    comp = (np.stack([r, g, b], -1) * a[..., None] + 0.4 * (1 - a[..., None]))
    Image.fromarray((comp * 255).astype(np.uint8)).save(os.path.join(d, f"{frame}-gameui-overgrey.png"))
print("done")
