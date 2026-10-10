#!/usr/bin/env python3
"""PSNR of hardware-mode replay outputs against the exact replay on the same dumped frames.
   fg-mode-psnr.py <dump-dir> <exact-tag> <hw-tag> <first> <last>"""
import sys, math
import numpy as np
d, exact, hw, first, last = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
worst = 1e9
for f in range(first + 1, last + 1):
    a = np.fromfile(f"{d}/{exact}-f{f}-out-640x360-f70.bin", dtype=np.uint8).reshape(360, 640, 4)[..., :3].astype(np.float64)
    b = np.fromfile(f"{d}/{hw}-f{f}-out-640x360-f70.bin", dtype=np.uint8).reshape(360, 640, 4)[..., :3].astype(np.float64)
    mse = ((a - b) ** 2).mean()
    psnr = 99.0 if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)
    diff = np.abs(a - b)
    print(f"frame {f}: PSNR {psnr:.2f} dB  mean|d| {diff.mean():.3f}  max|d| {diff.max():.0f}  frac>2 {(diff.max(-1) > 2).mean():.4f}")
    worst = min(worst, psnr)
print(f"worst PSNR {worst:.2f} dB")
