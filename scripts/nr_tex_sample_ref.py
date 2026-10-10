"""CPU model of src/nr_temporal/nr_tex_sample.metal (same coordinate conversion, clamping, 8-bit fraction quantisation modes and blend order)."""
import numpy as np
def coord_q8(u, v, w, h, mode=0):
    f32 = np.float32; tx = f32(u) * f32(w) - f32(0.5); ty = f32(v) * f32(h) - f32(0.5); fx0 = np.floor(tx); fy0 = np.floor(ty); fx = tx - fx0; fy = ty - fy0
    q = (lambda z: np.floor(z * f32(256)) / f32(256)) if mode == 0 else (lambda z: np.rint(z * f32(256)) / f32(256))
    fx, fy = q(fx), q(fy); cl = lambda a, n: int(min(max(int(a), 0), n - 1))
    return cl(fx0, w), cl(fy0, h), cl(fx0 + 1, w), cl(fy0 + 1, h), f32(fx), f32(fy)
def sample_q8(img, u, v, mode=0):
    """img: (H, W, C) float32 array of exact texel values; returns float32 (C,) with the fixed blend order of the Metal primitive"""
    h, w = img.shape[:2]; x0, y0, x1, y1, fx, fy = coord_q8(u, v, w, h, mode); f32 = np.float32
    a, b, d, e = img[y0, x0], img[y0, x1], img[y1, x0], img[y1, x1]
    top = a + (b - a) * fx; bot = d + (e - d) * fx; return (top + (bot - top) * fy).astype(f32)   # lerp form: exact when neighbours are equal (clamped edges)


def sample_rtx(img, u, v):
    """fitted RTX rule (2026-09-06): see ptx_warp_emu.tex_bilinear_rtx (single implementation); img float32 (H, W, C); returns float32 (C,)"""
    import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from ptx_warp_emu import tex_bilinear_rtx
    H, W = img.shape[0], img.shape[1]; r = tex_bilinear_rtx(img, W, H, np.array([u], dtype=np.float64), np.array([v], dtype=np.float64))
    return np.array([float(x[0]) for x in r], dtype=np.float32)
