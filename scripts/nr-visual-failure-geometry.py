#!/usr/bin/env python3
"""Measure the geometry of the NR visual failure in a captured live frame, and test whether it tracks
the input's value range.

Takes a capture directory written by the live adapter (capture.json + model-input.rgba16f +
model-output.rgba16f) and reports:
  * the vertical seam columns where the OUTPUT has a discontinuity, their period and phase;
  * whether that period drifts with row (a stride/shear bug) or is stationary (a fixed partition);
  * the dominant horizontal seam rows;
  * the correlation between local input over-range (>1.0, the HDR hypothesis) and local corruption,
    plus the corruption present in regions where the input is exactly zero.

Read-only. Comparison and measurement only; nothing here feeds execution.
"""
import argparse, json, os
import numpy as np


def load_rgba16f(path, w, h):
    a = np.fromfile(path, dtype="<f2").astype(np.float32)
    if a.size != w * h * 4:
        raise SystemExit(f"{path}: {a.size} halves, expected {w*h*4}")
    return a.reshape(h, w, 4)[..., :3]


def seams(a, axis, top=30):
    d = np.abs(np.diff(a, axis=axis)).mean(tuple(i for i in (0, 1, 2) if i != axis) if False else ((0, 2) if axis == 1 else (1, 2)))
    return d


def period_and_phase(prof, W, min_period=16, max_period=480):
    """Dominant periodic component of a 1D profile within a period band, cross-checked by autocorrelation.

    The band matters: a raw argmax lands on the 2 px pixel-level dither, which is a different phenomenon
    from the coarse seam lattice this is meant to measure."""
    x = prof - prof.mean()
    F = np.abs(np.fft.rfft(x))
    lo, hi = max(1, int(W // max_period)), int(W // min_period)
    band = np.zeros_like(F)
    band[lo:hi + 1] = F[lo:hi + 1]
    k = int(np.argmax(band))
    ac = np.correlate(x, x, "full")[len(x) - 1:]
    ac = ac / ac[0]
    peaks = [(l, float(ac[l])) for l in range(min_period, min(max_period + 1, len(ac) - 1))
             if ac[l] > ac[l - 1] and ac[l] > ac[l + 1] and ac[l] > 0.2]
    peaks.sort(key=lambda t: -t[1])
    return k, W / k if k else float("inf"), float(F[k]), peaks[:6]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("capture", help="directory holding capture.json and the model-*.rgba16f planes")
    p.add_argument("--output", default=None)
    a = p.parse_args()

    meta = json.load(open(os.path.join(a.capture, "capture.json")))
    w, h = meta["model_extent"]
    mi = load_rgba16f(os.path.join(a.capture, "model-input.rgba16f"), w, h)
    mo = load_rgba16f(os.path.join(a.capture, "model-output.rgba16f"), w, h)

    # --- vertical seams: columns where the output jumps
    dcol = np.abs(np.diff(mo, axis=1)).mean((0, 2))
    # the seam lattice rides on the hue deviation, not on scene luminance; measure the period there
    hue = mo[..., 2] - np.maximum(mo[..., 0], mo[..., 1])
    lattice_prof = np.abs(np.diff(hue, axis=1)).mean(0)
    th = dcol.mean() + 4 * dcol.std()
    strong = (np.where(dcol > th)[0] + 1).tolist()
    top = sorted((np.argsort(dcol)[::-1][:30] + 1).tolist())
    k, per, amp, acpeaks = period_and_phase(lattice_prof, w)
    k_grad, per_grad, amp_grad, _ = period_and_phase(dcol, w)

    # --- is the lattice stationary or sheared? phase of the dominant component per row
    dev = mo[..., 2] - np.maximum(mo[..., 0], mo[..., 1])
    e = np.exp(-2j * np.pi * k * np.arange(w) / w)
    c = dev @ e
    amp_r, ph_r = np.abs(c), np.angle(c)
    sel = np.where(amp_r > np.percentile(amp_r, 97))[0]
    shear = float("nan")
    if sel.size > 5:
        shear = float(-np.polyfit(sel, np.unwrap(ph_r[sel]), 1)[0] / (2 * np.pi) * per)

    # --- horizontal seams
    drow = np.abs(np.diff(mo, axis=0)).mean((1, 2))
    rowtop = sorted((np.argsort(drow)[::-1][:12] + 1).tolist())

    # --- does corruption track input over-range?
    inmax = mi.max(2)
    over = inmax > 1.0
    red = mo[..., 0] - np.maximum(mo[..., 1], mo[..., 2])
    blue = np.abs(mo[..., 2] - np.maximum(mo[..., 0], mo[..., 1]))

    def tiled(x, t, f=np.mean):
        hh, ww = (h // t) * t, (w // t) * t
        return f(x[:hh, :ww].reshape(h // t, t, w // t, t), axis=(1, 3))

    def cc(x, y):
        return float(np.corrcoef(x.ravel(), y.ravel())[0, 1])

    corr = {str(t): dict(over_range_vs_red=cc(tiled(over.astype(np.float32), t), tiled(red, t)),
                         log_inmax_vs_red=cc(np.log1p(tiled(inmax, t, np.max)), tiled(red, t)),
                         log_inmax_vs_blue=cc(np.log1p(tiled(inmax, t, np.max)), tiled(blue, t)))
            for t in (8, 32, 120)}

    zero = ~(mi > 0).any(2)
    zero_report = None
    if zero.any():
        zero_report = dict(pixels=int(zero.sum()),
                           output_max=float(mo[zero].max()),
                           output_mean_rgb=[float(v) for v in mo[zero].mean(0)],
                           output_nonzero_fraction=float((mo[zero] > 0).mean()))

    report = dict(schema="neuralstudio.nr.visual-failure-geometry.v1",
                  capture=os.path.abspath(a.capture), frame=meta.get("frame"), stream=meta.get("stream"),
                  model_extent=[w, h], model_content=meta.get("model_content"),
                  input_range=[float(mi.min()), float(mi.max())],
                  input_over_range_fraction=float(over.mean()),
                  vertical_seam_columns_4sigma=strong, vertical_seam_columns_top30=top,
                  dominant_column_period_px=per, dominant_column_harmonic=k, dominant_column_amplitude=amp,
                  gradient_column_period_px=per_grad, gradient_column_harmonic=k_grad,
                  autocorrelation_peaks=acpeaks, shear_px_per_row=shear,
                  horizontal_seam_rows_top12=rowtop,
                  corruption_vs_input_correlation=corr,
                  zero_input_region=zero_report)
    print(json.dumps(report, indent=1))
    if a.output:
        os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
        json.dump(report, open(a.output, "w"), indent=1)
        print(f"wrote {a.output}")


if __name__ == "__main__":
    main()
