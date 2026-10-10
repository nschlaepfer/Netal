"""ANE placement against token count, and against the shifted window.

The U-Net is not one size. enc0 dispatches 1129 threadgroups at 1080p and dec5 dispatches one: the
deep layers are tiny. If the planner takes the wide layers and refuses the narrow ones, the handoff
lands *between* layers, which is the failure the SR port was warned about.
"""
import numpy as np, coremltools as ct, sys
from coremltools.converters.mil import Builder as mb

rng = np.random.default_rng(0)
def w(*s): return rng.standard_normal(s).astype(np.float32) * 0.02

def build(N, C, shift):
    Wqkv, Wproj = w(3*C, C, 1, 1), w(C, C, 1, 1)
    Wup, Wdown = w(4*C, C, 1, 1), w(C, 4*C, 1, 1)
    k = min(32, N // 2)
    @mb.program(input_specs=[mb.TensorSpec(shape=(1, C, 1, N))], opset_version=ct.target.iOS18)
    def prog(x):
        y = mb.layer_norm(x=x, axes=[1], epsilon=1e-5)
        if shift:                       # the shifted window, as the roll it actually is
            y = mb.concat(values=[mb.slice_by_index(x=y, begin=[0,0,0,k], end=[1,C,1,N]),
                                  mb.slice_by_index(x=y, begin=[0,0,0,0], end=[1,C,1,k])], axis=3)
        qkv = mb.conv(x=y, weight=Wqkv)
        q, kk, v = mb.split(x=qkv, num_splits=3, axis=1)
        q = mb.reshape(x=q, shape=(1, C, N)); kk = mb.reshape(x=kk, shape=(1, C, N))
        v = mb.reshape(x=v, shape=(1, C, N))
        a = mb.softmax(x=mb.matmul(x=q, y=kk, transpose_x=True), axis=-1)
        o = mb.reshape(x=mb.matmul(x=v, y=a, transpose_y=True), shape=(1, C, 1, N))
        o = mb.conv(x=o, weight=Wproj)
        if shift:
            o = mb.concat(values=[mb.slice_by_index(x=o, begin=[0,0,0,N-k], end=[1,C,1,N]),
                                  mb.slice_by_index(x=o, begin=[0,0,0,0], end=[1,C,1,N-k])], axis=3)
        y = mb.add(x=x, y=o)
        z = mb.layer_norm(x=y, axes=[1], epsilon=1e-5)
        z = mb.conv(x=mb.gelu(x=mb.conv(x=z, weight=Wup)), weight=Wdown)
        return mb.add(x=y, y=z)
    return prog

# (label, tokens, channels) chosen from the real layer widths: norm32..norm160
cases = [("dec5-ish", 64, 160), ("enc4-ish", 96, 128), ("enc3-ish", 288, 96),
         ("dec2-ish", 2176, 64), ("dec1-ish", 8640, 64), ("dec0-ish", 17280, 32)]
for label, N, C in cases:
    for shift in (False, True):
        name = f"{label}{'-shift' if shift else ''}"
        try:
            m = ct.convert(build(N, C, shift), minimum_deployment_target=ct.target.macOS15,
                           compute_units=ct.ComputeUnit.CPU_AND_NE,
                           compute_precision=ct.precision.FLOAT16)
            m.save(f"sw-{name}.mlpackage"); print("wrote", name, f"N={N} C={C}")
        except Exception as e:
            print("FAILED", name, str(e)[:100])
