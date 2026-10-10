"""The custom block as one Core ML model - the fair ANE test.

Measuring 25 separate predictions charges Core ML dispatch 25 times, which is not what a real
integration would do. This chains block0's shape sequence into a single model so the ANE pays entry
once. Structure follows the status doc: convPre + downsample, conv0, conv1, an eight-launch conv2
family with residuals, then the two bottom layers.
"""
import numpy as np, coremltools as ct
from coremltools.converters.mil import Builder as mb
rng = np.random.default_rng(0)
def W(o, i, k=3): return rng.standard_normal((o, i, k, k)).astype(np.float32) * 0.02

@mb.program(input_specs=[mb.TensorSpec(shape=(1, 8, 368, 640))], opset_version=ct.target.iOS18)
def block0(x):
    def cv(t, o, i, k=3, pad=1):
        t = mb.conv(x=t, weight=W(o, i, k), pad=[pad]*4, pad_type="custom")
        return mb.leaky_relu(x=t, alpha=0.01)
    def pool(t): return mb.avg_pool(x=t, kernel_sizes=[2, 2], strides=[2, 2], pad_type="valid")
    t = pool(cv(x, 16, 8))                 # convPre  640x368x8  -> 320x184x16
    t = pool(cv(t, 32, 16))                # conv0    320x184    -> 160x92x32
    t = pool(cv(t, 64, 32))                # conv1    160x92     -> 80x46x64
    for _ in range(8):                     # conv2 family, residual on alternating launches
        t = mb.add(x=t, y=cv(t, 64, 64))
    t = mb.upsample_bilinear(x=t, scale_factor_height=2.0, scale_factor_width=2.0)
    t = cv(t, 96, 64)                      # bottom0  -> 160x92x96
    return cv(t, 8, 96)                    # bottom1

m = ct.convert(block0, minimum_deployment_target=ct.target.macOS15,
               compute_units=ct.ComputeUnit.CPU_AND_NE, compute_precision=ct.precision.FLOAT16)
m.save("fgblock0.mlpackage"); print("wrote fgblock0.mlpackage")
