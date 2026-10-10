"""One Core ML convolution per FG launch shape, from src/fg_conv/FAMILY_FRONTIER.md and the
custom block0/block1 extents. Real shapes, because the first ANE probe today used a made-up size
and returned the opposite answer."""
import numpy as np, coremltools as ct
from coremltools.converters.mil import Builder as mb

rng = np.random.default_rng(0)
# label, H, W, kernel, Cin, Cout
SHAPES = [
    ("L20  160x96 3x3", 96, 160, 3, 48, 32),
    ("L22   80x48 1x1", 48,  80, 1, 32, 64),
    ("L24   40x24 3x3", 24,  40, 3, 64, 128),
    ("L26   20x12 3x3", 12,  20, 3, 128, 128),
    ("L28    10x6 3x3",  6,  10, 3, 128, 256),
    ("L30     5x3 1x1",  3,   5, 1, 256, 512),
    ("L34    10x6 1x1",  6,  10, 1, 256, 256),
    ("L50  160x96 1x1", 96, 160, 1, 32, 32),
    # custom block extents: 640x368 and its 320x184 downsample, channels 8..96
    ("cust 640x368 3x3", 368, 640, 3, 8, 16),
    ("cust 320x184 3x3", 184, 320, 3, 16, 32),
    ("cust 320x184 3x3b",184, 320, 3, 32, 64),
    ("cust 160x92  3x3", 92, 160, 3, 64, 96),
]

for label, H, W, K, Cin, Cout in SHAPES:
    weight = (rng.standard_normal((Cout, Cin, K, K)).astype(np.float32) * 0.02)
    pad = K // 2
    @mb.program(input_specs=[mb.TensorSpec(shape=(1, Cin, H, W))], opset_version=ct.target.iOS18)
    def prog(x):
        y = mb.conv(x=x, weight=weight, pad=[pad, pad, pad, pad], pad_type="custom")
        return mb.leaky_relu(x=y, alpha=0.01)
    m = ct.convert(prog, minimum_deployment_target=ct.target.macOS15,
                   compute_units=ct.ComputeUnit.CPU_AND_NE,
                   compute_precision=ct.precision.FLOAT16)
    safe = label.split()[0] + "-" + f"{W}x{H}x{Cin}to{Cout}k{K}"
    m.save(f"fg-{safe}.mlpackage")
    print(f"{label:20s} -> fg-{safe}.mlpackage")
