#!/usr/bin/env python3
"""Prove the DLL probe works before any DLL exists.

Synthesises binaries that embed Netal's own model the way a DLL plausibly might, and checks that the
probe reaches the right verdict on each. Without this the probe would be untested code waiting on an
input nobody has, and its first run would be the one that matters.
"""
import os, pathlib, random, struct, subprocess, sys, tempfile, zlib

REPO = pathlib.Path(__file__).resolve().parent.parent
PROBE = REPO / "tools" / "netal-dlss-probe.py"
MODEL = REPO / "work" / "re9-sr-runtime-bundle" / "model"

failures = 0


def check(ok: bool, what: str) -> None:
    global failures
    print("  ok    %s" % what if ok else "  FAIL  %s" % what)
    if not ok:
        failures += 1


def filler(size: int, seed: int) -> bytes:
    """Something a binary might contain: instruction-like bytes, strings and tables - deliberately
    NOT uniform noise, which would be an unrealistically easy thing to tell weights apart from."""
    rng = random.Random(seed)
    out = bytearray()
    while len(out) < size:
        pick = rng.random()
        if pick < 0.55:
            out += bytes(rng.randrange(0x40, 0xC0) for _ in range(rng.randrange(4, 24)))
        elif pick < 0.8:
            out += bytes(rng.choice(b"abcdefghijklmnopqrstuvwxyz_.") for _ in range(rng.randrange(8, 40)))
            out += b"\x00"
        else:
            out += b"\x00" * rng.randrange(8, 64)
    return bytes(out[:size])


def synthesize(path: pathlib.Path, payload: bytes, seed: int) -> int:
    before = filler(3_000_000, seed)
    after = filler(2_000_000, seed + 1)
    path.write_bytes(before + payload + after)
    return len(before)


def run(path: pathlib.Path) -> str:
    result = subprocess.run([sys.executable, str(PROBE), str(path), "--reference",
                             str(REPO / "work" / "re9-sr-runtime-bundle")],
                            capture_output=True, text=True)
    return result.stdout


weights = (MODEL / "weights.bin").read_bytes()
print("reference weights.bin: %d bytes" % len(weights))

with tempfile.TemporaryDirectory() as tmp:
    tmp = pathlib.Path(tmp)

    print("\nstored verbatim")
    offset = synthesize(tmp / "plain.dll", weights, 1)
    out = run(tmp / "plain.dll")
    check("result=extractable" in out, "verdict is extractable")
    check("as verbatim fp16" in out, "encoding identified as verbatim")
    check(("0x%x" % offset) in out, "offset reported correctly (0x%x)" % offset)

    print("\nstored byte-swapped")
    swapped = bytearray(len(weights))
    swapped[0::2] = weights[1::2]
    swapped[1::2] = weights[0::2]
    synthesize(tmp / "swapped.dll", bytes(swapped), 3)
    out = run(tmp / "swapped.dll")
    check("result=extractable" in out, "verdict is extractable")
    check("byte-swapped" in out, "encoding identified as byte-swapped")

    print("\nstored as fp32, narrowed at load")
    values = struct.unpack("<%de" % (len(weights) // 2), weights)
    synthesize(tmp / "fp32.dll", struct.pack("<%df" % len(values), *values), 5)
    out = run(tmp / "fp32.dll")
    check("result=extractable" in out, "verdict is extractable")
    check("as fp32" in out, "encoding identified as fp32")

    print("\na DIFFERENT DLSS version - same layout, different numbers")
    # Perturb every weight slightly: no byte matches, but it is still a weight blob. This is the
    # case that matters most, because it is what every real DLL other than the reference will be.
    rng = random.Random(9)
    shifted = struct.pack("<%de" % len(values), *[v * rng.uniform(0.85, 1.15) for v in values])
    synthesize(tmp / "other.dll", shifted, 7)
    out = run(tmp / "other.dll")
    check("result=candidates" in out, "verdict is candidates, not extractable")
    check("MB  mean |w|" in out, "the weight region is located by its statistics")

    print("\ncompressed - the case the file route cannot serve")
    synthesize(tmp / "packed.dll", zlib.compress(weights, 6), 11)
    out = run(tmp / "packed.dll")
    check("result=not-plain" in out, "verdict is not-plain")

print("\nall checks passed" if failures == 0 else "\n%d FAILURES" % failures)
raise SystemExit(1 if failures else 0)
