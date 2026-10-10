#!/usr/bin/env python3
"""CPU discriminator for QMMA (mma.sync m16n8k32 e4m3 -> f16) candidate formulations against Codex's RTX 5090 run of the 64-case
operand probe (build/nr_qmma_operands/discriminators-rtx-v1: input-after.bin = per case, per lane a[4] b[2] c[2] raw registers;
<key>.bin = d[2]). Every candidate is evaluated per output element with exact Python integers / IEEE binary32 semantics (numpy float32)."""
import argparse, json, struct, sys
import numpy as np
def e4m3_fields(b):
    e = (b >> 3) & 15; m = b & 7
    return ((m, -9) if e == 0 else (8 + m, e - 10)), bool(b & 0x80)          # (sig, q): value = sig * 2^q
def e4m3_float(b):
    (sig, q), neg = e4m3_fields(b); v = float(sig) * 2.0 ** q; return -v if neg else v
def half_fields(h):
    e = (h >> 10) & 31; f = h & 1023
    if e == 0: return (f, -24), bool(h & 0x8000)
    return (1024 + f, e - 25), bool(h & 0x8000)
def half_from_bits(h): return float(np.frombuffer(struct.pack("<H", h), dtype=np.float16)[0])
def fixed_to_half_rne(total, scale):
    """exact signed integer total * 2^scale -> half bits, RNE (mirrors nr_fixed_to_half_rne)"""
    if total == 0: return 0
    neg = total < 0; mag = -total if neg else total
    lead = mag.bit_length() - 1; exponent = lead + scale
    def rshift_rne(v, sh):
        if sh <= 0: return v << (-sh)
        q, r = v >> sh, v & ((1 << sh) - 1); half = 1 << (sh - 1)
        if r > half or (r == half and (q & 1)): q += 1
        return q
    if exponent >= -14:
        sig = rshift_rne(mag, lead - 10)
        if sig >= 2048: sig >>= 1; exponent += 1
        res = 0x7c00 if exponent > 15 else (((exponent + 15) << 10) | (sig - 1024))
    else:
        sig = rshift_rne(mag, -(scale + 24)); res = 0x0400 if sig >= 1024 else sig
    return res | (0x8000 if neg else 0)
def rule_reference(arow, bcol, cbits):
    """the accepted fixed-point rule: max_exp over nonzero products of floor(log2|a|)+floor(log2|b|) and floor(log2|C|)"""
    terms = []; maxexp = None
    for ab, bb in zip(arow, bcol):
        (sa, qa), na = e4m3_fields(ab); (sb, qb), nb = e4m3_fields(bb)
        if sa == 0 or sb == 0: continue
        ex = (qa + sa.bit_length() - 1) + (qb + sb.bit_length() - 1); maxexp = ex if maxexp is None else max(maxexp, ex)
        terms.append((sa * sb, qa + qb, na != nb))
    (cs, cq), cn = half_fields(cbits)
    if cs: ce = cq + cs.bit_length() - 1; maxexp = ce if maxexp is None else max(maxexp, ce); terms.append((cs, cq, cn))
    if maxexp is None: return 0
    scale = maxexp - 25; total = 0
    for sig, q, neg in terms:
        sh = q - scale; mag = (sig << sh) if sh >= 0 else (sig >> -sh)
        total += -mag if neg else mag
    return fixed_to_half_rne(total, scale)
def rule_fscale(arow, bcol, cbits, operand_sum):
    """candidate 11 in binary32 arithmetic: products in fp32, max_exp from the product exponent (operand_sum=False, as on disk)
    or from the operand exponent sum (operand_sum=True, the corrected form); trunc(p * 2^(25-max_exp)); u32 pos/neg sums."""
    f32 = np.float32; A = [f32(e4m3_float(b)) for b in arow]; B = [f32(e4m3_float(b)) for b in bcol]
    P = [f32(a * b) for a, b in zip(A, B)]; maxexp = -127
    for a, b, p in zip(A, B, P):
        if p == 0: continue
        if operand_sum:
            ea = int((int(a.view(np.uint32)) >> 23) & 255) - 127; eb = int((int(b.view(np.uint32)) >> 23) & 255) - 127; maxexp = max(maxexp, ea + eb)
        else: maxexp = max(maxexp, int((int(p.view(np.uint32)) >> 23) & 255) - 127)
    cf = f32(half_from_bits(cbits))
    if cf != 0: maxexp = max(maxexp, int((int(cf.view(np.uint32)) >> 23) & 255) - 127)
    if maxexp == -127: return 0
    scale = maxexp - 25; fs = f32(2.0 ** (25 - maxexp)); pos = neg = 0
    for t in [f32(np.trunc(p * fs)) for p in P] + [f32(np.trunc(cf * fs))]:
        m = int(abs(float(t)))
        if t < 0: neg += m
        else: pos += m
    assert pos < 2 ** 32 and neg < 2 ** 32, "u32 bound exceeded"
    return fixed_to_half_rne(pos - neg, scale)
def main():
    p = argparse.ArgumentParser(); p.add_argument("--rtx-dir", required=True); p.add_argument("--key", default="5a-a"); p.add_argument("--cases", type=int, default=64); a = p.parse_args()
    inp = np.fromfile(f"{a.rtx_dir}/output/{a.key}.input-after.bin", dtype=np.uint32).reshape(a.cases, 32, 8)
    out = np.fromfile(f"{a.rtx_dir}/output/{a.key}.bin", dtype=np.uint32).reshape(a.cases, 32, 2)
    cands = {"reference_operand_sum_integer": lambda ar, bc, c: rule_reference(ar, bc, c),
             "fscale_product_exponent (candidate 11 as on disk)": lambda ar, bc, c: rule_fscale(ar, bc, c, False),
             "fscale_operand_exponent_sum (corrected)": lambda ar, bc, c: rule_fscale(ar, bc, c, True)}
    mism = {k: [] for k in cands}; regs = 0
    for ci in range(a.cases):
        lanes = inp[ci]
        # rebuild the A rows / B cols from the raw fragments (fragment layout of nr-qmma-discriminate.py)
        A = np.zeros((16, 32), dtype=np.int64); B = np.zeros((32, 8), dtype=np.int64); C = np.zeros((16, 8), dtype=np.int64)
        for l in range(32):
            g, t = l // 4, l % 4; a4 = [int(x) for x in lanes[l][0:4]]; b2 = [int(x) for x in lanes[l][4:6]]; c2 = [int(x) for x in lanes[l][6:8]]
            for j in range(4):
                A[g, 4 * t + j] = (a4[0] >> (8 * j)) & 255; A[g + 8, 4 * t + j] = (a4[1] >> (8 * j)) & 255
                A[g, 16 + 4 * t + j] = (a4[2] >> (8 * j)) & 255; A[g + 8, 16 + 4 * t + j] = (a4[3] >> (8 * j)) & 255
                B[4 * t + j, g] = (b2[0] >> (8 * j)) & 255; B[16 + 4 * t + j, g] = (b2[1] >> (8 * j)) & 255
            C[g, 2 * t] = c2[0] & 0xffff; C[g, 2 * t + 1] = c2[0] >> 16; C[g + 8, 2 * t] = c2[1] & 0xffff; C[g + 8, 2 * t + 1] = c2[1] >> 16
        for l in range(32):
            g, t = l // 4, l % 4
            for reg, r in ((0, g), (1, g + 8)):
                actual = int(out[ci, l, reg]); regs += 1
                for name, fn in cands.items():
                    lo = fn([int(x) for x in A[r]], [int(x) for x in B[:, 2 * t]], int(C[r, 2 * t])); hi = fn([int(x) for x in A[r]], [int(x) for x in B[:, 2 * t + 1]], int(C[r, 2 * t + 1]))
                    pred = lo | (hi << 16)
                    if pred != actual: mism[name].append({"case": ci, "lane": l, "register": reg, "actual": f"{actual:08x}", "predicted": f"{pred:08x}"})
    print(f"registers compared: {regs} ({a.cases} cases x 32 lanes x 2)")
    for name, m in mism.items(): print(f"  {name}: {len(m)} mismatching registers", m[:3])
    json.dump({"schema": "neuralstudio.nr.qmma-rule-check.v1", "rtx_dir": a.rtx_dir, "key": a.key, "registers": regs, "mismatches": {k: len(v) for k, v in mism.items()}, "counterexamples": {k: v[:8] for k, v in mism.items()}}, open("build/nr_ptx/qmma-rule-check.json", "w"), indent=1)
if __name__ == "__main__": main()
