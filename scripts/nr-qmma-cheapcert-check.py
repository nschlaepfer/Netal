#!/usr/bin/env python3
"""CPU check of the one-pack integer certificate for the three-product QMMA path. For an exact sum S (units 2^-24) and grid G = 2^(max_exp-25):
half ulp in units from |S|'s bit length (normal: 2^(bitlen-11) units for bitlen > 10, subnormal region: 1 unit); r = |S| mod ulp;
certified iff 33G < |r - ulp/2| and 33G < (|S| - 2^(bl-1)) + ulp/4 (no rounding midpoint inside [S-33G, S+33G], including the lower binade's last midpoint).
Soundness: certified => RNE_half(S) == accepted rule result (checked against the RTX-validated rule on every element); coverage vs the two-pack interval
certificate. Sources: RTX 64-case registers, the 80 adversarial cases, and a launch-28 census sample."""
import argparse, importlib.util, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
spec = importlib.util.spec_from_file_location("rc", os.path.join(HERE, "nr-qmma-rule-check.py")); RC = importlib.util.module_from_spec(spec); spec.loader.exec_module(RC)
def e4m3_int(b):
    (sig, q), neg = RC.e4m3_fields(b)
    if sig == 0: return 0, None
    m = sig << (q + 9); return (-m if neg else m), q + sig.bit_length() - 1
def terms_of(arow, bcol, cbits):
    ma = [e4m3_int(b) for b in arow]; mb = [e4m3_int(b) for b in bcol]; (cs, cq), cn = RC.half_fields(cbits)
    terms = []; maxexp = None
    for (m1, e1), (m2, e2) in zip(ma, mb):
        if m1 == 0 or m2 == 0: continue
        maxexp = e1 + e2 if maxexp is None else max(maxexp, e1 + e2); terms.append(m1 * m2 << 6)
    if cs: ce = cq + cs.bit_length() - 1; maxexp = ce if maxexp is None else max(maxexp, ce); terms.append(((-cs if cn else cs) << (cq + 24)))
    return sum(terms), maxexp
def cheap_cert(S, G):
    if G == 0: return True
    mag = abs(S); bl = mag.bit_length()
    ulp = 1 << (bl - 11) if bl > 10 else 1          # half has 11 significant bits; below 2^10 units (subnormal half region) the ulp is one unit
    r = mag % ulp; half_ulp = ulp // 2
    # rounding is monotone; the result is constant on [S-33G, S+33G] iff no half midpoint lies inside. In-binade midpoints sit at r = ulp/2;
    # the lower binade's last midpoint sits ulp/4 below the binade edge 2^(bl-1); the upper binade's first midpoint sits ulp above 2^bl
    # (already excluded by the in-binade test). Subnormal region (ulp = 1 unit): integers S are never within 1/2 unit of a midpoint unless G = 0.
    if bl <= 10: return False
    edge_dist = mag - (1 << (bl - 1))
    return (33 * G < abs(r - half_ulp)) and (33 * G < edge_dist + ulp // 4)
def run_elements(elements, stats):
    for arow, bcol, cbits, actual in elements:
        ref = RC.rule_reference(arow, bcol, cbits); stats["elements"] += 1
        if actual is not None and ref != actual: stats["rule_vs_actual_mismatch"] += 1
        S, maxexp = terms_of(arow, bcol, cbits)
        if maxexp is None: stats["zero"] += 1; continue
        gexp = maxexp - 25; G = (1 << (gexp + 24)) if gexp + 24 >= 0 else 0
        rne = RC.fixed_to_half_rne(S, -24)
        iv = RC.fixed_to_half_rne(S - 33 * G, -24) == RC.fixed_to_half_rne(S + 33 * G, -24)
        ch = cheap_cert(S, G)
        stats["interval_certified"] += iv; stats["cheap_certified"] += ch
        if ch and rne != ref: stats["cheap_certified_wrong"] += 1
        if ch and not iv: stats["cheap_not_subset_of_interval"] += 1
        if iv and rne != ref: stats["interval_certified_wrong"] += 1
def rtx_elements(rtx_dir, key="5a-a"):
    inp = np.fromfile(f"{rtx_dir}/output/{key}.input-after.bin", dtype=np.uint32).reshape(-1, 32, 8); out = np.fromfile(f"{rtx_dir}/output/{key}.bin", dtype=np.uint32).reshape(-1, 32, 2)
    return frag_elements(inp, out)
def frag_elements(inp, out):
    els = []
    for ci in range(inp.shape[0]):
        lanes = inp[ci]; A = np.zeros((16, 32), dtype=np.int64); B = np.zeros((32, 8), dtype=np.int64); C = np.zeros((16, 8), dtype=np.int64)
        for l in range(32):
            g, t = l // 4, l % 4; a4 = [int(x) for x in lanes[l][0:4]]; b2 = [int(x) for x in lanes[l][4:6]]; c2 = [int(x) for x in lanes[l][6:8]]
            for j in range(4):
                A[g, 4 * t + j] = (a4[0] >> (8 * j)) & 255; A[g + 8, 4 * t + j] = (a4[1] >> (8 * j)) & 255; A[g, 16 + 4 * t + j] = (a4[2] >> (8 * j)) & 255; A[g + 8, 16 + 4 * t + j] = (a4[3] >> (8 * j)) & 255
                B[4 * t + j, g] = (b2[0] >> (8 * j)) & 255; B[16 + 4 * t + j, g] = (b2[1] >> (8 * j)) & 255
            C[g, 2 * t] = c2[0] & 0xffff; C[g, 2 * t + 1] = c2[0] >> 16; C[g + 8, 2 * t] = c2[1] & 0xffff; C[g + 8, 2 * t + 1] = c2[1] >> 16
        for r in range(16):
            for c in range(8):
                l = r % 8 * 4 + c // 2; actual = (int(out[ci, l, r // 8]) >> (16 * (c % 2))) & 0xffff if out is not None else None
                els.append(([int(x) for x in A[r]], [int(x) for x in B[:, c]], int(C[r, c]), actual))
    return els
def main():
    p = argparse.ArgumentParser(); p.add_argument("--rtx-dir", required=True); p.add_argument("--adversarial", default=os.path.join(ROOT, "build/nr_ptx/qmma-adversarial-v1")); p.add_argument("--census-tokens", default="0,777,2159"); a = p.parse_args()
    report = {}
    for name, els in (("rtx64", rtx_elements(a.rtx_dir)), ("adversarial80", frag_elements(np.fromfile(os.path.join(a.adversarial, "cases.input.bin"), dtype=np.uint32).reshape(-1, 32, 8), np.fromfile(os.path.join(a.adversarial, "cases.expected.bin"), dtype=np.uint32).reshape(-1, 32, 2)))):
        st = {"elements": 0, "zero": 0, "rule_vs_actual_mismatch": 0, "interval_certified": 0, "cheap_certified": 0, "cheap_certified_wrong": 0, "interval_certified_wrong": 0, "cheap_not_subset_of_interval": 0}; run_elements(els, st); report[name] = st
    # launch-28 census sample (replay of the accepted rule step by step, like nr-tp-census.py)
    spec2 = importlib.util.spec_from_file_location("cen", os.path.join(HERE, "nr-tp-census.py")); CEN = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(CEN)
    ref = os.path.join(ROOT, "build/nr_chain/attempt-v7/middle-a5-a/a5-a"); x = np.fromfile(os.path.join(ref, "launch-027", "out.bin"), dtype=np.uint8); res = np.fromfile(os.path.join(ref, "launch-026", "out.bin"), dtype=np.uint8)
    w = np.fromfile(os.path.join(ROOT, "build/nr_split/fixtures/launch-028/weights.bin"), dtype=np.uint8); scale = w[262144:262144 + 1024].view(np.uint16)
    st = {"elements": 0, "zero": 0, "rule_vs_actual_mismatch": 0, "interval_certified": 0, "cheap_certified": 0, "cheap_certified_wrong": 0, "interval_certified_wrong": 0, "cheap_not_subset_of_interval": 0}; els = []
    for t in [int(v) for v in a.census_tokens.split(",")]:
        for n in range(512):
            acc = CEN.hmul(CEN.e4m3_half_bits(int(res[CEN.a_index(t, CEN.cfd(n))])), int(scale[n]))
            for s in range(16):
                arow = [int(x[CEN.a_index(t, 32 * s + j)]) for j in range(32)]; bcol = [int(w[s * 16384 + (n // 64) * 2048 + CEN.w_index(j, n % 64)]) for j in range(32)]
                els.append((arow, bcol, acc, None)); acc = RC.rule_reference(arow, bcol, acc)
    run_elements(els, st); report["launch28_census"] = st
    print(json.dumps(report, indent=1)); json.dump(report, open(os.path.join(ROOT, "build/nr_ptx/qmma-cheapcert-check.json"), "w"), indent=1)
    ok = all(v["cheap_certified_wrong"] == 0 and v["interval_certified_wrong"] == 0 for v in report.values()); print("SOUND" if ok else "UNSOUND"); sys.exit(0 if ok else 1)
if __name__ == "__main__": main()
