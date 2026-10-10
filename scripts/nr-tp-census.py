#!/usr/bin/env python3
"""Real-operand census for the three-product QMMA proposal on one split conv launch (default 28, proj_512_chained, 2160 tokens x 512 cols,
K=512 in 16 rounded steps). CPU-only. For sampled tokens: replays the accepted rule step by step (reference), checks the replay against the
strict reference output bytes, and counts how many (element, step) results the cheap certificates would certify (interval +-33G, or sign
counts) with max_exp from operand exponent sums; verifies every certified value equals the rule's value."""
import argparse, importlib.util, json, os, struct, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
spec = importlib.util.spec_from_file_location("rc", os.path.join(HERE, "nr-qmma-rule-check.py")); RC = importlib.util.module_from_spec(spec); spec.loader.exec_module(RC)
def a_index(t, c, channels=512):
    lane = 4 * (t % 8) + ((c % 32) % 16) // 4; reg = ((t % 16) // 8) + 2 * ((c % 32) // 16)
    return (t // 16) * (16 * channels) + (c // 32) * 512 + lane * 16 + reg * 4 + c % 4
def w_index(k, n): return (n // 16) * 512 + (n % 8) * 64 + ((k % 16) // 4) * 16 + ((n % 16) // 8) * 8 + k % 4 + (k // 16) * 4
def cfd(n): return (n & 1) + ((n & 6) << 1) + ((n & 8) >> 2) + (n & 16) + (n & ~31)
def e4m3_half_bits(b):
    e = (b >> 3) & 15; f = b & 7; s = (b & 128) << 8
    if e == 0: return s | np.float16(f * 0.001953125).view(np.uint16).item()
    if e == 15 and f == 7: return s | 0x7e00
    return s | ((e + 8) << 10) | (f << 7)
def half_to_e4m3(hb):
    s = (hb >> 8) & 128; e = (hb >> 10) & 31; f = hb & 1023
    def rne(v, sh):
        if sh == 0: return v
        q, r, tie = v >> sh, v & ((1 << sh) - 1), 1 << (sh - 1); return q + (1 if (r > tie or (r == tie and q & 1)) else 0)
    if e == 31: return s | (127 if f else 126)
    if (hb & 0x7fff) >= 0x5f00: return s | 126
    if e < 9: m = f + 1024 if e else f; return s | rne(m, 16 - e if e else 15)
    m = rne(1024 + f, 7); ee = e - 8
    if m == 16: m = 8; ee += 1
    return s | (ee << 3) | (m - 8)
def hmul(a_bits, b_bits): return np.float16(np.float16(RC.half_from_bits(a_bits)) * np.float16(RC.half_from_bits(b_bits))).view(np.uint16).item()
def e4m3_int(b):
    (sig, q), neg = RC.e4m3_fields(b)
    if sig == 0: return 0, None
    m = sig << (q + 9); return (-m if neg else m), q + sig.bit_length() - 1
def main():
    p = argparse.ArgumentParser(); p.add_argument("--ref-middle", default=os.path.join(ROOT, "build/nr_chain/attempt-v5/middle-a5-a/a5-a")); p.add_argument("--fixtures", default=os.path.join(ROOT, "build/nr_split/fixtures"))
    p.add_argument("--launch", type=int, default=28); p.add_argument("--x-launch", type=int, default=27); p.add_argument("--res-launch", type=int, default=26); p.add_argument("--tokens", default="0,1,7,8,15,16,17,300,301,777,1000,1023,1024,1500,1999,2000,2100,2159"); a = p.parse_args()
    x = np.fromfile(os.path.join(a.ref_middle, f"launch-{a.x_launch:03d}", "out.bin"), dtype=np.uint8); res = np.fromfile(os.path.join(a.ref_middle, f"launch-{a.res_launch:03d}", "out.bin"), dtype=np.uint8)
    w = np.fromfile(os.path.join(a.fixtures, f"launch-{a.launch:03d}", "weights.bin"), dtype=np.uint8); expect = np.fromfile(os.path.join(a.ref_middle, f"launch-{a.launch:03d}", "out.bin"), dtype=np.uint8)
    scale = w[262144:262144 + 1024].view(np.uint16)
    stats = {"launch": a.launch, "tokens": [], "element_steps": 0, "replay_output_mismatch": 0, "all_terms_aligned": 0, "cert_33": 0, "cert_33_wrong": 0, "cert_sign": 0, "cert_sign_wrong": 0, "zero_products": 0}
    per_step_uncert = [0] * 16
    for t in [int(v) for v in a.tokens.split(",")]:
        stats["tokens"].append(t)
        for n in range(512):
            acc = hmul(e4m3_half_bits(int(res[a_index(t, cfd(n))])), int(scale[n]))
            for s in range(16):
                arow = [int(x[a_index(t, 32 * s + j)]) for j in range(32)]; bcol = [int(w[s * 16384 + (n // 64) * 2048 + w_index(j, n % 64)]) for j in range(32)]
                ref = RC.rule_reference(arow, bcol, acc); stats["element_steps"] += 1
                ma = [e4m3_int(b) for b in arow]; mb = [e4m3_int(b) for b in bcol]
                (cs, cq), cn = RC.half_fields(acc); cval = ((-cs if cn else cs) << (cq + 24)) if cs else 0
                terms = []; maxexp = None
                for (m1, e1), (m2, e2) in zip(ma, mb):
                    if m1 == 0 or m2 == 0: continue
                    maxexp = e1 + e2 if maxexp is None else max(maxexp, e1 + e2); terms.append(m1 * m2 << 6)
                if cs: ce = cq + cs.bit_length() - 1; maxexp = ce if maxexp is None else max(maxexp, ce); terms.append(cval)
                S = sum(terms)
                if maxexp is None: stats["zero_products"] += 1; stats["cert_33"] += 1; stats["cert_sign"] += 1; stats["all_terms_aligned"] += 1
                else:
                    gexp = maxexp - 25; G = (1 << (gexp + 24)) if gexp + 24 >= 0 else 0
                    if not G or all(t_ % G == 0 for t_ in terms): stats["all_terms_aligned"] += 1
                    for name, npos, nneg in (("cert_33", 33, 33), ("cert_sign", sum(1 for t_ in terms if t_ > 0), sum(1 for t_ in terms if t_ < 0))):
                        hl = RC.fixed_to_half_rne(S - npos * G, -24); hh = RC.fixed_to_half_rne(S + nneg * G, -24)
                        if hl == hh:
                            stats[name] += 1
                            if hl != ref: stats[name + "_wrong"] += 1
                        elif name == "cert_33": per_step_uncert[s] += 1
                acc = ref
            if half_to_e4m3(acc) != int(expect[a_index(t, cfd(n))]): stats["replay_output_mismatch"] += 1
    stats["uncertified_33_per_step"] = per_step_uncert; stats["cert_33_rate"] = round(stats["cert_33"] / stats["element_steps"], 5); stats["cert_sign_rate"] = round(stats["cert_sign"] / stats["element_steps"], 5); stats["aligned_rate"] = round(stats["all_terms_aligned"] / stats["element_steps"], 5)
    print(json.dumps(stats, indent=1)); json.dump(stats, open(os.path.join(ROOT, "build/nr_tp/census-launch%03d.json" % a.launch), "w"), indent=1)
if __name__ == "__main__": main()
