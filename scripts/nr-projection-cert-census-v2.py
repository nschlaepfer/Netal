#!/usr/bin/env python3
"""OPT-NR-PROJ-CERT-001 CPU census v2 (Fable; v1 sealed unchanged). v2 changes after Codex's 07:09 review: (a) overflow-safe modeled fp32-chain allowance
   ceil((sum|p|+|seed|)/2^14) with range assertions, cited to Jeannerod & Rump 2013 (|s_hat - s| <= (n-1) u sum|x_i| for recursive RN summation without
   overflow/underflow; here n = 1025 exact terms, u = 2^-24), labelled an ideal-centred scenario with an allowance, not a Metal ceiling; (b) nonfinite inputs
   (attention row, weight column, residual, scale) and nonfinite seeds make an output ineligible BEFORE classification; (c) sealed source/command/version,
   pre- and post-run input hashes, per-output archive hash; the archive retains T3_bits, eligibility masks, endpoint codes and the modeled radius;
   (d) prospective group-cost accounting items recorded (prefix statistics are not free).
v1 text:: whole-projection certification on ViT block 31 `nr_vit_projection` (T=640, K=N=1024).
1. EXACT REFERENCE: the accepted staged arithmetic (four independent quarter chains of eight QMMA steps under the RTX-verified truncating/aligned
   rule with operand-exponent max_exp, exact half seed H(r*s) for quarter 0, three half RNE merges, the actual half->E4M3 codec), vectorized with
   exact integers. It must reproduce every accepted projection byte before any census statistic is accepted (stop condition otherwise).
2. ORACLE-FREE BOUNDS: from live inputs, weights and scales only, an outward-rounded enclosure of the exact staged result around the ideal exact
   dot P (exact seed + exact sum of all products): per-step product/carry truncation with the carry exponent taken from a proven carry interval,
   per-step half RNE rounding, the quarter structure and the three merges. Dyadic integers on a 2^-30 lattice; every bound term rounded outward.
3. CLASSIFICATION: certified iff every half in [P-B, P+B] maps to one E4M3 code; fallback when the interval contains zero, reaches the E4M3
   saturation region (>= 448) or half overflow, or an input is nonfinite. Expected bytes never enter steps 2-3.
4. VALIDATION (oracle, afterwards): actual T3 inside its enclosure for every output; false certificates counted (must be 0).
5. CLUSTERING at 16x8 instruction tiles and 16x64 workgroups; bound widths, margins, cancellation; a separately labelled MODELED native
   fp32-chain predictor ceiling (not validated on Metal).
Usage: nr-projection-cert-census.py --attention F --residual F --weights F --expected F --out DIR [--tokens N]"""
import argparse, hashlib, json, os, sys, time, platform
import numpy as np
L = 30                                   # value lattice 2^-30 (products lie on 2^-18, half values on 2^-24)
def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()
# ---------------- codecs (mirror src/nr_vit/nr_vit_common.metal) ----------------
def e4m3_fields(b):
    b = b.astype(np.int64); e = (b >> 3) & 15; m = b & 7
    sig = np.where(e == 0, m, 8 + m); q = np.where(e == 0, -9, e - 10); neg = (b & 0x80) != 0; nan = (e == 15) & (m == 7)
    return sig, q, neg, nan
def half_fields(h):
    h = h.astype(np.int64); e = (h >> 10) & 31; f = h & 1023
    sig = np.where(e == 0, f, 1024 + f); q = np.where(e == 0, -24, e - 25); neg = (h & 0x8000) != 0; return sig, q, neg, e == 31
BITLEN = np.array([0] + [int(v).bit_length() for v in range(1, 4096)], dtype=np.int64)
def floor_log2(mag):
    """floor(log2(mag)) for positive int64 (exact: verified against the power-of-two brackets)"""
    mag = mag.astype(np.int64); k = np.floor(np.log2(np.maximum(mag, 1).astype(np.float64))).astype(np.int64)
    k = np.where((np.int64(1) << np.clip(k, 0, 62)) > mag, k - 1, k); k = np.where((np.int64(1) << np.clip(k + 1, 0, 62)) <= mag, k + 1, k); return k
def rshift_rne(v, sh):
    """v >= 0 int64, sh >= 1: round-to-nearest-even right shift"""
    sh = np.clip(sh, 1, 62); q = v >> sh; r = v & ((np.int64(1) << sh) - 1); half = np.int64(1) << (sh - 1)
    return q + ((r > half) | ((r == half) & ((q & 1) == 1))).astype(np.int64)
def fixed_to_half_rne(total, scale):
    """exact signed integer total * 2^scale -> half bits (mirrors nr_hw_fixed_to_half_rne / fixed_to_half_rne in nr-qmma-rule-check.py)"""
    total = total.astype(np.int64); scale = np.broadcast_to(np.asarray(scale, dtype=np.int64), total.shape)
    neg = total < 0; mag = np.abs(total); nz = mag > 0; lead = floor_log2(np.maximum(mag, 1)); exponent = lead + scale
    normal = exponent >= -14; shift = lead - 10
    sig_n = np.where(shift > 0, rshift_rne(mag, np.maximum(shift, 1)), mag << np.clip(-shift, 0, 62))
    carry = sig_n >= 2048; sig_n = np.where(carry, sig_n >> 1, sig_n); exp_n = exponent + carry.astype(np.int64)
    res_n = np.where(exp_n > 15, 0x7c00, ((np.clip(exp_n, -15, 16) + 15) << 10) | np.clip(sig_n - 1024, 0, 1023))
    shift_s = -(scale + 24)
    sig_s = np.where(shift_s > 0, rshift_rne(mag, np.maximum(shift_s, 1)), mag << np.clip(-shift_s, 0, 62))
    res_s = np.where(sig_s >= 1024, 0x0400, sig_s)
    res = np.where(normal, res_n, res_s); res = np.where(nz, res, 0); return np.where(neg & nz, res | 0x8000, res).astype(np.int64)
def rne_shift(value, shift):
    value = value.astype(np.int64); shift = np.broadcast_to(np.asarray(shift, dtype=np.int64), value.shape)
    return np.where(shift == 0, value, rshift_rne(value, np.maximum(shift, 1)))
def to_e4m3(hbits):
    """nr_to_e4m3 on half bits"""
    b = hbits.astype(np.int64); sign = (b >> 8) & 128; exponent = (b >> 10) & 31; fraction = b & 1023
    nan_or_inf = exponent == 31; r_inf = sign | np.where(fraction != 0, 127, 126)
    sat = (b & 0x7fff) >= 0x5f00; r_sat = sign | 126
    small = exponent < 9; mant_small = np.where(exponent != 0, fraction + 1024, fraction); sh_small = np.where(exponent != 0, 16 - exponent, 15)
    r_small = sign | rne_shift(mant_small, np.clip(sh_small, 0, 62))
    mant = rne_shift(1024 + fraction, 7); e = exponent - 8; wrap = mant == 16; mant = np.where(wrap, 8, mant); e = e + wrap.astype(np.int64)
    r_norm = sign | np.minimum(126, (e << 3) + mant - 8)
    return np.where(nan_or_inf, r_inf, np.where(sat, r_sat, np.where(small, r_small, r_norm))).astype(np.uint8)
# ---------------- layouts ----------------
def a_index(t, c, channels=1024):
    lane = 4 * (t % 8) + ((c % 32) % 16) // 4; reg = ((t % 16) // 8) + 2 * ((c % 32) // 16)
    return (t // 16) * (16 * channels) + (c // 32) * 512 + lane * 16 + reg * 4 + c % 4
def d_index(t, c, channels=1024):
    return (t // 16) * (16 * channels) + (c // 64) * 1024 + ((c % 64) // 32) * 512 + (4 * (t % 8) + (c % 8) // 2) * 16 + ((c % 32) // 16) * 8 + ((t % 16) // 8) * 4 + ((c % 16) // 8) * 2 + (c % 2)
def slot_channel(j): return 2 * ((j % 16) // 4) + (j % 2) + 8 * ((j % 4) // 2) + 16 * (j // 16)
def cfd(n): return (n & 1) + ((n & 6) << 1) + ((n & 8) >> 2) + (n & 16) + (n & ~31)
def weight_index(k, n): return (n // 16) * 512 + (n % 8) * 64 + ((k % 16) // 4) * 16 + ((n % 16) // 8) * 8 + k % 4 + (k // 16) * 4
# ---------------- exact QMMA step (vectorized over outputs) ----------------
def qmma_step(sa, qa, na, sb, qb, nb, cbits):
    """sa,qa,na: (T,32) A operand fields; sb,qb,nb: (32,N) B fields; cbits (T,N) half bits -> (T,N) half bits. Exact rule."""
    T, N = sa.shape[0], sb.shape[1]
    psig = sa[:, None, :] * sb[None, :, :].transpose(0, 2, 1)            # (T,N,32) products of significands (<= 15*15)
    pq = qa[:, None, :] + qb[None, :, :].transpose(0, 2, 1)
    pneg = na[:, None, :] ^ nb[None, :, :].transpose(0, 2, 1)
    nz = psig > 0
    pex = pq + BITLEN[np.clip(sa, 0, 4095)][:, None, :] - 1 + BITLEN[np.clip(sb, 0, 4095)][None, :, :].transpose(0, 2, 1) - 1
    maxexp = np.where(nz, pex, -10 ** 6).max(axis=2)                       # (T,N)
    cs, cq, cn, _ = half_fields(cbits); cnz = cs > 0; cex = cq + BITLEN[np.clip(cs, 0, 4095)] - 1
    maxexp = np.where(cnz, np.maximum(maxexp, cex), maxexp); any_term = (maxexp > -10 ** 6)
    scale = maxexp - 25
    sh = pq - scale[:, :, None]; mag = np.where(sh >= 0, psig.astype(np.int64) << np.clip(sh, 0, 62), psig.astype(np.int64) >> np.clip(-sh, 0, 62))
    mag = np.where(nz, mag, 0); total = np.where(pneg, -mag, mag).sum(axis=2)
    shc = cq - scale; cmag = np.where(shc >= 0, cs.astype(np.int64) << np.clip(shc, 0, 62), cs.astype(np.int64) >> np.clip(-shc, 0, 62)); cmag = np.where(cnz, cmag, 0)
    total = total + np.where(cn, -cmag, cmag)
    out = fixed_to_half_rne(total, scale); return np.where(any_term, out, 0).astype(np.int64)
def half_add(x, y):
    """exact half + half -> half RNE (IEEE; -0 only when both are -0)"""
    sx, qx, nx, _ = half_fields(x); sy, qy, ny, _ = half_fields(y); q = np.minimum(qx, qy)
    tot = np.where(nx, -sx, sx) << np.clip(qx - q, 0, 62); tot = tot + (np.where(ny, -sy, sy) << np.clip(qy - q, 0, 62))
    r = fixed_to_half_rne(tot, q); both_negzero = (sx == 0) & (sy == 0) & nx & ny; return np.where(both_negzero, 0x8000, r)
def half_value_L(hbits):
    """exact half value on the 2^-L lattice (int64); NaN/Inf flagged by caller"""
    s, q, n, _ = half_fields(hbits); v = s << (q + L); return np.where(n, -v, v)
def half_grid_ceil(x):
    """smallest half-grid point >= x (x int64 on 2^-L lattice, finite range); returns lattice ints"""
    ax = np.abs(x); e = floor_log2(np.maximum(ax, 1)) - L; sp = np.int64(1) << np.clip(np.maximum(e, -14) - 10 + L, 0, 62)
    up = -((-x) // sp) * sp; return up
def half_grid_floor(x):
    ax = np.abs(x); e = floor_log2(np.maximum(ax, 1)) - L; sp = np.int64(1) << np.clip(np.maximum(e, -14) - 10 + L, 0, 62)
    return (x // sp) * sp
def half_bits_from_grid(x):
    """half bits of an exact half-grid value x (int64 on 2^-L lattice)"""
    neg = x < 0; ax = np.abs(x); nz = ax > 0; e = floor_log2(np.maximum(ax, 1)) - L
    normal = e >= -14; sig = np.where(normal, ax >> np.clip(e - 10 + L, 0, 62), ax >> (L - 24))
    bits = np.where(normal, ((np.clip(e, -14, 15) + 15) << 10) | np.clip(sig - 1024, 0, 1023), sig); bits = np.where(nz, bits, 0)
    return np.where(neg & nz, bits | 0x8000, bits).astype(np.int64)
def main():
    p = argparse.ArgumentParser(); p.add_argument("--attention", required=True); p.add_argument("--residual", required=True); p.add_argument("--weights", required=True); p.add_argument("--expected", required=True)
    p.add_argument("--out", required=True); p.add_argument("--tokens", type=int, default=640); p.add_argument("--chunk", type=int, default=64); a = p.parse_args()
    if os.path.exists(a.out): raise SystemExit("preserve previous census: " + a.out)
    os.makedirs(a.out); t_start = time.time(); T, K, N = a.tokens, 1024, 1024
    att = np.fromfile(a.attention, dtype=np.uint8); res = np.fromfile(a.residual, dtype=np.uint8); wb = np.fromfile(a.weights, dtype=np.uint8); exp_bytes = np.fromfile(a.expected, dtype=np.uint8)
    assert att.size == res.size == exp_bytes.size == 655360 and wb.size == 1050624, "input sizes"
    tt = np.arange(T)[:, None]; nn = np.arange(N)[None, :]
    # A operand per k-step: slot j holds attended channel 32s + slot_channel(j) (D layout)
    jj = np.arange(32); A_sig = np.zeros((T, 32, 32), np.int64); A_q = np.zeros_like(A_sig); A_neg = np.zeros((T, 32, 32), bool); nan_in = 0; row_nonfinite = np.zeros(T, bool); col_nonfinite = np.zeros(N, bool)
    for s in range(32):
        ch = 32 * s + slot_channel(jj); idx = d_index(tt, ch[None, :]); sg, q, ng, nan = e4m3_fields(att[idx]); A_sig[:, s], A_q[:, s], A_neg[:, s] = sg, q, ng; nan_in += int(nan.sum()); row_nonfinite |= nan.any(axis=1)
    # B operand per k-step: weights blob [s*32768 + (n//64)*2048 + weight_index(j, n%64)]
    B_sig = np.zeros((32, 32, N), np.int64); B_q = np.zeros_like(B_sig); B_neg = np.zeros((32, 32, N), bool)
    for s in range(32):
        idx = s * 32768 + (nn // 64) * 2048 + weight_index(jj[:, None], nn % 64); sg, q, ng, nan = e4m3_fields(wb[idx]); B_sig[s], B_q[s], B_neg[s] = sg, q, ng; nan_in += int(nan.sum()); col_nonfinite |= nan.any(axis=0)
    # residual r[t,n] (A layout at cfd(n)) and half scale[n]; exact seed H(r*s)
    r_sig, r_q, r_neg, r_nan = e4m3_fields(res[a_index(tt, cfd(nn))]); nan_in += int(r_nan.sum())
    pre_hashes = {k: sha_file(v) for k, v in (("attention", a.attention), ("residual", a.residual), ("weights", a.weights), ("expected_comparison_only", a.expected))}
    scale_bits = np.frombuffer(wb[1048576:1048576 + 2 * N].tobytes(), dtype=np.uint16).astype(np.int64)
    s_sig, s_q, s_neg, s_nan = half_fields(scale_bits); nan_in += int(s_nan.sum())
    prod = r_sig * s_sig[None, :]; seed_bits = fixed_to_half_rne(np.where(r_neg ^ s_neg[None, :], -prod, prod), r_q + s_q[None, :])
    seed_bits = np.where((prod == 0) & (r_neg ^ s_neg[None, :]), 0x8000, seed_bits)   # IEEE: (-0)*(+x) = -0 (half(product) in Metal keeps the sign of an exact zero)
    seed_nonfinite = ((seed_bits >> 10) & 31) == 31
    nonfinite_out = row_nonfinite[:, None] | col_nonfinite[None, :] | r_nan | s_nan[None, :] | seed_nonfinite   # (T,N): unsupported domain -> never eligible
    if nan_in: print("nonfinite inputs present:", nan_in, "-> outputs excluded from eligibility:", int(nonfinite_out.sum()))
    # ---------------- exact reference + ideal dot + oracle-free bounds, per token chunk ----------------
    out_code = np.zeros((T, N), np.uint8); P_L = np.zeros((T, N), np.int64); B_L = np.zeros((T, N), np.int64); Bfp32_L = np.zeros((T, N), np.int64)
    absum_L = np.zeros((T, N), np.int64); T3_bits = np.zeros((T, N), np.int64); overflow = np.zeros((T, N), bool)
    def round_bound(U):   # half RNE error bound (lattice ints) at magnitude bound U (lattice ints): half ulp/2 at floor(log2 U), or 2^-25 in the subnormal range
        e = floor_log2(np.maximum(U, 1)) - L; return np.where(U >= (np.int64(1) << (L - 14)), np.int64(1) << np.clip(e - 11 + L, 0, 62), np.int64(1) << (L - 25))
    for c0 in range(0, T, a.chunk):
        c1 = min(T, c0 + a.chunk); sl = slice(c0, c1); Tc = c1 - c0
        quarter_bits = []; quarter_ideal = []; quarter_err = []
        seed_L = half_value_L(seed_bits[sl])
        for z in range(4):
            c = seed_bits[sl] if z == 0 else np.zeros((Tc, N), np.int64)
            S = seed_L.copy() if z == 0 else np.zeros((Tc, N), np.int64); E = np.zeros((Tc, N), np.int64)
            for s in range(8 * z, 8 * z + 8):
                sa, qa, na = A_sig[sl, s], A_q[sl, s], A_neg[sl, s]; sb, qb, nb = B_sig[s], B_q[s], B_neg[s]
                # exact staged step
                c = qmma_step(sa, qa, na, sb, qb, nb, c)
                # ideal products on the lattice and their exact sum
                psig = sa[:, None, :] * sb.T[None, :, :]; pq = qa[:, None, :] + qb.T[None, :, :]; pneg = na[:, None, :] ^ nb.T[None, :, :]; nz = psig > 0
                pL = psig.astype(np.int64) << np.clip(pq + L, 0, 62); step_sum = np.where(pneg, -pL, pL).sum(axis=2); absum_L[sl] += pL.sum(axis=2)
                # bound: reachable carry exponent from the carry interval [S-E, S+E]; max_exp* = max(product exponents, carry exponent); Delta* = 2^(M*-25)
                pex = pq + BITLEN[np.clip(sa, 0, 4095)][:, None, :] - 1 + BITLEN[np.clip(sb, 0, 4095)].T[None, :, :] - 1
                m = nz.sum(axis=2); pmax = np.where(nz, pex, -10 ** 6).max(axis=2)
                cU = np.abs(S) + E; ce = floor_log2(np.maximum(cU, 1)) - L; Mstar = np.where(cU > 0, np.maximum(pmax, ce), pmax)
                has = Mstar > -10 ** 6; dexp = Mstar - 25   # Delta* exponent
                trunc_p = np.where(has & (dexp > -18), m * (np.int64(1) << np.clip(dexp + L, 0, 62)), 0)          # products exact on 2^-18 when Delta <= 2^-18
                trunc_c = np.where(has & (cU > 0) & (dexp > -24), np.int64(1) << np.clip(dexp + L, 0, 62), 0)     # carry exact when Delta <= its lattice (>= 2^-24)
                S = S + step_sum; Epre = E + trunc_p + trunc_c; U = np.abs(S) + Epre
                overflow[sl] |= U >= (np.int64(65504) << L)
                E = Epre + round_bound(U)
            quarter_bits.append(c); quarter_ideal.append(S); quarter_err.append(E)
        # merges T1 = H(T0 + C1), T2 = H(T1 + C2), T3 = H(T2 + C3)
        Tb = quarter_bits[0]; Ti = quarter_ideal[0]; Te = quarter_err[0]
        for z in (1, 2, 3):
            Tb = half_add(Tb, quarter_bits[z]); Ti = Ti + quarter_ideal[z]; Epre = Te + quarter_err[z]; U = np.abs(Ti) + Epre
            overflow[sl] |= U >= (np.int64(65504) << L); Te = Epre + round_bound(U)
        T3_bits[sl] = Tb; out_code[sl] = to_e4m3(Tb); P_L[sl] = Ti; B_L[sl] = Te
        # MODELED native fp32-chain predictor (NOT validated on Metal): 1024 sequential fp32 roundings, each <= 2^-24 * |partial| <= 2^-24 * sum|p|
        L1 = absum_L[sl] + np.abs(seed_L); assert int(L1.max()) < (1 << 62) and int(Te.max()) < (1 << 62), "lattice range"
        Bfp32_L[sl] = Te + ((L1 + ((1 << 14) - 1)) >> 14)   # (n-1)*u*sum|x| with n-1 = 1024, u = 2^-24 -> L1 / 2^14, outward (ceil); no multiplication before the shift
        print(f"chunk {c0}-{c1} done, {time.time() - t_start:.0f} s", flush=True)
    # ---------------- 1. exact reference vs accepted bytes ----------------
    ref_bytes = np.zeros(655360, np.uint8); ref_bytes[a_index(tt, cfd(nn)).ravel()] = out_code.ravel()
    mism = int((ref_bytes != exp_bytes).sum()); print("exact reference vs accepted projection bytes: mismatching", mism, "of", exp_bytes.size)
    rec = {"schema": "neuralstudio.nr.projection-cert-census.v2", "target": "ViT block 31 nr_vit_projection, T=%d K=N=1024" % T,
           "inputs": {k: {"file": os.path.abspath(v), "sha256": sha_file(v)} for k, v in (("attention", a.attention), ("residual", a.residual), ("weights", a.weights), ("expected_comparison_only", a.expected))},
           "reference": {"mismatching_bytes": mism, "bytes": int(exp_bytes.size), "nonfinite_inputs": nan_in}}
    if mism:
        rec["result"] = "STOP: exact reference does not reproduce the accepted bytes"; json.dump(rec, open(os.path.join(a.out, "census.json"), "w"), indent=1); print(rec["result"]); sys.exit(1)
    # ---------------- 3. classification (oracle-free) ----------------
    lo = P_L - B_L; hi = P_L + B_L; contains_zero = (lo <= 0) & (hi >= 0)
    sat = np.maximum(np.abs(lo), np.abs(hi)) >= (np.int64(448) << L)
    eligible = ~contains_zero & ~sat & ~overflow & ~nonfinite_out
    h_min = half_grid_ceil(lo); h_max = half_grid_floor(hi); empty = h_min > h_max
    code_lo = to_e4m3(half_bits_from_grid(np.where(eligible & ~empty, h_min, 0))); code_hi = to_e4m3(half_bits_from_grid(np.where(eligible & ~empty, h_max, 0)))
    certified = eligible & ~empty & (code_lo == code_hi)
    # modeled native predictor (fp32 chain) ceiling
    lo2 = P_L - Bfp32_L; hi2 = P_L + Bfp32_L; elig2 = ~((lo2 <= 0) & (hi2 >= 0)) & (np.maximum(np.abs(lo2), np.abs(hi2)) < (np.int64(448) << L)) & ~overflow & ~nonfinite_out
    hmin2 = half_grid_ceil(lo2); hmax2 = half_grid_floor(hi2); cert2 = elig2 & (hmin2 <= hmax2) & (to_e4m3(half_bits_from_grid(np.where(elig2, hmin2, 0))) == to_e4m3(half_bits_from_grid(np.where(elig2, hmax2, 0))))
    # ---------------- 4. validation with the oracle (afterwards) ----------------
    T3_L = half_value_L(T3_bits); inside = (T3_L >= lo) & (T3_L <= hi); enclosure_violations = int((~inside).sum())
    false_cert = certified & (code_lo != out_code); n_false = int(false_cert.sum())
    # oracle-assisted ceiling: exact predictor with the TRUE error (|T3 - P|) instead of the bound: how many outputs are within one code of P anyway
    # ---------------- 5. clustering / statistics ----------------
    tiles = certified.reshape(T // 16, 16, N // 8, 8).all(axis=(1, 3)); groups = certified.reshape(T // 16, 16, N // 64, 64).all(axis=(1, 3))
    tiles2 = cert2.reshape(T // 16, 16, N // 8, 8).all(axis=(1, 3)); groups2 = cert2.reshape(T // 16, 16, N // 64, 64).all(axis=(1, 3))
    Bf = B_L.astype(np.float64) / 2 ** L; Pf = P_L.astype(np.float64) / 2 ** L; absf = absum_L.astype(np.float64) / 2 ** L
    cancel = np.where(absf > 0, np.abs(Pf) / np.maximum(absf, 1e-30), 1.0)
    # E4M3 code width at |P| (ulp) for margin reporting
    e_out = np.floor(np.log2(np.maximum(np.abs(Pf), 2 ** -9))); e4m3_ulp = np.where(np.abs(Pf) >= 2 ** -6, 2.0 ** (e_out - 3), 2.0 ** -9)
    fallback_reason = np.full((T, N), "certified", dtype=object)
    fallback_reason[~certified & contains_zero] = "interval contains zero (sign not proved)"; fallback_reason[~certified & ~contains_zero & sat] = "saturation region"
    fallback_reason[~certified & ~contains_zero & ~sat & overflow] = "half overflow reachable"; fallback_reason[~certified & eligible & ~empty & (code_lo != code_hi)] = "interval straddles an E4M3 decision boundary"
    fallback_reason[~certified & eligible & empty] = "no half in interval (bound below half grid; conservative fallback)"
    fallback_reason[~certified & nonfinite_out] = "nonfinite input/seed (unsupported domain)"
    reasons = {k: int(v) for k, v in zip(*np.unique(fallback_reason.astype(str), return_counts=True))}
    def q(x, ps=(0, 10, 50, 90, 99, 100)): return {str(pp): float(np.percentile(x, pp)) for pp in ps}
    rec.update({"outputs": int(T * N), "certified": int(certified.sum()), "certified_fraction": float(certified.mean()), "fallback_reasons": reasons,
                "false_certificates": n_false, "enclosure_violations": enclosure_violations,
                "clustering": {"tiles_16x8_total": int(tiles.size), "tiles_16x8_fully_certified": int(tiles.sum()), "workgroups_16x64_total": int(groups.size), "workgroups_16x64_fully_certified": int(groups.sum()),
                               "workgroups_needing_any_fallback": int((~groups).sum()), "fallback_outputs_per_uncertified_workgroup": q((~certified).reshape(T // 16, 16, N // 64, 64).sum(axis=(1, 3))[~groups]) if (~groups).any() else None},
                "bound_width_B_percentiles": q(Bf), "bound_width_in_e4m3_ulps_percentiles": q(Bf / e4m3_ulp), "abs_P_percentiles": q(np.abs(Pf)), "cancellation_ratio_percentiles": q(cancel),
                "modeled_fp32_chain_IDEAL_CENTRED_SCENARIO_not_a_metal_ceiling": {"certified": int(cert2.sum()), "certified_fraction": float(cert2.mean()), "tiles_16x8_fully_certified": int(tiles2.sum()), "workgroups_16x64_fully_certified": int(groups2.sum()),
                 "allowance": "(n-1)*u*(sum|products|+|seed|) with n = 1025 exact terms, u = 2^-24, computed as ceil(L1/2^14) on the lattice (overflow-safe)", "citation": "Jeannerod & Rump, SIAM J. Matrix Anal. Appl. 34(2), 2013, doi:10.1137/120894488: |s_hat - s| <= (n-1) u sum|x_i| for recursive round-to-nearest summation",
                 "assumptions": "exact FP8 products and exact half seed as the n terms; binary32 round-to-nearest additions in an unknown order; no overflow/underflow; the interval is centred on the exact P, NOT on an actually computed fp32 predictor - a deployable certificate must centre on its own predictor and include the errors of any approximate prefix statistics used for M* and rounding bounds",
                                                                     "note": "Metal MMA precision/order/underflow unvalidated; this scenario proves nothing about Metal"},
                "method": {"lattice": "2^-%d dyadic integers, outward rounding" % L, "predictor": "exact seed H(r*s) + exact sum of all 1024 products (ideal dot); no expected bytes, no oracle carries",
                           "per_step_bound": "product truncation m*Delta* (0 if Delta* <= 2^-18), carry truncation Delta* (0 if Delta* <= 2^-24), Delta* = 2^(M*-25) with M* = max(operand-exponent sums, exponent of |S|+E over the carry interval), then half RNE bound (ulp/2 at |S|+E, 2^-25 subnormal)",
                           "merges": "three half additions: errors added, each with its own RNE bound", "classification": "all halves in [P-B,P+B] map to one E4M3 code via the actual codec; fallback on zero-containing intervals, saturation (>= 448), half overflow, nonfinite inputs",
                           "validation": "oracle used only afterwards: T3 inside its enclosure for every output; false certificates counted"},
                "seconds": round(time.time() - t_start, 1), "result": "pass" if (mism == 0 and n_false == 0 and enclosure_violations == 0) else "fail"})
    rec["fallback_reasons"]["nonfinite input/seed (unsupported domain)"] = int((~certified & nonfinite_out).sum())
    arch = os.path.join(a.out, "per-output.npz")
    np.savez_compressed(arch, certified=certified, P_L=P_L, B_L=B_L, Bfp32_L=Bfp32_L, code=out_code, T3_bits=T3_bits, code_lo=code_lo, code_hi=code_hi, contains_zero=contains_zero, saturation=sat, overflow=overflow,
                        nonfinite=nonfinite_out, eligible=eligible, empty_interval=empty, cert_fp32_scenario=cert2, reason=fallback_reason.astype(str))
    post_hashes = {k: sha_file(v) for k, v in (("attention", a.attention), ("residual", a.residual), ("weights", a.weights), ("expected_comparison_only", a.expected))}
    rec["seal"] = {"script": os.path.abspath(__file__), "script_sha256": sha_file(__file__), "command": sys.argv, "python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform(),
                   "inputs_sha256_pre_run": pre_hashes, "inputs_sha256_post_run": post_hashes, "inputs_unchanged": pre_hashes == post_hashes, "per_output_archive": arch, "per_output_archive_sha256": sha_file(arch),
                   "archive_fields": "certified, P_L, B_L, Bfp32_L (2^-30 lattice ints), code (actual E4M3), T3_bits (actual half), code_lo/code_hi (endpoint E4M3 codes), contains_zero, saturation, overflow, nonfinite, eligible, empty_interval, cert_fp32_scenario, reason"}
    rec["prospective_group_cost_items_unmeasured"] = ["input/weight layout preparation for the native GEMM", "native GEMM producing the predictor (its own error must be centred and bounded)",
        "per-step prefix statistics for M*: 32 per-step max operand exponents and 32 per-step partial-sum magnitudes per output (the exact ideal prefix sums S are input-derived but cost a per-step reduction; a cheaper proved alternative would be needed)",
        "bound construction (per-step truncation/rounding terms, three merges)", "classification against the E4M3 codec", "compaction of ~20% (today) or >=1.6% (ceiling) uncertified outputs, ~16-281 per 16x64 workgroup", "exact three-product fallback for those outputs", "output merge, synchronization, memory traffic"]
    rec["receipt_root_sha256"] = hashlib.sha256(json.dumps(rec, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
    json.dump(rec, open(os.path.join(a.out, "census.json"), "w"), indent=1, default=str)
    print(json.dumps({k: rec[k] for k in ("certified", "certified_fraction", "fallback_reasons", "false_certificates", "enclosure_violations", "clustering", "modeled_fp32_chain_IDEAL_CENTRED_SCENARIO_not_a_metal_ceiling", "result")}, indent=1, default=str)); print("seal: inputs unchanged", rec["seal"]["inputs_unchanged"], "archive", rec["seal"]["per_output_archive_sha256"][:16], "root", rec["receipt_root_sha256"][:16])
if __name__ == "__main__": main()
