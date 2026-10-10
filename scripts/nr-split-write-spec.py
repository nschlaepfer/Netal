#!/usr/bin/env python3
"""Write spec/nr/split_swin.json from the emulation reports and the sealed acceptance receipt."""
import json, os, sys, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def main():
    emu = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "build/nr_split/emulate/launch-*/report.json"))):
        d = json.load(open(f)); emu[d["launch"]] = {"kernel": d["kernel"], "weights": d["weights"], "changed": d["oracle_changed_bytes"], "mismatch": d["mismatching_bytes"], "extra": d["bytes_written_outside_oracle_change"]}
    want = list(range(25, 58)) + list(range(100, 133)); missing = [n for n in want if n not in emu]
    if missing: raise SystemExit("missing emulation reports: " + str(missing))
    all_exact = all(emu[n]["mismatch"] == 0 and emu[n]["extra"] == 0 for n in want)
    rec = json.load(open(os.path.join(ROOT, "build/nr_split/attempt-v2/verification.json")))
    spec = {"schema": "neuralstudio.nr.split-swin-spec.v1", "status": "accepted" if all_exact and rec["result"] == "pass" else "open",
            "family": "cc_split_swin_16h + final head + dec_input_upsample (blocks 23-30, 39-47; launches 25-57, 100-132)",
            "runtime_sha256": "e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e",
            "numerics": {
                "qmma": "exact E4M3 products, fp32 FMA chain per 32-slot step onto half C, one half rounding per step",
                "ptxas_contraction": "single-use mul.f16x2 feeding add.f16x2 becomes an f16 fma: first operand if it is a product, otherwise the second (dec_input_upsample)",
                "rsqrt_rcp": "RTX-captured half tables (build/nr_rsq_half attempt-v2, build/nr_rcp_half rtx-v1)",
                "split_ffn": "H1 = X W1 (512->512); per 64-col group: H2 = e4m3(H1_g) W2_g (64->256), a = e4m3(cubic_silu(H2)), OUT_g = a W3_g (256->64); output column n at channel cfd(n)",
                "residual_conv": "Y = e4m3(chain_16(W . X, C0 = half(res[cfd(n)] * scale[n]))) at channel cfd(n)",
                "qkv": "q = e4m3(half(half(t*rsqrt(sumsq32)) * h(head_scale_f32))), k = e4m3(half(t*rsqrt(sumsq32))), v = e4m3(t); ViT sum-of-squares tree",
                "attention": "score = half(fp32 chain over 32 dims in slot order from the half bias); v = clamp(fma(s,0x29c0,0x3d34),0x3c20,0x3e47); e = as_half((bits(v)<<5)+0x8000); rowsum = tree8 per lane/parity, ((L0+L1)+L2)+L3, lo+hi; inv = rcp(max(sum,0x0410)); P = e4m3(half(e*inv)); out = e4m3(two 32-key steps of P.v) at 32h+cfd(d)",
                "windows": "8x8 tokens = four 4x4 tiles, m-tile order (ty,tx),(ty,tx+1),(ty+1,tx),(ty+1,tx+1); shift from constants +32/+36; out-of-grid tiles zero q/k/v, their queries unstored",
                "pool": "pooled = e4m3(half(half(half(a+b)+half(c+d))*0.25)) over pre-rounding conv halves; padded 32x20 grid written in full, padding zero",
                "final_head": "512->1024 plain 16-step chain, ViT 2D layout (16384-byte tiles), 30x18 valid tokens only",
                "dec_input_upsample": "nearest 2x of the ViT output; conv 1024->512 in four K quarters rounded to half, ((p0+p1)+p2)+p3; out = e4m3(fma(res[cfd(n)], scale[n], total))",
                "view_layout": "byte(r,c,ch) = (ch/32)*69120 + (r/2 + 18*((ch%32)/16))*1920 + (c/2 + 30*(r%2))*32 + ch%16 + 16*(c%2)"},
            "emulation": {"launches": want, "all_exact": all_exact, "reports": {str(n): emu[n] for n in want}},
            "metal_acceptance": {"receipt": "build/nr_split/attempt-v2/verification.json", "receipt_root_sha256": rec["receipt_root_sha256"], "result": rec["result"], "fresh_processes": rec["fresh_processes"],
                                 "connected_chain": rec["connected_chain"], "boundary_inputs": rec["boundary_inputs"], "poisons": rec["poisons"], "ordinals": rec["ordinals"],
                                 "gpu_ms_one_pass_a5_a": round(sum(r["gpu_ms"] for r in rec["runs"] if r["poison"] == "a5" and r["ordinal"] == "a"), 1)},
            "documentation": "docs/NR_SPLIT_SWIN.md"}
    os.makedirs(os.path.join(ROOT, "spec/nr"), exist_ok=True); json.dump(spec, open(os.path.join(ROOT, "spec/nr/split_swin.json"), "w"), indent=1)
    print("spec written:", spec["status"], "all_exact", all_exact)
if __name__ == "__main__": main()
