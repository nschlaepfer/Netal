#!/usr/bin/env python3
"""Connected run of NR launches 25-132 on Metal: split-Swin blocks 23-30 -> final head -> ViT repack 2D->1D ->
ViT-1D blocks 31-38 -> repack 1D->2D -> decoder input upsample -> split-Swin blocks 40-47 -> output view.
Every launch is its own probe process fed from the previous launches' live Metal outputs; recorded bytes enter
only at launch 25 (block-23 input view) and launch 26 (its residual view). Two poisons x two ordinals."""
import argparse, hashlib, json, os, subprocess, sys
KTYPE = {"cc_split_swin_16h_ffwd_inpview_512_tilesync_fp8": "ffwd_view", "cc_split_swin_16h_ffwd_proj_inpview_512_chained_fp8": "conv_resview",
         "cc_split_swin_16h_qkv_512_chained_fp8": "qkv_attend", "cc_split_swin_16h_proj_512_chained_fp8": "conv", "cc_split_swin_16h_ffwd_512_chained_fp8": "ffwd",
         "cc_split_swin_16h_ffwd_proj_512_chained_fp8": "conv", "cc_split_swin_16h_proj_pool_512_chained_fp8": "conv_pool", "cc_split_swin_16h_final_head_512_wait_fp8": "final_head",
         "cc_dec_input_upsample_1024_512_tilesync_fp8": "dec_upsample", "cc_split_swin_16h_proj_512_outview_wait_fp8": "conv_outview"}
def a_index(t, c, channels=1024):      # A-identity token layout (nr_a_index)
    lane = 4 * (t % 8) + ((c % 32) % 16) // 4; reg = ((t % 16) // 8) + 2 * ((c % 32) // 16)
    return (t // 16) * (16 * channels) + (c // 32) * 512 + lane * 16 + reg * 4 + c % 4
def run(cmd, label):
    r = subprocess.run(cmd, capture_output=True, text=True); ok = r.returncode == 0
    tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr.strip()[-200:]
    print(f"{label} {'pass' if ok else 'FAIL'}  {tail}", flush=True)
    if r.returncode == 2: print(r.stderr[-600:])
    return ok
def main():
    p = argparse.ArgumentParser(); p.add_argument("--split-attempt", required=True); p.add_argument("--vit-attempt", required=True); p.add_argument("--out", required=True)
    p.add_argument("--split-fixtures", default="build/nr_split/fixtures"); p.add_argument("--vit-fixtures", default="build/nr_vit/fixtures"); p.add_argument("--encoder-entry", default="", help="FRESH producer file for the launch 25/26 entrance (block22-middle.e4m3, 1105920 bytes, pool 86649856); required unless --allow-recorded-entry"); p.add_argument("--strict", action="store_true", help="every tensor input of every launch must come from a same-run live producer (split_probe --strict-inputs, repack single-direction with live input); fails closed otherwise"); p.add_argument("--allow-recorded-entry", action="store_true", help="diagnostic only: let launches 25/26 read the recorded fixture entrance"); p.add_argument("--poisons", default="a5,5a"); p.add_argument("--ordinals", default="a,b"); a = p.parse_args()
    index = json.load(open(os.path.join(a.split_fixtures, "index.json"))); failures = 0
    ENTRY = 86649856     # pool offset of the middle entrance (launch 25 input, launch 26 residual input)
    if not a.encoder_entry and not a.allow_recorded_entry: raise SystemExit("fail-closed: --encoder-entry (fresh block22-middle.e4m3) is required; --allow-recorded-entry only for diagnostics")
    entry_prov = None
    if a.encoder_entry:
        eb = open(a.encoder_entry, "rb").read()
        if len(eb) != 1105920: raise SystemExit(f"fail-closed: encoder entry has {len(eb)} bytes, expected 1105920")
        entry_prov = {"file": os.path.abspath(a.encoder_entry), "bytes": len(eb), "sha256": hashlib.sha256(eb).hexdigest(), "pool_offset": ENTRY, "live": True}
    os.makedirs(a.out, exist_ok=True)
    json.dump({"schema": "neuralstudio.nr.middle-entry-provenance.v1", "encoder_entry": entry_prov or {"live": False, "source": "recorded fixture (diagnostic run)"}, "middle_output_pool_offset": 123145216, "middle_output_role": "out_view.bin of launch-132"}, open(os.path.join(a.out, "provenance.json"), "w"), indent=1)
    sprobe = os.path.join(a.split_attempt, "split_probe"); slib = os.path.join(a.split_attempt, "split.metallib")
    vprobe = os.path.join(a.vit_attempt, "vit_probe"); rprobe = os.path.join(a.vit_attempt, "repack_probe"); vlib = os.path.join(a.vit_attempt, "vit.metallib")
    for poison in a.poisons.split(","):
        for ordinal in a.ordinals.split(","):
            base = os.path.join(a.out, f"{poison}-{ordinal}"); os.makedirs(base, exist_ok=True); live = {}
            if a.encoder_entry: live[ENTRY] = os.path.abspath(a.encoder_entry)        # fresh producer replaces the fixture entrance for launches 25 and 26
            def split_launch(n):
                meta = index["launches"][str(n)]; ktype = KTYPE[meta["kernel"]]; fx = os.path.join(a.split_fixtures, f"launch-{n:03d}"); out = os.path.join(base, f"launch-{n:03d}")
                cmd = [sprobe, slib, ktype, poison, out, "--fixture", fx, "--tables", a.split_fixtures, "--shift", f"{meta['shift'][0]},{meta['shift'][1]}"] + (["--strict-inputs"] if a.strict else [])
                for b in meta["bindings"].values():
                    if b["direction"] == "in" and b["pool_offset"] in live: cmd += ["--in", f"{b['role']}={live[b['pool_offset']]}"]
                    elif b["direction"] == "in" and a.strict: raise SystemExit(f"strict: launch {n} input role {b['role']} (pool {b['pool_offset']}) has no same-run live producer")
                ok = run(cmd, f"{poison}-{ordinal} launch {n:3d} {ktype:13s}")
                for b in meta["bindings"].values():
                    if b["direction"] == "out": live[b["pool_offset"]] = os.path.join(out, b["role"] + ".bin")
                return ok
            for n in range(25, 58): failures += not split_launch(n)
            # launch 57 output is the ViT 2D input; the recorded repack reads a 720896-byte region whose 64 KB tail does
            # not influence its output (checked: zeroed tail reproduces the recording), so the live 655360 bytes are zero-padded
            # The final head writes only the 30 x 18 valid pooled tokens; the recorded buffer was zero-initialised memory,
            # so the 100 padding tokens of the 32 x 20 grid (which the ViT does process) are zero: clear the probe's poison there.
            vit_in = os.path.join(base, "vit_input_padded.bin"); raw = bytearray(open(live[179096576], "rb").read())
            for token in range(640):
                tile, slot = token // 16, token % 16; R, C = 4 * (tile // 8) + slot // 4, 4 * (tile % 8) + slot % 4
                if R >= 18 or C >= 30:
                    for c in range(1024): raw[a_index(token, c)] = 0
            open(vit_in, "wb").write(bytes(raw) + bytes(720896 - len(raw)))
            rp58 = os.path.join(base, "repack58"); failures += not run([rprobe, vlib, a.vit_fixtures, poison, rp58, "--in58", vit_in, "--only", "58"] + (["--require-live"] if a.strict else []), f"{poison}-{ordinal} launch  58 repack2d->1d ")
            prev = os.path.join(rp58, "nr_vit_repack_2d_to_1d.bin")
            for block in range(31, 39):
                out = os.path.join(base, f"block{block}")
                failures += not run([vprobe, vlib, a.vit_fixtures, str(block), poison, out, "--input-tokens", prev], f"{poison}-{ordinal} vit block {block}    ")
                prev = os.path.join(out, "projection.bin")
            rp99 = os.path.join(base, "repack99"); failures += not run([rprobe, vlib, a.vit_fixtures, poison, rp99, "--in99", prev, "--only", "99"] + (["--require-live"] if a.strict else []), f"{poison}-{ordinal} launch  99 repack1d->2d ")
            live[211864576] = os.path.join(rp99, "nr_vit_repack_1d_to_2d.bin")   # launch 99 output: the ViT 2D output consumed by launch 100
            for n in range(100, 133): failures += not split_launch(n)
    print("failures", failures); sys.exit(1 if failures else 0)
if __name__ == "__main__": main()
