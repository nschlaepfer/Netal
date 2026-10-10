#!/usr/bin/env python3
"""Run the post block (launch 155) emulation on a fixed random CTA sample under several semantic hypotheses and tabulate the
output-image agreement with the recorded RTX output."""
import json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
VARIANTS = [("base", {}), ("c_last", {"EMU_MMA_ORDER": "c_last"}), ("sust_rne", {"EMU_SUST_RNE": "1"}), ("hmma_fp32", {"EMU_HMMA_MODE": "fp32"}), ("hmma_exact", {"EMU_HMMA_MODE": "exact"}), ("div_exact", {"EMU_DIV_MODE": "exact"}), ("ex2_frac", {"EMU_EX2_POS": "frac"}), ("fuse_none", {"EMU_FUSE": "none"}), ("fuse_first", {"EMU_FUSE": "first"}), ("fuse_second", {"EMU_FUSE": "second"})]
def main():
    sample = sys.argv[1] if len(sys.argv) > 1 else "200"; out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "build/nr_dec/post-emu/variants")
    only = [v for v in sys.argv[3].split(",") if v] if len(sys.argv) > 3 else None
    os.makedirs(out, exist_ok=True)
    for name, env in VARIANTS:
        if only and name not in only: continue
        d = os.path.join(out, name)
        if not os.path.exists(os.path.join(d, "report.json")):
            subprocess.run(["rm", "-rf", d]); r = subprocess.run([sys.executable, os.path.join(HERE, "nr-vit-emulate-launch.py"), "--launch", "155", "--procs", "6", "--sample", sample, "--output", d], env=dict(os.environ, **env), capture_output=True, text=True)
            if r.returncode: print(name, "FAILED", r.stderr[-800:]); continue
        t = json.load(open(os.path.join(d, "report.json")))["texture"]
        print(f"{name:11s} written {t['values_written']} mismatching {t['mismatching_values']} ulp {t['ulp_histogram']}", flush=True)
if __name__ == "__main__": main()
