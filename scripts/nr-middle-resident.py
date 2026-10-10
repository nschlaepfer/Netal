#!/usr/bin/env python3
"""Stage and run the GPU-resident middle chain (middle_chain_probe) on a fresh encoder entry with the same fixtures-index roles, shifts and
weights as the strict harness; check every launch/stage output hash against a strict reference middle run and report GPU vs wall time."""
import argparse, hashlib, json, os, subprocess, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import importlib.util
spec = importlib.util.spec_from_file_location("tnm", os.path.join(ROOT, "scripts/test-nr-middle.py")); TNM = importlib.util.module_from_spec(spec); spec.loader.exec_module(TNM)
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
def main():
    p = argparse.ArgumentParser(); p.add_argument("--attempt", required=True); p.add_argument("--reference-middle", required=True, help="strict middle run dir (.../middle-<key>/<key>)"); p.add_argument("--encoder-entry", required=True); p.add_argument("--poison", required=True)
    p.add_argument("--chain-only", action="store_true", help="write chain.json and exit (used by the fused body runner)"); p.add_argument("--vit-tp-lib", default="", help="optional three-product ViT library (expand/contract/projection kernels)"); p.add_argument("--split-tp-lib", default="", help="optional three-product split conv library"); p.add_argument("--split-attempt", required=True); p.add_argument("--vit-attempt", required=True); p.add_argument("--split-fixtures", default=os.path.join(ROOT, "build/nr_split/fixtures")); p.add_argument("--vit-fixtures", default=os.path.join(ROOT, "build/nr_vit/fixtures"))
    a = p.parse_args(); att = os.path.abspath(a.attempt)
    if os.path.exists(att): raise SystemExit("preserve previous attempt: " + att)
    os.makedirs(att); idx = json.load(open(os.path.join(a.split_fixtures, "index.json")))
    def launch(n):
        m = idx["launches"][str(n)]; return {"launch": n, "kernel": TNM.KTYPE[m["kernel"]], "shift": m["shift"], "weights": os.path.abspath(os.path.join(a.split_fixtures, f"launch-{n:03d}", "weights.bin")), "roles": {b["role"]: {"pool_offset": b["pool_offset"], "bytes": b["bytes"], "direction": b["direction"]} for b in m["bindings"].values()}}
    pad = [t for t in range(640) if (4 * ((t // 16) // 8) + (t % 16) // 4) >= 18 or (4 * ((t // 16) % 8) + (t % 16) % 4) >= 30]
    chain = {"poison": a.poison, "split_lib": os.path.abspath(os.path.join(a.split_attempt, "split.metallib")), "vit_lib": os.path.abspath(os.path.join(a.vit_attempt, "vit.metallib")), "tables": os.path.abspath(a.split_fixtures), "pool_bytes": 212519936, "vit_tp_lib": os.path.abspath(a.vit_tp_lib) if a.vit_tp_lib else "", "split_tp_lib": os.path.abspath(a.split_tp_lib) if a.split_tp_lib else "",
             "entry_file": os.path.abspath(a.encoder_entry), "entry_offset": 86649856, "split_a": [launch(n) for n in range(25, 58)], "split_b": [launch(n) for n in range(100, 133)], "final_offset": 123145216,
             "vit": {"final_head_offset": 179096576, "padding_tokens": pad, "repack99_out_offset": 211864576, "blocks": [{"block": b, **{f"w{i}": os.path.abspath(os.path.join(a.vit_fixtures, f"block{b}", f"weights.layer{i}.bin")) for i in (0, 1, 2, 4)}} for b in range(31, 39)]}}
    cj = os.path.join(att, "chain.json"); json.dump(chain, open(cj, "w"), indent=1)
    if a.chain_only: print("chain", cj); return
    t0 = time.time(); r = subprocess.run([os.path.join(a.split_attempt, "middle_chain_probe"), cj, os.path.join(att, "run")], capture_output=True, text=True); wall = time.time() - t0
    open(os.path.join(att, "probe.log"), "w").write(r.stdout + r.stderr)
    if r.returncode: raise SystemExit("probe failed: " + r.stderr[-800:])
    run = json.load(open(os.path.join(att, "run", "chain-run.json"))); ref = a.reference_middle; failures = []; checks = 0
    for L in run["launches"]:
        n, kind = L["launch"], L["kind"]
        if kind == "vit_block":
            for stage, h in L["outputs"].items():
                f = os.path.join(ref, f"block{n}", f"{stage}.bin")
                if not os.path.exists(f): failures.append(f"block {n}: missing reference {stage}"); continue
                checks += 1
                if sha(f) != h: failures.append(f"block {n} {stage}: hash differs from the strict reference")
            for stage, (once, multi, zero) in L["coverage"].items():
                if multi or zero: failures.append(f"block {n} {stage}: coverage once={once} multi={multi} zero={zero}")
        elif kind.startswith("nr_vit_repack"):
            f = os.path.join(ref, "repack58" if n == 58 else "repack99", kind + ".bin"); checks += 1
            if sha(f) != L["outputs"][kind]: failures.append(f"repack {n}: hash differs")
            once, multi, zero = L["coverage_dwords"]
            if multi or zero: failures.append(f"repack {n}: coverage {L['coverage_dwords']}")
        else:
            for role, h in L["outputs"].items():
                f = os.path.join(ref, f"launch-{n:03d}", f"{role}.bin"); checks += 1
                if not os.path.exists(f): failures.append(f"launch {n}: missing reference {role}"); continue
                if sha(f) != h: failures.append(f"launch {n} {role}: hash differs from the strict reference")
            for role, (once, multi, zero) in L["coverage"].items():
                if multi or (zero and role != "out_vit"): failures.append(f"launch {n} {role}: coverage once={once} multi={multi} zero={zero}")
    final_ref = sha(os.path.join(ref, "launch-132", "out_view.bin")); final_mine = sha(os.path.join(att, "run", "out_view_132.bin"))
    if final_ref != final_mine: failures.append("final launch-132 view differs")
    receipt = {"schema": "neuralstudio.nr.middle-resident.v1", "attempt": att, "reference_middle": os.path.abspath(ref), "encoder_entry_sha256": sha(a.encoder_entry), "poison": a.poison, "checks": checks, "gpu_ms_total": run["gpu_ms_total"], "probe_wall_ms_total": run["wall_ms_total"], "driver_wall_s": round(wall, 1),
               "final_out_view_sha256": final_mine, "failures": failures, "result": "pass" if not failures else "fail"}
    receipt["receipt_root_sha256"] = hashlib.sha256(json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()).hexdigest(); json.dump(receipt, open(os.path.join(att, "resident.json"), "w"), indent=1)
    print("resident middle", receipt["result"], "checks", checks, "failures", failures[:6]); print(f"GPU {run['gpu_ms_total']/1000:.2f} s, probe wall {run['wall_ms_total']/1000:.2f} s, driver wall {wall:.1f} s"); sys.exit(0 if not failures else 1)
if __name__ == "__main__": main()
