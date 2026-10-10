#!/usr/bin/env python3
"""Audit a regression directory (build/nr_full/regression-*): every config/replica receipt, per-frame pass, history bit-equality, state advance, EX2
faults, and (where an RTX matrix with boundary ports exists) the six producer ports per frame. Seals summary.json. Usage: nr-regression-audit.py ROOT CODEX_CHECKOUT"""
import json, hashlib, os, sys
from importlib.machinery import SourceFileLoader
ROOT, CX = sys.argv[1], sys.argv[2]; sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
pc = SourceFileLoader("pc", os.path.join(os.path.dirname(os.path.abspath(__file__)), "nr-varied-ports-compare.py")).load_module()
ports = {"black3": f"{CX}/build/nr_varied_temporal/case1-matrix-v1.json", "checker3": f"{CX}/build/nr_varied_temporal/case2-matrix-v1.json", "images3": f"{CX}/build/nr_image_sequence/matrix-audit-v1.json", "uniform3": f"{CX}/build/nr_motion_reference/matrix-audit-v1.json", "spatial3": f"{CX}/build/nr_spatial_motion/matrix-audit-v1.json"}
requested = ["black3", "checker3", "dense-case3-reset", "images3", "midreset8", "uniform3", "spatial3"]; cfgs = {}
for cfg in sorted(os.listdir(ROOT)):
    if not os.path.isdir(os.path.join(ROOT, cfg)): continue
    runs = []
    for tag in sorted(os.listdir(os.path.join(ROOT, cfg))):
        rp = os.path.join(ROOT, cfg, tag, "receipt.json")
        if not os.path.exists(rp): runs.append({"tag": tag, "result": "no-receipt"}); continue
        d = json.load(open(rp)); fr = d.get("frames", []); good = [f for f in fr if "body" in f]; pk = None
        if cfg in ports and d["result"] == "pass":
            try: pk, _ = pc.compare(rp, ports[cfg])
            except Exception as e: pk = f"error:{e}"
        runs.append({"tag": tag, "pid": d.get("pid"), "result": d["result"], "error": d.get("error"), "receipt_sha256": sha(rp), "executable_sha256": d.get("executable_sha256"), "frames": len(fr), "all_frames_pass": bool(fr) and all(f.get("result") == "pass" for f in fr) and len(good) == len(fr), "mismatching_values": [f["reference"]["mismatching_values"] for f in good], "history_diffs": [f["history_copy_value_differences"] for f in good], "state_advanced": [f["temporal_state_advanced"] for f in good], "ex2_faults": [f["ex2_half_faults"] for f in good], "finals": [f["surface_sha256_caller_readback"] for f in good], "six_ports_equal_rtx": pk})
    cfgs[cfg] = {"runs": runs, "replicas_passed": sum(1 for r in runs if r["result"] == "pass" and r["all_frames_pass"]), "all_pass": bool(runs) and all(r["result"] == "pass" and r["all_frames_pass"] and r.get("six_ports_equal_rtx") in (True, None) for r in runs)}
    print(f"{cfg:45s} runs={len(runs)} passed={cfgs[cfg]['replicas_passed']} all_pass={cfgs[cfg]['all_pass']} ports={[r.get('six_ports_equal_rtx') for r in runs]}")
S = {"schema": "neuralstudio.nr.regression-halffma.v1", "label": "Prior-mode regression on Codex's accepted half-fma candidate (zero-motion cases: harness v19t/half-fma-v1, body v11, desc-v23t; motion cases: harness v20t/half-fma-spatial-v1, desc-v24t, sampler_variant 3). Each requested configuration: pilot + three fresh poison replicas (four processes), storage admitted per config, references CPU-only, history live, fail-stop on any replica in the follow-up gate. Misconfigured/extra runs preserved under renamed directories and excluded from the requested set. No default/accepted object replaced.", "requested_configs": {c: cfgs.get(c, {"runs": [], "all_pass": False, "replicas_passed": 0}) for c in requested}, "other_dirs": {c: cfgs[c] for c in cfgs if c not in requested}, "harness": {"v19t_sha256": sha("build/nr_full/v19t/nr_full_frame"), "v20t_sha256": sha("build/nr_full/v20t/nr_full_frame"), "half_fma_v1_object": sha(f"{CX}/build/nr_encoder_temporal/half-fma-v1/nr_encoder_runtime.o"), "half_fma_spatial_v1_object": sha(f"{CX}/build/nr_encoder_temporal/half-fma-spatial-v1/nr_encoder_runtime.o"), "desc_v23t": sha("build/nr_body/desc-v23t/chain.json"), "desc_v24t": sha("build/nr_body/desc-v24t/chain.json")}, "all_requested_pass": all(cfgs.get(c, {}).get("all_pass") and cfgs.get(c, {}).get("replicas_passed") == 4 for c in requested), "receipt_root_rule": "sha256 of this document serialised with indent=1 without receipt_root_sha256"}
S["receipt_root_sha256"] = hashlib.sha256(json.dumps(S, indent=1).encode()).hexdigest(); json.dump(S, open(os.path.join(ROOT, "summary.json"), "w"), indent=1); print("REGRESSION ROOT", S["receipt_root_sha256"], "ALL REQUESTED PASS", S["all_requested_pass"])
