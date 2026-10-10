#!/usr/bin/env python3
"""Serve-mode driver: one fused middle+decoder process (middle_chain_probe --serve) stays alive; each frame's chain.json is built with
nr-body-resident.py --chain-only, sent to the server, then checked with --check-only (same checks as the one-shot runner). Records the per-frame
server wall (includes nothing but the frame: pipelines, libraries, device are resident) so the fixed per-process cost is visible as frame 1 vs later.
Correctness is unchanged: fresh poison per frame, every check per frame."""
import argparse, json, os, subprocess, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); PY = sys.executable
def main():
    p = argparse.ArgumentParser(); p.add_argument("--attempt", required=True); p.add_argument("--reference-attempt", required=True); p.add_argument("--keys", default="a5-a,a5-b,5a-a,5a-b"); p.add_argument("--encoder-runs", required=True, help="dir containing <key>/ encoder runs")
    p.add_argument("--split-attempt", required=True); p.add_argument("--vit-attempt", required=True); p.add_argument("--dec-lib", required=True); p.add_argument("--post-lib", required=True); p.add_argument("--copy-lib", required=True); p.add_argument("--probe", required=True, help="serve-capable middle_chain_probe")
    p.add_argument("--production", action="store_true"); p.add_argument("--vit-tp-lib", default=""); p.add_argument("--split-tp-lib", default=""); p.add_argument("--repeat", type=int, default=1, help="pass over the keys this many times (later passes show the resident-process frame time)")
    a = p.parse_args(); att = os.path.abspath(a.attempt)
    if os.path.exists(att): raise SystemExit("preserve previous attempt: " + att)
    os.makedirs(att); keys = a.keys.split(","); common = ["--reference-attempt", a.reference_attempt, "--split-attempt", a.split_attempt, "--vit-attempt", a.vit_attempt, "--dec-lib", a.dec_lib, "--post-lib", a.post_lib, "--copy-lib", a.copy_lib, "--probe", a.probe] + (["--production"] if a.production else []) + (["--vit-tp-lib", a.vit_tp_lib] if a.vit_tp_lib else []) + (["--split-tp-lib", a.split_tp_lib] if a.split_tp_lib else [])
    frames = []
    for rep in range(a.repeat):
        for key in keys:
            fdir = os.path.join(att, f"frame-{rep}-{key}"); r = subprocess.run([PY, os.path.join(ROOT, "scripts/nr-body-resident.py"), "--attempt", fdir, "--key", key, "--encoder-run", os.path.join(a.encoder_runs, key), "--chain-only"] + common, capture_output=True, text=True)
            if r.returncode: raise SystemExit(f"chain build failed for {key}: {r.stderr[-600:]}")
            frames.append((rep, key, fdir))
    t_start = time.time(); srv = subprocess.Popen([a.probe, "--serve"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1); results = []
    for rep, key, fdir in frames:
        t0 = time.time(); srv.stdin.write(f"{os.path.join(fdir, 'chain.json')} {os.path.join(fdir, 'run')}\n"); srv.stdin.flush()
        line = ""
        while True:   # the server prints per-launch progress lines; the frame ends with "done <rc> <ms>"
            l = srv.stdout.readline()
            if not l: break
            if l.startswith("done "): line = l.strip(); break
        wall = time.time() - t0
        parts = line.split(); rc = int(parts[1]) if len(parts) >= 2 else 2; ms = float(parts[2]) if len(parts) >= 3 else -1.0
        open(os.path.join(fdir, "serve-wall-ms.txt"), "w").write(str(ms))
        chk = subprocess.run([PY, os.path.join(ROOT, "scripts/nr-body-resident.py"), "--attempt", fdir, "--key", key, "--encoder-run", os.path.join(a.encoder_runs, key), "--check-only"] + common, capture_output=True, text=True)
        rec = json.load(open(os.path.join(fdir, "resident.json"))) if os.path.exists(os.path.join(fdir, "resident.json")) else {"result": "missing"}
        results.append({"pass": rep, "key": key, "server_rc": rc, "server_frame_ms": round(ms, 1), "driver_roundtrip_s": round(wall, 2), "result": rec.get("result"), "checks": rec.get("checks"), "gpu_s_total": rec.get("gpu_s_total"), "middle_gpu_s": rec.get("middle_gpu_s"), "decoder_gpu_s": rec.get("decoder_gpu_s"), "check_rc": chk.returncode})
        print(f"pass {rep} {key}: server rc {rc} frame {ms:.0f} ms, GPU {rec.get('gpu_s_total')} s, result {rec.get('result')} ({rec.get('checks')} checks)")
    srv.stdin.close(); srv.wait(timeout=60); total = time.time() - t_start
    summary = {"schema": "neuralstudio.nr.body-serve.v1", "attempt": att, "mode": "production" if a.production else "checked", "frames": results, "total_wall_s": round(total, 2), "probe": os.path.abspath(a.probe), "result": "pass" if all(f["result"] == "pass" and f["server_rc"] == 0 for f in results) else "fail"}
    json.dump(summary, open(os.path.join(att, "serve.json"), "w"), indent=1); print("serve", summary["result"], "frames", len(results), "total wall", round(total, 1), "s")
    sys.exit(0 if summary["result"] == "pass" else 1)
if __name__ == "__main__": main()
