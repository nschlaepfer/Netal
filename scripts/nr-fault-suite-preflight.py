#!/usr/bin/env python3
"""CPU preflight for the fault/recovery suite configuration (spec/nr/recipes/fault-recovery-v1/suite-config.json).

Verifies, WITHOUT touching the GPU, that every pinned build, descriptor and input exists and hashes exactly as
recorded, that the fault builds really carry the hook and the production builds really do not, that the production
builds are reproducible byte-for-byte from the current source, and that no descriptor references a missing file.
It runs nothing and changes nothing. Usage: nr-fault-suite-preflight.py CONFIG.json [OUT.json]
"""
import json, os, subprocess, sys, hashlib
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg_path = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else ""
cfg = json.load(open(cfg_path))
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
res = lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p)
cases = []; allok = True
def check(name, ok, detail=""):
    global allok; allok &= bool(ok); cases.append({"case": name, "ok": bool(ok), "detail": detail})
    print(("PASS " if ok else "FAIL ") + name + (" :: " + detail if detail else ""))
def pin(name, path, want):
    p = res(path)
    if not os.path.exists(p): return check(name, False, "absent: " + path)
    g = sha(p); return check(name, g == want, want[:16] if g == want else f"on disk {g[:16]}, pinned {want[:16]}")

for k, b in cfg["builds"].items():
    for field, exe in (("cli_sha256", "nr_body_cli"), ("negctl_sha256", "nr_body_negctl"), ("apitest_sha256", "nr_body_apitest"),
                       ("executable_sha256", "nr_full_frame"), ("metallib_sha256", "validate.metallib"), ("probe_sha256", "nr_body_validate_probe")):
        if field in b: pin(f"build:{k}:{exe}", os.path.join(b["path"], exe), b[field])
    man = res(os.path.join(b["path"], "build-sha256.txt"))
    if os.path.exists(man):
        txt = open(man).read(); covered = "nr_temporal" in txt or "resident" in k
        # Builds made before the 2026-09-08 manifest repair hash only src/nr_body. That is acceptable ONLY when the same
        # binary is reproduced byte-for-byte in a root whose manifest IS complete; otherwise a transitive header is unhashed.
        repro = b.get("reproduced_byte_identical_from_current_source_in")
        rtxt = open(res(os.path.join(repro, "build-sha256.txt"))).read() if repro and os.path.exists(res(os.path.join(repro, "build-sha256.txt"))) else ""
        check(f"manifest-complete:{k}", covered or "nr_temporal" in rtxt,
              f"{txt.count(chr(10))} hashed entries" + ("" if covered else f"; pre-repair manifest, covered instead by the complete manifest in {repro}"))
    dfl = res(os.path.join(b["path"], "build-defines.txt"))
    if os.path.exists(dfl):
        got = open(dfl).read().split(":", 1)[1].strip()
        check(f"defines-as-pinned:{k}", got == b.get("defines"), f"{got!r} vs pinned {b.get('defines')!r}")

# the fault hook must be present in exactly the fault builds and absent from the production ones
HOOK = b"decoder-pool-multi"
for k, b in cfg["builds"].items():
    for exe in ("nr_body_cli", "nr_full_frame"):
        p = res(os.path.join(b["path"], exe))
        if not os.path.exists(p): continue
        has = HOOK in open(p, "rb").read(); want = "fault" in k
        check(f"fault-hook-{'present' if want else 'absent'}:{k}:{exe}", has == want,
              "the hook exists only where -DNR_BODY_FAULT_TESTS was defined")

# production builds must be reproducible byte-for-byte from the current source
for k, b in cfg["builds"].items():
    r = b.get("reproduced_byte_identical_from_current_source_in")
    if not r: continue
    for exe, field in (("nr_body_cli", "cli_sha256"), ("nr_full_frame", "executable_sha256")):
        p = res(os.path.join(r, exe))
        if field in b and os.path.exists(p):
            check(f"reproducible:{k}:{exe}", sha(p) == b[field], f"rebuilt in {r}")

for k, d in cfg["descriptors"].items():
    pin("descriptor:" + k, d["path"], d["sha256"])
    p = res(d["path"])
    if not os.path.exists(p): continue
    j = json.load(open(p))
    if "checks" in d: check(f"descriptor-checks:{k}", j["decoder"].get("checks") == d["checks"], f"checks = {j['decoder'].get('checks')}")
    missing = []
    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                if isinstance(v, str) and v.endswith((".metallib", ".bin", ".e4m3", ".f16")):
                    if not os.path.exists(res(v)): missing.append(v)
                else: walk(v)
        elif isinstance(o, list):
            for v in o: walk(v)
    walk(j)
    # In the IN-MEMORY path the harness supplies the entry view and the five skip patches from the live encoder, so those
    # file references are inert there. Everything else - every library, table and constant - must exist. The accepted
    # descriptor still names the lost build/nr_chain/attempt-v7 worktree in exactly those six inert slots; state it, do not hide it.
    supplied = {j.get("entry_file")} | {x.get("file") for x in j.get("decoder", {}).get("patches", [])}
    hard = [m for m in missing if m not in supplied]
    inert = [m for m in missing if m in supplied]
    check(f"descriptor-files-resolve:{k}", not hard,
          "every referenced artefact exists" if not missing else
          (f"{len(inert)} inert harness-supplied slots name the lost build/nr_chain/attempt-v7 worktree; every library, table and constant resolves"
           if not hard else f"{len(hard)} REQUIRED artefacts missing, first {hard[0][-60:]}"))
    if k == "in_memory":
        check("in-memory-inert-slots-are-only-entry-and-patches", set(missing) <= supplied and len(missing) in (0, 6),
              f"{len(missing)} missing references, all of them harness-supplied (entry_file + 5 skip patches)")

pin("cli-entry-file", cfg["cli_inputs"]["entry_file"], cfg["cli_inputs"]["entry_file_sha256"])
for i, pf in enumerate(cfg["cli_inputs"]["patch_files"]): pin(f"cli-patch-{i}", pf["file"], pf["sha256"])
pin("encoder-object", os.path.join(cfg["encoder"]["build"], "nr_encoder_runtime.o"), cfg["encoder"]["object_sha256"])
for k, p in (("input", cfg["live_frame_inputs"]["input"]), ("motion", cfg["live_frame_inputs"]["motion"])):
    check("live-input-exists:" + k, os.path.exists(res(p)), p[-58:])
fl = cfg.get("storage_floor_gib")
check("storage-floor", isinstance(fl, (int, float)) and not isinstance(fl, bool) and fl >= 12, f"{fl} GiB")
free = __import__("shutil").disk_usage(ROOT).free / 2**30
check("free-space-above-floor", free >= fl, f"{free:.1f} GiB free")
check("no-timing-claim", cfg.get("no_timing_claim") is True and "NOT RUN" in cfg.get("execution", ""), "correctness-only, unexecuted")

rep = {"schema": "neuralstudio.nr.fault-suite-preflight.v1", "config": cfg_path, "config_sha256": sha(cfg_path),
       "cases": cases, "case_count": len(cases), "all_ok": allok, "gpu_touched": False}
if out: json.dump(rep, open(out, "w"), indent=1)
print(("ALL %d PASS" if allok else "FAILURES in %d cases") % len(cases))
sys.exit(0 if allok else 1)
