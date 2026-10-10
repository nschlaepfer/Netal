#!/usr/bin/env python3
"""CPU tests for the opt-in --single-warp generator prototype, v3 (Codex reviews 05:04, 05:11 and 06:17).

v3: fixtures go to a fresh retained mkdtemp directory recorded in the report - nothing is deleted, so the
counterexamples of every run survive for review.

v2 answers the 05:11 blockers:
  * all THREE dispatcher fences must be present, at the original sites, with the original flags (v1 had one);
  * equal OPERATION COUNTS are not proof - the whole switch body must be textually identical to the accepted
    kernel after normalising ONLY the exit bookkeeping (the exit_count atomic increment);
  * the generator must refuse PREDICATED synchronisation, not just unpredicated `bar.sync`;
  * the host guard must reject threadgroup shapes 0/31/33/64 and a pipeline whose execution width is not 32
    (tools/nr/nr_single_warp_guard_test.cc, compiled and run here), and the CANDIDATE HARNESS must contain it.

No GPU. Usage:
  test-nr-single-warp.py ACCEPTED.metal CANDIDATE.metal PTX_DIR [OUT.json] [DEFAULT_HARNESS] [CANDIDATE_HARNESS]
"""
import json, os, re, subprocess, sys, hashlib, tempfile, datetime
acc_p, cand_p, ptx_dir = sys.argv[1], sys.argv[2], sys.argv[3]
out = sys.argv[4] if len(sys.argv) > 4 else ""
harness_default = sys.argv[5] if len(sys.argv) > 5 else ""
harness_cand = sys.argv[6] if len(sys.argv) > 6 else ""
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
acc, cand = open(acc_p).read(), open(cand_p).read()
cases = []; allok = True
def check(name, ok, detail=""):
    global allok; allok &= bool(ok); cases.append({"case": name, "ok": bool(ok), "detail": detail})
    print(("PASS " if ok else "FAIL ") + name + (" :: " + detail if detail else "")); return ok
def gen(entry_ptx, entry, name, extra, out_file):
    cmd = [sys.executable, os.path.join(ROOT, "scripts/ptx_warp_to_msl.py"), "--ptx", entry_ptx,
           "--entry", entry, "--name", name, "--output", out_file, "--tp-scratch", "--single-warp"] + extra
    return subprocess.run(cmd, capture_output=True, text=True)
def last(r): 
    t = (r.stdout + r.stderr).strip()
    return t.splitlines()[-1][:130] if t else ""
# Codex review 2026-09-08 06:17: this used a FIXED /tmp path and rmtree'd it, destroying the counterexamples of every
# previous run. Each run now gets its own retained directory, recorded in the report; nothing is ever deleted.
tmp = tempfile.mkdtemp(prefix="nr-sw-test-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ-"))
print("test fixtures retained in " + tmp)
POST = "cc_tinlayout_fused_post_block_swin_1h_32_fp8"
post_ptx = os.path.join(ptx_dir, POST + ".ptx")

# ---------- 1. generator rejections (fail closed) ----------
r = gen(os.path.join(ptx_dir, "cg2r_copy_kernel.ptx"), "cg2r_copy_kernel", "ptx_cg2r_copy", [], tmp + "/copy.metal")
check("reject-launch156-copy-kernel", r.returncode != 0 and "copy kernel" in (r.stdout + r.stderr), last(r))
for entry in ("cc_tinlayout_fused_swin_1h_32_1_chained_fp8", "cc_tinlayout_fused_swin_8h_256_8_chained_fp8"):
    r = gen(os.path.join(ptx_dir, entry + ".ptx"), entry, "ptx_x", [], tmp + "/x.metal")
    check("reject-kernel-with-barrier-arrivals:" + entry.split("fused_")[1][:22],
          r.returncode != 0 and "synchronisation instruction" in (r.stdout + r.stderr), last(r))

# 1b. PREDICATED and other synchronisation forms injected into an otherwise ELIGIBLE kernel (05:11 blocker 3).
src_ptx = open(post_ptx).read()
body_at = src_ptx.index("{", src_ptx.index(".visible .entry " + POST))
def inject(line, tag):
    mutated = src_ptx[:body_at + 1] + "\n\t" + line + "\n" + src_ptx[body_at + 1:]
    f = os.path.join(tmp, "inj_" + tag + ".ptx"); open(f, "w").write(mutated)
    r = gen(f, POST, "ptx_post_block_swin_1h_32", [], tmp + "/inj_" + tag + ".metal")
    check("reject-injected-sync:" + tag, r.returncode != 0 and "synchronisation instruction" in (r.stdout + r.stderr), last(r))
inject("@%p1 bar.sync 0;", "predicated-bar-sync")
inject("@!%p1 bar.sync 0;", "predicated-negated-bar-sync")
inject("bar.sync 0;", "plain-bar-sync")
inject("@%p1 barrier.cta.sync 0;", "predicated-barrier-cta-sync")
inject("membar.cta;", "membar-cta")
inject("@%p1 fence.sc.cta;", "predicated-fence-sc-cta")
# control: the UNMUTATED eligible kernel must still be accepted, so the rejections above are not vacuous
r = gen(post_ptx, POST, "ptx_post_block_swin_1h_32", [], tmp + "/ok.metal")
check("accept-eligible-post-kernel", r.returncode == 0 and os.path.exists(tmp + "/ok.metal"), last(r))

# ---------- 2. emitted dispatcher shape: THREE fences, original sites, original flags ----------
BOTH = "simdgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);"
TG   = "simdgroup_barrier(mem_flags::mem_threadgroup);"
check("candidate-has-three-simdgroup-fences", cand.count("simdgroup_barrier(") == 3,
      f"{cand.count('simdgroup_barrier(')} simdgroup_barrier calls (accepted has {acc.count('threadgroup_barrier')-1} in-loop threadgroup barriers)")
check("fence-flags-match-accepted-sites",
      cand.count(BOTH) == 2 and cand.count(TG) == 1 and acc.count("threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);") == 2
      and acc.count("threadgroup_barrier(mem_flags::mem_threadgroup);") == 2,
      "candidate 2x(threadgroup|device) + 1x(threadgroup); accepted 2x(threadgroup|device) in-loop + 1x(threadgroup) in-loop + 1x prologue")
loop = re.search(r"for \(uint iter_ = 0u; iter_ < 4000000u; \+\+iter_\) \{\n(.*?)\n    pc_next = pc \+ 1u;", cand, re.S)
check("dispatcher-loop-head-recognised", loop is not None, "")
if loop:
    lines = [l.split("//")[0].rstrip() for l in loop.group(1).splitlines()]
    want = ["    " + BOTH, "    const bool all_exited = simd_all(done);", "    " + TG, "    if (all_exited) break;", "    " + BOTH, "    if (done) continue;"]
    check("loop-head-is-exactly-the-three-fence-shape", lines == want, " | ".join(l.strip() for l in lines))
# v1 defect (found 2026-09-08 by cross-checking the two independent single-warp paths): the opt-in prologue comment
# ran into the scratch initialiser and commented it out, leaving nr_tp_scratch_arr[0].enabled UNINITIALISED
# threadgroup memory. Assert the initialiser survives as CODE, not inside a comment, and matches the accepted text.
SCRATCH = "if (tlin < 1u) nr_tp_scratch_arr[tlin].enabled = (nthreads <= 32u * 1u);"
check("scratch-init-survives-as-code", SCRATCH in re.sub(r"//[^\n]*", "", cand) and SCRATCH in re.sub(r"//[^\n]*", "", acc),
      "nr_tp_scratch_arr[tlin].enabled is initialised in both the accepted and the candidate kernel")
scr_lines = [i for i, l in enumerate(cand.splitlines()) if SCRATCH in l]
check("prologue-has-no-code-swallowed-by-a-comment",
      len(scr_lines) == 1 and cand.splitlines()[scr_lines[0]].split(SCRATCH)[0].strip() == "",
      f"the scratch initialiser is at line {scr_lines[0] + 1 if scr_lines else -1} with nothing but indentation before it")
check("prologue-threadgroup-barrier-retained", cand.count("threadgroup_barrier") == 1 and "nr_tp_scratch_arr[tlin].enabled" in cand,
      "the scratch-init threadgroup_barrier is the only threadgroup_barrier left")
code_only = re.sub(r"//[^\n]*", "", cand)   # the emitted comments mention the counters they replaced; the CODE must not
acc_code = re.sub(r"//[^\n]*", "", acc)
check("no-threadgroup-atomics", "bar_count" not in code_only and "exit_count" not in code_only and "threadgroup atomic_" not in code_only
      and code_only.count("atomic_") == acc_code.count("atomic_") - 8,
      "bar_count/exit_count and the threadgroup atomic declaration are gone; the 3 device coverage-atomic uses are untouched "
      f"(candidate {code_only.count('atomic_')} vs accepted {acc_code.count('atomic_')} atomic_ mentions)")
check("iteration-cap-unchanged", "iter_ < 4000000u" in cand and "iter_ < 4000000u" in acc, "4,000,000 iteration bound kept")
check("no-early-kernel-return", cand.count("return;") == acc.count("return;"), f"return statements {cand.count('return;')} vs {acc.count('return;')}")

# ---------- 3. FULL case-body equality after ONLY exit-bookkeeping normalisation (05:11 blocker 4) ----------
def body(src):
    i = src.index("    pc_next = pc + 1u;")
    return src[i:]
EXIT_A = "atomic_fetch_add_explicit(&exit_count, 1u, memory_order_relaxed); done = true; break;"
EXIT_B = "default: done = true; atomic_fetch_add_explicit(&exit_count, 1u, memory_order_relaxed); break;"
acc_body, cand_body = body(acc), body(cand)
norm = acc_body.replace(EXIT_A, "done = true; break;").replace(EXIT_B, "default: done = true; break;")
check("exit-bookkeeping-normalisation-applied", norm != acc_body and "exit_count" not in norm,
      f"{acc_body.count(EXIT_A)} ret sites + {acc_body.count(EXIT_B)} default site normalised")
eq = norm == cand_body
detail = "every switch case byte-identical after removing only the exit_count increment"
if not eq:
    import difflib
    d = [l for l in difflib.unified_diff(norm.splitlines(), cand_body.splitlines(), lineterm="", n=0) if l[:1] in "+-" and l[:3] not in ("+++", "---")]
    detail = f"{len(d)} differing lines, first: {d[0][:110] if d else ''}"
check("full-case-body-equality", eq, detail)
check("case-body-is-the-bulk-of-the-kernel",
      acc_body.count("case ") == cand_body.count("case ") and acc_body.count("case ") >= 32
      and len(acc_body.splitlines()) >= int(0.4 * len(acc.splitlines())),
      f"{acc_body.count('case ')} switch cases, {len(acc_body.splitlines())} of {len(acc.splitlines())} lines compared")

# ---------- 3b. two INDEPENDENT single-warp paths must agree on the code ----------
# scripts/nr-single-warp-prototype.py rewrites an already-generated accepted kernel; the translator emits the same
# shape from the PTX. Neither is derived from the other, so agreement on every non-comment line is real evidence.
proto_out = tmp + "/proto.metal"
rp = subprocess.run([sys.executable, os.path.join(ROOT, "scripts/nr-single-warp-prototype.py"),
                     "--input", acc_p, "--output", proto_out, "--report", tmp + "/proto.json"], capture_output=True, text=True)
if check("independent-source-transformer-accepts", rp.returncode == 0, last(rp)):
    strip = lambda t: [x for x in (re.sub(r"//.*", "", l).rstrip() for l in t.splitlines()) if x]
    pa, pb = strip(open(proto_out).read()), strip(cand)
    d = [l for l in __import__("difflib").unified_diff(pa, pb, lineterm="", n=0) if l[:1] in "+-" and l[:3] not in ("+++", "---")]
    check("two-independent-paths-emit-the-same-code", not d, f"{len(pa)} code lines agree" if not d else f"{len(d)} differing lines, first: {d[0][:110]}")

# ---------- 4. host guard: shape/width rejections, compiled and RUN on the CPU ----------
gt = tmp + "/nr_single_warp_guard_test"
rc = subprocess.run(["clang++", "-std=c++20", "-O0", "-Wall", "-Wextra", "-Werror", "-I", os.path.join(ROOT, "src/nr_temporal"),
                     os.path.join(ROOT, "tools/nr/nr_single_warp_guard_test.cc"), "-o", gt], capture_output=True, text=True)
if check("host-guard-test-compiles", rc.returncode == 0, (rc.stderr or "").strip().splitlines()[-1][:120] if rc.returncode else ""):
    rg = subprocess.run([gt], capture_output=True, text=True)
    passed = [l for l in rg.stdout.splitlines() if l.startswith("PASS")]; failed = [l for l in rg.stdout.splitlines() if l.startswith("FAIL")]
    check("host-guard-rejects-bad-shapes-and-widths", rg.returncode == 0 and not failed,
          f"{len(passed)} predicate cases passed (0/31/33/64, non-unit h/d, widths 0/1/16/31/64, maxTotal<32)")
    for tag in ("reject-shape-w0", "reject-shape-w31", "reject-shape-w33", "reject-shape-w64", "reject-width-16", "reject-width-64"):
        check("host-guard-case-present:" + tag, any(l.startswith("PASS " + tag) for l in passed), "")

# ---------- 5. the guard is actually LINKED INTO the candidate harness (05:11 blocker 1) ----------
GUARDSTR = b"single-warp launch-155 guard refused the dispatch"
if harness_cand:
    b = open(harness_cand, "rb").read()
    check("candidate-harness-contains-the-guard", GUARDSTR in b and b"single-warp-guard:launch155" in b,
          f"{os.path.basename(harness_cand)} sha {hashlib.sha256(b).hexdigest()[:16]}")
if harness_default:
    b = open(harness_default, "rb").read()
    check("default-harness-has-no-guard", GUARDSTR not in b, f"{os.path.basename(harness_default)} sha {hashlib.sha256(b).hexdigest()[:16]} (accepted build unaffected)")

# ---------- 5b. the guard runs before ANY encoder for launch 155 is opened (Codex 06:17 repair 2) ----------
body_src = open(os.path.join(ROOT, "src/nr_body/nr_body_runtime_v17.mm")).read()
regions = [(m.start(), body_src.index("#endif\n", m.start())) for m in re.finditer(r"#ifdef NR_BODY_SINGLE_WARP", body_src)]
check("guard-source-regions", len(regions) == 2, f"{len(regions)} #ifdef NR_BODY_SINGLE_WARP regions (include + call site)")
if len(regions) == 2:
    gstart, gend = regions[1]
    tail = body_src[gend:]
    nxt_enc = min([x for x in (tail.find("blit();"), tail.find("compute();")) if x >= 0] or [-1])
    prev_pipe = body_src.rfind("id<MTLComputePipelineState> dps = pso(", 0, gstart)
    between = body_src[prev_pipe:gstart]
    check("guard-is-after-the-actual-pipeline", prev_pipe > 0 and "blit();" not in between and "compute();" not in between,
          "the guard sees the real pipeline for this launch, with no encoder opened between them")
    check("guard-precedes-every-encoder-for-155", nxt_enc >= 0,
          f"the next encoder is opened {nxt_enc} chars after the guard block, so a refusal throws with none open")
    check("guard-fails-closed-before-dispatch",
          "require(swok," in body_src[gstart:gend] and body_src.index("dispatchThreadgroups", gend) > gend,
          "require(swok, ...) throws before the dispatch is encoded")
# the transitive header must be covered by the immutable build manifest (Codex 06:17 repair 1)
for h in (harness_default, harness_cand):
    if not h: continue
    man = os.path.join(os.path.dirname(h), "build-sha256.txt")
    txt = open(man).read() if os.path.exists(man) else ""
    check("build-manifest-covers-the-guard-header:" + os.path.basename(os.path.dirname(h)),
          "nr_single_warp_guard.h" in txt and "nr_body_runtime_v17.mm" in txt and "nr_temporal.h" in txt,
          f"{txt.count(chr(10))} hashed entries including the shared nr_temporal headers")

# ---------- 6. control-logic simulation of the THREE-fence loop ----------
def simulate(done_at, cap=20):
    done = [False] * 32; runs = [0] * 32
    for it in range(cap):
        all_exited = all(done)                 # computed between fence 1 and fence 2 by every lane
        if all_exited: return it, runs         # break after fence 2
        for l in range(32):                    # fence 3, then the per-lane guard
            if done[l]: continue
            runs[l] += 1
            if runs[l] >= done_at[l]: done[l] = True
    return cap, runs
it, runs = simulate([3] * 32); check("control-all-done-together", it == 3 and set(runs) == {3}, f"exit at iteration {it}")
stag = [1 + (l % 5) for l in range(32)]; it, runs = simulate(stag)
check("control-staggered-done", it == max(stag) and runs == stag, f"exit at iteration {it}, every lane ran its own state count")
it, runs = simulate([999] * 32, cap=20); check("control-no-done-hits-iteration-cap", it == 20, "loop is bounded when no lane finishes")
it, runs = simulate([1] + [4] * 31); check("control-one-lane-early-others-continue", it == 4 and runs[0] == 1 and runs[1] == 4, "an early lane does not terminate the group")

rep = {"schema": "neuralstudio.nr.single-warp-tests.v3", "accepted": acc_p, "accepted_sha256": hashlib.sha256(acc.encode()).hexdigest(),
       "candidate": cand_p, "candidate_sha256": hashlib.sha256(cand.encode()).hexdigest(),
       "harness_default": harness_default, "harness_candidate": harness_cand, "fixtures_dir": tmp, "fixtures_retained": True,
       "cases": cases, "all_ok": allok, "case_count": len(cases),
       "note": "CPU only: generator, emitted-shape, full case-body equality, host-guard predicate and linkage. NOT proof of Metal execution correctness."}
if out: json.dump(rep, open(out, "w"), indent=1)
print(("ALL %d PASS" if allok else "FAILURES in %d cases") % len(cases))
sys.exit(0 if allok else 1)
