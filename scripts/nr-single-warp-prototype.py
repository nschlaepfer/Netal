#!/usr/bin/env python3
"""Prototype transformer for the launch-155 single-warp specialisation (docs/NR_LAUNCH155_CENSUS.md, gate 1).

CPU only. It reads an ACCEPTED generated kernel, refuses unless the kernel is provably eligible, and writes a NEW
specialised source beside it. It never edits the translator, the accepted sources or any library, and it proves
that the only textual changes are in the dispatcher: every arithmetic, memory and texture line must survive
byte-for-byte, which is checked and reported.

Eligibility (all required, else hard refusal):
  * zero CTA barrier arrivals  (no `bar_count, 1u` increment anywhere)
  * the emulated shared memory `smem` is declared but never referenced
  * the dispatcher has the expected shape (three threadgroup_barrier calls and the bar_count/exit_count protocol)

Specialisation (v2, after Codex's 05:11 review - fence elimination was NOT approved): ALL THREE dispatcher fences
are retained at their original sites with their original flags, as `simdgroup_barrier`; the only change is the exit
BOOKKEEPING (`simd_all(done)` in place of the shared exit counter) and the removal of the bar.sync release arm,
which is dead because eligibility proves the kernel has zero barrier arrivals. Arithmetic, rounding, tables,
textures and the iteration cap are untouched, and the proof below is full case-body equality, not operation counts.

Usage: nr-single-warp-prototype.py --input GEN.metal --output OUT.metal [--report R.json]
"""
import argparse, hashlib, json, re, sys, collections
sha = lambda b: hashlib.sha256(b).hexdigest()
OPS = ("nr_mma_e4m3", "nr_tex2d", "ex2", "rcp", "rsqrt", "nr_ld_param64", "nr_ld_param32", "nr_tp_scratch", "fault")
def main():
    p = argparse.ArgumentParser(); p.add_argument("--input", required=True); p.add_argument("--output", required=True); p.add_argument("--report", default="")
    a = p.parse_args(); src = open(a.input).read()
    refuse = []
    if src.count("bar_count, 1u"): refuse.append(f"kernel has {src.count('bar_count, 1u')} CTA barrier arrivals: NOT eligible")
    if src.count("smem") != 1: refuse.append(f"emulated shared memory is referenced {src.count('smem')-1} times beyond its declaration: NOT eligible")
    if src.count("threadgroup_barrier") != 4: refuse.append(f"unexpected dispatcher shape: {src.count('threadgroup_barrier')} threadgroup_barrier sites (expected 4)")
    if "bar_count, exit_count;" not in src: refuse.append("dispatcher protocol not recognised")
    if refuse:
        print("REFUSED:"); [print("  -", r) for r in refuse]; sys.exit(3)
    out = src
    # 1. prologue: drop the threadgroup atomics and their initialisation, keep the scratch init and its barrier
    out = out.replace("  threadgroup atomic_uint bar_count, exit_count;\n", "  // single-warp specialisation: no CTA-barrier protocol (this kernel has zero bar.sync arrivals)\n")
    out = re.sub(r"if \(tlin == 0u\) \{ atomic_store_explicit\(&bar_count, 0u, memory_order_relaxed\); atomic_store_explicit\(&exit_count, 0u, memory_order_relaxed\); \} ", "", out)
    # 2. dispatcher head: three threadgroup barriers + shared counters -> one simdgroup fence + simd_all
    old_head = re.search(r"    threadgroup_barrier\(mem_flags::mem_threadgroup \| mem_flags::mem_device\);\n"
                         r"    const uint waiting = .*?\n"
                         r"    const bool all_exited = .*?\n"
                         r"    threadgroup_barrier\(mem_flags::mem_threadgroup\);\n"
                         r"    if \(all_exited\) break;\n"
                         r"    if \(release\) \{[^\n]*\}\n"
                         r"    threadgroup_barrier\(mem_flags::mem_threadgroup \| mem_flags::mem_device\);\n"
                         r"    if \(done \|\| bar_wait\) continue;\n", out, re.S)
    if not old_head: print("REFUSED: dispatcher loop head did not match the expected pattern"); sys.exit(3)
    new_head = ("    simdgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);   // fence 1 of 3 - original site, original flags\n"
                "    const bool all_exited = simd_all(done);                                  // exit bookkeeping only: uniform vote replaces exit_count == nthreads\n"
                "    simdgroup_barrier(mem_flags::mem_threadgroup);                           // fence 2 of 3 - original site, original flags\n"
                "    if (all_exited) break;\n"
                "    simdgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);   // fence 3 of 3 - original site, original flags\n"
                "    if (done) continue;\n")
    out = out[:old_head.start()] + new_head + out[old_head.end():]
    # 3. exit sites: the shared counter is gone; `done` alone terminates via simd_all
    out = out.replace("atomic_fetch_add_explicit(&exit_count, 1u, memory_order_relaxed); done = true;", "done = true;")
    out = out.replace("done = true; atomic_fetch_add_explicit(&exit_count, 1u, memory_order_relaxed);", "done = true;")
    out = out.replace("uint pc = 0u, pc_next = 0u; bool done = false, bar_wait = false;", "uint pc = 0u, pc_next = 0u; bool done = false;")
    open(a.output, "w").write(out)
    # proof: only dispatcher lines changed; every operation count identical
    a_lines, b_lines = src.splitlines(), out.splitlines()
    import difflib
    changed = [l for l in difflib.unified_diff(a_lines, b_lines, lineterm="", n=0) if l[:1] in "+-" and l[:3] not in ("+++", "---")]
    counts = {op: (src.count(op), out.count(op)) for op in OPS}
    bad_ops = {k: v for k, v in counts.items() if v[0] != v[1]}
    # Codex 05:11: equal operation COUNTS are not proof. The whole switch body must be textually identical after
    # normalising ONLY the exit bookkeeping, so any reordering, retyping or respelling inside a case is a failure.
    cut = lambda t: t[t.index("    pc_next = pc + 1u;"):]
    a_body = cut(src).replace("atomic_fetch_add_explicit(&exit_count, 1u, memory_order_relaxed); done = true; break;", "done = true; break;")
    a_body = a_body.replace("default: done = true; atomic_fetch_add_explicit(&exit_count, 1u, memory_order_relaxed); break;", "default: done = true; break;")
    b_body = cut(out); body_equal = a_body == b_body
    body_cases = a_body.count("case "); body_lines = len(a_body.splitlines())
    rep = {"schema": "neuralstudio.nr.single-warp-prototype.v2", "input": a.input, "input_sha256": sha(src.encode()),
           "output": a.output, "output_sha256": sha(out.encode()), "changed_lines": len(changed),
           "changed_line_samples": changed[:14], "operation_counts_input_vs_output": counts,
           "operation_counts_identical": not bad_ops,
           "full_case_body_equal_after_exit_bookkeeping_normalisation": body_equal,
           "case_body_cases_compared": body_cases, "case_body_lines_compared": body_lines,
           "fences": {"retained": out.count("simdgroup_barrier("), "flags_threadgroup_device": out.count("simdgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);"),
                      "flags_threadgroup": out.count("simdgroup_barrier(mem_flags::mem_threadgroup);")},
           "eligibility": {"bar_sync_arrivals": 0, "smem_references_beyond_declaration": 0},
           "note": "prototype source only: not compiled into any accepted library, not run on the GPU, no arithmetic change intended or observed"}
    if a.report: json.dump(rep, open(a.report, "w"), indent=1)
    print(f"changed lines: {len(changed)} | fences retained: {out.count('simdgroup_barrier(')} | "
          f"full case-body equality ({body_cases} cases, {body_lines} lines): {body_equal} | operation counts identical: {not bad_ops}"
          + ("" if not bad_ops else f" | DIFFERING: {bad_ops}"))
    for l in changed[:10]: print("   ", l[:150])
    sys.exit(0 if (not bad_ops and body_equal and out.count("simdgroup_barrier(") == 3) else 1)
if __name__ == "__main__": main()
