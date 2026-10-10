#!/usr/bin/env python3
"""Seal exact uint32 full-runtime correctness and measurements."""
from __future__ import annotations
import argparse,hashlib,json,shutil,subprocess
from pathlib import Path
from typing import Any

def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def cp(a:Path,b:Path)->None:
    if not a.is_file():raise SystemExit("missing input: "+str(a))
    shutil.copyfile(a,b)
def write(p:Path,v:dict[str,Any])->None:p.write_text(json.dumps(v,indent=2,sort_keys=True)+"\n")

def main()->int:
    q=argparse.ArgumentParser()
    for name in ("repo","u32","clz","rtx","benchmark","profile","evidence"):
        q.add_argument("--"+name,type=Path,required=True)
    a=q.parse_args();repo=a.repo.resolve();u32=a.u32.resolve();clz=a.clz.resolve();rtx=a.rtx.resolve()
    bench=a.benchmark.resolve();profile=a.profile.resolve();evidence=a.evidence.resolve()
    if evidence.exists():raise SystemExit("preserve evidence: "+str(evidence))
    evidence.mkdir(parents=True)
    sources=["scripts/build-macos-fg-runtime.sh","scripts/build-macos-fg-runtime-compact-ex2.sh",
      "scripts/build-macos-fg-runtime-u32.sh","scripts/benchmark-macos-fg-runtime-u32.sh",
      "scripts/profile-macos-fg-runtime-u32.sh","scripts/verify-fg-runtime-u32.py",
      "scripts/seal-fg-runtime-u32.py","scripts/verify-fg-runtime-live-alias.py",
      "src/fg_runtime/fg_runtime.h","src/fg_runtime/fg_runtime.mm","src/fg_runtime/benchmark_host.mm",
      "src/fg_runtime/offline_host.mm","src/fg_runtime/tail_output_bridge.h",
      "src/fg_runtime/tail_output_bridge.mm","src/fg_runtime/tail_resource_cache.inc",
      "src/fg_conv_simd_l20/hmma_exact_u32.metalh",
      "src/fg_conv_simd_l20/hmma_exact_u32_override.metalh"]
    ur=u32.relative_to(repo);cr=clz.relative_to(repo);br=bench.relative_to(repo);pr=profile.relative_to(repo)
    archive=rtx.parent/(rtx.name+".tar.gz")
    builds=[str(ur/p) for p in (Path("backend-receipt.json"),Path("compact-promotion-receipt.json"),
      Path("u32-promotion-receipt.json"),Path("fg-runtime-benchmark"),Path("fg-runtime-offline"),
      Path("build/runtime_tail.mm"),Path("build/hmma-u32-sources.txt"))]
    builds += [str(cr/"fg-runtime-benchmark"),str(cr/"build/runtime_tail.mm"),str(br/"comparison.json"),
      str(pr/"profile.json"),str(archive.relative_to(repo)),
      "build/fg_math_ex2/compact/attempt-v4/packed-a/unit-corrections.bin",
      "build/fg_math_ex2/compact/attempt-v4/packed-a/fraction-corrections.bin"]
    libs=("conv_3x3","conv_1x1","conv_pre","conv0","conv1","conv2","bot0","bot1",
      "block1_conv0","block1_conv1","block1_conv2","block1_bot0","block1_bot1")
    for name in libs:builds += [str(ur/"libs"/f"{name}.metallib"),str(cr/"libs"/f"{name}.metallib")]
    overlay=Path(str(u32)+"-overlay");orr=overlay.relative_to(repo)
    for name in ("conv_3x3_live_zero.metal","block0_bot0_live_zero.metal","l52_compact.metal","l54_compact.metal"):
        builds.append(str(orr/name))
    for rel in sources+builds:
        if not (repo/rel).is_file():raise SystemExit("missing pin: "+rel)
    bv=json.loads((bench/"comparison.json").read_text());pv=json.loads((profile/"profile.json").read_text())
    provenance={"schema":"neuralstudio.fg.runtime-u32-provenance.v1",
      "dependencies":{"clz_runtime_commit":"02077c88fd29c2a052e2487caa17d12177fa44d8",
        "u32_primitive_commit":"c27cf534588b9fa2ec4a0428ce9041f464eb75df",
        "u32_parent_verification_root":"e59c12715f9de63cfda8da6ff7773481adf930b0fb5b6f27f378de4e045cb8c9",
        "live_alias_checkpoint_root":"2b13a70a5d44659a6912c872a01adfd0e9eceaa02b668743cef82a0ea088a156"},
      "u32_attempt":str(ur),"clz_attempt":str(cr),"rtx_attempt":str(rtx.relative_to(repo)),
      "benchmark_attempt":str(br),"profile_attempt":str(pr),
      "benchmark_sha256":sha(bench/"comparison.json"),"profile_sha256":sha(profile/"profile.json"),
      "rtx_archive_sha256":sha(archive),"source_sha256":{x:sha(repo/x) for x in sources},
      "build_artifact_sha256":{x:sha(repo/x) for x in builds},"changed_metallib_count":13,
      "unchanged_metallib_count":24,"expected_or_archived_intermediate_ingress":False,
      "address_fields_recorded":False,"result":"pass"}
    cp(u32/"backend-receipt.json",evidence/"backend-receipt.json")
    cp(u32/"compact-promotion-receipt.json",evidence/"compact-promotion-receipt.json")
    cp(u32/"u32-promotion-receipt.json",evidence/"promotion-receipt.json")
    cp(u32/"build/hmma-u32-sources.txt",evidence/"compiled-hmma-sources.txt")
    for run in sorted(p for p in u32.glob("fg-runtime-poison-*-process-*") if p.is_dir()):
        cp(run/"run-receipt.json",evidence/f"correctness-{run.name}.json")
    cp(bench/"comparison.json",evidence/"benchmark-comparison.json")
    cp(profile/"profile.json",evidence/"boundary-profile.json")
    for mode in ("clz","u32"):
        for i in range(1,4):
            cp(bench/mode/f"run-{i}.json",evidence/f"benchmark-{mode}-run-{i}.json")
            cp(profile/mode/f"run-{i}.json",evidence/f"profile-{mode}-run-{i}.json")
    cross=repo/"build/fg_runtime/u32-seal-v1"
    cp(cross/"cross-backend-verification.json",evidence/"cross-backend-verification.json")
    cp(cross/"cross-backend-negative-audit.json",evidence/"cross-backend-negative-audit.json")
    write(evidence/"provenance.json",provenance)
    (evidence/"README.md").write_text(f"""# Exact uint32 HMMA promotion for the full live FG runtime

The bounded uint32 HMMA implementation is promoted through all 13
HMMA-consuming Metal libraries. Four fresh Mac processes across `0xA5` and
`0x5A` match the RTX oracle at all 1,149 checkpoints and all 15 retrieved RGBA8
frames. The root remains `2b13a70a5d44659a6912c872a01adfd0e9eceaa02b668743cef82a0ea088a156`.
No expected or archived intermediate entered execution, canaries and caller
inputs remain unchanged, and allocations after warmup remain zero.

The controlled order `C1,U1,U2,C2,C3,U3` measured CLZ at
{bv['clz']['p50_evaluate_wall_ms']:.6f} ms p50 and uint32 at
{bv['u32']['p50_evaluate_wall_ms']:.6f} ms p50, a
{bv['u32_vs_clz_p50_speedup']:.6f}x improvement. Only the 13 HMMA metallibs
changed; 24 other runtime metallibs are byte-identical.

The 24-span profile attributes {pv['generic_l20_l51_gpu_speedup']:.6f}x to
L20-L51 and {pv['custom_l56_l83_gpu_speedup']:.6f}x to L56-L83. Summed GPU
spans improve {pv['u32_vs_clz_gpu_span_p50_speedup']:.6f}x and instrumented
evaluate improves {pv['u32_vs_clz_evaluate_p50_speedup']:.6f}x.

These are pinned 640x360 synthetic component measurements on M3 Max, not game
FPS, input latency, frame pacing, dynamic-resolution, subrect, or Netal claims.
""")
    (evidence/"failure-ledger.md").write_text("""# Preserved limits

1. Rejected FP32/SIMD alternatives remain in the parent uint32 package; only
   the bounded, CPU-differentially proven uint32 contract is promoted.
2. Raw benchmark runs retain a legacy profile label; explicit live mode,
   aggregate profile, source/metallib pins and exact roots identify execution.
3. Timing is synchronous component timing, not production/game performance.
4. Batched submission remains a separate, unaccepted experiment.
""")
    verifier=repo/"scripts/verify-fg-runtime-u32.py";common=["--repo",str(repo),"--u32",str(u32),
      "--clz",str(clz),"--rtx",str(rtx),"--benchmark",str(bench),"--profile",str(profile),
      "--provenance",str(evidence/"provenance.json")]
    subprocess.run(["python3",str(verifier),"verify",*common,"--output",str(evidence/"verification.json")],check=True)
    subprocess.run(["python3",str(verifier),"negative-audit",*common,"--output",str(evidence/"negative-audit.json")],check=True)
    names=sorted(p.name for p in evidence.iterdir() if p.is_file() and p.name!="ARTIFACTS.sha256")
    (evidence/"ARTIFACTS.sha256").write_text("".join(f"{sha(evidence/n)}  {n}\n" for n in names))
    print("evidence="+str(evidence));print("verification_sha256="+sha(evidence/"verification.json"));print("result=pass");return 0
if __name__=="__main__":raise SystemExit(main())
