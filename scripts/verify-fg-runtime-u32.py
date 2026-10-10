#!/usr/bin/env python3
"""Verify the exact uint32 full-runtime promotion and measurements."""

from __future__ import annotations
import argparse,copy,hashlib,json,math,statistics
from pathlib import Path
from typing import Any,Callable

ROOT="2b13a70a5d44659a6912c872a01adfd0e9eceaa02b668743cef82a0ea088a156"
PARENT="e59c12715f9de63cfda8da6ff7773481adf930b0fb5b6f27f378de4e045cb8c9"
LIBS=("conv_3x3","conv_1x1","conv_pre","conv0","conv1","conv2","bot0","bot1",
      "block1_conv0","block1_conv1","block1_conv2","block1_bot0","block1_bot1")
LABELS=("status","L3","L4","L5-high","L5-low","L5-pack","L6","L7-texture-to-buffer",
        "L7-kernel","L7-buffer-to-texture","L8","L19","L20-L51","L52","L53","L54",
        "L55","L55-to-L84-blit","L56-L83","L84","L85","L86","L87","caller-output-copy")
DOMAINS=[0]*12+[1]*11+[2]

class Error(RuntimeError):pass
def req(v:bool,m:str)->None:
    if not v:raise Error(m)
def load(p:Path)->dict[str,Any]:return json.loads(p.read_text())
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def canon(v:Any)->str:return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",", ":")).encode()).hexdigest()
def close(a:float,b:float)->bool:return math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-9)
def pct(v:list[float],f:float)->float:
    o=sorted(v);p=(len(o)-1)*f;lo=math.floor(p);hi=math.ceil(p)
    return o[lo] if lo==hi else o[lo]+(o[hi]-o[lo])*(p-lo)
def outputs(repo:Path)->list[str]:return load(repo/"evidence/specs/fg-runtime-v1/live-alias/verification.json")["output_sha256"]

def correctness(args:argparse.Namespace,backend_override=None,promotion_override=None,runs_override=None)->dict[str,Any]:
    out=outputs(args.repo);backend=backend_override or load(args.u32/"backend-receipt.json")
    promotion=promotion_override or load(args.u32/"u32-promotion-receipt.json")
    req(backend.get("hmma_mode")=="u32" and backend.get("checkpoint_root_sha256")==ROOT and
        backend.get("output_sha256")==out and backend.get("fresh_process_count")==4 and
        backend.get("poison_bytes")==[90,165] and backend.get("auxiliary_mode")=="live" and
        backend.get("metal_resource_allocations_per_generated_frame")==[210]+[0]*14 and
        backend.get("expected_or_archived_intermediate_ingress") is False and backend.get("result")=="pass",
        "backend policy")
    req(promotion.get("schema")=="neuralstudio.fg.runtime-u32-promotion.v1" and
        promotion.get("hmma_mode")=="u32" and promotion.get("hmma_source_count")==13 and
        promotion.get("parent_u32_verification_root_sha256")==PARENT and
        promotion.get("checkpoint_root_sha256")==ROOT and promotion.get("output_sha256")==out and
        promotion.get("fresh_process_count")==4 and promotion.get("poison_bytes")==[90,165] and
        promotion.get("expected_or_archived_intermediate_ingress") is False and promotion.get("result")=="pass",
        "promotion policy")
    for rel,want in promotion["hmma_source_sha256"].items():req(sha(args.repo/rel)==want,"source pin "+rel)
    req(sha(args.repo/"src/fg_conv_simd_l20/hmma_exact_u32.metalh")==promotion["u32_source_sha256"] and
        sha(args.repo/"src/fg_conv_simd_l20/hmma_exact_u32_override.metalh")==promotion["override_source_sha256"],
        "uint32 helper pins")
    overlay=args.u32.parent/(args.u32.name+"-overlay")
    for name,want in promotion["compiled_overlay_sha256"].items():req(sha(overlay/name)==want,"overlay "+name)
    req(set(promotion["hmma_metallib_sha256"])==set(LIBS),"metallib coverage")
    for name,want in promotion["hmma_metallib_sha256"].items():
        req(sha(args.u32/"libs"/f"{name}.metallib")==want,"metallib "+name)
    dirs=sorted(p for p in args.u32.glob("fg-runtime-poison-*-process-*") if p.is_dir())
    rdirs=sorted(p for p in args.rtx.glob("fg-production-contract-l3-l87-poison-*-process-*") if p.is_dir())
    runs=runs_override or [load(p/"run-receipt.json") for p in dirs]
    rruns=[load(p/"temporal-receipt.json") for p in rdirs]
    req(len(dirs)==len(rdirs)==len(runs)==4,"process count")
    req(len({r.get("process_id") for r in runs})==4 and len({r.get("process_id") for r in rruns})==4,
        "fresh processes")
    req(sorted(r.get("poison_byte") for r in runs)==[90,90,165,165] and
        sorted(r.get("poison_byte") for r in rruns)==[90,90,165,165],"poison")
    for m,r in zip(runs,rruns):
        c=m.get("checkpoint_sha256")
        req(m.get("runtime_input_mismatched_bytes")==0 and m.get("runtime_canary_mismatched_bytes")==0 and
            m.get("resource_allocations_per_generated_frame")==[210]+[0]*14 and
            m.get("expected_or_archived_intermediate_ingress") is False and m.get("result")=="pass",
            "Mac integrity")
        req(isinstance(c,dict) and len(c)==1149 and canon(c)==ROOT and c==r.get("checkpoint_sha256") and
            m.get("output_sha256")==out and m.get("output_sha256")==r.get("final_output_sha256"),
            "RTX parity")
    md=args.u32/"fg-runtime-poison-a5-process-a";rd=args.rtx/"fg-production-contract-l3-l87-poison-a5-process-a"
    for i,want in enumerate(out,1):
        mb=(md/f"generated-{i}.rgba8").read_bytes();rb=(rd/f"generated-{i}/l20-l87/output.bin").read_bytes()
        req(mb==rb and len(mb)==921600 and hashlib.sha256(mb).hexdigest()==want,f"output {i}")
    result={"schema":"neuralstudio.fg.runtime-u32-correctness-verification.v1","checkpoint_root_sha256":ROOT,
            "output_sha256":out,"hmma_source_count":13,"fresh_processes_per_backend":4,
            "poison_bytes":[90,165],"retrieved_outputs_byte_exact":True,
            "expected_or_archived_intermediate_ingress":False,"result":"pass"}
    result["verification_root_sha256"]=canon(result);return result

def raw_policy(run:dict[str,Any],out:list[str],run_id:str,profile:bool)->None:
    req(run.get("run_id")==run_id and run.get("ex2_mode")=="compact" and
        run.get("auxiliary_mode")=="live" and run.get("poison_byte")==165 and
        run.get("profiling_enabled") is profile and run.get("counter_attachment") is False and
        run.get("pre_output_sha256")==out and run.get("measured_output_sha256")==out and
        run.get("post_output_sha256")==out and run.get("output_hashes_comparable") is True and
        run.get("measured_metal_resource_allocations")==[0]*15 and
        run.get("caller_owned_input_mismatches")==0 and
        run.get("expected_or_archived_intermediate_ingress") is False and
        run.get("address_fields_recorded") is False and run.get("result")=="pass","raw policy")

def benchmark(args:argparse.Namespace,override=None,runs_override=None)->dict[str,Any]:
    v=override or load(args.benchmark/"comparison.json");out=outputs(args.repo)
    req(v.get("schema")=="neuralstudio.fg.runtime-u32-benchmark-comparison.v1" and
        v.get("production_benchmark") is False and v.get("profiling_instrumentation") is False and
        v.get("execution_order")==["clz-1","u32-1","u32-2","clz-2","clz-3","u32-3"] and
        v.get("accepted_checkpoint_root_sha256")==ROOT and v.get("accepted_output_sha256")==out and
        v.get("expected_or_archived_intermediate_ingress") is False and v.get("result")=="pass",
        "benchmark policy")
    attempts={"clz":args.clz,"u32":args.u32};pids=[]
    for mode in ("clz","u32"):
        paths=sorted((args.benchmark/mode).glob("run-*.json"));runs=(runs_override or {}).get(mode) or [load(p) for p in paths]
        req(len(paths)==len(runs)==3,mode+" run count");vals=[]
        for i,r in enumerate(runs,1):raw_policy(r,out,f"fg-runtime-u32-{mode}-{i}",False);vals+=r["measured_evaluate_wall_ms"];pids.append(r["process_id"])
        block=v[mode];stats={"p50_evaluate_wall_ms":pct(vals,.5),"p95_evaluate_wall_ms":pct(vals,.95),
            "p99_evaluate_wall_ms":pct(vals,.99),"mean_evaluate_wall_ms":statistics.mean(vals),
            "stddev_evaluate_wall_ms":statistics.pstdev(vals),"max_evaluate_wall_ms":max(vals)}
        req(block.get("hmma_mode")==mode and block.get("process_ids")==[r["process_id"] for r in runs] and
            block.get("raw_receipt_sha256")==[sha(p) for p in paths] and block.get("evaluate_wall_ms")==vals,
            mode+" benchmark pins")
        for k,x in stats.items():req(close(block.get(k),x),mode+" "+k)
        for name in LIBS:req(block["hmma_metallib_sha256"][name]==sha(attempts[mode]/"libs"/f"{name}.metallib"),mode+" lib")
    req(len(set(pids))==6,"benchmark freshness")
    a={p.name:sha(p) for p in args.clz.joinpath("libs").glob("*.metallib")}
    b={p.name:sha(p) for p in args.u32.joinpath("libs").glob("*.metallib")}
    changed={n.removesuffix(".metallib") for n in a if a[n]!=b[n]}
    req(set(a)==set(b) and changed==set(LIBS),"only HMMA changed")
    req(close(v["u32_vs_clz_p50_speedup"],v["clz"]["p50_evaluate_wall_ms"]/v["u32"]["p50_evaluate_wall_ms"]),"speedup")
    n={k:x for k,x in v.items() if k not in ("clz","u32","normalized_root_sha256")}
    req(canon(n)==v.get("normalized_root_sha256"),"benchmark root")
    r={"schema":"neuralstudio.fg.runtime-u32-benchmark-verification.v1","comparison_sha256":sha(args.benchmark/"comparison.json"),
       "normalized_root_sha256":v["normalized_root_sha256"],"clz_p50_ms":v["clz"]["p50_evaluate_wall_ms"],
       "u32_p50_ms":v["u32"]["p50_evaluate_wall_ms"],"u32_vs_clz_p50_speedup":v["u32_vs_clz_p50_speedup"],
       "changed_metallib_count":13,"unchanged_metallib_count":len(a)-13,"production_benchmark":False,"result":"pass"}
    r["verification_root_sha256"]=canon(r);return r

def profile(args:argparse.Namespace,override=None,runs_override=None)->dict[str,Any]:
    v=override or load(args.profile/"profile.json");out=outputs(args.repo)
    req(v.get("schema")=="neuralstudio.fg.runtime-u32-boundary-profile.v1" and
        v.get("diagnostic_instrumentation") is True and v.get("production_benchmark") is False and
        v.get("command_spans_per_frame")==24 and v.get("selected_wait_boundaries")==23 and
        v.get("accepted_checkpoint_root_sha256")==ROOT and v.get("accepted_output_sha256")==out and
        v.get("expected_or_archived_intermediate_ingress") is False and v.get("result")=="pass","profile policy")
    pids=[]
    for mode in ("clz","u32"):
        paths=sorted((args.profile/mode).glob("run-*.json"));runs=(runs_override or {}).get(mode) or [load(p) for p in paths]
        req(len(paths)==len(runs)==3,mode+" profile count")
        cpu=[[] for _ in LABELS];gpu=[[] for _ in LABELS];walls=[];cs=[];gs=[];us=[]
        for i,r in enumerate(runs,1):
            raw_policy(r,out,f"fg-runtime-u32-profile-{mode}-{i}",True);pids.append(r["process_id"])
            for wall,spans in zip(r["measured_evaluate_wall_ms"],r["measured_command_spans"]):
                req(len(spans)==24 and [s["ordinal"] for s in spans]==list(range(24)) and [s["domain"] for s in spans]==DOMAINS,mode+" spans")
                c=sum(s["cpu_commit_wait_ms"] for s in spans);g=sum(s["gpu_ms"] for s in spans)
                walls.append(wall);cs.append(c);gs.append(g);us.append(wall-c)
                for j,s in enumerate(spans):cpu[j].append(s["cpu_commit_wait_ms"]);gpu[j].append(s["gpu_ms"])
        block=v[mode];req(block.get("process_ids")==[r["process_id"] for r in runs] and
            block.get("raw_receipt_sha256")==[sha(p) for p in paths],mode+" profile pins")
        ag={"evaluate_wall_p50_ms":statistics.median(walls),"command_commit_wait_sum_p50_ms":statistics.median(cs),
            "gpu_span_sum_p50_ms":statistics.median(gs),"unspanned_host_work_p50_ms":statistics.median(us)}
        for k,x in ag.items():req(close(block.get(k),x),mode+" "+k)
        for j,label in enumerate(LABELS):
            s=block["spans"][j];req(s.get("ordinal")==j and s.get("label")==label and s.get("domain")==DOMAINS[j] and
                close(s.get("cpu_commit_wait_p50_ms"),statistics.median(cpu[j])) and
                close(s.get("gpu_p50_ms"),statistics.median(gpu[j])) and close(s.get("gpu_max_ms"),max(gpu[j])),mode+f" span {j}")
    req(len(set(pids))==6,"profile freshness")
    calc={"u32_vs_clz_evaluate_p50_speedup":v["clz"]["evaluate_wall_p50_ms"]/v["u32"]["evaluate_wall_p50_ms"],
      "u32_vs_clz_gpu_span_p50_speedup":v["clz"]["gpu_span_sum_p50_ms"]/v["u32"]["gpu_span_sum_p50_ms"],
      "generic_l20_l51_gpu_speedup":v["clz"]["spans"][12]["gpu_p50_ms"]/v["u32"]["spans"][12]["gpu_p50_ms"],
      "custom_l56_l83_gpu_speedup":v["clz"]["spans"][18]["gpu_p50_ms"]/v["u32"]["spans"][18]["gpu_p50_ms"]}
    for k,x in calc.items():req(close(v.get(k),x),k)
    n={k:x for k,x in v.items() if k not in ("clz","u32","normalized_root_sha256")};req(canon(n)==v.get("normalized_root_sha256"),"profile root")
    r={"schema":"neuralstudio.fg.runtime-u32-profile-verification.v1","profile_sha256":sha(args.profile/"profile.json"),
       "normalized_root_sha256":v["normalized_root_sha256"],**calc,"production_benchmark":False,"result":"pass"}
    r["verification_root_sha256"]=canon(r);return r

def provenance(args:argparse.Namespace,override=None)->dict[str,Any]:
    req(args.provenance is not None,"provenance required");v=override or load(args.provenance)
    req(v.get("schema")=="neuralstudio.fg.runtime-u32-provenance.v1" and
        v.get("benchmark_sha256")==sha(args.benchmark/"comparison.json") and
        v.get("profile_sha256")==sha(args.profile/"profile.json") and
        v.get("rtx_archive_sha256")==sha(args.rtx.parent/(args.rtx.name+".tar.gz")) and
        v.get("expected_or_archived_intermediate_ingress") is False and v.get("result")=="pass","provenance")
    for rel,want in v.get("source_sha256",{}).items():req(sha(args.repo/rel)==want,"provenance source "+rel)
    for rel,want in v.get("build_artifact_sha256",{}).items():req(sha(args.repo/rel)==want,"provenance build "+rel)
    req(len(v.get("source_sha256",{}))>=12 and len(v.get("build_artifact_sha256",{}))>=20,"provenance coverage")
    r={"schema":"neuralstudio.fg.runtime-u32-provenance-verification.v1","source_pin_count":len(v["source_sha256"]),
       "build_pin_count":len(v["build_artifact_sha256"]),"result":"pass"};r["verification_root_sha256"]=canon(r);return r

def verify(args:argparse.Namespace)->dict[str,Any]:
    c=correctness(args);b=benchmark(args);p=profile(args);v=provenance(args) if args.provenance else None
    r={"schema":"neuralstudio.fg.runtime-u32-verification.v1","checkpoint_root_sha256":ROOT,
       "correctness_root_sha256":c["verification_root_sha256"],"benchmark_root_sha256":b["verification_root_sha256"],
       "profile_root_sha256":p["verification_root_sha256"],"provenance_root_sha256":v["verification_root_sha256"] if v else None,
       "clz_p50_ms":b["clz_p50_ms"],"u32_p50_ms":b["u32_p50_ms"],"u32_vs_clz_speedup":b["u32_vs_clz_p50_speedup"],
       "generic_l20_l51_gpu_speedup":p["generic_l20_l51_gpu_speedup"],"custom_l56_l83_gpu_speedup":p["custom_l56_l83_gpu_speedup"],
       "expected_or_archived_intermediate_ingress":False,
       "claims":{"netal_integration":False,"production_performance":False},"result":"pass"}
    r["verification_root_sha256"]=canon(r);return r

def negative(args:argparse.Namespace)->dict[str,Any]:
    rejected=[];backend=load(args.u32/"backend-receipt.json");promo=load(args.u32/"u32-promotion-receipt.json")
    runs=[load(p/"run-receipt.json") for p in sorted(q for q in args.u32.glob("fg-runtime-poison-*-process-*") if q.is_dir())]
    comp=load(args.benchmark/"comparison.json");br={m:[load(p) for p in sorted((args.benchmark/m).glob("run-*.json"))] for m in ("clz","u32")}
    prof=load(args.profile/"profile.json");pr={m:[load(p) for p in sorted((args.profile/m).glob("run-*.json"))] for m in ("clz","u32")}
    cases=[]
    def add(name,fn):cases.append((name,fn))
    def be():v=copy.deepcopy(backend);v["hmma_mode"]="clz";correctness(args,backend_override=v)
    def src():v=copy.deepcopy(promo);v["u32_source_sha256"]="0"*64;correctness(args,promotion_override=v)
    def chk():v=copy.deepcopy(runs);v[0]["checkpoint_sha256"][next(iter(v[0]["checkpoint_sha256"]))]="f"*64;correctness(args,runs_override=v)
    def proc():v=copy.deepcopy(runs);v[1]["process_id"]=v[0]["process_id"];correctness(args,runs_override=v)
    def bp():v=copy.deepcopy(comp);v["production_benchmark"]=True;benchmark(args,override=v)
    def bs():v=copy.deepcopy(comp);v["u32_vs_clz_p50_speedup"]=9.;benchmark(args,override=v)
    def bo():v=copy.deepcopy(br);v["u32"][0]["post_output_sha256"][0]="0"*64;benchmark(args,runs_override=v)
    def pp():v=copy.deepcopy(prof);v["production_benchmark"]=True;profile(args,override=v)
    def ps():v=copy.deepcopy(prof);v["u32"]["spans"][18]["gpu_p50_ms"]=1.;profile(args,override=v)
    def pi():v=copy.deepcopy(pr);v["clz"][0]["measured_command_spans"][0][12]["ordinal"]=99;profile(args,runs_override=v)
    add("backend-mode",be);add("u32-source",src);add("checkpoint",chk);add("process",proc)
    add("benchmark-production",bp);add("benchmark-speedup",bs);add("benchmark-output",bo)
    add("profile-production",pp);add("profile-span",ps);add("profile-identity",pi)
    if args.provenance:
        pv=load(args.provenance)
        def psr():v=copy.deepcopy(pv);v["source_sha256"][next(iter(v["source_sha256"]))]="0"*64;provenance(args,v)
        add("provenance-source",psr)
    for name,fn in cases:
        try:fn()
        except Error as e:rejected.append({"mutation":name,"observed_error":str(e)})
        else:raise Error("mutation accepted "+name)
    base=verify(args);return {"schema":"neuralstudio.fg.runtime-u32-negative-audit.v1",
      "baseline_verification_root_sha256":base["verification_root_sha256"],"mutations":rejected,
      "all_mutations_rejected":True,"result":"pass"}

def main()->int:
    p=argparse.ArgumentParser();p.add_argument("mode",choices=("verify","negative-audit"))
    for name in ("repo","u32","clz","rtx","benchmark","profile"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--provenance",type=Path);p.add_argument("--output",type=Path);a=p.parse_args()
    r=verify(a) if a.mode=="verify" else negative(a);text=json.dumps(r,indent=2,sort_keys=True)+"\n"
    if a.output:
        if a.output.exists():raise Error("preserve existing output")
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(text)
    print(text,end="");return 0
if __name__=="__main__":
    try:raise SystemExit(main())
    except (Error,OSError,json.JSONDecodeError) as e:raise SystemExit("verification_error="+str(e))
