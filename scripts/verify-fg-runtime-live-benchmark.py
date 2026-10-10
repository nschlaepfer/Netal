#!/usr/bin/env python3
"""Verify the no-refill live-alias runtime timing diagnostic."""

from __future__ import annotations
import argparse,copy,hashlib,json,math,statistics
from pathlib import Path
from typing import Any,Callable

class VerificationError(RuntimeError):pass
def require(v:bool,m:str)->None:
    if not v:raise VerificationError(m)
def load(p:Path)->dict[str,Any]:return json.loads(p.read_text())
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(v:Any)->str:return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",", ":")).encode()).hexdigest()
def close(a:float,b:float)->bool:return math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-9)
def percentile(values:list[float],f:float)->float:
    o=sorted(values);p=(len(o)-1)*f;lo=math.floor(p);hi=math.ceil(p)
    return o[lo] if lo==hi else o[lo]+(o[hi]-o[lo])*(p-lo)

def validate_run(run:dict[str,Any],expected:list[str])->None:
    require(run.get("schema")=="neuralstudio.fg.runtime-benchmark-run.v1" and
            run.get("ex2_mode")=="compact" and run.get("auxiliary_mode")=="live" and
            run.get("poison_byte")==0xA5 and
            run.get("timing_class")=="synchronous-evaluate-wall-not-input-latency" and
            run.get("profiling_enabled") is False and run.get("counter_attachment") is False and
            run.get("warmup_generated_frames")==15 and run.get("measured_generated_frames")==15 and
            run.get("postcheck_generated_frames")==15 and
            len(run.get("measured_evaluate_wall_ms",[]))==15 and
            run.get("measured_command_spans")==[[] for _ in range(15)] and
            run.get("pre_output_sha256")==expected and run.get("measured_output_sha256")==expected and
            run.get("post_output_sha256")==expected and run.get("output_hashes_comparable") is True and
            run.get("measured_metal_resource_allocations")==[0]*15 and
            run.get("caller_owned_input_mismatches")==0 and
            run.get("expected_or_archived_intermediate_ingress") is False and
            run.get("address_fields_recorded") is False and run.get("result")=="pass","raw run")

def validate(attempt:Path,reference:Path,override:dict[str,Any]|None=None,
             run_override:list[dict[str,Any]]|None=None)->dict[str,Any]:
    receipt=override or load(attempt/"benchmark.json");expected=load(reference)["output_sha256"]
    paths=sorted(attempt.glob("run-*.json"));runs=run_override or [load(p) for p in paths]
    require(len(paths)==3 and len(runs)==3,"raw run count")
    require(len({r.get("process_id") for r in runs})==3 and
            [r.get("run_id") for r in runs]==[f"fg-runtime-live-hotpath-{i}" for i in range(1,4)],
            "fresh processes")
    values=[]
    for run in runs:
        validate_run(run,expected)
        values.extend(run["measured_evaluate_wall_ms"])
    require(receipt.get("schema")=="neuralstudio.fg.runtime-live-hotpath-benchmark.v1" and
            receipt.get("diagnostic_component_only") is True and receipt.get("production_benchmark") is False and
            receipt.get("timing_class")=="synchronous-evaluate-wall-not-input-latency" and
            receipt.get("profile")=="640x360-compact-live-early-aliases" and
            receipt.get("profiling_instrumentation") is False and
            receipt.get("untimed_correctness_before_and_after") is True and
            receipt.get("output_hashes_comparable") is True and
            receipt.get("independent_process_count")==3 and receipt.get("measured_frame_count")==45 and
            receipt.get("accepted_output_sha256")==expected and receipt.get("result")=="pass","aggregate policy")
    require(receipt.get("raw_receipt_sha256")==[sha(p) for p in paths] and
            receipt.get("evaluate_wall_ms")==values,"raw provenance")
    stats={"p50_evaluate_wall_ms":percentile(values,.5),"p95_evaluate_wall_ms":percentile(values,.95),
           "p99_evaluate_wall_ms":percentile(values,.99),"max_evaluate_wall_ms":max(values),
           "mean_evaluate_wall_ms":statistics.mean(values),"stddev_evaluate_wall_ms":statistics.pstdev(values)}
    for k,v in stats.items():require(close(receipt.get(k),v),k)
    require(close(receipt.get("diagnostic_ratio_vs_profiled_reference"),
                  receipt["profiled_reference_p50_ms"]/stats["p50_evaluate_wall_ms"]),"ratio")
    normalized={k:v for k,v in receipt.items() if k not in ("evaluate_wall_ms","raw_receipt_sha256","normalized_root_sha256")}
    require(canonical(normalized)==receipt.get("normalized_root_sha256"),"normalized root")
    result={"schema":"neuralstudio.fg.runtime-live-hotpath-benchmark-verification.v1",
            "benchmark_sha256":sha(attempt/"benchmark.json"),"normalized_root_sha256":receipt["normalized_root_sha256"],
            **stats,"diagnostic_ratio_vs_profiled_reference":receipt["diagnostic_ratio_vs_profiled_reference"],
            "independent_process_count":3,"measured_frame_count":45,"output_hashes_comparable":True,
            "production_benchmark":False,"result":"pass"}
    result["verification_root_sha256"]=canonical(result);return result

def validate_provenance(attempt:Path,provenance_path:Path,repo:Path,
                        override:dict[str,Any]|None=None)->dict[str,Any]:
    provenance=override or load(provenance_path)
    benchmark=load(attempt/"benchmark.json")
    require(provenance.get("schema")=="neuralstudio.fg.runtime-live-hotpath-provenance.v1" and
            provenance.get("benchmark_attempt")=="build/fg_runtime/benchmark/live-hotpath-v1" and
            provenance.get("benchmark_sha256")==sha(attempt/"benchmark.json") and
            provenance.get("binary_sha256")==benchmark.get("binary_sha256") and
            provenance.get("runtime_tail_sha256")==benchmark.get("runtime_tail_sha256") and
            provenance.get("expected_or_archived_intermediate_ingress") is False and
            provenance.get("address_fields_recorded") is False and provenance.get("result")=="pass",
            "provenance policy")
    for relative,expected in provenance.get("source_sha256",{}).items():
        path=repo/relative
        require(path.is_file() and sha(path)==expected,"source pin "+relative)
    require(len(provenance.get("source_sha256",{}))>=10,"source pin coverage")
    artifacts=provenance.get("build_artifact_sha256",{})
    for relative,expected in artifacts.items():
        path=repo/relative
        require(path.is_file() and sha(path)==expected,"build pin "+relative)
    require(len(artifacts)>=6,"build pin coverage")
    result={"schema":"neuralstudio.fg.runtime-live-hotpath-provenance-verification.v1",
            "benchmark_sha256":provenance["benchmark_sha256"],
            "source_pin_count":len(provenance["source_sha256"]),
            "build_pin_count":len(artifacts),"result":"pass"}
    result["verification_root_sha256"]=canonical(result)
    return result

def negative(attempt:Path,reference:Path,provenance_path:Path|None=None,repo:Path|None=None)->dict[str,Any]:
    original=load(attempt/"benchmark.json")
    muts:dict[str,Callable[[dict[str,Any]],None]]={
      "production":lambda x:x.__setitem__("production_benchmark",True),
      "output":lambda x:x["accepted_output_sha256"].__setitem__(0,"0"*64),
      "raw":lambda x:x["raw_receipt_sha256"].__setitem__(1,"f"*64),
      "p50":lambda x:x.__setitem__("p50_evaluate_wall_ms",1.0),
      "ratio":lambda x:x.__setitem__("diagnostic_ratio_vs_profiled_reference",1.0),
      "root":lambda x:x.__setitem__("normalized_root_sha256","0"*64)}
    rejected=[]
    for name,mutate in muts.items():
        value=copy.deepcopy(original);mutate(value)
        try:validate(attempt,reference,value)
        except VerificationError as e:rejected.append({"mutation":name,"observed_error":str(e)})
        else:raise VerificationError("mutation accepted: "+name)
    original_runs=[load(p) for p in sorted(attempt.glob("run-*.json"))]
    run_muts:dict[str,Callable[[list[dict[str,Any]]],None]]={
      "process_reuse":lambda x:x[1].__setitem__("process_id",x[0]["process_id"]),
      "freshness_id":lambda x:x[2].__setitem__("run_id",x[1]["run_id"]),
      "poison":lambda x:x[0].__setitem__("poison_byte",0x5A),
      "bracket_output":lambda x:x[0]["post_output_sha256"].__setitem__(0,"0"*64),
      "caller_immutability":lambda x:x[1].__setitem__("caller_owned_input_mismatches",1),
      "archive_ingress":lambda x:x[1].__setitem__("expected_or_archived_intermediate_ingress",True),
      "measured_allocation":lambda x:x[2]["measured_metal_resource_allocations"].__setitem__(3,1),
      "profiling":lambda x:x[2].__setitem__("profiling_enabled",True)}
    for name,mutate in run_muts.items():
        value=copy.deepcopy(original_runs);mutate(value)
        try:validate(attempt,reference,run_override=value)
        except VerificationError as e:rejected.append({"mutation":name,"observed_error":str(e)})
        else:raise VerificationError("mutation accepted: "+name)
    if provenance_path is not None:
        require(repo is not None,"repo required for provenance negative")
        provenance=load(provenance_path)
        first=next(iter(provenance["source_sha256"]))
        mutated=copy.deepcopy(provenance);mutated["source_sha256"][first]="0"*64
        try:validate_provenance(attempt,provenance_path,repo,mutated)
        except VerificationError as e:rejected.append({"mutation":"source_pin","observed_error":str(e)})
        else:raise VerificationError("mutation accepted: source_pin")
    baseline=validate(attempt,reference)
    return {"schema":"neuralstudio.fg.runtime-live-hotpath-benchmark-negative-audit.v1",
            "baseline_verification_root_sha256":baseline["verification_root_sha256"],
            "mutations":rejected,"all_mutations_rejected":True,"result":"pass"}

def main()->int:
    p=argparse.ArgumentParser();p.add_argument("mode",choices=("verify","verify-provenance","negative-audit"));p.add_argument("--attempt",type=Path,required=True);p.add_argument("--reference",type=Path);p.add_argument("--provenance",type=Path);p.add_argument("--repo",type=Path);p.add_argument("--output",type=Path);a=p.parse_args()
    if a.mode=="verify":
        require(a.reference is not None,"reference required");r=validate(a.attempt,a.reference)
    elif a.mode=="verify-provenance":
        require(a.provenance is not None and a.repo is not None,"provenance and repo required");r=validate_provenance(a.attempt,a.provenance,a.repo)
    else:
        require(a.reference is not None,"reference required");r=negative(a.attempt,a.reference,a.provenance,a.repo)
    text=json.dumps(r,indent=2,sort_keys=True)+"\n"
    if a.output:
        if a.output.exists():raise VerificationError("preserve existing output")
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(text)
    print(text,end="");return 0
if __name__=="__main__":
    try:raise SystemExit(main())
    except (VerificationError,OSError,json.JSONDecodeError) as e:raise SystemExit("verification_error="+str(e))
