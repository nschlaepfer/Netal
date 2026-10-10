#!/usr/bin/env python3
"""Verify the timing-perturbed FG runtime command-boundary profile."""

from __future__ import annotations

import argparse, copy, hashlib, json, math, statistics
from pathlib import Path
from typing import Any, Callable

LABELS = [
    "status", "L3", "L4", "L5-high", "L5-low", "L5-pack", "L6",
    "L7-texture-to-buffer", "L7-kernel", "L7-buffer-to-texture", "L8",
    "L19", "L20-L51", "L52", "L53", "L54", "L55", "L55-to-L84-blit",
    "L56-L83", "L84", "L85", "L86", "L87", "caller-output-copy",
]
DOMAINS = [0] * 12 + [1] * 11 + [2]


class VerificationError(RuntimeError): pass


def require(value: bool, message: str) -> None:
    if not value: raise VerificationError(message)


def load(path: Path) -> dict[str, Any]: return json.loads(path.read_text())
def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",", ":")).encode()).hexdigest()
def close(a: float,b: float) -> bool: return math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-9)


def validate(attempt: Path, reference: Path,
             override: dict[str,Any]|None=None)->dict[str,Any]:
    profile=override or load(attempt/"profile.json")
    expected=load(reference)["final_output_sha256"]
    paths=sorted(attempt.glob("run-*.json"));runs=[load(path) for path in paths]
    require(len(runs)==3 and len({r["process_id"] for r in runs})==3,"fresh profile processes")
    require(profile.get("schema")=="neuralstudio.fg.runtime-boundary-profile.v1" and
            profile.get("diagnostic_instrumentation") is True and
            profile.get("production_benchmark") is False and
            profile.get("untimed_correctness_before_and_after") is True and
            profile.get("output_hashes_comparable") is True and
            profile.get("command_spans_per_frame")==24 and
            profile.get("static_selected_wait_boundaries")==23 and
            profile.get("additional_caller_output_copy_boundary")==1 and
            profile.get("result")=="pass","profile policy")
    require(profile.get("raw_receipt_sha256")==[sha(path) for path in paths],"raw profile pins")
    cpu=[[] for _ in LABELS];gpu=[[] for _ in LABELS];walls=[];cpu_sums=[];gpu_sums=[];unattributed=[]
    for run in runs:
        require(run.get("profiling_enabled") is True and run.get("result")=="pass" and
                run.get("pre_output_sha256")==expected and
                run.get("measured_output_sha256")==expected and
                run.get("post_output_sha256")==expected and
                run.get("output_hashes_comparable") is True and
                run.get("measured_metal_resource_allocations")==[0]*15,"profile run")
        for wall,spans in zip(run["measured_evaluate_wall_ms"],run["measured_command_spans"]):
            require(len(spans)==24 and [s["domain"] for s in spans]==DOMAINS and
                    [s["ordinal"] for s in spans]==list(range(24)),"profile span identity")
            walls.append(wall);cs=sum(s["cpu_commit_wait_ms"] for s in spans);gs=sum(s["gpu_ms"] for s in spans)
            cpu_sums.append(cs);gpu_sums.append(gs);unattributed.append(wall-cs)
            for i,span in enumerate(spans):cpu[i].append(span["cpu_commit_wait_ms"]);gpu[i].append(span["gpu_ms"])
    require(len(walls)==45 and profile.get("measured_frame_count")==45,"profile frame coverage")
    expected_scalar={"evaluate_wall_p50_ms":statistics.median(walls),
                     "command_commit_wait_sum_p50_ms":statistics.median(cpu_sums),
                     "gpu_span_sum_p50_ms":statistics.median(gpu_sums),
                     "unattributed_host_work_p50_ms":statistics.median(unattributed)}
    for key,value in expected_scalar.items():require(close(profile.get(key),value),key)
    spans=profile.get("spans");require(isinstance(spans,list) and len(spans)==24,"profile span summaries")
    for i,span in enumerate(spans):
        require(span.get("ordinal")==i and span.get("label")==LABELS[i] and span.get("domain")==DOMAINS[i] and
                close(span.get("cpu_commit_wait_p50_ms"),statistics.median(cpu[i])) and
                close(span.get("gpu_p50_ms"),statistics.median(gpu[i])),f"span summary {i}")
    normalized={k:v for k,v in profile.items() if k not in ("raw_receipt_sha256","normalized_root_sha256")}
    require(canonical(normalized)==profile.get("normalized_root_sha256"),"profile normalized root")
    result={"schema":"neuralstudio.fg.runtime-profile-verification.v1",
            "profile_sha256":sha(attempt/"profile.json"),
            "normalized_root_sha256":profile["normalized_root_sha256"],
            **expected_scalar,"command_spans_per_frame":24,
            "dominant_gpu_span":"L56-L83","output_hashes_comparable":True,
            "production_benchmark":False,"result":"pass"}
    result["verification_root_sha256"]=canonical(result);return result


def negative(attempt:Path,reference:Path)->dict[str,Any]:
    original=load(attempt/"profile.json")
    mutations:dict[str,Callable[[dict[str,Any]],None]]={
      "production":lambda x:x.__setitem__("production_benchmark",True),
      "span_count":lambda x:x.__setitem__("command_spans_per_frame",23),
      "raw_pin":lambda x:x["raw_receipt_sha256"].__setitem__(0,"0"*64),
      "wall":lambda x:x.__setitem__("evaluate_wall_p50_ms",1.0),
      "span_label":lambda x:x["spans"][18].__setitem__("label","wrong"),
      "root":lambda x:x.__setitem__("normalized_root_sha256","f"*64)}
    rejected=[]
    for name,mutate in mutations.items():
        changed=copy.deepcopy(original);mutate(changed)
        try:validate(attempt,reference,changed)
        except VerificationError as error:rejected.append({"mutation":name,"observed_error":str(error)})
        else:raise VerificationError("mutation accepted: "+name)
    baseline=validate(attempt,reference)
    return {"schema":"neuralstudio.fg.runtime-profile-negative-audit.v1",
            "baseline_verification_root_sha256":baseline["verification_root_sha256"],
            "mutations":rejected,"all_mutations_rejected":True,"result":"pass"}


def main()->int:
    parser=argparse.ArgumentParser();parser.add_argument("mode",choices=("verify","negative-audit"));parser.add_argument("--attempt",type=Path,required=True);parser.add_argument("--reference",type=Path,required=True);parser.add_argument("--output",type=Path);a=parser.parse_args()
    result=validate(a.attempt,a.reference) if a.mode=="verify" else negative(a.attempt,a.reference)
    text=json.dumps(result,indent=2,sort_keys=True)+"\n"
    if a.output:
        if a.output.exists():raise VerificationError("preserve existing output")
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(text)
    print(text,end="");return 0

if __name__=="__main__":
    try:raise SystemExit(main())
    except (VerificationError,OSError,json.JSONDecodeError) as error:raise SystemExit("verification_error="+str(error))
