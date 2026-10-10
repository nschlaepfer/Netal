#!/usr/bin/env python3
"""Derive an isolated encoder telemetry object; all model assets stay unchanged."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def once(s,a,b):
    assert s.count(a)==1,'changed anchor '+a[:100]
    return s.replace(a,b)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--spatial',action='store_true');a=p.parse_args()
    repo=Path(__file__).resolve().parents[1];parent=repo/('build/nr_encoder_temporal/half-fma-spatial-v1' if a.spatial else 'build/nr_encoder_temporal/half-fma-v1')
    r=json.loads((parent/'build.json').read_text());b=dict(r);h=b.pop('receipt_root_sha256')
    assert canonical(b)==h==('9289bb7b21bd36571fffbdf057757fc6397baea3d9377c813965558e0430bfec' if a.spatial else '0a57b3bac8616c7103ae30a2df5a91296f3bda2ab9f84960725bb943df053b2e')
    for kind,folder in [('source_files','source'),('assets','assets')]:
        for n,digest in r[kind].items():assert sha(parent/folder/n)==digest
    assert not a.output.exists();a.output.mkdir(parents=True);src=a.output/'source';shutil.copytree(parent/'source',src);shutil.copytree(parent/'assets',a.output/'assets')
    shutil.copyfile(repo/'src/nr_encoder_runtime/nr_encoder_telemetry.h',src/'nr_encoder_telemetry.h')
    profile=src/'generated/profile_timing.h';text=profile.read_text()
    text=once(text,'struct NrTimingRecord {std::string label;double gpu_ms,submit_wait_ms,host_scope_ms;};','struct NrTimingRecord {std::string label;double gpu_ms,submit_wait_ms,host_scope_ms;double gpu_start_s=0,gpu_end_s=0;bool command=false;};')
    text=once(text,'gpu,NrElapsed(start),0});','gpu,NrElapsed(start),0,command.GPUStartTime,command.GPUEndTime,true});')
    text=once(text,'inline void NrDumpProfile(const fs::path& directory){','inline NSMutableDictionary* NrProfileDictionary(){')
    text=once(text,'@"host_scope_ms":@(r.host_scope_ms)}','@"host_scope_ms":@(r.host_scope_ms),@"command":@(r.command),@"gpu_start_s":@(r.gpu_start_s),@"gpu_end_s":@(r.gpu_end_s),@"gpu_timestamp_valid":@(!r.command||(r.gpu_start_s>0&&r.gpu_end_s>=r.gpu_start_s))}')
    text=once(text,'@"neuralstudio.nr.encoder-timing.v1"','@"neuralstudio.nr.encoder-timing.v2"')
    text=once(text,'  NSError*error=nil;NSData*data=[NSJSONSerialization dataWithJSONObject:result options:NSJSONWritingPrettyPrinted error:&error];', '''  return [result mutableCopy];
}
inline std::string NrProfileJson(){
  auto result=NrProfileDictionary();
  NSError*error=nil;NSData*data=[NSJSONSerialization dataWithJSONObject:result options:NSJSONWritingPrettyPrinted error:&error];''')
    old='  std::ofstream f(directory/"profile.json",std::ios::binary);f.write(static_cast<const char*>(data.bytes),static_cast<std::streamsize>(data.length));if(!f)throw std::runtime_error("profile write");'
    text=once(text,old,'''  return std::string(static_cast<const char*>(data.bytes),data.length);
}
inline void NrDumpProfile(const fs::path& directory){
  auto data=NrProfileJson();std::ofstream f(directory/"profile.json",std::ios::binary);f.write(data.data(),data.size());if(!f)throw std::runtime_error("profile write");''')
    profile.write_text(text)
    target=src/'nr_encoder_runtime.mm';text=target.read_text()
    text=once(text,'#include "nr_encoder_runtime.h"','#include "nr_encoder_runtime.h"\n#include "nr_encoder_telemetry.h"')
    text=once(text,'#include "generated/profile_timing.h"\n','')
    text=once(text,'struct NrEncoderState {','#include "generated/profile_timing.h"\n\nstruct NrEncoderState {\n  bool telemetry_enabled=false;uint64_t telemetry_generation=0;NSMutableDictionary* last_telemetry=nil;')
    text=once(text,'[e endEncoding];[c commit];[c waitUntilCompleted];','[e endEncoding];auto telemetry_start=NrClock::now();[c commit];[c waitUntilCompleted];const double telemetry_wait_ms=NrElapsed(telemetry_start);')
    text=once(text,'  if(Canary(b))throw std::runtime_error("input texture copy guard damage");','''  if(Canary(b))throw std::runtime_error("input texture copy guard damage");
  if(s.telemetry_enabled){double gpu=(c.GPUEndTime-c.GPUStartTime)*1000.0;
    nr_timings.push_back({"encoder.input_texture_blit",gpu,telemetry_wait_ms,0,c.GPUStartTime,c.GPUEndTime,true});}''')
    text=once(text,'    if(NrTemporalInput){','    ctx.state->last_telemetry=nil;\n    if(NrTemporalInput){')
    text=once(text,'accepted.code=0;result=std::move(accepted);return 0;','''accepted.code=0;
    if(ctx.state->telemetry_enabled){
      NSMutableDictionary* snapshot=NrProfileDictionary();
      if(!snapshot)throw std::runtime_error("telemetry snapshot construction failed");
      snapshot[@"input_sha256"]=[NSString stringWithUTF8String:pre.input_sha.c_str()];snapshot[@"frame_index"]=@(in.frame_index);snapshot[@"reset"]=@(in.reset);
      snapshot[@"generation"]=@(ctx.state->telemetry_generation+1);snapshot[@"tensor_diagnostics_enabled"]=@(diagnostics!=nullptr);
      snapshot[@"guards_unchanged"]=@YES;snapshot[@"input_blits_included"]=@YES;snapshot[@"scope"]=@"Instrumented checked encoder; host scopes overlap phases and GPU waits. No resident or quiet-window speed claim.";
      snapshot[@"serialization_in_copy_accessor"]=@YES;
      ctx.state->last_telemetry=snapshot;
    }
    ++ctx.state->telemetry_generation;result=std::move(accepted);return 0;''')
    text+='''
int nr_encoder_enable_telemetry(NrEncoderContext& ctx,bool enabled,std::string& error){
  try{std::unique_lock<std::mutex> lease(encoder_execution_mutex,std::try_to_lock);
    if(!lease.owns_lock()||!ctx.state||ctx.state->device!=ctx.device||ctx.state->queue!=ctx.queue)throw std::runtime_error("telemetry: unavailable or changed context");
    ctx.state->telemetry_enabled=enabled;ctx.state->last_telemetry=nil;error.clear();return 0;
  }catch(const std::exception&e){error=e.what();return 2;}}
int nr_encoder_copy_telemetry(const NrEncoderContext& ctx,std::string& json,std::string& error){
  json.clear();try{std::unique_lock<std::mutex> lease(encoder_execution_mutex,std::try_to_lock);
    if(!lease.owns_lock()||!ctx.state||ctx.state->device!=ctx.device||ctx.state->queue!=ctx.queue||!ctx.state->telemetry_enabled||!ctx.state->last_telemetry)throw std::runtime_error("telemetry: no successful snapshot for this context");
    NSError* issue=nil;NSData* encoded=[NSJSONSerialization dataWithJSONObject:ctx.state->last_telemetry options:0 error:&issue];
    if(!encoded)throw std::runtime_error("telemetry JSON encode failed");json.assign(static_cast<const char*>(encoded.bytes),encoded.length);error.clear();return 0;
  }catch(const std::exception&e){error=e.what();return 2;}}
'''
    target.write_text(text)
    cmd=['xcrun','-sdk','macosx','clang++','-std=c++20','-O2','-fobjc-arc','-Wall','-Wextra','-Werror','-Wno-deprecated-declarations','-c',str(target),'-o',str(a.output/'nr_encoder_runtime.o')]
    with (a.output/'compile.log').open('x') as log:run=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    assets={n:sha(a.output/'assets'/n) for n in r['assets']};assert assets==r['assets'],'model asset changed'
    result={'schema':'neuralstudio.nr.encoder-telemetry-build.v1','parent_root':h,'assets':assets,'source_files':{str(p.relative_to(src)):sha(p) for p in src.rglob('*') if p.is_file()},
            'command':cmd,'exit_code':run.returncode,'result':'compiled_candidate' if run.returncode==0 else 'compile_failed',
            'model_assets_unchanged':True,'existing_result_ABI_unchanged':True,'serialization_in_copy_accessor':True,'accepted':False,'production_changed':False,'generator_sha256':sha(Path(__file__))}
    if run.returncode==0:result['object_sha256']=sha(a.output/'nr_encoder_runtime.o')
    result['receipt_root_sha256']=canonical(result);(a.output/'build.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('assets','source_files','command')},indent=2))
    if run.returncode:raise SystemExit(3)


if __name__=='__main__':main()
