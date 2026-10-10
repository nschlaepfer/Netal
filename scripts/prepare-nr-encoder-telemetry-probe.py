#!/usr/bin/env python3
"""Extend the accepted producer probe with output-only telemetry API checks."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def once(s,a,b):
    assert s.count(a)==1,a[:100]
    return s.replace(a,b)


def main():
    p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    repo=Path(__file__).resolve().parents[1];runtime=a.runtime.resolve();r=json.loads((runtime/'build.json').read_text());b=dict(r);h=b.pop('receipt_root_sha256');assert canonical(b)==h and r['exit_code']==0
    for n,d in r['source_files'].items():assert sha(runtime/'source'/n)==d
    assert sha(runtime/'nr_encoder_runtime.o')==r['object_sha256']
    assert not a.output.exists();a.output.mkdir(parents=True);src=a.output/'source';src.mkdir()
    for n in ('nr_encoder_runtime.h','nr_encoder_telemetry.h'):shutil.copyfile(runtime/'source'/n,src/n)
    text=(repo/'src/nr_encoder_runtime/nr_encoder_runtime_probe.mm').read_text()
    text=once(text,'#include "nr_encoder_runtime.h"','#include "nr_encoder_runtime.h"\n#include "nr_encoder_telemetry.h"')
    text=text.replace('--profile','--telemetry')
    text=once(text,'  NSMutableArray* frames=', '''  std::string telemetry,error2;NrEncoderContext absent;
  if(nr_encoder_enable_telemetry(absent,true,error2)!=2||nr_encoder_copy_telemetry(absent,telemetry,error2)!=2)throw std::runtime_error("unprepared telemetry admitted");
  if(nr_encoder_copy_telemetry(ctx,telemetry,error2)!=2||!telemetry.empty())throw std::runtime_error("default telemetry not disabled");
  if(nr_encoder_enable_telemetry(ctx,profile,error2))throw std::runtime_error(error2);
  if(nr_encoder_copy_telemetry(ctx,telemetry,error2)!=2)throw std::runtime_error("telemetry available before success");
  NSMutableArray* frames=''')
    text=once(text,'    [negative addObject:', '    if(nr_encoder_copy_telemetry(ctx,telemetry,error2)!=2||!telemetry.empty())throw std::runtime_error("stale telemetry after rejected frame");\n    [negative addObject:')
    text=once(text,'    const fs::path diagnostics=out/("diagnostics-frame"+std::to_string(i));\n','')
    text=once(text,'profile?diagnostics.c_str():nullptr','nullptr')
    anchor='    // Expected artifacts are opened ONLY after native execution has returned.'
    text=once(text,anchor,'''    if(profile){
      if(nr_encoder_copy_telemetry(ctx,telemetry,error2))throw std::runtime_error(error2);
      NSData* payload=[NSData dataWithBytes:telemetry.data() length:telemetry.size()];NSError* err=nil;
      NSDictionary* observed=[NSJSONSerialization JSONObjectWithData:payload options:0 error:&err];
      if(!observed||[observed[@"generation"] unsignedLongLongValue]!=i+1||[observed[@"frame_index"] unsignedIntValue]!=0||![observed[@"reset"] boolValue]||[observed[@"tensor_diagnostics_enabled"] boolValue]||![observed[@"input_sha256"] isEqualToString:S(imageSha)])throw std::runtime_error("telemetry frame identity");
      std::ofstream f(out/("telemetry-frame"+std::to_string(i)+".json"));f<<telemetry;if(!f)throw std::runtime_error("telemetry write");
      NrEncoderContext other;if(nr_encoder_copy_telemetry(other,telemetry,error2)!=2||!telemetry.empty())throw std::runtime_error("cross-context telemetry leak");
    }else if(nr_encoder_copy_telemetry(ctx,telemetry,error2)!=2||!telemetry.empty())throw std::runtime_error("disabled telemetry leaked");
'''+anchor)
    text=once(text,'    if(mismatches)return 3;','''    if(mismatches)return 3;
    if(i==0&&profile){NrEncoderFrame bad;bad.color=texture;bad.reset=false;NrEncoderResult rejected;
      if(nr_encoder_run_frame(ctx,bad,current_poison,rejected)!=2||nr_encoder_copy_telemetry(ctx,telemetry,error2)!=2||!telemetry.empty())throw std::runtime_error("failed frame retained telemetry");
      if(nr_encoder_enable_telemetry(ctx,false,error2)||nr_encoder_copy_telemetry(ctx,telemetry,error2)!=2)throw std::runtime_error("disable telemetry failed");
      if(nr_encoder_enable_telemetry(ctx,true,error2)||nr_encoder_copy_telemetry(ctx,telemetry,error2)!=2)throw std::runtime_error("enable telemetry retained stale result");
    }''')
    target=src/'probe.mm';target.write_text(text)
    cmd=['xcrun','-sdk','macosx','clang++','-std=c++20','-O2','-fobjc-arc','-Wall','-Wextra','-Werror','-Wno-deprecated-declarations',str(target),str(runtime/'nr_encoder_runtime.o'),'-framework','Foundation','-framework','Metal','-o',str(a.output/'probe')]
    with (a.output/'compile.log').open('x') as log:rc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT).returncode
    result={'schema':'neuralstudio.nr.encoder-telemetry-probe-build.v1','runtime_root':h,'runtime_build':str(runtime),'runtime_object_sha256':r['object_sha256'],'command':cmd,'exit_code':rc,'files':{str(p.relative_to(a.output)):sha(p) for p in a.output.rglob('*') if p.is_file()},'generator_sha256':sha(Path(__file__))}
    result['receipt_root_sha256']=canonical(result);(a.output/'build.json').write_text(json.dumps(result,indent=2)+'\n');print(result['receipt_root_sha256'])
    if rc:raise SystemExit(3)


if __name__=='__main__':main()
