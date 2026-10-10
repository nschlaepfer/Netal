#!/usr/bin/env python3
"""Build isolated exact Block2 GPU-initialization candidate from frozen zero-v3.

No shader, neural arithmetic, weight, coverage or output-ABI change. This only
builds; it does not execute the model or promote the candidate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]
PARENT='e4a60e4442cf88efb5cf09bfa631da81268ea86a39fce96c877ad030dacdc6aa'


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def canonical(data):return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def once(text,old,new):
    if text.count(old)!=1:raise ValueError('unique patch anchor required: '+old[:60])
    return text.replace(old,new,1)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise ValueError('preserve existing build')
    parent=ROOT/'build/nr_encoder_telemetry/zero-v3';r=json.loads((parent/'build.json').read_text());h=r.pop('receipt_root_sha256')
    if h!=PARENT or canonical(r)!=PARENT:raise ValueError('parent seal')
    pins={parent/'assets'/n:v for n,v in r['assets'].items()}
    pins.update({parent/'source'/n:v for n,v in r['source_files'].items()});pins[parent/'nr_encoder_runtime.o']=r['object_sha256']
    for path,want in pins.items():
        if sha(path)!=want:raise ValueError('parent bytes changed: '+str(path))
    a.out.mkdir(parents=True);shutil.copytree(parent/'assets',a.out/'assets');shutil.copytree(parent/'source',a.out/'source')
    common=a.out/'source/generated/common.inc';text=common.read_text()
    text=once(text,'size_t Canary(Buffer& b) {', '''// OPT-ENC-FILL-001: allocation/guards only. Caller MUST enqueue a full data
// fill before any GPU read; this helper is used only by the Block2 branch.
Buffer MakeGpuFillStorage(id<MTLDevice> dev,size_t bytes) {
  if(!bytes || bytes%4 || Guard%4 || bytes>SIZE_MAX-2*Guard)throw std::runtime_error("GPU fill extent/alignment");
  Buffer b{[dev newBufferWithLength:bytes+2*Guard options:MTLResourceStorageModeShared|MTLResourceHazardTrackingModeTracked],bytes};
  if(!b.metal || b.metal.hazardTrackingMode!=MTLHazardTrackingModeTracked)throw std::runtime_error("GPU fill allocation/tracking");
  std::memset(b.metal.contents,0xc3,Guard);std::memset(Data(b)+bytes,0xc3,Guard);
  return b;
}
size_t Canary(Buffer& b) {''')
    common.write_text(text)
    encoder=a.out/'source/generated/encoder.inc';text=encoder.read_text()
    text=once(text,'  NrPhaseClock phases(output.filename().string(),"allocation_and_poison");',
        '  const bool gpu_fill = output.filename()=="block2"; // graph identity, never frame/pixel/output data\n  NrPhaseClock phases(output.filename().string(),gpu_fill?"allocation_and_guards":"allocation_and_poison");')
    text=once(text,'for(size_t i=0;i<10;++i){y.push_back(Make(p.device,size_t(rows)*elements[i]*bytes[i],poison));counts.push_back(Make(p.device,size_t(rows)*elements[i]*4,0));}',
        'for(size_t i=0;i<10;++i){const size_t nb=size_t(rows)*elements[i]*bytes[i],nc=size_t(rows)*elements[i]*4; y.push_back(gpu_fill?MakeGpuFillStorage(p.device,nb):Make(p.device,nb,poison));counts.push_back(gpu_fill?MakeGpuFillStorage(p.device,nc):Make(p.device,nc,0));}')
    anchor='  for(size_t i=0;i<10;++i) {\n    auto encoder=[command computeCommandEncoder];'
    replacement='''  if(gpu_fill){
    phases.next("gpu_fill_encoding");
    auto fill=[command blitCommandEncoder];if(!fill)throw std::runtime_error("Block2 GPU fill encoder unavailable");
    for(size_t i=0;i<10;++i){
      [fill fillBuffer:y[i].metal range:NSMakeRange(Guard,y[i].bytes) value:poison];
      [fill fillBuffer:counts[i].metal range:NSMakeRange(Guard,counts[i].bytes) value:0];
    }
    [fill endEncoding];phases.next("encode");
  }
  for(size_t i=0;i<10;++i) {
    auto encoder=[command computeCommandEncoder];'''
    text=once(text,anchor,replacement);encoder.write_text(text)
    cmd=['xcrun','-sdk','macosx','clang++','-std=c++20','-O2','-fobjc-arc','-Wall','-Wextra','-Werror','-Wno-deprecated-declarations','-c',str(a.out/'source/nr_encoder_runtime.mm'),'-o',str(a.out/'nr_encoder_runtime.o')]
    with (a.out/'compile.log').open('x') as log:run=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    assets={n:sha(a.out/'assets'/n) for n in r['assets']}
    if assets!=r['assets']:raise ValueError('model assets changed')
    for path,want in pins.items():
        if sha(path)!=want:raise ValueError('parent drift after build: '+str(path))
    report={'schema':'neuralstudio.nr.block2-gpu-fill-build.v1','parent_root':PARENT,'candidate':'OPT-ENC-FILL-001',
        'result':'compiled_unvalidated_candidate' if run.returncode==0 else 'compile_failed','command':cmd,'exit_code':run.returncode,
        'assets':assets,'source_files':{str(p.relative_to(a.out/'source')):sha(p) for p in (a.out/'source').rglob('*') if p.is_file()},
        'model_assets_unchanged':True,'accepted_runtime_changed':False,'gpu_executed':False,'parity_accepted':False,
        'scope':'Block2 only: 20 buffer data spans filled before compute in the same tracked command buffer; guards and all validation unchanged',
        'timing_contract':'Block2 GPU timestamp includes fills; allocation_and_guards host phase excludes data initialization now; compare complete block/frame cost',
        'references':['https://developer.apple.com/documentation/metal/mtlblitcommandencoder/fill(buffer:range:value:)','https://developer.apple.com/documentation/metal/resource-synchronization'],
        'builder_sha256':sha(Path(__file__))}
    if run.returncode==0:report['object_sha256']=sha(a.out/'nr_encoder_runtime.o')
    report['receipt_root_sha256']=canonical(report)
    (a.out/'build.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('assets','source_files','command')},indent=2))
    return run.returncode


if __name__=='__main__':raise SystemExit(main())
