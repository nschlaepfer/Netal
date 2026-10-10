#!/usr/bin/env python3
"""Freeze an archived-window diagnostic host using unchanged accepted kernels."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def canonical(r):
    return hashlib.sha256(json.dumps(r, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    bundle = repo / 'build/nr_encoder_fscale/bundle-v1'
    parent = json.loads((bundle / 'manifest.json').read_text())
    body = dict(parent)
    root = body.pop('receipt_root_sha256')
    assert canonical(body) == root == '3350d5f53e391a3be260c2d1d5b60065a9e23e24ba3d9f871f8bf9037ef2d633'
    for n, h in parent['files'].items():
        assert sha(bundle / n) == h, n
    assert not a.output.exists()
    a.output.mkdir(parents=True)
    source = a.output / 'source'
    source.mkdir()
    for n in ('mlp_probe.mm', 'profile_timing.h'):
        shutil.copyfile(bundle / 'encoder/source' / n, source / n)
    code = (bundle / 'encoder/source/probe.mm').read_text()
    assert code.count('Buffer Downsample(') == 1
    code = code.split('Buffer Downsample(', 1)[0]
    code += r'''
int main(int argc,const char**argv){@autoreleasepool{try{
  if(argc!=6)throw std::runtime_error("usage: probe LIB INPUT WEIGHT POISON OUTPUT");
  const auto in=Read(argv[2]),weights=Read(argv[3]);
  if(in.size()!=2048||weights.size()!=22720||Sha(weights.data(),weights.size())!="f0d05533694577fa313f2655b00ada84f231f87ebc66ebbb7e8f0f7ea1ac88d2")throw std::runtime_error("input/weight contract");
  unsigned poison=std::stoul(argv[4],nullptr,16);if(poison!=0xa5&&poison!=0x5a)throw std::runtime_error("poison contract");
  fs::path output(argv[5]);if(fs::exists(output))throw std::runtime_error("preserve diagnostic");
  Pipelines p;p.device=MTLCreateSystemDefaultDevice();p.queue=[p.device newCommandQueue];NSError*error=nil;
  auto lib=[p.device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&error];
  if(!lib)throw std::runtime_error("library failed");
  NSArray<NSString*>* names=@[@"nr_swin32_expand",@"nr_swin32_activate",@"nr_swin32_contract",@"nr_swin32_qkv",@"nr_swin32_normalize",@"nr_swin32_scores",@"nr_swin32_probabilities",@"nr_swin32_values",@"nr_swin32_project",@"nr_swin32_pack"];
  for(size_t i=0;i<10;++i){p.states[i]=[p.device newComputePipelineStateWithFunction:[lib newFunctionWithName:names[i]] error:&error];if(!p.states[i])throw std::runtime_error("pipeline failed");}
  auto x=Make(p.device,in.size(),poison),w=Make(p.device,weights.size(),poison);
  std::memcpy(Data(x),in.data(),in.size());std::memcpy(Data(w),weights.data(),weights.size());
  auto result=Block(p,x,w,64,poison,output);Write(output/"output.e4m3",result);
  std::ofstream f(output/"run.json");f<<"{\"schema\":\"neuralstudio.nr.c32-window-diagnostic.v1\",\"pid\":"<<getpid()<<",\"poison\":"<<poison
   <<",\"input_sha256\":\""<<Sha(in.data(),in.size())<<"\",\"weight_sha256\":\""<<Sha(weights.data(),weights.size())
   <<"\",\"output_sha256\":\""<<Sha(Data(result),result.bytes)<<"\",\"expected_file_opened\":false,\"archived_component_input\":true,\"full_model_inference\":false,\"coverage_errors\":0,\"canary_errors\":0,\"immutable_errors\":0,\"result\":\"pass_diagnostic_execution\"}\n";
  std::cout<<"output_sha256="<<Sha(Data(result),result.bytes)<<"\n";
  return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}}
'''
    target = source / 'probe.mm'
    target.write_text(code)
    cmd = ['xcrun', '-sdk', 'macosx', 'clang++', '-std=c++20', '-O2', '-fobjc-arc', '-Wall', '-Wextra', '-Werror', '-Wno-deprecated-declarations', str(target), '-framework', 'Foundation', '-framework', 'Metal', '-o', str(a.output / 'probe')]
    with (a.output / 'compile.log').open('x') as f:
        run = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    r = {'schema': 'neuralstudio.nr.c32-window-probe-build.v1', 'parent_bundle_root': root,
         'result': 'compiled' if run.returncode == 0 else 'failed', 'command': cmd, 'exit_code': run.returncode,
         'model_arithmetic_changed': False, 'role': 'archived component diagnostic, never live full-model acceptance',
         'files': {str(p.relative_to(a.output)): sha(p) for p in a.output.rglob('*') if p.is_file()}}
    r['receipt_root_sha256'] = canonical(r)
    (a.output / 'build.json').write_text(json.dumps(r, indent=2) + '\n')
    print(json.dumps(r, indent=2))
    if run.returncode:
        raise SystemExit(3)


if __name__ == '__main__':
    main()
