#!/usr/bin/env python3
"""Extract in-memory host entrypoints from the sealed, tested encoder bundle.

Never edit that bundle or shaders. All transformations are recorded and fail
closed on changed source anchors; generated code is retained for inspection.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from nr_candidate_primitive import validate_primitive

ROOT='3350d5f53e391a3be260c2d1d5b60065a9e23e24ba3d9f871f8bf9037ef2d633'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def once(s,old,new):
    if s.count(old)!=1:raise ValueError('source anchor count: '+old[:100])
    return s.replace(old,new)
def between(s,start,end,replacement):
    if s.count(start)!=1 or s.count(end)!=1:raise ValueError('source range anchors')
    a=s.index(start);b=s.index(end,a);return s[:a]+replacement+s[b:]
def diagnostic(s):
    return s.replace('std::ofstream','NrDiagnosticStream').replace('fs::create_directories(output);','if(NrEmitDiagnostics)fs::create_directories(output);').replace('fs::create_directories(out);','if(NrEmitDiagnostics)fs::create_directories(out);')
def quiet_progress(s,expected):
    if s.count('std::cout')!=expected:raise ValueError('progress call-site inventory changed')
    return s.replace('std::cout','if(NrEmitDiagnostics)std::cout')
def reuse_block0_scratch(s):
    """Reuse only synchronous chunk-local buffers; re-poison every logical use."""
    start='  for(uint32_t base=0;base<Windows;base+=Chunk){@autoreleasepool{'
    s=once(s,start,'''  // Frame-local scratch never escapes to a returned region. The previous
  // chunk has completed and passed all coverage/canary checks before reuse.
  std::array<Buffer,22> scratch{};size_t scratch_allocations=0,scratch_reuses=0;
'''+start+'''
    size_t scratch_slot=0;
    auto ScratchMake=[&](id<MTLDevice> dev,size_t bytes,uint8_t fill){
      if(scratch_slot>=scratch.size())throw std::runtime_error("scratch slot overflow");
      auto& b=scratch[scratch_slot++];
      if(b.metal&&Canary(b))throw std::runtime_error("scratch guard damaged before reuse");
      if(!b.metal||b.bytes!=bytes){b=Make(dev,bytes,fill);++scratch_allocations;}
      else{std::memset(Data(b),fill,bytes);++scratch_reuses;}
      return b;
    };
''')
    old='''    Buffer raw=Make(device,size_t(rows)*64,poison),packed=Make(device,size_t(rows)*32,poison),rc=Make(device,size_t(rows)*128,0),pc=Make(device,size_t(rows)*128,0);
    std::vector<Buffer> y,c;for(size_t i=0;i<9;++i){y.push_back(Make(device,size_t(rows)*elements[i]*sizes[i],poison));c.push_back(Make(device,size_t(rows)*elements[i]*4,0));}'''
    s=once(s,old,old.replace('Make(device,','ScratchMake(device,')+'\n    if(scratch_slot!=scratch.size())throw std::runtime_error("scratch slot underflow");')
    end='  GuardOutput(skip,sc);GuardOutput(encoder,ec);'
    s=once(s,end,'''  if(scratch_allocations!=44||scratch_reuses!=1452)throw std::runtime_error("scratch reuse inventory changed");
  if(NrEmitDiagnostics){NrDiagnosticStream f(out/"scratch-reuse.json");
    f<<"{\\"allocations\\":"<<scratch_allocations<<",\\"reuses\\":"<<scratch_reuses<<",\\"re_poison_every_use\\":true}\\n";}
'''+end)
    return s
def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',required=True,type=Path);p.add_argument('--layout-parent',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--quiet-progress',action='store_true',help='candidate: progress output only when diagnostics are explicitly enabled');p.add_argument('--reuse-block0-scratch',action='store_true',help='candidate: re-poison and reuse completed chunk-local scratch');a=p.parse_args();repo=Path(__file__).resolve().parents[1]
    parent=json.loads((a.bundle/'manifest.json').read_text());body=dict(parent);root=body.pop('receipt_root_sha256')
    assert canonical(body)==root==ROOT;validate_primitive(a.bundle,parent)
    for name,h in parent['files'].items():assert sha(a.bundle/name)==h,name
    lp=json.loads((a.layout_parent/'plan.json').read_text());b=dict(lp);h=b.pop('receipt_root_sha256');assert canonical(b)==h
    assert sha(a.layout_parent/'entry.metallib')==lp['layout_artifacts']['entry.metallib']
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False);source=a.output/'source';gen=source/'generated';gen.mkdir(parents=True);assets=a.output/'assets';assets.mkdir()
    for name in ('nr_encoder_runtime.h','nr_encoder_runtime.mm'):shutil.copyfile(repo/'src/nr_encoder_runtime'/name,source/name)
    common=(a.bundle/'encoder/source/mlp_probe.mm').read_text().split('\nint main(',1)[0]
    sink=r'''
static thread_local bool NrEmitDiagnostics=false;
static thread_local uint64_t NrCheckedValues=0,NrCheckedOutputs=0;
struct NrDiagnosticMode {bool old; explicit NrDiagnosticMode(bool on):old(NrEmitDiagnostics){NrEmitDiagnostics=on;} ~NrDiagnosticMode(){NrEmitDiagnostics=old;}};
struct NrNullBuffer:std::streambuf {std::streamsize xsputn(const char*,std::streamsize n)override{return n;} int overflow(int c)override{return traits_type::not_eof(c);}};
struct NrDiagnosticStream:std::ostream {
  NrNullBuffer discard;std::ofstream file;
  explicit NrDiagnosticStream(const fs::path& p,std::ios::openmode mode=std::ios::out):std::ostream(nullptr){
    if(NrEmitDiagnostics){file.open(p,mode|std::ios::out);rdbuf(file.rdbuf());if(!file)setstate(std::ios::failbit);}
    else rdbuf(&discard);
  }
};
'''
    common=once(common,'namespace fs=std::filesystem;','namespace fs=std::filesystem;\n'+sink)
    common=once(common,'void Write(const fs::path& p,Buffer& b) {','void Write(const fs::path& p,Buffer& b) {\n  if(!NrEmitDiagnostics)return;')
    (gen/'common.inc').write_text(common)
    profile=(a.bundle/'encoder/source/profile_timing.h').read_text();(gen/'profile_timing.h').write_text(profile)
    pre=(a.bundle/'pre/source/nr_pre/pre_probe.mm').read_text();pre=pre[pre.index('struct PreShape'):]
    pre=between(pre,'int main(int argc,','  if(weights.size()', '''PreOutputs Execute(NrEncoderState& state,id<MTLTexture> color,uint8_t poison,const fs::path& out){@autoreleasepool{
  if(NrEmitDiagnostics)fs::create_directories(out);
  const auto& weights=Constant(state,"block0.weights");
''')
    pre=once(pre,'id<MTLDevice> device=MTLCreateSystemDefaultDevice();auto queue=[device newCommandQueue];NSError*error=nil;','id<MTLDevice> device=state.device;auto queue=state.queue;NSError*error=nil;')
    pre=once(pre,'auto lib=[device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&error];if(!lib)throw std::runtime_error(error.localizedDescription.UTF8String);','auto lib=state.libraries.at("pre.metallib");')
    pre=pre.replace('image.size()','size_t(1920*1080*8)')
    pre=once(pre,'std::memcpy(Data(input),image.data(),size_t(1920*1080*8));','TextureToBuffer(state,color,input);')
    pre=once(pre,'auto d=Read(argv[7]),e=Read(argv[8]);','const auto& d=Constant(state,"instruction-deltas.u8");const auto& e=Constant(state,"instruction-escapes.u32");')
    pre=pre.replace('argc==9','true')
    pre=pre[:pre.index('  // Comparison-only oracle')]+'''  if(immutable||canary)throw std::runtime_error("Block0 immutable/canary failure");
  if(NrEmitDiagnostics)NrDumpProfile(out);
  return {skip,encoder,inputHash,featureHash};
}}
'''
    pre=once(pre,'if(errors)throw std::runtime_error("write coverage/canary failure");','if(errors)throw std::runtime_error("write coverage/canary failure");NrCheckedValues+=counts.bytes/4;++NrCheckedOutputs;')
    pre=diagnostic(pre);assert 'argv' not in pre and 'Read(' not in pre and 'expected' not in pre.lower()
    if a.reuse_block0_scratch:pre=reuse_block0_scratch(pre)
    (gen/'pre.inc').write_text(pre)
    enc=(a.bundle/'encoder/source/probe.mm').read_text();enc=enc[enc.index('struct Pipelines'):]
    enc=between(enc,'int main(int argc,','  NSArray<NSString*>* names=', '''EncoderOutputs Execute(NrEncoderState& state,Buffer input,uint8_t poison,const fs::path& output){@autoreleasepool{
  constexpr uint32_t count=22;EncoderOutputs outputs;
  if(NrEmitDiagnostics)fs::create_directories(output);
  Pipelines p;p.device=state.device;p.queue=state.queue;NSError* error=nil;
  auto lib=state.libraries.at("prefix.metallib");
''')
    enc=between(enc,'  const auto initial=Read(argv[2]);','  const std::array<const char*,4> hashes=', '  if(input.bytes!=17694720)throw std::runtime_error("live encoder input extent");\n')
    enc=enc.replace('Sha(initial.data(),initial.size())','Sha(Data(input),input.bytes)')
    enc=enc.replace('auto bytes=Read(fs::path(argv[3])/','const auto& bytes=Constant(state,fs::path("weights")/')
    for name,arg in [('wide128',7),('wide256',8)]:
        var='lib'+name[4:]
        enc=once(enc,f'auto {var}=[p.device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[{arg}]]] error:&error];',f'auto {var}=state.libraries.at("{name}.metallib");')
    enc=once(enc,'const std::string name="block"+std::to_string(b)+"-native.e4m3";','outputs.skips[3-i]=native_skips[b];\n        const std::string name="block"+std::to_string(b)+"-native.e4m3";')
    enc=once(enc,'Write(output/"block22-middle.e4m3",native_middle);','outputs.entry=native_middle;Write(output/"block22-middle.e4m3",native_middle);')
    enc=once(enc,'NrDumpProfile(output);return 0;','if(NrEmitDiagnostics)NrDumpProfile(output);return outputs;')
    enc=once(enc,'}catch(const std::exception& e){std::cerr<<e.what()<<"\\n";return 1;}}}','}}')
    enc=once(enc,'if(errors)throw std::runtime_error("output write coverage/canary mismatch");','if(errors)throw std::runtime_error("output write coverage/canary mismatch");NrCheckedValues+=counts.bytes/4;++NrCheckedOutputs;')
    enc=diagnostic(enc);assert 'argv' not in enc and 'Read(' not in enc
    (gen/'encoder.inc').write_text(enc)
    for name in ('wide_execute.h','wide_boundaries.h','native_encoder_exports.h'):(gen/name).write_text(diagnostic((a.bundle/'encoder/source'/name).read_text()))
    if a.quiet_progress:
        for name,n in [('pre.inc',1),('encoder.inc',6),('wide_boundaries.h',1)]:
            target=gen/name;target.write_text(quiet_progress(target.read_text(),n))
    mapping={'pre.metallib':a.bundle/'pre/pre.metallib','prefix.metallib':a.bundle/'encoder/prefix.metallib','wide128.metallib':a.bundle/'encoder/wide128.metallib','wide256.metallib':a.bundle/'encoder/wide256.metallib','entry.metallib':a.layout_parent/'entry.metallib','block0.weights':repo/'build/nr_pre/rtx-feature-source-v1/weights.bin','instruction-deltas.u8':repo/'build/nr_noise_math/corrections-v1/instruction-deltas.u8','instruction-escapes.u32':repo/'build/nr_noise_math/corrections-v1/instruction-escapes.u32'}
    mapping.update({'weights/'+x.name:x for x in (a.bundle/'encoder/weights').glob('*.bin')})
    hashes={}
    for name,path in mapping.items():
        dest=assets/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest);hashes[name]=sha(dest);assert hashes[name]==sha(path)
    (gen/'asset_hashes.h').write_text('#pragma once\nstatic const char* NrEncoderParentRoot="'+root+'";\nstatic const std::pair<const char*,const char*> NrEncoderAssetHashes[]={\n'+''.join('  {"'+n+'","'+h+'"},\n' for n,h in sorted(hashes.items()))+'};\n')
    cmd=['xcrun','-sdk','macosx','clang++','-std=c++20','-O2','-fobjc-arc','-Wall','-Wextra','-Werror','-Wno-deprecated-declarations','-c',str(source/'nr_encoder_runtime.mm'),'-o',str(a.output/'nr_encoder_runtime.o')]
    with (a.output/'compile.log').open('x') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    record={'schema':'neuralstudio.nr.encoder-runtime-build.v1','parent_root':root,'layout_parent_root':h,'assets':hashes,'source_files':{str(p.relative_to(source)):sha(p) for p in sorted(source.rglob('*')) if p.is_file()},'command':cmd,'compile_exit':r.returncode,'generator_sha256':sha(Path(__file__)),'shader_changes':False,'validation':'pending'}
    if a.quiet_progress:record['quiet_progress_candidate']={'guarded_sites':8,'diagnostics_enabled_restores_progress':True,'native_validation':'pending'}
    if a.reuse_block0_scratch:record['block0_scratch_candidate']={'scope':'frame-local completed chunk scratch only','allocations_per_frame':44,'reuses_per_frame':1452,'re_poison_every_use':True,'unchanged_coverage_checks':True,'native_validation':'pending'}
    if r.returncode==0:record['object_sha256']=sha(a.output/'nr_encoder_runtime.o')
    record['receipt_root_sha256']=canonical(record);(a.output/'build.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({'compile_exit':r.returncode,'root':record['receipt_root_sha256']},indent=2))
    if r.returncode:raise SystemExit(3)
if __name__=='__main__':main()
