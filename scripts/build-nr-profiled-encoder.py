#!/usr/bin/env python3
"""Derive profiled hosts, optionally testing a byte-identical poison fill."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
OLD_FILL='std::memset(b.metal.contents,0xc3,bytes+2*Guard);std::memset(Data(b),poison,bytes);'
NEW_FILL='std::memset(b.metal.contents,0xc3,Guard);std::memset(Data(b)+bytes,0xc3,Guard);std::memset(Data(b),poison,bytes);'
def single_pass_fill(source):
    if source.count(OLD_FILL)!=1:raise ValueError('guarded allocation implementation changed')
    return source.replace(OLD_FILL,NEW_FILL)
def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--detailed-phases',action='store_true');p.add_argument('--single-pass-poison',action='store_true');a=p.parse_args();a.output=a.output.resolve();repo=Path(__file__).resolve().parents[1]
    parent=json.loads((a.parent/'manifest.json').read_text());body=dict(parent);h=body.pop('receipt_root_sha256');assert canonical(body)==h
    for name,digest in parent['files'].items():assert sha(a.parent/name)==digest,name
    assert not a.output.exists();shutil.copytree(a.parent,a.output,ignore=shutil.ignore_patterns('manifest.json'))
    header=repo/'src/nr_swin/profile_timing.h'
    for dest in ('pre/source/nr_pre/profile_timing.h','encoder/source/profile_timing.h'):shutil.copyfile(header,a.output/dest)
    def edit(path,replacements):
        text=path.read_text()
        for old,new in replacements:
            assert text.count(old)==1,old[:100];text=text.replace(old,new)
        path.write_text(text)
    pre=a.output/'pre/source/nr_pre/pre_probe.mm'
    edit(pre,[('#undef main','#undef main\n#include "profile_timing.h"'),
      ('void Complete(id<MTLCommandBuffer> command){[command commit];[command waitUntilCompleted];if(command.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error(command.error.localizedDescription.UTF8String);}',
       'void Complete(id<MTLCommandBuffer> command){NrComplete(command);}'),
      ('auto command=[queue commandBuffer];auto enc=', 'auto command=[queue commandBuffer];command.label=@"block0.features";auto enc='),
      ('PreChunk s{std::min(Chunk,Windows-base)*64,base};','NrHostScope profile_scope("block0.chunk");PreChunk s{std::min(Chunk,Windows-base)*64,base};'),
      ('auto cmd=[queue commandBuffer];auto adapter=', 'auto cmd=[queue commandBuffer];cmd.label=@"block0.chunk";auto adapter='),
      ('return (sm|em|immutable|canary)?3:0;','NrDumpProfile(out);return (sm|em|immutable|canary)?3:0;')])
    enc=a.output/'encoder/source/probe.mm'
    edit(enc,[('#undef main','#undef main\n#include "profile_timing.h"'),
      ('  [command commit];[command waitUntilCompleted];\n  if(command.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error(command.error.localizedDescription.UTF8String);','  NrComplete(command);'),
      ('  WindowMap shape{960,576,','  NrHostScope profile_scope("C32.rewindow");\n  WindowMap shape{960,576,'),
      ('  const std::array<unsigned,10> elements=', '  NrHostScope profile_scope(output.filename().string());\n  const std::array<unsigned,10> elements='),
      ('  constexpr size_t rows=480*288;', '  NrHostScope profile_scope("block4.downsample");\n  constexpr size_t rows=480*288;'),
      ('auto command=[p.queue commandBuffer];auto encoder=', 'auto command=[p.queue commandBuffer];command.label=@"C32.rewindow";auto encoder='),
      ('  auto command=[p.queue commandBuffer];\n  for(size_t i=0;i<10;++i)', '  auto command=[p.queue commandBuffer];command.label=[NSString stringWithUTF8String:output.filename().string().c_str()];\n  for(size_t i=0;i<10;++i)'),
      ('  auto command=[p.queue commandBuffer];\n  for(unsigned i=0;i<2;++i)', '  auto command=[p.queue commandBuffer];command.label=@"block4.downsample";\n  for(unsigned i=0;i<2;++i)'),
      ('return 0;','NrDumpProfile(output);return 0;')])
    edit(a.output/'encoder/source/wide_execute.h',[
      ('  const bool weight_extent=', '  NrHostScope profile_scope(output.filename().string());\n  const bool weight_extent='),
      ('auto command=[queue commandBuffer];','auto command=[queue commandBuffer];command.label=[NSString stringWithUTF8String:output.filename().string().c_str()];')])
    edit(a.output/'encoder/source/wide_boundaries.h',[
      ('  WindowMap shape{width,height,from.width,','  NrHostScope profile_scope("C"+std::to_string(channels)+".rewindow");\n  WindowMap shape{width,height,from.width,'),
      ('  const uint32_t destination_width=', '  NrHostScope profile_scope("block"+std::to_string(block)+".downsample");\n  const uint32_t destination_width='),
      ('auto command=[p.queue commandBuffer];auto encoder=', 'auto command=[p.queue commandBuffer];command.label=[NSString stringWithUTF8String:("C"+std::to_string(channels)+".rewindow").c_str()];auto encoder='),
      ('NSError* error=nil;auto command=[p.queue commandBuffer];','NSError* error=nil;auto command=[p.queue commandBuffer];command.label=[NSString stringWithUTF8String:("block"+std::to_string(block)+".downsample").c_str()];')])
    edit(a.output/'encoder/source/native_encoder_exports.h',[
      ('  const size_t rows=', '  NrHostScope profile_scope("C"+std::to_string(channels)+".export");\n  const size_t rows='),
      ('auto command=[p.queue commandBuffer];auto encoder=', 'auto command=[p.queue commandBuffer];command.label=[NSString stringWithUTF8String:("C"+std::to_string(channels)+".export").c_str()];auto encoder=')])
    if a.detailed_phases:
        edit(pre,[
          ('NrHostScope profile_scope("block0.chunk");','NrHostScope profile_scope("block0.chunk");NrPhaseClock phases("block0.chunk","allocation_and_poison");'),
          ('auto cmd=[queue commandBuffer];','phases.next("encode");auto cmd=[queue commandBuffer];'),
          ('[e endEncoding];Complete(cmd);','[e endEncoding];phases.next("submit_wait");Complete(cmd);phases.next("coverage");'),
          ('    if(base==0){','    phases.next("sample_io");if(base==0){'),
          ('    std::cout<<"windows="','    phases.next("progress_and_release");std::cout<<"windows="')])
        edit(enc,[
          ('  const std::array<unsigned,10> elements=', '  NrPhaseClock phases(output.filename().string(),"allocation_and_poison");\n  const std::array<unsigned,10> elements='),
          ('  const auto before=Sha(Data(input),input.bytes),weight_before=', '  phases.next("input_hashes");const auto before=Sha(Data(input),input.bytes),weight_before='),
          ('  auto command=[p.queue commandBuffer];command.label=[NSString stringWithUTF8String:output.filename().string().c_str()];','  phases.next("encode");auto command=[p.queue commandBuffer];command.label=[NSString stringWithUTF8String:output.filename().string().c_str()];'),
          ('  Complete(command);fs::create_directories(output);', '  phases.next("submit_wait");Complete(command);phases.next("coverage_and_samples");fs::create_directories(output);'),
          ('  if(before!=Sha(Data(input),input.bytes)||weight_before!=', '  phases.next("immutability");if(before!=Sha(Data(input),input.bytes)||weight_before!='),
          ('  if(projection)*projection=y[8];','  phases.next("return_and_release");if(projection)*projection=y[8];')])
        edit(a.output/'encoder/source/wide_execute.h',[
          ('  const bool weight_extent=', '  NrPhaseClock phases(output.filename().string(),"input_checks");\n  const bool weight_extent='),
          ('for(size_t i=0;i<11;++i){pipelines[i]=', 'for(size_t i=0;i<11;++i){phases.next("pipeline_creation");pipelines[i]='),
          ('    y.push_back(Make(device,','    phases.next("allocation_and_poison");y.push_back(Make(device,'),
          ('  auto command=[queue commandBuffer];', '  phases.next("encode");auto command=[queue commandBuffer];'),
          ('  Complete(command);fs::create_directories(output);','  phases.next("submit_wait");Complete(command);phases.next("coverage_and_samples");fs::create_directories(output);'),
          ('  if(input_before!=Sha(', '  phases.next("immutability");if(input_before!=Sha('),
          ('  if(projection)*projection=y[9];', '  phases.next("return_and_release");if(projection)*projection=y[9];')])
    if a.single_pass_poison:
        for relative in ('pre/source/nr_swin/mlp_probe.mm','encoder/source/mlp_probe.mm'):
            source=a.output/relative;source.write_text(single_pass_fill(source.read_text()))
    pre_manifest=a.output/'pre/source/source-manifest.json';pre_meta=json.loads(pre_manifest.read_text())
    pre_meta['profiling_parent_manifest_sha256']=sha(a.parent/'pre/source/source-manifest.json')
    pre_meta['transformation']='host-only timing scopes, command labels and GPU timestamp readback; unchanged shaders'
    if a.single_pass_poison:pre_meta['transformation']+='; byte-identical single-pass payload poison fill'
    pre_meta['files']={str(p.relative_to(pre_manifest.parent)):sha(p) for p in sorted(pre_manifest.parent.rglob('*')) if p.is_file() and p!=pre_manifest}
    pre_manifest.write_text(json.dumps(pre_meta,indent=2)+'\n')
    commands=[];cc=['xcrun','-sdk','macosx','clang++','-std=c++20','-O2','-fobjc-arc','-Wall','-Wextra','-Werror','-Wno-deprecated-declarations']
    for name,source in [('pre',pre),('encoder',enc)]:
        command=cc+[str(source),'-framework','Foundation','-framework','Metal','-o',str(a.output/name/'probe')]
        with (a.output/(name+'-profile-build.log')).open('w') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        commands.append({'command':command,'exit_code':r.returncode});assert r.returncode==0,name
    (a.output/'profile-build.json').write_text(json.dumps(commands,indent=2)+'\n')
    for path in a.parent.rglob('*.metallib'):assert sha(path)==sha(a.output/path.relative_to(a.parent)),'shader changed'
    parent['files']={str(p.relative_to(a.output)):sha(p) for p in sorted(a.output.rglob('*')) if p.is_file()}
    parent['profiling_parent_root']=h;parent['timing_only_host_changes']=not a.single_pass_poison;parent['host_allocation_change']='single-pass-payload-poison' if a.single_pass_poison else None;parent['detailed_host_phases']=a.detailed_phases;parent.pop('receipt_root_sha256',None)
    parent['receipt_root_sha256']=canonical(parent);(a.output/'manifest.json').write_text(json.dumps(parent,indent=2)+'\n');print(parent['receipt_root_sha256'])
if __name__=='__main__':main()
