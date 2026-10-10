#!/usr/bin/env python3
"""Derive observation-only clear/Block0 snapshots from the accepted color job.

All156 original launches still execute. Only snapshot retention is limited.
GPU input arrays are also read back after evaluation to check immutability.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def once(text,old,new):
    if text.count(old)!=1:raise ValueError('parent changed: '+old[:80])
    return text.replace(old,new)
def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--encoder-boundary',action='store_true');a=p.parse_args();repo=Path(__file__).resolve().parents[1]
    parent=json.loads((a.parent/'source-receipt.json').read_text());b=dict(parent);h=b.pop('receipt_root_sha256');assert canonical(b)==h
    for name,digest in parent['files'].items():assert sha(a.parent/name)==digest,name
    assert not a.output.exists();shutil.copytree(a.parent,a.output,ignore=shutil.ignore_patterns('source-receipt.json'))
    proxy=repo/'src/media_host/nvcuda_forward_trace.cpp';source=once(proxy.read_text(),'void CaptureSnapshotDelta(LONG launch_id) {','void CaptureSnapshotDelta(LONG launch_id) {\n  if (launch_id > 2) return; // observation-only bounded retention')
    if a.encoder_boundary:source=once(source,'if (launch_id > 2) return;','if (launch_id != 2 && launch_id != 24) return;')
    (a.output/'prefix_proxy.cpp').write_text(source)
    build=a.output/'build-nr-color-job.ps1';text=build.read_text()
    insertion='''$compileExit = $LASTEXITCODE
$proxyCommand = '"{0}" >nul && cl.exe /nologo /std:c++20 /EHsc /MT /W4 /WX /LD /DUNICODE /D_UNICODE /Fo"{1}" "{2}" /link /OUT:"{3}"' -f $vcvars,(Join-Path $build 'prefix-proxy.obj'),(Join-Path $source 'prefix_proxy.cpp'),(Join-Path $build 'nvcuda.dll')
if ($compileExit -eq 0) { & $env:ComSpec /d /s /c $proxyCommand *> (Join-Path $build 'proxy-compile.log'); $compileExit = $LASTEXITCODE }
'''
    text=once(text,'$compileExit = $LASTEXITCODE\n',insertion)
    text=once(text,'command=$command;exit_code=', 'command=$command;proxy_command=$proxyCommand;exit_code=');build.write_text(text)
    capture=a.output/'capture-color.ps1';text=capture.read_text()
    for old,new in [('$counts.snapshots -ne 156','$counts.snapshots -ne 2'),('$snapshotFiles.Count -ne 156','$snapshotFiles.Count -ne 2'),('Expected 156 pool snapshots','Expected 2 prefix pool snapshots'),('Expected 156 snapshot delta files','Expected 2 prefix snapshot delta files')]:text=once(text,old,new)
    text=once(text,"    'runtime.evaluate_result=0x1',","    'runtime.evaluate_result=0x1',\n    'runtime.input_arrays_immutable=1',")
    text=once(text,"    'output.rgba16le'\n)","    'output.rgba16le',\n    'input-color-after.rgba16le',\n    'input-depth-after.r32le',\n    'input-motion-after.rg16le'\n)")
    text=once(text,'    snapshot_capture_enabled = [bool]$CaptureSnapshots','    snapshot_capture_enabled = [bool]$CaptureSnapshots\n    snapshot_launch_limit = '+('24' if a.encoder_boundary else '2')+'\n    snapshot_launch_ids = @('+('2,24' if a.encoder_boundary else '1,2')+')\n    input_arrays_immutable = $true');capture.write_text(text)
    host=a.output/'color_cases_trace.cpp';text=host.read_text()
    check='''        std::vector<uint16_t> color_after(color_host.size()), motion_after(motion_host.size());
        std::vector<uint32_t> depth_after(depth_host.size());
        if (!CopyArrayToHost(cu_memcpy_2d,color,color_after.data(),static_cast<size_t>(kWidth)*8,kHeight) ||
            !CopyArrayToHost(cu_memcpy_2d,depth,depth_after.data(),static_cast<size_t>(kWidth)*4,kHeight) ||
            !CopyArrayToHost(cu_memcpy_2d,motion,motion_after.data(),static_cast<size_t>(kWidth)*4,kHeight) ||
            color_after!=color_host || depth_after!=depth_host || motion_after!=motion_host ||
            !WriteIoArtifact("input-color-after.rgba16le",color_after.data(),color_after.size()*2) ||
            !WriteIoArtifact("input-depth-after.r32le",depth_after.data(),depth_after.size()*4) ||
            !WriteIoArtifact("input-motion-after.rg16le",motion_after.data(),motion_after.size()*2)) {
          std::cerr << "error=input_array_mutation_or_readback_failure\\n";return 11;
        }
        std::cout << "runtime.input_arrays_immutable=1\\n";
        const HalfOutputStatistics statistics = AnalyzeHalfOutput(output_host);'''
    text=once(text,'        const HalfOutputStatistics statistics = AnalyzeHalfOutput(output_host);',check);host.write_text(text)
    run=a.output/'run-nr-color-job.ps1';text=run.read_text();text=once(text,"'-ArmRealExecution','-ArtifactDirectory'","'-ArmRealExecution','-CaptureSnapshots','-ArtifactDirectory'")
    old="  if ($CaseId -eq 0 -and $r.output_sha256 -ne 'a07835db4be8469dad1e3d3befcc495be396790f89408be62ddb8a5a710585a2') { throw 'Control output differs from accepted RTX frame' }"
    new="""  $referenceOutputs=@('a07835db4be8469dad1e3d3befcc495be396790f89408be62ddb8a5a710585a2','1391787165701ef795a2259432a8672ec3bba779b9c09c7abe602848d8677f29','4e3df0c2d6c15ba20f35bbd2f3374d98d52687bd6345e4fe1191f33aafe16379','153fa0d408035d0f88bf483bedb87ca26e250b8bf6e7927f71c2f799dc03c534')
  if ($r.output_sha256 -ne $referenceOutputs[$CaseId]) { throw 'Observation changed final output vs accepted untraced case' }
  if ($r.counts.snapshots -ne 2 -or -not $r.input_arrays_immutable) { throw 'Prefix/input-after evidence missing' }"""
    run.write_text(once(text,old,new))
    parent.pop('receipt_root_sha256');parent['parent_source_root']=h;parent['snapshot_launch_limit']=24 if a.encoder_boundary else 2;parent['snapshot_launch_ids']=[2,24] if a.encoder_boundary else [1,2];parent['proxy_parent_sha256']=sha(proxy);parent['post_eval_input_array_checks']=True
    parent['scope']='All156 original launches execute. Retain only the declared two boundary snapshots; compare complete output with accepted untraced color case and input arrays before/after. No model/runtime patch or expected tensor GPU input.'
    parent['files']={str(p.relative_to(a.output)):sha(p) for p in sorted(a.output.rglob('*')) if p.is_file()};parent['receipt_root_sha256']=canonical(parent)
    (a.output/'source-receipt.json').write_text(json.dumps(parent,indent=2)+'\n');print(parent['receipt_root_sha256'])
if __name__=='__main__':main()
