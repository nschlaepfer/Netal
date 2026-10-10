#!/usr/bin/env python3
"""Extend the sealed two-call observer to three calls; no model modification."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

PARENT='a02bc60f138e7d3d5924f51684d5cefad6fc113da4a4d4de278d262c7d579145'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(r): return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def once(s,a,b):
    if s.count(a)!=1: raise ValueError('changed source anchor '+a[:90])
    return s.replace(a,b)
def host(s):
    s=once(s,'(NrEvaluationIndex ? "frame-1-" : "")','(NrEvaluationIndex ? ("frame-" + std::to_string(NrEvaluationIndex) + "-") : "")')
    s=once(s,'NrEvaluationIndex<2','NrEvaluationIndex<3')
    return once(s,'runtime.sequence_frames=2','runtime.sequence_frames=3')
def proxy(s):
    return once(s,'&& launch_id != 180) return;','&& launch_id != 180 && launch_id != 314 && launch_id != 336) return;')
def hook(s): return once(s,'frame < 2 && after < 2','frame < 3 && after < 2')
def capture(s):
    for a,b in [('$counts.launches -ne 312','$counts.launches -ne 468'),('$counts.argument_bindings -ne 312','$counts.argument_bindings -ne 468'),('$argumentDumps.Count -ne 312','$argumentDumps.Count -ne 468'),('runtime.sequence_frames=2','runtime.sequence_frames=3'),('sequence_frames = 2','sequence_frames = 3'),('$counts.snapshots -ne 4','$counts.snapshots -ne 6'),('$snapshotFiles.Count -ne 4','$snapshotFiles.Count -ne 6'),('Expected 4 bounded history-chain pool snapshots','Expected 6 bounded history-chain pool snapshots'),('Expected 4 bounded history-chain snapshot delta files','Expected 6 bounded history-chain snapshot delta files'),('snapshot_launch_limit = 180','snapshot_launch_limit = 336'),('snapshot_launch_ids = @(2,24,158,180)','snapshot_launch_ids = @(2,24,158,180,314,336)'),('$historyEvents.Count -ne 4','$historyEvents.Count -ne 6'),('Expected four read-only history observations','Expected six read-only history observations'),('$i -lt 4;$i++','$i -lt 6;$i++'),("foreach ($prefix in @('', 'frame-1-'))", "foreach ($prefix in @('', 'frame-1-', 'frame-2-'))")]: s=once(s,a,b)
    s=once(s,"    'runtime.sequence_nonfinite=1:0',","    'runtime.sequence_nonfinite=1:0',\n    'runtime.sequence_nonfinite=2:0',")
    line='Require-OutputLine ("runtime.sequence_frame=1 reset="+$env:NEURALSTUDIO_NR_SECOND_RESET)'
    s=once(s,line,line+'\nRequire-OutputLine ("runtime.sequence_frame=2 reset="+$env:NEURALSTUDIO_NR_SECOND_RESET)')
    files=['history-before.rgba16le','history-after.rgba16le','output.rgba16le','input-color-after.rgba16le','input-depth-after.r32le','input-motion-after.rg16le']
    s=once(s,"    'frame-1-output.rgba16le',",'\n'.join("    'frame-2-"+n+"'," for n in files)+"\n    'frame-1-output.rgba16le',")
    s=once(s,"$failedCalls = @($events | Where-Object {", "if ((Get-FileHash -LiteralPath (Join-Path $ArtifactDirectory 'frame-1-history-after.rgba16le') -Algorithm SHA256).Hash -ne (Get-FileHash -LiteralPath (Join-Path $ArtifactDirectory 'frame-2-history-before.rgba16le') -Algorithm SHA256).Hash) { throw 'History mutated before third evaluation' }\n$failedCalls = @($events | Where-Object {")
    s=once(s,'$secondOutputHash=',"$thirdOutputHash=(Get-FileHash -LiteralPath (Join-Path $ArtifactDirectory 'frame-2-output.rgba16le') -Algorithm SHA256).Hash.ToLowerInvariant()\n$secondOutputHash=")
    s=once(s,'    second_output_sha256 = $secondOutputHash','    second_output_sha256 = $secondOutputHash\n    third_output_sha256 = $thirdOutputHash')
    return s
def runner(s):
    s=once(s,'resets=@(1,$SecondReset);','resets=@(1,$SecondReset,$SecondReset);')
    return once(s,'  $record.first_output_sha256=', '  $record.third_output_sha256=$cap.third_output_sha256\n  $record.first_output_sha256=')
def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=json.loads((a.parent/'source-receipt.json').read_text());b=dict(r);h=b.pop('receipt_root_sha256');assert canonical(b)==h==PARENT
    for n,digest in r['files'].items(): assert sha(a.parent/n)==digest,n
    assert not a.output.exists();shutil.copytree(a.parent,a.output,ignore=shutil.ignore_patterns('source-receipt.json'))
    for name,transform in [('color_cases_trace.cpp',host),('prefix_proxy.cpp',proxy),('nr_history_readback.inl',hook),('capture-color.ps1',capture),('run-nr-reset-sequence.ps1',runner)]:
        (a.output/name).write_text(transform((a.parent/name).read_text()))
    repo=Path(__file__).resolve().parents[1]
    out={'schema':'neuralstudio.nr.three-frame-history-source.v1','parent_source_root':h,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),'frames':3,'supported_resets':[[1,1,1],[1,0,0]],'snapshot_launch_ids':[2,24,158,180,314,336],'same_feature_handle':True,'history_readback_direction':'device_to_host_only','expected_bytes_staged':False,'runtime_modified':False,'executed':False,'files':{str(p.relative_to(a.output)):sha(p) for p in sorted(a.output.rglob('*')) if p.is_file()}}
    out['receipt_root_sha256']=canonical(out);(a.output/'source-receipt.json').write_text(json.dumps(out,indent=2)+'\n');print(out['receipt_root_sha256'])
if __name__=='__main__':main()
