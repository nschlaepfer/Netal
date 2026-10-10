#!/usr/bin/env python3
"""Eight live RTX evaluations, with all-reset control or a reset at frame4."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

PARENT='88d2006591a451aac66a16547134c80a7e1c7ae2c714978a2ec4931a1e1fffe6'
SNAPSHOTS=[156*i+j for i in range(8) for j in (2,24)]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def once(s,a,b):
    if s.count(a)!=1:raise ValueError('changed anchor '+a[:90])
    return s.replace(a,b)
def host(s):
    s=once(s,'NrEvaluationIndex<3','NrEvaluationIndex<8')
    s=once(s,"NrEvaluationIndex ? reset_text[0]-'0' : 1","(NrEvaluationIndex%4u)==0u ? 1 : reset_text[0]-'0'")
    return once(s,'runtime.sequence_frames=3','runtime.sequence_frames=8')
def capture(s):
    for a,b in [('$counts.launches -ne 468','$counts.launches -ne 1248'),('$counts.argument_bindings -ne 468','$counts.argument_bindings -ne 1248'),('$argumentDumps.Count -ne 468','$argumentDumps.Count -ne 1248'),('runtime.sequence_frames=3','runtime.sequence_frames=8'),('sequence_frames = 3','sequence_frames = 8'),('$historyEvents.Count -ne 6','$historyEvents.Count -ne 16'),('Expected six read-only history observations','Expected sixteen read-only history observations'),('$i -lt 6;$i++','$i -lt 16;$i++'),('$counts.snapshots -ne 6','$counts.snapshots -ne 16'),('$snapshotFiles.Count -ne 6','$snapshotFiles.Count -ne 16'),('Expected 6 bounded history-chain pool snapshots','Expected 16 bounded history-chain pool snapshots'),('Expected 6 bounded history-chain snapshot delta files','Expected 16 bounded history-chain snapshot delta files'),('snapshot_launch_limit = 336','snapshot_launch_limit = 1116'),('snapshot_launch_ids = @(2,24,158,180,314,336)','snapshot_launch_ids = @('+','.join(map(str,SNAPSHOTS))+')')]:s=once(s,a,b)
    old='Require-OutputLine ("runtime.sequence_frame=1 reset="+$env:NEURALSTUDIO_NR_SECOND_RESET)\nRequire-OutputLine ("runtime.sequence_frame=2 reset="+$env:NEURALSTUDIO_NR_SECOND_RESET)'
    s=once(s,old,'''for($i=0;$i -lt 8;$i++) {
  $reset=if(($i%4) -eq 0){1}else{[int]$env:NEURALSTUDIO_NR_SECOND_RESET}
  Require-OutputLine ("runtime.sequence_frame="+$i+" reset="+$reset)
  Require-OutputLine ("runtime.sequence_nonfinite="+$i+":0")
}''')
    extra=[f'frame-{i}-'+n for i in range(3,8) for n in ('history-before.rgba16le','history-after.rgba16le','output.rgba16le','input-color-after.rgba16le','input-depth-after.r32le','input-motion-after.rg16le')]
    s=once(s,"    'frame-1-output.rgba16le',",'\n'.join("    '"+n+"'," for n in extra)+"\n    'frame-1-output.rgba16le',")
    s=once(s,"foreach ($prefix in @('', 'frame-1-', 'frame-2-'))", "foreach ($prefix in @('', 'frame-1-', 'frame-2-', 'frame-3-', 'frame-4-', 'frame-5-', 'frame-6-', 'frame-7-'))")
    s=once(s,'$failedCalls = @($events | Where-Object {', '''for($i=3;$i -lt 8;$i++) {
  $before='frame-'+$i+'-history-before.rgba16le';$after='frame-'+($i-1)+'-history-after.rgba16le'
  if((Get-FileHash (Join-Path $ArtifactDirectory $before) -Algorithm SHA256).Hash -ne (Get-FileHash (Join-Path $ArtifactDirectory $after) -Algorithm SHA256).Hash){throw 'History changed between evaluations'}
}
$failedCalls = @($events | Where-Object {''')
    s=once(s,'$thirdOutputHash=', '''$allOutputHashes=@();for($i=0;$i -lt 8;$i++){$n=if($i -eq 0){'output.rgba16le'}else{'frame-'+$i+'-output.rgba16le'};$allOutputHashes+=(Get-FileHash (Join-Path $ArtifactDirectory $n) -Algorithm SHA256).Hash.ToLowerInvariant()}
$thirdOutputHash=''')
    return once(s,'    third_output_sha256 = $thirdOutputHash','    third_output_sha256 = $thirdOutputHash\n    all_output_sha256 = $allOutputHashes')
def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=json.loads((a.parent/'source-receipt.json').read_text());b=dict(r);h=b.pop('receipt_root_sha256');assert canonical(b)==h==PARENT
    for n,d in r['files'].items():assert sha(a.parent/n)==d,n
    assert not a.output.exists();shutil.copytree(a.parent,a.output,ignore=shutil.ignore_patterns('source-receipt.json'))
    (a.output/'color_cases_trace.cpp').write_text(host((a.parent/'color_cases_trace.cpp').read_text()))
    (a.output/'capture-color.ps1').write_text(capture((a.parent/'capture-color.ps1').read_text()))
    hook=once((a.parent/'nr_history_readback.inl').read_text(),'frame < 3 && after < 2','frame < 8 && after < 2');(a.output/'nr_history_readback.inl').write_text(hook)
    proxy=(a.parent/'prefix_proxy.cpp').read_text();old='if (launch_id != 2 && launch_id != 24 && launch_id != 158 && launch_id != 180 && launch_id != 314 && launch_id != 336) return;';proxy=once(proxy,old,'if ('+' && '.join('launch_id != '+str(n) for n in SNAPSHOTS)+') return;');(a.output/'prefix_proxy.cpp').write_text(proxy)
    run=(a.parent/'run-nr-reset-sequence.ps1').read_text();run=once(run,'resets=@(1,$SecondReset,$SecondReset);','resets=@(1,$SecondReset,$SecondReset,$SecondReset,1,$SecondReset,$SecondReset,$SecondReset);');run=once(run,'  $record.third_output_sha256=$cap.third_output_sha256','  $record.third_output_sha256=$cap.third_output_sha256\n  $record.all_output_sha256=$cap.all_output_sha256');(a.output/'run-nr-reset-sequence.ps1').write_text(run)
    record={'schema':'neuralstudio.nr.eight-frame-source.v1','parent_source_root':h,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[1],text=True).strip(),'frames':8,'reset_sequences':[[1]*8,[1,0,0,0,1,0,0,0]],'snapshot_launch_ids':SNAPSHOTS,'same_model_handle':True,'expected_bytes_staged':False,'model_unchanged':True,'executed':False,'files':{str(p.relative_to(a.output)):sha(p) for p in sorted(a.output.rglob('*')) if p.is_file()}}
    record['receipt_root_sha256']=canonical(record);(a.output/'source-receipt.json').write_text(json.dumps(record,indent=2)+'\n');print(record['receipt_root_sha256'])
if __name__=='__main__':main()
