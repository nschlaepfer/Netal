#!/usr/bin/env python3
"""Sixteen uninterrupted evaluations or an all-reset control on one handle."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
PARENT='c7d9a43fa089bb1f564b2119db4666b830b1673679b6fa539310a2987910a05d'
SNAPS=[156*i+j for i in range(16) for j in (2,24)]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def once(s,a,b):
    if s.count(a)!=1:raise ValueError('changed anchor '+a[:80])
    return s.replace(a,b)
def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=json.loads((a.parent/'source-receipt.json').read_text());b=dict(r);h=b.pop('receipt_root_sha256');assert canonical(b)==h==PARENT
    for n,d in r['files'].items():assert sha(a.parent/n)==d,n
    assert not a.output.exists();shutil.copytree(a.parent,a.output,ignore=shutil.ignore_patterns('source-receipt.json'))
    host=(a.parent/'color_cases_trace.cpp').read_text()
    for x,y in [('NrEvaluationIndex<8','NrEvaluationIndex<16'),('(NrEvaluationIndex%4u)==0u ? 1 :', 'NrEvaluationIndex==0u ? 1 :'),('runtime.sequence_frames=8','runtime.sequence_frames=16')]:host=once(host,x,y)
    (a.output/'color_cases_trace.cpp').write_text(host)
    hook=once((a.parent/'nr_history_readback.inl').read_text(),'frame < 8 && after < 2','frame < 16 && after < 2');(a.output/'nr_history_readback.inl').write_text(hook)
    proxy=(a.parent/'prefix_proxy.cpp').read_text();old='if ('+' && '.join('launch_id != '+str(156*i+j) for i in range(8) for j in (2,24))+') return;';proxy=once(proxy,old,'if ('+' && '.join('launch_id != '+str(n) for n in SNAPS)+') return;');(a.output/'prefix_proxy.cpp').write_text(proxy)
    cap=(a.parent/'capture-color.ps1').read_text()
    for x,y in [('$counts.launches -ne 1248','$counts.launches -ne 2496'),('$counts.argument_bindings -ne 1248','$counts.argument_bindings -ne 2496'),('$argumentDumps.Count -ne 1248','$argumentDumps.Count -ne 2496'),('runtime.sequence_frames=8','runtime.sequence_frames=16'),('sequence_frames = 8','sequence_frames = 16'),('$historyEvents.Count -ne 16','$historyEvents.Count -ne 32'),('Expected sixteen read-only history observations','Expected thirty-two read-only history observations'),('$i -lt 16;$i++','$i -lt 32;$i++'),('$counts.snapshots -ne 16','$counts.snapshots -ne 32'),('$snapshotFiles.Count -ne 16','$snapshotFiles.Count -ne 32'),('Expected 16 bounded history-chain pool snapshots','Expected 32 bounded history-chain pool snapshots'),('Expected 16 bounded history-chain snapshot delta files','Expected 32 bounded history-chain snapshot delta files'),('snapshot_launch_limit = 1116','snapshot_launch_limit = 2364'),('snapshot_launch_ids = @('+','.join(str(156*i+j) for i in range(8) for j in (2,24))+')','snapshot_launch_ids = @('+','.join(map(str,SNAPS))+')'),('if(($i%4) -eq 0)','if($i -eq 0)')]:cap=once(cap,x,y)
    assert cap.count('$i -lt 8;')==3;cap=cap.replace('$i -lt 8;','$i -lt 16;')
    old="foreach ($prefix in @('', 'frame-1-', 'frame-2-', 'frame-3-', 'frame-4-', 'frame-5-', 'frame-6-', 'frame-7-'))";new="foreach ($prefix in @(''"+''.join(", 'frame-"+str(i)+"-'" for i in range(1,16))+"))";cap=once(cap,old,new)
    extra=[f'frame-{i}-'+n for i in range(8,16) for n in ('history-before.rgba16le','history-after.rgba16le','output.rgba16le','input-color-after.rgba16le','input-depth-after.r32le','input-motion-after.rg16le')]
    cap=once(cap,"    'frame-1-output.rgba16le',",'\n'.join("    '"+n+"'," for n in extra)+"\n    'frame-1-output.rgba16le',");(a.output/'capture-color.ps1').write_text(cap)
    run=(a.parent/'run-nr-reset-sequence.ps1').read_text();old='resets=@(1,$SecondReset,$SecondReset,$SecondReset,1,$SecondReset,$SecondReset,$SecondReset);';run=once(run,old,'resets=@(1'+',$SecondReset'*15+');');(a.output/'run-nr-reset-sequence.ps1').write_text(run)
    record={'schema':'neuralstudio.nr.sixteen-source.v1','parent_source_root':h,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[1],text=True).strip(),'frames':16,'reset_sequences':[[1]*16,[1]+[0]*15],'snapshot_launch_ids':SNAPS,'same_model_handle':True,'expected_bytes_staged':False,'model_unchanged':True,'executed':False,'files':{str(p.relative_to(a.output)):sha(p) for p in sorted(a.output.rglob('*')) if p.is_file()}}
    record['receipt_root_sha256']=canonical(record);(a.output/'source-receipt.json').write_text(json.dumps(record,indent=2)+'\n');print(record['receipt_root_sha256'])
if __name__=='__main__':main()
