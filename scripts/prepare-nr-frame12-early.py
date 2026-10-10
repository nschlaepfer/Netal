#!/usr/bin/env python3
"""Add four read-only frame12 encoder snapshots to the sealed 16-frame observer."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def once(s,a,b):
    assert s.count(a)==1,a;return s.replace(a,b)
def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=json.loads((a.parent/'source-receipt.json').read_text());b=dict(r);h=b.pop('receipt_root_sha256');assert canonical(b)==h=='d1b151dc5a8a0101bbb9f9963fdf8d7bf1a3650ef4b21729acf67367cfcd723a'
    for n,d in r['files'].items():assert sha(a.parent/n)==d,n
    old=r['snapshot_launch_ids'];new=sorted(old+[1875,1876,1877,1878]);assert len(new)==36
    assert not a.output.exists();shutil.copytree(a.parent,a.output,ignore=shutil.ignore_patterns('source-receipt.json'))
    proxy=(a.parent/'prefix_proxy.cpp').read_text();proxy=once(proxy,'if ('+' && '.join('launch_id != '+str(n) for n in old)+') return;','if ('+' && '.join('launch_id != '+str(n) for n in new)+') return;');(a.output/'prefix_proxy.cpp').write_text(proxy)
    cap=(a.parent/'capture-color.ps1').read_text()
    for x,y in [('$counts.snapshots -ne 32','$counts.snapshots -ne 36'),('$snapshotFiles.Count -ne 32','$snapshotFiles.Count -ne 36'),('Expected 32 bounded history-chain pool snapshots','Expected 36 bounded history-chain pool snapshots'),('Expected 32 bounded history-chain snapshot delta files','Expected 36 bounded history-chain snapshot delta files'),('snapshot_launch_ids = @('+','.join(map(str,old))+')','snapshot_launch_ids = @('+','.join(map(str,new))+')')]:cap=once(cap,x,y)
    (a.output/'capture-color.ps1').write_text(cap)
    out={'schema':'neuralstudio.nr.frame12-early-source.v1','parent_source_root':h,'frames':16,'snapshot_launch_ids':new,'new_snapshots':[1875,1876,1877,1878],'host_model_unchanged':True,'observation_only':True,'expected_bytes_staged':False,'executed':False,'files':{str(p.relative_to(a.output)):sha(p) for p in sorted(a.output.rglob('*')) if p.is_file()}}
    out['receipt_root_sha256']=canonical(out);(a.output/'source-receipt.json').write_text(json.dumps(out,indent=2)+'\n');print(out['receipt_root_sha256'])
if __name__=='__main__':main()
