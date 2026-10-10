#!/usr/bin/env python3
"""Read-only storage admission estimate for one exact complementary checked pair."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    p=argparse.ArgumentParser();p.add_argument('--peer',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();peer=a.peer.resolve()
    dirs=[peer/'build/nr_full/checked-pairs-v2/black3-v29t-v11-checked/checked-1',peer/'build/nr_full/checked-pairs-v2/black3-v29t-checked/checked-1']
    rows=[]
    for d in dirs:
        files=[p for p in d.rglob('*') if p.is_file()]
        r=json.loads((d/'receipt.json').read_text());assert r['result']=='pass' and len(r['frames'])==3 and all(x['body_mode']=='checked' for x in r['frames'])
        rows.append({'completed_run':str(d),'logical_bytes':sum(p.stat().st_size for p in files),'file_count':len(files),'receipt_sha256':hashlib.sha256((d/'receipt.json').read_bytes()).hexdigest()})
    base=sum(r['logical_bytes'] for r in rows);reserve=(base*5+3)//4+128*1024**2;free=shutil.disk_usage(peer).free;floor=12*1024**3
    r={'schema':'neuralstudio.nr.complementary-pair-storage-plan.v1','result':'fits_at_observation' if free-reserve>=floor else 'does_not_fit','observed_runs':rows,
       'task':'ONE additional [5a,a5,5a] black3 process per side, same v29t binaries/configuration/dimensions/retention as these completed runs',
       'observed_logical_bytes_total':base,'conservative_peak_disk_bytes':reserve,'headroom_rule':'1.25 times observed bytes for BOTH runs, plus128MiB metadata/variation reserve',
       'free_bytes_observed':free,'unchanged_floor_bytes':floor,'projected_free_bytes':free-reserve,
       'execution_authorized_by_this_file':False,'GPU_executed':False,'files_deleted':False,
       'conditions':['Recheck exact binaries, paths, dimensions and retention; this is not a budget for other recipes.','Re-measure free space under the coordinated lock immediately before launch; refuse if the full pair plus margin does not fit.','Keep the12GiB floor unchanged. No evidence deletion, thinning, or broad reserve reduction.','This does not provide capacity for all remaining gates; user storage action remains requested.']}
    r['receipt_root_sha256']=hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest();assert not a.output.exists();a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))


if __name__=='__main__':main()
