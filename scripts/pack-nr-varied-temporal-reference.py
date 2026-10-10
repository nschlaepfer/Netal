#!/usr/bin/env python3
"""Compress one completed three-frame reference; not inference or replay."""
import argparse
import importlib.util
import json
from pathlib import Path
import re

spec=importlib.util.spec_from_file_location('transport',Path(__file__).with_name('pack-nr-reset-retrieval.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def main():
    p=argparse.ArgumentParser();p.add_argument('--job-root',type=Path,required=True);p.add_argument('--label',required=True);p.add_argument('--minimal',action='store_true');a=p.parse_args();job=a.job_root.resolve()
    assert job.drive.upper()=='D:' and job.parent.name=='CodexLab' and re.fullmatch('nr-color-reset-[a-z0-9-]+',job.name)
    assert re.fullmatch('case[12]-reset100-a5-a',a.label)
    files=m.capture_files(job,a.label);files['reference.json']=job/(a.label+'-reference.json')
    for n in ('frame-2-output.rgba16le','frame-2-input-color-after.rgba16le','frame-2-input-depth-after.r32le','frame-2-input-motion-after.rg16le','history-after.rgba16le','frame-1-history-after.rgba16le','frame-2-history-after.rgba16le'):files[n]=job/a.label/n
    if a.minimal:
        names={'input-color.rgba16le','output.rgba16le','frame-1-output.rgba16le','frame-2-output.rgba16le','reference.json','capture-receipt.json','artifact-inventory.json','host-output.txt','build-receipt.json','job-receipt.json'}
        files={n:p for n,p in files.items() if n in names};assert set(files)==names
    label=a.label+('-minimal' if a.minimal else '')
    r=m.pack(files,job/'transport-v1',label);print(json.dumps({'label':label,'root':r['receipt_root_sha256'],'archive_bytes':r['archive']['bytes'],'parts':len(r['parts'])}))
if __name__=='__main__':main()
