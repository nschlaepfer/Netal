#!/usr/bin/env python3
"""Pack only requested completed-capture evidence into immutable small parts."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile
CHUNK=512*1024
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canonical(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def capture_files(job,label):
    names=['capture-receipt.json','artifact-inventory.json','host-output.txt','rtx-cuda-trace.jsonl','input-color.rgba16le','input-depth.r32le','input-motion.rg16le','output.rgba16le','frame-1-output.rgba16le']
    names += [prefix+n for prefix in ('','frame-1-') for n in ('input-color-after.rgba16le','input-depth-after.r32le','input-motion-after.rg16le')]
    files={n:job/label/n for n in names};files.update({'build-receipt.json':job/'build-receipt.json','job-receipt.json':job/(label+'-receipt.json')});return files
def pack(files,directory,label):
    assert re.fullmatch('[a-z0-9-]+',label) and all(Path(n).name==n and n not in ('.','..') for n in files)
    directory.mkdir(parents=True,exist_ok=True);archive=directory/(label+'.zip');manifest=directory/(label+'-manifest.json')
    assert not archive.exists() and not manifest.exists(),'preserve existing transport bundle'
    pins={n:{'bytes':p.stat().st_size,'sha256':sha(p)} for n,p in files.items()}
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for n,p in files.items():z.write(p,n)
    for n,p in files.items():assert sha(p)==pins[n]['sha256'],'source changed while packing'
    parts=[]
    with archive.open('rb') as f:
        while data:=f.read(CHUNK):
            path=directory/(label+f'.part{len(parts):04d}');assert not path.exists();path.write_bytes(data);parts.append({'name':path.name,'bytes':len(data),'sha256':sha(path)})
    r={'schema':'neuralstudio.nr.capture-transport.v1','label':label,'archive':{'name':archive.name,'bytes':archive.stat().st_size,'sha256':sha(archive)},'files':pins,'parts':parts,'source_files_unchanged':True,'model_executed':False}
    r['receipt_root_sha256']=canonical(r);manifest.write_text(json.dumps(r,indent=2)+'\n');return r
def main():
    p=argparse.ArgumentParser();p.add_argument('--job-root',type=Path,required=True);p.add_argument('--label',required=True);a=p.parse_args();job=a.job_root.resolve()
    assert job.drive.upper()=='D:' and job.parent.name=='CodexLab' and re.fullmatch('nr-color-reset-[a-z0-9-]+',job.name)
    assert re.fullmatch('[a-z0-9-]+',a.label)
    r=pack(capture_files(job,a.label),job/'transport-v1',a.label);print(json.dumps({'root':r['receipt_root_sha256'],'archive_bytes':r['archive']['bytes'],'parts':len(r['parts'])}))
if __name__=='__main__':main()
