#!/usr/bin/env python3
"""Package per-frame original images and comparison-only mixed-sequence outputs."""
import argparse
import importlib.util
import json
from pathlib import Path
spec=importlib.util.spec_from_file_location('pack',Path(__file__).with_name('pack-nr-reset-retrieval.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def main():
    p=argparse.ArgumentParser();p.add_argument('--job-root',type=Path,required=True);a=p.parse_args();job=a.job_root.resolve()
    assert job.drive.upper()=='D:' and job.parent.name=='CodexLab' and job.name=='nr-color-reset-images-20260906-v1'
    label='mixed-reset100-a5-a';names=['input-color.rgba16le','frame-1-input-color.rgba16le','frame-2-input-color.rgba16le','input-motion.rg16le','output.rgba16le','frame-1-output.rgba16le','frame-2-output.rgba16le','capture-receipt.json','artifact-inventory.json','host-output.txt']
    files={n:job/label/n for n in names};files.update({'reference.json':job/(label+'-reference.json'),'build-receipt.json':job/'build-receipt.json','job-receipt.json':job/(label+'-receipt.json')})
    r=m.pack(files,job/'transport-v1',label);print(json.dumps({'label':label,'root':r['receipt_root_sha256'],'archive_bytes':r['archive']['bytes'],'parts':len(r['parts'])}))
if __name__=='__main__':main()
