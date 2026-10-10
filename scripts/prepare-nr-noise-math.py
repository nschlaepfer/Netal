#!/usr/bin/env python3
"""Prepare exhaustive math domains with no model, image or frame inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();root=Path(__file__).resolve().parents[1]
    a.output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(root/'src/nr_pre/noise_math_oracle.cu',a.output/'noise_math_oracle.cu')
    shutil.copyfile(root/'scripts/run-nr-noise-math.ps1',a.output/'run.ps1')
    r={'schema':'neuralstudio.nr.noise-math-inputs.v1','files':{x.name:sha(x) for x in sorted(a.output.iterdir())},
       'domain':'Every n/2^24, n=1..2^24, plus all 23-bit mantissas at sqrt exponents 0 and 1',
       'model_inputs':False,'expected_answers_staged':False,'radius_column':'comparison-only range-reduction audit; must never become a runtime radius lookup'}
    r['receipt_root_sha256']=hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    (a.output/'inputs.json').write_text(json.dumps(r,indent=2)+'\n');print(r['receipt_root_sha256'])
if __name__=='__main__':main()
