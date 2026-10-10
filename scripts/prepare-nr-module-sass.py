#!/usr/bin/env python3
"""Stage the two selected C32 modules for offline ptxas inspection."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();root=Path(__file__).resolve().parents[1]
    a.output.mkdir(parents=True,exist_ok=False)
    expected=['9fa036fc715cb56c9138a1f931db1314fdd444ed30d1c31f66307a6babc7176a','0ebd93ef1f85ac4b1981a2dc31e49c00273e107ebe397bb13b0a72bc97face99']
    for i,h in enumerate(expected):
        src=root/f'build/nr_compressed_ptx/extraction-v2/module-{i:02d}.ptx';assert sha(src)==h;shutil.copyfile(src,a.output/src.name)
    shutil.copyfile(root/'scripts/disassemble-nr-selected-modules.ps1',a.output/'run.ps1')
    shutil.copyfile(root/'src/nr_probes/jit_cubin.cu',a.output/'jit_cubin.cu')
    r={'schema':'neuralstudio.nr.module-sass-inputs.v1','files':{x.name:sha(x) for x in sorted(a.output.iterdir())},'purpose':'offline compiler contraction evidence, not live JIT identity'}
    r['receipt_root_sha256']=hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    (a.output/'inputs.json').write_text(json.dumps(r,indent=2)+'\n');print(r['receipt_root_sha256'])
if __name__=='__main__':main()
