#!/usr/bin/env python3
"""Rebuild a pinned correction ROM from explicit synthetic NVIDIA and M3 scalar sweeps."""
import argparse
from collections import Counter
import hashlib
import json
import mmap
from pathlib import Path
import sys
import tempfile
from netal_sr_calibration import PROFILES

COUNT = 1 << 23


def pack(nvidia, base, operation, mantissas=COUNT):
    if operation not in PROFILES or mantissas <= 0 or mantissas % 4 or len(nvidia) != 2*mantissas or len(base) != len(nvidia):
        raise ValueError('Invalid sweep geometry')
    deltas=PROFILES[operation]['deltas']; packed=bytearray((mantissas if operation=='rcp' else 2*mantissas)//4); histogram=Counter()
    for index in range(len(nvidia)):
        delta=int(nvidia[index])-int(base[index])
        if delta not in deltas: raise ValueError(f'Unsupported ULP correction at {index}: {delta}')
        code=deltas.index(delta); slot=index % mantissas if operation=='rcp' else index
        if operation=='rcp' and index >= mantissas:
            if ((packed[slot//4] >> (2*(slot%4))) & 3) != code: raise ValueError('RCP exponent-band disagreement')
        else: packed[slot//4] |= code << (2*(slot%4))
        restored=int(base[index])+deltas[(packed[slot//4] >> (2*(slot%4))) & 3]
        if restored!=int(nvidia[index]):raise ValueError('Reconstruction mismatch')
        histogram[delta]+=1
    return bytes(packed),dict(sorted(histogram.items()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operation',choices=PROFILES,required=True)
    parser.add_argument('--nvidia-sweep',type=Path,required=True)
    parser.add_argument('--m3-sweep',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();profile=PROFILES[args.operation]
    if sys.byteorder!='little':parser.error('Sweep reader requires a little-endian host')
    if args.output.exists() or args.output.is_symlink():parser.error('Output exists')
    for path,key in [(args.nvidia_sweep,'nvidia'),(args.m3_sweep,'m3')]:
        if path.stat().st_size!=2*COUNT*4 or hashlib.sha256(path.read_bytes()).hexdigest()!=profile[key]:parser.error('Sweep identity mismatch: '+key)
    with args.nvidia_sweep.open('rb') as nf,args.m3_sweep.open('rb') as bf:
        with mmap.mmap(nf.fileno(),0,access=mmap.ACCESS_READ) as nm,mmap.mmap(bf.fileno(),0,access=mmap.ACCESS_READ) as bm:
            nv=memoryview(nm).cast('I');base=memoryview(bm).cast('I')
            try:data,histogram=pack(nv,base,args.operation)
            finally: nv.release();base.release()
    if len(data)!=profile['bytes'] or hashlib.sha256(data).hexdigest()!=profile['sha256']:raise ValueError('ROM contract drift')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.calibration-',dir=args.output.parent) as temporary:
        stage=Path(temporary)/'result';stage.mkdir()
        (stage/f'{args.operation}-correction-rom.bin').write_bytes(data)
        receipt={'schema':'netal.scalar-rom-build.v1','operation':args.operation,'profile':profile,'entries_checked':2*COUNT,
                 'delta_histogram':histogram,'reconstruction_mismatches':0,'model_inputs':0,'fresh_nvidia_execution':False,
                 'complete_runtime':False,'publication_clearance':False}
        (stage/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');stage.rename(args.output)
    print(f'result=pass entries_checked={2*COUNT} model_inputs=0')


if __name__=='__main__':main()
