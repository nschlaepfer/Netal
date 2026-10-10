#!/usr/bin/env python3
"""GPU integration checks for reference-present, missing, and corrupt oracle receipts.

Requires a prepared runtime and a compatible four-frame 1080p capture supplied locally.
This does not ship test inputs or prove live-game behavior. In the shared lab, run under
its GPU reservation mechanism. CI runs the source-only tests instead.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe',required=True,type=Path)
    parser.add_argument('--bundle',required=True,type=Path)
    parser.add_argument('--capture',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    output=args.output.absolute();output.mkdir(parents=True,exist_ok=False)
    env={k:v for k,v in os.environ.items() if not k.startswith('NEURALSTUDIO_SR_')}
    env.update(NEURALSTUDIO_SR_PWIN_PIPELINE='0',NEURALSTUDIO_SR_TEMPORAL_FRAME_COUNT='4')
    summary=[]
    for kind in ('reference','missing','truncated'):
        case=output/kind;case.mkdir()
        capture=args.capture.resolve()
        if kind!='reference':
            capture=case/'input';capture.mkdir()
            for name in ('color.rgba16le','depth.r32le','motion.rg16le','exposure.r32le'):
                shutil.copyfile(args.capture/('frame-000-input-'+name),capture/('frame-000-input-'+name))
            params=args.capture/'frame-000-params.json'
            if params.exists():shutil.copyfile(params,capture/params.name)
            if kind=='truncated':(capture/'frame-000-output.rgba16le').write_bytes(b'bad')
        env['NEURALSTUDIO_SR_TEMPORAL_FRAME_COUNT']='4' if kind=='reference' else '1'
        receipt=case/'receipt.txt'
        with (case/'process.log').open('w') as log:
            run=subprocess.run([str(args.probe.resolve()),str(args.bundle.resolve()),str(capture),str(receipt)],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900)
        data=dict(line.split('=',1) for line in receipt.read_text().splitlines() if '=' in line) if receipt.exists() else {}
        if kind=='reference':
            assert run.returncode==0 and data.get('result')=='pass' and data.get('oracle_frames')=='4'
            assert data.get('persistent_temporal_rtx_parity')=='true'
            assert data.get('physical_compute_dispatches')=='unmeasured' and data.get('logical_network_stages')=='56'
            for frame in range(4):
                assert data[f'frame_{frame}_expected_sha256']==data[f'frame_{frame}_output_sha256']
                for name in ('color','depth','motion','exposure'):
                    assert len(data[f'frame_{frame}_input_{name}_sha256'])==64
        elif kind=='missing':
            assert run.returncode!=0 and data.get('result')=='unverified' and data.get('oracle_frames')=='0'
            assert data.get('persistent_temporal_rtx_parity')=='false' and data.get('frame_0_exact')=='unverified'
            assert data.get('output_canaries_all_intact')=='true' and len(data.get('frame_0_output_sha256',''))==64
        else:
            assert run.returncode==7 and data.get('result')!='pass'
        summary.append({'case':kind,'returncode':run.returncode,'receipt_result':data.get('result'),
                        'parity':data.get('persistent_temporal_rtx_parity'),'check':'pass'})
        print(kind,'claim checks passed',flush=True)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':
    main()
