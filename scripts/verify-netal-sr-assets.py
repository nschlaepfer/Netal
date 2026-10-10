#!/usr/bin/env python3
"""Compare generated SR body data against a separately supplied reference runtime.

Shared Metal-library bytes are recorded, not treated as an inference equivalence test.
No reference binaries or captured inputs are copied into source or generated assets.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def verify(generated, reference):
    receipt=json.loads((generated/'SOURCE_BUILD_RECEIPT.json').read_text())
    specification=importlib.util.spec_from_file_location('boundary_abi',Path(__file__).resolve().parents[1]/'tools/netal_sr_abi.py')
    abi=importlib.util.module_from_spec(specification);specification.loader.exec_module(abi)
    mismatches=[];matched=[];libraries=[];canonical_abi_matches=0;profile_contract_matches=0
    for relative,identity in receipt['files'].items():
        path=generated/relative
        if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=identity['sha256']:
            mismatches.append({'path':relative,'reason':'generated inventory mismatch'});continue
        target=reference/relative
        if relative.startswith('pipeline/layouts/'):
            # Derived metadata has no counterpart in the installed runtime; binary outputs are
            # compared below, and layout correctness is covered by their use in pipeline JSONs.
            continue
        if path.name in abi.CONTRACTS and 'abi' in path.parts:
            if not target.is_file() or path.read_bytes()!=abi.canonicalize(path.name,path.read_bytes()) or path.read_bytes()!=abi.canonicalize(path.name,target.read_bytes()):
                mismatches.append({'path':relative,'reason':'canonical boundary ABI differs'})
            else:
                canonical_abi_matches+=1;matched.append(relative)
            continue
        if relative.startswith('profiles/') and path.name=='manifest.json':
            if not target.is_file():
                mismatches.append({'path':relative,'reason':'reference profile missing'});continue
            left=json.loads(path.read_text());right=json.loads(target.read_text())
            fields=('schema','output','render_reference','arenas','arrays','stages','hiluma_input','hiluma_output','exposure_copy')
            if any(left.get(k)!=right.get(k) for k in fields):
                mismatches.append({'path':relative,'reason':'runtime profile contract differs'})
            else:
                profile_contract_matches+=1;matched.append(relative)
            continue
        if relative.endswith('/layout.json'):
            target=target.with_name('manifest.json')
            if not target.is_file():
                mismatches.append({'path':relative,'reason':'reference profile missing'});continue
            left=json.loads(path.read_text());right=json.loads(target.read_text())
            if any(left[k]!=right.get(k) for k in ('output','arenas','stages')):
                mismatches.append({'path':relative,'reason':'profile geometry mismatch'})
            else:matched.append(relative)
            continue
        if relative=='pwin/enc0/arguments-template.bin':
            target=reference/'pwin/enc0-arguments-template.bin'
        elif relative=='pwin/enc0/replay-config.bin':
            target=reference/'profiles/1920x1080/pwin/enc0/replay-config.bin'
        if not target.is_file():
            mismatches.append({'path':relative,'reason':'reference asset missing'});continue
        if path.suffix=='.metallib':
            libraries.append({'path':relative,'generated_sha256':identity['sha256'],
                              'reference_sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
            continue
        equal=(json.loads(path.read_text())==json.loads(target.read_text())) if path.suffix=='.json' else path.read_bytes()==target.read_bytes()
        if not equal:mismatches.append({'path':relative,'reason':'content differs'})
        else:matched.append(relative)
    return {'schema':'netal.sr.body-comparison.v1','data_match':not mismatches,'matched_data_assets':len(matched),
            'canonical_abi_matches':canonical_abi_matches,'profile_contract_matches':profile_contract_matches,
            'mismatches':mismatches,'libraries':libraries,'inference_validated':False,'complete_runtime':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generated',required=True,type=Path)
    parser.add_argument('--reference',required=True,type=Path)
    parser.add_argument('--report',required=True,type=Path)
    args=parser.parse_args()
    result=verify(args.generated,args.reference)
    args.report.write_text(json.dumps(result,indent=2)+'\n')
    print(f'data_match={result["data_match"]} matched={result["matched_data_assets"]} inference_validated=false')
    return 0 if result['data_match'] else 1


if __name__=='__main__':
    raise SystemExit(main())
