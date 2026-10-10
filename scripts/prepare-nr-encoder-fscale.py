#!/usr/bin/env python3
"""Prepare an isolated encoder primitive transfer from a frozen peer source.

Does not compile, execute, promote, or modify either accepted implementation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def transform(reference,peer):
    signature='inline half nr_qmma_finish(thread const NrQmmaAccumulator& q)'
    if reference.count(signature)!=1:raise ValueError('reference signature changed')
    step=peer.split('inline half nr_qmma_step(thread const half* a, thread const half* b, half c) {')
    if len(step)!=2:raise ValueError('peer step signature changed')
    branches=step[1].split('#elif NR_QMMA_FSCALE\n')
    if len(branches)!=2:raise ValueError('peer float-scaled branch changed')
    body=branches[1].split('#elif NR_QMMA_NOARRAY')[0]
    if 'const int ex = int((as_type<uint>(fa)' not in body or 'maxexp = max(maxexp, ex)' not in body:
        raise ValueError('operand-exponent form required')
    if body.count('nr_hw_fixed_to_half_rne(')!=1:raise ValueError('peer RNE call changed')
    body=body.replace('nr_hw_fixed_to_half_rne(','nr_qmma_fixed_half(')
    wrapper='''
// Adapted from the pinned peer float-scaled branch. The original integer
// reference and its RNE pack remain available, with the same finite guard.
inline half nr_qmma_finish(thread const NrQmmaAccumulator& q) {
  thread const half* a=q.a;thread const half* b=q.b;const half c=q.prior;
  if((as_type<ushort>(c)&0x7c00u)==0x7c00u)return as_type<half>(ushort(0x7e00u));
  for(uint k=0;k<32u;++k)if((as_type<ushort>(a[k])&0x7c00u)==0x7c00u || (as_type<ushort>(b[k])&0x7c00u)==0x7c00u)return as_type<half>(ushort(0x7e00u));
'''+body+'\n}\n'
    return reference.replace(signature,signature.replace('nr_qmma_finish','nr_qmma_finish_integer_reference'))+wrapper

def main():
    p=argparse.ArgumentParser();p.add_argument('--peer-common',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();repo=Path(__file__).resolve().parents[1]
    accepted=repo/'build/nr_qmma_fixed/macos-v1';r=json.loads((accepted/'verification.json').read_text());b=dict(r);h=b.pop('receipt_root_sha256')
    assert canonical(b)==h=='e478976e560da01d2fb449252b41e5cfbdcc1f0a0ed976274d4b11f74136862f' and r['result']=='pass'
    for name,digest in r['files'].items():assert sha(accepted/name)==digest,name
    a.output.mkdir(parents=True,exist_ok=False);shutil.copytree(accepted/'source',a.output/'source')
    source=a.output/'source/nr_swin/qmma_fixed.metalh';original=source.read_text();peer=a.peer_common.read_text()
    source.write_text(transform(original,peer));shutil.copyfile(a.peer_common,a.output/'peer-common.metal')
    out={'schema':'neuralstudio.nr.encoder-fscale-source.v1','reference_primitive_root':h,'peer_common_path':str(a.peer_common.resolve()),'peer_common_sha256':sha(a.peer_common),
         'reference_header_sha256':hashlib.sha256(original.encode()).hexdigest(),'files':{str(p.relative_to(a.output)):sha(p) for p in sorted(a.output.rglob('*')) if p.is_file()},
         'adaptation':'Existing NrQmmaAccumulator interface, original integer RNE pack, original finite-contract NaN guards; peer operand-exponent float-scaled body. No layer/layout change.',
         'compiled':False,'executed':False,'promoted':False}
    out['receipt_root_sha256']=canonical(out);(a.output/'source-receipt.json').write_text(json.dumps(out,indent=2)+'\n');print(out['receipt_root_sha256'])
if __name__=='__main__':main()
