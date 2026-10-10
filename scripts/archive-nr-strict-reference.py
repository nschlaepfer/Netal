#!/usr/bin/env python3
"""Preserve independently accepted evidence without modifying the peer attempt.

This is a comparison/evidence archive, NEVER a live model-input directory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--audit',required=True,type=Path);p.add_argument('--peer-checkout',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    audit=json.loads(a.audit.read_text());body=dict(audit);root=body.pop('receipt_root_sha256')
    assert canonical(body)==root and audit['strict_acceptance'] is True and audit['four_frames_match_rtx'] is True and not audit['findings']
    attempt=Path(audit['attempt']).resolve();peer=a.peer_checkout.resolve();a.output.mkdir(parents=True,exist_ok=False);inventory=[]
    def preserve(source,destination,expected=None):
        source=Path(source).resolve();target=a.output/destination;assert source.is_file() and not target.exists()
        digest=sha(source);assert expected is None or digest==expected,source
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target);assert sha(target)==digest
        inventory.append({'source':str(source),'file':str(destination),'bytes':target.stat().st_size,'sha256':digest})
    preserve(a.audit,Path('strict-audit.json'))
    preserve(attempt/'full-chain.json',Path('full-chain.json'),audit['chain_receipt_sha256'])
    for source,h in audit['observed_receipt_hashes'].items():
        source=Path(source)
        if source==attempt/'full-chain.json':continue
        try:relative=Path('receipts')/source.relative_to(attempt)
        except ValueError:relative=Path('external-receipts')/h/source.name
        preserve(source,relative,h)
    for run in ('a5-a','a5-b','5a-a','5a-b'):
        for launch in (155,156):
            for name in ('tex_out.bin','tex_coverage.bin'):
                source=attempt/f'body-{run}/launch-{launch}/run'/name
                preserve(source,Path('frames')/run/f'launch-{launch}'/name,audit['final_frame_sha256'][run] if launch==156 and name=='tex_out.bin' else None)
    chain=json.loads((attempt/'full-chain.json').read_text());binaries=set()
    for step in chain['steps']:
        cmd=step['command']
        for i,arg in enumerate(cmd[:-1]):
            if arg not in ('--split-attempt','--vit-attempt','--lib','--probe','--post-lib','--post-probe','--copy-lib'):continue
            path=Path(cmd[i+1]);path=path if path.is_absolute() else peer/path
            if arg=='--split-attempt':binaries.update(path/n for n in ('split.metallib','split_probe'))
            elif arg=='--vit-attempt':binaries.update(path/n for n in ('vit.metallib','vit_probe','repack_probe'))
            else:binaries.add(path)
    for source in sorted(binaries):preserve(source,Path('binaries')/sha(source)/source.name)
    for name in ('scripts/test-nr-full-chain.py','scripts/test-nr-middle.py','scripts/test-nr-body.py','scripts/nr-ptx-run-launch.py','scripts/ptx_warp_to_msl.py','src/nr_ptx/nr_ptx_rt.metal','src/nr_ptx/ptx_probe.mm','src/nr_split/split_probe.mm','src/nr_vit/repack_probe.mm'):
        preserve(peer/name,Path('observed-peer-source')/name)
    manifest={'schema':'neuralstudio.nr.strict-reference-archive.v1','strict_audit_root':root,'original_attempt':str(attempt),
              'peer_head_observed':subprocess.check_output(['git','rev-parse','HEAD'],cwd=peer,text=True).strip(),
              'inventory':inventory,'total_bytes':sum(x['bytes'] for x in inventory),'runtime_input_use_forbidden':True,
              'source_snapshot_scope':'Sources observed at archive time; compiled binaries are separately pinned. Do not infer these sources were frozen before the peer run.',
              'result':'pass'}
    manifest['receipt_root_sha256']=canonical(manifest);target=a.output/'archive.json';target.write_text(json.dumps(manifest,indent=2)+'\n')
    for path in a.output.rglob('*'):
        if path.is_file():path.chmod(0o444)
    print(json.dumps({'root':manifest['receipt_root_sha256'],'files':len(inventory),'bytes':manifest['total_bytes'],'result':'pass'},indent=2))
if __name__=='__main__':main()
