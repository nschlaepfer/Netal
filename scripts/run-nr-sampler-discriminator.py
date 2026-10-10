#!/usr/bin/env python3
"""Zero-motion ablation: fitted encoder sampler, unchanged body/post and inputs.

Run under the shared GPU lock. Reference hashes are comparison-only, read
after execution. All original and new failures remain immutable.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canonical(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sealed(p):
    r=read(p);b=dict(r);h=b.pop('receipt_root_sha256');assert canonical(b)==h;return r
def main():
    p=argparse.ArgumentParser();p.add_argument('--peer-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();peer=a.peer_root.resolve();out=a.output.resolve();repo=Path(__file__).resolve().parents[1]
    baseline=peer/'build/nr_full/temporal-sixteen-v1/replica-1-pilot/receipt.json';old=read(baseline);build=peer/'build/nr_full/v17t';enc=repo/'build/nr_encoder_temporal/spatial-v1';eb=sealed(enc/'build.json');base_enc=sealed(repo/'build/nr_encoder_temporal/build-v2-sass/build.json')
    assert eb['receipt_root_sha256']=='7c5531e9d15b68b4252060bc2e5daeaa55aa62fe155bcb3ec59fe89441ca5844'
    assert all(eb['assets'][n]==h for n,h in base_enc['assets'].items()),'old shaders changed'
    pins={str(baseline):sha(baseline)}
    for line in (build/'build-sha256.txt').read_text().splitlines():
        h,n=line.split(None,1);path=(peer/n.strip()).resolve();assert sha(path)==h;pins[str(path)]=h
    assert sha(build/'source/src/nr_body/nr_body_runtime_v11.mm')==sha(peer/'build/nr_full/v16t/source/src/nr_body/nr_body_runtime_v11.mm')
    assert sha(build/'source/src/nr_body/nr_full_frame.mm')==sha(peer/'build/nr_full/v16t/source/src/nr_body/nr_full_frame.mm')
    assert sha(build/'source/encoder/nr_encoder_runtime.o')==eb['object_sha256']
    for n,h in eb['assets'].items():assert sha(enc/'assets'/n)==h;pins[str(enc/'assets'/n)]=h
    for f in old['frames']:
        pins[f['input_file']]=sha(f['input_file'])
        for label,h in f['body']['loaded'].items():
            name=label[4:] if label.startswith('lib:') else label[5:].rsplit(':',1)[0] if label.startswith('file:') else label.split(':',1)[1]
            assert sha(name)==h;pins[name]=h
    cmd=list(old['command']);cmd[0]=str(build/'nr_full_frame');cmd[2]=str(enc/'assets')
    assert '--encoder-sampler-variant' not in cmd
    cmd+=['--encoder-sampler-variant','3']
    for flag,value in [('--receipt',str(out/'receipt.json')),('--save-dir',str(out)),('--diag-dir',str(out/'diag'))]:cmd[cmd.index(flag)+1]=value
    descriptor=Path(cmd[1]);descriptor=descriptor if descriptor.is_absolute() else peer/descriptor;pins[str(descriptor)]=sha(descriptor)
    motion=Path(cmd[cmd.index('--motion')+1]);motion=motion if motion.is_absolute() else peer/motion;pins[str(motion)]=sha(motion)
    assert sha(motion)=='788ae0147bdf979a6575938ca2d7d4403788588f7be2010f03776c968fd1ab49'
    assert not out.exists() and shutil.disk_usage(repo).free>13*1024**3;out.mkdir(parents=True)
    plan={'schema':'neuralstudio.nr.encoder-sampler-ablation-plan.v1','command':cmd,'artifacts':pins,'baseline_receipt_sha256':sha(baseline),'only_algorithm_change':'encoder temporal sampler0 hardware -> sampler3 fitted; body/post/harness source unchanged','started_utc':datetime.now(timezone.utc).isoformat()};plan['receipt_root_sha256']=canonical(plan);(out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    with (out/'native.log').open('x') as log:r=subprocess.run(cmd,cwd=peer,stdout=log,stderr=subprocess.STDOUT)
    drift=[name for name,h in pins.items() if sha(name)!=h]
    result={'schema':'neuralstudio.nr.encoder-sampler-ablation.v1','plan_root':plan['receipt_root_sha256'],'exit_code':r.returncode,'artifact_drift':drift,'ended_utc':datetime.now(timezone.utc).isoformat(),'full_acceptance':False}
    if (out/'receipt.json').exists():
        native=read(out/'receipt.json');ref=sealed(repo/'build/nr_sixteen/receipts-v1/continuous-a5-a/continuous-a5-a-reference.json');mapping={'entry':'block22-middle','skip_c256':'block22-native','skip_c128':'block14-native','skip_c64':'block8-native','skip_c32':'block4-native','block0_skip':'block0-skip'};rows=[]
        for i,f in enumerate(native['frames']):
            if 'body' not in f or 'regions' not in f:rows.append({'frame':i,'execution_failure':f});continue
            actual=sha(out/f'final-{i}.rgba16le');rows.append({'frame':i,'final_sha256':actual,'final_matches_RTX':actual==ref['outputs'][i],'bad_encoder_ports':[n for n,k in mapping.items() if f['regions'][n]['sha256']!=ref['boundaries'][2*i+1]['ports'][k]]})
        result.update({'pid':native['pid'],'native_receipt_sha256':sha(out/'receipt.json'),'comparison_reference_root':ref['receipt_root_sha256'],'frames':rows})
    result['receipt_root_sha256']=canonical(result);(out/'observation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    if r.returncode or drift:raise SystemExit(3)
if __name__=='__main__':main()
