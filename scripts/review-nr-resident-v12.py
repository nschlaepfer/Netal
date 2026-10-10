#!/usr/bin/env python3
"""Read-only resident source/checkpoint review; never labels archived CLI input live."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read(p):return json.loads(p.read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument('--peer',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    runtime=a.peer/'build/nr_body/runtime-v12/source/src/nr_body/nr_body_runtime_v12.mm'
    validator=a.peer/'src/nr_body/nr_body_validate.metal';pins={str(x):sha(x) for x in (runtime,validator)}
    source=runtime.read_text();vs=validator.read_text()
    assert '(i << 8u) | min(c, 255u)' in vs
    packed=lambda i,c:((i<<8)|min(c,255))&0xffffffff
    indices=[7,(1<<24)+7];keys=[packed(i,2) for i in indices]
    assert keys[0]==keys[1] and indices[0]!=indices[1]
    old_path=a.peer/'build/nr_full/temporal-sixteen-halffma-v1/replica-3-5a/diag/frame-0/chain-run.json'
    old=read(old_path);pins[str(old_path)]=sha(old_path)
    wanted={(x['launch'],k):v for x in old['launches'] if x['kind']!='ptx' for k,v in x['outputs'].items()}
    assert len(wanted)==125 and old['poison']==90
    cases=[]
    for name in ('checked-no-fault-faultbuild','checked-no-fault-prodbuild'):
        d=a.peer/'build/nr_body/v12-evidence-v1/cli'/name;path=d/'chain-run.json';r=read(path);manifest=read(d/'manifest.json')
        assert manifest['code']==0 and r['poison']==90 and r['command_buffers']==1
        got={(x['launch'],k[:-7]):v for x in r['launches'] for k,v in x.get('digests',{}).items() if k.endswith('_sha256')}
        assert wanted.keys()<=got.keys() and all(wanted[k]==got[k] for k in wanted)
        output=d/'tex_out_156.bin';assert sha(output)==r['final_surface_sha256']=='a07835db4be8469dad1e3d3befcc495be396790f89408be62ddb8a5a710585a2'
        pins[str(path)]=sha(path);pins[str(d/'manifest.json')]=sha(d/'manifest.json');pins[str(output)]=sha(output)
        cases.append({'case':name,'pid':r['pid'],'matching_middle_checkpoint_hashes':125,'extra_checkpoint_keys':[list(k) for k in got.keys()-wanted.keys()],'actual_final_sha256':sha(output),'role':'archived-encoder-input component diagnostic, not live complete-model inference'})
    pre_call=next((i+1,l.strip()) for i,l in enumerate(source.splitlines()) if 'digest("decoder pre-image (restaged pool)"' in l)
    pre_sha=next((i+1,l.strip()) for i,l in enumerate(source.splitlines()) if 'res.decoder_pre_image_sha256 = Sha(img.data(), img.size())' in l)
    assert 'nullptr, true, false' in pre_call[1]
    findings=[
      {'id':'preimage-is-reconstructed','kind':'acceptance_gap','evidence_lines':[pre_call[0],pre_sha[0]],'detail':'Actual GPU restage is not snapshotted. Reported decoder_pre_image_sha256 hashes a CPU reconstruction; only a 64-bit digest binds it to GPU bytes. Restore a strong actual-byte snapshot/check in the new path before claiming the original boundary guarantee.'},
      {'id':'decoder-pool-checkpoints-weakened','kind':'acceptance_gap','detail':'Checked v12 omits the 24 actual decoder pool SHA-256 snapshots, retaining only 64-bit digests. Availability of the separate v11 runner does not prove those checkpoints for the batched v12 path.'},
      {'id':'first-failure-index-wrap','kind':'diagnostic_bug','detail':'Packing (element << 8) into uint truncates element indices above 2^24-1. The 212,519,936-byte pool has 53,129,984 words, so valid decoder indices exceed that range. Detection flags still work, but the reported position/order can be wrong.','indices':indices,'same_packed_keys':keys,'reported_index':keys[1]>>8},
      {'id':'additive-api-not-yet-present','kind':'incomplete_contract','detail':'This source implements a synchronous nr_body_run_frame with one commit/wait; it does not yet define the planned public encode/complete APIs. Label this accurately rather than treating phase comments as an implemented interface.'},
    ]
    assert 'int nr_body_encode_frame(' not in source and 'int nr_body_complete_frame(' not in source
    for path,h in pins.items():assert sha(Path(path))==h,'review source changed'
    r={'schema':'neuralstudio.nr.resident-v12-independent-review.v1','result':'component_evidence_with_unresolved_acceptance_gaps','cases':cases,'findings':findings,'pins':pins,
       'GPU_executed':False,'peer_files_changed':False,'resident_promoted':False,'full_model_parity_claimed':False,'verifier_sha256':sha(Path(__file__))}
    r['receipt_root_sha256']=canonical(r);assert not a.output.exists();a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='pins'},indent=2))


if __name__=='__main__':main()
