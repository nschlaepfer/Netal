#!/usr/bin/env python3
"""Independent four-process live-history encoder audit; not full-model acceptance."""
import argparse
import hashlib
import json
from pathlib import Path
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canonical(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read(p):return json.loads(p.read_text())
def sealed(p):
    r=read(p);b=dict(r);h=b.pop('receipt_root_sha256');assert canonical(b)==h;return r
def audit(attempts):
    repo=Path(__file__).resolve().parents[1];reference=sealed(repo/'build/nr_motion_reference/matrix-audit-v1.json');expected=reference['boundary_ports'][3]
    sizes={'block0-skip':70778880,'block22-middle':1105920,'block22-native':2211840,'block14-native':4423680,'block8-native':8847360,'block4-native':17694720}
    rows=[];pids=set();checked=0;fingerprints=set()
    for attempt,poison in zip(attempts,(165,165,90,90)):
        plan=sealed(attempt/'plan.json');observation=sealed(attempt/'observation.json');native=read(attempt/'run/run.json')
        assert observation['result']=='pass_temporal_encoder_boundaries' and observation['exit_code']==0 and observation['artifact_drift']==[]
        assert observation['plan_root']==plan['receipt_root_sha256'] and observation['native_receipt_sha256']==sha(attempt/'run/run.json')
        assert plan['rtx_reference_root']==reference['receipt_root_sha256'] and plan['history_source']=='live first full native inference -> GPU copy; no archived activation ingress'
        for p,h in plan['artifacts'].items():assert sha(Path(p))==h,'pinned artifact drift '+p
        assert native['pid'] not in pids and native['pid']>0;pids.add(native['pid'])
        assert native['sampler_variant']==3 and native['motion_sha256']==reference['motion_sha256'] and native['poison_first']==poison and native['poison_second']==(poison^255)
        assert native['first_encoder_code']==native['first_body_code']==native['temporal_encoder_code']==0 and not native['error']
        assert native['history_origin']=="GPU copy of this process's live first native generated frame" and native['archived_history_input'] is False and native['history_immutable'] is True
        assert native['first_encoder_guarded_outputs']==1039 and native['first_encoder_coverage_values']==5900107776
        assert native['first_body_checks']==241 and native['first_body_constants_unchanged'] and native['first_body_constants_rehashed']==153
        assert len(native['first_body_loaded'])==160
        for n,h in native['first_body_loaded'].items():
            p=n[4:] if n.startswith('lib:') else n[5:].rsplit(':',1)[0];assert plan['artifacts'][p]==h
        assert native['first_output_sha256']=='a07835db4be8469dad1e3d3befcc495be396790f89408be62ddb8a5a710585a2'
        assert sha(attempt/'run/first-live-output.rgba16le')==native['first_output_sha256']
        assert native['guarded_outputs_checked']==1040 and native['coverage_values_checked']==5935497216
        neg=native['negative_controls'];assert len(neg)==11 and [n['test'] for n in neg]==list(range(11)) and all(n['code']==2 and n['error'] for n in neg)
        assert set(native['ports'])==set(sizes)
        for n,size in sizes.items():
            p=attempt/'run'/(n+'.e4m3');meta=native['ports'][n];assert p.stat().st_size==meta['bytes']==size and meta['offset']==256 and meta['matches_rtx']
            assert sha(p)==meta['sha256']==expected[n];checked+=size
        features=attempt/'run/temporal-diagnostics/block0/features-shared.f16';assert features.stat().st_size==70778880 and sha(features)==native['feature_sha256'];fingerprints.add(native['feature_sha256'])
        assert native['full_temporal_model'] is False
        rows.append({'attempt':str(attempt.resolve()),'pid':native['pid'],'poison':native['poison_second'],'observation_root':observation['receipt_root_sha256'],'native_receipt_sha256':sha(attempt/'run/run.json')})
    assert len(rows)==4 and len(pids)==4 and len(fingerprints)==1
    return {'schema':'neuralstudio.nr.live-motion-encoder-matrix-audit.v1','result':'pass_bounded_live_motion_encoder','runs':rows,'fresh_processes':4,'live_first_frame_histories':4,'temporal_encoder_frames':4,'actual_boundary_bytes_checked':checked,'negative_controls':44,'feature_sha256':next(iter(fingerprints)),'rtx_reference_root':reference['receipt_root_sha256'],'archived_history_ingress':False,'full_temporal_model':False,'scope':'Four original-image/uniform fractional-motion(0.25,-0.5) frame1 encoder runs after live native frame0 history production, both poisons, full coverage/immutability/provenance checks. The second-frame body/output is not executed here. No general motion, long-sequence or performance claim.','verifier_sha256':sha(Path(__file__))}
def main():
    p=argparse.ArgumentParser();p.add_argument('--attempts',type=Path,nargs=4,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
    try:r=audit(a.attempts)
    except Exception as e:r={'schema':'neuralstudio.nr.live-motion-encoder-matrix-audit.v1','result':'fail','error':str(e)}
    r['receipt_root_sha256']=canonical(r);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
    if r['result']=='fail':raise SystemExit(3)
if __name__=='__main__':main()
