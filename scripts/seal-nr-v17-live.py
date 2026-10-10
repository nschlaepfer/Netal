#!/usr/bin/env python3
"""Seal the eight independently checked v17 live cases without scope promotion."""
import argparse,hashlib,json
from pathlib import Path
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canonical(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
ROOTS={
 'black3':'5bd81e7b5094114f3a1efd55a1a952f9abfaea1d3e2aaacda46e1a3326d92a50',
 'checker3':'f12b395bef7441c1301749ca37d9cdf9326bbf4af5060df715827599483a57c4',
 'dense':'c8f20b1b04ef20018bc5002e4a1bf720828849c63ccf983aaf58c53778b7d9cd',
 'images3':'d29d7af5a072479a446e90ae114bfc176d48adf13900cc68aa7faf60d400e8d2',
 'midreset8':'5ae6608089b51a78f2c8f12463c105e1cd244634507de973d61fb78e5eb6f271',
 'uniform3':'95df8deac6f4018ad94b7150014ebaa76750ea1b61b2f48e9f242eb40c317114',
 'spatial3':'495d225c4217f4236777324e084107261d68de68abc82196dff23cdc5cc1b06d',
 'combined3':'b4babe31803872ce77b44b98e4b1a5fe7b72f87cee653e1191f8b783121ac1dd'}
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();assert not a.out.exists();repo=Path(__file__).resolve().parents[1]
    rows=[];pids=set();artifacts={};frames=0;nb=0;ports=0
    for name,expected in ROOTS.items():
        suffix='v2' if name=='black3' else 'v1'
        path=repo/'build/nr_resident_review'/f'live-v17-{name}-{suffix}.json'
        r=json.loads(path.read_text());b=dict(r);h=b.pop('receipt_root_sha256');assert canonical(b)==h==expected
        assert r['body_version']==17 and r['configuration']==name and r['result']=='pass_live_outputs_ports_history_with_explicit_omissions'
        assert not r['strong_live_intermediate_acceptance'] and not r['production_promotion'] and not r['full_goal_complete']
        assert r['verifier_sha256']==sha(repo/'scripts/audit-nr-resident-live.py')=='be1be337fa9a49b362939855b7c891fadc1de83be25f9c4904978508b94aaede'
        ps={f['pid'] for f in r['runs']};assert len(ps)==4 and not ps&pids;pids|=ps
        for f,h0 in r['artifacts'].items():
            assert f not in artifacts or artifacts[f]==h0;artifacts[f]=h0
        frames+=len(r['runs']);nb+=r['actual_final_post_history_bytes_checked'];ports+=r['encoder_boundary_hashes_checked']
        rows.append({'case':name,'root':h,'frames':len(r['runs']),'audit_sha256':sha(path)})
    for path,h in artifacts.items():assert sha(path)==h,'artifact drift: '+path
    assert (len(pids),frames,ports,nb)==(32,108,648,5374771200)
    r={'schema':'neuralstudio.nr.v17-live-set.v1','result':'pass_bounded_live_outputs_ports_history','cases':rows,'fresh_processes':32,'frames':108,'RTX_encoder_port_hashes_checked':648,'actual_output_post_history_bytes_checked':nb,'artifact_count':len(artifacts),'full_strong_checkpoint_suite_accepted':False,'default_promoted':False,'speedup_accepted':False,'full_goal_complete':False,'limits':['Eight fixed input/control recipes only.','Production mode retains coverage/digests but omits strong intermediate snapshots; full checkpoint suite is a separate gate.','No use of old cumulative frame timers as latency; corrected benchmark remains separate.'],'verifier_sha256':sha(__file__)}
    r['receipt_root_sha256']=canonical(r);a.out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
if __name__=='__main__':main()
