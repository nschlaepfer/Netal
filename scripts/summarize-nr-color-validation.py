#!/usr/bin/env python3
"""Seal the four independently verified cases against the original graph."""
import argparse
import hashlib
import json
from pathlib import Path

def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def original_roots(capture):
    files={x['path']:x for x in read(capture/'artifact-inventory.json')['files']}
    events=[json.loads(x) for x in (capture/'rtx-cuda-trace.jsonl').read_text(encoding='utf-8-sig').splitlines() if x.strip()]
    modules={e['module_id']:files['cuda-dumps/'+e['dump']]['sha256'] for e in events if e.get('call')=='cuModuleLoadData'}
    functions={e['function_id']:e for e in events if e.get('call')=='cuModuleGetFunction'}
    graph=[]
    for e in events:
        if e.get('call')!='cuLaunchKernel':continue
        f=functions[e['function_id']];assert f['name']==e['name']
        graph.append({'module_sha256':modules[f['module_id']],'kernel':e['name'],**{k:e[k] for k in ('grid','block','shared_bytes','argument_bytes')}})
    weights=[(k,v['bytes'],v['sha256']) for k,v in sorted(files.items()) if k.startswith('cuda-dumps/transfer-')]
    assert len(graph)==156 and len(weights)==153
    return canonical(graph),canonical(weights)
def main():
    p=argparse.ArgumentParser();p.add_argument('--case-directories',nargs=4,type=Path,required=True);p.add_argument('--original-capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
    graph,weights=original_roots(a.original_capture);pids=set();cases=[];outputs=set()
    for i,path in enumerate(a.case_directories):
        r=read(path/'verification.json');b=dict(r);h=b.pop('receipt_root_sha256');assert canonical(b)==h and r['result']=='pass' and r['case']==i
        assert r['fresh_processes']==4 and r['output_poison_invariance'] and len(r['runs'])==4
        for run in r['runs']:
            assert run['pid'] not in pids;pids.add(run['pid'])
            assert run['graph_root']==graph and run['weight_root']==weights
        output=r['runs'][0]['output_sha256'];assert all(x['output_sha256']==output for x in r['runs']);outputs.add(output)
        cases.append({'case':i,'directory':str(path.resolve()),'verification_root':h,'input_sha256':r['runs'][0]['input_sha256'],'output_sha256':output})
    assert len(pids)==16 and len(outputs)==4
    result={'schema':'neuralstudio.nr.varied-color-rtx-validation.v1','cases':cases,'fresh_processes':16,'original_selected_graph_root':graph,'original_weight_root':weights,
      'result':'pass','output_poison_invariance':True,'varied_input_mac_parity':False,'temporal_or_descriptor_coverage':False,'gpu_input_after_immutability_proven':False,'internal_pool_write_coverage_proven':False,
      'scope':'Four fixed-size reset color cases; four RTX processes each. Downloaded evidence independently verified and selected graph/weight identities equal the original capture. Not full acceptance of new colors on Metal or ANE.'}
    result['receipt_root_sha256']=canonical(result);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
