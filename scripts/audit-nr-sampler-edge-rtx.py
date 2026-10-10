#!/usr/bin/env python3
"""Audit small retrieved RTX payloads; full texture readbacks remain on the rig."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def need(ok,why):
    if not ok:raise ValueError(why)
PINS={'nr_sampler_edge_oracle.cu':'018524e2cc4f108b5b2bceb4a282fc804d3817b745bb43c14d1e936505f56b8a','nr_sampler_edge_inputs.h':'827da78ed8f5a980c7544124ca65238b0b1517c6ff820c8584791174c025f256','nr_finite_half_texture_oracle.cu':'19aeb1a1339636acd57513e9ae7371c03a255f0c7e57ae2c0eb667f9b1dc7e5f','run-nr-sampler-edge-rtx.ps1':'080e59ae2c643f3ab65e69d4f09f7cfd94b4e0184fb0b5fe83f67076f32533e5'}
def audit(root,fixture,manifest_sha):
    manifest=root/'job-receipt.json';need(sha(manifest)==manifest_sha,'verified transport manifest identity');j=read(manifest);f=read(fixture/'fixture.json')
    need(j['schema']=='neuralstudio.nr.sampler-edge-job.v1' and j['host']=='NICO-RIG','job identity')
    need(j['gpu']=='NVIDIA GeForce RTX 5090, 616.64, 32607 MiB','observed device/driver pin')
    need(j['sources_before']==j['sources_after']==PINS,'immutable staged sources')
    need('sm_120' in j['compile_command'] and any('13.3' in s for s in j['nvcc']),'compiler/target')
    expected=[('a5-a','a5'),('a5-b','a5'),('5a-a','5a'),('5a-b','5a')]
    need([(r['label'],r['poison']) for r in j['runs']]==expected and all(r['exit_code']==0 for r in j['runs']),'complete fresh-run inventory')
    need(f['input_sha256']=='98bb7f6defd425955e46f043c722a397453d70cb4812fe71c28cadcfe6624093','synthetic image identity')
    need(f['queries_sha256']=='2f59a69b4ce32cd8c18bbee570d8d69fbfe37a2b05f3aa885436ea4372012b1a' and f['queries']==1078 and len(f['cases'])==22,'query/case inventory')
    def artifact(rel,local=True):
        record=j['artifacts'][rel]
        if local:need((root/rel).stat().st_size==record['bytes'] and sha(root/rel)==record['sha256'],'retrieved identity '+rel)
        return record
    pids=set();data=[];observed=[]
    for label,pz in expected:
        for name in ('run.json','queries.bin','samples.bin','coverage.bin'):artifact(label+'/'+name)
        artifact(label+'.log');r=read(root/label/'run.json')
        need(r['schema']=='neuralstudio.nr.sampler-edge-oracle.v1' and r['result']=='pass','runtime completion')
        need(r['pid'] not in pids and r['pid']>0,'fresh distinct process');pids.add(r['pid'])
        need(r['poison']==int(pz,16) and (r['width'],r['height'],r['cases'],r['queries'])==(1920,1080,22,1078),'runtime configuration')
        need(r['coverage_exactly_one'] and r['guards_intact'] and r['input_immutable'],'runtime safety checks')
        need((root/label/'queries.bin').read_bytes()==(fixture/'queries.bin').read_bytes(),'exact independently regenerated queries')
        coverage=(root/label/'coverage.bin').read_bytes();need(len(coverage)==1078*4 and all(x[0]==1 for x in struct.iter_unpack('<I',coverage)),'raw exactly-once coverage')
        for name in ('input.rgba16le','input-after.rgba16le'):
            a=artifact(label+'/'+name,False);need(a['bytes']==16588800 and a['sha256']==f['input_sha256'],'remote raw texture identity')
        samples=(root/label/'samples.bin').read_bytes();need(len(samples)==1078*16,'full output write extent');data.append(samples)
        observed.append({'label':label,'pid':r['pid'],'poison':r['poison'],'samples_sha256':sha(root/label/'samples.bin')})
    need(all(d==data[0] for d in data),'fresh-process and two-poison output invariance')
    artifact('compile.log')
    rows=[];words=[v[0] for v in struct.iter_unpack('<I',data[0])]
    for i,c in enumerate(f['cases']):
        v=words[i*49*4:(i+1)*49*4];is_nan=lambda b:(b&0x7f800000)==0x7f800000 and (b&0x7fffff)!=0
        rows.append({'case':c['name'],'values':len(v),'nan':sum(map(is_nan,v)),'infinity':sum((x&0x7fffffff)==0x7f800000 for x in v),'negative_zero':v.count(0x80000000),'positive_zero':v.count(0),'unique_bits':[f'{x:08x}' for x in sorted(set(v))]})
    return {'schema':'neuralstudio.nr.sampler-edge-rtx-audit.v1','result':'pass_repeated_rtx_primitive_execution','manifest_sha256':manifest_sha,'device':j['gpu'],'sources':PINS,'executable':artifact('nr_sampler_edge_oracle.exe',False),'runs':observed,'queries_per_process':1078,'values_per_process':4312,'cases':rows,'rtx_mac_parity':False,'full_model_acceptance':False,'limits':['Full raw texture readbacks retained remotely; their remote SHA hashes match the independently regenerated CPU input.','Canary integrity is checked by pinned oracle code; raw coverage payloads are also independently scanned.','Executable SHA recorded after the job, not separately before each process.','No Metal output, nonfinite-coordinate behavior, other-format generality or performance claim.']}
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--fixture',type=Path,required=True);p.add_argument('--manifest-sha',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();need(not a.out.exists(),'preserve earlier audit')
    try:r=audit(a.root,a.fixture,a.manifest_sha)
    except Exception as e:r={'schema':'neuralstudio.nr.sampler-edge-rtx-audit.v1','result':'fail','error':str(e)}
    r['verifier_sha256']=sha(Path(__file__));r['receipt_root_sha256']=hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest();a.out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='cases'},indent=2));return int(r['result']=='fail')
if __name__=='__main__':raise SystemExit(main())
