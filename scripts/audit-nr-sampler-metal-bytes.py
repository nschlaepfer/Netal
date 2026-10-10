#!/usr/bin/env python3
"""Independently verify retained primitive outputs; explicitly scope driver gaps."""
import argparse,hashlib,json
from pathlib import Path
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def need(v,msg):
    if not v:raise ValueError(msg)
def main():
    p=argparse.ArgumentParser();p.add_argument('--peer',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--guarded',action='store_true');a=p.parse_args();need(not a.out.exists(),'preserve audit')
    peer=a.peer.resolve();repo=Path(__file__).resolve().parents[1];d=peer/('build/nr_temporal/sampler-edge-metal-v3' if a.guarded else 'build/nr_temporal/sampler-edge-metal-v1');r=json.loads((d/'receipt.json').read_text());pins={}
    def observe(path):
        h=sha(path);pins[str(path)]=h;return h
    need(r['schema']==('neuralstudio.nr.sampler-edge-metal.v2' if a.guarded else 'neuralstudio.nr.sampler-edge-metal.v1') and r['pass'],'reported component pass')
    build=peer/r['probe_build'];need(build==peer/'build/nr_temporal/tex-probe-metal-v4-nonfinite','reviewed build')
    for line in r['build_sha256'].splitlines():
        h,path=line.split(None,1);full=(peer/path.strip()).resolve();need(full.is_relative_to(build) and observe(full)==h,'retained build pin')
    need(observe(build/'probe.metallib')=='d920ba9a813d5292f8f9f9fadaefe5b400ebc92cd6f47143b40bfa77d0b5f2f5','sampler library')
    need(observe(build/'nr_tex_probe')=='5d97d361f83467c066e7bd8b9c4c3c7bdca0e1ce9d20dd8a36b20077ce3e7a6f','probe driver')
    need(observe(build/'source/nr_tex_sample.metal')=='34822e9e9a5f8f058a71b073963542dfaafce1928c4b1cdd219c29f212de5966','reviewed sampler source')
    need(observe(d/'edge-atlas.rgba16le')==r['atlas_sha256']=='98bb7f6defd425955e46f043c722a397453d70cb4812fe71c28cadcfe6624093','actual atlas')
    need(observe(d/'edge-queries.bin')==r['queries_sha256']=='2f59a69b4ce32cd8c18bbee570d8d69fbfe37a2b05f3aa885436ea4372012b1a','actual query file')
    expected=repo/'build/nr_sampler_edges/rtx-v1/a5-a/samples.bin';want=observe(expected);need(want=='4584ac72a39a59f660d345d3cb7c6cc4d5146ef5325bccd7db24d20b835e84cf','RTX oracle pin')
    need(len(r['edge_processes'])==4,'complete edge inventory');rows=[]
    for i,(entry,poison) in enumerate(zip(r['edge_processes'],('a5','a5','5a','5a'))):
        path=d/f'edge-samples-{i}-{poison}.bin'
        need(entry['process']==i and entry['poison']==poison and entry['rc']==0 and entry['values']==4312 and entry['unequal_to_rtx']==0,'edge run metadata')
        need(path.stat().st_size==17248 and observe(path)==entry['metal_samples_sha256']==want,'raw edge bits')
        rows.append({'kind':'edge','run':i,'poison':poison,'values':4312,'bytes':17248})
    regressions=[(kind,tag) for kind in ('step','finite-half') for tag in ('a5-a','a5-b','5a-a','5a-b')]
    need([x['capture'] for x in r['finite_regressions']]==[f'{k} {t}' for k,t in regressions],'complete finite regression inventory')
    for entry,(kind,tag) in zip(r['finite_regressions'],regressions):
        cap=repo/'build'/('nr_tex_micro' if kind=='step' else 'nr_tex_finite_half')/tag/'samples.bin';path=d/f'{kind}-{tag}-metal.bin';n=1212 if kind=='step' else 5524
        need(entry['rc']==0 and entry['unequal']==0 and entry['values']==n,'finite metadata')
        need(path.stat().st_size==cap.stat().st_size==n*4 and observe(path)==observe(cap)==entry['metal_samples_sha256']==entry['rtx_samples_sha256'],'raw finite regression bytes')
        rows.append({'kind':kind,'run':tag,'values':n,'bytes':n*4})
    observe(d/'receipt.json')
    process_pids=set()
    if a.guarded:
        need(r['driver']=='build/nr_temporal/tex-probe-driver-v2/nr_tex_probe_v2' and observe(peer/r['driver'])==r['driver_sha256']=='23796066baad179aa5b6c25f8718d8bb10ed2c808b0af220b931352e0c345713','guarded driver identity')
        for line in (peer/'build/nr_temporal/tex-probe-driver-v2/build-sha256.txt').read_text().splitlines():
            h,relative=line.split(None,1);p0=(peer/relative.strip()).resolve();need((p0.is_relative_to(peer/'build/nr_temporal/tex-probe-driver-v2') or p0==build/'probe.metallib') and observe(p0)==h,'guarded driver source/build')
        need(observe(peer/'build/nr_temporal/tex-probe-driver-v2/nr_tex_probe_v2.mm')=='b0e2543d6c969dba1cfbd116839fdaac5c982255f4f42658b44b0dd55b53094f','reviewed guarded driver source')
        for row in rows:
            if row['kind']=='edge':
                poison=row['poison'];sample=d/f'edge-samples-{row["run"]}-{poison}.bin';query=d/'edge-queries.bin';input_file=d/'edge-atlas.rgba16le'
            else:
                kind,tag=row['kind'],row['run'];poison='5a' if tag.startswith('5a') else 'a5';sample=d/f'{kind}-{tag}-metal.bin';query=repo/'build'/('nr_tex_micro' if kind=='step' else 'nr_tex_finite_half')/tag/'queries.bin';input_file=d/f'{kind}.rgba16le'
            rp=Path(str(sample)+'.receipt.json');pr=json.loads(rp.read_text());observe(rp)
            need(pr['schema']=='neuralstudio.nr.tex-probe-process.v2' and pr['pid']>0 and pr['pid'] not in process_pids,'distinct guarded process');process_pids.add(pr['pid'])
            need(pr['poison']==poison and pr['queries']==row['values']//4 and pr['pass'],'process configuration')
            need(pr['command_buffer_completed'] and pr['command_buffer_error']=='' and pr['redzones_intact']=={'queries':True,'samples':True,'coverage':True},'GPU completion and redzones')
            need(pr['queries_unchanged'] and pr['coverage_not_exactly_once']==0 and pr['input_texture_unchanged'] and pr['sample_words_still_poison']==0,'write coverage and input immutability')
            need(pr['library_sha256']==observe(build/'probe.metallib') and (peer/pr['library']).resolve()==build/'probe.metallib','per-process library identity')
            need(pr['queries_sha256']==observe(query) and pr['input_sha256']==observe(input_file) and pr['samples_sha256']==observe(sample),'per-process actual input/output identities')
    for path,h in pins.items():need(sha(path)==h,'review-time drift')
    result={'schema':'neuralstudio.nr.sampler-metal-independent-bytes.v1','result':'pass_raw_primitive_byte_parity','rows':rows,'total_values_compared':sum(x['values'] for x in rows),'total_bytes_compared':sum(x['bytes'] for x in rows),'pins':pins,'full_model_acceptance':False,'strict_driver_gate_complete':False,'limits':['v1 records process ordinals, not OS PIDs/per-process source identities; wrapper source launches a fresh subprocess for each run.','Driver checks coverage and input texture equality but lacks redzone canaries, query-buffer immutability check and explicit command-buffer status rejection.','Primitive numerical evidence only; new post replay and connected model regression remain required before promotion.']}
    if a.guarded:
        result.update(result='pass_guarded_primitive_byte_parity',strict_driver_gate_complete=True,guarded_processes=len(process_pids),limits=['Guard/coverage verdicts are enforced by the pinned driver; process IDs, hashes and actual sample payloads independently checked.','Library bytes hashed before load; retained library/driver/build identities rechecked now, not a per-process executable before/after audit.','Primitive numerical/guard evidence only; new post replay and connected model regression remain required before promotion.'])
    result['verifier_sha256']=sha(__file__);result['receipt_root_sha256']=hashlib.sha256(json.dumps(result,sort_keys=True,separators=(',',':')).encode()).hexdigest();a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('rows','pins')},indent=2))
if __name__=='__main__':main()
