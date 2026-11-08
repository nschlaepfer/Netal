#!/usr/bin/env python3
"""Reproduce separate opt-in MPP/fast-half sidecars from a sealed SR runtime.

  python3 scripts/build-netal-sr-fast-pipelines.py --bundle SOURCE_RUNTIME \
    --variant fast-half-mlp --output NEW_SIDECAR_DIRECTORY

Build the matching library separately with build-netal-sr-mpp-kernels.py using
the same --variant. Copy these sidecars and that library into a separate runtime,
then reseal it. This tool copies no model/weight bytes and certifies no inference.
Existing default generators, source runtime, and FP8 pack recipes are unchanged.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/pwin_pipeline'))
import mpp_backend
import netal_sr_runtime_manifest as runtime
from netal_sr_layouts import profile

MODEL='DLSS_SR_Feature1_Quality_K'
LAYERS=('enc0','enc1','enc2','enc3','enc4','dec5','dec4','dec3','dec2','dec1','dec0')
SHIFTED={'enc3','enc4','dec5','dec4','dec3'}
ENTRY=re.compile(r'dltss_pwin_(enc[0-4]|dec[0-5])_layer(?:\.phase([1-7]))?\.json')
VARIANTS=('direct-mlp','fast-half-mlp')


def sha(data):return hashlib.sha256(data).hexdigest()


def unique_object(pairs):
    out={}
    for key,value in pairs:
        if key in out:raise ValueError('Duplicate JSON key')
        out[key]=value
    return out


def read_json(path):return json.loads(Path(path).read_text(),object_pairs_hook=unique_object)


def checked(value,name):
    if type(value)is not int or not 0<=value<=0xffffffff:raise ValueError('Invalid u32 '+name)
    return value


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


def source_identity(bundle):
    if bundle.is_symlink()or not bundle.is_dir()or(bundle/'manifest.json').is_symlink():raise ValueError('Use a real sealed source runtime directory')
    manifest=read_json(bundle/'manifest.json')
    if manifest.get('schema')!=runtime.SCHEMA or manifest.get('model')!=MODEL:raise ValueError('Unsupported RuntimeManifest v2/model identity')
    runtime.verify_engine(bundle,manifest)
    inventory={x['path']:x for x in manifest['files']}
    if not runtime.REQUIRED<=inventory.keys():raise ValueError('Incomplete source runtime inventory')
    pins={}
    for name,(size,digest)in runtime.MODEL.items():
        data=(bundle/name).read_bytes();row=inventory.get(name,{})
        if len(data)!=size or sha(data)!=digest or row.get('bytes')!=size or row.get('sha256')!=digest:raise ValueError('Unsupported pinned model bytes: '+name)
        pins[name]={'bytes':size,'sha256':digest}
    weights=(bundle/'model/weights.bin').read_bytes()
    repacker=module('netal_fast_supported_repack',ROOT/'tools/pwin_pipeline/repack.py')
    layouts,weight_records={},{}
    for layer in LAYERS:
        spec_path=ROOT/f'tools/pwin_pipeline/spec-{layer}.json';spec_bytes=spec_path.read_bytes()
        packed,layout=repacker.repack(weights,json.loads(spec_bytes,object_pairs_hook=unique_object))
        relative=f'pipeline/{layer}-weights.bin';layout_path=f'pipeline/layouts/layout-{layer}.json'
        data=(bundle/relative).read_bytes();layout_bytes=(bundle/layout_path).read_bytes()
        if data!=packed or read_json(bundle/layout_path)!=layout:raise ValueError('Repacked source weights/layout disagree with supported model: '+layer)
        layouts[layer]=layout
        weight_records[layer]={'weights_path':relative,'bytes':len(data),'sha256':sha(data),
                               'layout_path':layout_path,'layout_sha256':sha(layout_bytes),
                               'spec_sha256':sha(spec_bytes)}
    return layouts,{'schema':runtime.SCHEMA,'asset_root_sha256':manifest['asset_root_sha256'],
                    'runtime_manifest_sha256':sha((bundle/'manifest.json').read_bytes()),
                    'model':MODEL,'model_pins':pins,'repacked_weights':weight_records}


def graph_files(bundle):
    result={}
    for path in sorted(bundle.rglob('pipeline/*.json')):
        if not ENTRY.fullmatch(path.name):continue
        relative=path.relative_to(bundle)
        parent=relative.parent.as_posix()
        if parent!='pipeline'and not re.fullmatch(r'profiles/[1-9][0-9]*x[1-9][0-9]*/pipeline',parent):raise ValueError('Unsupported sidecar location: '+str(relative))
        result[relative]=path
    if not result:raise ValueError('No canonical PWin sidecars')
    groups={}
    for relative in result:
        layer,phase=ENTRY.fullmatch(relative.name).groups()
        groups.setdefault(relative.parent,{}).setdefault(layer,set()).add(int(phase)if phase else 0)
    if Path('pipeline')not in groups:raise ValueError('Missing root graph')
    for group,layers in groups.items():
        if set(layers)!=set(LAYERS):raise ValueError('Incomplete eleven-layer graph: '+str(group))
        for layer,phases in layers.items():
            if phases!=(set(range(8))if layer in SHIFTED else{0}):raise ValueError('Incomplete or extra shift phases: '+str(group)+'/'+layer)
    # A stray disabled sidecar must not masquerade as complete ordinary coverage.
    if any(ENTRY.fullmatch(p.name[:-4])for p in bundle.rglob('*.json.off')):raise ValueError('Disabled PWin sidecars are unsupported')
    return result,groups


def canonical_graphs(groups,layouts):
    """Generate complete canonical descriptors from public formulas, not source JSON."""
    with tempfile.TemporaryDirectory(prefix='netal-fast-canonical-')as temporary:
        base=Path(temporary)/'bundle';layout_dir=Path(temporary)/'layouts';layout_dir.mkdir()
        dimensions={(1920,1080)}
        for group in groups:
            if group!=Path('pipeline'):dimensions.add(tuple(map(int,group.parts[1].split('x'))))
        geometry={}
        for width,height in sorted(dimensions):
            files,metadata=profile(width,height);geometry[f'profiles/{width}x{height}/pipeline']=metadata
            destinations=[base/f'profiles/{width}x{height}']+([base]if(width,height)==(1920,1080)else[])
            for destination in destinations:
                for relative,data in files.items():
                    path=destination/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        (base/'pwin/enc0-arguments-template.bin').write_bytes((base/'pwin/enc0/arguments-template.bin').read_bytes())
        geometry['pipeline']=geometry['profiles/1920x1080/pipeline']
        for layer,layout in layouts.items():(layout_dir/f'layout-{layer}.json').write_text(json.dumps(layout))
        env={k:v for k,v in os.environ.items()if not k.startswith('PWIN_')}
        result=subprocess.run([sys.executable,str(ROOT/'tools/pwin_pipeline/build-profile-pipelines.py'),str(base),
                               '--layout-dir',str(layout_dir)],cwd=ROOT,env=env,capture_output=True,text=True)
        if result.returncode:raise ValueError('Public canonical graph generation failed: '+result.stderr)
        graphs={}
        for group in groups:
            for path in(base/group).glob('*.json'):
                if ENTRY.fullmatch(path.name):graphs[path.relative_to(base)]=read_json(path)
        return graphs,geometry


def profile_admission(bundle,groups,geometry):
    """Match Session's actual profile allocations/templates to span geometry."""
    abi=module('netal_fast_boundary_abi',ROOT/'tools/netal_sr_abi.py')
    base={name:(bundle/'abi'/name).read_bytes()for name in abi.CONTRACTS}
    for name,data in base.items():
        if abi.canonicalize(name,data)!=data:raise ValueError('Noncanonical root boundary template: '+name)
    if base!=abi.for_geometry(base,(1920,1080),(1280,720)):
        raise ValueError('Root boundary template geometry mismatch')
    arenas,records={},{}
    equal=lambda a,b:json.dumps(a,sort_keys=True,allow_nan=False)==json.dumps(b,sort_keys=True,allow_nan=False)
    for group in groups:
        key=group.as_posix();metadata=geometry[key]
        if group==Path('pipeline'):
            files,_=profile(1920,1080);prefix=bundle
            arenas[key]=metadata['arenas']
            records[key]={'output':[1920,1080],'render_reference':[1280,720],'allocation_source':'fixed legacy Session geometry'}
        else:
            prefix=bundle/group.parent;manifest_path=prefix/'manifest.json';layout_path=prefix/'layout.json'
            if not manifest_path.is_file():raise ValueError('Profile manifest missing: '+str(group.parent))
            if not layout_path.is_file():raise ValueError('Profile DataTemplateLayout missing: '+str(group.parent))
            actual=read_json(manifest_path);layout=read_json(layout_path)
            if not isinstance(actual,dict)or not isinstance(layout,dict):raise ValueError('Invalid profile geometry metadata')
            reference=actual.get('render_reference');output=metadata['output']
            if(not isinstance(reference,list)or len(reference)!=2 or
                    any(type(v)is not int or not 1<=v<=output[i]for i,v in enumerate(reference))):
                raise ValueError('Invalid profile reference geometry')
            files,_=profile(*output)
            expected=abi.profile_manifest(metadata,reference,files)
            for field in('schema','output','render_reference','arenas','hiluma_input','hiluma_output','exposure_copy','stages'):
                if not equal(actual.get(field),expected[field]):raise ValueError('Profile manifest geometry mismatch: '+key+'/'+field)
            arrays=actual.get('arrays')
            if not isinstance(arrays,list)or any(not isinstance(a,dict)or type(a.get('id'))is not int for a in arrays):raise ValueError('Invalid profile internal arrays')
            ids=[a['id']for a in arrays]
            if len(ids)!=len(set(ids)):raise ValueError('Duplicate profile array identity')
            internal=sorted((a for a in arrays if 1<=a['id']<=6),key=lambda a:a['id'])
            if not equal(internal,expected['arrays'][:6]):raise ValueError('Profile internal array geometry mismatch')
            for field in('schema','output','arenas','stages'):
                if not equal(layout.get(field),metadata[field]):raise ValueError('Profile DataTemplateLayout geometry mismatch: '+key+'/'+field)
            if not equal(layout.get('render_reference'),reference):raise ValueError('Profile DataTemplateLayout reference geometry mismatch')
            templates=abi.for_geometry(base,output,reference)
            for name,data in templates.items():
                if (prefix/'abi'/name).read_bytes()!=data:raise ValueError('Profile boundary template geometry mismatch: '+key+'/'+name)
            # Use the checked actual Session allocations, not merely formulas.
            arenas[key]=actual['arenas']
            records[key]={'output':output,'render_reference':reference,
                          'manifest_sha256':sha(manifest_path.read_bytes()),'layout_sha256':sha(layout_path.read_bytes()),
                          'allocation_source':'verified actual profile manifest',
                          'boundary_templates':{name:sha(data)for name,data in templates.items()}}
        for relative,data in files.items():
            if (prefix/relative).read_bytes()!=data:raise ValueError('Profile PWin template geometry mismatch: '+key+'/'+relative)
        if group==Path('pipeline')and(bundle/'pwin/enc0-arguments-template.bin').read_bytes()!=files['pwin/enc0/arguments-template.bin']:
            raise ValueError('Root enc0 alias geometry mismatch')
        records[key]['pwin_geometry_sha256']={relative:sha(data)for relative,data in files.items()}
    return arenas,records


def half_mlp_graph(q):
    result=copy.deepcopy(q)
    for d in result['dispatches']:
        if d['function']=='pwin_mlp2_tokens':
            words=d['bytes'][0]['u32'];d['grid']=[(words[0]+63)//64,1,1];d['block']=[128,1,1]
    return result


def core(q):return {key:q[key]for key in('library','scratch_bytes','files','dispatches')}


def binding_map(d):
    selected={}
    for b in d['buffers']:
        index=checked(b['index'],'buffer index')
        if index>16 or index in selected:raise ValueError('Invalid or duplicate buffer binding')
        checked(b['offset'],'buffer offset');selected[index]=b
    return selected


def span(buffers,index,elements,size,sizes):
    if type(elements)is not int or elements<=0 or elements-1>0xffffffff:raise ValueError('GPU uint32 element addressing overflow')
    b=buffers[index];offset=b['offset'];length=sizes[b['name']]
    if offset%size or offset+elements*size>length:raise ValueError('Transformed GEMM/MLP buffer span exceeds allocation')


def validate_transformed(layer,q,weight_bytes,arenas):
    if q['files']!=[{'name':'pweights','path':f'pipeline/{layer}-weights.bin'}]:raise ValueError('Unsupported original file bindings')
    checked(q['scratch_bytes'],'scratch size')
    sizes={'scratch':q['scratch_bytes'],'pweights':weight_bytes,**{f'arena{k}':v for k,v in arenas.items()}}
    gemms=mlps=0
    for d in q['dispatches']:
        buffers=binding_map(d);byte_slots=set()
        for item in d.get('bytes',[]):
            index=checked(item['index'],'bytes index')
            if index>16 or index in byte_slots or index in buffers:raise ValueError('Invalid/duplicate/conflicting byte binding')
            byte_slots.add(index)
            for word in item['u32']:checked(word,'parameter')
        for b in buffers.values():
            if b['name']not in sizes or b['offset']>=sizes[b['name']]:raise ValueError('Buffer binding outside known allocation')
        for value in d.get('constants',{}).values():checked(value,'function constant')
        if d['function']=='pwin_gemm_mpp':
            gemms+=1
            if set(buffers)!={0,1,2,3,5,6}or len(d.get('bytes',[]))!=1 or byte_slots!={4}or set(d['constants'])!={'0','1'}:raise ValueError('Unsupported GEMM ABI')
            p=d['bytes'][0]['u32'];mpp_backend.validate_params(p)
            t,k,n,xs,ys,cs,epi,cmode=p[:8];ws=p[11]or n;limit=p[12]or n
            if d['constants']!={'0':k,'1':n}or d['grid']!=[(t+63)//64,(limit+31)//32,1]or d['block']!=[128,1,1]:raise ValueError('Incorrect GEMM constants/dispatch')
            span(buffers,0,(t-1)*xs+k,2,sizes);span(buffers,1,(k-1)*ws+n,2,sizes);span(buffers,3,(t-1)*ys+limit,2,sizes)
            if cmode==0:span(buffers,2,15*cs+n,4,sizes)
            else:span(buffers,5,(t-1)*cs+n,2,sizes);span(buffers,6,n,2,sizes)
            if epi==3:span(buffers,5,(t-1)*p[13]+limit,2,sizes)
        elif d['function']=='pwin_mlp2_tokens':
            mlps+=1;c=d['constants'];p=d['bytes'][0]['u32'];required={'0','1','2','3','7','14'}
            if set(buffers)!={0,1,2,3,4,5,6,8}or len(d.get('bytes',[]))!=1 or byte_slots!={7}or not required<=c.keys()or set(c)-required-{'15'}or len(p)!=8:raise ValueError('Unsupported half MLP ABI')
            k,n,xs,cs,ys,hd=(c[str(i)]for i in(0,1,2,3,7,14));ws=c.get('15',0)or n;t=p[0]
            if((k,hd,n)not in((32,128,32),(64,256,64))or t==0 or p[2]==0 or p[1]not in(0,2,5,6)or p[4]not in(0,1)or p[5]>=k//16 or p[6:]!=[0,0]or xs<k or cs<n or ys<n or ws<n):raise ValueError('Unsupported half MLP parameters')
            if d['grid']!=[(t+63)//64,1,1]or d['block']!=[128,1,1]:raise ValueError('Incorrect half MLP dispatch')
            for index,elements,size in [(0,(t-1)*xs+k,2),(1,k*hd,2),(2,16*hd,4),(3,(hd-1)*ws+n,2),(4,(t-1)*cs+n,2),(5,n,2),(6,(t-1)*ys+n,2)]:span(buffers,index,elements,size,sizes)
            if p[4]:span(buffers,8,k,2,sizes)
        elif d['function'].startswith(('pwin_gemm','pwin_mlp')):raise ValueError('Unknown GEMM/MLP cannot silently remain')
    return gemms,mlps


def build(bundle,output,variant):
    bundle,output=Path(bundle).absolute(),Path(output).absolute()
    if variant not in VARIANTS:raise ValueError('Unsupported fast sidecar variant')
    if output.exists()or output.is_symlink():raise ValueError('Output exists; select a fresh sidecar directory')
    if output.resolve().is_relative_to(bundle.resolve()):raise ValueError('Output must be outside source runtime')
    layouts,identity=source_identity(bundle);paths,groups=graph_files(bundle)
    canonical,geometry=canonical_graphs(groups,layouts)
    admitted_arenas,profile_records=profile_admission(bundle,groups,geometry)
    outputs,records,counts={},[],{}
    for relative,path in paths.items():
        layer,phase=ENTRY.fullmatch(relative.name).groups();source=read_json(path);reference=canonical[relative]
        mapped=mpp_backend.convert_pipeline(reference);candidate=half_mlp_graph(mapped)
        if core(source)not in(core(reference),core(mapped),core(candidate)):raise ValueError('Source graph differs from public canonical descriptors: '+str(relative))
        # Convert actual GEMM3 source with the existing converter; already-MPP
        # sources have just been compared with its canonical functional result.
        q=half_mlp_graph(mpp_backend.convert_pipeline(source)if any(d['function']=='pwin_gemm3_tokens'for d in source['dispatches'])else source)
        gemms,mlps=validate_transformed(layer,q,identity['repacked_weights'][layer]['bytes'],admitted_arenas[relative.parent.as_posix()])
        q=core(q)  # Do not propagate unsupported source claims/experimental metadata.
        q.update({'library':'pwin_mpp_all','gemm_backend':'mpp','mlp_backend':'mpp-half','kernel_variant':variant,'mpp_variant':variant,
                  'candidate_approximate':True,'reference_exact':False,
                  'gemm_arithmetic':'full-K FP16 inputs/weights, FP32 dot, then real-unit bias'if variant=='fast-half-mlp'else'FP16 inputs/weights, FP32 accumulation with original handoff points',
                  'gemm_intermediate_fp16_handoffs':'deliberately omitted'if variant=='fast-half-mlp'else'preserved',
                  'mlp_intermediate_fp16_handoffs':'preserved','weights_modified':False})
        data=(json.dumps(q,indent=2)+'\n').encode();outputs[relative]=data
        records.append({'path':relative.as_posix(),'source_sha256':sha(path.read_bytes()),'output_sha256':sha(data),
                        'gemm_dispatches':gemms,'half_mlp_dispatches':mlps,'dispatches':len(q['dispatches'])})
        counts.setdefault(relative.parent,{})[layer]=max(counts.get(relative.parent,{}).get(layer,0),len(q['dispatches']))
    for group,per_layer in counts.items():
        if sum(per_layer.values())+3>=160:raise ValueError('Graph exceeds the per-frame argument-table budget: '+str(group))
    source_files=[Path(__file__),ROOT/'scripts/netal_sr_runtime_manifest.py',ROOT/'scripts/netal_sr_layouts.py',
                  ROOT/'tools/netal_sr_abi.py',
                  ROOT/'tools/pwin_pipeline/repack.py',ROOT/'tools/pwin_pipeline/mpp_backend.py',
                  ROOT/'tools/pwin_pipeline/build-profile-pipelines.py',*sorted((ROOT/'tools/pwin_pipeline').glob('make-*.py'))]
    receipt={'schema':'netal.sr.fast-pipelines.v1','variant':variant,'source':identity,'pipelines':records,'profile_admission':profile_records,
             'generator_sources':{p.relative_to(ROOT).as_posix():sha(p.read_bytes())for p in source_files},
             'expected_library_usage':{'path':'metallib/pwin_mpp_all.metallib','sidecar_library':'pwin_mpp_all',
                  'builder':'scripts/build-netal-sr-mpp-kernels.py','builder_variant':variant,
                  'gemm_entry':'pwin_gemm_mpp','mlp_entry':'pwin_mlp2_tokens'},
             'max_pwin_dispatches':max(sum(c.values())for c in counts.values()),'boundary_dispatch_allowance':3,
             'argument_table_limit':160,'validation_scope':'sealed source identity, deterministic weights/layouts, canonical operator graphs and transformed GEMM/MLP spans',
             'contains_model_weights':False,'weights_modified':False,'inference_validated':False,
             'image_quality_validated':False,'physical_device_validated':False,'gameplay_validated':False}
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.sr-fast-sidecars-',dir=output.parent)as temporary:
        stage=Path(temporary)/'output';stage.mkdir()
        for relative,data in outputs.items():
            destination=stage/relative;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(data)
        (stage/'fast-pipelines-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        if output.exists()or output.is_symlink():raise ValueError('Output appeared during generation; preserved')
        stage.rename(output)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--variant',choices=VARIANTS,required=True)
    args=parser.parse_args()
    try:result=build(args.bundle,args.output,args.variant)
    except(ValueError,OSError,KeyError,TypeError)as error:parser.error(str(error))
    print('result=pass pipelines='+str(len(result['pipelines']))+' variant='+result['variant']+' max_pwin_dispatches='+str(result['max_pwin_dispatches'])+' inference_validated=false')
