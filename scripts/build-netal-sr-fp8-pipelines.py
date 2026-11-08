#!/usr/bin/env python3
"""Generate separate approximate FP8 sidecars from a verified private weight pack.

No source bundle or weights are modified. Copy the returned sidecars and pack
payloads into a separate runtime, install the matching FP8 library, then reseal.
Quantization is an explicit dispatch; all timings must include that work.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'tools/pwin_pipeline'))
from netal_sr_fp8_pack import verify_pack, matrix_bytes, sha
from mpp_backend import convert_pipeline as mpp_pipeline, validate_params

KEEP_HALF = {'enc0.W1', 'dec0.W7'}
ENTRY = re.compile(r'dltss_pwin_(enc[0-4]|dec[0-5])_layer(?:\.phase([1-7]))?\.json')
LAYERS = {'enc0','enc1','enc2','enc3','enc4','dec5','dec4','dec3','dec2','dec1','dec0'}
SHIFTED = {'enc3','enc4','dec5','dec4','dec3'}


def align(value, multiple=256):
    return (value + multiple - 1) // multiple * multiple


def binding(dispatch, index):
    selected = [b for b in dispatch['buffers'] if b['index'] == index]
    if len(selected) != 1:
        raise ValueError(f'Missing or duplicate buffer {index}')
    return selected[0]


def tensor_for(layer, dispatch, manifest):
    p = dispatch['bytes'][0]['u32']
    validate_params(p)
    wb = binding(dispatch, 1)
    if wb['name'] != 'pweights':
        raise ValueError('FP8 conversion requires original pweights bindings')
    found = [t for t in manifest['tensors'] if t['layer'] == layer and
             t['source']['offset'] == wb['offset'] and t['shape'] == p[1:3] and
             t['source']['row_stride_elements'] == (p[11] or p[2])]
    if len(found) != 1:
        raise ValueError(f'No unique packed matrix for {layer}: offset={wb["offset"]}, K,N={p[1:3]}')
    return found[0]


def check_tensor(t):
    k, n = t['shape']
    w, s = t['payload'], t['scales']
    if t['precision'] != 'e4m3fn' or s is None:
        raise ValueError('Non-exempt GEMM is not E4M3FN')
    if (w['order'] != 'N,K' or w['padded_k'] != align(k, 32) or w['padded_n'] != align(n, 32) or
            w['shape'] != [w['padded_n'], w['padded_k']] or w['row_stride_bytes'] != align(k, 128) or
            w['bytes'] != w['padded_n'] * w['row_stride_bytes'] or w['offset'] % 128 or
            s['count'] != w['padded_n'] or s['bytes'] != s['count'] * 4 or s['offset'] % 128 or
            s['dtype'] != 'float32-le' or s['axis'] != 'output_column' or
            s['direction'] != 'decoded_value * dequant_scale'):
        raise ValueError('Packed FP8 shape, stride, scale axis or alignment is unsupported')


def check_original_scratch(q):
    legacy=q['scratch_bytes']
    if type(legacy) is not int or not 0 < legacy <= 0xffffffff:
        raise ValueError('Invalid original scratch reservation')
    for d in q['dispatches']:
        for b in d.get('buffers',[]):
            if b['name']=='scratch' and (type(b['offset']) is not int or not 0 <= b['offset'] < legacy):
                raise ValueError('Original scratch binding is outside its reservation')
        if d['function'] != 'pwin_gemm_mpp':continue
        p=d['bytes'][0]['u32'];validate_params(p)
        t,k,n,xs,ys,cs,epi,cmode=p[:8];limit=p[12]or n
        spans={0:2*((t-1)*xs+k),3:2*((t-1)*ys+limit),6:2*n}
        if cmode:spans[5]=2*((t-1)*cs+n)
        if epi==3:spans[5]=max(spans.get(5,0),2*((t-1)*p[13]+limit))
        for index,span in spans.items():
            b=binding(d,index)
            if b['name']=='scratch' and b['offset']+span > legacy:
                raise ValueError('Original GEMM scratch tensor exceeds its reservation')


def convert(layer, original, manifest, half_mlp=True):
    q = mpp_pipeline(original) if any(d['function'] == 'pwin_gemm3_tokens' for d in original['dispatches']) else json.loads(json.dumps(original))
    if q['library'] != 'pwin_mpp_all' or any(d['function'].startswith('pwin_quantize') for d in q['dispatches']):
        raise ValueError('Convert an unquantized Reference/Batched/MPP sidecar only once')
    if q.get('files') != [{'name':'pweights','path':f'pipeline/{layer}-weights.bin'}]:
        raise ValueError('Unsupported original file bindings')
    check_original_scratch(q)
    mapped = {}
    for index, d in enumerate(q['dispatches']):
        if d['function'] == 'pwin_gemm_mpp':
            t = tensor_for(layer, d, manifest)
            if t['id'] in KEEP_HALF:
                if t['precision'] != 'float16-le':
                    raise ValueError('Sensitive FP16 matrix was quantized')
            else:
                check_tensor(t)
                mapped[index] = t
        elif d['function'].startswith('pwin_gemm'):
            raise ValueError('Unknown GEMM cannot silently remain in FP8 candidate')
    if not mapped:
        raise ValueError('No FP8 GEMM in layer')
    legacy = q['scratch_bytes']
    aq_offset = align(legacy)
    aq_bytes = max(align(q['dispatches'][i]['bytes'][0]['u32'][0],64) * t['payload']['row_stride_bytes'] for i,t in mapped.items())
    scale_offset = align(aq_offset + aq_bytes)
    scale_bytes = max(align(q['dispatches'][i]['bytes'][0]['u32'][0],64) * 4 for i in mapped)
    q['scratch_bytes'] = align(scale_offset + scale_bytes)
    if q['scratch_bytes'] > 0xffffffff:
        raise ValueError('FP8 scratch addressing exceeds uint32')
    new, files, used, mlps, fallback = [], {f['name']:f for f in q['files']}, [], 0, []
    for index, d in enumerate(q['dispatches']):
        if d['function'] == 'pwin_mlp2_tokens' and half_mlp:
            p = d['bytes'][0]['u32']
            c = d['constants']
            if ((c['0'],c['14'],c['1']) not in ((32,128,32),(64,256,64)) or len(p) != 8 or
                    any(type(v) is not int or not 0 <= v <= 0xffffffff for v in p) or
                    p[0] == 0 or p[2] == 0 or p[1] not in (0,2,5,6) or p[4] not in (0,1) or
                    p[5] >= c['0']//16 or p[6:] != [0,0] or c['2'] < c['0'] or c['3'] < c['1'] or
                    c['7'] < c['1'] or (c.get('15',0) or c['1']) < c['1'] or
                    any(p[0]*c[i] > 0xffffffff for i in ('2','3','7'))):
                raise ValueError('Unsupported fused MLP shape/ABI')
            d['grid'], d['block'] = [(p[0]+63)//64,1,1], [128,1,1]
            mlps += 1
        if index not in mapped:
            if d['function'] == 'pwin_gemm_mpp':
                fallback.append(tensor_for(layer,d,manifest)['id'])
            new.append(d)
            continue
        t = mapped[index]
        p = d['bytes'][0]['u32'].copy()
        rows = align(p[0],64)
        stride = t['payload']['row_stride_bytes']
        if rows * stride > aq_bytes or rows * 4 > scale_bytes:
            raise ValueError('FP8 scratch span exceeds reservation')
        quant = {'function':'pwin_quantize_rows_e4m3','grid':[rows//4,1,1],'block':[128,1,1],
                 'buffers':[dict(binding(d,0)),{'index':1,'name':'scratch','offset':aq_offset},
                            {'index':2,'name':'scratch','offset':scale_offset}],
                 'bytes':[{'index':3,'u32':[p[0],p[1],p[3],stride,t['payload']['padded_k'],rows,0,0]}]}
        quant['buffers'][0]['index'] = 0
        p[3],p[11],p[15] = stride,stride,t['payload']['padded_k']
        qb = [dict(binding(d,i)) for i in (2,3,5,6)]
        # Binding names include the payload kind; each layer has a separate file.
        for name, field in [('pweights_fp8','payload'),('pweights_fp8_scales','scales')]:
            row={'name':name,'path':t[field]['file']}
            if name in files and files[name] != row:
                raise ValueError('Layer refers to multiple FP8 payload files')
            files[name]=row
        qb += [{'index':0,'name':'scratch','offset':aq_offset},
               {'index':1,'name':'pweights_fp8','offset':t['payload']['offset']},
               {'index':7,'name':'scratch','offset':scale_offset},
               {'index':8,'name':'pweights_fp8_scales','offset':t['scales']['offset']}]
        native={'function':'pwin_gemm_fp8_e4m3','grid':d['grid'],'block':[128,1,1],
                'buffers':sorted(qb,key=lambda x:x['index']), 'bytes':[{'index':4,'u32':p}],
                'constants':{'0':p[1],'1':p[2]}}
        new.extend([quant,native]);used.append(t['id'])
    q['dispatches'],q['files'] = new,list(files.values())
    q.update({'library':'pwin_mpp_all','gemm_backend':'fp8-e4m3','candidate_approximate':True,
              'reference_exact':False,'mlp_backend':'mpp-half' if half_mlp else 'original',
              'fp8_pack_root_sha256':manifest['pack_root_sha256'],
              'fp8_gemm_coverage':{'dispatches':len(mapped),'tensor_ids':used,'fp16_fallbacks':fallback},
              'fp8_activation_quantization':'GPU per-row amax/448; FP32 scales; zero row scale1',
              'fp8_handoffs':'original intermediate K16 FP16 handoffs deliberately omitted',
              'requires_ordered_dispatches':True,
              'fp8_scratch':{'legacy_bytes':legacy,'activation_offset':aq_offset,'activation_bytes':aq_bytes,
                             'scale_offset':scale_offset,'scale_bytes':scale_bytes}})
    return q


def build(bundle, pack, output, half_mlp=True, source_runtime=None):
    if os.environ.get('NEURALSTUDIO_SR_METAL3_SHARED_ENCODER') == '2':
        raise ValueError('FP8 scratch dependencies forbid unordered Metal3 encoder mode2')
    if output.exists() or output.is_symlink():
        raise ValueError('Output exists; select a fresh sidecar directory')
    manifest=verify_pack(pack, runtime_directory=source_runtime or bundle)
    outputs, verified, counts, phases = {},set(),{},{}
    for path in sorted(bundle.rglob('pipeline/*.json')):
        match=ENTRY.fullmatch(path.name)
        if not match: continue
        layer,phase=match.groups()
        original=json.loads(path.read_text())
        q=convert(layer,original,manifest,half_mlp)
        # Verify each consumed matrix against the actual original weight bytes,
        # not merely an identity label or a matching offset.
        for identity in q['fp8_gemm_coverage']['tensor_ids'] + q['fp8_gemm_coverage']['fp16_fallbacks']:
            if identity in verified: continue
            t=next(t for t in manifest['tensors'] if t['id']==identity)
            b=(bundle/t['source']['path']).read_bytes()
            logical=matrix_bytes(b,t['source']['offset'],*t['shape'],t['source']['row_stride_elements'])
            if sha(logical) != t['source']['logical_sha256']:
                raise ValueError('Weight source disagrees with verified FP8 pack: '+identity)
            verified.add(identity)
        relative=path.relative_to(bundle)
        outputs[relative]=q
        group=relative.parent.as_posix()
        counts.setdefault(group,{})[layer]=max(counts.get(group,{}).get(layer,0),len(q['dispatches']))
        phases.setdefault((group,layer),set()).add(int(phase) if phase else 0)
    if not outputs:
        raise ValueError('No canonical layer sidecars found')
    for group, per_layer in counts.items():
        if set(per_layer) != LAYERS:
            raise ValueError('Incomplete FP8 layer coverage: '+group)
        for layer in LAYERS:
            if phases[(group,layer)] != (set(range(8)) if layer in SHIFTED else {0}):
                raise ValueError('Incomplete FP8 shift-phase coverage: '+group+'/'+layer)
        if sum(per_layer.values())+3 > 160:
            raise ValueError('FP8 dispatch graph exceeds the session160 argument-table ring')
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.sr-fp8-sidecars-',dir=output.parent) as temporary:
        stage=Path(temporary)/'output';stage.mkdir()
        records=[]
        for relative,q in outputs.items():
            dest=stage/relative;dest.parent.mkdir(parents=True,exist_ok=True)
            data=(json.dumps(q,indent=2)+'\n').encode();dest.write_bytes(data)
            records.append({'path':relative.as_posix(),'sha256':hashlib.sha256(data).hexdigest(),
                            'fp8_gemms':q['fp8_gemm_coverage']['dispatches']})
        receipt={'schema':'netal.sr.fp8-pipelines.v1','candidate_approximate':True,'reference_exact':False,
                 'converter_source_sha256':sha(Path(__file__).read_bytes()),
                 'pack_root_sha256':manifest['pack_root_sha256'],'mlp_backend':'mpp-half' if half_mlp else 'original',
                 'copy_payload_files':manifest['files'],'pipelines':records,
                 'max_frame_pwin_dispatches':max(sum(x.values()) for x in counts.values()),
                 'requires_ordered_dispatches':True,'physical_device_validated':False,'image_quality_validated':False}
        (stage/'fp8-pipelines-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        stage.rename(output)
    return receipt


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--pack',type=Path,required=True,help='verified pack directory containing manifest.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-runtime',type=Path,help='canonical Batched runtime used to derive the pack; defaults to bundle')
    parser.add_argument('--mlp',choices=('half','original'),default='half')
    args=parser.parse_args()
    try:
        result=build(args.bundle.absolute(),args.pack.absolute(),args.output.absolute(),args.mlp=='half',
                     args.source_runtime.absolute() if args.source_runtime else None)
    except (ValueError,KeyError,TypeError) as error:
        parser.error(str(error))
    print('result=pass pipelines='+str(len(result['pipelines']))+' max_frame_dispatches='+str(result['max_frame_pwin_dispatches']))
