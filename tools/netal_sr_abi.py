"""Canonical SR boundary metadata for the supported locally supplied quality model.

Templates are obtained from local DLL execution. This source contains ABI layout rules and
hashes, not proprietary tensor bytes or captured process addresses.
"""
import hashlib
import json
from pathlib import Path
import struct

CONTRACTS = {
    'hiluma-input-reset-template.bin': {
        'function': 'hiluma_engine_input_depthreg_mvlo_hdr_v2_rel', 'bytes': 248,
        'pointers': [168,176,184,192,200,208,216,224,240], 'padding': [(4,4)],
        'sha256': '89f604cae8b48dd859b1fb4c8382d73221ef49c99f9385f87114a36112fc3cca'},
    'hiluma-output-reset-template.bin': {
        'function': 'hiluma_engine_output_depthreg_mvlo_hdr_max_v2_rel', 'bytes': 328,
        'pointers': [200,224,232,240,248,256,264,272,280,288,296,304,312], 'padding': [(194,6)],
        'sha256': '102f33027be6af8866e87826ffaac8e2d8f92674f6dfc2772b4636c0b2fd8dc2'},
    'exposure-reset-template.bin': {
        'function': 'cuda_copy_exposure_kernel', 'bytes': 16, 'pointers': [0,8], 'padding': [],
        'sha256': '374708fff7719dd5979ec875d56cd2286f6d3cf7ec317a3b25632aab28ec37bb'},
}


def canonicalize(name, data):
    contract=CONTRACTS[name]
    if len(data)!=contract['bytes']:raise ValueError('Boundary argument size mismatch: '+name)
    result=bytearray(data)
    for offset in contract['pointers']:result[offset:offset+8]=bytes(8)
    for offset,size in contract['padding']:result[offset:offset+size]=bytes(size)
    return bytes(result)


def extract(trace, dumps):
    launches=[json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
    if any(not isinstance(row,dict) for row in launches):raise ValueError('Invalid boundary trace record')
    files={};rows={}
    for name,contract in CONTRACTS.items():
        matching=[row for row in launches if row.get('call')=='cuLaunchKernel' and row.get('name')==contract['function']]
        if len(matching)!=1:raise ValueError('Expected one supported boundary launch: '+name)
        row=matching[0];filename=row.get('argument_dump','')
        if not filename or Path(filename).name!=filename or filename in ('.','..'):
            raise ValueError('Unsafe boundary argument path')
        path=dumps/filename
        if path.is_symlink() or not path.is_file() or path.stat().st_size!=contract['bytes'] or row.get('argument_bytes')!=contract['bytes']:
            raise ValueError('Invalid boundary argument record')
        raw=path.read_bytes();data=canonicalize(name,raw)
        if hashlib.sha256(data).hexdigest()!=contract['sha256']:
            raise ValueError('Boundary scalars do not match the supported reference: '+name)
        files[name]=data
        rows[name]={'bytes':len(data),'sha256':contract['sha256'],'function':contract['function'],
                    'sanitized_pointer_offsets':contract['pointers'],'zeroed_padding_ranges':contract['padding']}
    return files,{'schema':'netal.sr.boundary-abi.v1','reference_render':[1280,720],
                   'reference_output':[1920,1080],'files':rows,'complete_runtime':False}


def validate_package(directory):
    if directory.is_symlink() or (directory/'manifest.json').is_symlink():raise ValueError('Linked boundary package is not accepted')
    manifest=json.loads((directory/'manifest.json').read_text())
    if not isinstance(manifest,dict) or not isinstance(manifest.get('files'),dict):raise ValueError('Invalid boundary manifest')
    if manifest.get('schema')!='netal.sr.boundary-abi.v1':raise ValueError('Unsupported boundary ABI schema')
    if manifest.get('reference_render')!=[1280,720] or manifest.get('reference_output')!=[1920,1080]:raise ValueError('Boundary reference geometry mismatch')
    files={}
    for name,contract in CONTRACTS.items():
        path=directory/name
        if path.is_symlink() or not path.is_file():raise ValueError('Missing or linked boundary template: '+name)
        data=path.read_bytes()
        if len(data)!=contract['bytes'] or hashlib.sha256(data).hexdigest()!=contract['sha256']:
            raise ValueError('Unsupported boundary template: '+name)
        row=manifest.get('files',{}).get(name,{})
        if not isinstance(row,dict):raise ValueError('Invalid boundary inventory entry')
        if row.get('sha256')!=contract['sha256'] or row.get('bytes')!=contract['bytes']:
            raise ValueError('Boundary manifest mismatch: '+name)
        files[name]=data
    return files,manifest


def f32(value):return struct.unpack('<f',struct.pack('<f',value))[0]
def round_up(value,multiple):return (value+multiple-1)//multiple*multiple


def for_geometry(base, output, render):
    ow,oh=output;iw,ih=render
    if any(not isinstance(x,int) or not 1<=x<=16384 for x in (ow,oh,iw,ih)):
        raise ValueError('Invalid boundary geometry')
    if iw>ow or ih>oh:raise ValueError('Render dimensions exceed output')
    data={name:bytearray(base[name]) for name in CONTRACTS}
    inp=data['hiluma-input-reset-template.bin'];out=data['hiluma-output-reset-template.bin']
    def put_u(b,offset,value):struct.pack_into('<I',b,offset,value)
    def put_f(b,offset,value):struct.pack_into('<f',b,offset,f32(value))
    ratio_x=f32(iw*f32(1/ow));ratio_y=f32(ih*f32(1/oh))
    extent_w=max(256,round_up((ow+3)//4,32));extent_h=max(256,round_up((oh+3)//4,32))
    for offset,value in [(8,f32(1/ratio_x)),(12,f32(1/ratio_y)),(40,ratio_x),(44,ratio_y),(48,ratio_x),(52,ratio_y),
                          (120,1/ow),(124,1/round_up(oh,16)),(128,1/(4*extent_w)),(132,1/(4*extent_h)),(136,1/ow),(140,1/oh)]:put_f(inp,offset,value)
    put_u(inp,64,extent_w)
    for offset in (80,96,112):put_u(inp,offset,iw-1)
    for offset in (84,100,116):put_u(inp,offset,ih-1)
    for offset,value in [(0,f32(1/ratio_x)),(4,f32(1/ratio_y)),(40,ratio_x),(44,ratio_y),(48,f32(ratio_x/2)),(52,f32(ratio_y/2)),
                         (56,ratio_x),(60,ratio_y),(144,1/ow),(148,1/round_up(oh,16)),(152,1/ow),(156,1/oh)]:put_f(out,offset,value)
    for offset,value in ((80,extent_w),(84,extent_h),(136,ow),(140,oh)):put_u(out,offset,value)
    for offset in (104,120,216):put_u(out,offset,iw-1)
    for offset in (108,124,220):put_u(out,offset,ih-1)
    try:
        struct.pack_into('<3e',out,188,8.0,f32(-16.0*ow/iw),f32(-16.0*oh/ih))
    except OverflowError as error:
        raise ValueError('Render/output scale exceeds the supported half-precision range') from error
    return {name:bytes(value) for name,value in data.items()}


def profile_manifest(metadata, render, files):
    ow,oh=metadata['output'];extent_w,extent_h=metadata['stages'][0]['extent']
    arrays=[]
    for identity,(width,height,channels) in enumerate([(2*ow,2*round_up(oh,16),1),(2*ow,2*round_up(oh,16),1),
                                                      (ow,round_up(oh,16),4),(ow,round_up(oh,16),4),
                                                      (extent_w,extent_h,4),(1,1,1)],1):
        arrays.append(dict(id=identity,width=width,height=height,format=16,channels=channels))
    iw,ih=render
    for identity,width,height,fmt,channels in [(7,iw,ih,16,4),(9,iw,ih,32,1),(10,iw,ih,16,2),(11,1,1,32,1)]:
        arrays.append(dict(id=identity,width=width,height=height,format=fmt,channels=channels))
    return {'schema':'neuralstudio.re9.sr_profile.v1','origin':'source-generated topology and local DLL ABI',
            'output':[ow,oh],'render_reference':list(render),'arenas':metadata['arenas'],'arrays':arrays,
            'hiluma_input':{'grid':[extent_w//4,extent_h//16,1],'block':[16,16,1]},
            'hiluma_output':{'grid':[(ow+15)//16,(oh+15)//16,1],'block':[16,16,1]},
            'exposure_copy':{'grid':[1,1,1],'block':[1,1,1]},'stages':metadata['stages'],
            'files':{name:{'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()} for name,data in files.items()}}
