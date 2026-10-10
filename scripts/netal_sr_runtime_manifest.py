"""Complete SR asset inventory; integrity is not authorship or gameplay validation."""
import hashlib
import json
from pathlib import Path
import re
import struct
import tempfile

SCHEMA='netal.sr.runtime.v2'
MODEL={'model/weights.bin':(3545824,'ddaa59cd82deed53bad9d65b82b454f1439fc4875e4220b3ffafafb6cd0296a1'),
       'model/constants.bin':(524288,'305f4c6773d1cb333b577094e28a46beb1867898716d8290121862a2cb06e773')}
STAGES=['enc0','enc1','enc2','enc3','enc4','dec5','dec4','dec3','dec2','dec1','dec0']
REQUIRED=set(MODEL)|{f'metallib/dltss_pwin_{s}_layer_exact.metallib' for s in STAGES}|{
 'metallib/hiluma_input_poison_a.metallib','metallib/hiluma_output_exact_math_safe.metallib',
 'metallib/hiluma_input_depthinv_exact.metallib','metallib/hiluma_output_depthinv_exact.metallib',
 'metallib/netal_copy_exposure.metallib','metallib/dlss_sr_live_texture_convert.metallib',
 'metallib/hiluma_input_depthinv.metallib','math/sm120-texture-weights.bin',
 'math/sm120-mufu/lg2-input-table.bin','math/sm120-mufu/lg2-output-table.bin',
 'math/sm120-mufu/ex2-input-table.bin','math/sm120-mufu/ex2-output-table.bin',
 'math/rcp-correction-rom.bin','math/rsq-correction-rom.bin','math/rcp-f16-to-f16-lut.bin','math/ex2-f16x2-lut.bin',
 'abi/hiluma-input-reset-template.bin','abi/hiluma-output-reset-template.bin','abi/exposure-reset-template.bin'}


def safe_path(value):
    return isinstance(value,str) and bool(re.fullmatch(r'[A-Za-z0-9_.\-/]+',value)) and all(p not in ('','.','..') for p in value.split('/')) and not any(x in value.lower() for x in ('expected','capture'))


def ledger_root(rows):
    if not isinstance(rows,list) or any(not isinstance(row,dict) or not isinstance(row.get('path'),str) for row in rows):raise ValueError('Invalid asset inventory')
    seen=set(); material=bytearray((SCHEMA+'\n').encode())
    for row in sorted(rows,key=lambda r:r['path']):
        path=row['path'];size=row['bytes'];digest=row['sha256']
        if not safe_path(path) or path=='manifest.json' or path in seen or type(size) is not int or size<0 or not isinstance(digest,str) or not re.fullmatch('[0-9a-f]{64}',digest):raise ValueError('Invalid or duplicate asset row')
        seen.add(path);material.extend(f'{path}\0{size}\0{digest}\n'.encode())
    return hashlib.sha256(material).hexdigest()


def verify_engine(root,manifest):
    if manifest.get('schema')!=SCHEMA or ledger_root(manifest.get('files'))!=manifest.get('asset_root_sha256'):raise ValueError('Runtime inventory root mismatch')
    root=Path(root);rows={row['path']:row for row in manifest['files'] if not row['path'].startswith('model/')}
    actual=set()
    for path in root.rglob('*'):
        if path.is_symlink():raise ValueError('Linked runtime asset')
        if not path.is_file():continue
        relative=path.relative_to(root).as_posix()
        if relative=='manifest.json' or relative.startswith('model/'):continue
        actual.add(relative);row=rows.get(relative);data=path.read_bytes()
        if not row or len(data)!=row['bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']:raise ValueError('Runtime engine asset mismatch: '+relative)
    if actual!=rows.keys():raise ValueError('Missing runtime engine asset')


def pointer_slots(root,relative,data):
    name=Path(relative).name
    layouts={'hiluma-input-reset-template.bin':(248,[168,176,184,192,200,208,216,224,240]),
             'hiluma-output-reset-template.bin':(328,[200,224,232,240,248,256,264,272,280,288,296,304,312]),
             'exposure-reset-template.bin':(16,[0,8]), 'enc0-arguments-template.bin':(176,[8,24,32,48,64])}
    if name in layouts:
        size,slots=layouts[name]
        if len(data)!=size:raise ValueError('Boundary/template size mismatch')
        return slots
    if name=='arguments-template.bin' and ('/pwin/' in relative or relative.startswith('pwin/')):
        config_path=Path(root)/Path(relative).parent/'replay-config.bin'
        if config_path.is_symlink():raise ValueError('Linked PWin config')
        config=config_path.read_bytes()
        if len(data)!=176 or len(config)!=296 or config[:7]!=b'NSPWIN1':raise ValueError('PWin ABI shape mismatch')
        count=struct.unpack_from('<I',config,164)[0]
        if count>8:raise ValueError('PWin relocation count')
        return [struct.unpack_from('<I',config,168+i*16)[0] for i in range(count)]
    return []

def create(root):
    root=Path(root);rows=[]
    for path in sorted(root.rglob('*')):
        if path.is_symlink():raise ValueError('Symlink in runtime')
        if not path.is_file():continue
        relative=path.relative_to(root).as_posix()
        if relative=='manifest.json':continue
        if not safe_path(relative):raise ValueError('Unsafe runtime path: '+relative)
        data=path.read_bytes()
        for offset in pointer_slots(root,relative,data):
            if offset+8>len(data) or any(data[offset:offset+8]):raise ValueError('Captured pointer or invalid mask: '+relative)
        rows.append({'path':relative,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    names={r['path']:r for r in rows}
    if not REQUIRED<=names.keys():raise ValueError('Missing required runtime assets: '+', '.join(sorted(REQUIRED-names.keys())))
    for name,(size,digest) in MODEL.items():
        if names[name]['bytes']!=size or names[name]['sha256']!=digest:raise ValueError('Unsupported model bytes: '+name)
    return {'schema':SCHEMA,'model':'DLSS_SR_Feature1_Quality_K','asset_root_sha256':ledger_root(rows),'files':rows,
            'claims':{'inventory_complete':True,'gameplay_validated':False,'publication_clearance':False}}


def seal(root,replace=False):
    root=Path(root);manifest=root/'manifest.json'
    if manifest.exists() and not replace:raise ValueError('Manifest exists; replacement must be explicit')
    result=create(root)
    # Put the temporary file outside the inventoried directory.
    with tempfile.NamedTemporaryFile(mode='w',dir=root.parent,prefix='.sr-manifest-',delete=False) as stream:
        temporary=Path(stream.name);json.dump(result,stream,indent=2);stream.write('\n')
    try:temporary.replace(manifest)
    finally:temporary.unlink(missing_ok=True)
    return result
