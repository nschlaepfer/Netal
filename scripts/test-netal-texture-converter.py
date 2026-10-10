#!/usr/bin/env python3
"""Opt-in GPU conformance against the pinned prepared converter; no model or game is run."""
import argparse,hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REFERENCE='1a60b8a9da8162feb87dca08cd8252526183e7314fb733ad7533862e241bf65c'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--reference',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();output=args.output.absolute()
if output.exists() or output.is_symlink():parser.error('Output exists')
if hashlib.sha256(args.reference.read_bytes()).hexdigest()!=REFERENCE:parser.error('Reference converter identity mismatch')
output.mkdir(parents=True)
with (output/'build.log').open('w') as log:
    subprocess.run(['python3',str(ROOT/'scripts/build-netal-sr-texture-kernels.py'),'--output',str(output/'libraries')],check=True,stdout=log,stderr=subprocess.STDOUT)
    subprocess.run(['zsh',str(ROOT/'scripts/build-netal-texture-converter-probe.sh')],check=True,stdout=log,stderr=subprocess.STDOUT)
    compiler=Path(subprocess.check_output(['xcrun','--find','metal'],text=True).strip())
    code=(ROOT/'src/sr_kernels/netal_live_texture_convert.metal').read_text()
    statement='destination.write(float4(source[position.y * size.x + position.x]), position);'
    if code.count(statement)!=1:raise ValueError('No-write control source drift')
    control=output/'no-write.metal';control.write_text(code.replace(statement,'(void)source; (void)destination;'))
    subprocess.run([str(compiler),'-std=metal3.1','-fno-fast-math','-c',str(control),'-o',str(output/'no-write.air')],check=True,stdout=log,stderr=subprocess.STDOUT)
    subprocess.run([str(compiler.with_name('metallib')),str(output/'no-write.air'),'-o',str(output/'no-write.metallib')],check=True,stdout=log,stderr=subprocess.STDOUT)
records=[]
for name,library in [('source',output/'libraries/dlss_sr_live_texture_convert.metallib'),('no-write',output/'no-write.metallib')]:
    with (output/(name+'.log')).open('w') as log:
        result=subprocess.run([str(ROOT/'build/netal-tools/texture-converter-probe'),str(args.reference.resolve()),str(library)],stdout=log,stderr=subprocess.STDOUT)
    text=(output/(name+'.log')).read_text()
    valid=(result.returncode==0 and 'result=pass' in text) if name=='source' else (result.returncode!=0 and 'error=converter mismatch' in text)
    if not valid:raise RuntimeError('Unexpected conformance result: '+name)
    records.append({'case':name,'exit':result.returncode,'library_sha256':hashlib.sha256(library.read_bytes()).hexdigest(),'output':text})
receipt={'schema':'netal.texture-converter-conformance.v1','reference_sha256':REFERENCE,'probe_source_sha256':hashlib.sha256((ROOT/'src/macos_host/netal_texture_converter_probe.mm').read_bytes()).hexdigest(),
         'cases':records,'model_inputs':0,'gameplay_validated':False,'other_devices_validated':False}
(output/'validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('result=pass converter conformance and no-write control; gameplay unverified')
