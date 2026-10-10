#!/usr/bin/env python3
"""Build SR body assets from source and the supported local model.

Produces layout/config files, repacked weights, pipeline descriptions, source boundary
libraries and three formula-generated math resources. Remaining legacy kernels and math
resources make this an incomplete runtime; it deliberately has no root manifest.json.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from netal_sr_layouts import profile, STAGES
from netal_sr_math import generate as generate_math
from netal_sr_calibration import load as load_calibration

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIZES = ['1024x576','1280x720','1280x800','1920x1080','2560x1440','3440x1440','3840x2160']
KERNELS = ['pwin_gemm','pwin_gemm2','pwin_gemm3','pwin_norm','pwin_mix','pwin_mix2','pwin_mix3',
           'pwin_mix4','pwin_mix5','pwin_mixp','pwin_mixp2','pwin_mixm','pwin_mixm2','pwin_merge',
           'pwin_d2s','pwin_pad','pwin_pad2','pwin_mlp','pwin_mlp2','pwin_attn','pwin_dbg']


def load(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def digest(data):
    return hashlib.sha256(data).hexdigest()


def build(model, output, sizes, compile_metal=True, include_boundary=True):
    if output.exists() or output.is_symlink():
        raise ValueError('Output already exists; choose a new directory')
    extractor=load('model_reference',ROOT/'tools/netal-dlss-extract.py')
    weights=(model/'weights.bin').read_bytes()
    if len(weights)!=extractor.KNOWN['weights.bin'] or digest(weights)!=extractor.REFERENCE['files']['weights.bin']:
        raise ValueError('Weights do not match the supported model')
    abi=extractor.abi_module()
    boundaries,abi_manifest=abi.validate_package(model/'abi') if include_boundary else (None,None)
    repacker=load('weight_repack',ROOT/'tools/pwin_pipeline/repack.py')
    dimensions=[]
    for size in sizes:
        w,h=map(int,size.split('x')); profile(w,h); dimensions.append((w,h))
    if len(set(dimensions))!=len(dimensions):
        raise ValueError('Duplicate output geometry')
    output.parent.mkdir(parents=True,exist_ok=True)
    sources={}
    def source(path):
        path=path.resolve()
        data=path.read_bytes();sources[path.relative_to(ROOT).as_posix()]=digest(data);return data
    for path in [Path(__file__),ROOT/'tools/netal-dlss-extract.py',ROOT/'tools/netal_sr_abi.py',ROOT/'scripts/netal_sr_layouts.py',ROOT/'scripts/netal_sr_math.py',ROOT/'tools/pwin_pipeline/repack.py',ROOT/'tools/pwin_pipeline/build-profile-pipelines.py']:
        source(path)
    for path in (ROOT/'tools/pwin_pipeline').glob('make-*.py'):
        source(path)
    source(ROOT/'scripts/netal_sr_calibration.py')
    source(ROOT/'runtime/arithmetic-calibration/manifest.json')
    calibration=load_calibration(ROOT)
    with tempfile.TemporaryDirectory(prefix='.sr-assets-',dir=output.parent) as temporary:
        stage=Path(temporary)/'assets';stage.mkdir()
        def write(relative,data):
            target=stage/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        # These render dimensions reproduce the recorded compatibility profiles. Other sizes
        # use a two-thirds reference; the runtime still patches the game's actual render size.
        renders={'1024x576':(683,384),'1280x720':(854,480),'1280x800':(854,534),
                 '1920x1080':(1280,720),'2560x1440':(1920,1080),'3440x1440':(2293,960),'3840x2160':(2560,1440)}
        profile_dimensions=dimensions if (1920,1080) in dimensions else dimensions+[(1920,1080)]
        for w,h in profile_dimensions:
            files,metadata=profile(w,h)
            render=renders.get(f'{w}x{h}',((2*w+2)//3,(2*h+2)//3))
            if boundaries is not None:
                files.update({'abi/'+name:data for name,data in abi.for_geometry(boundaries,(w,h),render).items()})
                manifest=abi.profile_manifest(metadata,render,files)
                write(f'profiles/{w}x{h}/manifest.json',(json.dumps(manifest,indent=2)+'\n').encode())
                metadata['boundary_abi_included']=True
                metadata['render_reference']=list(render)
            for name,data in files.items():write(f'profiles/{w}x{h}/'+name,data)
            write(f'profiles/{w}x{h}/layout.json',(json.dumps(metadata,indent=2)+'\n').encode())
        for name,data in generate_math().items():write('math/'+name,data)
        for name,data in calibration.items():write('math/'+name,data)
        root_files,_=profile(1920,1080)
        for name,data in root_files.items():write(name,data)
        write('pwin/enc0-arguments-template.bin',root_files['pwin/enc0/arguments-template.bin'])
        if boundaries is not None:
            for name,data in boundaries.items():write('abi/'+name,data)
        for spec in STAGES:
            definition=json.loads(source(ROOT/f'tools/pwin_pipeline/spec-{spec.name}.json'))
            packed,layout=repacker.repack(weights,definition)
            write(f'pipeline/{spec.name}-weights.bin',packed)
            write(f'pipeline/layouts/layout-{spec.name}.json',(json.dumps(layout,indent=1)+'\n').encode())
        result=subprocess.run([sys.executable,str(ROOT/'tools/pwin_pipeline/build-profile-pipelines.py'),
                               str(stage),'--layout-dir',str(stage/'pipeline/layouts')],
                              capture_output=True,text=True,check=True)
        compiler=None
        boundary_kernels=[]
        reference_kernels=[]
        texture_kernels=[]
        if compile_metal:
            combined=b''.join(b'// ==== '+name.encode()+b'.metal\n'+source(ROOT/f'tools/pwin_pipeline/{name}.metal') for name in KERNELS)
            metal=Path(temporary)/'pwin_all.metal';metal.write_bytes(combined)
            air=Path(temporary)/'pwin_all.air'
            library=stage/'metallib/pwin_all.metallib';library.parent.mkdir(exist_ok=True)
            metal_tool=Path(subprocess.check_output(['xcrun','--find','metal'],text=True).strip())
            library_tool=metal_tool.with_name('metallib')
            if not library_tool.is_file():
                library_tool=Path(subprocess.check_output(['xcrun','--find','metallib'],text=True).strip())
            subprocess.run([str(metal_tool),'-std=metal4.0','-fno-fast-math','-c',str(metal),'-o',str(air)],check=True)
            subprocess.run([str(library_tool),str(air),'-o',str(library)],check=True)
            compiler=subprocess.check_output([str(metal_tool),'--version'],text=True).strip()
            reference_builder=ROOT/'scripts/build-netal-sr-reference-kernels.py'
            source(reference_builder)
            reference_output=Path(temporary)/'reference-kernels'
            subprocess.run([sys.executable,str(reference_builder),'--output',str(reference_output)],check=True)
            reference_kernels=json.loads((reference_output/'build-receipt.json').read_text())['kernels']
            for kernel in reference_kernels:
                source(ROOT/kernel['source'])
                shutil.copy2(reference_output/kernel['library'],stage/'metallib'/kernel['library'])
            if include_boundary:
                boundary_builder=ROOT/'scripts/build-netal-sr-boundary-kernels.py'
                source(boundary_builder)
                boundary_output=Path(temporary)/'boundary-kernels'
                subprocess.run([sys.executable,str(boundary_builder),'--output',str(boundary_output)],check=True)
                boundary_receipt=json.loads((boundary_output/'build-receipt.json').read_text())
                boundary_kernels=boundary_receipt['kernels']
                for kernel in boundary_kernels:
                    source(ROOT/kernel['source'])
                    shutil.copy2(boundary_output/kernel['library'],stage/'metallib'/kernel['library'])
                texture_builder=ROOT/'scripts/build-netal-sr-texture-kernels.py'
                source(texture_builder)
                texture_output=Path(temporary)/'texture-kernels'
                subprocess.run([sys.executable,str(texture_builder),'--output',str(texture_output)],check=True)
                texture_kernels=json.loads((texture_output/'build-receipt.json').read_text())['kernels']
                for kernel in texture_kernels:
                    source(ROOT/kernel['source'])
                    shutil.copy2(texture_output/kernel['library'],stage/'metallib'/kernel['library'])
        records={p.relative_to(stage).as_posix():{'bytes':p.stat().st_size,'sha256':digest(p.read_bytes())} for p in sorted(stage.rglob('*')) if p.is_file()}
        receipt={'schema':'netal.sr.source-body-assets.v1','weights_sha256':digest(weights),
                 'profiles':[f'{w}x{h}' for w,h in dimensions], 'sources':sources,'compiler':compiler,
                 'files':records,'complete_runtime':False,'inference_validated':False,'gameplay_validated':False,
                 'boundary_abi_included':boundaries is not None,
                 'boundary_kernels':boundary_kernels,'redistribution_cleared':False,
                 'reference_kernels':reference_kernels,
                 'texture_kernels':texture_kernels,
                 'scalar_calibration':'pinned measured RCP/RSQ data; see runtime/arithmetic-calibration/manifest.json',
                 'remaining':['general scalar math beyond sampled contracts','complete runtime assembly and validation']
                    + ([] if texture_kernels else ['live texture libraries'])
                    + ([] if boundaries is not None else ['boundary ABI'])
                    + ([] if boundary_kernels else ['source boundary libraries'])
                    + ([] if compile_metal else ['shared Metal body library','Reference body libraries'])}
        write('SOURCE_BUILD_RECEIPT.json',(json.dumps(receipt,indent=2)+'\n').encode())
        stage.rename(output)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--resolutions',nargs='+',default=DEFAULT_SIZES)
    parser.add_argument('--body-only',action='store_true',help='legacy diagnostic mode: omit boundary metadata')
    parser.add_argument('--no-compile',action='store_true',help='generate CPU assets only, without the Metal library')
    args=parser.parse_args()
    try:
        receipt=build(args.model.resolve(),args.output.absolute(),args.resolutions,not args.no_compile,not args.body_only)
    except (OSError,ValueError,subprocess.CalledProcessError) as error:
        print(f'result=fail {error}',file=sys.stderr)
        if isinstance(error,subprocess.CalledProcessError) and error.stderr:print(error.stderr[-3000:],file=sys.stderr)
        return 1
    print(f'result=pass body_assets={len(receipt["files"])} complete_runtime=false')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
