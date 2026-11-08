#!/usr/bin/env python3
"""Build a model-free, optional physical-iOS27 FP8 operator lab (no signing).

An unavailable SDK/compiler is a build error, never simulated device success.
Existing iOS26/macOS SR targets do not depend on this experimental archive.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from netal_apple_build import check_macho_platform, metal_toolchain

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'src/ios_fp8_lab'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    output=args.output_dir.absolute()
    if output.exists() or output.is_symlink():parser.error('Output exists; choose a new directory')
    try:compiler,linker,flags,target=metal_toolchain('iphoneos','27.0')
    except ValueError as error:parser.error(str(error))
    sdk=subprocess.check_output(['xcrun','--sdk','iphoneos','--show-sdk-path'],text=True).strip()
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.fp8-lab-',dir=output.parent) as temporary:
        stage=Path(temporary)/'output';stage.mkdir()
        air=Path(temporary)/'lab.air';obj=Path(temporary)/'lab.o'
        library=stage/'netal_fp8_lab.metallib';archive=stage/'libnetal_ios_fp8_lab.a'
        metal_flags=['-std=metal4.1','-fno-fast-math']
        subprocess.run([str(compiler),*flags,*metal_flags,'-c',str(SOURCE/'NetalFP8MatmulLab.metal'),'-o',str(air)],check=True)
        subprocess.run([str(linker),str(air),'-o',str(library)],check=True)
        subprocess.run(['xcrun','--sdk','iphoneos','clang++','-std=c++17','-fobjc-arc','-fblocks','-Wall','-Wextra','-Werror',
            '-target','arm64-apple-ios27.0','-isysroot',sdk,'-c',str(SOURCE/'NetalFP8MatmulLab.mm'),'-o',str(obj)],check=True)
        check_macho_platform(obj,2,'27.0')
        subprocess.run(['xcrun','--sdk','iphoneos','libtool','-static','-o',str(archive),str(obj)],check=True)
        header=stage/'NetalFP8MatmulLab.h';header.write_bytes((SOURCE/header.name).read_bytes())
        receipt={'schema':'netal.fp8.matmul-lab.build.v1',**target,'msl':'4.1','metal_flags':metal_flags,
            'compiler':subprocess.check_output([str(compiler),'--version'],text=True).strip(),
            'sources':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in sorted(SOURCE.glob('*')) if p.is_file()],
            'library':library.name,'library_sha256':sha(library),'archive':archive.name,'archive_sha256':sha(archive),
            'kernel_entries':['netal_lab_fp16','netal_lab_e4m3','netal_lab_e5m2','netal_lab_pack_e4m3','netal_lab_pack_e5m2'],
            'physical_device_executed':False,'native_fp8_instruction_verified':False,'full_sr_validated':False,
            'models_included':False,'runtime_hardware_allowlist':['A20 Pro'],'timing_transport':'MTLCommandBuffer driver GPU timestamps'}
        (stage/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        stage.rename(output)
    print('result=pass physical_device_executed=false native_instruction_verified=false')

if __name__=='__main__':main()
