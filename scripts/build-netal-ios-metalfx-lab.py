#!/usr/bin/env python3
"""Build a model-free physical-iOS Metal 4/MetalFX comparison archive; no signing.

The optional capture is enabled by the consuming app, never by this archive.
No proprietary shader, model, trace, game data or device identity is embedded.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from netal_apple_build import check_macho_platform, version_tuple

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'src/ios_metalfx_lab'

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def build(output,deployment_target='26.0'):
    output=Path(output).absolute()
    if output.exists() or output.is_symlink():raise ValueError('Output exists; choose a new directory')
    if version_tuple(deployment_target)<(26,0,0):raise ValueError('Metal 4 requires iOS 26+')
    sdk=subprocess.check_output(['xcrun','--sdk','iphoneos','--show-sdk-path'],text=True).strip()
    sdk_version=subprocess.check_output(['xcrun','--sdk','iphoneos','--show-sdk-version'],text=True).strip()
    if version_tuple(deployment_target)>version_tuple(sdk_version):raise ValueError('Deployment target exceeds SDK')
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.metalfx-lab-',dir=output.parent) as temp:
        stage=Path(temp)/'output';stage.mkdir();obj=Path(temp)/'lab.o'
        archive=stage/'libnetal_ios_metalfx_lab.a'
        flags=['-std=c++17','-fobjc-arc','-fblocks','-Wall','-Wextra','-Werror','-Wno-deprecated-declarations',
               '-target','arm64-apple-ios'+deployment_target,'-isysroot',sdk,'-O2']
        subprocess.run(['xcrun','--sdk','iphoneos','clang++',*flags,'-c',str(SOURCE/'NetalMetalFXLab.mm'),'-o',str(obj)],check=True)
        check_macho_platform(obj,2,deployment_target)
        subprocess.run(['xcrun','--sdk','iphoneos','libtool','-static','-o',str(archive),str(obj)],check=True)
        (stage/'NetalMetalFXLab.h').write_bytes((SOURCE/'NetalMetalFXLab.h').read_bytes())
        sources=[p for p in SOURCE.glob('*') if p.is_file()]+[ROOT/'src/ios_probe/NetalProbeMetrics.h']
        receipt={'schema':'netal.metalfx.lab.build.v1','sdk':'iphoneos','sdk_version':sdk_version,
          'deployment_target':deployment_target,'cpu_target':'arm64-apple-ios'+deployment_target,
          'archive':archive.name,'archive_sha256':digest(archive),'required_frameworks':['Foundation','Metal','MetalFX'],
          'sources':[{'path':str(p.relative_to(ROOT)),'sha256':digest(p)} for p in sorted(sources)],
          'models_included':False,'physical_device_executed':False,'capture_enabled_by_archive':False,
          'gameplay_validated':False,'latency_validated':False}
        (stage/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');stage.rename(output)
    return receipt

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir','--output',dest='output',type=Path,required=True)
    p.add_argument('--deployment-target',default='26.0');a=p.parse_args()
    try:build(a.output,a.deployment_target)
    except (ValueError,OSError,subprocess.CalledProcessError) as e:p.exit(1,'MetalFX lab build failed: '+str(e)+'\n')
    print('result=pass physical_device_executed=false latency_validated=false')

if __name__=='__main__':main()
