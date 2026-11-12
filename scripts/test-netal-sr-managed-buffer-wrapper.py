#!/usr/bin/env python3
"""CPU B1 controls using the actual public wrapper and authenticated helpers.

No variant rewrites the wrapper. EXPECT_FIXED selects assertions only. The native
encoder is excluded; endpoint substitutions and source hashes are recorded.
"""
from pathlib import Path
import os,sys,hashlib,json,subprocess,importlib.util,tempfile
ROOT=Path(__file__).resolve().parents[1]

def run(root,out,expect_fixed):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location('helper_fixture',root/'scripts/test-netal-sr-managed-helpers.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper);helper.ROOT=root;helper.run(out/'base-helpers')
    source=(root/'src/macos_host/dlss_sr_session.mm').read_text()
    base=(out/'base-helpers/actual-helpers.mm').read_text();base=base[:base.index('int main(){@autoreleasepool{')]
    start=source.index('- (BOOL)encodeManagedFrameWithLease:');end=source.index('- (BOOL)enableManagedMTL4FramesWithError:',start)
    wrapper=source[start:end]
    base=base.replace('- (BOOL)enableManagedMTL4FramesWithError:',wrapper+'\n- (BOOL)enableManagedMTL4FramesWithError:',1)
    # Include the unchanged actual nil/device/authentication guard; stop before residency/encoder work.
    start=source.index('- (BOOL)neuralStudioEncodeManagedFrameWithLease:')
    end=source.index('  if (![self neuralStudioRetainManagedResources:@[colorBuffer',start)
    guard=source[start:end]+'  return NO; // Only actual guard rejection paths are executed.\n}\n'
    oldstart=base.index('- (BOOL)neuralStudioEncodeManagedFrameWithLease:')
    oldend=base.index('\n}\n',oldstart)+3
    base=base[:oldstart]+guard+base[oldend:]
    tail=(ROOT/'scripts/fixtures/netal-managed-buffer-wrapper-main.mm').read_text()
    code='#include <thread>\n'+base+tail;file=out/'public-wrapper.mm';file.write_text(code)
    sdk=subprocess.check_output(['xcrun','--sdk','macosx','--show-sdk-path'],text=True).strip()
    args=['xcrun','--sdk','macosx','clang++','-isysroot',sdk,'-mmacosx-version-min=26.0','-std=c++20','-fobjc-arc','-DEXPECT_FIXED='+str(int(expect_fixed)),'-Wall','-Wextra','-Werror','-Wno-incomplete-implementation','-Wno-objc-property-implementation','-Wno-objc-designated-initializers','-I',str(root/'src/macos_host'),'-framework','Foundation',str(file),'-o',str(out/'public-wrapper')]
    with(out/'compile.log').open('w')as log:subprocess.run(args,check=True,stdout=log,stderr=subprocess.STDOUT)
    result=subprocess.run([str(out/'public-wrapper')],check=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=10)
    (out/'traces.log').write_bytes(result.stdout)
    receipt={'schema':'netal.managed.public-buffer-wrapper.cpu.v1','session_sha256':hashlib.sha256(source.encode()).hexdigest(),'public_wrapper_sha256':hashlib.sha256(wrapper.encode()).hexdigest(),'actual_guard_sha256':hashlib.sha256(guard.encode()).hexdigest(),'harness_sha256':hashlib.sha256(code.encode()).hexdigest(),'expect_fixed':expect_fixed,'wrapper_rewritten':False,'native_encoder_model_gpu_device_app_execution':False,'scope':'actual public wrapper/Begin/guard/Fail/Finish/Discard/issue helpers; distinct-thread explicit and registered-error cuts; fake CPU endpoints'}
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(result.stdout.decode(),end='')

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,default=ROOT);parser.add_argument('--expect-fixed',type=int,choices=[0,1],default=1);parser.add_argument('--output',type=Path);args=parser.parse_args()
    if args.output:run(args.source_root,args.output,bool(args.expect_fixed))
    else:
        with tempfile.TemporaryDirectory(prefix='netal-public-wrapper-',dir=os.getenv('TMPDIR'))as out:run(args.source_root,Path(out),bool(args.expect_fixed))
