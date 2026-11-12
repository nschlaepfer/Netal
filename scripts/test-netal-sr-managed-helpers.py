#!/usr/bin/env python3
"""Actual managed helper/context/guard methods exercised with CPU endpoints only.

Does not link Metal, open a device, execute a model/app, or issue GPU commands.
Source extraction intentionally excludes the native encoder; source/archive builds are separate.
"""
from pathlib import Path
import hashlib,json,os,subprocess,tempfile,shutil
ROOT=Path(__file__).resolve().parents[1]

def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(output).free<2*1024**3: raise RuntimeError('CPU compile guard: less than 2 GiB storage available')
    source=(ROOT/'src/macos_host/dlss_sr_session.mm').read_text()
    shared=source[source.index('NSErrorDomain const NeuralStudioDLSSSRSessionErrorDomain'):source.index('constexpr NSUInteger kInputWidth')]+"\n} // extracted private helper namespace\n"
    methods=source[source.index('- (BOOL)enableManagedMTL4FramesWithError:'):source.rfind('\n@end')]
    methods=methods.replace('MTL4CommitOptions','FakeOptions').replace('MTLResidencySetDescriptor','FakeDescriptor').replace('options:options];','options:(id)options];')
    start=source.index('- (BOOL)neuralStudioEncodeManagedFrameWithLease:')
    stop=source.index('  for (id<MTLBuffer> buffer',start)
    guard=source[start:stop]+'  return NO; // Only the actual stale/nil-input guard prefix is exercised.\n}\n'
    fixture="""FakeState::FakeState():managed(gManagedSessionIdentity.fetch_add(1)) {}
@implementation NeuralStudioDLSSSRSession
- (instancetype)initCPUFixture:(id)device {
 self=[super init];if(self){_state=new FakeState;_state->device=device;_state->residency=[NSObject new];}return self;
}
- (void)dealloc {delete _state;}
"""
    adapter=(ROOT/'src/macos_host/dlss_sr_texture_adapter.mm').read_text()
    ast=adapter.index('- (BOOL)encodeManagedTemporalFrameToMTL4CommandBuffer:')
    aend=adapter.index('- (BOOL)neuralStudioEncodeTemporalFrameToMTL4CommandBuffer:',ast)
    managed_adapter=adapter[ast:aend]
    sigend=adapter.index(' {',aend)
    internal_sig=adapter[aend:sigend]
    adapter_fixture='@interface NeuralStudioDLSSTextureAdapter () { @public NeuralStudioDLSSSRSession* _session; std::mutex _managedMutex; NSUInteger _cpuImportCalls; }\n- (instancetype)initCPUFixture:(NeuralStudioDLSSSRSession*)session;\n'+internal_sig+';\n@end\n@implementation NeuralStudioDLSSTextureAdapter\n@synthesize pendingFence;\n- (instancetype)initCPUFixture:(NeuralStudioDLSSSRSession*)session {self=[super init];if(self)_session=session;return self;}\n'+managed_adapter+internal_sig+' {++_cpuImportCalls;return NO;}\n@end\n'
    code=(ROOT/'scripts/fixtures/netal-managed-helpers-prefix.mm').read_text()+shared+fixture+methods+guard+'\n@end\n'+adapter_fixture+(ROOT/'scripts/fixtures/netal-managed-helpers-suffix.mm').read_text()
    harness=output/'actual-helpers.mm';harness.write_text(code)
    binary=output/'actual-helpers'
    sdk=subprocess.check_output(['xcrun','--sdk','macosx','--show-sdk-path'],text=True).strip()
    command=['xcrun','--sdk','macosx','clang++','-isysroot',sdk,'-mmacosx-version-min=26.0','-std=c++20','-fobjc-arc','-Wall','-Wextra','-Werror','-Wno-incomplete-implementation','-Wno-objc-property-implementation','-Wno-objc-designated-initializers','-I',str(ROOT/'src/macos_host'),'-framework','Foundation',str(harness),'-o',str(binary)]
    with (output/'compile.log').open('w')as log:subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
    result=subprocess.run([str(binary)],check=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=10)
    (output/'traces.log').write_bytes(result.stdout)
    receipt={'schema':'netal.managed.cpu-helper-controls.v2','session_sha256':hashlib.sha256(source.encode()).hexdigest(),'adapter_sha256':hashlib.sha256(adapter.encode()).hexdigest(),'harness_sha256':hashlib.sha256(code.encode()).hexdigest(),'scope':'actual helpers/context/reaping/registry/managed adapter and stale/nil-input guard prefix; fake CPU endpoints; native encoder excluded','runtime_model_gpu_device_app_execution':False,'source_runtime_proof':False,'result':'pass'}
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(result.stdout.decode(),end='')

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path);args=parser.parse_args()
    if args.output:run(args.output)
    else:
        with tempfile.TemporaryDirectory(prefix='netal-managed-cpu-',dir=os.getenv('TMPDIR'))as folder:run(folder)
