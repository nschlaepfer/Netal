#!/usr/bin/env python3
"""CPU controls for actual admission/unlock and provider callout checkpoints.

The native encoder is excluded; no Metal/model/GPU/device/app execution occurs.
The optional sealed V2 source-root mode records its existing final/issue rejection,
not a GPU lifetime or unsafe-submission claim.
"""
from pathlib import Path
import sys,os,json,subprocess,importlib.util,hashlib
ROOT=Path(__file__).resolve().parents[1]
def run(root,out,version):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location('fixture',root/'scripts/test-netal-sr-managed-helpers.py');fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture);fixture.ROOT=root
    fixture.run(out/'base-helpers')
    source=(root/'src/macos_host/dlss_sr_session.mm').read_text();base=(out/'base-helpers/actual-helpers.mm').read_text();base=base[:base.index('int main(){@autoreleasepool{')]
    base=base.replace('struct FakeState {','struct FixtureStage { std::vector<std::vector<int>> pipeline_dispatches_by_phase; };\nusing Stage=FixtureStage;\nstruct FixtureArena {id<MTLBuffer>buffer;};\nstruct FakeState {\n std::vector<Stage>pwin_stages; id qualified_backend; FixtureArena arena6;')
    start=source.index('- (BOOL)encodeFrameWithSink:',source.index('@implementation NeuralStudioDLSSSRSession'))
    a=source.index('  std::unique_lock<std::mutex> lock(_state->mutex);',start);b=source.index('  // Claim this frame',a)
    admission=source[a:b]
    p=source.index('    @try {\n      prepared_core = [_state->qualified_backend prepareNeuralCoreForCommandBuffer:');q=source.index('\n  _state->dispatch_timing.next_index',p)
    prepare=source[p:q]
    a=source.index('  if (prepared_core != nil) {',q)+len('  if (prepared_core != nil) {');b=source.index('    sink.metal4 = [sink.metal4_command computeCommandEncoder];',a)
    encode=source[a:b]
    method='''
    static constexpr NSUInteger kPwinShiftPeriod=8,kRingTables=160;
    static NSUInteger OutputRepeat(){return 1;}
    static void*g_dispatch_timing=nullptr;
    struct FrameSink {id<MTLCommandBuffer>metal3=nil;id<MTL4CommandBuffer>metal4_command;id<MTL4ComputeCommandEncoder>metal4;NeuralStudioDLSSSRFrameLease*managed_lease;};
    @interface NeuralStudioDLSSSRSession (BoundaryCPU)
    - (BOOL)reviewerActualProviderBoundary:(NeuralStudioDLSSSRFrameLease*)lease;
    @end
    @implementation NeuralStudioDLSSSRSession (BoundaryCPU)
    - (BOOL)reviewerActualProviderBoundary:(NeuralStudioDLSSSRFrameLease*)lease {
     NSError* __autoreleasing *error=nullptr;
     FrameSink sink{nil,lease->_command,(id)[FakeEncoder new],lease};
    '''+admission+'''
     (void)command;const std::uint64_t frame_index=lease->_token.frame;const BOOL reset=lease->_reset;
     netal::sr::NativeFeatureLayouts views;CHECK(netal::sr::NativeHWCFeatureLayouts(4,2,0,640,frame_index%8,views));
     id<NeuralStudioDLSSSRPreparedNeuralCoreInvocation>prepared_core=nil;
     NeuralStudioDLSSSRNeuralCoreEncodingContext*core_context=nil;
     if(_state->qualified_backend!=nil){
    '''+prepare+'''
     ++NativeInputRecords; // CPU stand-in for continuation after the actual prepare/checkpoint
     if(prepared_core!=nil){
    '''+encode+'''
     }
     return YES;
    }
    @end
    '''
    # Counters are declared before the extracted category and defined only once.
    main=(ROOT/'scripts/fixtures/netal-provider-boundary-main.mm').read_text();declarations=main[:main.index('@interface BoundaryInvocation')];main=main[main.index('@interface BoundaryInvocation'):]
    code=base+declarations+method+main;(out/'provider-boundary.mm').write_text(code)
    args=['xcrun','--sdk','macosx','clang++','-std=c++20','-fobjc-arc','-Wall','-Wextra','-Werror','-Wno-incomplete-implementation','-Wno-objc-property-implementation','-Wno-objc-designated-initializers','-I',str(root/'src/macos_host'),'-framework','Foundation',str(out/'provider-boundary.mm'),'-o',str(out/'provider-boundary')]
    with(out/'compile.log').open('w')as log:subprocess.run(args,check=True,stdout=log,stderr=subprocess.STDOUT)
    results=[]
    for mode in range(1,9):
     p=subprocess.run([str(out/'provider-boundary'),str(mode),str(version==3 and 1 or 0)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=3);assert p.returncode==0,p.stdout.decode();results.append(json.loads(p.stdout))
    receipt={'source_version':version,'session_sha256':hashlib.sha256(source.encode()).hexdigest(),'harness_sha256':hashlib.sha256(code.encode()).hexdigest(),'scope':'actual surrounding unique_lock/authenticated admission/managed unlock plus prepare/encode/checkpoint fragments and actual context/helpers; fake CPU provider; native encoder excluded','results':results,'gpu_execution':False,'unsafe_finalization_or_submission':False}
    (out/'PROVIDER_BOUNDARY_RESULTS.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))

if __name__=='__main__':
    import argparse,tempfile
    parser=argparse.ArgumentParser(description='CPU provider callout controls with actual surrounding admission/unlock. No native encoder/model/GPU execution.')
    parser.add_argument('--source-root',type=Path,default=ROOT)
    parser.add_argument('--expect-version',type=int,choices=[2,3],default=3)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.output:run(args.source_root,args.output,args.expect_version)
    else:
        with tempfile.TemporaryDirectory(prefix='netal-provider-boundary-',dir=os.getenv('TMPDIR'))as out:
            run(args.source_root,Path(out),args.expect_version)
