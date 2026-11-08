#import "NetalFP8MatmulLab.h"
#include "NetalFP8Reference.h"
#import <TargetConditionals.h>
#include <array>
#include <chrono>
#include <cstring>
#include <string>
#include <vector>

namespace {
using namespace netal_fp8_lab;
constexpr NSUInteger kGuard=256;
constexpr std::array<std::array<unsigned,2>,5> kShapes={{{16,32},{32,48},{64,256},{640,160},{160,768}}};
constexpr std::array<unsigned,8> kOracleRows={0,1,7,31,63,64,127,1023};
struct Params {std::uint32_t m,n,k,paddedK,aStride,bStride,yStride;float scaleA,scaleB;};
static_assert(sizeof(Params)==36,"Metal ABI");
struct Buffer {id<MTLBuffer> value;NSUInteger bytes;};
Buffer MakeBuffer(id<MTLDevice> device,NSUInteger bytes) {
    id<MTLBuffer> buffer=[device newBufferWithLength:bytes+2*kGuard options:MTLResourceStorageModeShared];
    if(buffer)std::memset(buffer.contents,0xa5,buffer.length);
    return {buffer,bytes};
}
template<typename T>T* Data(const Buffer& buffer) {
    return reinterpret_cast<T*>(static_cast<std::uint8_t*>(buffer.value.contents)+kGuard);
}
bool Guards(const Buffer& buffer) {
    if(!buffer.value)return false;
    const auto* bytes=static_cast<const std::uint8_t*>(buffer.value.contents);
    return Guard(bytes,kGuard)&&Guard(bytes+kGuard+buffer.bytes,kGuard);
}
NSDictionary* ErrorRecord(const Error& error) {
    return @{ @"maximum_absolute_error":@(error.maximum),@"rms_error":@(error.RMS()),
              @"finite_values_checked":@(error.count),@"nonfinite_values":@(error.nonfinite)};
}
bool Fail(NSError** error,NSString* message) {
    if(error)*error=[NSError errorWithDomain:@"org.netal.fp8-lab" code:1
                                   userInfo:@{NSLocalizedDescriptionKey:message}];
    return false;
}
NSString* FormatName(int mode) {return mode==0?@"fp16":mode==1?@"e4m3":@"e5m2";}
float Fixture(unsigned row,unsigned col,unsigned salt,bool noise) {
    const std::uint32_t hash=(row*747796405u+col*2891336453u+salt*277803737u)^0x9e3779b9u;
    return noise?(float((hash>>8)&65535u)/32768.0f-1.0f):float(int((hash>>16)%17)-8)/8.0f;
}
void EncodePack(id<MTLComputeCommandEncoder> encoder,id<MTLComputePipelineState> pipeline,
                const Buffer& source,const Buffer& packed,const Buffer& decoded,float scale) {
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:source.value offset:kGuard atIndex:0];
    [encoder setBuffer:packed.value offset:kGuard atIndex:1];
    [encoder setBuffer:decoded.value offset:kGuard atIndex:2];
    [encoder setBytes:&scale length:sizeof(scale) atIndex:3];
    [encoder dispatchThreads:MTLSizeMake(source.bytes/sizeof(float)/4,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
}
void EncodeMatmul(id<MTLComputeCommandEncoder> encoder,id<MTLComputePipelineState> pipeline,
                  const Buffer& a,const Buffer& b,const Buffer& y,const Buffer& bias,const Params& params) {
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:a.value offset:kGuard atIndex:0];[encoder setBuffer:b.value offset:kGuard atIndex:1];
    [encoder setBuffer:y.value offset:kGuard atIndex:2];[encoder setBytes:&params length:sizeof(params) atIndex:3];
    [encoder setBuffer:bias.value offset:kGuard atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake(params.m/64,(params.n+31)/32,1)
           threadsPerThreadgroup:MTLSizeMake(128,1,1)];
}
// MTLCommandBuffer retains referenced resources. The captured array also owns
// all buffers until completion if the bounded host wait times out. No timeout
// allows reusing/freeing the in-flight buffers or pretending timing succeeded.
bool Submit(id<MTLCommandQueue> queue,NSArray* resources,
             void (^encode)(id<MTLComputeCommandEncoder>),double* gpuMS,NSError** error) {
    id<MTLCommandBuffer> command=[queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
    if(!command||!encoder)return Fail(error,@"Could not create GPU commands.");
    encode(encoder);[encoder endEncoding];
    dispatch_semaphore_t done=dispatch_semaphore_create(0);
    __block double start=0,end=0;__block BOOL completed=NO;__block NSError* failure=nil;
    [command addCompletedHandler:^(id<MTLCommandBuffer> finished){
        (void)resources.count;start=finished.GPUStartTime;end=finished.GPUEndTime;
        completed=finished.status==MTLCommandBufferStatusCompleted;failure=finished.error;
        dispatch_semaphore_signal(done);
    }];
    [command commit];
    if(dispatch_semaphore_wait(done,dispatch_time(DISPATCH_TIME_NOW,30*NSEC_PER_SEC)))
        return Fail(error,@"GPU timeout; in-flight resources remain retained. Close the app before another test.");
    if(!completed||failure)return Fail(error,failure.localizedDescription?:@"GPU command failed.");
    *gpuMS=Milliseconds(start,end,true);
    return std::isfinite(*gpuMS)||Fail(error,@"Driver GPU timestamps are unavailable; no performance result was recorded.");
}
bool PackCheck(const Buffer& source,const Buffer& packed,const Buffer& decoded,float scale,Format format,
                NSUInteger* saturated) {
    const auto* inputs=Data<float>(source);const auto* bytes=Data<std::uint8_t>(packed);
    const auto* outputs=Data<float>(decoded);bool valid=Guards(source)&&Guards(packed)&&Guards(decoded);
    for(NSUInteger i=0;i<source.bytes/sizeof(float);++i) {
        const float normalized=inputs[i]/scale;
        if(std::abs(normalized)>Maximum(format))++*saturated;
        valid=valid&&std::isfinite(inputs[i])&&bytes[i]==EncodeFinite(normalized,format)&&
              outputs[i]==Decode(bytes[i],format)*scale;
    }
    return valid;
}
bool TensorType(id<MTLDevice> device,MTLTensorDataType type) {
    MTLTensorDescriptor* descriptor=[MTLTensorDescriptor new];NSInteger extent=32;
    descriptor.dimensions=[[MTLTensorExtents alloc] initWithRank:1 values:&extent];
    descriptor.dataType=type;descriptor.storageMode=MTLStorageModePrivate;descriptor.usage=MTLTensorUsageCompute;
    NSError* error=nil;return [device newTensorWithDescriptor:descriptor error:&error]!=nil&&error==nil;
}
bool A20Name(NSString* name) {
    NSArray<NSString*>* words=[[name lowercaseString] componentsSeparatedByCharactersInSet:NSCharacterSet.whitespaceAndNewlineCharacterSet];
    NSMutableArray<NSString*>* normalized=[NSMutableArray array];
    for(NSString* word in words)if(word.length)[normalized addObject:word];
    if([normalized.firstObject isEqual:@"apple"])[normalized removeObjectAtIndex:0];
    if([normalized.lastObject isEqual:@"gpu"])[normalized removeLastObject];
    return [normalized isEqualToArray:@[@"a20",@"pro"]];
}
} // namespace

NSDictionary<NSString*,id>* NetalRunFP8MatmulLab(id<MTLDevice> device,NSURL* libraryURL,
    NSUInteger sampleCount,NSUInteger repetitionsPerSample,NSError** error) {
    return NetalRunFP8MatmulLabWithStop(device,libraryURL,sampleCount,repetitionsPerSample,nil,error);
}
NSDictionary<NSString*,id>* NetalRunFP8MatmulLabWithStop(id<MTLDevice> device,NSURL* libraryURL,
    NSUInteger sampleCount,NSUInteger repetitionsPerSample,BOOL (^shouldStop)(void),NSError** error) {
    if(error)*error=nil;
#if !TARGET_OS_IOS || TARGET_OS_SIMULATOR || TARGET_OS_MACCATALYST
    (void)device;(void)libraryURL;(void)sampleCount;(void)repetitionsPerSample;(void)shouldStop;
    Fail(error,@"This experiment requires physical iOS on A20 Pro.");return nil;
#else
    if(NSProcessInfo.processInfo.operatingSystemVersion.majorVersion<27||
       !device||!A20Name(device.name)||!libraryURL.isFileURL||sampleCount<3||sampleCount>15||
       repetitionsPerSample<1||repetitionsPerSample>128) {
        Fail(error,@"Requires physical A20 Pro, a local compiled library, 3–15 samples and 1–128 repetitions.");return nil;
    }
    if(![device supportsFamily:MTLGPUFamilyMetal4]||
       !([device supportsFamily:MTLGPUFamilyApple10]||[device supportsFamily:MTLGPUFamilyApple11])||
       !TensorType(device,MTLTensorDataTypeFloat16)||!TensorType(device,MTLTensorDataTypeMetalFloat8E4M3)||
       !TensorType(device,MTLTensorDataTypeMetalFloat8E5M2)) {
        Fail(error,@"Required Metal/tensor allocation capabilities are unavailable.");return nil;
    }
    id<MTLLibrary> library=[device newLibraryWithURL:libraryURL error:error];if(!library)return nil;
    std::array<id<MTLComputePipelineState>,3> gemm;
    std::array<id<MTLComputePipelineState>,2> pack;
    for(int mode=0;mode<3;++mode) {
        NSString* name=[@"netal_lab_" stringByAppendingString:FormatName(mode)];
        id<MTLFunction> function=[library newFunctionWithName:name];
        if(!function){Fail(error,@"A required matmul entry is missing.");return nil;}
        gemm[mode]=[device newComputePipelineStateWithFunction:function error:error];
        if(!gemm[mode])return nil;
        if(gemm[mode].maxTotalThreadsPerThreadgroup<128||gemm[mode].threadExecutionWidth!=32){Fail(error,@"This GPU does not support the lab's threadgroup layout.");return nil;}
        if(mode) {
            function=[library newFunctionWithName:[@"netal_lab_pack_" stringByAppendingString:FormatName(mode)]];
            if(!function){Fail(error,@"A required packing entry is missing.");return nil;}
            pack[mode-1]=[device newComputePipelineStateWithFunction:function error:error];if(!pack[mode-1])return nil;
        }
    }
    id<MTLCommandQueue> queue=[device newCommandQueue];
    if(!queue){Fail(error,@"Could not create a GPU queue.");return nil;}
    NSMutableArray* records=[NSMutableArray array];
    auto stop=[&](){
        if(shouldStop&&shouldStop()){Fail(error,@"Stopped between completed GPU commands because the caller requested cancellation or the app became inactive.");return true;}
        if(NSProcessInfo.processInfo.thermalState==NSProcessInfoThermalStateCritical){Fail(error,@"Stopped between completed GPU commands because the phone is critically hot.");return true;}
        return false;
    };
    if(stop())return nil;
    const NSInteger thermalStart=NSProcessInfo.processInfo.thermalState;
    // Verify saturation and subnormal conversion independently of GEMM. Model
    // fixtures remain bounded and never depend on a saturated input.
    Buffer special=MakeBuffer(device,16*sizeof(float)),specialPacked=MakeBuffer(device,16),specialDecoded=MakeBuffer(device,16*sizeof(float));
    if(!special.value||!specialPacked.value||!specialDecoded.value){Fail(error,@"Allocation failed.");return nil;}
    const float specialValues[]={0,-0.0f,1,-1,1.0625f,1.1875f,0.001953125f,-0.001953125f,448,-448,57344,-57344,1e10f,-1e10f,0.0000152587890625f,-0.0000152587890625f};
    std::memcpy(Data<float>(special),specialValues,sizeof(specialValues));
    NSMutableArray* packingChecks=[NSMutableArray array];
    for(int mode=1;mode<3;++mode) {
        if(stop())return nil;
        double time;NSUInteger saturated=0;
        if(!Submit(queue,@[special.value,specialPacked.value,specialDecoded.value],^(id<MTLComputeCommandEncoder> e){
            EncodePack(e,pack[mode-1],special,specialPacked,specialDecoded,1.0f);
        },&time,error))return nil;
        const bool ok=PackCheck(special,specialPacked,specialDecoded,1.0f,mode==1?Format::E4M3:Format::E5M2,&saturated);
        [packingChecks addObject:@{@"format":FormatName(mode),@"passed":@(ok),@"saturated_finite_inputs":@(saturated),@"gpu_ms":@(time)}];
        if(!ok){Fail(error,@"GPU FP8 packing disagrees with the independent OFP8 reference.");return nil;}
    }
    for(const auto& shape:kShapes)for(int fixture=0;fixture<2;++fixture) {
        if(stop())return nil;
        Params params={1024,shape[1],shape[0],(shape[0]+31)/32*32,0,0,0,0.5f,0.25f};
        params.aStride=(params.paddedK+127)/128*128;params.bStride=(params.n+127)/128*128;params.yStride=(params.n+31)/32*32;
        const NSUInteger ac=params.m*params.aStride,bc=params.paddedK*params.bStride,yc=params.m*params.yStride;
        Buffer a=MakeBuffer(device,ac*sizeof(float)),b=MakeBuffer(device,bc*sizeof(float)),bias=MakeBuffer(device,params.yStride*sizeof(float));
        std::array<Buffer,3> aa,bb,yy;std::array<Buffer,2> da,db;
        NSMutableArray* resources=[NSMutableArray array];
        for(int mode=0;mode<3;++mode) {
            aa[mode]=MakeBuffer(device,ac*(mode?1:2));bb[mode]=MakeBuffer(device,bc*(mode?1:2));yy[mode]=MakeBuffer(device,yc*sizeof(float));
            if(mode){da[mode-1]=MakeBuffer(device,ac*sizeof(float));db[mode-1]=MakeBuffer(device,bc*sizeof(float));}
        }
        auto retain=[&](const Buffer& v){if(v.value)[resources addObject:v.value];return v.value!=nil;};
        bool allocated=retain(a)&&retain(b)&&retain(bias);
        for(int mode=0;mode<3;++mode){allocated=retain(aa[mode])&&retain(bb[mode])&&retain(yy[mode])&&allocated;if(mode)allocated=retain(da[mode-1])&&retain(db[mode-1])&&allocated;}
        if(!allocated){Fail(error,@"Fixture buffers could not be allocated.");return nil;}
        NSUInteger totalAllocated=0;for(id<MTLBuffer> resource in resources)totalAllocated+=resource.length;
        for(int mode=0;mode<3;++mode)for(unsigned row=0;row<params.m;++row)for(unsigned col=0;col<params.n;++col)
            Data<float>(yy[mode])[row*params.yStride+col]=std::numeric_limits<float>::quiet_NaN();
        std::fill(Data<float>(a),Data<float>(a)+ac,0);std::fill(Data<float>(b),Data<float>(b)+bc,0);
        for(unsigned row=0;row<params.m;++row)for(unsigned k=0;k<params.k;++k)Data<float>(a)[row*params.aStride+k]=Fixture(row,k,1,fixture)*params.scaleA;
        for(unsigned k=0;k<params.k;++k)for(unsigned col=0;col<params.n;++col)Data<float>(b)[k*params.bStride+col]=Fixture(k,col,3,fixture)*params.scaleB;
        for(unsigned col=0;col<params.yStride;++col)Data<float>(bias)[col]=float(int(col%11)-5)/32.0f;
        for(NSUInteger i=0;i<ac;++i)Data<_Float16>(aa[0])[i]=_Float16(Data<float>(a)[i]);
        for(NSUInteger i=0;i<bc;++i)Data<_Float16>(bb[0])[i]=_Float16(Data<float>(b)[i]);
        std::array<double,2> packingMS{};
        std::array<NSUInteger,2> saturation{};
        for(int mode=1;mode<3;++mode) {
            if(stop())return nil;
            if(!Submit(queue,resources,^(id<MTLComputeCommandEncoder> e){
                EncodePack(e,pack[mode-1],a,aa[mode],da[mode-1],params.scaleA);
                EncodePack(e,pack[mode-1],b,bb[mode],db[mode-1],params.scaleB);
            },&packingMS[mode-1],error))return nil;
            const Format format=mode==1?Format::E4M3:Format::E5M2;
            if(!PackCheck(a,aa[mode],da[mode-1],params.scaleA,format,&saturation[mode-1])||
               !PackCheck(b,bb[mode],db[mode-1],params.scaleB,format,&saturation[mode-1])||saturation[mode-1]) {
                Fail(error,@"Bounded fixture packing, saturation, or guards failed.");return nil;
            }
        }
        std::array<std::vector<double>,3> times;
        std::array<std::vector<double>,2> combined;
        // One warm-up each. Rotating order limits systematic order/heat bias.
        for(NSUInteger sample=0;sample<=sampleCount;++sample)for(int order=0;order<3;++order) {
            if(stop())return nil;
            const int mode=int((sample+order)%3);Params run=params;if(!mode){run.scaleA=1;run.scaleB=1;}
            double ms;
            if(!Submit(queue,resources,^(id<MTLComputeCommandEncoder> e){
                for(NSUInteger repeat=0;repeat<repetitionsPerSample;++repeat){
                    EncodeMatmul(e,gemm[mode],aa[mode],bb[mode],yy[mode],bias,run);
                    [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
                }
            },&ms,error))return nil;
            if(sample)times[mode].push_back(ms/repetitionsPerSample);
            if(mode&&sample) {
                if(stop())return nil;
                if(!Submit(queue,resources,^(id<MTLComputeCommandEncoder> e){
                    for(NSUInteger repeat=0;repeat<repetitionsPerSample;++repeat){
                        EncodePack(e,pack[mode-1],a,aa[mode],da[mode-1],params.scaleA);
                        EncodePack(e,pack[mode-1],b,bb[mode],db[mode-1],params.scaleB);
                        [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
                        EncodeMatmul(e,gemm[mode],aa[mode],bb[mode],yy[mode],bias,run);
                        [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    }
                },&ms,error))return nil;
                combined[mode-1].push_back(ms/repetitionsPerSample);
            }
        }
        for(int mode=0;mode<3;++mode) {
            bool guards=Guards(a)&&Guards(b)&&Guards(bias)&&Guards(aa[mode])&&Guards(bb[mode])&&Guards(yy[mode]);
            if(mode)guards=guards&&Guards(da[mode-1])&&Guards(db[mode-1]);
            NSUInteger nonfinite=0;
            for(unsigned row=0;row<params.m;++row) {
                for(unsigned col=0;col<params.n;++col)if(!std::isfinite(Data<float>(yy[mode])[row*params.yStride+col]))++nonfinite;
                guards=guards&&Guard(reinterpret_cast<std::uint8_t*>(Data<float>(yy[mode])+row*params.yStride+params.n),(params.yStride-params.n)*sizeof(float));
            }
            Error original,quantized;
            for(unsigned row:kOracleRows)for(unsigned col=0;col<params.n;++col) {
                double full=Data<float>(bias)[col],quant=full;
                for(unsigned k=0;k<params.k;++k) {
                    full+=double(Data<float>(a)[row*params.aStride+k])*Data<float>(b)[k*params.bStride+col];
                    quant+=mode?double(Data<float>(da[mode-1])[row*params.aStride+k])*Data<float>(db[mode-1])[k*params.bStride+col]:
                        double(Data<_Float16>(aa[0])[row*params.aStride+k])*double(Data<_Float16>(bb[0])[k*params.bStride+col]);
                }
                const double actual=Data<float>(yy[mode])[row*params.yStride+col];original.Add(actual,full);quantized.Add(actual,quant);
            }
            // Quantized operands are binary fractions; this loose numerical
            // bound checks matmul/epilogue units, not perceptual image quality.
            const bool correct=guards&&!nonfinite&&!quantized.nonfinite&&quantized.maximum<=0.002;
            NSMutableDictionary* record=[@{@"m":@(params.m),@"k":@(params.k),@"n":@(params.n),@"padded_k":@(params.paddedK),
                @"fixture":fixture?@"bounded_noise":@"exact_grid",@"arithmetic":FormatName(mode),@"passed":@(correct),
                @"gpu_ms_per_dispatch_samples":[NSMutableArray array],@"gpu_ms_per_dispatch_median":@(Median(times[mode])),
                @"original_input_error":ErrorRecord(original),@"quantized_input_math_error":ErrorRecord(quantized),
                @"guards_and_output_padding_intact":@(guards),@"all_output_nonfinite_count":@(nonfinite),
                @"bias_after_scale_product":@YES,@"scale_a":@(mode?params.scaleA:1),@"scale_b":@(mode?params.scaleB:1),
                @"input_storage_bytes":@(aa[mode].bytes+bb[mode].bytes),@"allocation_bytes_with_output_and_guards":@(aa[mode].value.length+bb[mode].value.length+yy[mode].value.length)} mutableCopy];
            record[@"original_fp32_source_bytes"]=@(a.bytes+b.bytes);
            record[@"all_fixture_allocations_bytes"]=@(totalAllocated);
            record[@"diagnostic_fp32_decode_bytes"]=@(mode?da[mode-1].bytes+db[mode-1].bytes:0);
            for(double ms:times[mode])[record[@"gpu_ms_per_dispatch_samples"] addObject:@(ms)];
            if(mode){
                record[@"quantize_both_operands_gpu_ms"]=@(packingMS[mode-1]);record[@"saturated_fixture_inputs"]=@(saturation[mode-1]);
                record[@"quantize_and_matmul_gpu_ms_median"]=@(Median(combined[mode-1]));
                NSMutableArray* samples=[NSMutableArray array];for(double ms:combined[mode-1])[samples addObject:@(ms)];
                record[@"quantize_and_matmul_gpu_ms_samples"]=samples;
                record[@"source_fp8_to_half_fallback"]=@NO;
            }
            [records addObject:record];
            if(!correct){Fail(error,@"Synthetic matmul failed the CPU math oracle, finite-output, or canary check.");return nil;}
        }
    }
    return @{@"schema":@"netal.fp8.matmul-lab.v1",@"state":@"synthetic_operator_pass",@"gpu_name":device.name,
        @"os_version":NSProcessInfo.processInfo.operatingSystemVersionString,@"minimum_os":@"27.0",@"msl_version":@"4.1",
        @"fp8_tensor_allocation":@YES,@"fp8_input_tensorops_executed":@YES,@"hardware_native_fp8_instruction_verified":@NO,
        @"full_sr_validated":@NO,@"dlss_model_executed":@NO,@"latency_target_validated":@NO,
        @"timing_source":@"MTLCommandBuffer driver GPUStartTime/GPUEndTime",@"timing_scope":@"amortized dispatch plus buffer barrier; combined includes both packing kernels and FP32 unpack diagnostics",
        @"warmup_samples_per_mode":@1,@"measured_samples_per_mode":@(sampleCount),@"repetitions_per_sample":@(repetitionsPerSample),
        @"oracle_rows":@[@0,@1,@7,@31,@63,@64,@127,@1023],@"oracle_columns":@"all logical columns",
        @"all_outputs_checked_finite":@YES,@"math_error_limit_absolute":@0.002,
        @"thermal_state_start":@(thermalStart),@"thermal_state_end":@(NSProcessInfo.processInfo.thermalState),
        @"packing_checks":packingChecks,@"records":records};
#endif
}
