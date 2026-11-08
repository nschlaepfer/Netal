// Source-only, real-model SR execution probe. No model, fixture tensors or game assets are embedded.
#import <UIKit/UIKit.h>
#import <Metal/Metal.h>
#import <NetalIOSSR/NetalIOSSR.h>
#import <CommonCrypto/CommonDigest.h>
#import <mach/mach.h>
#include "NetalProbeMetrics.h"
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>

namespace {
using Clock = std::chrono::steady_clock;
constexpr NSUInteger kGuard = 256;
constexpr std::uint16_t kOutputPoison = 0x7e00; // quiet NaN: untouched output cannot pass finite checks.
constexpr std::array<std::array<float, 2>, 18> kJitter = {{
    {-0.3125f,-0.351851851f},{0.1875f,-0.0185185075f},
    {-0.0625f,0.314814806f},{0.4375f,-0.240740746f},
    {-0.46875f,0.092592597f},{0.03125f,0.42592591f},
    {-0.21875f,-0.42592591f},{0.0f,-0.166666657f},
    {-0.25f,0.166666687f},{0.25f,-0.388888896f},
    {-0.375f,-0.0555555522f},{0.125f,0.277777791f},
    {-0.125f,-0.277777791f},{0.375f,0.055555582f},
    {-0.4375f,0.388888896f},{0.0625f,-0.462962955f},
    {-0.1875f,-0.129629612f},{0.3125f,0.203703701f}}};

double Elapsed(Clock::time_point start) {
    return std::chrono::duration<double,std::milli>(Clock::now()-start).count();
}
id Number(double value) { return std::isfinite(value) ? static_cast<id>(@(value)) : NSNull.null; }
NSDictionary* Summary(const std::vector<double>& samples) {
    return @{ @"samples":@(samples.size()), @"p50":Number(netal_probe::Percentile(samples,0.5)),
              @"p95":Number(netal_probe::Percentile(samples,0.95)),
              @"p99":Number(netal_probe::Percentile(samples,0.99)) };
}
NSString* SHA(const void* bytes, NSUInteger count) {
    CC_SHA256_CTX context; CC_SHA256_Init(&context);
    const auto* cursor=static_cast<const unsigned char*>(bytes);
    for (NSUInteger left=count; left;) {
        CC_LONG part=static_cast<CC_LONG>(MIN(left,static_cast<NSUInteger>(1u<<30)));
        CC_SHA256_Update(&context,cursor,part);cursor+=part;left-=part;
    }
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];CC_SHA256_Final(digest,&context);
    NSMutableString* value=[NSMutableString string];
    for (unsigned char byte : digest) [value appendFormat:@"%02x",byte];
    return value;
}
NSString* Thermal() {
    switch (NSProcessInfo.processInfo.thermalState) {
        case NSProcessInfoThermalStateNominal:return @"nominal";
        case NSProcessInfoThermalStateFair:return @"fair";
        case NSProcessInfoThermalStateSerious:return @"serious";
        case NSProcessInfoThermalStateCritical:return @"critical";
    }
    return @"unknown";
}
id Footprint() {
    task_vm_info_data_t info={};mach_msg_type_number_t count=TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(),TASK_VM_INFO,reinterpret_cast<task_info_t>(&info),&count)!=KERN_SUCCESS)
        return NSNull.null;
    return @(info.phys_footprint);
}
NSURL* Documents() {
    return [NSFileManager.defaultManager URLsForDirectory:NSDocumentDirectory inDomains:NSUserDomainMask].firstObject;
}
NSDictionary* JSON(NSURL* url) {
    NSData* data=[NSData dataWithContentsOfURL:url];
    id value=data ? [NSJSONSerialization JSONObjectWithData:data options:0 error:nil] : nil;
    return [value isKindOfClass:NSDictionary.class] ? value : nil;
}
BOOL WriteReceipt(NSDictionary* receipt) {
    NSData* data=[NSJSONSerialization dataWithJSONObject:receipt options:NSJSONWritingPrettyPrinted|NSJSONWritingSortedKeys error:nil];
    return data && [data writeToURL:[Documents() URLByAppendingPathComponent:@"probe-receipt.json"] options:NSDataWritingAtomic error:nil];
}
NSDictionary* SaveCompletedOutput(id<MTLBuffer> output,NSUInteger outputBytes,NSDictionary* frame) {
    if(!frame)return @{@"requested":@YES,@"state":@"unavailable",@"reason":@"No frame completed."};
    // Called after the timing run (or a between-frame stop), with no GPU work in flight.
    // Only the caller's procedural model output is saved, inside this app's Documents directory.
    NSData* data=[NSData dataWithBytesNoCopy:static_cast<unsigned char*>(output.contents)+kGuard
                                    length:outputBytes freeWhenDone:NO];
    NSError* error=nil;
    if(![data writeToURL:[Documents() URLByAppendingPathComponent:@"probe-output.rgba16le"] options:NSDataWritingAtomic error:&error])
        return @{@"requested":@YES,@"state":@"failed",@"reason":error.localizedDescription?:@"Output snapshot could not be saved."};
    return @{@"requested":@YES,@"state":@"written",@"path":@"probe-output.rgba16le",
        @"format":@"RGBA16F little-endian, tightly packed, no header",@"bytes":@(outputBytes),
        @"sha256":SHA(data.bytes,data.length),@"frame_index":frame[@"index"],
        @"reset":frame[@"reset"],@"jitter":frame[@"jitter"],@"written_after_timing_run":@YES};
}
bool BoundedInteger(id object, NSUInteger fallback, NSUInteger minimum, NSUInteger maximum, NSUInteger* result) {
    if (!object) {*result=fallback;return true;}
    if (![object isKindOfClass:NSNumber.class] || CFGetTypeID((__bridge CFTypeRef)object)==CFBooleanGetTypeID())return false;
    const double value=[object doubleValue];
    if (!std::isfinite(value) || value!=std::floor(value) || value<minimum || value>maximum)return false;
    *result=static_cast<NSUInteger>(value);return true;
}
bool Canary(const unsigned char* bytes, NSUInteger begin, NSUInteger end) {
    for (NSUInteger i=begin;i<end;++i)if(bytes[i]!=0xa5)return false;
    return true;
}

BOOL MPPDescriptorValid(NSDictionary* doc,NSUInteger* gemms) {
    if(![doc[@"library"] isEqual:@"pwin_mpp_all"] || ![doc[@"gemm_backend"] isEqual:@"mpp"] ||
       ![doc[@"dispatches"] isKindOfClass:NSArray.class])return NO;
    NSUInteger found=0;
    for(id raw in doc[@"dispatches"]) {
        if(![raw isKindOfClass:NSDictionary.class])return NO;
        NSDictionary* dispatch=raw;NSString* function=dispatch[@"function"];
        if(![function isKindOfClass:NSString.class])return NO;
        if(![function hasPrefix:@"pwin_gemm"])continue;
        if(![function isEqual:@"pwin_gemm_mpp"] || ![dispatch[@"block"] isEqual:@[@128,@1,@1]])return NO;
        NSDictionary* constants=dispatch[@"constants"];
        NSArray* byteArguments=dispatch[@"bytes"];
        if(![constants isKindOfClass:NSDictionary.class] || ![byteArguments isKindOfClass:NSArray.class])return NO;
        NSArray* params=nil;NSUInteger parameterBlocks=0;
        for(NSDictionary* argument in byteArguments)if([argument[@"index"] isEqual:@4]){params=argument[@"u32"];++parameterBlocks;}
        if(parameterBlocks!=1 || ![params isKindOfClass:NSArray.class] || params.count!=16)return NO;
        for(id number in params)if(![number isKindOfClass:NSNumber.class] ||
            CFGetTypeID((__bridge CFTypeRef)number)==CFBooleanGetTypeID() ||
            [number doubleValue]<0 || [number doubleValue]>UINT32_MAX || [number doubleValue]!=[number unsignedIntValue])return NO;
        const NSUInteger tokens=[params[0] unsignedIntegerValue],k=[params[1] unsignedIntegerValue],n=[params[2] unsignedIntegerValue];
        if(!tokens || !k || !n || ![constants[@"0"] isEqual:params[1]] || ![constants[@"1"] isEqual:params[2]] ||
           ![params[15] isEqual:@0])return NO;
        const NSUInteger columns=[params[12] unsignedIntegerValue]?:n;
        if(![dispatch[@"grid"] isEqual:@[@((tokens+63)/64),@((columns+31)/32),@1]])return NO;
        ++found;
    }
    if(gemms)*gemms=found;return found>0;
}

// A translating, periodic plane: non-zero normalized current-to-previous motion, inverted depth,
// linear HDR color, exposure and a supported jitter sequence. This is an execution stimulus,
// not a replacement for an independent temporal image-quality oracle.
void Inputs(id<MTLBuffer> color,id<MTLBuffer> motion,id<MTLBuffer> depth,id<MTLBuffer> exposure,
            NSUInteger width,NSUInteger height,NSUInteger frame,float jitterX,float jitterY) {
    auto* c=static_cast<_Float16*>(color.contents);
    auto* m=static_cast<_Float16*>(motion.contents);
    auto* d=static_cast<float*>(depth.contents);
    for (NSUInteger y=0;y<height;++y)for(NSUInteger x=0;x<width;++x) {
        const NSUInteger index=y*width+x;
        const float sx=static_cast<float>(x)+jitterX-static_cast<float>(frame);
        const float sy=static_cast<float>(y)+jitterY;
        const float wave=0.5f+0.5f*std::sin(sx*0.075f)*std::cos(sy*0.075f);
        c[index*4]=static_cast<_Float16>(wave*2.0f);
        c[index*4+1]=static_cast<_Float16>(0.25f+0.7f*wave);
        c[index*4+2]=static_cast<_Float16>(1.0f-wave*0.8f);
        c[index*4+3]=static_cast<_Float16>(1.0f);
        m[index*2]=static_cast<_Float16>(-1.0f/static_cast<float>(width));
        m[index*2+1]=static_cast<_Float16>(0.0f);d[index]=0.75f;
    }
    *static_cast<float*>(exposure.contents)=1.0f;
}

NSMutableDictionary* RunProbe(void (^progress)(NSString*),std::shared_ptr<std::atomic<int>> stopFlag) {
    NSMutableDictionary* receipt=[@{
        @"schema":@"netal.ios.sr.device-probe.v1", @"state":@"unavailable",
        @"started_utc":[[NSISO8601DateFormatter new] stringFromDate:NSDate.date],
        @"scope":@"SR temporal buffer-to-buffer; boundary transforms, model, history and output included",
        @"excludes":@[@"game rendering",@"texture import/export",@"presentation",@"CPU procedural input generation"],
        @"stimulus":@"procedural translating HDR plane; current-to-previous motion; inverted depth; exposure=1; 18-phase jitter",
        @"cpu_work_scope":@"CPU encode includes session encoding only; completion wall includes encode, submit and callback wait. Procedural input generation, output validation, hashing and receipt writes are excluded; this serial probe is not gameplay throughput.",
        @"output_poison_half_bits":@"0x7e00",
        @"rtx_parity_validated":@NO,@"visual_quality_validated":@NO,@"gameplay_validated":@NO,
        @"hardware_acceleration_validated":@NO,@"latency_target_validated":@NO,
        @"execution_smoke_validated":@NO,@"gpu_timing_validated":@NO,
        @"dlss_4_5_supported":@NO,@"os":NSProcessInfo.processInfo.operatingSystemVersionString,
        @"thermal_start":Thermal(),@"memory_footprint_start_bytes":Footprint(),
        @"gpu_full_pass_ms":Summary({}),@"cpu_encode_ms":Summary({}),@"completion_wall_ms":Summary({})
    } mutableCopy];
    auto fail=[&](NSString* reason,BOOL restart=NO) -> NSMutableDictionary* {
        if([receipt[@"state"] isEqual:@"running"])receipt[@"state"]=@"failed";
        receipt[@"reason"]=reason;receipt[@"restart_required"]=@(restart);return receipt;
    };
    // A launch scheme may inherit developer diagnostics that repeat passes, drop barriers or
    // change arithmetic. Do not silently benchmark those overrides under a normal backend label.
    NSMutableArray* clearedEnvironment=[NSMutableArray array];
    for(NSString* key in NSProcessInfo.processInfo.environment)if([key hasPrefix:@"NEURALSTUDIO_SR_"]){
        [clearedEnvironment addObject:key];unsetenv(key.UTF8String);
    }
    receipt[@"cleared_sr_environment_keys"]=[clearedEnvironment sortedArrayUsingSelector:@selector(compare:)];
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();
    NetalIOSSRCapabilities* capability=[NetalIOSSRCapabilities capabilitiesForDevice:device];
    receipt[@"hardware"]=@{@"gpu":device.name?:@"unavailable",@"chip":capability.chipName,
        @"eligible":@(capability.eligible),@"reason":capability.reason,
        @"neural_accelerator_candidate":@(capability.supportsNeuralAccelerators),
        @"fp8_tensor_allocation_supported":@(capability.supportsFP8TensorTypes)};
    if(!capability.eligible)return fail(capability.reason);
    NSURL* root=[Documents() URLByAppendingPathComponent:@"runtime" isDirectory:YES];
    NSDictionary* manifest=JSON([root URLByAppendingPathComponent:@"manifest.json"]);
    if(!manifest)return fail(@"Add your supported runtime bundle to Documents/runtime, then run again.");
    if(![manifest[@"schema"] isEqual:@"netal.sr.runtime.v2"] || ![manifest[@"model"] isEqual:@"DLSS_SR_Feature1_Quality_K"])
        return fail(@"This probe requires the supported DLSS 4 K runtime. DLSS 4.5 is unavailable.");
    if(![manifest[@"files"] isKindOfClass:NSArray.class])return fail(@"The runtime asset inventory is missing.");
    NSMutableArray* modelAssets=[NSMutableArray array];
    for(id item in manifest[@"files"])if([item isKindOfClass:NSDictionary.class] &&
        [item[@"path"] isKindOfClass:NSString.class] && [item[@"path"] hasPrefix:@"model/"])
        [modelAssets addObject:item];
    receipt[@"model"]=@{@"name":manifest[@"model"],@"asset_root_sha256":manifest[@"asset_root_sha256"]?:NSNull.null,
        @"assets":modelAssets,@"identity_validated":@NO};
    NSURL* configURL=[Documents() URLByAppendingPathComponent:@"benchmark.json"];
    NSDictionary* config=JSON(configURL);
    if([NSFileManager.defaultManager fileExistsAtPath:configURL.path] && !config)
        return fail(@"benchmark.json is not a JSON object.");
    config=config?:@{};
    bool saveLastOutput=false;
    if(!netal_probe::ReadOptionalBoolean(config[@"save_last_output"],false,&saveLastOutput))
        return fail(@"save_last_output must be a JSON boolean (true or false).");
    receipt[@"side_output"]=saveLastOutput?
        @{@"requested":@YES,@"state":@"unavailable",@"reason":@"No completed output was saved."}:
        @{@"requested":@NO,@"state":@"disabled"};
    NSString* profile=config[@"profile"]?:@"1280x720";
    NSString* backend=config[@"backend"]?:@"batched";
    if([backend isEqual:@"mpp"])backend=@"mpp-candidate";
    NSString* api=config[@"metal_api"]?:@"metal4";
    NSRegularExpression* size=[NSRegularExpression regularExpressionWithPattern:@"^[1-9][0-9]{1,3}x[1-9][0-9]{1,3}$" options:0 error:nil];
    if(![profile isKindOfClass:NSString.class] || [size numberOfMatchesInString:profile options:0 range:NSMakeRange(0,profile.length)]!=1)
        return fail(@"Use a supported output profile such as 1280x720.");
    if(![@[@"reference",@"batched",@"mpp-candidate"] containsObject:backend] ||
       ![@[@"metal3",@"metal4"] containsObject:api])return fail(@"Unknown backend or Metal API in benchmark.json.");
    NSUInteger warmup=0,measured=0;
    if(!BoundedInteger(config[@"warmup_frames"],3,1,16,&warmup) ||
       !BoundedInteger(config[@"measured_frames"],60,1,600,&measured))
        return fail(@"Use 1–16 warm-up frames and 1–600 measured frames.");
    setenv("NEURALSTUDIO_SR_PROFILE",profile.UTF8String,1);
    setenv("NEURALSTUDIO_SR_PWIN_PIPELINE",[backend isEqual:@"reference"]?"0":"1",1);
    // Explicit selection avoids inherited environment accidentally changing the arithmetic route.
    unsetenv("NEURALSTUDIO_SR_PWIN_SUFFIX");
    unsetenv("NEURALSTUDIO_SR_PROFILE_SYNTHESIZE");
    unsetenv("NEURALSTUDIO_SR_RENDER");
    receipt[@"configuration"]=@{@"profile":profile,@"requested_backend":backend,@"metal_api":api,
        @"warmup_frames":@(warmup),@"measured_frames":@(measured),@"serial_single_in_flight":@YES,
        @"save_last_output":@(saveLastOutput),
        @"per_frame_timeout_seconds":@30,@"run_deadline_seconds":@180};
    progress(@"Preparing the model…");
    NSError* error=nil;
    const auto prepareStart=Clock::now();
    NeuralStudioDLSSSRSession* session=[[NeuralStudioDLSSSRSession alloc] initWithDevice:device runtimeBundleURL:root error:&error];
    receipt[@"cold_prepare_wall_ms"]=@(Elapsed(prepareStart));
    if(!session)return fail([@"The runtime could not be prepared: " stringByAppendingString:error.localizedDescription?:@"unknown error"]);
    receipt[@"model"]=@{@"name":manifest[@"model"],@"asset_root_sha256":session.runtimeBundleRoot,
        @"assets":modelAssets,@"identity_validated":@YES};
    receipt[@"geometry"]=@{@"input_width":@(session.inputWidth),@"input_height":@(session.inputHeight),
        @"output_width":@(session.outputWidth),@"output_height":@(session.outputHeight)};
    receipt[@"pipeline_layers"]=@{@"total":@(session.pipelineLayersTotal),@"present":@(session.pipelineLayersPresent),
        @"missing":session.pipelineLayersMissing,@"fallback_visible":@YES};
    // Record exactly what the selected profile advertises. Loaded layer count above detects missing
    // dispatch sidecars. Sidecars do not prove Neural Accelerator instruction execution.
    NSMutableArray* pipelines=[NSMutableArray array];
    NSString* profilePrefix=[NSString stringWithFormat:@"profiles/%@/",profile];
    const BOOL synthesized=![NSFileManager.defaultManager fileExistsAtPath:[root URLByAppendingPathComponent:[profilePrefix stringByAppendingString:@"manifest.json"]].path];
    NSString* pipelinePrefix=synthesized?[NSString stringWithFormat:@"pipeline-synth/%@/",profile]:[profilePrefix stringByAppendingString:@"pipeline/"];
    receipt[@"geometry_synthesized"]=@(synthesized);
    receipt[@"selected_pipeline_directory"]=pipelinePrefix;
    BOOL sawMPP=NO,sawSIMD=NO;
    BOOL allMPPDescriptorsValid=YES;NSUInteger mppGEMMs=0;
    NSMutableSet* auditedMPPPaths=[NSMutableSet set];
    for(NSDictionary* item in manifest[@"files"]) {
        NSString* path=item[@"path"];
        if(![path hasPrefix:pipelinePrefix] || ![path hasSuffix:@".json"])continue;
        NSDictionary* sidecar=JSON([root URLByAppendingPathComponent:path]);
        NSString* mode=[sidecar[@"gemm_backend"] isKindOfClass:NSString.class]?sidecar[@"gemm_backend"]:@"unspecified";
        if([mode containsString:@"mpp"])sawMPP=YES;else sawSIMD=YES;
        NSUInteger gemms=0;const BOOL checked=MPPDescriptorValid(sidecar,&gemms);
        allMPPDescriptorsValid&=checked;if(checked){mppGEMMs+=gemms;[auditedMPPPaths addObject:path];}
        [pipelines addObject:@{@"path":path,@"sha256":item[@"sha256"]?:NSNull.null,@"gemm_backend_metadata":mode,
            @"mpp_descriptor_validated":@(checked),@"mpp_gemm_dispatches":@(gemms)}];
    }
    // Missing shifted phases otherwise reuse phase zero in the session. Require all known shifted
    // phases for an MPP claim rather than silently timing a partly converted profile.
    for(NSString* stage in @[@"enc0",@"enc1",@"enc2",@"enc3",@"enc4",@"dec5",@"dec4",@"dec3",@"dec2",@"dec1",@"dec0"]) {
        NSString* entry=[NSString stringWithFormat:@"dltss_pwin_%@_layer",stage];
        if(![auditedMPPPaths containsObject:[pipelinePrefix stringByAppendingFormat:@"%@.json",entry]])allMPPDescriptorsValid=NO;
        if([@[@"enc3",@"enc4",@"dec5",@"dec4",@"dec3"] containsObject:stage])for(NSUInteger phase=1;phase<8;++phase)
            if(![auditedMPPPaths containsObject:[pipelinePrefix stringByAppendingFormat:@"%@.phase%lu.json",entry,static_cast<unsigned long>(phase)]])allMPPDescriptorsValid=NO;
    }
    receipt[@"pipeline_sidecars"]=pipelines;
    receipt[@"backend_mode"]=[backend isEqual:@"reference"]?@"transpiled_reference":
        (session.pipelineLayersPresent!=session.pipelineLayersTotal?@"mixed_pipeline_and_reference_fallback":
        (sawMPP?(sawSIMD?@"mixed_mpp_and_other_pipeline_candidates":@"mpp_pipeline_candidate"):@"batched_pipeline_candidate"));
    receipt[@"backend_dispatch_verified"]=@NO;
    receipt[@"mpp_selected_descriptors_validated"]=@(allMPPDescriptorsValid&&mppGEMMs>0);
    receipt[@"mpp_descriptor_gemm_dispatches"]=@(mppGEMMs);
    if([backend isEqual:@"mpp-candidate"] && (!sawMPP || !allMPPDescriptorsValid || session.pipelineLayersPresent!=11 || session.pipelineLayersPresent!=session.pipelineLayersTotal))
        return fail(@"The selected profile does not contain a complete MPP candidate. No MPP result was recorded.");
    const NSUInteger inputPixels=session.inputWidth*session.inputHeight;
    const NSUInteger outputBytes=session.outputWidth*session.outputHeight*8;
    id<MTLBuffer> color=[device newBufferWithLength:inputPixels*8 options:MTLResourceStorageModeShared];
    id<MTLBuffer> motion=[device newBufferWithLength:inputPixels*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> depth=[device newBufferWithLength:inputPixels*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> exposure=[device newBufferWithLength:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> output=[device newBufferWithLength:outputBytes+2*kGuard options:MTLResourceStorageModeShared];
    if(!color||!motion||!depth||!exposure||!output)return fail(@"The phone could not allocate the working buffers.");
    const BOOL metal4=[api isEqual:@"metal4"];
    id<MTLCommandQueue> queue=metal4?nil:[device newCommandQueue];
    id<MTL4CommandQueue> queue4=metal4?[device newMTL4CommandQueue]:nil;
    id<MTL4CommandAllocator> allocator=metal4?[device newCommandAllocator]:nil;
    MTLResidencySetDescriptor* rd=[MTLResidencySetDescriptor new];rd.initialCapacity=5;
    id<MTLResidencySet> residency=metal4?[device newResidencySetWithDescriptor:rd error:&error]:nil;
    if((!metal4&&!queue)||(metal4&&(!queue4||!allocator||!residency)))return fail(@"The GPU command queue could not be created.");
    if(metal4){for(id<MTLBuffer> b in @[color,motion,depth,exposure,output])[residency addAllocation:b];[residency commit];}
    std::vector<double> gpuTimes,cpuTimes,wallTimes;
    NSMutableArray* frames=[NSMutableArray array];receipt[@"frames"]=frames;
    BOOL safe=YES,timingsValid=YES;
    receipt[@"thermal_before_frames"]=Thermal();receipt[@"memory_footprint_before_frames_bytes"]=Footprint();
    receipt[@"state"]=@"running";
    const auto runStart=Clock::now();
    for(NSUInteger frame=0;frame<warmup+measured;++frame){@autoreleasepool {
        const int requested=stopFlag->load();
        const auto stop=netal_probe::StopBeforeNextFrame(requested==1,requested!=2,
            NSProcessInfo.processInfo.thermalState==NSProcessInfoThermalStateCritical,Elapsed(runStart),180000.0);
        if(stop!=netal_probe::StopReason::None){
            receipt[@"state"]=@"incomplete";receipt[@"encoded_frames"]=@(session.encodedFrameCount);
            receipt[@"thermal_end"]=Thermal();receipt[@"memory_footprint_end_bytes"]=Footprint();
            receipt[@"run_wall_ms"]=@(Elapsed(runStart));
            NSString* reason=@"deadline";
            if(stop==netal_probe::StopReason::Cancelled)reason=@"cancelled";
            if(stop==netal_probe::StopReason::Inactive)reason=@"app_inactive";
            if(stop==netal_probe::StopReason::CriticalThermal)reason=@"critical_thermal";
            receipt[@"stop_reason"]=reason;
            if(saveLastOutput)receipt[@"side_output"]=SaveCompletedOutput(output,outputBytes,frames.lastObject);
            return fail([@"Stopped between frames (" stringByAppendingFormat:@"%@). Partial results were kept.",reason]);
        }
        progress([NSString stringWithFormat:@"Testing frame %lu of %lu…",static_cast<unsigned long>(frame+1),static_cast<unsigned long>(warmup+measured)]);
        const auto jitter=kJitter[frame%kJitter.size()];
        Inputs(color,motion,depth,exposure,session.inputWidth,session.inputHeight,frame,jitter[0],jitter[1]);
        std::memset(output.contents,0xa5,output.length);
        auto* poisonWords=reinterpret_cast<std::uint16_t*>(static_cast<unsigned char*>(output.contents)+kGuard);
        std::fill(poisonWords,poisonWords+outputBytes/2,kOutputPoison);
        dispatch_semaphore_t completed=dispatch_semaphore_create(0);
        __block double gpuStart=0,gpuEnd=0;
        __block NSError* commandError=nil;
        __block BOOL commandCompleted=NO;
        id<MTLCommandBuffer> command=metal4?nil:[queue commandBuffer];
        id<MTL4CommandBuffer> command4=metal4?[device newCommandBuffer]:nil;
        if((metal4&&!command4)||(!metal4&&!command))return fail(@"The GPU command buffer could not be created.");
        // Metal 4 does not retain the allocations it uses. The completion block owns the entire
        // in-flight context, including the session's history/ring, even if the bounded wait times out.
        // Neither a timeout nor a cancel permits these objects to be released or reused early.
        NSArray* inFlightResources=metal4?@[session,color,motion,depth,exposure,output,residency,allocator,queue4,command4]:
            @[session,color,motion,depth,exposure,output,queue,command];
        if(metal4){[command4 beginCommandBufferWithAllocator:allocator];[command4 useResidencySet:residency];}
        const auto wallStart=Clock::now();const auto encodeStart=Clock::now();
        const BOOL reset=frame==0;
        const BOOL encoded=metal4?
            [session encodeFrameToMTL4CommandBuffer:command4 colorBuffer:color colorOffset:0 motionBuffer:motion motionOffset:0 depthBuffer:depth depthOffset:0 exposureBuffer:exposure exposureOffset:0 outputBuffer:output outputOffset:kGuard jitterOffsetX:jitter[0] jitterOffsetY:jitter[1] motionVectorScaleX:session.inputWidth motionVectorScaleY:session.inputHeight reset:reset depthReversed:YES error:&error]:
            [session encodeFrameToCommandBuffer:command colorBuffer:color colorOffset:0 motionBuffer:motion motionOffset:0 depthBuffer:depth depthOffset:0 exposureBuffer:exposure exposureOffset:0 outputBuffer:output outputOffset:kGuard jitterOffsetX:jitter[0] jitterOffsetY:jitter[1] motionVectorScaleX:session.inputWidth motionVectorScaleY:session.inputHeight reset:reset depthReversed:YES error:&error];
        const double cpuMS=Elapsed(encodeStart);
        if(!encoded)return fail([@"The frame could not be encoded: " stringByAppendingString:error.localizedDescription?:@"unknown error"]);
        if(metal4){
            [command4 endCommandBuffer];
            MTL4CommitOptions* options=[MTL4CommitOptions new];
            [options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback){
                (void)[inFlightResources count];
                gpuStart=feedback.GPUStartTime;gpuEnd=feedback.GPUEndTime;commandError=feedback.error;
                commandCompleted=feedback.error==nil;dispatch_semaphore_signal(completed);
            }];
            id<MTL4CommandBuffer> list[]={command4};[queue4 commit:list count:1 options:options];
        }else{
            [command addCompletedHandler:^(id<MTLCommandBuffer> finished){
                (void)[inFlightResources count];
                gpuStart=finished.GPUStartTime;gpuEnd=finished.GPUEndTime;commandError=finished.error;
                commandCompleted=finished.status==MTLCommandBufferStatusCompleted;dispatch_semaphore_signal(completed);
            }];[command commit];
        }
        if(dispatch_semaphore_wait(completed,dispatch_time(DISPATCH_TIME_NOW,30*NSEC_PER_SEC))!=0){
            receipt[@"timed_out_in_flight_resources_retained"]=@YES;
            receipt[@"state"]=@"failed";return fail(@"The GPU did not finish within 30 seconds. Close the app before another test.",YES);
        }
        const double wallMS=Elapsed(wallStart);
        const double gpuMS=netal_probe::GPUMilliseconds(gpuStart,gpuEnd,commandCompleted,commandError!=nil);
        if(metal4)[allocator reset];
        if(!commandCompleted||commandError){receipt[@"state"]=@"failed";return fail(commandError.localizedDescription?:@"The GPU command failed.");}
        const auto* bytes=static_cast<const unsigned char*>(output.contents);
        const auto* pixels=reinterpret_cast<const _Float16*>(bytes+kGuard);
        const BOOL canaries=Canary(bytes,0,kGuard)&&Canary(bytes,kGuard+outputBytes,output.length);
        const auto* words=reinterpret_cast<const std::uint16_t*>(bytes+kGuard);
        NSUInteger nonfinite=0,changed=0,untouched=0;
        for(NSUInteger i=0;i<outputBytes/2;++i)if(!std::isfinite(static_cast<float>(pixels[i])))++nonfinite;
        for(NSUInteger i=0;i<outputBytes/2;++i)if(words[i]==kOutputPoison)++untouched;else changed+=2;
        safe&=canaries&&nonfinite==0&&untouched==0&&changed==outputBytes;
        timingsValid&=std::isfinite(gpuMS);
        NSMutableDictionary* row=[@{@"index":@(frame),@"warmup":@(frame<warmup),@"reset":@(reset),
            @"jitter":@[@(jitter[0]),@(jitter[1])],@"gpu_ms":Number(gpuMS),
            @"gpu_timing_status":std::isfinite(gpuMS)?@"measured":@"unavailable",
            @"cpu_encode_ms":@(cpuMS),@"completion_wall_ms":@(wallMS),
            @"canaries_intact":@(canaries),@"nonfinite_components":@(nonfinite),
            @"untouched_poison_components":@(untouched),
            @"changed_output_bytes":@(changed),@"output_sha256":SHA(bytes+kGuard,outputBytes),
            @"memory_footprint_bytes":Footprint(),@"thermal":Thermal()} mutableCopy];
        [frames addObject:row];
        if(frame==0)receipt[@"cold_first_frame"]=row;
        if(frame>=warmup){gpuTimes.push_back(gpuMS);cpuTimes.push_back(cpuMS);wallTimes.push_back(wallMS);}
        receipt[@"gpu_full_pass_ms"]=Summary(gpuTimes);receipt[@"cpu_encode_ms"]=Summary(cpuTimes);
        receipt[@"completion_wall_ms"]=Summary(wallTimes);
        // Keep a recoverable partial receipt if iOS terminates a long/hot run.
        receipt[@"state"]=@"running";WriteReceipt(receipt);
        if(!safe){receipt[@"state"]=@"failed";return fail(@"Output validation failed. No performance conclusion is valid.");}
    }}
    uint64_t waits=0,timeouts=0;int inFlight=0;
    [session neuralStudioRingStatsWaits:&waits timeouts:&timeouts maxInFlight:&inFlight];
    receipt[@"ring"]=@{@"waits":@(waits),@"timeouts":@(timeouts),@"maximum_in_flight":@(inFlight)};
    receipt[@"encoded_frames"]=@(session.encodedFrameCount);
    receipt[@"thermal_end"]=Thermal();receipt[@"memory_footprint_end_bytes"]=Footprint();
    receipt[@"run_wall_ms"]=@(Elapsed(runStart));
    receipt[@"execution_smoke_validated"]=@(safe&&session.encodedFrameCount==warmup+measured);
    receipt[@"gpu_timing_validated"]=@(timingsValid&&gpuTimes.size()==measured);
    receipt[@"state"]=timingsValid?@"completed":@"completed_timing_unavailable";
    receipt[@"reason"]=timingsValid?@"Finished. These timings cover the SR model test, not a running game.":
        @"Finished, but the driver did not provide valid GPU timings. No latency result is available.";
    // The optional snapshot is independent of execution/timing validity and all latency samples.
    if(saveLastOutput)receipt[@"side_output"]=SaveCompletedOutput(output,outputBytes,frames.lastObject);
    return receipt;
}
} // namespace

@interface NetalProbeViewController : UIViewController {
    std::shared_ptr<std::atomic<int>> _stopFlag;
}
@property(nonatomic,strong) UILabel* status;
@property(nonatomic,strong) UIButton* runButton;
@property(nonatomic) BOOL running;
@end
@implementation NetalProbeViewController
- (void)viewDidLoad {
    [super viewDidLoad];self.view.backgroundColor=UIColor.systemBackgroundColor;
    UILabel* title=[UILabel new];title.text=@"Netal performance test";title.font=[UIFont preferredFontForTextStyle:UIFontTextStyleTitle1];title.numberOfLines=0;
    self.status=[UILabel new];self.status.numberOfLines=0;self.status.text=@"Tests your local upscaling model. Add the supported runtime to Documents/runtime, then tap Run. This can take a few minutes.";
    self.runButton=[UIButton buttonWithType:UIButtonTypeSystem];[self.runButton setTitle:@"Run" forState:UIControlStateNormal];
    [self.runButton addTarget:self action:@selector(runTest) forControlEvents:UIControlEventTouchUpInside];
    [NSNotificationCenter.defaultCenter addObserver:self selector:@selector(stopForInactiveApp:) name:UIApplicationWillResignActiveNotification object:nil];
    [NSNotificationCenter.defaultCenter addObserver:self selector:@selector(stopForInactiveApp:) name:UIApplicationDidEnterBackgroundNotification object:nil];
    UIStackView* stack=[[UIStackView alloc]initWithArrangedSubviews:@[title,self.status,self.runButton]];
    stack.axis=UILayoutConstraintAxisVertical;stack.spacing=24;stack.translatesAutoresizingMaskIntoConstraints=NO;[self.view addSubview:stack];
    [NSLayoutConstraint activateConstraints:@[[stack.leadingAnchor constraintEqualToAnchor:self.view.safeAreaLayoutGuide.leadingAnchor constant:24],
        [stack.trailingAnchor constraintEqualToAnchor:self.view.safeAreaLayoutGuide.trailingAnchor constant:-24],
        [stack.topAnchor constraintEqualToAnchor:self.view.safeAreaLayoutGuide.topAnchor constant:32]]];
}
- (void)stopForInactiveApp:(NSNotification*)notification {
    (void)notification;if(self.running&&_stopFlag){int expected=0;_stopFlag->compare_exchange_strong(expected,2);}
}
- (void)dealloc {[NSNotificationCenter.defaultCenter removeObserver:self];}
- (void)runTest {
    if(self.running){if(_stopFlag)_stopFlag->store(1);self.status.text=@"Finishing the current frame…";self.runButton.enabled=NO;return;}
    self.running=YES;self.runButton.enabled=YES;[self.runButton setTitle:@"Cancel" forState:UIControlStateNormal];
    UIApplication.sharedApplication.idleTimerDisabled=YES;
    _stopFlag=std::make_shared<std::atomic<int>>(0);const auto stopFlag=_stopFlag;
    const UIBackgroundTaskIdentifier backgroundTask=[UIApplication.sharedApplication beginBackgroundTaskWithName:@"Finish SR frame" expirationHandler:^{int expected=0;stopFlag->compare_exchange_strong(expected,2);}];
    self.status.text=@"Starting…";
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,0),^{@autoreleasepool{
        NSMutableDictionary* receipt=RunProbe(^(NSString* message){dispatch_async(dispatch_get_main_queue(),^{if(stopFlag->load()==0)self.status.text=message;});},stopFlag);
        const BOOL written=WriteReceipt(receipt);
        dispatch_async(dispatch_get_main_queue(),^{
            self.running=NO;self.runButton.enabled=![receipt[@"restart_required"] boolValue];
            [self.runButton setTitle:@"Run" forState:UIControlStateNormal];
            if(backgroundTask!=UIBackgroundTaskInvalid)[UIApplication.sharedApplication endBackgroundTask:backgroundTask];
            UIApplication.sharedApplication.idleTimerDisabled=NO;
            NSString* detail=receipt[@"reason"]?:@"Test stopped.";
            NSDictionary* gpu=receipt[@"gpu_full_pass_ms"];
            if([receipt[@"gpu_timing_validated"] boolValue])detail=[detail stringByAppendingFormat:@"\n\nGPU: median %.2f ms; 95th percentile %.2f ms.",[gpu[@"p50"] doubleValue],[gpu[@"p95"] doubleValue]];
            if([receipt[@"side_output"][@"state"] isEqual:@"failed"])detail=[detail stringByAppendingString:@"\n\nThe optional output snapshot could not be saved."];
            self.status.text=[detail stringByAppendingString:written?@"\n\nResults saved to probe-receipt.json in Documents.":@"\n\nThe results file could not be saved."];
        });
    }});
}
@end

@interface NetalProbeSceneDelegate : UIResponder <UIWindowSceneDelegate> {
    BOOL _didAutoRun;
}
@property(nonatomic,strong) UIWindow* window;
@end
@implementation NetalProbeSceneDelegate
- (void)scene:(UIScene*)scene willConnectToSession:(UISceneSession*)session options:(UISceneConnectionOptions*)options {
    (void)session;(void)options;
    if(![scene isKindOfClass:UIWindowScene.class])return;
    self.window=[[UIWindow alloc]initWithWindowScene:(UIWindowScene*)scene];
    self.window.rootViewController=[NetalProbeViewController new];[self.window makeKeyAndVisible];
}
- (void)sceneDidBecomeActive:(UIScene*)scene {
    (void)scene;
    if(!_didAutoRun&&std::getenv("NETAL_IOS_PROBE_AUTORUN")){
        _didAutoRun=YES;
        [(NetalProbeViewController*)self.window.rootViewController runTest];
    }
}
@end

@interface NetalProbeAppDelegate : UIResponder <UIApplicationDelegate>
@end
@implementation NetalProbeAppDelegate
- (BOOL)application:(UIApplication*)application didFinishLaunchingWithOptions:(NSDictionary*)options {
    (void)application;(void)options;return YES;
}
- (UISceneConfiguration*)application:(UIApplication*)application configurationForConnectingSceneSession:(UISceneSession*)session options:(UISceneConnectionOptions*)options {
    (void)application;(void)options;
    UISceneConfiguration* configuration=[[UISceneConfiguration alloc]initWithName:@"Default Configuration" sessionRole:session.role];
    configuration.delegateClass=NetalProbeSceneDelegate.class;
    return configuration;
}
@end
int main(int argc,char* argv[]){@autoreleasepool{return UIApplicationMain(argc,argv,nil,NSStringFromClass(NetalProbeAppDelegate.class));}}
