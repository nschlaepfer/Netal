// Source-only, real-model SR execution probe. No model, fixture tensors or game assets are embedded.
#import <UIKit/UIKit.h>
#import <Metal/Metal.h>
#import <NetalIOSSR/NetalIOSSR.h>
#if defined(NETAL_IOS_FP8_LAB) && NETAL_IOS_FP8_LAB
#import "NetalFP8MatmulLab.h"
#endif
#if defined(NETAL_IOS_METALFX_LAB) && NETAL_IOS_METALFX_LAB
#import "NetalMetalFXLab.h"
#endif
#import <CommonCrypto/CommonDigest.h>
#import <mach/mach.h>
#include "NetalProbeMetrics.h"
#include "NetalProbeFP8Audit.h"
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>

// Method declarations keep old source-only archives usable for comparison. The probe checks
// respondsToSelector before calling; current Core implements both read-only selection properties.
@interface NeuralStudioDLSSSRSession (ProbeLibrarySelection)
- (NSString*)depthInputLibraryName;
- (NSString*)depthOutputLibraryName;
@end

namespace {
using Clock = std::chrono::steady_clock;
constexpr NSUInteger kGuard = 256;
constexpr std::uint16_t kOutputPoison = 0x7e00; // quiet NaN: untouched output cannot pass finite checks.
std::atomic<NSUInteger> gFirstEncodedOutputRepeat{0};
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
        const auto rgba=netal_probe::PlaneColor(x,y,jitterX,jitterY,
            {static_cast<std::uint32_t>(width),static_cast<std::uint32_t>(height)},frame);
        for(NSUInteger channel=0;channel<4;++channel)c[index*4+channel]=static_cast<_Float16>(rgba[channel]);
        m[index*2]=static_cast<_Float16>(-netal_probe::kPlaneVelocityU);
        m[index*2+1]=static_cast<_Float16>(0.0f);d[index]=0.75f;
    }
    *static_cast<float*>(exposure.contents)=1.0f;
}

NSMutableDictionary* RunOperatorLab(NSDictionary* base,NSDictionary* config,id<MTLDevice> device,
    void (^progress)(NSString*),std::shared_ptr<std::atomic<int>> stopFlag) {
    NSMutableDictionary* receipt=[@{@"schema":@"netal.ios.operator-lab.device-probe.v1",@"state":@"unavailable",
        @"probe_mode":@"fp8_lab",@"started_utc":base[@"started_utc"],@"os":base[@"os"],
        @"hardware":base[@"hardware"],@"probe_source_sha256":base[@"probe_source_sha256"],
        @"cleared_sr_environment_keys":base[@"cleared_sr_environment_keys"],
        @"thermal_start":base[@"thermal_start"],@"memory_footprint_start_bytes":base[@"memory_footprint_start_bytes"],
        @"scope":@"optional model-free synthetic matrix operator experiment",
        @"excludes":@[@"DLSS model execution",@"upscaling",@"game rendering",@"presentation"],
        @"operator_execution_validated":@NO,@"hardware_native_fp8_instruction_verified":@NO,
        @"full_sr_validated":@NO,@"gameplay_validated":@NO,@"latency_target_validated":@NO,
        @"restart_required":@NO} mutableCopy];
    NSUInteger samples=0,repetitions=0;
    if(!BoundedInteger(config[@"fp8_sample_count"],3,3,15,&samples) ||
       !BoundedInteger(config[@"fp8_repetitions_per_sample"],8,1,128,&repetitions)){
        receipt[@"reason"]=@"Use 3–15 samples and 1–128 repetitions for the optional operator test.";return receipt;
    }
    receipt[@"configuration"]=@{@"sample_count":@(samples),@"repetitions_per_sample":@(repetitions)};
#if defined(NETAL_IOS_FP8_LAB) && NETAL_IOS_FP8_LAB
    NSDictionary* identity=NSBundle.mainBundle.infoDictionary[@"NetalFP8LabIdentity"];
    NSURL* library=[NSBundle.mainBundle URLForResource:@"netal_fp8_lab" withExtension:@"metallib"];
    NSData* libraryData=library?[NSData dataWithContentsOfURL:library]:nil;
    if(![identity isKindOfClass:NSDictionary.class] || !libraryData){
        receipt[@"reason"]=@"The optional operator library or its build identity is missing.";return receipt;
    }
    NSString* actualHash=SHA(libraryData.bytes,libraryData.length);
    receipt[@"lab_build_identity"]=identity;receipt[@"actual_metallib_sha256"]=actualHash;
    if(![actualHash isEqual:identity[@"metallib_sha256"]]){
        receipt[@"reason"]=@"The optional operator library does not match this app's build identity.";return receipt;
    }
    if(stopFlag->load()!=0 || NSProcessInfo.processInfo.thermalState==NSProcessInfoThermalStateCritical){
        receipt[@"state"]=@"incomplete";receipt[@"reason"]=@"Stopped before the next graphics test.";return receipt;
    }
    if(@available(iOS 27.0,*)){
        progress(@"Running the optional graphics test…");
        NSError* error=nil;const auto start=Clock::now();
        NSDictionary* result=NetalRunFP8MatmulLabWithStop(device,library,samples,repetitions,^BOOL{
            return stopFlag->load()!=0;
        },&error);
        receipt[@"lab_completion_wall_ms"]=@(Elapsed(start));receipt[@"thermal_end"]=Thermal();
        receipt[@"memory_footprint_end_bytes"]=Footprint();
        if(!result){
            NSString* reason=error.localizedDescription?:@"The optional test did not produce a validated result.";
            const BOOL timeout=[reason.lowercaseString containsString:@"timeout"];
            receipt[@"state"]=timeout?@"failed":(stopFlag->load()!=0?@"incomplete":@"unavailable");
            receipt[@"restart_required"]=@(timeout);receipt[@"reason"]=reason;return receipt;
        }
        receipt[@"operator_lab"]=result;
        const BOOL passed=[result[@"state"] isEqual:@"synthetic_operator_pass"];
        receipt[@"operator_execution_validated"]=@(passed);
        receipt[@"state"]=passed?(stopFlag->load()!=0?@"incomplete":@"completed_operator_lab"):@"unavailable";
        receipt[@"reason"]=passed?@"Graphics operator test finished. These results cover synthetic matrix operations, not a running game.":
            @"The optional test did not produce a validated result.";
        return receipt;
    }
    receipt[@"reason"]=@"The optional operator test requires iOS 27 or newer.";
#else
    (void)device;(void)progress;(void)stopFlag;
    receipt[@"reason"]=@"The optional operator test is not included in this app build.";
#endif
    return receipt;
}

NSMutableDictionary* RunMetalFXComparison(NSDictionary* base,NSDictionary* config,NSString* mode,id<MTLDevice> device,
    void (^progress)(NSString*),std::shared_ptr<std::atomic<int>> stopFlag) {
    NSMutableDictionary* receipt=[@{@"schema":@"netal.ios.metalfx-request.v1",@"state":@"unavailable",
        @"probe_mode":mode,@"scope":@"requested optional model-free system-scaler comparison",
        @"execution_smoke_validated":@NO,@"gpu_timing_validated":@NO,@"gameplay_validated":@NO,
        @"netal_model_allocated":@NO,@"apple_internal_architecture_known":@NO} mutableCopy];
    NSString* selected=[mode isEqual:@"metalfx_spatial"]?@"spatial":@"temporal";
    if(config[@"metalfx_mode"]&&![config[@"metalfx_mode"] isEqual:selected]){
        receipt[@"reason"]=@"The selected system-scaler mode conflicts with metalfx_mode.";return receipt;
    }
#if defined(NETAL_IOS_METALFX_LAB) && NETAL_IOS_METALFX_LAB
    NSDictionary* identity=NSBundle.mainBundle.infoDictionary[@"NetalMetalFXLabIdentity"];
    if(![identity isKindOfClass:NSDictionary.class]){
        receipt[@"reason"]=@"The optional system-scaler comparison build identity is missing.";return receipt;
    }
    NSMutableDictionary* options=[config mutableCopy];options[@"metalfx_mode"]=selected;
    progress(@"Running the system graphics comparison…");NSError* error=nil;
    NSDictionary* result=NetalRunMetalFXLabWithStop(device,options,^BOOL{return stopFlag->load()!=0;},&error);
    if(result)receipt=[result mutableCopy];
    else receipt[@"reason"]=error.localizedDescription?:@"The optional comparison did not return a result.";
    receipt[@"metalfx_lab_build_identity"]=identity;
#else
    (void)device;(void)progress;(void)stopFlag;
    receipt[@"reason"]=@"The optional system-scaler comparison is not included in this app build.";
#endif
    receipt[@"probe_mode"]=mode;receipt[@"probe_source_sha256"]=base[@"probe_source_sha256"];
    receipt[@"started_utc"]=base[@"started_utc"];receipt[@"device_hardware"]=base[@"hardware"];
    receipt[@"cleared_sr_environment_keys"]=base[@"cleared_sr_environment_keys"];
    return receipt;
}

NSMutableDictionary* RunProbe(void (^progress)(NSString*),std::shared_ptr<std::atomic<int>> stopFlag) {
    NSMutableDictionary* receipt=[@{
        @"schema":@"netal.ios.sr.device-probe.v1", @"state":@"unavailable",
        @"started_utc":[[NSISO8601DateFormatter new] stringFromDate:NSDate.date],
        @"scope":@"SR temporal buffer-to-buffer; boundary transforms, model, history and output included",
        @"excludes":@[@"game rendering",@"texture import/export",@"presentation",@"CPU procedural input generation"],
        @"stimulus":@"procedural translating HDR plane; current-to-previous motion; inverted depth; exposure=1; 18-phase jitter",
        @"stimulus_contract":@{@"version":@2,@"cycles_u":@(netal_probe::kPlaneCyclesU),
            @"cycles_v":@(netal_probe::kPlaneCyclesV),@"velocity_u_per_frame":@(netal_probe::kPlaneVelocityU),
            @"velocity_v_per_frame":@0,@"pixel_centers":@YES,@"jitter_units":@"input pixels",
            @"jitter_world_normalization":@"jitter_x/input_width, jitter_y/input_height",
            @"motion_units":@"normalized current-to-previous UV, without jitter",
            @"jitter_sequence":@"existing accepted 18-phase sequence",@"jitter_phase_count":@18,
            @"nvidia_preset_validated":@NO,@"nvidia_optimal_settings_api_implemented":@NO},
        @"probe_source_sha256":NSBundle.mainBundle.infoDictionary[@"NetalProbeSourceSHA256"]?:NSNull.null,
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
    NSURL* configURL=[Documents() URLByAppendingPathComponent:@"benchmark.json"];
    NSDictionary* config=JSON(configURL);
    if([NSFileManager.defaultManager fileExistsAtPath:configURL.path] && !config)
        return fail(@"benchmark.json is not a JSON object.");
    config=config?:@{};
    id probeMode=config[@"probe_mode"]?:@"sr";
    if(![@[@"sr",@"fp8_lab",@"metalfx_spatial",@"metalfx_temporal"] containsObject:probeMode])return fail(@"Unknown test mode in benchmark.json.");
    receipt[@"probe_mode"]=probeMode;
    const BOOL metalfxMode=[probeMode isEqual:@"metalfx_spatial"]||[probeMode isEqual:@"metalfx_temporal"];
    const BOOL captureBuild=[NSBundle.mainBundle.infoDictionary[@"MetalCaptureEnabled"] boolValue];
    receipt[@"metal_capture_enabled_build"]=@(captureBuild);
    if(captureBuild&&!metalfxMode)return fail(@"This dedicated capture build cannot run ordinary performance tests. Select a system-scaler capture or use a performance build.");
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();
    NetalIOSSRCapabilities* capability=[NetalIOSSRCapabilities capabilitiesForDevice:device];
    receipt[@"hardware"]=@{@"gpu":device.name?:@"unavailable",@"chip":capability.chipName,
        @"eligible":@(capability.eligible),@"reason":capability.reason,
        @"neural_accelerator_candidate":@(capability.supportsNeuralAccelerators),
        @"fp8_tensor_allocation_supported":@(capability.supportsFP8TensorTypes)};
    // Select the separately built, model-free experiment before looking for any SR model files.
    if([probeMode isEqual:@"fp8_lab"])return RunOperatorLab(receipt,config,device,progress,stopFlag);
    if(metalfxMode)return RunMetalFXComparison(receipt,config,probeMode,device,progress,stopFlag);
    if(!capability.eligible)return fail(capability.reason);
    NSURL* root=[Documents() URLByAppendingPathComponent:@"runtime" isDirectory:YES];
    NSDictionary* manifest=JSON([root URLByAppendingPathComponent:@"manifest.json"]);
    if(!manifest)return fail(@"Add your supported runtime bundle to Documents/runtime, then run again.");
    if(![manifest[@"schema"] isEqual:@"netal.sr.runtime.v2"] || ![manifest[@"model"] isEqual:@"DLSS_SR_Feature1_Quality_K"])
        return fail(@"This probe requires the supported DLSS 4 K runtime. DLSS 4.5 is unavailable.");
    if(![manifest[@"files"] isKindOfClass:NSArray.class])return fail(@"The runtime asset inventory is missing.");
    NSMutableDictionary* ledger=[NSMutableDictionary dictionary];
    for(id raw in manifest[@"files"])if([raw isKindOfClass:NSDictionary.class]&&[raw[@"path"] isKindOfClass:NSString.class])ledger[raw[@"path"]]=raw;
    NSMutableArray* modelAssets=[NSMutableArray array];
    for(id item in manifest[@"files"])if([item isKindOfClass:NSDictionary.class] &&
        [item[@"path"] isKindOfClass:NSString.class] && [item[@"path"] hasPrefix:@"model/"])
        [modelAssets addObject:item];
    receipt[@"model"]=@{@"name":manifest[@"model"],@"asset_root_sha256":manifest[@"asset_root_sha256"]?:NSNull.null,
        @"assets":modelAssets,@"identity_validated":@NO};
    bool saveLastOutput=false;
    if(!netal_probe::ReadOptionalBoolean(config[@"save_last_output"],false,&saveLastOutput))
        return fail(@"save_last_output must be a JSON boolean (true or false).");
    bool fastInputBoundary=false,fastOutputBoundary=false;NSUInteger outputRepeat=1;
    if(!netal_probe::ReadOptionalBoolean(config[@"fast_input_boundary"],false,&fastInputBoundary))
        return fail(@"fast_input_boundary must be a JSON boolean (true or false).");
    if(!netal_probe::ReadOptionalBoolean(config[@"fast_output_boundary"],false,&fastOutputBoundary))
        return fail(@"fast_output_boundary must be a JSON boolean (true or false).");
    if(!BoundedInteger(config[@"diagnostic_output_repeat"],1,1,8,&outputRepeat))
        return fail(@"diagnostic_output_repeat must be an integer from 1 to 8, not a boolean.");
    const NSUInteger previousRepeat=gFirstEncodedOutputRepeat.load();
    if(previousRepeat&&previousRepeat!=outputRepeat)
        return fail(@"Close and reopen the app before changing diagnostic_output_repeat; Core caches it for the process.",YES);
    if(fastInputBoundary&&(!netal_probe::FP8Hex(ledger[@"metallib/hiluma_input_depthinv.metallib"][@"sha256"])||
       ![NSFileManager.defaultManager fileExistsAtPath:[root URLByAppendingPathComponent:@"metallib/hiluma_input_depthinv.metallib"].path]))
        return fail(@"The fast input-boundary candidate needs its manifest-pinned library.");
    if(fastOutputBoundary&&(!netal_probe::FP8Hex(ledger[@"metallib/hiluma_output_depthinv_fast.metallib"][@"sha256"])||
       ![NSFileManager.defaultManager fileExistsAtPath:[root URLByAppendingPathComponent:@"metallib/hiluma_output_depthinv_fast.metallib"].path]))
        return fail(@"The fast output-boundary candidate needs its manifest-pinned library.");
    receipt[@"side_output"]=saveLastOutput?
        @{@"requested":@YES,@"state":@"unavailable",@"reason":@"No completed output was saved."}:
        @{@"requested":@NO,@"state":@"disabled"};
    NSString* profile=config[@"profile"]?:@"1280x720";
    id renderValue=config[@"render_size"];
    netal_probe::Dimensions requestedOutput,requestedRender;
    auto parseSize=[](id value,netal_probe::Dimensions* dimensions){
        if(![value isKindOfClass:NSString.class])return false;
        NSString* text=value;
        return netal_probe::ParseSize(std::string_view(text.UTF8String,[text lengthOfBytesUsingEncoding:NSUTF8StringEncoding]),dimensions);
    };
    if(!parseSize(profile,&requestedOutput) || requestedOutput.width<10 || requestedOutput.height<10 ||
       requestedOutput.width>9999 || requestedOutput.height>9999)
        return fail(@"Use a supported output profile such as 1280x720.");
    if(renderValue && (!parseSize(renderValue,&requestedRender) || !netal_probe::RenderFitsOutput(requestedRender,requestedOutput)))
        return fail(@"render_size must be positive WxH, no larger than the output, with an aspect ratio within 1% of the output.");
    NSString* backend=config[@"backend"]?:@"batched";
    if([backend isEqual:@"mpp"])backend=@"mpp-candidate";
    NSString* api=config[@"metal_api"]?:@"metal4";
    if(![@[@"reference",@"batched",@"mpp-candidate",@"fp8-candidate"] containsObject:backend] ||
       ![@[@"metal3",@"metal4"] containsObject:api])return fail(@"Unknown backend or Metal API in benchmark.json.");
    const BOOL fp8Candidate=[backend isEqual:@"fp8-candidate"];
    if(fp8Candidate){
        if(@available(iOS 27.0,*)){
            if(![capability.chipName isEqual:@"A20 Pro"]||!capability.supportsFP8TensorTypes)
                return fail(@"This mixed-precision candidate requires A20 Pro and working FP8 tensor allocation.");
        }else return fail(@"This mixed-precision candidate requires iOS 27 or newer.");
    }
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
    if(renderValue)setenv("NEURALSTUDIO_SR_RENDER",[(NSString*)renderValue UTF8String],1);
    if(fp8Candidate)setenv("NEURALSTUDIO_SR_METAL3_SHARED_ENCODER","1",1);
    if(fastInputBoundary)setenv("NEURALSTUDIO_SR_DEPTH_INPUT_LIBRARY","hiluma_input_depthinv",1);
    if(fastOutputBoundary)setenv("NEURALSTUDIO_SR_DEPTH_OUTPUT_LIBRARY","hiluma_output_depthinv_fast",1);
    setenv("NEURALSTUDIO_SR_OUTPUT_REPEAT",[@(outputRepeat).stringValue UTF8String],1);
    receipt[@"configuration"]=@{@"profile":profile,@"requested_backend":backend,@"metal_api":api,
        @"warmup_frames":@(warmup),@"measured_frames":@(measured),@"serial_single_in_flight":@YES,
        @"save_last_output":@(saveLastOutput),
        @"render_size":renderValue?:NSNull.null,
        @"ordered_dispatches":@YES,@"metal3_shared_encoder_explicitly_forced":@(fp8Candidate),
        @"fast_input_boundary":@(fastInputBoundary),@"diagnostic_output_repeat":@(outputRepeat),
        @"fast_output_boundary":@(fastOutputBoundary),
        @"per_frame_timeout_seconds":@30,@"run_deadline_seconds":@180};
    progress(@"Preparing the model…");
    NSError* error=nil;
    const auto prepareStart=Clock::now();
    NeuralStudioDLSSSRSession* session=[[NeuralStudioDLSSSRSession alloc] initWithDevice:device runtimeBundleURL:root error:&error];
    receipt[@"cold_prepare_wall_ms"]=@(Elapsed(prepareStart));
    if(!session)return fail([@"The runtime could not be prepared: " stringByAppendingString:error.localizedDescription?:@"unknown error"]);
    const BOOL inputSelectionKnown=[session respondsToSelector:@selector(depthInputLibraryName)];
    const BOOL outputSelectionKnown=[session respondsToSelector:@selector(depthOutputLibraryName)];
    NSString* actualInput=inputSelectionKnown?session.depthInputLibraryName:nil;
    NSString* actualOutput=outputSelectionKnown?session.depthOutputLibraryName:nil;
    auto libraryReceipt=[&](NSString* actual,NSString* requested){
        NSString* path=actual.length?[NSString stringWithFormat:@"metallib/%@.metallib",actual]:nil;
        id hash=path?ledger[path][@"sha256"]:nil;
        return @{@"requested":requested,@"actual":actual?:static_cast<id>(NSNull.null),
            @"sha256":hash?:NSNull.null,@"selection_verified":@(actual.length>0&&netal_probe::FP8Hex(hash))};
    };
    receipt[@"boundary_libraries"]=@{@"input":libraryReceipt(actualInput,fastInputBoundary?@"hiluma_input_depthinv":@"bundle_default"),
        @"output":libraryReceipt(actualOutput,fastOutputBoundary?@"hiluma_output_depthinv_fast":@"bundle_default")};
    if((actualInput.length&&!netal_probe::FP8Hex(ledger[[NSString stringWithFormat:@"metallib/%@.metallib",actualInput]][@"sha256"]))||
       (actualOutput.length&&!netal_probe::FP8Hex(ledger[[NSString stringWithFormat:@"metallib/%@.metallib",actualOutput]][@"sha256"])))
        return fail(@"Core selected a boundary library missing from the pinned runtime inventory.");
    if(fastInputBoundary&&inputSelectionKnown&&![actualInput isEqual:@"hiluma_input_depthinv"])
        return fail(@"Core did not select the requested fast input-boundary candidate.");
    if(fastOutputBoundary&&(!outputSelectionKnown||![actualOutput isEqual:@"hiluma_output_depthinv_fast"]))
        return fail(@"Core did not verify selection of the requested fast output-boundary candidate.");
    receipt[@"profiling_only"]=@(outputRepeat>1);receipt[@"normal_sr_latency_sample"]=@(outputRepeat==1);
    if(outputRepeat>1)receipt[@"scope"]=@"diagnostic SR buffer-to-buffer run with repeated output passes; not normal SR latency";
    receipt[@"model"]=@{@"name":manifest[@"model"],@"asset_root_sha256":session.runtimeBundleRoot,
        @"assets":modelAssets,@"identity_validated":@YES};
    receipt[@"geometry"]=@{@"input_width":@(session.inputWidth),@"input_height":@(session.inputHeight),
        @"output_width":@(session.outputWidth),@"output_height":@(session.outputHeight),
        @"input_to_output_x":@(static_cast<double>(session.inputWidth)/session.outputWidth),
        @"input_to_output_y":@(static_cast<double>(session.inputHeight)/session.outputHeight),
        @"output_to_input_x":@(static_cast<double>(session.outputWidth)/session.inputWidth),
        @"output_to_input_y":@(static_cast<double>(session.outputHeight)/session.inputHeight),
        @"input_to_output_pixel_fraction":@(static_cast<double>(session.inputWidth)*session.inputHeight/(static_cast<double>(session.outputWidth)*session.outputHeight))};
    receipt[@"requested_geometry"]=@{@"output_width":@(requestedOutput.width),@"output_height":@(requestedOutput.height),
        @"input_width":renderValue?static_cast<id>(@(requestedRender.width)):NSNull.null,
        @"input_height":renderValue?static_cast<id>(@(requestedRender.height)):NSNull.null};
    if(session.outputWidth!=requestedOutput.width || session.outputHeight!=requestedOutput.height ||
       (renderValue&&(session.inputWidth!=requestedRender.width || session.inputHeight!=requestedRender.height)))
        return fail(@"The runtime geometry does not match the requested input/output sizes. No frames were submitted.");
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
    NSMutableDictionary* fp8Tensors=[NSMutableDictionary dictionary];
    NSMutableDictionary* fp8ExpectedPaths=[NSMutableDictionary dictionary];
    NSMutableDictionary* fp8FrameCounts=[NSMutableDictionary dictionary];
    NSMutableDictionary* frameDispatchCounts=[NSMutableDictionary dictionary];
    NSMutableSet* fp8AuditedPaths=[NSMutableSet set];
    NSDictionary* fp8Pack=nil;NSString* fp8Root=nil;
    NSDictionary* geometryManifest=JSON([root URLByAppendingPathComponent:[profilePrefix stringByAppendingString:@"manifest.json"]]);
    NSDictionary* profileArenas=geometryManifest[@"arenas"];
    NSDictionary* fp8Arenas=@{@"arena6":profileArenas[@"6"]?:@0,@"arena7":profileArenas[@"7"]?:@0,@"arena8":profileArenas[@"8"]?:@0};
    for(NSString* stage in @[@"enc0",@"enc1",@"enc2",@"enc3",@"enc4",@"dec5",@"dec4",@"dec3",@"dec2",@"dec1",@"dec0"]){
        frameDispatchCounts[stage]=@1; // The transpiled per-window layer is one dispatch when no batched sidecar exists.
        fp8ExpectedPaths[[pipelinePrefix stringByAppendingFormat:@"dltss_pwin_%@_layer.json",stage]]=stage;
        if([@[@"enc3",@"enc4",@"dec5",@"dec4",@"dec3"] containsObject:stage])for(NSUInteger phase=1;phase<8;++phase)
            fp8ExpectedPaths[[pipelinePrefix stringByAppendingFormat:@"dltss_pwin_%@_layer.phase%lu.json",stage,static_cast<unsigned long>(phase)]]=stage;
    }
    if(fp8Candidate){
        fp8Pack=JSON([root URLByAppendingPathComponent:@"pipeline/fp8/pack.json"]);fp8Root=fp8Pack[@"pack_root_sha256"];
        if(!ledger[@"pipeline/fp8/pack.json"]||![fp8Pack[@"schema"] isEqual:@"netal.sr.quantized-pack.v1"]||
           ![fp8Pack[@"source_model"] isEqual:@"DLSS_SR_Feature1_Quality_K"]||!netal_probe::FP8Hex(fp8Root)||
           !netal_probe::FP8Bool(fp8Pack[@"derived_approximate"],true)||!netal_probe::FP8Bool(fp8Pack[@"dlss45"],false)||
           ![fp8Pack[@"files"] isKindOfClass:NSArray.class]||![fp8Pack[@"tensors"] isKindOfClass:NSArray.class])
            return fail(@"The mixed-precision candidate needs its manifest-pinned, supported pack.json.");
        for(id raw in fp8Pack[@"files"]){
            if(![raw isKindOfClass:NSDictionary.class])return fail(@"Invalid FP8 pack inventory.");
            NSDictionary* row=raw;NSDictionary* pinned=ledger[row[@"path"]];
            if(!pinned||![pinned[@"bytes"] isEqual:row[@"bytes"]]||![pinned[@"sha256"] isEqual:row[@"sha256"]])
                return fail(@"FP8 pack payload inventory disagrees with the validated runtime.");
        }
        for(id raw in fp8Pack[@"tensors"]){
            if(![raw isKindOfClass:NSDictionary.class])return fail(@"Invalid FP8 tensor metadata.");
            NSDictionary* tensor=raw;NSString* identity=tensor[@"id"];
            if(![identity isKindOfClass:NSString.class]||fp8Tensors[identity])return fail(@"Duplicate FP8 tensor identity.");
            fp8Tensors[identity]=tensor;
        }
    }
    BOOL allMPPDescriptorsValid=YES;NSUInteger mppGEMMs=0;
    NSMutableSet* auditedMPPPaths=[NSMutableSet set];
    for(NSDictionary* item in manifest[@"files"]) {
        NSString* path=item[@"path"];
        if(![path hasPrefix:pipelinePrefix] || ![path hasSuffix:@".json"])continue;
        NSDictionary* sidecar=JSON([root URLByAppendingPathComponent:path]);
        NSString* graphLayer=fp8ExpectedPaths[path];
        if(![backend isEqual:@"reference"]&&graphLayer&&[sidecar[@"dispatches"] isKindOfClass:NSArray.class]&&
           [sidecar[@"dispatches"] count]>[frameDispatchCounts[graphLayer] unsignedIntegerValue])
            frameDispatchCounts[graphLayer]=@([sidecar[@"dispatches"] count]);
        if(fp8Candidate){
            NSString* layer=fp8ExpectedPaths[path];
            if(!layer)return fail(@"Unexpected sidecar in the selected mixed-precision profile.");
            NSDictionary* checked=netal_probe::AuditFP8Descriptor(sidecar,layer,ledger,fp8Tensors,fp8Arenas,fp8Root);
            if(![checked[@"validated"] boolValue])return fail([@"Mixed-precision sidecar audit failed: " stringByAppendingString:checked[@"reason"]]);
            [fp8AuditedPaths addObject:path];
            if([sidecar[@"dispatches"] count]>[fp8FrameCounts[layer] unsignedIntegerValue])fp8FrameCounts[layer]=@([sidecar[@"dispatches"] count]);
        }
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
    NSUInteger baselineDispatches=3;for(NSNumber* count in frameDispatchCounts.allValues)baselineDispatches+=count.unsignedIntegerValue;
    if(!netal_probe::DispatchRingFits(baselineDispatches,outputRepeat))
        return fail(@"The requested output-repeat diagnostic exceeds the 160-entry frame dispatch ring.");
    receipt[@"dispatch_ring_bound"]=@{@"baseline_maximum_dispatches":@(baselineDispatches),@"output_repeat":@(outputRepeat),
        @"maximum_dispatches":@(baselineDispatches+outputRepeat-1),@"capacity":@160,@"verified_from_selected_descriptors":@YES};
    if(fp8Candidate){
        NSUInteger pwinDispatches=0;for(NSNumber* count in fp8FrameCounts.allValues)pwinDispatches+=count.unsignedIntegerValue;
        if(fp8AuditedPaths.count!=46||session.pipelineLayersPresent!=11||session.pipelineLayersTotal!=11||
           pwinDispatches!=134||pwinDispatches+3>=160)return fail(@"Incomplete mixed-precision layer/phase coverage or dispatch ring overflow.");
        receipt[@"backend_mode"]=@"mixed_fp8_e4m3_with_sensitive_fp16_and_half_mlps";
        receipt[@"fp8_candidate"]=@{@"selected_descriptors_validated":@YES,@"library":@"pwin_mpp_all",
            @"validation_scope":@"all-dispatch structural base bounds and transformed FP8/FP16/MLP operand spans; original operator semantics remain offline-verified",
            @"general_shader_memory_safety_proven":@NO,@"original_operator_semantics_verified_on_device":@NO,
            @"coverage_basis":@"validated selected descriptor graph; hardware instructions are not inspected",
            @"library_sha256":ledger[@"metallib/pwin_mpp_all.metallib"][@"sha256"]?:NSNull.null,
            @"pack_root_sha256":fp8Root,@"pack_manifest_sha256":ledger[@"pipeline/fp8/pack.json"][@"sha256"],
            @"profile_sidecars":@(fp8AuditedPaths.count),@"layers":@11,@"fp8_gemms_per_frame":@40,
            @"gpu_row_quantizers_per_frame":@40,@"sensitive_fp16_gemms_per_frame":@2,@"half_mlps_per_frame":@6,
            @"pwin_dispatches_per_frame":@(pwinDispatches),@"boundary_dispatches_per_frame":@3,
            @"diagnostic_extra_output_dispatches_per_frame":@(outputRepeat-1),
            @"maximum_total_dispatches_per_frame":@(pwinDispatches+3+outputRepeat-1),@"requires_ordered_dispatches":@YES,
            @"original_k16_handoffs_preserved":@NO,@"candidate_approximate":@YES,
            @"pack_derivation_verified_on_device":@NO,@"native_fp8_instruction_verified":@NO,
            @"remaining_arithmetic":@"2 sensitive FP16 GEMMs, 6 half MLPs, and other FP16/FP32 operators"};
    }
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
        NSUInteger unseen=0;gFirstEncodedOutputRepeat.compare_exchange_strong(unseen,outputRepeat);
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
    if(outputRepeat>1)receipt[@"reason"]=timingsValid?
        @"Diagnostic test finished. GPU timings include repeated output passes and are not ordinary SR latency.":
        @"Diagnostic test finished, but valid GPU timestamps are unavailable.";
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
    self.status=[UILabel new];self.status.numberOfLines=0;self.status.text=@"Tests graphics performance on your phone using your selected setup. Tap Run to begin. This can take a few minutes.";
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
    if(self.running){if(_stopFlag)_stopFlag->store(1);self.status.text=@"Finishing the current step…";self.runButton.enabled=NO;return;}
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
