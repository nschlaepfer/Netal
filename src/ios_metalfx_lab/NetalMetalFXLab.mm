// Original model-free MetalFX adapter. API references:
// developer.apple.com/documentation/metalfx and /documentation/xcode/capturing-a-metal-workload-programmatically
#import "NetalMetalFXLab.h"
#import <MetalFX/MetalFX.h>
#import <TargetConditionals.h>
#import <CommonCrypto/CommonDigest.h>
#include "NetalMetalFXLabSupport.h"
#include <array>
#include <chrono>
#include <cstring>

namespace {
using Clock=std::chrono::steady_clock;
using netal_probe::Dimensions;
constexpr std::array<std::array<float,2>,18> kJitter={{{-0.3125f,-0.351851851f},{0.1875f,-0.0185185075f},
 {-0.0625f,0.314814806f},{0.4375f,-0.240740746f},{-0.46875f,0.092592597f},{0.03125f,0.42592591f},
 {-0.21875f,-0.42592591f},{0,-0.166666657f},{-0.25f,0.166666687f},{0.25f,-0.388888896f},
 {-0.375f,-0.0555555522f},{0.125f,0.277777791f},{-0.125f,-0.277777791f},{0.375f,0.055555582f},
 {-0.4375f,0.388888896f},{0.0625f,-0.462962955f},{-0.1875f,-0.129629612f},{0.3125f,0.203703701f}}};
double MS(Clock::time_point start){return std::chrono::duration<double,std::milli>(Clock::now()-start).count();}
id Number(double x){return std::isfinite(x)?static_cast<id>(@(x)):NSNull.null;}
NSDictionary* Summary(const std::vector<double>& x){return @{@"samples":@(x.size()),
 @"p50":Number(netal_probe::Percentile(x,.5)),@"p95":Number(netal_probe::Percentile(x,.95)),
 @"p99":Number(netal_probe::Percentile(x,.99))};}
NSString* Thermal(){switch(NSProcessInfo.processInfo.thermalState){
 case NSProcessInfoThermalStateNominal:return @"nominal";case NSProcessInfoThermalStateFair:return @"fair";
 case NSProcessInfoThermalStateSerious:return @"serious";case NSProcessInfoThermalStateCritical:return @"critical";}
 return @"unknown";}
bool Count(id value,NSUInteger fallback,NSUInteger low,NSUInteger high,NSUInteger* out){
 if(!value){*out=fallback;return true;}
 if(![value isKindOfClass:NSNumber.class]||CFGetTypeID((__bridge CFTypeRef)value)==CFBooleanGetTypeID())return false;
 double x=[value doubleValue];if(!std::isfinite(x)||std::floor(x)!=x||x<low||x>high)return false;*out=NSUInteger(x);return true;}
bool ParseDimensions(id value,NSString* fallback,Dimensions* out){
 value=value?:fallback;if(![value isKindOfClass:NSString.class])return false;
 const char* p=[value UTF8String];return p&&netal_probe::ParseSize(p,out);}
NSString* SHA(const void* bytes,NSUInteger length){
 unsigned char digest[CC_SHA256_DIGEST_LENGTH];CC_SHA256(bytes,CC_LONG(length),digest);
 NSMutableString* result=[NSMutableString string];for(unsigned char x:digest)[result appendFormat:@"%02x",x];return result;}
struct Surface {
 __strong id<MTLTexture> texture=nil;
 __strong id<MTLBuffer> host=nil;
 __strong id<MTLBuffer> verification=nil;
 NSUInteger width=0,height=0,pixelBytes=0,row=0,bytes=0;
};
bool MakeSurface(id<MTLDevice> device,Dimensions size,MTLPixelFormat format,NSUInteger pixelBytes,
                 MTLTextureUsage usage,Surface* out){
 std::size_t row=0,bytes=0;if(!netal_metalfx::RowLayout(size.width,size.height,pixelBytes,&row,&bytes))return false;
 auto desc=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:size.width height:size.height mipmapped:NO];
 desc.storageMode=MTLStorageModePrivate;desc.usage=usage;
 out->texture=[device newTextureWithDescriptor:desc];
 out->host=[device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
 out->verification=[device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
 out->width=size.width;out->height=size.height;out->pixelBytes=pixelBytes;out->row=row;out->bytes=bytes;
 if(!out->texture||!out->host||!out->verification)return false;
 std::memset(out->host.contents,0xa5,bytes);std::memset(out->verification.contents,0xa5,bytes);return true;
}
void Upload(id<MTL4ComputeCommandEncoder> encoder,const Surface& s,id<MTLBuffer> source){
 [encoder copyFromBuffer:source sourceOffset:netal_metalfx::kGuard sourceBytesPerRow:s.row sourceBytesPerImage:0
  sourceSize:MTLSizeMake(s.width,s.height,1) toTexture:s.texture destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0,0,0)];
}
void Download(id<MTL4ComputeCommandEncoder> encoder,const Surface& s,id<MTLBuffer> target){
 [encoder copyFromTexture:s.texture sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0)
  sourceSize:MTLSizeMake(s.width,s.height,1) toBuffer:target destinationOffset:netal_metalfx::kGuard
  destinationBytesPerRow:s.row destinationBytesPerImage:0];
}
void Barrier(id<MTL4ComputeCommandEncoder> encoder){
 [encoder barrierAfterEncoderStages:MTLStageBlit|MTLStageDispatch beforeEncoderStages:MTLStageBlit|MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
}
bool Guards(const Surface& s,id<MTLBuffer> buffer){
 const auto* p=static_cast<const unsigned char*>(buffer.contents);
 for(NSUInteger i=0;i<netal_metalfx::kGuard;++i)if(p[i]!=0xa5||p[s.bytes-1-i]!=0xa5)return false;
 for(NSUInteger y=0;y<s.height;++y)for(NSUInteger i=s.width*s.pixelBytes;i<s.row;++i)
  if(p[netal_metalfx::kGuard+y*s.row+i]!=0xa5)return false;
 return true;
}
bool EqualRows(const Surface& s){
 if(!Guards(s,s.verification))return false;
 for(NSUInteger y=0;y<s.height;++y)
  if(std::memcmp(static_cast<unsigned char*>(s.host.contents)+netal_metalfx::kGuard+y*s.row,
                 static_cast<unsigned char*>(s.verification.contents)+netal_metalfx::kGuard+y*s.row,s.width*s.pixelBytes))return false;
 return true;
}
void Inputs(Surface& color,Surface& motion,Surface& depth,Surface& exposure,NSUInteger frame,float jx,float jy){
 for(NSUInteger y=0;y<color.height;++y){
  auto* c=reinterpret_cast<_Float16*>(static_cast<unsigned char*>(color.host.contents)+netal_metalfx::kGuard+y*color.row);
  auto* m=reinterpret_cast<_Float16*>(static_cast<unsigned char*>(motion.host.contents)+netal_metalfx::kGuard+y*motion.row);
  auto* d=reinterpret_cast<float*>(static_cast<unsigned char*>(depth.host.contents)+netal_metalfx::kGuard+y*depth.row);
  for(NSUInteger x=0;x<color.width;++x){
   const auto rgba=netal_probe::PlaneColor(x,y,jx,jy,{std::uint32_t(color.width),std::uint32_t(color.height)},frame);
   for(NSUInteger channel=0;channel<4;++channel)c[x*4+channel]=_Float16(rgba[channel]);
   m[x*2]=_Float16(-netal_probe::kPlaneVelocityU);m[x*2+1]=_Float16(0);d[x]=.75f;
  }
 }
 // Actual FP32 -> FP16 conversion, not a four-byte bit copy into R16Float.
 *reinterpret_cast<_Float16*>(static_cast<unsigned char*>(exposure.host.contents)+netal_metalfx::kGuard)=_Float16(1.f);
}
NSUInteger InvalidRGB(const Surface& output){
 NSUInteger count=0;for(NSUInteger y=0;y<output.height;++y){
  const auto* p=reinterpret_cast<const _Float16*>(static_cast<const unsigned char*>(output.host.contents)+netal_metalfx::kGuard+y*output.row);
  for(NSUInteger x=0;x<output.width;++x)for(NSUInteger c=0;c<3;++c)if(!std::isfinite(float(p[x*4+c])))++count;
 }return count;
}
} // namespace

NSDictionary<NSString*,id>* NetalRunMetalFXLabWithStop(id<MTLDevice> device,NSDictionary<NSString*,id>* config,
 BOOL (^shouldStop)(void),NSError** error){
 NSMutableDictionary* r=[@{@"schema":@"netal.metalfx.device-lab.v1",@"state":@"unavailable",
  @"reason":@"MetalFX comparison is unavailable.",@"scope":@"Metal 4 buffer-to-texture input transfers, synchronization, MetalFX scaling, texture-to-buffer output transfer",
  @"excludes":@[@"game rendering",@"CPU input generation/padded-row packing",@"output validation",@"presentation"],
  @"serial_latency_only":@YES,
  @"gameplay_validated":@NO,@"visual_quality_validated":@NO,@"rtx_parity_validated":@NO,@"hardware_acceleration_validated":@NO,
  @"apple_internal_architecture_known":@NO,@"netal_model_allocated":@NO,@"gpu_timing_validated":@NO,
  @"execution_smoke_validated":@NO,@"gpu_full_pass_ms":Summary({}),@"cpu_encode_ms":Summary({}),@"completion_wall_ms":Summary({}),
  @"os":NSProcessInfo.processInfo.operatingSystemVersionString,@"gpu":device.name?:@"unavailable",@"thermal_start":Thermal()} mutableCopy];
 auto fail=[&](NSString* text,NSString* state=@"failed",BOOL restart=NO)->NSDictionary*{
  r[@"reason"]=text;r[@"state"]=state;r[@"restart_required"]=@(restart);r[@"thermal_end"]=Thermal();
  if(error)*error=[NSError errorWithDomain:@"NetalMetalFXLab" code:1 userInfo:@{NSLocalizedDescriptionKey:text}];return r;};
#if !TARGET_OS_IOS || TARGET_OS_SIMULATOR || TARGET_OS_MACCATALYST
 (void)config;(void)shouldStop;return fail(@"This lab requires physical iOS.",@"unavailable");
#else
 if(!device)return fail(@"A Metal device is unavailable.",@"unavailable");
 if(![config isKindOfClass:NSDictionary.class])return fail(@"MetalFX configuration must be an object.");
 if(![device supportsFamily:MTLGPUFamilyMetal4])return fail(@"The GPU does not support Metal 4.",@"unavailable");
 NSString* mode=config[@"metalfx_mode"]?:@"temporal";
 if(![@[@"spatial",@"temporal"] containsObject:mode])return fail(@"metalfx_mode must be spatial or temporal.");
 BOOL temporal=[mode isEqual:@"temporal"],capture=NO;
 if(!netal_probe::ReadOptionalBoolean(config[@"capture"],false,&capture))return fail(@"capture must be a JSON boolean.");
 Dimensions in{},out{};NSUInteger warmup=0,measured=0;
 if(!ParseDimensions(config[@"render_size"],@"640x360",&in)||!ParseDimensions(config[@"profile"],@"1280x720",&out)||!netal_metalfx::ValidGeometry(in,out))
  return fail(@"Invalid render/output dimensions (positive, <=4096, input<=output, matching aspect ratio).");
 if(!Count(config[@"warmup_frames"],8,8,32,&warmup)||!Count(config[@"measured_frames"],60,1,240,&measured))
  return fail(@"Use 8–32 warm-up frames and 1–240 measured frames.");
 r[@"configuration"]=@{@"metalfx_mode":mode,@"render_size":[NSString stringWithFormat:@"%ux%u",in.width,in.height],
  @"profile":[NSString stringWithFormat:@"%ux%u",out.width,out.height],@"warmup_frames":@(warmup),@"measured_frames":@(measured),@"capture":@(capture)};
 r[@"geometry"]=@{@"input_width":@(in.width),@"input_height":@(in.height),@"output_width":@(out.width),@"output_height":@(out.height)};
 r[@"backend"]=temporal?@"Apple MetalFX temporal, Metal 4":@"Apple MetalFX spatial HDR, Metal 4";
 r[@"stimulus_contract"]=@{@"version":@2,@"cycles_u":@(netal_probe::kPlaneCyclesU),@"cycles_v":@(netal_probe::kPlaneCyclesV),
  @"velocity_u_per_frame":@(netal_probe::kPlaneVelocityU),@"jitter_phase_count":@18,@"jitter_units":@"input pixels",
  @"depth_reversed":@YES,@"pre_exposure":@1,@"exposure":@1,@"motion_units":@"normalized current-to-previous UV"};
 r[@"alpha_quality_validated"]=@NO;r[@"rgb_output_validation_only"]=@YES;
 const BOOL captureBuild=[NSBundle.mainBundle.infoDictionary[@"MetalCaptureEnabled"] boolValue];
 r[@"capture_instrumentation_enabled"]=@(captureBuild);
 if(capture&&!captureBuild)
  return fail(@"Capture requires a separate MetalCaptureEnabled build. Ordinary timing builds do not enable capture.",@"unavailable");
 if(captureBuild&&!capture)
  return fail(@"This capture-enabled build cannot produce ordinary performance measurements. Use a performance build or request capture.",@"unavailable");
 if(shouldStop&&shouldStop())return fail(@"Stopped before scaler creation.",@"incomplete");
 const auto setup=Clock::now();NSError* local=nil;
 id<MTL4Compiler> compiler=[device newCompilerWithDescriptor:[MTL4CompilerDescriptor new] error:&local];
 if(!compiler)return fail(local.localizedDescription?:@"Metal 4 compiler creation failed.",@"unavailable");
 id<MTL4FXSpatialScaler> spatial=nil;id<MTL4FXTemporalScaler> temporalScaler=nil;
 if(temporal){
  if(![MTLFXTemporalScalerDescriptor supportsMetal4FX:device])return fail(@"Metal 4 temporal scaling is unsupported.",@"unavailable");
  const float min=[MTLFXTemporalScalerDescriptor supportedInputContentMinScaleForDevice:device],max=[MTLFXTemporalScalerDescriptor supportedInputContentMaxScaleForDevice:device];
  r[@"supported_scale_range"]=@{@"minimum":Number(min),@"maximum":Number(max),@"units":@"output/input"};
  if(!netal_metalfx::WithinScaleRange(in,out,min,max))return fail(@"The requested temporal scale is unsupported by this device.",@"unavailable");
  auto d=[MTLFXTemporalScalerDescriptor new];d.inputWidth=in.width;d.inputHeight=in.height;d.outputWidth=out.width;d.outputHeight=out.height;
  d.colorTextureFormat=MTLPixelFormatRGBA16Float;d.outputTextureFormat=MTLPixelFormatRGBA16Float;d.motionTextureFormat=MTLPixelFormatRG16Float;
  d.depthTextureFormat=MTLPixelFormatDepth32Float;d.autoExposureEnabled=NO;d.requiresSynchronousInitialization=YES;
#if __IPHONE_OS_VERSION_MAX_ALLOWED >= 270000
  if(@available(iOS 27.0,*)){d.outputResolutionMotionVectorsEnabled=NO;d.jitteredMotionVectorsEnabled=NO;}
#endif
  temporalScaler=[d newTemporalScalerWithDevice:device compiler:compiler];
  if(!temporalScaler)return fail(@"Metal 4 temporal scaler creation failed; no substitute was used.",@"unavailable");
 }else{
  if(![MTLFXSpatialScalerDescriptor supportsMetal4FX:device])return fail(@"Metal 4 spatial scaling is unsupported.",@"unavailable");
  auto d=[MTLFXSpatialScalerDescriptor new];d.inputWidth=in.width;d.inputHeight=in.height;d.outputWidth=out.width;d.outputHeight=out.height;
  d.colorTextureFormat=MTLPixelFormatRGBA16Float;d.outputTextureFormat=MTLPixelFormatRGBA16Float;d.colorProcessingMode=MTLFXSpatialScalerColorProcessingModeHDR;
  spatial=[d newSpatialScalerWithDevice:device compiler:compiler];
  if(!spatial)return fail(@"Metal 4 spatial scaler creation failed; no substitute was used.",@"unavailable");
 }
 r[@"synchronous_initialization"]=@(temporal);
 if((shouldStop&&shouldStop())||NSProcessInfo.processInfo.thermalState==NSProcessInfoThermalStateCritical)
  return fail(@"Stopped after scaler creation, before working-resource allocation.",@"incomplete");
 Surface color,motion,depth,exposure,output;
 if(!MakeSurface(device,in,MTLPixelFormatRGBA16Float,8,temporal?temporalScaler.colorTextureUsage:spatial.colorTextureUsage,&color)||
    !MakeSurface(device,in,MTLPixelFormatRG16Float,4,temporal?temporalScaler.motionTextureUsage:MTLTextureUsageShaderRead,&motion)||
    !MakeSurface(device,in,MTLPixelFormatDepth32Float,4,temporal?temporalScaler.depthTextureUsage:MTLTextureUsageShaderRead,&depth)||
    !MakeSurface(device,{1,1},MTLPixelFormatR16Float,2,MTLTextureUsageShaderRead,&exposure)||
    !MakeSurface(device,out,MTLPixelFormatRGBA16Float,8,temporal?temporalScaler.outputTextureUsage:spatial.outputTextureUsage,&output))
  return fail(@"Working textures/buffers could not be allocated.");
 if(temporal){temporalScaler.colorTexture=color.texture;temporalScaler.depthTexture=depth.texture;temporalScaler.motionTexture=motion.texture;
  temporalScaler.exposureTexture=exposure.texture;temporalScaler.outputTexture=output.texture;temporalScaler.preExposure=1;
  temporalScaler.motionVectorScaleX=in.width;temporalScaler.motionVectorScaleY=in.height;temporalScaler.depthReversed=YES;
  temporalScaler.inputContentWidth=in.width;temporalScaler.inputContentHeight=in.height;
 }else{spatial.colorTexture=color.texture;spatial.outputTexture=output.texture;spatial.inputContentWidth=in.width;spatial.inputContentHeight=in.height;}
 id<MTL4CommandQueue> queue=[device newMTL4CommandQueue];id<MTL4CommandAllocator> allocator=[device newCommandAllocator];
 MTLResidencySetDescriptor* rd=[MTLResidencySetDescriptor new];rd.initialCapacity=20;
 id<MTLResidencySet> residency=[device newResidencySetWithDescriptor:rd error:&local];
 id<MTLBuffer> poison=[device newBufferWithLength:output.bytes options:MTLResourceStorageModeShared];
 if(!queue||!allocator||!residency||!poison)return fail(@"Metal 4 queue, allocator, residency or validation allocation failed.");
 std::memset(poison.contents,0xa5,output.bytes);
 for(NSUInteger y=0;y<output.height;++y){auto* p=reinterpret_cast<std::uint16_t*>(static_cast<unsigned char*>(poison.contents)+netal_metalfx::kGuard+y*output.row);
  std::fill(p,p+output.width*4,netal_metalfx::kPoison);}
 NSMutableArray* resources=[NSMutableArray arrayWithObjects:device,compiler,queue,allocator,residency,poison,nil];
 if(spatial)[resources addObject:spatial];if(temporalScaler)[resources addObject:temporalScaler];
 for(Surface* s:{&color,&motion,&depth,&exposure,&output}){[resources addObjectsFromArray:@[s->texture,s->host,s->verification]];
  [residency addAllocation:s->texture];[residency addAllocation:s->host];[residency addAllocation:s->verification];}
 [residency addAllocation:poison];[residency commit];
 NSUInteger callerBufferBytes=poison.length,callerTextureBytes=0;
 for(Surface* s:{&color,&motion,&depth,&exposure,&output}){callerBufferBytes+=s->host.length+s->verification.length;callerTextureBytes+=s->texture.allocatedSize;}
 r[@"caller_memory"]=@{@"buffer_bytes":@(callerBufferBytes),@"texture_allocated_bytes":@(callerTextureBytes),
  @"excludes_framework_internal_allocations":@YES};
 r[@"cold_prepare_wall_ms"]=@(MS(setup));r[@"row_strides"]=@{@"color":@(color.row),@"motion":@(motion.row),@"depth":@(depth.row),@"output":@(output.row),@"alignment":@256};
 const auto runStart=Clock::now();BOOL timedOut=NO,captureFinalizationPending=NO;
 auto stop=[&]()->bool{return (shouldStop&&shouldStop())||NSProcessInfo.processInfo.thermalState==NSProcessInfoThermalStateCritical||MS(runStart)>180000.;};
 auto submit=[&](void (^encode)(id<MTL4CommandBuffer>),BOOL captured,double* gpu,double* cpu,double* wall)->bool{
  auto abandonUnsubmittedCapture=[&](){
   if(captured&&MTLCaptureManager.sharedCaptureManager.isCapturing)[MTLCaptureManager.sharedCaptureManager stopCapture];
  };
  if(stop()){abandonUnsubmittedCapture();return false;}
  [allocator reset];id<MTL4CommandBuffer> command=[device newCommandBuffer];
  if(!command){abandonUnsubmittedCapture();return false;}
  NSMutableArray* keep=[resources mutableCopy];[keep addObject:command];
  dispatch_semaphore_t done=dispatch_semaphore_create(0);
  dispatch_semaphore_t captureStopped=dispatch_semaphore_create(0);
  __block double begin=0,end=0;__block NSError* commandError=nil;
  [command beginCommandBufferWithAllocator:allocator];[command useResidencySet:residency];
  const auto wallStart=Clock::now(),cpuStart=Clock::now();encode(command);[command endCommandBuffer];*cpu=MS(cpuStart);
  MTL4CommitOptions* options=[MTL4CommitOptions new];
  [options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback){(void)keep.count;begin=feedback.GPUStartTime;end=feedback.GPUEndTime;commandError=feedback.error;
   dispatch_semaphore_signal(done);
   // Stop outside the driver's callback. This also finalizes a late capture after a GPU timeout.
   if(captured)dispatch_async(dispatch_get_global_queue(QOS_CLASS_UTILITY,0),^{
    (void)keep.count; // Trace finalization may still inspect the captured resources.
    if(MTLCaptureManager.sharedCaptureManager.isCapturing)[MTLCaptureManager.sharedCaptureManager stopCapture];
    dispatch_semaphore_signal(captureStopped);
   });
  }];
  id<MTL4CommandBuffer> list[]={command};[queue commit:list count:1 options:options];
  if(dispatch_semaphore_wait(done,dispatch_time(DISPATCH_TIME_NOW,30*NSEC_PER_SEC))){timedOut=YES;return false;}
  *wall=MS(wallStart);*gpu=netal_probe::GPUMilliseconds(begin,end,true,commandError!=nil);
  if(captured&&dispatch_semaphore_wait(captureStopped,dispatch_time(DISPATCH_TIME_NOW,10*NSEC_PER_SEC))){
   captureFinalizationPending=YES;r[@"capture_finalization_pending"]=@YES;return false;
  }
  return commandError==nil;
 };
 auto uploadInputs=[&](id<MTL4ComputeCommandEncoder> e){Upload(e,color,color.host);if(temporal){Upload(e,motion,motion.host);Upload(e,depth,depth.host);Upload(e,exposure,exposure.host);}};
 Inputs(color,motion,depth,exposure,0,kJitter[0][0],kJitter[0][1]);
 double gpu=0,cpu=0,wall=0;
 // An untimed roundtrip verifies every input format and padded row independently.
 if(!submit(^(id<MTL4CommandBuffer> command){id<MTL4ComputeCommandEncoder> e=[command computeCommandEncoder];
  uploadInputs(e);Barrier(e);Download(e,color,color.verification);if(temporal){Download(e,motion,motion.verification);Download(e,depth,depth.verification);Download(e,exposure,exposure.verification);}[e endEncoding];
 },NO,&gpu,&cpu,&wall))return fail(@"Input transfer validation did not complete.",stop()?@"incomplete":@"failed",timedOut);
 if(!EqualRows(color)||(temporal&&(!EqualRows(motion)||!EqualRows(depth)||!EqualRows(exposure))))return fail(@"Buffer/texture format or row roundtrip failed.");
 r[@"input_roundtrip_validated"]=@YES;
 r[@"input_roundtrip_gpu_ms"]=Number(gpu);
 auto scale=[&](id<MTL4CommandBuffer> command,BOOL poisonOutput){
  id<MTL4ComputeCommandEncoder> pre=[command computeCommandEncoder];uploadInputs(pre);if(poisonOutput)Upload(pre,output,poison);
  [pre barrierAfterStages:MTLStageBlit beforeQueueStages:MTLStageAll visibilityOptions:MTL4VisibilityOptionDevice];[pre endEncoding];
  if(temporal)[temporalScaler encodeToCommandBuffer:command];else[spatial encodeToCommandBuffer:command];
  id<MTL4ComputeCommandEncoder> post=[command computeCommandEncoder];
  [post barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageBlit visibilityOptions:MTL4VisibilityOptionDevice];
  Download(post,output,output.host);[post endEncoding];
 };
 // Private output NaN poison catches partial scaler writes, outside timed frames.
 if(temporal){temporalScaler.reset=YES;temporalScaler.jitterOffsetX=kJitter[0][0];temporalScaler.jitterOffsetY=kJitter[0][1];}
 if(!submit(^(id<MTL4CommandBuffer> c){scale(c,YES);},NO,&gpu,&cpu,&wall))return fail(@"Output-write validation did not complete.",stop()?@"incomplete":@"failed",timedOut);
 if(!Guards(output,output.host)||InvalidRGB(output))return fail(@"The scaler left nonfinite/unwritten RGB or damaged output guards.");
 r[@"private_output_nan_poison_validated"]=@YES;r[@"validation_work_excluded_from_timing"]=@YES;
 r[@"first_scaler_validation_gpu_ms"]=Number(gpu);
 r[@"first_scaler_validation_includes_poison_upload"]=@YES;
 std::vector<double> times,cpus,walls;NSMutableArray* rows=[NSMutableArray array];r[@"frames"]=rows;
 NSURL* traceURL=nil;BOOL traceWritten=NO,timingValid=YES;
 const NSUInteger total=capture?warmup+1:warmup+measured;
 for(NSUInteger frame=0;frame<total;++frame){@autoreleasepool{
  if(stop()){r[@"encoded_frames"]=@(rows.count);return fail(@"Stopped between completed frames (cancelled, inactive, critical thermal or deadline).",@"incomplete");}
  const auto jitter=kJitter[frame%kJitter.size()];Inputs(color,motion,depth,exposure,frame,jitter[0],jitter[1]);
  std::memset(output.host.contents,0xa5,output.bytes);
  if(temporal){temporalScaler.reset=frame==0;temporalScaler.jitterOffsetX=jitter[0];temporalScaler.jitterOffsetY=jitter[1];}
  BOOL capturing=capture&&frame==warmup;
  if(capturing){
   MTLCaptureManager* manager=MTLCaptureManager.sharedCaptureManager;
   if(![manager supportsDestination:MTLCaptureDestinationGPUTraceDocument])return fail(@"GPU trace documents are unsupported in this process.",@"unavailable");
   NSURL* docs=[NSFileManager.defaultManager URLsForDirectory:NSDocumentDirectory inDomains:NSUserDomainMask].firstObject;
   NSString* name=[NSString stringWithFormat:@"metalfx-%@-%@.gputrace",mode,NSUUID.UUID.UUIDString];traceURL=[docs URLByAppendingPathComponent:name];
   MTLCaptureDescriptor* d=[MTLCaptureDescriptor new];d.captureObject=device;d.destination=MTLCaptureDestinationGPUTraceDocument;d.outputURL=traceURL;
   if(![manager startCaptureWithDescriptor:d error:&local])return fail(local.localizedDescription?:@"GPU capture could not start.",@"unavailable");
  }
  if(!submit(^(id<MTL4CommandBuffer> c){scale(c,NO);},capturing,&gpu,&cpu,&wall))
   return fail(captureFinalizationPending?@"Capture finalization is still pending. Restart the app before another run.":
    @"The GPU command did not complete safely. A timed-out run retains its in-flight resources.",
    stop()?@"incomplete":@"failed",timedOut||captureFinalizationPending);
  const NSUInteger bad=InvalidRGB(output);const BOOL guards=Guards(output,output.host);
  if(bad||!guards)return fail(@"MetalFX output has nonfinite RGB or damaged guards; no performance conclusion is valid.");
  [rows addObject:@{@"index":@(frame),@"warmup":@(frame<warmup),@"gpu_ms":capture?NSNull.null:Number(gpu),
   @"cpu_encode_ms":capture?NSNull.null:@(cpu),@"completion_wall_ms":capture?NSNull.null:@(wall),@"rgb_nonfinite":@(bad),@"canaries_intact":@(guards),@"thermal":Thermal()}];
  if(frame>=warmup&&!capture){times.push_back(gpu);cpus.push_back(cpu);walls.push_back(wall);timingValid&=std::isfinite(gpu);}
  if(capturing){traceWritten=[NSFileManager.defaultManager fileExistsAtPath:traceURL.path];break;}
 }}
 r[@"encoded_frames"]=@(rows.count);r[@"thermal_end"]=Thermal();r[@"run_wall_ms"]=@(MS(runStart));
 r[@"execution_smoke_validated"]=@YES;
 if(capture){r[@"state"]=traceWritten?@"captured":@"failed";r[@"capture"]=@{@"written":@(traceWritten),@"path":traceURL.lastPathComponent?:@"",@"private_only":@YES,
   @"may_contain_framework_resources":@YES,@"performance_measurement":@NO};r[@"reason"]=traceWritten?@"Captured one warmed MetalFX frame. Capture timings are intentionally unavailable.":@"Capture stopped without a trace document.";
 }else{r[@"gpu_full_pass_ms"]=Summary(times);r[@"cpu_encode_ms"]=Summary(cpus);r[@"completion_wall_ms"]=Summary(walls);
  r[@"gpu_timing_validated"]=@(timingValid&&times.size()==measured);r[@"state"]=timingValid?@"completed":@"completed_timing_unavailable";
  r[@"reason"]=@"Finished isolated MetalFX buffer-to-buffer comparison. This is not gameplay FPS or Apple model-architecture evidence.";}
 // The hash covers exact padded readback bytes; a separate RGB quality oracle is still required.
 r[@"last_output_readback_sha256"]=SHA(output.host.contents,output.bytes);return r;
#endif
}
