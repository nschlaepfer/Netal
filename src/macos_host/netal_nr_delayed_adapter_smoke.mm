// Standalone simulated-renderer test of the ACTUAL delayed reset worker.
// No game, drawable, expected model tensor, or archived temporal history enters
// execution. An optional replay file is ONLY the external RG11B10F game input.
#define NETAL_NR_DELAYED_ONCE 1
#include "netal_nr_experimental_bridge.mm"
#include <chrono>
#include <stdexcept>
#include <thread>

@interface DelayedScalerFixture : NSObject
@property(nonatomic,strong) id<MTLTexture> outputTexture, colorTexture, motionTexture, depthTexture, exposureTexture;
@property(nonatomic,strong) id<MTLFence> fence;
@property(nonatomic) NSUInteger inputContentWidth, inputContentHeight, outputWidth, outputHeight;
@end
@implementation DelayedScalerFixture
@end

static void Need(bool value,const char* why) { if(!value)throw std::runtime_error(why); }
static double Milliseconds(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
}
static void Save(NSData* data,NSString* path) {
  NSError* error=nil;
  Need(data && [data writeToFile:path options:NSDataWritingWithoutOverwriting error:&error],
    error ? error.description.UTF8String : "receipt write failed");
}
static void SaveJSON(id object,NSString* path) {
  NSError* error=nil;
  auto data=[NSJSONSerialization dataWithJSONObject:object options:NSJSONWritingPrettyPrinted|NSJSONWritingSortedKeys error:&error];
  Need(data!=nil,"JSON serialization failed");Save(data,path);
}

int main(int argc,char** argv) { @autoreleasepool {
  try {
    Need(argc==5,"usage: delayed-adapter-smoke CLOSED_BUNDLE CODECS POISON NEW_PROCESS_DIR");
    NSString* out=@(argv[4]);
    Need(![[NSFileManager defaultManager] fileExistsAtPath:out],"preserve prior process directory");
    Need([[NSFileManager defaultManager] createDirectoryAtPath:out withIntermediateDirectories:NO attributes:nil error:nil],"process directory creation");
    char* end=nullptr;const long poison=std::strtol(argv[3],&end,16);
    Need(end && !*end && (poison==0xa5 || poison==0x5a),"explicit a5/5a poison required");
    constexpr NSUInteger width=1920,height=1200,bytes=width*height*4;
    NSData* replay=nil;
    if(const char* path=std::getenv("NR_ADAPTER_EXTERNAL_INPUT")) {
      Need(path[0]=='/',"external input must be absolute");
      replay=[NSData dataWithContentsOfFile:@(path)];
      Need(replay && replay.length==bytes,"external game-input size/read failure");
    }
    capture_root=std::string(argv[4])+"/captures";
    arm_file=std::string(argv[4])+"/arm";
    Save([NSData data],@(arm_file.c_str()));
    delayed_once_mode=true;once_mode=false;color_mode=ColorMode::BoundedReinhardSrgb;
    delayed_phase_gates=std::getenv("NR_ADAPTER_PHASE_GATES") && std::strcmp(std::getenv("NR_ADAPTER_PHASE_GATES"),"1")==0;
    Need(setenv("NETAL_NR_RUNTIME_BUNDLE",argv[1],1)==0 && setenv("NETAL_NR_FORMAT_LIBRARY",argv[2],1)==0,"prepare environment");
    Need(setenv("NETAL_NR_ARITHMETIC","original",1)==0,"original arithmetic environment");
    auto device=MTLCreateSystemDefaultDevice();Need(device!=nil,"Metal device");
    const auto prepare_start=std::chrono::steady_clock::now();StartPrepare(device);
    for(;;) {
      {std::lock_guard<std::mutex> lock(state_mutex);
        if(ready==Ready::Failed)throw std::runtime_error(reset_prepare_report.error);
        if(ready==Ready::Ready){Need(bool(reset_worker),"reset worker absent");break;}}
      Need(Milliseconds(prepare_start)<300000,"worker prepare timeout");
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    const double prepare_ms=Milliseconds(prepare_start);
    std::printf("actual_pinned_NR=1 simulated_renderer=1 game_connected=0 offline_external_input_replay=%d model=%s weights=%s prepare_ms=%.3f\n",
      replay!=nil,reset_prepare_report.model_identity.c_str(),reset_prepare_report.weights_identity.c_str(),prepare_ms);
    NSError* error=nil;
    // Frames change after capture, so an accidentally repeated frozen result is
    // observable. The replay is used only for the FIRST external source frame.
    NSString* shader=@"#include <metal_stdlib>\nusing namespace metal;\nkernel void source_frame(texture2d<half,access::write> out [[texture(0)]],constant uint& frame [[buffer(0)]],uint2 p [[thread_position_in_grid]]){if(p.x>=out.get_width()||p.y>=out.get_height())return;float x=float(p.x)/out.get_width(),y=float(p.y)/out.get_height();float t=float(frame%17)/64.0f;out.write(half4(0.1f+0.6f*x+t,0.15f+0.6f*y,0.2f+0.3f*x*y,1.0f),p);}";
    auto library=[device newLibraryWithSource:shader options:nil error:&error];Need(library!=nil,"producer shader");
    auto producer=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"source_frame"] error:&error];Need(producer!=nil,"producer pipeline");
    auto td=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRG11B10Float width:width height:height mipmapped:NO];
    td.storageMode=MTLStorageModePrivate;td.hazardTrackingMode=MTLHazardTrackingModeTracked;
    td.usage=MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite|MTLTextureUsageRenderTarget;
    auto texture=[device newTextureWithDescriptor:td];
    auto before=[device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    auto after=[device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    auto poison_buffer=[device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    auto external=replay ? [device newBufferWithBytes:replay.bytes length:bytes options:MTLResourceStorageModeShared] : nil;
    Need(texture && before && after && poison_buffer && (!replay || external),"frame resources");
    std::memset(poison_buffer.contents,int(poison),bytes);
    auto queue=[device newCommandQueue];Need(queue!=nil,"simulated renderer queue");
    DelayedScalerFixture* scaler=[DelayedScalerFixture new];scaler.outputTexture=texture;
    scaler.inputContentWidth=1152;scaler.inputContentHeight=720;scaler.outputWidth=width;scaler.outputHeight=height;
    NSMutableArray* frame_records=[NSMutableArray array];
    const uint64_t first_frame=poison==0xa5?123456:0;
    unsigned publications=0,original_frames_during_compute=0,after_done=0;
    unsigned compute_gate_original_frames=0,ready_gate_original_frames=0;
    const auto start=std::chrono::steady_clock::now();
    using Phase=netal_nr_adapter::DelayedOnceState::Phase;
    for(uint32_t index=0;index<20000;++index) { @autoreleasepool {
      Need(Milliseconds(start)<300000,"delayed worker/publication deadline exceeded");
      const uint64_t frame=first_frame+index;
      const auto frame_start=std::chrono::steady_clock::now();
      auto cb=[queue commandBufferWithUnretainedReferences];Need(cb && !cb.retainedReferences,"unretained caller contract");
      std::memset(before.contents,int(poison^255),bytes);std::memset(after.contents,int(poison),bytes);
      auto b=[cb blitCommandEncoder];Need(b!=nil,"source poison encoder");
      [b copyFromBuffer:poison_buffer sourceOffset:0 sourceBytesPerRow:width*4 sourceBytesPerImage:bytes sourceSize:MTLSizeMake(width,height,1)
        toTexture:texture destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0,0,0)];[b endEncoding];
      if(index==0 && external) {
        b=[cb blitCommandEncoder];Need(b!=nil,"external upload encoder");
        [b copyFromBuffer:external sourceOffset:0 sourceBytesPerRow:width*4 sourceBytesPerImage:bytes sourceSize:MTLSizeMake(width,height,1)
          toTexture:texture destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0,0,0)];[b endEncoding];
      } else {
        auto c=[cb computeCommandEncoder];Need(c!=nil,"producer encoder");
        [c setComputePipelineState:producer];[c setTexture:texture atIndex:0];[c setBytes:&index length:4 atIndex:0];
        [c dispatchThreads:MTLSizeMake(width,height,1) threadsPerThreadgroup:MTLSizeMake(8,8,1)];[c endEncoding];
      }
      const auto readback=[&](id<MTLBuffer> buffer) {
        auto copy=[cb blitCommandEncoder];Need(copy!=nil,"readback encoder");
        [copy copyFromTexture:texture sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0) sourceSize:MTLSizeMake(width,height,1)
          toBuffer:buffer destinationOffset:0 destinationBytesPerRow:width*4 destinationBytesPerImage:bytes];[copy endEncoding];
      };
      readback(before);
      auto old_job=DelayedJob();const auto old_phase=old_job?old_job->state.snapshot().phase:Phase::Idle;
      const auto append_start=std::chrono::steady_clock::now();
      AppendDelayedOnce((id<MTLFXTemporalScaler>)scaler,cb,frame,7);
      const double append_ms=Milliseconds(append_start);
      Need(cb.status==MTLCommandBufferStatusNotEnqueued,"bridge submitted caller command");
      readback(after);[cb commit];[cb waitUntilCompleted];
      const double frame_wall_ms=Milliseconds(frame_start);
      Need(cb.status==MTLCommandBufferStatusCompleted,cb.error?cb.error.description.UTF8String:"simulated game GPU failure");
      auto job=DelayedJob();Need(bool(job),"no source capture admitted");
      // waitUntilCompleted need not synchronize the bridge's handler body.
      const auto callback_deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
      auto s=job->state.snapshot();
      while((s.phase==Phase::Capturing || (s.phase==Phase::Publishing && s.publication_frame==frame) || (old_phase==Phase::Recovering && s.phase==Phase::Recovering)) &&
        std::chrono::steady_clock::now()<callback_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));s=job->state.snapshot();
      }
      Need(s.phase!=Phase::Failed,"delayed bridge failed; preserve log/captures");
      const bool published=s.export_completed && s.publication_frame==frame;
      const bool equal=std::memcmp(before.contents,after.contents,bytes)==0;
      if(published){++publications;Need(index>0 && !equal,"publication not later or output identity only");
        Save([NSData dataWithBytes:after.contents length:bytes],[out stringByAppendingPathComponent:@"published.rg11b10f"]);}
      else Need(equal,"nonpublication frame altered; original renderer not preserved");
      if(index==0){Need(equal,"capture command changed source");
        Need(!external || std::memcmp(external.contents,before.contents,bytes)==0,"external source upload mismatch");
        Save([NSData dataWithBytes:before.contents length:bytes],[out stringByAppendingPathComponent:@"source.rg11b10f"]);}
      if(old_phase==Phase::Computing && !published)++original_frames_during_compute;
      if(delayed_phase_gates && s.phase==Phase::WaitingForCompute) {
        Need(equal && !s.model_computed,"compute gate did not preserve original output");
        __block bool started=false;dispatch_sync(job->host_queue,^{started=bool(job->reset_job);});
        Need(!started,"NR began before manual compute release");
        if(++compute_gate_original_frames==4)Save([NSData data],@(job->compute_path.c_str()));
      }
      if(delayed_phase_gates && s.phase==Phase::Ready) {
        Need(equal && s.model_computed && !s.publication_may_be_encoded,"ready gate leaked output");
        if(++ready_gate_original_frames==4)Save([NSData data],@(job->publish_path.c_str()));
      }
      const double gpu_ms=(cb.GPUEndTime-cb.GPUStartTime)*1000;
      [frame_records addObject:@{@"frame":@(frame),@"index":@(index),@"phase_before":@(int(old_phase)),@"phase_after":@(int(s.phase)),
        @"published":@(published),@"original_bytes_equal":@(equal),@"append_cpu_ms":@(append_ms),@"whole_command_gpu_ms":@(gpu_ms),@"frame_wall_ms":@(frame_wall_ms)}];
      std::printf("frame=%llu phase=%d published=%d original_equal=%d append_ms=%.3f gpu_ms=%.3f wall_ms=%.3f\n",
        (unsigned long long)frame,int(s.phase),published,equal,append_ms,gpu_ms,frame_wall_ms);std::fflush(stdout);
      if(s.phase==Phase::Done){Need(equal && s.original_rendering_recovered,"recovery did not preserve original frame");
        if(++after_done==3){Save([NSData dataWithBytes:after.contents length:bytes],[out stringByAppendingPathComponent:@"recovered.rg11b10f"]);break;}}
      std::this_thread::sleep_for(std::chrono::milliseconds(16));
    } }
    auto job=DelayedJob();Need(bool(job),"missing completed job");const auto state=job->state.snapshot();
    Need(state.phase==Phase::Done && publications==1 && nr_published.load()==1 && after_done==3,"exactly one publish/recovery not established");
    Need(original_frames_during_compute>=2,"insufficient original frames observed while worker ran");
    Need(!state.drawable_presented,"standalone fixture must not claim drawable presentation");
    Need(!delayed_phase_gates || (compute_gate_original_frames>=4 && ready_gate_original_frames==4),"manual gates not exercised");
    const auto write_deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
    for(NSString* phase in @[@"source",@"input",@"computed",@"published"]) {
      NSString* path=[NSString stringWithFormat:@"%s/%@/frame-%llu-stream-7/capture.json",capture_root.c_str(),phase,(unsigned long long)first_frame];
      while(![[NSFileManager defaultManager] fileExistsAtPath:path] && std::chrono::steady_clock::now()<write_deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
      Need([[NSFileManager defaultManager] fileExistsAtPath:path],"phase capture incomplete");
    }
    __block nr::ResetProgress progress;
    dispatch_sync(job->host_queue,^{progress=nr::reset_worker_poll(job->reset_job);});
    Need(progress.state==nr::ResetState::Completed && progress.output_publishable && progress.chunks_done==progress.chunks_total,"worker not fully completed");
    const auto durations=nr::reset_worker_chunk_durations(job->reset_job);
    uint32_t dropped_samples=0;bool drop_count_available=false;
#if NR_DELAYED_HAS_DROPPED_COUNTER
    dropped_samples=nr::reset_worker_dropped_duration_samples(job->reset_job);drop_count_available=true;
    Need(dropped_samples==0,"worker duration samples were dropped");
#endif
    SaveJSON(@{@"schema":@"neuralstudio.nr.delayed-adapter-smoke.v1",@"result":@"pass",@"pid":@(getpid()),@"poison":@(argv[3]),
      @"synthetic_source":@(!replay),@"offline_external_input_replay":@(replay!=nil),@"actual_model":@YES,
      @"model":@(reset_prepare_report.model_identity.c_str()),@"weights":@(reset_prepare_report.weights_identity.c_str()),
      @"caller_retained_references":@NO,@"source_frame":@(first_frame),@"publication_frame":@(state.publication_frame),
      @"phase_gates":@(delayed_phase_gates),@"compute_gate_original_frames":@(compute_gate_original_frames),@"ready_gate_original_frames":@(ready_gate_original_frames),
      @"publication_count":@(publications),@"original_frames_during_compute":@(original_frames_during_compute),@"original_recovered":@YES,
      @"subsequent_original_frames":@(after_done),@"prepare_ms":@(prepare_ms),@"chunks":@(progress.chunks_done),
      @"logical_units_completed":@(progress.chunks_done),@"gpu_submissions_recorded":@(durations.size()),
      @"duration_sample_drop_count":drop_count_available?@(dropped_samples):(id)[NSNull null],
      @"total_worker_gpu_ms":@(progress.total_gpu_ms),@"worst_worker_chunk_gpu_ms":@(progress.worst_chunk_gpu_ms),@"frames":frame_records,
      @"game_connected":@NO,@"screen_presented":@NO,@"visual_accepted":@NO,@"exact_rtx_parity":@NO,@"playable":@NO,
      @"limits":@"Simulated renderer with readback on each frame; two external-texture poisons do not prove internal write coverage; no live game, motion/history, quality or performance acceptance."},
      [out stringByAppendingPathComponent:@"receipt.json"]);
    std::printf("result=pass delayed_actual_model=1 publication_count=1 original_recovery=1 simulated_renderer_only=1\n");return 0;
  }catch(const std::exception& e){std::fprintf(stderr,"result=FAIL detail=%s\n",e.what());return 1;}
} }
