// Synthetic producer OR explicitly labelled external game-input replay -> ACTUAL
// NR -> guarded packed publication in ONE command. Never loads model tensors.
// Standalone harness only: it owns its queue, waits and reads back completed data.
// It does not launch a game and cannot establish in-game/visual/RTX acceptance.
#include "netal_nr_experimental_bridge.mm"
#include <chrono>
#include <fstream>
#include <thread>

@interface AdapterScalerFixture : NSObject
@property(nonatomic,strong) id<MTLTexture> outputTexture, colorTexture, motionTexture, depthTexture, exposureTexture;
@property(nonatomic,strong) id<MTLFence> fence;
@property(nonatomic) NSUInteger inputContentWidth, inputContentHeight;
@property(nonatomic) NSUInteger outputWidth, outputHeight;
@end
@implementation AdapterScalerFixture
@end

static void Check(bool yes,const char* why) { if(!yes) throw std::runtime_error(why); }
int main(int argc,char** argv) { @autoreleasepool {
  try {
    Check(argc==5 || argc==6,"usage: live-adapter-smoke BUNDLE CODECS POISON_BYTE NEW_OUTPUT_FILE [--unretained]");
    const bool unretained=argc==6;
    Check(!unretained || !std::strcmp(argv[5],"--unretained"),"unknown command-buffer option");
    Check(ParseColorMode(std::getenv("NETAL_NR_COLOR_MODE"),color_mode),"unknown color profile");
    const char* source_fault_text=std::getenv("NR_ADAPTER_SOURCE_FAULT");
    Check(!source_fault_text || !std::strcmp(source_fault_text,"inf"),"unknown synthetic source fault");
    const bool source_fault=source_fault_text!=nullptr;
    Check(!source_fault || color_mode==ColorMode::BoundedReinhardSrgb,"source fault test requires bounded color mode");
    const char* requested_height=std::getenv("NR_ADAPTER_GAME_HEIGHT");
    Check(!requested_height || !std::strcmp(requested_height,"1080") || !std::strcmp(requested_height,"1200"),"unsupported game-height profile");
    const NSUInteger game_height=requested_height ? std::strtoul(requested_height,nullptr,10) : 1080;
    const NSUInteger bytes=1920*game_height*4;
    NSData* replay=nil;
    if(const char* path=std::getenv("NR_ADAPTER_EXTERNAL_INPUT")) {
      Check(path[0]=='/',"external input requires absolute path");
      replay=[NSData dataWithContentsOfFile:@(path)];
      Check(replay!=nil && replay.length==bytes,"external RG11B10F input size/read failure");
    }
    Check(!source_fault || !replay,"source fault test requires synthetic source, not a capture");
    if(const char* dir=std::getenv("NR_ADAPTER_CAPTURE_DIR"))capture_root=dir;
    Check(![[NSFileManager defaultManager] fileExistsAtPath:@(argv[4])],"preserve prior output");
    char* end=nullptr; long poison=std::strtol(argv[3],&end,16);
    Check(end && !*end && poison>=0 && poison<=255,"poison byte");
    auto device=MTLCreateSystemDefaultDevice(); Check(device!=nil,"no device");
    nr::PrepareOptions opt; opt.runtime_bundle=argv[1]; nr::Report report;
    Check(nr::prepare(device,opt,context,report),report.error.c_str());
    std::printf("synthetic_producer=%d offline_external_input_replay=%d source_fault_test=%d actual_pinned_NR=1 game_connected=0 model=%s weights=%s\n",replay==nil,replay!=nil,source_fault,report.model_identity.c_str(),report.weights_identity.c_str());
    NSError* error=nil;
    auto codecs=[device newLibraryWithURL:[NSURL fileURLWithPath:@(argv[2])] error:&error]; Check(codecs!=nil,"codec library");
    auto pipeline=[&](NSString* name) {
      auto f=[codecs newFunctionWithName:name]; Check(f!=nil,"codec function");
      auto p=[device newComputePipelineStateWithFunction:f error:&error]; Check(p!=nil,"codec pipeline"); return p;
    };
    import_rg11=pipeline(@"neuralstudio_sr_import_rg11b10f");
    scan_finite=pipeline(@"netal_nr_scan_finite"); publish_finite=pipeline(@"netal_nr_publish_finite");
    fit_input=pipeline(@"netal_nr_fit_input");fit_publish=pipeline(@"netal_nr_fit_publish");
    if(color_mode==ColorMode::BoundedReinhardSrgb) {
      color_forward=pipeline(@"netal_nr_color_to_bounded");
      color_inverse=pipeline(@"netal_nr_color_from_bounded");
    }
    ready=Ready::Ready; once_mode=true;
    // Default source is generated on the GPU in this SAME uncommitted buffer.
    // Explicit replay replaces ONLY external color, not any model activation,
    // expected output or temporal history. Fit/padding and all NR stages rerun.
    NSString* source=@"#include <metal_stdlib>\nusing namespace metal;\nkernel void producer(texture2d<half,access::write> out [[texture(0)]],constant uint& fault [[buffer(0)]],uint2 p [[thread_position_in_grid]]) { if(p.x>=out.get_width()||p.y>=out.get_height())return; float x=float(p.x)/float(out.get_width()), y=float(p.y)/float(out.get_height()); half4 v=half4(0.1f+0.6f*x,0.15f+0.6f*y,0.2f+0.3f*x*y,1.0f); if(fault && p.x==out.get_width()/2 && p.y==out.get_height()/2)v.x=as_type<half>(ushort(0x7c00)); out.write(v,p); }";
    auto plib=[device newLibraryWithSource:source options:nil error:&error]; Check(plib!=nil,"producer library");
    auto producer=[device newComputePipelineStateWithFunction:[plib newFunctionWithName:@"producer"] error:&error]; Check(producer!=nil,"producer pipeline");
    auto td=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRG11B10Float width:1920 height:game_height mipmapped:NO];
    td.storageMode=MTLStorageModePrivate; td.hazardTrackingMode=MTLHazardTrackingModeTracked;
    td.usage=MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite|MTLTextureUsageRenderTarget;
    auto game=[device newTextureWithDescriptor:td]; Check(game!=nil,"game-shaped texture");
    auto external=replay ? [device newBufferWithBytes:replay.bytes length:bytes options:MTLResourceStorageModeShared] : nil;
    Check(!replay || external!=nil,"external input buffer");
    if(external) std::printf("external_input_sha256=%s external_input_bytes=%lu\n",netal_nr_adapter::CaptureSHA(external).UTF8String,(unsigned long)bytes);
    auto poison_buffer=[device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    auto result=[device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    auto producer_readback=[device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    Check(poison_buffer!=nil && result!=nil && producer_readback!=nil,"buffers");
    std::memset(poison_buffer.contents,int(poison),bytes); std::memset(result.contents,int(poison^255),bytes);
    AdapterScalerFixture* fixture=[AdapterScalerFixture new]; fixture.outputTexture=game;
    fixture.inputContentWidth=1152; fixture.inputContentHeight=game_height*3/5;
    fixture.outputWidth=1920; fixture.outputHeight=game_height;
    auto queue=[device newCommandQueue];
    auto cb=unretained ? [queue commandBufferWithUnretainedReferences] : [queue commandBuffer];
    Check(cb!=nil && bool(cb.retainedReferences)==!unretained,"command-buffer retention contract");
    std::printf("caller_retained_references=%d Metal_validation_requested=%s\n",cb.retainedReferences,std::getenv("MTL_DEBUG_LAYER") ?: "unspecified");
    auto b=[cb blitCommandEncoder];
    [b copyFromBuffer:poison_buffer sourceOffset:0 sourceBytesPerRow:1920*4 sourceBytesPerImage:bytes sourceSize:MTLSizeMake(1920,game_height,1)
       toTexture:game destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0,0,0)]; [b endEncoding];
    if(external) {
      b=[cb blitCommandEncoder];
      [b copyFromBuffer:external sourceOffset:0 sourceBytesPerRow:1920*4 sourceBytesPerImage:bytes sourceSize:MTLSizeMake(1920,game_height,1)
         toTexture:game destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0,0,0)]; [b endEncoding];
    } else {
      auto c=[cb computeCommandEncoder]; [c setComputePipelineState:producer]; [c setTexture:game atIndex:0];
      const uint32_t fault=uint32_t(source_fault);[c setBytes:&fault length:4 atIndex:0];
      [c dispatchThreads:MTLSizeMake(1920,game_height,1) threadsPerThreadgroup:MTLSizeMake(8,8,1)]; [c endEncoding];
    }
    b=[cb blitCommandEncoder];
    [b copyFromTexture:game sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0) sourceSize:MTLSizeMake(1920,game_height,1)
       toBuffer:producer_readback destinationOffset:0 destinationBytesPerRow:1920*4 destinationBytesPerImage:bytes]; [b endEncoding];
    // Global identities differ across fresh processes; reset-local noise must
    // remain zero. The capture above is output-only and never enters the model.
    const uint64_t global_frame=poison==0xa5?123456:0;
    AppendDiagnosticOnce((id<MTLFXTemporalScaler>)fixture,cb,global_frame,7);
    Check(attempted.load(),"bridge did not attempt admission");
    Check(cb.status==MTLCommandBufferStatusNotEnqueued,"bridge submitted the caller buffer");
    b=[cb blitCommandEncoder];
    [b copyFromTexture:game sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0) sourceSize:MTLSizeMake(1920,game_height,1)
       toBuffer:result destinationOffset:0 destinationBytesPerRow:1920*4 destinationBytesPerImage:bytes]; [b endEncoding];
    [cb commit]; [cb waitUntilCompleted];
    Check(cb.status==MTLCommandBufferStatusCompleted,cb.error ? cb.error.description.UTF8String : "GPU failure");
    Check(!external || std::memcmp(external.contents,producer_readback.contents,bytes)==0,"external input upload/readback mismatch");
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(10);
    while(!nr_completed.load() && std::chrono::steady_clock::now()<deadline) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    Check(nr_completed.load()==1 && nr_published.load()==uint64_t(!source_fault),"unexpected publication outcome");
    Check(source_fault ? (nr_rejection_bits.load()&0x10u)!=0 : nr_rejection_bits.load()==0,"unexpected color source rejection flag");
    if(!capture_root.empty()) {
      const auto write_deadline=std::chrono::steady_clock::now()+std::chrono::seconds(20);
      while(!captures_finished.load() && std::chrono::steady_clock::now()<write_deadline)std::this_thread::sleep_for(std::chrono::milliseconds(1));
      Check(captures_succeeded.load()==1,"post-completion frame capture did not complete");
    }
    size_t changed=0;
    for(NSUInteger i=0;i<bytes;++i) changed+=((const uint8_t*)result.contents)[i]!=((const uint8_t*)producer_readback.contents)[i];
    Check(source_fault ? changed==0 : changed>0,source_fault ? "invalid source did not preserve whole original frame" : "identity output cannot establish a model transformation in this test");
    std::ofstream out(argv[4],std::ios::binary); out.write((const char*)result.contents,bytes); out.close(); Check(bool(out),"output write");
    std::ofstream source_out(std::string(argv[4])+".producer",std::ios::binary);
    source_out.write((const char*)producer_readback.contents,bytes); source_out.close(); Check(bool(source_out),"producer capture write");
    if(source_fault) {
      auto cd=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float width:1920 height:1080 mipmapped:NO];
      cd.storageMode=MTLStorageModeShared;cd.usage=MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite;
      auto next_color=[device newTextureWithDescriptor:cd];cd.pixelFormat=MTLPixelFormatRG16Float;
      auto next_motion=[device newTextureWithDescriptor:cd];Check(next_color!=nil && next_motion!=nil,"reset probe resources");
      nr::FrameInputs next;next.color=next_color;next.motion=next_motion;next.stream_id=7;next.frame_id=global_frame+1;next.reset=false;
      next.motion_units=nr::MotionUnits::PixelsFullRes;next.motion_direction=nr::MotionDirection::CurrentToPrevious;next.motion_y_down=true;
      auto next_cb=[queue commandBuffer];nr::TicketPtr next_ticket;nr::Report next_report;
      auto st=nr::encode(context,next,next_cb,next_ticket,next_report);
      Check(st==nr::EncodeStatus::RejectedBeforeEncoding && !next_ticket && next_cb.status==MTLCommandBufferStatusNotEnqueued && next_report.error.find("reset")!=std::string::npos,"rejected publication did not require history reset");
      std::printf("source_fault_fallback_bytes_equal=1 next_temporal_requires_reset=1 source_fault_test=1\n");
    }
    std::printf("result=pass same_command_producer_model_consumer=1 completed=%llu published=%llu bytes=%lu poison=%02lx global_frame=%llu changed_bytes=%zu game_connected=0 no_parity_or_speed_claim=1\n",
      (unsigned long long)nr_completed.load(),(unsigned long long)nr_published.load(),(unsigned long)bytes,poison,(unsigned long long)global_frame,changed);
    return 0;
  } catch(const std::exception& e) { std::fprintf(stderr,"result=FAIL detail=%s\n",e.what()); return 1; }
} }
