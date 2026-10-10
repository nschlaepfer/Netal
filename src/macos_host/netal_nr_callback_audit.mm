// Independent API misuse/recovery audit. Deliberate wrong/premature completion
// must reject WITHOUT consuming the genuine pending ticket or releasing scratch.
// This standalone test owns its command queue and waits; no game is running.
#include "../nr_experimental/nr_experimental_api.h"
#import <Foundation/Foundation.h>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
namespace nr=netal_nr_experimental;
static void Need(bool ok,const char* message) { if(!ok)throw std::runtime_error(message); }
int main(int argc,char** argv) { @autoreleasepool {
  try {
    Need(argc==3,"usage: callback-audit BUNDLE wrong-command|premature");
    const bool wrong=!std::strcmp(argv[2],"wrong-command");
    Need(wrong || !std::strcmp(argv[2],"premature"),"unknown case");
    auto device=MTLCreateSystemDefaultDevice(); auto queue=[device newCommandQueue]; Need(queue!=nil,"queue");
    nr::ContextPtr ctx; nr::Report r; nr::PrepareOptions o; o.runtime_bundle=argv[1];
    Need(nr::prepare(device,o,ctx,r),r.error.c_str());
    auto d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float width:1920 height:1080 mipmapped:NO];
    d.storageMode=MTLStorageModeShared; d.usage=MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite;
    auto color=[device newTextureWithDescriptor:d]; Need(color!=nil,"color");
    std::vector<uint16_t> values(1920*1080*4,0x3800); // generated finite half0.5, no archived input
    [color replaceRegion:MTLRegionMake2D(0,0,1920,1080) mipmapLevel:0 withBytes:values.data() bytesPerRow:1920*8];
    auto genuine=[queue commandBuffer]; auto unrelated=[queue commandBuffer];
    nr::FrameInputs input;input.color=color;input.reset=true;input.stream_id=1;input.frame_id=57;
    nr::TicketPtr ticket; const auto status=nr::encode(ctx,input,genuine,ticket,r);
    Need(status==nr::EncodeStatus::Encoded && ticket && r.model_encoded,"model admission failed");
    Need(nr::output_texture(ticket)!=nil,"output unavailable before commit");
    // No second encode or scratch reuse is attempted while the genuine work is
    // pending. The audit only tests that correct later finalization remains possible.
    nr::Report misuse;
    nr::complete(ticket,wrong?unrelated:genuine,misuse);
    std::printf("case=%s misuse_rejected=%d error=%s genuine_status=%ld\n",argv[2],!misuse.error.empty(),misuse.error.c_str(),(long)genuine.status);
    Need(!misuse.error.empty(),"misuse was not rejected");
    [genuine commit];[genuine waitUntilCompleted];
    Need(genuine.status==MTLCommandBufferStatusCompleted,genuine.error?genuine.error.description.UTF8String:"GPU failed");
    nr::Report recovery;nr::complete(ticket,genuine,recovery);
    const bool ok=recovery.error.empty() && recovery.model_encoded;
    std::printf("case=%s genuine_gpu_completed=1 genuine_finalization_succeeded=%d history_advanced=%d error=%s\n",
      argv[2],ok,recovery.history_advanced,recovery.error.c_str());
    std::printf("result=%s game_connected=0 no_parity_or_performance_claim=1\n",ok?"PASS":"COUNTEREXAMPLE");
    return ok?0:1;
  } catch(const std::exception& e) {std::fprintf(stderr,"result=AUDIT_ERROR detail=%s\n",e.what());return 2;}
} }
