#include "nr_encoder_runtime.h"
#include "generated/common.inc"
#include <map>
#include <mutex>
#include "generated/asset_hashes.h"

struct NrEncoderState {
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  std::map<std::string,id<MTLLibrary>> libraries;
  std::map<std::string,std::vector<uint8_t>> constants;
};
static const std::vector<uint8_t>& Constant(NrEncoderState& s,const fs::path& key) {
  return s.constants.at(key.generic_string());
}
struct PreOutputs { Buffer skip,encoder; std::string input_sha,feature_sha; };
struct EncoderOutputs { Buffer entry; std::array<Buffer,4> skips; };
static void TextureToBuffer(NrEncoderState& s,id<MTLTexture> tex,Buffer& b) {
  auto c=[s.queue commandBuffer];auto e=[c blitCommandEncoder];
  if(!c||!e)throw std::runtime_error("input texture copy encoder unavailable");
  [e copyFromTexture:tex sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0)
    sourceSize:MTLSizeMake(1920,1080,1) toBuffer:b.metal destinationOffset:Guard
    destinationBytesPerRow:1920*8 destinationBytesPerImage:1920*1080*8];
  [e endEncoding];[c commit];[c waitUntilCompleted];
  if(c.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error("input texture copy failed");
  if(Canary(b))throw std::runtime_error("input texture copy guard damage");
}
#include "generated/profile_timing.h"
namespace Pre {
#include "generated/pre.inc"
}
namespace Encoder {
#include "generated/encoder.inc"
}
static Buffer Layout(NrEncoderState& s,Buffer& input,uint8_t poison) {
  if(input.bytes!=17694720)throw std::runtime_error("Block0 layout input extent");
  auto before=Sha(Data(input),input.bytes);NSError* error=nil;
  auto p=[s.device newComputePipelineStateWithFunction:[s.libraries.at("entry.metallib") newFunctionWithName:@"nr_pre_entry_layout"] error:&error];
  if(!p)throw std::runtime_error("layout pipeline unavailable");
  Buffer out=Make(s.device,input.bytes,poison),counts=Make(s.device,input.bytes*4,0);
  auto cmd=[s.queue commandBuffer];auto e=[cmd computeCommandEncoder];[e setComputePipelineState:p];
  [e setBuffer:input.metal offset:Guard atIndex:0];[e setBuffer:out.metal offset:Guard atIndex:1];[e setBuffer:counts.metal offset:Guard atIndex:2];
  [e dispatchThreads:MTLSizeMake(input.bytes,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[e endEncoding];Pre::Complete(cmd);Pre::GuardOutput(out,counts);
  if(before!=Sha(Data(input),input.bytes)||Canary(input))throw std::runtime_error("layout input mutation");
  return out;
}
static NrEncoderRegion Region(Buffer b) {
  if(!b.metal||Canary(b))throw std::runtime_error("invalid returned allocation");
  return {b.metal,Guard,b.bytes,Sha(Data(b),b.bytes)};
}
static std::mutex encoder_execution_mutex;
int nr_encoder_prepare(NrEncoderContext& ctx,const char* root,std::string& error) {
  @autoreleasepool {try {
    if(!root||!*root||ctx.state)throw std::runtime_error("asset root required; context must be unprepared");
    if(!ctx.device)ctx.device=MTLCreateSystemDefaultDevice();
    if(!ctx.queue)ctx.queue=[ctx.device newCommandQueue];
    if(!ctx.device||!ctx.queue||ctx.queue.device!=ctx.device)throw std::runtime_error("device/queue mismatch");
    auto s=std::make_shared<NrEncoderState>();s->device=ctx.device;s->queue=ctx.queue;
    for(const auto& asset:NrEncoderAssetHashes) {
      fs::path p=fs::path(root)/asset.first;auto bytes=Read(p);
      if(Sha(bytes.data(),bytes.size())!=asset.second)throw std::runtime_error("pinned asset mismatch: "+p.string());
      if(p.extension()==".metallib") {
        NSError* e=nil;auto lib=[ctx.device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:p.c_str()]] error:&e];
        if(!lib)throw std::runtime_error("library load failed: "+p.string());s->libraries[asset.first]=lib;
      }else s->constants[asset.first]=std::move(bytes);
    }
    ctx.state=s;error.clear();return 0;
  }catch(const std::exception& e){error=e.what();return 2;}}
}
int nr_encoder_run_frame(NrEncoderContext& ctx,const NrEncoderFrame& in,uint8_t poison,NrEncoderResult& result,const char* diagnostics) {
  result=NrEncoderResult{};
  @autoreleasepool {try {
    std::unique_lock<std::mutex> lease(encoder_execution_mutex,std::try_to_lock);
    if(!lease.owns_lock())throw std::runtime_error("concurrent encoder invocation unsupported");
    if(!ctx.state||ctx.state->device!=ctx.device||ctx.state->queue!=ctx.queue)throw std::runtime_error("unprepared or changed context");
    if(!in.reset||in.frame_index!=0)throw std::runtime_error("unverified temporal/frame contract");
    if(!in.color||in.color.device!=ctx.device||in.color.textureType!=MTLTextureType2D||in.color.pixelFormat!=MTLPixelFormatRGBA16Float||in.color.width!=1920||in.color.height!=1080||in.color.mipmapLevelCount!=1||in.color.sampleCount!=1)throw std::runtime_error("input texture must be same-device 1920x1080 RGBA16F 2D");
    if(in.producer_completion) {
      auto p=in.producer_completion;
      if(p.commandQueue.device!=ctx.device||p.status<MTLCommandBufferStatusCommitted)throw std::runtime_error("producer command is not committed or wrong-device");
      [p waitUntilCompleted];if(p.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error("producer command failed");
    }
    fs::path out=diagnostics?diagnostics:"";
    if(diagnostics&&(out.empty()||fs::exists(out)))throw std::runtime_error("diagnostics must name a fresh directory");
    NrDiagnosticMode mode(diagnostics!=nullptr);NrCheckedValues=0;NrCheckedOutputs=0;nr_timings.clear();nr_phases.clear();
    auto pre=Pre::Execute(*ctx.state,in.color,poison,out/"block0");
    Buffer entry=Layout(*ctx.state,pre.encoder,poison);
    auto enc=Encoder::Execute(*ctx.state,entry,poison,out/"encoder");
    Buffer after=Make(ctx.device,1920*1080*8,0);TextureToBuffer(*ctx.state,in.color,after);
    if(Sha(Data(after),after.bytes)!=pre.input_sha)throw std::runtime_error("input texture changed during encoder");
    NrEncoderResult accepted;accepted.entry=Region(enc.entry);accepted.block0_skip=Region(pre.skip);
    for(size_t i=0;i<4;++i)accepted.skips[i]=Region(enc.skips[i]);
    accepted.input_sha256=pre.input_sha;accepted.feature_sha256=pre.feature_sha;accepted.bundle_root=NrEncoderParentRoot;
    accepted.guarded_outputs_checked=NrCheckedOutputs;accepted.coverage_values_checked=NrCheckedValues;
    accepted.input_immutable=true;accepted.gpu_complete=true;accepted.code=0;result=std::move(accepted);return 0;
  }catch(const std::exception& e){result.error=e.what();return 2;}}
}
