// Standalone adapter-only synthetic test. Never loads weights/model captures.
// This harness owns its queue and waits; the live bridge must NEVER do so.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

static void Need(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
static id<MTLComputePipelineState> Pipeline(id<MTLDevice> d, id<MTLLibrary> lib, NSString* name) {
  NSError* e = nil;
  id<MTLFunction> f = [lib newFunctionWithName:name]; Need(f != nil, name.UTF8String);
  auto p = [d newComputePipelineStateWithFunction:f error:&e];
  Need(p != nil, e ? e.description.UTF8String : "pipeline"); return p;
}
static id<MTLTexture> Texture(id<MTLDevice> d, MTLPixelFormat f, uint32_t w, uint32_t h) {
  auto td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:f width:w height:h mipmapped:NO];
  td.storageMode = MTLStorageModePrivate; td.hazardTrackingMode = MTLHazardTrackingModeTracked;
  td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
  auto t = [d newTextureWithDescriptor:td]; Need(t != nil,"texture allocation"); return t;
}
static void Finish(id<MTLCommandBuffer> cb) {
  [cb commit]; [cb waitUntilCompleted];
  Need(cb.status == MTLCommandBufferStatusCompleted, cb.error ? cb.error.description.UTF8String : "command failed");
}
static void Dispatch(id<MTLComputeCommandEncoder> c, uint32_t w, uint32_t h) {
  [c dispatchThreads:MTLSizeMake(w,h,1) threadsPerThreadgroup:MTLSizeMake(8,8,1)]; [c endEncoding];
}

static void ImportCensus(id<MTLDevice> d, id<MTLCommandQueue> q, id<MTLLibrary> lib) {
  const uint32_t w=64,h=32, n=w*h, dims[2]={w,h};
  std::vector<uint32_t> raw(n);
  std::vector<uint16_t> expected(n*4);
  for (uint32_t i=0;i<n;++i) {
    // Exhaust all finite encodings of each unsigned packed channel, including
    // zero, subnormals and maximum finite. IEEE exponent bias equals half's15.
    uint32_t r=i%1984, g=(1983-i%1984), b=i%992;
    raw[i]=r|(g<<11)|(b<<22);
    expected[i*4]=r<<4; expected[i*4+1]=g<<4; expected[i*4+2]=b<<5; expected[i*4+3]=0x3c00;
  }
  auto source=[d newBufferWithBytes:raw.data() length:raw.size()*4 options:MTLResourceStorageModeShared];
  auto decoded=[d newBufferWithLength:n*8 options:MTLResourceStorageModeShared];
  auto texture=Texture(d,MTLPixelFormatRG11B10Float,w,h);
  auto p=Pipeline(d,lib,@"neuralstudio_sr_import_rg11b10f");
  auto cb=[q commandBuffer]; auto blit=[cb blitCommandEncoder];
  [blit copyFromBuffer:source sourceOffset:0 sourceBytesPerRow:w*4 sourceBytesPerImage:n*4 sourceSize:MTLSizeMake(w,h,1)
     toTexture:texture destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0,0,0)]; [blit endEncoding];
  auto c=[cb computeCommandEncoder]; [c setComputePipelineState:p]; [c setTexture:texture atIndex:0];
  [c setBuffer:decoded offset:0 atIndex:0]; [c setBytes:dims length:sizeof dims atIndex:1]; Dispatch(c,w,h); Finish(cb);
  const auto* actual=(const uint16_t*)decoded.contents;
  size_t mismatch=0; for(size_t i=0;i<expected.size();++i) if(actual[i]!=expected[i]) {
    if(mismatch<4) std::printf("import_counterexample index=%zu actual=%04x expected=%04x\n",i,actual[i],expected[i]); ++mismatch;
  }
  std::printf("test=packed_finite_import_census pixels=%u mismatches=%zu\n",n,mismatch);
  Need(mismatch==0,"packed import does not match CPU format semantics");
}

static void PublicationCase(id<MTLDevice> d, id<MTLCommandQueue> q,
    id<MTLComputePipelineState> scan,id<MTLComputePipelineState> publish,
    MTLPixelFormat format,uint32_t w,uint32_t h,int poison,int channel,uint16_t exceptional,size_t location,
    id<MTLComputePipelineState> inverse=nil,uint32_t sticky_flag=0) {
  const size_t n=size_t(w)*h, bpp=format==MTLPixelFormatRGBA16Float?8:4;
  const uint32_t dims[4]={w,h,uint32_t(format==MTLPixelFormatRG11B10Float),0};
  std::vector<uint8_t> original(n*bpp,poison);
  std::vector<uint16_t> candidate(n*4);
  for(size_t i=0;i<n;++i) {
    candidate[i*4]=0x3c00; candidate[i*4+1]=0x4000; candidate[i*4+2]=0x4400; candidate[i*4+3]=0x3c00;
    if(inverse)candidate[i*4]=candidate[i*4+1]=candidate[i*4+2]=0x3800;
  }
  const bool nonfinite=channel>=0 && (exceptional&0x7c00)==0x7c00;
  const bool overflow=channel>=0 && channel<3 && bpp==4 && !(exceptional&0x8000) &&
    exceptional>(channel==2?0x7be0:0x7bf0);
  // Inverse cases deliberately contain one invalid model-space RGB value.
  if(inverse)Need(channel>=0 && channel<3 && exceptional>=0x3c00,"inverse rejection fixture invalid");
  const uint32_t expected_flag=sticky_flag | (inverse?0x20u:(uint32_t(nonfinite)|(uint32_t(overflow)<<1)));
  const bool reject=expected_flag!=0;
  if(channel>=0) candidate[(location%n)*4+channel]=exceptional;
  const auto tracked=MTLResourceStorageModeShared|MTLResourceHazardTrackingModeTracked;
  auto orig=[d newBufferWithBytes:original.data() length:original.size() options:tracked];
  auto live=[d newBufferWithBytes:candidate.data() length:candidate.size()*2 options:tracked];
  auto readback=[d newBufferWithLength:original.size() options:tracked];
  auto flag=[d newBufferWithLength:4 options:tracked];
  auto seed=[d newBufferWithBytes:&sticky_flag length:4 options:tracked];
  std::memset(flag.contents,poison,4); std::memset(readback.contents,~poison,original.size());
  auto game=Texture(d,format,w,h), staged=Texture(d,format,w,h);
  auto cb=[q commandBufferWithUnretainedReferences];
  NSMutableArray* keepalive=[NSMutableArray arrayWithObjects:orig,live,readback,flag,seed,game,staged,scan,publish,nil];
  if(inverse)[keepalive addObject:inverse];
  [cb addCompletedHandler:^(id<MTLCommandBuffer>) { (void)keepalive.count; }];
  auto b=[cb blitCommandEncoder];
  [b copyFromBuffer:orig sourceOffset:0 sourceBytesPerRow:w*bpp sourceBytesPerImage:n*bpp sourceSize:MTLSizeMake(w,h,1)
     toTexture:game destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0,0,0)];
  [b copyFromTexture:game sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0) sourceSize:MTLSizeMake(w,h,1)
     toTexture:staged destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0,0,0)];
  [b fillBuffer:flag range:NSMakeRange(0,4) value:0];
  if(sticky_flag)[b copyFromBuffer:seed sourceOffset:0 toBuffer:flag destinationOffset:0 size:4];
  [b endEncoding];
  if(inverse) {
    auto c=[cb computeCommandEncoder];[c setComputePipelineState:inverse];
    [c setBuffer:live offset:0 atIndex:0];[c setBuffer:flag offset:0 atIndex:1];
    [c setBytes:dims length:8 atIndex:2];Dispatch(c,w,h);
  }
  const id<MTLComputePipelineState> passes[]={scan,publish};
  for(int pass=0;pass<2;++pass) {
    auto c=[cb computeCommandEncoder]; [c setComputePipelineState:passes[pass]];
    [c setBuffer:live offset:0 atIndex:0]; [c setBuffer:flag offset:0 atIndex:1];
    [c setBytes:dims length:sizeof dims atIndex:2]; if(pass==1) [c setTexture:staged atIndex:0]; Dispatch(c,w,h);
  }
  b=[cb blitCommandEncoder];
  [b copyFromTexture:staged sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0) sourceSize:MTLSizeMake(w,h,1)
     toTexture:game destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0,0,0)];
  [b copyFromTexture:game sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0) sourceSize:MTLSizeMake(w,h,1)
     toBuffer:readback destinationOffset:0 destinationBytesPerRow:w*bpp destinationBytesPerImage:n*bpp]; [b endEncoding];
  Finish(cb);
  Need(*(uint32_t*)flag.contents==expected_flag,"publication rejection flag mismatch");
  std::vector<uint8_t> expected;
  if(reject) expected=original;
  else if(bpp==8) expected.assign((uint8_t*)candidate.data(),(uint8_t*)candidate.data()+candidate.size()*2);
  else {
    const uint32_t rgb=(15u<<6)|((16u<<6)<<11)|((17u<<5)<<22);
    expected.resize(n*4); for(size_t i=0;i<n;++i) std::memcpy(expected.data()+i*4,&rgb,4);
  }
  const auto* actual=(const uint8_t*)readback.contents;
  size_t mismatches=0; for(size_t i=0;i<expected.size();++i) mismatches+=expected[i]!=actual[i];
  std::printf("test=publication format=%lu width=%u height=%u poison=%02x channel=%d exceptional=%04x location=%zu rejected=%d color_inverse=%d sticky=%u unretained=1 checked_bytes=%zu mismatches=%zu\n",
    (unsigned long)format,w,h,poison,channel,exceptional,location,reject,inverse!=nil,sticky_flag,expected.size(),mismatches);
  Need(mismatches==0,"whole-frame publication/fallback mismatch");
}
int main(int argc,char** argv) { @autoreleasepool {
  try {
    Need(argc==2,"usage: netal_nr_adapter_smoke FORMATS.metallib");
    auto d=MTLCreateSystemDefaultDevice(); Need(d!=nil,"no Metal device"); auto q=[d newCommandQueue];
    NSError* e=nil; auto lib=[d newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&e];
    Need(lib!=nil,e?e.description.UTF8String:"library");
    std::printf("adapter_only=1 model_executed=0 device=%s\n",d.name.UTF8String);
    ImportCensus(d,q,lib);
    auto scan=Pipeline(d,lib,@"netal_nr_scan_finite"), publish=Pipeline(d,lib,@"netal_nr_publish_finite");
    auto inverse=Pipeline(d,lib,@"netal_nr_color_from_bounded");
    size_t cases=0;
    for(auto format:{MTLPixelFormatRGBA16Float,MTLPixelFormatRG11B10Float}) for(int poison:{0xa5,0x5a}) {
      // Non-multiple grid tests the bounds guards; full1080p tests all pixels.
      PublicationCase(d,q,scan,publish,format,17,9,poison,-1,0,0); ++cases;
      PublicationCase(d,q,scan,publish,format,1920,1080,poison,-1,0,0); ++cases;
      for(int channel=0;channel<4;++channel) for(uint16_t fault:{uint16_t(0x7c00),uint16_t(0xfc00),uint16_t(0x7e01)}) {
        PublicationCase(d,q,scan,publish,format,17,9,poison,channel,fault,channel==0?0:channel==3?152:77); ++cases;
      }
      PublicationCase(d,q,scan,publish,format,1920,1080,poison,0,0x7e01,1920*1080-1); ++cases;
      // A finite half near65504 must not become infinity in a packed output.
      // RGBA16F keeps the exact finite value; packed output rejects the frame.
      for(int channel=0;channel<3;++channel) {
        PublicationCase(d,q,scan,publish,format,17,9,poison,channel,0x7bff,152); ++cases;
      }
      for(int channel=0;channel<3;++channel) for(uint16_t fault:{uint16_t(0x3c00),uint16_t(0xbc00),uint16_t(0x7c00),uint16_t(0x7e01)}) {
        PublicationCase(d,q,scan,publish,format,17,9,poison,channel,fault,channel==0?0:channel==2?152:77,inverse);++cases;
      }
      PublicationCase(d,q,scan,publish,format,1920,1200,poison,0,0x3c00,1920*1200-1,inverse);++cases;
      PublicationCase(d,q,scan,publish,format,17,9,poison,-1,0,0,nil,0x10);++cases;
    }
    std::printf("result=pass import_census=1 publication_cases=%zu no_model_or_game_claim=1\n",cases); return 0;
  } catch(const std::exception& e) { std::fprintf(stderr,"result=FAIL error=%s\n",e.what()); return 1; }
} }
