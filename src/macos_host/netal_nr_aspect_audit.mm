// Aspect/format unit tests only. This harness owns its queue and CPU readbacks.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "netal_nr_aspect.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
using netal_nr_adapter::AspectLayout;
static void Need(bool b,const char* s){if(!b)throw std::runtime_error(s);}
static id<MTLComputePipelineState> Pipe(id<MTLDevice>d,id<MTLLibrary>l,NSString*n){NSError*e=nil;auto p=[d newComputePipelineStateWithFunction:[l newFunctionWithName:n] error:&e];Need(p!=nil,e?e.description.UTF8String:"pipeline");return p;}
static id<MTLTexture> Tex(id<MTLDevice>d,MTLPixelFormat f,unsigned w,unsigned h){auto t=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:f width:w height:h mipmapped:NO];t.storageMode=MTLStorageModePrivate;t.usage=MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite;t.hazardTrackingMode=MTLHazardTrackingModeTracked;auto r=[d newTextureWithDescriptor:t];Need(r!=nil,"texture");return r;}
static void Done(id<MTLCommandBuffer>b){[b commit];[b waitUntilCompleted];Need(b.status==MTLCommandBufferStatusCompleted,b.error?b.error.description.UTF8String:"GPU failure");}
static void Dispatch(id<MTLComputeCommandEncoder>c,unsigned w,unsigned h){[c dispatchThreads:MTLSizeMake(w,h,1) threadsPerThreadgroup:MTLSizeMake(8,8,1)];[c endEncoding];}
static void Run(id<MTLDevice>d,id<MTLCommandQueue>q,id<MTLComputePipelineState>fit,id<MTLComputePipelineState>publish,
                unsigned height,MTLPixelFormat format,int poison,bool gradient,bool reject){
  AspectLayout a;Need(netal_nr_adapter::MakeAspectLayout(1920,height,a),"layout");
  const size_t n=1920*height,mn=1920*1080,bpp=format==MTLPixelFormatRGBA16Float?8:4;
  std::vector<_Float16> pixels(n*4);
  for(unsigned y=0;y<height;++y)for(unsigned x=0;x<1920;++x){size_t i=(size_t(y)*1920+x)*4;
    pixels[i]=gradient?0.1f+0.8f*(x+0.5f)/1920:1;pixels[i+1]=gradient?0.1f+0.8f*(y+0.5f)/height:2;pixels[i+2]=gradient?0.25f:4;pixels[i+3]=1;}
  auto input=[d newBufferWithBytes:pixels.data() length:n*8 options:MTLResourceStorageModeShared];
  auto model=[d newBufferWithLength:mn*8 options:MTLResourceStorageModeShared];
  auto result=[d newBufferWithLength:n*bpp options:MTLResourceStorageModeShared];
  auto original=[d newBufferWithLength:n*bpp options:MTLResourceStorageModeShared];
  auto flag=[d newBufferWithLength:4 options:MTLResourceStorageModeShared];
  Need(input && model && result && original && flag,"buffer allocation");
  std::memset(model.contents,poison,mn*8);std::memset(original.contents,poison,n*bpp);std::memset(result.contents,poison^255,n*bpp);*(uint32_t*)flag.contents=reject?1:0;
  auto source=Tex(d,MTLPixelFormatRGBA16Float,1920,height),canvas=Tex(d,MTLPixelFormatRGBA16Float,1920,1080),target=Tex(d,format,1920,height);
  auto cb=[q commandBufferWithUnretainedReferences];auto b=[cb blitCommandEncoder];
  [b copyFromBuffer:input sourceOffset:0 sourceBytesPerRow:1920*8 sourceBytesPerImage:n*8 sourceSize:MTLSizeMake(1920,height,1) toTexture:source destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0,0,0)];
  [b copyFromBuffer:model sourceOffset:0 sourceBytesPerRow:1920*8 sourceBytesPerImage:mn*8 sourceSize:MTLSizeMake(1920,1080,1) toTexture:canvas destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0,0,0)];
  [b copyFromBuffer:original sourceOffset:0 sourceBytesPerRow:1920*bpp sourceBytesPerImage:n*bpp sourceSize:MTLSizeMake(1920,height,1) toTexture:target destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0,0,0)];[b endEncoding];
  auto c=[cb computeCommandEncoder];[c setComputePipelineState:fit];[c setTexture:source atIndex:0];[c setTexture:canvas atIndex:1];[c setBytes:&a length:sizeof a atIndex:0];Dispatch(c,1920,1080);
  b=[cb blitCommandEncoder];[b copyFromTexture:canvas sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0) sourceSize:MTLSizeMake(1920,1080,1) toBuffer:model destinationOffset:0 destinationBytesPerRow:1920*8 destinationBytesPerImage:mn*8];[b endEncoding];Done(cb);
  auto* data=(_Float16*)model.contents;double input_error=0;
  for(unsigned y=0;y<1080;++y)for(unsigned x=0;x<1920;++x){size_t i=(size_t(y)*1920+x)*4;
    bool inside=x>=a.x && x<a.x+a.width && y>=a.y && y<a.y+a.height;
    if(!inside){Need(data[i]==0 && data[i+1]==0 && data[i+2]==0 && data[i+3]==1,"padding coverage");
      // Valid but conspicuous padding must not bleed into the returned game image.
      data[i]=data[i+1]=data[i+2]=9;continue;}
    double r=gradient?0.1+0.8*(x-a.x+0.5)/a.width:1,g=gradient?0.1+0.8*(y-a.y+0.5)/a.height:2;
    input_error=std::max({input_error,std::abs(double(data[i])-r),std::abs(double(data[i+1])-g)});
    Need(data[i+2]==(gradient?_Float16(.25):_Float16(4)) && data[i+3]==1,"content channel/alpha");
  }
  Need(input_error<0.0015,"input sampling coordinates/ratio");
  cb=[q commandBufferWithUnretainedReferences];c=[cb computeCommandEncoder];[c setComputePipelineState:publish];
  [c setBuffer:model offset:0 atIndex:0];[c setBuffer:flag offset:0 atIndex:1];[c setBytes:&a length:sizeof a atIndex:2];[c setTexture:target atIndex:0];Dispatch(c,1920,height);
  b=[cb blitCommandEncoder];[b copyFromTexture:target sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0) sourceSize:MTLSizeMake(1920,height,1) toBuffer:result destinationOffset:0 destinationBytesPerRow:1920*bpp destinationBytesPerImage:n*bpp];[b endEncoding];Done(cb);
  double output_error=0;size_t mismatches=0;
  if(reject){auto*p=(uint8_t*)result.contents;for(size_t i=0;i<n*bpp;++i)mismatches+=p[i]!=uint8_t(poison);}
  else if(format==MTLPixelFormatRG11B10Float){Need(!gradient,"packed gradient not in this exact-format test");uint32_t expected=(15u<<6)|((16u<<6)<<11)|((17u<<5)<<22);auto*p=(uint32_t*)result.contents;for(size_t i=0;i<n;++i)mismatches+=p[i]!=expected;}
  else {auto*p=(_Float16*)result.contents;for(size_t i=0;i<n*4;++i){double e=std::abs(double(p[i])-double(pixels[i]));Need(std::isfinite(e),"finite output");output_error=std::max(output_error,e);}}
  Need(!mismatches && output_error<0.002,"publication crop/edge/padding/fallback mismatch");
  std::printf("PASS height=%u format=%lu poison=%02x gradient=%d reject=%d rect=%u,%u,%u,%u input_max_error=%.8g output_max_error=%.8g checked_pixels=%zu\n",height,(unsigned long)format,poison,gradient,reject,a.x,a.y,a.width,a.height,input_error,output_error,n);
}
int main(int argc,char**argv){@autoreleasepool{try{
  Need(argc==2,"usage: aspect-audit FORMATS.metallib");AspectLayout a;
  Need(netal_nr_adapter::MakeAspectLayout(1920,1200,a) && a.x==96 && a.y==0 && a.width==1728 && a.height==1080,"16:10 exact fit");
  Need(netal_nr_adapter::MakeAspectLayout(1920,1080,a) && a.identity() && !a.x && !a.y,"16:9 identity");
  Need(!netal_nr_adapter::MakeAspectLayout(0,1200,a),"zero rejected");
  Need(!netal_nr_adapter::SupportedGameExtent(3840,2160),"unsupported live extent not promoted");
  auto d=MTLCreateSystemDefaultDevice();auto q=[d newCommandQueue];NSError*e=nil;auto l=[d newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&e];Need(l!=nil,"library");
  auto fit=Pipe(d,l,@"netal_nr_fit_input"),pub=Pipe(d,l,@"netal_nr_fit_publish");
  for(int p:{0xa5,0x5a}){for(auto f:{MTLPixelFormatRGBA16Float,MTLPixelFormatRG11B10Float}){Run(d,q,fit,pub,1200,f,p,false,false);Run(d,q,fit,pub,1200,f,p,false,true);Run(d,q,fit,pub,1080,f,p,false,false);}Run(d,q,fit,pub,1200,MTLPixelFormatRGBA16Float,p,true,false);}
  std::printf("result=PASS cases=14 unretained=1 model_executed=0 game_connected=0\n");return 0;
}catch(const std::exception&e){std::fprintf(stderr,"result=FAIL error=%s\n",e.what());return 1;}}}
