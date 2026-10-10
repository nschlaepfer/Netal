// Standalone GPU color-adapter audit. No NR, game, or expected-model bytes used.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

static void Check(bool ok,const char* why) { if(!ok)throw std::runtime_error(why); }
static float Value(uint16_t bits) { _Float16 x;std::memcpy(&x,&bits,2);return float(x); }
static uint16_t Half(double value) { _Float16 x=_Float16(value);uint16_t b;std::memcpy(&b,&x,2);return b; }
static double Forward(double x) {
  double y=x/(1+x);return y<=0.0031308?12.92*y:1.055*std::pow(y,1.0/2.4)-0.055;
}
static double Inverse(double q) {
  double y=q<=0.04045?q/12.92:std::pow((q+0.055)/1.055,2.4);return y/(1-y);
}
struct Result { std::vector<uint16_t> pixels;uint32_t rejected=0; };
struct Audit {
  id<MTLDevice> dev;id<MTLCommandQueue> queue;
  id<MTLComputePipelineState> forward,inverse;
  id<MTLTexture> texture(NSUInteger w,NSUInteger h) {
    auto d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float width:w height:h mipmapped:NO];
    d.storageMode=MTLStorageModeShared;d.hazardTrackingMode=MTLHazardTrackingModeTracked;
    d.usage=MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite;
    auto t=[dev newTextureWithDescriptor:d];Check(t!=nil,"texture allocation");return t;
  }
  Result run(const std::vector<uint16_t>& source,uint32_t w,uint32_t h,bool decode=false,uint32_t initial=0,uint32_t target_w=0) {
    Check(source.size()==size_t(w)*h*4,"input shape");if(!target_w)target_w=w;
    auto flag=[dev newBufferWithLength:4 options:MTLResourceStorageModeShared|MTLResourceHazardTrackingModeTracked];
    Check(flag!=nil,"flag allocation");std::memcpy(flag.contents,&initial,4);
    auto input=texture(w,h),output=texture(target_w,h);
    [input replaceRegion:MTLRegionMake2D(0,0,w,h) mipmapLevel:0 withBytes:source.data() bytesPerRow:w*8];
    Result r;r.pixels.assign(size_t(target_w)*h*4,0xa5a5);
    [output replaceRegion:MTLRegionMake2D(0,0,target_w,h) mipmapLevel:0 withBytes:r.pixels.data() bytesPerRow:target_w*8];
    auto buffer=[dev newBufferWithBytes:source.data() length:source.size()*2 options:MTLResourceStorageModeShared|MTLResourceHazardTrackingModeTracked];
    Check(buffer!=nil,"export copy allocation");
    auto cb=[queue commandBufferWithUnretainedReferences];Check(cb!=nil,"command allocation");
    auto c=[cb computeCommandEncoder];Check(c!=nil,"encoder allocation");
    if(decode) {
      uint32_t shape[2]={w,h};[c setComputePipelineState:inverse];
      [c setBuffer:buffer offset:0 atIndex:0];[c setBuffer:flag offset:0 atIndex:1];[c setBytes:shape length:8 atIndex:2];
    } else {
      [c setComputePipelineState:forward];[c setTexture:input atIndex:0];[c setTexture:output atIndex:1];[c setBuffer:flag offset:0 atIndex:0];
    }
    [c dispatchThreads:MTLSizeMake(w,h,1) threadsPerThreadgroup:MTLSizeMake(16,8,1)];[c endEncoding];
    [cb commit];[cb waitUntilCompleted];Check(cb.status==MTLCommandBufferStatusCompleted,"GPU command failure");
    if(decode) { r.pixels.resize(source.size());std::memcpy(r.pixels.data(),buffer.contents,source.size()*2); }
    else [output getBytes:r.pixels.data() bytesPerRow:target_w*8 fromRegion:MTLRegionMake2D(0,0,target_w,h) mipmapLevel:0];
    std::memcpy(&r.rejected,flag.contents,4);
    std::vector<uint16_t> unchanged(source.size());
    [input getBytes:unchanged.data() bytesPerRow:w*8 fromRegion:MTLRegionMake2D(0,0,w,h) mipmapLevel:0];
    Check(unchanged==source,"source texture changed");return r;
  }
};
static std::vector<uint16_t> Domain(uint32_t end) {
  std::vector<uint16_t> v(size_t(end)*4);
  for(uint32_t i=0;i<end;++i){for(unsigned j=0;j<3;++j)v[i*4+j]=uint16_t(i);v[i*4+3]=0x3c00;}return v;
}
static void Compare(const char* name,const std::vector<uint16_t>& input,const Result& r,bool inverse) {
  Check(r.rejected==0,"valid domain rejected");uint32_t max_ulp=0;size_t different=0;std::vector<uint16_t> reported;
  for(size_t i=0;i<input.size();++i) {
    if(i%4==3){Check(r.pixels[i]==input[i],"alpha changed");continue;}
    auto expected=Half(inverse?Inverse(Value(input[i])):Forward(Value(input[i])));
    Check(std::isfinite(Value(r.pixels[i])) && Value(r.pixels[i])>=0,"nonfinite or negative conversion");
    if(!inverse)Check(Value(r.pixels[i])<=1,"forward value above one");
    uint32_t delta=uint32_t(std::abs(int(r.pixels[i])-int(expected)));max_ulp=std::max(max_ulp,delta);different+=delta!=0;
    if(delta && reported.size()<16 && std::find(reported.begin(),reported.end(),input[i])==reported.end()) {
      reported.push_back(input[i]);
      std::printf("rounding_cell direction=%s input=%04x expected=%04x gpu=%04x\n",inverse?"inverse":"forward",unsigned(input[i]),unsigned(expected),unsigned(r.pixels[i]));
    }
  }
  std::printf("case=%s compared_components=%zu different_components=%zu max_half_ulp=%u exact=%d\n",name,input.size()/4*3,different,max_ulp,different==0);
  Check(max_ulp==0,"color conversion is not exactly rounded against CPU reference");
}
int main(int argc,char** argv) { @autoreleasepool {try {
  Check(argc==2 || argc==5,"usage: color-audit METALLIB [EXTERNAL_RGBA16F EXPECTED_BOUNDED_RGBA16F NEW_GPU_OUTPUT]");
  Audit a;a.dev=MTLCreateSystemDefaultDevice();Check(a.dev!=nil,"no Metal device");a.queue=[a.dev newCommandQueue];
  NSError* error=nil;auto lib=[a.dev newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&error];Check(lib!=nil,"load color library");
  a.forward=[a.dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"netal_nr_color_to_bounded"] error:&error];
  a.inverse=[a.dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"netal_nr_color_from_bounded"] error:&error];
  Check(a.forward!=nil && a.inverse!=nil,"color pipelines");
  auto f=Domain(0x7c00);Compare("forward_all_nonnegative_finite_half",f,a.run(f,256,124),false);
  auto b=Domain(0x3c00);Compare("inverse_all_half_below_one",b,a.run(b,256,60,true),true);
  for(uint16_t bad:{uint16_t(0xbc00),uint16_t(0x7c00),uint16_t(0x7e00)}) {
    auto r=a.run({bad,0,0,0x3c00},1,1);Check(r.rejected==0x10 && r.pixels==std::vector<uint16_t>({0,0,0,0x3c00}),"invalid forward handling");
  }
  for(uint16_t bad:{uint16_t(0x3c00),uint16_t(0xbc00),uint16_t(0x7c00),uint16_t(0x7e00)}) {
    auto r=a.run({bad,0,0,0x3c00},1,1,true);Check(r.rejected==0x20 && r.pixels==std::vector<uint16_t>({0,0,0,0x3c00}),"invalid inverse handling");
  }
  Check(a.run({0,0,0,0x3c00},1,1,false,0x80).rejected==0x80,"forward cleared prior rejection");
  Check(a.run({0,0,0,0x3c00},1,1,true,0x80).rejected==0x80,"inverse cleared prior rejection");
  Check(a.run({0x8000,0,0,0x3c00},1,1).pixels[0]==0x8000,"forward negative zero changed");
  Check(a.run({0x8000,0,0,0x3c00},1,1,true).pixels[0]==0x8000,"inverse negative zero changed");
  auto top=a.run({0x7bff,0,0,0x3c00},1,1);
  Check(top.pixels[0]==0x3c00 && a.run(top.pixels,1,1,true).rejected==0x20,"quantized singular inverse not rejected");
  auto shape=a.run({0,0,0,0x3c00,0,0,0,0x3c00},2,1,false,0,1);
  Check(shape.rejected==0x10 && shape.pixels==std::vector<uint16_t>(4,0xa5a5),"shape mismatch not rejected intact");
  if(argc==5) {
    NSData* raw=[NSData dataWithContentsOfFile:@(argv[2])];Check(raw.length==16588800,"external input size");
    std::vector<uint16_t> source(raw.length/2);std::memcpy(source.data(),raw.bytes,raw.length);
    auto r=a.run(source,1920,1080);Compare("external_game_color",source,r,false);
    NSData* expected=[NSData dataWithContentsOfFile:@(argv[3])];Check(expected.length==raw.length,"bounded reference size");
    size_t mismatch=0;const auto* e=static_cast<const uint16_t*>(expected.bytes);
    for(size_t i=0;i<r.pixels.size();++i)mismatch+=r.pixels[i]!=e[i];
    std::printf("external_gpu_vs_cpu_bounded_half_differences=%zu exact=%d\n",mismatch,mismatch==0);
    Check(mismatch==0,"GPU transform does not reproduce accepted bounded input");
    NSData* saved=[NSData dataWithBytes:r.pixels.data() length:r.pixels.size()*2];
    Check([saved writeToFile:@(argv[4]) options:NSDataWritingWithoutOverwriting error:&error],"preserve prior GPU output / write failure");
  }
  std::printf("result=pass model_executed=0 game_executed=0 color_contract_established=0\n");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"result=FAIL detail=%s\n",e.what());return 1;}}}
