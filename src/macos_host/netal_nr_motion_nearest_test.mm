// Standalone GPU primitive test. Synthetic inputs only; no NR model or game.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <CommonCrypto/CommonDigest.h>
#include "netal_nr_motion_geometry.h"
#include <cstring>
#include <stdexcept>
#include <vector>
#include <unistd.h>
using namespace netal_nr_adapter;
struct alignas(16) MotionParams { AspectLayout aspect; uint32_t input_width,input_height; float scale_x,scale_y; };
static_assert(sizeof(MotionParams)==48 && offsetof(MotionParams,input_width)==32 && offsetof(MotionParams,scale_x)==40);
static void need(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
static uint16_t half_bits(float v){__fp16 h=v;uint16_t b;std::memcpy(&b,&h,2);return b;}
static float half_value(uint16_t b){__fp16 h;std::memcpy(&h,&b,2);return float(h);}
static NSString* hash_bytes(const void* p,size_t n){
  unsigned char h[CC_SHA256_DIGEST_LENGTH];CC_SHA256(p,CC_LONG(n),h);
  NSMutableString* s=[NSMutableString string];for(auto c:h)[s appendFormat:@"%02x",c];return s;
}
static void save(NSData* data,NSString* path){NSError* e=nil;need([data writeToFile:path options:NSDataWritingWithoutOverwriting error:&e],"preserve/write artifact");}
int main(int argc,char** argv){@autoreleasepool{
  try {
    need(argc==4,"usage: motion-nearest-test LIBRARY POISON NEW_DIRECTORY");
    NSString* root=@(argv[3]);need(![[NSFileManager defaultManager] fileExistsAtPath:root],"preserve previous run");
    need([[NSFileManager defaultManager] createDirectoryAtPath:root withIntermediateDirectories:NO attributes:nil error:nil],"create output directory");
    char* end=nullptr;const long poison=strtol(argv[2],&end,16);need(end && !*end && (poison==0xa5 || poison==0x5a),"a5/5a poison required");
    auto device=MTLCreateSystemDefaultDevice();need(device!=nil,"Metal device unavailable");NSError* err=nil;
    auto library=[device newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&err];need(library!=nil,"load motion library");
    auto pipeline=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"netal_nr_motion_nearest_candidate"] error:&err];need(pipeline!=nil,"motion pipeline");
    auto queue=[device newCommandQueue];need(queue!=nil,"command queue");NSMutableArray* cases=[NSMutableArray array];
    for(NSString* name in @[@"pattern",@"identity",@"signed-scale",@"source-nan",@"overflow",@"bad-metadata"]){@autoreleasepool{
      const bool identity=[name isEqualToString:@"identity"];
      MotionGeometryInput m;m.game_width=1920;m.game_height=identity?1080:1200;
      m.color_width=m.motion_width=identity?1920:1152;m.color_height=m.motion_height=identity?1080:720;
      m.motion_at_output_resolution=m.motion_includes_jitter=0;
      m.color_x=m.color_y=m.motion_x=m.motion_y=m.output_x=m.output_y=0;
      if([name isEqualToString:@"signed-scale"]){m.scale_x=-1;m.scale_y=.5;}
      MotionGeometry plan;need(PlanInputPixelMotion(m,plan)==MotionGeometryError::None,"CPU geometry rejected");
      MotionParams a{plan.aspect,plan.input_width,plan.input_height,float(plan.vector_scale_x),float(plan.vector_scale_y)};
      std::vector<uint16_t> input(size_t(m.color_width)*m.color_height*2);
      for(uint32_t y=0;y<m.color_height;++y)for(uint32_t x=0;x<m.color_width;++x){
        const size_t i=(size_t(y)*m.color_width+x)*2;
        input[i]=half_bits(float(int(x%31)-15)/4);input[i+1]=half_bits(float(int(y%23)-11)/8);
      }
      const size_t fault=(size_t(19)*m.color_width+17)*2;
      if([name isEqualToString:@"source-nan"])input[fault]=0x7e00;
      if([name isEqualToString:@"overflow"])input[fault]=half_bits(60000);
      const bool bad_metadata=[name isEqualToString:@"bad-metadata"];
      if(bad_metadata)--a.input_width;
      auto d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRG16Float width:m.color_width height:m.color_height mipmapped:NO];
      d.storageMode=MTLStorageModeShared;d.hazardTrackingMode=MTLHazardTrackingModeTracked;d.usage=MTLTextureUsageShaderRead;
      auto source=[device newTextureWithDescriptor:d];need(source!=nil,"source texture");
      [source replaceRegion:MTLRegionMake2D(0,0,m.color_width,m.color_height) mipmapLevel:0 withBytes:input.data() bytesPerRow:m.color_width*4];
      d.width=1920;d.height=1080;d.usage=MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite;
      auto output=[device newTextureWithDescriptor:d];need(output!=nil,"output texture");
      constexpr size_t pixels=1920u*1080u;std::vector<uint16_t> actual(pixels*2);std::memset(actual.data(),int(poison),actual.size()*2);
      [output replaceRegion:MTLRegionMake2D(0,0,1920,1080) mipmapLevel:0 withBytes:actual.data() bytesPerRow:1920*4];
      auto flags=[device newBufferWithLength:64 options:MTLResourceStorageModeShared];
      auto coverage=[device newBufferWithLength:(pixels+16)*4 options:MTLResourceStorageModeShared];need(flags && coverage,"audit buffers");
      auto f=static_cast<uint32_t*>(flags.contents);auto c=static_cast<uint32_t*>(coverage.contents);
      constexpr uint32_t canary=0x9ac15e37;for(unsigned i=0;i<16;++i)f[i]=canary;f[0]=0;
      std::memset(c,0,pixels*4);for(unsigned i=0;i<16;++i)c[pixels+i]=canary;
      auto cb=[queue commandBuffer];auto encoder=[cb computeCommandEncoder];need(encoder!=nil,"compute encoder");
      [encoder setComputePipelineState:pipeline];[encoder setTexture:source atIndex:0];[encoder setTexture:output atIndex:1];
      [encoder setBuffer:flags offset:0 atIndex:0];[encoder setBytes:&a length:sizeof(a) atIndex:1];[encoder setBuffer:coverage offset:0 atIndex:2];
      // Oversized grid tests the bounds check as well as exactly-once coverage.
      [encoder dispatchThreads:MTLSizeMake(1923,1082,1) threadsPerThreadgroup:MTLSizeMake(8,8,1)];[encoder endEncoding];
      [cb commit];[cb waitUntilCompleted];need(cb.status==MTLCommandBufferStatusCompleted,"motion GPU error");
      [output getBytes:actual.data() bytesPerRow:1920*4 fromRegion:MTLRegionMake2D(0,0,1920,1080) mipmapLevel:0];
      save([NSData dataWithBytes:actual.data() length:actual.size()*2],[root stringByAppendingPathComponent:[name stringByAppendingString:@".rg16f"]]);
      uint32_t expected_flags=bad_metadata?4:0;size_t differing=0;unsigned uncovered=0;
      for(uint32_t y=0;y<1080;++y)for(uint32_t x=0;x<1920;++x){
        const size_t i=size_t(y)*1920+x;uint16_t ex=0,ey=0;
        if(!bad_metadata && MotionLocation(plan,x,y).region==MotionRegion::Content){
          // Independent fixture oracle: 3 output centers select 2 input centers
          // with pattern [0,1,1], not a copy of the shader's multiply/divide formula.
          const auto sample=[](uint32_t p){return 2*(p/3)+(p%3!=0);};
          const uint32_t sx=identity?x:sample(x-plan.aspect.x);
          const uint32_t sy=identity?y:sample(y-plan.aspect.y);
          const size_t j=(size_t(sy)*m.color_width+sx)*2;const float rx=half_value(input[j]),ry=half_value(input[j+1]);
          if(!std::isfinite(rx)||!std::isfinite(ry))expected_flags|=1;
          else {const float vx=rx*a.scale_x,vy=ry*a.scale_y;
            if(!std::isfinite(vx)||!std::isfinite(vy)||std::abs(vx)>65504||std::abs(vy)>65504)expected_flags|=2;
            else {ex=half_bits(vx);ey=half_bits(vy);}}
        }
        differing+=(actual[2*i]!=ex)+(actual[2*i+1]!=ey);uncovered+=(c[i]!=1);
      }
      std::vector<uint16_t> unchanged(input.size());
      [source getBytes:unchanged.data() bytesPerRow:m.color_width*4 fromRegion:MTLRegionMake2D(0,0,m.color_width,m.color_height) mipmapLevel:0];
      bool guards=true;for(unsigned i=1;i<16;++i)guards&=f[i]==canary;for(unsigned i=0;i<16;++i)guards&=c[pixels+i]==canary;
      const bool pass=differing==0 && uncovered==0 && guards && f[0]==expected_flags && unchanged==input;
      [cases addObject:@{@"case":name,@"pass":@(pass),@"flags":@(f[0]),@"expected_flags":@(expected_flags),
        @"differing_half_components":@(differing),@"incorrect_coverage_pixels":@(uncovered),@"canaries_intact":@(guards),
        @"input_unchanged":@(unchanged==input),@"input_sha256":hash_bytes(input.data(),input.size()*2),
        @"output_sha256":hash_bytes(actual.data(),actual.size()*2),@"conversion_accepted":@(f[0]==0)}];
      printf("case=%s pass=%d flags=%u differences=%zu coverage_errors=%u\n",name.UTF8String,pass,f[0],differing,uncovered);
    }}
    bool pass=true;for(NSDictionary* c in cases)pass&=[c[@"pass"] boolValue];
    NSDictionary* receipt=@{@"schema":@"netal.nr.motion-nearest-gpu-test.v1",@"pass":@(pass),@"pid":@(getpid()),
      @"poison":@(argv[2]),@"cases":cases,@"game_connected":@NO,@"temporal_admission":@NO,
      @"scope":@"Synthetic nearest-sampling candidate and geometry test only. No NR, game motion, interpolation-policy selection, temporal quality, NVIDIA sampling parity or performance claim."};
    save([NSJSONSerialization dataWithJSONObject:receipt options:NSJSONWritingPrettyPrinted|NSJSONWritingSortedKeys error:nil],[root stringByAppendingPathComponent:@"receipt.json"]);
    return pass?0:1;
  }catch(const std::exception& e){fprintf(stderr,"FAIL %s\n",e.what());return 2;}
}}
