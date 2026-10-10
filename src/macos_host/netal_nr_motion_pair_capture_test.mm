#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cassert>
#include <cstdio>
#include <vector>
#include "netal_nr_motion_pair_capture.h"
using namespace netal_nr_adapter;
static NSDictionary* Metadata(){return @{
  @"color_offset_x":@0,@"color_offset_y":@0,@"motion_offset_x":@0,@"motion_offset_y":@0,
  @"output_offset_x":@0,@"output_offset_y":@0,@"motion_at_output_resolution":@0,@"motion_includes_jitter":@0,
  @"input_width":@1152,@"input_height":@720,@"output_width":@1920,@"output_height":@1200,
  @"motion_scale_x":@1,@"motion_scale_y":@1,@"jitter_x":@0,@"jitter_y":@0,@"pre_exposure":@1,@"reset":@NO};}
static id<MTLTexture> Texture(id<MTLDevice> dev,MTLPixelFormat format){
  auto d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:1152 height:720 mipmapped:NO];
  d.storageMode=MTLStorageModeShared;d.hazardTrackingMode=MTLHazardTrackingModeTracked;
  return [dev newTextureWithDescriptor:d];
}
int main(int argc,const char** argv){@autoreleasepool{
  assert(argc==2);NSString* root=[NSString stringWithUTF8String:argv[1]];
  assert(![NSFileManager.defaultManager fileExistsAtPath:root]);
  assert([NSFileManager.defaultManager createDirectoryAtPath:root withIntermediateDirectories:YES attributes:nil error:nil]);
  NSString* arm=[root stringByAppendingPathComponent:@"arm"];
  assert([[NSData data] writeToFile:arm options:NSDataWritingWithoutOverwriting error:nil]);
  auto dev=MTLCreateSystemDefaultDevice();assert(dev);auto q=[dev newCommandQueue];assert(q);
  static_assert(MTLPixelFormatRGBA16Float==115); // Verified observed input format, NOT packed scaler output.
  auto color=Texture(dev,MTLPixelFormatRGBA16Float),motion=Texture(dev,MTLPixelFormatRG16Float);
  assert(color && motion);
  const size_t bytes=1152*720*4;
  const size_t colorBytes=bytes*2;
  std::vector<uint32_t> colorData(colorBytes/4),motionData(bytes/4);
  for(size_t i=0;i<colorBytes/4;++i)colorData[i]=uint32_t(i*12345);
  for(size_t i=0;i<bytes/4;++i)motionData[i]=uint32_t(i*76543);
  [color replaceRegion:MTLRegionMake2D(0,0,1152,720) mipmapLevel:0 withBytes:colorData.data() bytesPerRow:1152*8];
  [motion replaceRegion:MTLRegionMake2D(0,0,1152,720) mipmapLevel:0 withBytes:motionData.data() bytesPerRow:1152*4];
  MotionPairInput in;in.color=color;in.motion=motion;in.frame=100;in.stream=6;in.command=1;in.metadata=Metadata();
  MotionPairCapture pair(root.UTF8String,arm.UTF8String);
  auto cb=[q commandBufferWithUnretainedReferences];assert(pair.encode(cb,in));
  [cb commit];[cb waitUntilCompleted];assert(cb.status==MTLCommandBufferStatusCompleted);
  in.frame++;in.command++;
  auto cb2=[q commandBufferWithUnretainedReferences];assert(pair.encode(cb2,in));
  [cb2 commit];[cb2 waitUntilCompleted];assert(cb2.status==MTLCommandBufferStatusCompleted);
  pair.drain_writer_for_test();
  for(unsigned i=0;i<2;++i){
    NSString* dir=[root stringByAppendingPathComponent:[NSString stringWithFormat:@"pair-%u-frame-%u",i,100+i]];
    NSData* c=[NSData dataWithContentsOfFile:[dir stringByAppendingPathComponent:@"color.rgba16f"]];
    NSData* m=[NSData dataWithContentsOfFile:[dir stringByAppendingPathComponent:@"motion.rg16f"]];
    assert(c.length==colorBytes && m.length==bytes);
    assert(!memcmp(c.bytes,colorData.data(),colorBytes));assert(!memcmp(m.bytes,motionData.data(),bytes));
    NSDictionary* r=[NSJSONSerialization JSONObjectWithData:[NSData dataWithContentsOfFile:[dir stringByAppendingPathComponent:@"capture.json"]] options:0 error:nil];
    assert([r[@"gpu_completed"] boolValue] && [r[@"encoded"] boolValue]);
    assert(![r[@"actual_nr"] boolValue] && ![r[@"temporal_admitted"] boolValue]);
  }
  std::vector<uint32_t> check(colorBytes/4);
  [color getBytes:check.data() bytesPerRow:1152*8 fromRegion:MTLRegionMake2D(0,0,1152,720) mipmapLevel:0];assert(check==colorData);
  check.resize(bytes/4);
  [motion getBytes:check.data() bytesPerRow:1152*4 fromRegion:MTLRegionMake2D(0,0,1152,720) mipmapLevel:0];assert(check==motionData);
  assert(!pair.encode([q commandBuffer],in)); // Never a third frame.
  unsigned refused=0;
  for(NSString* key in @[@"color_offset_x",@"color_offset_y",@"motion_offset_x",@"motion_offset_y",
      @"output_offset_x",@"output_offset_y",@"motion_at_output_resolution",@"motion_includes_jitter",
      @"input_width",@"input_height",@"output_width",@"output_height"]){
    MotionPairCapture negative(root.UTF8String,arm.UTF8String);NSMutableDictionary* bad=[Metadata() mutableCopy];bad[key]=@(-1);
    in.metadata=bad;assert(!negative.encode([q commandBuffer],in));++refused;
  }
  MotionPairCapture noarm(root.UTF8String,[[root stringByAppendingPathComponent:@"missing-arm"] UTF8String]);
  in.metadata=Metadata();assert(!noarm.encode([q commandBuffer],in));
  // Genuine first encoding then source gap: no silent resampling of a later pair.
  NSString* gaproot=[root stringByAppendingPathComponent:@"gap"];
  MotionPairCapture gap(gaproot.UTF8String,arm.UTF8String);in.frame=200;
  auto c3=[q commandBuffer];assert(gap.encode(c3,in));[c3 commit];[c3 waitUntilCompleted];
  in.frame=202;assert(!gap.encode([q commandBuffer],in));in.frame=201;assert(!gap.encode([q commandBuffer],in));
  gap.drain_writer_for_test();
  // Caller drops its texture owners BEFORE submitting the unretained buffer.
  NSString* leaseRoot=[root stringByAppendingPathComponent:@"lifetime"];
  MotionPairCapture leasePair(leaseRoot.UTF8String,arm.UTF8String);
  MotionPairInput lease;lease.frame=300;lease.command=20;lease.stream=6;lease.metadata=Metadata();
  lease.color=Texture(dev,MTLPixelFormatRGBA16Float);lease.motion=Texture(dev,MTLPixelFormatRG16Float);
  [lease.color replaceRegion:MTLRegionMake2D(0,0,1152,720) mipmapLevel:0 withBytes:colorData.data() bytesPerRow:1152*8];
  [lease.motion replaceRegion:MTLRegionMake2D(0,0,1152,720) mipmapLevel:0 withBytes:motionData.data() bytesPerRow:1152*4];
  auto c4=[q commandBufferWithUnretainedReferences];assert(leasePair.encode(c4,lease));
  lease.color=nil;lease.motion=nil;[c4 commit];[c4 waitUntilCompleted];assert(c4.status==MTLCommandBufferStatusCompleted);
  leasePair.drain_writer_for_test();
  NSData* held=[NSData dataWithContentsOfFile:[leaseRoot stringByAppendingPathComponent:@"pair-0-frame-300/color.rgba16f"]];
  assert(held.length==colorBytes && !memcmp(held.bytes,colorData.data(),colorBytes));
  printf("PASS pid=%d two consecutive copies, source unchanged, unretained CB dropped-owner lifetime, bounded admission, %u metadata refusals, source gap fail-closed; no game/model\n",getpid(),refused);
}}
