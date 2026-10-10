#pragma once
// Opt-in ONE-FRAME diagnostic. All data leaves execution only after completion;
// no file produced here is ever read by the inference path.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <CommonCrypto/CommonDigest.h>
#include <array>
#include <cstdio>
#include <memory>
#include <string>
#include "netal_nr_aspect.h"

namespace netal_nr_adapter {
struct CapturePlane {
  id<MTLBuffer> data=nil;
  const char* name="";
  uint32_t width=0,height=0,bpp=0;
  MTLPixelFormat format=MTLPixelFormatInvalid;
  bool copied=false;
};
struct FrameCapture {
  std::array<CapturePlane,4> planes;
  std::string root, model_identity, arithmetic;
  std::string color_mode="raw", color_reconstruction="identity";
  std::string execution_mode="inline-reset-once";
  uint64_t publication_frame=0, publication_command_id=0;
  AspectLayout layout;
  uint64_t frame=0,stream=0;
  uint64_t command_id=0,game_texture_id=0;
  bool reset=true;
};
inline std::shared_ptr<FrameCapture> MakeFrameCapture(id<MTLDevice> dev,const char* root,
    const AspectLayout& a,MTLPixelFormat game_format,uint64_t frame,uint64_t stream) {
  if(!root || root[0]!='/')return {};
  auto c=std::make_shared<FrameCapture>();c->root=root;c->layout=a;c->frame=frame;c->stream=stream;
  const uint32_t game_bpp=game_format==MTLPixelFormatRG11B10Float?4:8;
  c->planes={CapturePlane{nil,"game-input.bin",a.game_width,a.game_height,game_bpp,game_format,false},
             CapturePlane{nil,"model-input.rgba16f",a.model_width,a.model_height,8,MTLPixelFormatRGBA16Float,false},
             CapturePlane{nil,"model-output.rgba16f",a.model_width,a.model_height,8,MTLPixelFormatRGBA16Float,false},
             CapturePlane{nil,"game-output.bin",a.game_width,a.game_height,game_bpp,game_format,false}};
  for(auto& p:c->planes){p.data=[dev newBufferWithLength:size_t(p.width)*p.height*p.bpp
      options:MTLResourceStorageModeShared|MTLResourceHazardTrackingModeTracked];if(!p.data)return {};}
  return c;
}
inline void CopyCapture(id<MTLBlitCommandEncoder> b,const std::shared_ptr<FrameCapture>& c,
                        size_t plane,id<MTLTexture> texture) {
  if(!c)return;
  auto& p=c->planes.at(plane);
  [b copyFromTexture:texture sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0)
     sourceSize:MTLSizeMake(p.width,p.height,1) toBuffer:p.data destinationOffset:0
     destinationBytesPerRow:size_t(p.width)*p.bpp destinationBytesPerImage:size_t(p.width)*p.height*p.bpp];
  p.copied=true; // encoded, NOT yet CPU-ready
}
inline NSString* CaptureSHA(id<MTLBuffer> b){
  unsigned char d[CC_SHA256_DIGEST_LENGTH];CC_SHA256(b.contents,CC_LONG(b.length),d);
  char out[65];for(unsigned i=0;i<32;++i)std::snprintf(out+2*i,3,"%02x",d[i]);
  return [NSString stringWithUTF8String:out];
}
// Synchronous writer called by a BACKGROUND block after genuine GPU completion.
// Manifest written last. Refuse prior directories/files; preserve partial failures.
inline bool WriteFrameCapture(const std::shared_ptr<FrameCapture>& c,bool gpu_ok,bool published,
                              double gpu_ms,NSString** error_out) {
  NSString* root=[NSString stringWithUTF8String:c->root.c_str()];
  NSString* dir=[root stringByAppendingPathComponent:[NSString stringWithFormat:@"frame-%llu-stream-%llu",
      (unsigned long long)c->frame,(unsigned long long)c->stream]];
  NSError* error=nil;auto fm=[NSFileManager defaultManager];
  if([fm fileExistsAtPath:dir]){if(error_out)*error_out=@"capture directory already exists";return false;}
  if(![fm createDirectoryAtPath:dir withIntermediateDirectories:YES attributes:nil error:&error]){
    if(error_out)*error_out=error.localizedDescription;return false;}
  NSMutableArray* files=[NSMutableArray array];
  if(gpu_ok)for(const auto& p:c->planes)if(p.copied){
    NSString* name=[NSString stringWithUTF8String:p.name];
    NSData* data=[NSData dataWithBytesNoCopy:p.data.contents length:p.data.length freeWhenDone:NO];
    if(![data writeToFile:[dir stringByAppendingPathComponent:name] options:NSDataWritingWithoutOverwriting error:&error]){
      if(error_out)*error_out=error.localizedDescription;return false;}
    [files addObject:@{@"file":name,@"width":@(p.width),@"height":@(p.height),@"bytes_per_pixel":@(p.bpp),
                      @"metal_pixel_format":@(p.format),@"bytes":@(p.data.length),@"sha256":CaptureSHA(p.data)}];
  }
  const auto& a=c->layout;
  NSDictionary* record=@{@"schema":@"netal.nr.live-frame-capture.v1",@"frame":@(c->frame),@"stream":@(c->stream),
    @"command_id":@(c->command_id),@"game_texture_id":@(c->game_texture_id),@"ids_are_process_local":@YES,
    @"execution_mode":[NSString stringWithUTF8String:c->execution_mode.c_str()],
    @"source_frame":@(c->frame),@"publication_frame":c->publication_command_id?@(c->publication_frame):(id)[NSNull null],
    @"publication_command_id":@(c->publication_command_id),
    @"gpu_completed":@(gpu_ok),@"nr_export_completed":@(published),@"reset_only":@(c->reset),
    @"game_extent":@[@(a.game_width),@(a.game_height)],@"model_extent":@[@(a.model_width),@(a.model_height)],
    @"model_content":@[@(a.x),@(a.y),@(a.width),@(a.height)],@"files":files,
    @"model_identity":[NSString stringWithUTF8String:c->model_identity.c_str()],
    @"color_mode":[NSString stringWithUTF8String:c->color_mode.c_str()],
    @"color_reconstruction":[NSString stringWithUTF8String:c->color_reconstruction.c_str()],
    @"arithmetic":[NSString stringWithUTF8String:c->arithmetic.c_str()],@"whole_command_gpu_ms":@(gpu_ms),
    @"presentation_verified":@NO,@"exact_parity_claim":@NO,
    @"note":@"Captured execution outputs, not screenshots. This live bridge never reads capture files. Separately labelled offline replays start from external color, never expected outputs or internal/history tensors. Manifest written after completion; presentation association remains separate."};
  NSData* json=[NSJSONSerialization dataWithJSONObject:record options:NSJSONWritingPrettyPrinted|NSJSONWritingSortedKeys error:&error];
  if(!json || ![json writeToFile:[dir stringByAppendingPathComponent:@"capture.json"] options:NSDataWritingWithoutOverwriting error:&error]){
    if(error_out)*error_out=error.localizedDescription;return false;}
  return true;
}
} // namespace netal_nr_adapter
