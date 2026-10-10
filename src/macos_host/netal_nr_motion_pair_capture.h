#pragma once
// Bounded two-frame, READ-ONLY capture. No model, conversion, or history import.
// Called before the original scaler; waits/updates its fence without submitting
// or waiting for the caller command buffer. All CPU reads follow completion.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unistd.h>
#include "netal_nr_frame_capture.h"
#include "../nr_experimental/nr_encoder_scope.h"

namespace netal_nr_adapter {
struct MotionPairInput {
  id<MTLTexture> color=nil, motion=nil;
  id<MTLFence> fence=nil;
  uint64_t frame=0, stream=0, command=0;
  // Caller takes immutable scalar metadata at the interception boundary.
  NSDictionary* metadata=nil;
};
struct MotionPairFrame {
  MotionPairInput input;
  std::array<CapturePlane,2> planes;
  bool encoded=false;
};
class MotionPairCapture {
  std::mutex mutex_;
  std::string root_, arm_;
  uint64_t first_=0, stream_=0, first_command_=0;
  unsigned selected_=0;
  bool stopped_=false;
  NSDictionary* geometry_=nil;
  dispatch_queue_t writer_=dispatch_queue_create("netal.nr.motion-pair-writer",DISPATCH_QUEUE_SERIAL);

  static bool TextureOK(id<MTLTexture> t,id<MTLDevice> dev,MTLPixelFormat format,id<MTLFence> fence) {
    return t && t.device==dev && t.textureType==MTLTextureType2D && t.sampleCount==1 &&
      t.width==1152 && t.height==720 && t.pixelFormat==format && !t.framebufferOnly &&
      t.storageMode!=MTLStorageModeMemoryless &&
      (t.hazardTrackingMode!=MTLHazardTrackingModeUntracked || fence);
  }
  static bool MetadataOK(NSDictionary* m) {
    if(!m)return false;
    const NSArray* zero=@[@"color_offset_x",@"color_offset_y",@"motion_offset_x",@"motion_offset_y",
      @"output_offset_x",@"output_offset_y",@"motion_at_output_resolution",@"motion_includes_jitter"];
    for(NSString* k in zero)if(![m[k] isKindOfClass:[NSNumber class]] || [m[k] doubleValue]!=0)return false;
    for(NSString* k in @[@"motion_scale_x",@"motion_scale_y",@"jitter_x",@"jitter_y",@"pre_exposure"])
      if(![m[k] isKindOfClass:[NSNumber class]] || !std::isfinite([m[k] doubleValue]))return false;
    return [m[@"input_width"] longLongValue]==1152 && [m[@"input_height"] longLongValue]==720 &&
      [m[@"output_width"] longLongValue]==1920 && [m[@"output_height"] longLongValue]==1200;
  }
  static NSDictionary* Geometry(NSDictionary* m) {
    NSMutableDictionary* g=[m mutableCopy];
    // Jitter/reset/exposure may vary and MUST remain recorded per frame.
    for(NSString* k in @[@"jitter_x",@"jitter_y",@"reset",@"pre_exposure"]) [g removeObjectForKey:k];
    return [g copy];
  }
  static void Write(const std::string& root,const std::shared_ptr<MotionPairFrame>& f,
                    unsigned index,bool gpu_ok,NSString* gpu_error) {
    @autoreleasepool {
      NSError* err=nil;
      NSString* dir=[[NSString stringWithUTF8String:root.c_str()] stringByAppendingPathComponent:
        [NSString stringWithFormat:@"pair-%u-frame-%llu",index,(unsigned long long)f->input.frame]];
      NSFileManager* fm=[NSFileManager defaultManager];
      if([fm fileExistsAtPath:dir] || ![fm createDirectoryAtPath:dir withIntermediateDirectories:YES attributes:nil error:&err]) {
        NSLog(@"netal_nr motion_pair_write_failed directory=%@ error=%@",dir,err);return;
      }
      NSMutableArray* files=[NSMutableArray array];
      // Failed GPU work is never read on the CPU or represented as valid data.
      if(gpu_ok && f->encoded)for(const auto& p:f->planes){
        NSString* name=[NSString stringWithUTF8String:p.name];
        NSData* data=[NSData dataWithBytesNoCopy:p.data.contents length:p.data.length freeWhenDone:NO];
        if(![data writeToFile:[dir stringByAppendingPathComponent:name] options:NSDataWritingWithoutOverwriting error:&err]){
          NSLog(@"netal_nr motion_pair_write_failed file=%@ error=%@",name,err);return;
        }
        [files addObject:@{@"file":name,@"sha256":CaptureSHA(p.data),@"width":@(p.width),@"height":@(p.height),
          @"bytes":@(p.data.length),@"bytes_per_row":@(p.width*p.bpp),@"metal_pixel_format":@(p.format)}];
      }
      NSDictionary* record=@{@"schema":@"netal.nr.motion-pair-frame.v1",@"pair_index":@(index),
        @"pid":@(getpid()),@"source_frame":@(f->input.frame),@"stream":@(f->input.stream),
        @"source_frame_semantics":@"global temporal-scaler invocation ordinal; not independently proven presented-frame ID",
        @"command_id":@(f->input.command),@"ids_are_process_local":@YES,
        @"encoded":@(f->encoded),@"gpu_completed":@(gpu_ok),@"gpu_error":gpu_error?:@"",
        @"metadata":f->input.metadata,@"files":files,@"capture_position":@"before-original-scaler",
        @"actual_nr":@NO,@"temporal_admitted":@NO,@"game_convention_verified":@NO,
        @"note":@"Raw input textures only; pair acceptance additionally requires two consecutive same-stream successful records. No model or history consumes these files."};
      NSData* json=[NSJSONSerialization dataWithJSONObject:record options:NSJSONWritingPrettyPrinted|NSJSONWritingSortedKeys error:&err];
      if(!json || ![json writeToFile:[dir stringByAppendingPathComponent:@"capture.json"] options:NSDataWritingWithoutOverwriting error:&err])
        NSLog(@"netal_nr motion_pair_manifest_failed error=%@",err);
    }
  }
public:
  MotionPairCapture(std::string root,std::string arm):root_(std::move(root)),arm_(std::move(arm)){}
  // False means no complete capture was encoded; the original renderer still runs.
  bool encode(id<MTLCommandBuffer> cb,const MotionPairInput& input) {
    std::lock_guard<std::mutex> lock(mutex_);
    if(stopped_ || selected_==2 || root_.empty() || root_[0]!='/' || arm_.empty())return false;
    if(access((arm_+".cancel").c_str(),F_OK)==0){stopped_=true;return false;}
    if(!selected_ && access(arm_.c_str(),F_OK)!=0)return false;
    if(!cb || cb.status!=MTLCommandBufferStatusNotEnqueued || !input.frame || !input.stream || !input.command ||
       !TextureOK(input.color,cb.device,MTLPixelFormatRGBA16Float,input.fence) ||
       !TextureOK(input.motion,cb.device,MTLPixelFormatRG16Float,input.fence) || !MetadataOK(input.metadata)) {
      // A geometry discontinuity must not silently skip to a later second frame.
      if(selected_)stopped_=true;
      return false;
    }
    NSDictionary* geometry=Geometry(input.metadata);
    if(selected_ && (input.frame!=first_+1 || input.stream!=stream_ || input.command==first_command_ || ![geometry isEqual:geometry_])) {
      stopped_=true;NSLog(@"netal_nr motion_pair_refused nonconsecutive_or_changed_geometry");return false;
    }
    auto f=std::make_shared<MotionPairFrame>();f->input=input;f->input.metadata=[input.metadata copy];
    f->planes={CapturePlane{nil,"color.rgba16f",1152,720,8,MTLPixelFormatRGBA16Float,false},
               CapturePlane{nil,"motion.rg16f",1152,720,4,MTLPixelFormatRG16Float,false}};
    for(auto& p:f->planes){
      p.data=[cb.device newBufferWithLength:size_t(p.width)*p.height*p.bpp options:MTLResourceStorageModeShared|MTLResourceHazardTrackingModeTracked];
      if(!p.data){stopped_=true;return false;}
      std::memset(p.data.contents,0xA5,p.data.length);
    }
    if(!selected_){first_=input.frame;stream_=input.stream;first_command_=input.command;geometry_=geometry;}
    const unsigned index=selected_++;
    // Lease registered BEFORE encoding, also for unretained caller buffers.
    // Capture values by value, never references to this stack or this object's lifetime.
    const std::string root=root_;dispatch_queue_t writer=writer_;
    [cb addCompletedHandler:^(id<MTLCommandBuffer> done){
      const bool ok=done.status==MTLCommandBufferStatusCompleted;
      NSString* error=[done.error.localizedDescription copy];
      // GPU lifetime is complete. Do not retain game textures through disk I/O.
      f->input.color=nil;f->input.motion=nil;f->input.fence=nil;
      dispatch_async(writer,^{try{@try{Write(root,f,index,ok,error);}
        @catch(NSException* e){NSLog(@"netal_nr motion_pair_writer_exception %@",e);}}
        catch(...){NSLog(@"netal_nr motion_pair_writer_cpp_exception");}});
    }];
    netal_nr_experimental::FrameEncoderState ownership;
    auto b=[cb blitCommandEncoder];netal_nr_experimental::EncoderScope scope(ownership,b);
    if(!b){stopped_=true;return false;}
    b.label=@"Netal capture-only consecutive color/motion";
    if(input.fence)[b waitForFence:input.fence];
    for(unsigned i=0;i<2;++i){auto& p=f->planes[i];
      [b copyFromTexture:i?input.motion:input.color sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0)
        sourceSize:MTLSizeMake(p.width,p.height,1) toBuffer:p.data destinationOffset:0
        destinationBytesPerRow:p.width*p.bpp destinationBytesPerImage:p.data.length];
      p.copied=true;
    }
    if(input.fence)[b updateFence:input.fence];
    f->encoded=scope.close();
    if(!f->encoded)stopped_=true;
    return f->encoded;
  }
  void stop(){std::lock_guard<std::mutex> lock(mutex_);stopped_=true;}
  // Standalone tests only, never call from the game callback.
  void drain_writer_for_test(){dispatch_sync(writer_,^{});}
};
} // namespace netal_nr_adapter
