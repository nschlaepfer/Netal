#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <CoreGraphics/CoreGraphics.h>
#import <ImageIO/ImageIO.h>
#import <CommonCrypto/CommonDigest.h>
#include <simd/simd.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
#include "../fg_runtime.h"
#include "../../fg_host/offline_sequence.h"

namespace fs = std::filesystem;
constexpr unsigned W = 640, H = 360;
using Bytes = std::vector<uint8_t>;

static std::string Hash(const Bytes& bytes) {
  unsigned char digest[CC_SHA256_DIGEST_LENGTH];
  CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size()), digest);
  const char* hex = "0123456789abcdef";
  std::string result;
  for (auto b : digest) { result += hex[b >> 4]; result += hex[b & 15]; }
  return result;
}
static NSString* Str(const std::string& value) { return [NSString stringWithUTF8String:value.c_str()]; }
static void Binary(const fs::path& path, const Bytes& bytes) {
  std::ofstream file(path,std::ios::binary);
  file.write(reinterpret_cast<const char*>(bytes.data()),std::streamsize(bytes.size())); file.close();
  if(!file) throw std::runtime_error("binary evidence write failed");
}
static id<MTLTexture> Texture(id<MTLDevice> device, MTLPixelFormat format) {
  auto* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:W height:H mipmapped:NO];
  d.storageMode = MTLStorageModeShared;
  d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
  auto t = [device newTextureWithDescriptor:d];
  if (!t) throw std::runtime_error("fixture texture allocation failed");
  return t;
}
static Bytes Read(id<MTLTexture> texture) {
  Bytes bytes(W * H * 4);
  [texture getBytes:bytes.data() bytesPerRow:W * 4 fromRegion:MTLRegionMake2D(0,0,W,H) mipmapLevel:0];
  return bytes;
}
static void PNG(const fs::path& path, const Bytes& rgba) {
  auto provider = CGDataProviderCreateWithData(nullptr, rgba.data(), rgba.size(), nullptr);
  auto space = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
  auto image = CGImageCreate(W,H,8,32,W*4,space,
      CGBitmapInfo(kCGBitmapByteOrder32Big) | CGBitmapInfo(kCGImageAlphaPremultipliedLast),provider,nullptr,false,kCGRenderingIntentDefault);
  auto destination = CGImageDestinationCreateWithURL((__bridge CFURLRef)[NSURL fileURLWithPath:Str(path.string())],CFSTR("public.png"),1,nullptr);
  if (!image || !destination) throw std::runtime_error("PNG creation failed");
  CGImageDestinationAddImage(destination,image,nullptr);
  bool ok = CGImageDestinationFinalize(destination);
  CFRelease(destination); CGImageRelease(image); CGColorSpaceRelease(space); CGDataProviderRelease(provider);
  if (!ok) throw std::runtime_error("PNG write failed");
}
struct Frame { id<MTLTexture> color, motion, depth, hud; };
struct SceneRenderer {
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  id<MTLComputePipelineState> pipeline, hudPipeline;
  SceneRenderer(id<MTLDevice> d, const fs::path& shader) : device(d), queue([d newCommandQueue]) {
    NSError* error = nil;
    auto source = [NSString stringWithContentsOfFile:Str(shader.string()) encoding:NSUTF8StringEncoding error:&error];
    if (!source) throw std::runtime_error("fixture shader missing");
    auto* options = [MTLCompileOptions new]; options.fastMathEnabled = NO;
    auto library = [device newLibraryWithSource:source options:options error:&error];
    if (!library) throw std::runtime_error(error.localizedDescription.UTF8String);
    pipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"simulate_native"] error:&error];
    if (!pipeline) throw std::runtime_error(error.localizedDescription.UTF8String);
    auto hudSource=[NSString stringWithContentsOfFile:Str((shader.parent_path().parent_path()/"hud_compositor.metal").string()) encoding:NSUTF8StringEncoding error:&error];
    if(!hudSource) throw std::runtime_error("HUD compositor source missing");
    auto hudLibrary=[device newLibraryWithSource:hudSource options:options error:&error];
    if(!hudLibrary) throw std::runtime_error(error.localizedDescription.UTF8String);
    hudPipeline=[device newComputePipelineStateWithFunction:[hudLibrary newFunctionWithName:@"fg_composite_hud"] error:&error];
    if(!hudPipeline) throw std::runtime_error(error.localizedDescription.UTF8String);
  }
  Frame render(float time, float previous, int profile, bool reset, bool includeHUD=true, float hudTime=std::numeric_limits<float>::quiet_NaN()) {
    Frame f{Texture(device,MTLPixelFormatRGBA8Unorm),Texture(device,MTLPixelFormatRG16Float),Texture(device,MTLPixelFormatR32Float),Texture(device,MTLPixelFormatRGBA8Unorm)};
    auto command = [queue commandBuffer]; auto encoder = [command computeCommandEncoder];
    vector_float4 parameters{time,previous,float(profile),reset ? 1.0f : 0.0f};
    uint32_t hudEnabled=includeHUD ? 1u : 0u; if(std::isnan(hudTime)) hudTime=time;
    [encoder setComputePipelineState:pipeline];
    [encoder setTexture:f.color atIndex:0]; [encoder setTexture:f.motion atIndex:1]; [encoder setTexture:f.depth atIndex:2];
    [encoder setTexture:f.hud atIndex:3];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:0];
    [encoder setBytes:&hudEnabled length:sizeof(hudEnabled) atIndex:1];
    [encoder setBytes:&hudTime length:sizeof(hudTime) atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(W,H,1) threadsPerThreadgroup:MTLSizeMake(16,8,1)];
    [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) throw std::runtime_error("fixture rendering failed");
    return f;
  }
  id<MTLTexture> composite(id<MTLTexture> scene, id<MTLTexture> hud) {
    auto output=Texture(device,MTLPixelFormatRGBA8Unorm);
    auto command=[queue commandBuffer]; auto encoder=[command computeCommandEncoder];
    [encoder setComputePipelineState:hudPipeline];
    [encoder setTexture:scene atIndex:0]; [encoder setTexture:hud atIndex:1]; [encoder setTexture:output atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(W,H,1) threadsPerThreadgroup:MTLSizeMake(16,8,1)];
    [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
    if(command.status!=MTLCommandBufferStatusCompleted) throw std::runtime_error("HUD composition failed");
    return output;
  }
};
static NSDictionary* ErrorMetrics(const Bytes& actual, const Bytes& reference, int occlusionFrame=-1, bool linearOcclusion=false) {
  double total = 0, edge = 0, referenceEdge = 0; size_t count = 0, referenceCount = 0, pixels=0;
  double lo=W, hi=0;
  if(occlusionFrame>=0) for(double t : {occlusionFrame-1.0,occlusionFrame-0.5,double(occlusionFrame)}) {
    double x=linearOcclusion ? 64+8*t : 320+120*std::sin(t*0.11); lo=std::min(lo,x); hi=std::max(hi,x);
  }
  for (unsigned y=1;y<H-1;y++) for (unsigned x=1;x<W-1;x++) {
    // Score swept foreground boundaries, including revealed/covered background,
    // using analytic geometry rather than a mask produced by the model.
    if(occlusionFrame>=0 && (y<77 || y>283 || !((x>=lo-31 && x<=hi-25)||(x>=lo+25 && x<=hi+31)))) continue;
    size_t i=(y*W+x)*4; double e=0; int gradient=0, referenceGradient=0;
    for (int c=0;c<3;c++) {
      e += std::abs(int(actual[i+c])-int(reference[i+c]))/(255.0*3);
      for (int offset : {-4,4,-int(W*4),int(W*4)}) {
        gradient=std::max(gradient,std::abs(int(actual[i+c])-int(actual[i+offset+c])));
        gradient=std::max(gradient,std::abs(int(reference[i+c])-int(reference[i+offset+c])));
        referenceGradient=std::max(referenceGradient,std::abs(int(reference[i+c])-int(reference[i+offset+c])));
      }
    }
    total+=e; pixels++; if (gradient>=24) { edge+=e; count++; }
    if (referenceGradient>=24) { referenceEdge+=e; referenceCount++; }
  }
  return @{ @"mae":@(pixels ? total/pixels : 0), @"measured_pixels":@(pixels), @"edge_mae":@(count ? edge/count : 0), @"edge_pixels":@(count),
    @"reference_edge_mae":@(referenceCount ? referenceEdge/referenceCount : 0), @"reference_edge_pixels":@(referenceCount) };
}
static NSDictionary* HUDMetrics(const Bytes& actual, const Bytes& model, const Bytes& ui, bool separate) {
  size_t opaque=0, opaqueErrors=0, backgroundErrors=0, translucent=0; int opaqueMax=0, blendMax=0; double opaqueError=0;
  for(size_t i=0;i<ui.size();i+=4) {
    if(separate && actual[i+3]!=255) throw std::runtime_error("HUD composition output is not opaque");
    int alpha=ui[i+3]; bool mismatch=false, backgroundMismatch=false;
    for(int c=0;c<3;c++) {
      if(ui[i+c]>alpha) throw std::runtime_error("HUD producer violated premultiplied color contract");
      int error=std::abs(int(actual[i+c])-int(ui[i+c]));
      mismatch|=error!=0; backgroundMismatch|=actual[i+c]!=model[i+c];
      if(alpha==255) { opaqueMax=std::max(opaqueMax,error); opaqueError+=error/(255.0*3); }
      if(alpha>0 && alpha<255) {
        int expected=std::clamp(int(std::lround(ui[i+c]+model[i+c]*(255-alpha)/255.0)),0,255);
        blendMax=std::max(blendMax,std::abs(int(actual[i+c])-expected));
      }
    }
    if(alpha==255) { opaque++; opaqueErrors+=mismatch; }
    if(alpha==0) backgroundErrors+=backgroundMismatch;
    if(alpha>0 && alpha<255) translucent++;
  }
  bool contract=opaque>0 && opaqueErrors==0 && backgroundErrors==0 && blendMax<=1;
  if(separate && !contract) throw std::runtime_error("HUD composition violated opaque/blend/background contract");
  return @{ @"opaque_pixels":@(opaque), @"opaque_error_pixels":@(opaqueErrors), @"opaque_max_byte_error":@(opaqueMax),
    @"opaque_mae":@(opaque ? opaqueError/opaque : 0), @"translucent_pixels":@(translucent),
    @"compositor_tested":@(separate), @"compositor_max_blend_byte_error":separate ? @(blendMax) : [NSNull null],
    @"background_changed_pixels":separate ? @(backgroundErrors) : [NSNull null], @"compositor_contract_passed":@(separate && contract) };
}
static Bytes CompositeBaseline(const Bytes& scene, const Bytes& ui) {
  Bytes output(scene.size());
  for(size_t i=0;i<output.size();i+=4) {
    for(int c=0;c<3;c++) output[i+c]=uint8_t(std::clamp(int(std::lround(ui[i+c]+scene[i+c]*(255-ui[i+3])/255.0)),0,255));
    output[i+3]=255;
  }
  return output;
}
// Diagnostics deliberately perturb only the submitted motion buffer. Neither
// scene color, analytic midpoint nor the accepted native runtime is modified.
static NSDictionary* MotionExperiment(const Frame& f, const std::string& experiment) {
  auto bytes=Read(f.motion); const auto original=Hash(bytes);
  double maxX=0, maxY=0; unsigned invalid=0;
  for(unsigned i=0;i<W*H;i++) {
    __fp16 values[2]; std::memcpy(values,bytes.data()+i*4,4);
    // Preserve invalid-vector markers instead of scaling them into infinities.
    if(values[0]==65504 || values[1]==65504) { invalid++; continue; }
    for(int axis=0;axis<2;axis++) {
      float factor=1;
      if(experiment=="zero") factor=0;
      else if(experiment=="extent-scaled" || experiment=="extent-negated" || experiment=="extent-flip-y") {
        factor=axis==0 ? W : H;
        if(experiment=="extent-negated" || (experiment=="extent-flip-y" && axis==1)) factor=-factor;
      }
      float value=float(values[axis])*factor;
      if(!std::isfinite(value) || std::abs(value)>=65504) throw std::runtime_error("motion experiment exceeds finite non-sentinel half range");
      values[axis]=value;
      if(std::abs(float(values[axis]))>=65504) throw std::runtime_error("motion experiment rounds into the reserved invalid marker");
    }
    maxX=std::max(maxX,std::abs(double(values[0]))); maxY=std::max(maxY,std::abs(double(values[1])));
    std::memcpy(bytes.data()+i*4,values,4);
  }
  if(experiment!="none") [f.motion replaceRegion:MTLRegionMake2D(0,0,W,H) mipmapLevel:0 withBytes:bytes.data() bytesPerRow:W*4];
  return @{ @"fixture_motion_sha256":Str(original), @"submitted_max_abs_x":@(maxX), @"submitted_max_abs_y":@(maxY), @"invalid_motion_pixels":@(invalid) };
}
static void CheckInputs(const Frame& f, bool reset) {
  const auto motion=Read(f.motion), depth=Read(f.depth);
  for (unsigned i=0;i<W*H;i++) {
    __fp16 mx, my; float z;
    std::memcpy(&mx,motion.data()+i*4,2); std::memcpy(&my,motion.data()+i*4+2,2);
    std::memcpy(&z,depth.data()+i*4,4);
    if (!std::isfinite(z) || z<0 || z>1 || !std::isfinite(float(mx)) || !std::isfinite(float(my)))
      throw std::runtime_error("invalid simulator depth or motion");
    if (reset && (mx!=0 || my!=0)) throw std::runtime_error("reset fixture has nonzero motion");
  }
}
struct SessionOwner { NSFGSession* session=nullptr; ~SessionOwner() { if(session) NSFGSessionDestroy(session); } };

int main(int argc, const char** argv) { @autoreleasepool {
  try {
    if(argc!=9) throw std::runtime_error("usage: native-fg-quality BUNDLE SHADER NEW_OUTPUT SCENE FRAMES MOTION_EXPERIMENT EXPORT_INPUTS HUD_MODE");
    fs::path bundle=argv[1], output=argv[3]; std::string scene=argv[4];
    std::string experiment=argv[6];
    const std::string exportArgument=argv[7];
    if(exportArgument!="yes" && exportArgument!="no") throw std::runtime_error("invalid export selection");
    const bool exportInputs=exportArgument=="yes";
    const std::string hudMode=argv[8];
    if(hudMode!="embedded" && hudMode!="separate") throw std::runtime_error("unknown HUD mode");
    const bool separateHUD=hudMode=="separate";
    if(experiment!="none" && experiment!="zero" && experiment!="extent-scaled" && experiment!="extent-negated" && experiment!="extent-flip-y")
      throw std::runtime_error("unknown motion experiment");
    int frames=std::stoi(argv[5]);
    if(frames<2 || frames>64 || fs::exists(output)) throw std::runtime_error("invalid frames or existing output");
    int profile = scene=="courtyard" ? 0 : scene=="orbit" ? 1 : scene=="fast-turn" ? 2 : scene=="translation-x" ? 4 : scene=="translation-y" ? 5 : scene=="hud-stress" ? 6 : scene=="occlusion" ? 7 : scene=="occlusion-linear" ? 8 : 3;
    if(profile==3 && scene!="scene-cut" && scene!="scene-cut-odd") throw std::runtime_error("unknown scene");
    if(profile==3 && frames<15) throw std::runtime_error("cut sequences need at least15 frames");
    auto device=MTLCreateSystemDefaultDevice(); if(!device) throw std::runtime_error("Metal unavailable");
    SceneRenderer renderer(device,argv[2]);
    // Independently check the fixture's stationary correspondence before model use.
    auto stationary=renderer.render(3,3,profile,false);
    auto stationaryMotion=Read(stationary.motion);
    double stationaryMax=0;
    for(size_t i=0;i<W*H*2;i++) { __fp16 value; std::memcpy(&value,stationaryMotion.data()+i*2,2); stationaryMax=std::max(stationaryMax,std::abs(double(value))); }
    if(stationaryMax>0.002) throw std::runtime_error("stationary reprojection is not zero");
    std::string libs=(bundle/"libs").string(), model=(bundle/"model").string(), rom=(bundle/"rom.bin").string(),
      sigmoid=(bundle/"sigmoid.bin").string(), unit=(bundle/"ex2-unit.bin").string(), fraction=(bundle/"ex2-fraction.bin").string();
    auto queue=[device newCommandQueue];
    NSFGCreateInfo create{};
    create.metal_device=(__bridge void*)device; create.metal_command_queue=(__bridge void*)queue;
    create.metallib_directory=libs.c_str(); create.model_directory=model.c_str(); create.rcp_rom_path=rom.c_str();
    create.sigmoid_lut_path=sigmoid.c_str(); create.ex2_unit_table_path=unit.c_str(); create.ex2_fraction_table_path=fraction.c_str();
    create.width=W; create.height=H; create.ex2_mode=NSFG_EX2_COMPACT;
    create.auxiliary_mode=NSFG_AUXILIARY_LIVE_EARLY_RESOURCES; create.poison_byte=0xa5; create.enable_audit=true;
    SessionOwner owner;
    auto status=NSFGSessionCreate(&create,&owner.session);
    if(status!=NSFG_STATUS_OK) throw std::runtime_error(std::string("session creation: ")+NSFGStatusName(status));
    fs::create_directories(output);
    NSMutableArray* rows=[NSMutableArray array]; id<MTLTexture> previousColor=nil;
    int failures=0, generated=0; const int resetAt=scene=="scene-cut-odd" ? 13 : 12;
    const float timeOffset=scene=="scene-cut-odd" ? -1.0f : 0;
    Bytes exported=exportInputs ? neuralstudio::fg::offline::Header(frames) : Bytes{};
    for(int index=0;index<frames;index++) { @autoreleasepool {
      bool reset=index==0 || (profile==3 && index==resetAt);
      float time=index+timeOffset, prior=index-1+timeOffset;
      auto current=renderer.render(time,reset ? time : prior,profile,reset,!separateHUD); CheckInputs(current,reset);
      auto motionDiagnostic=MotionExperiment(current,experiment); CheckInputs(current,reset);
      const auto colorBefore=Read(current.color), motionBefore=Read(current.motion), depthBefore=Read(current.depth);
      const auto hudBefore=Read(current.hud);
      if(exportInputs) neuralstudio::fg::offline::Append(exported,{uint64_t(index),reset,colorBefore,motionBefore,depthBefore});
      const auto previousBefore=previousColor ? Read(previousColor) : Bytes{};
      auto target=Texture(device,MTLPixelFormatRGBA8Unorm);
      Bytes poison(W*H*4,0xa5); [target replaceRegion:MTLRegionMake2D(0,0,W,H) mipmapLevel:0 withBytes:poison.data() bytesPerRow:W*4];
      NSFGFrameInfo frame{};
      frame.current_backbuffer_rgba8=(__bridge void*)current.color;
      frame.previous_backbuffer_rgba8=(__bridge void*)(reset ? current.color : previousColor);
      frame.motion_rg16f=(__bridge void*)current.motion; frame.depth_r32f=(__bridge void*)current.depth;
      frame.generated_target_rgba8=(__bridge void*)target; frame.frame_id=index; frame.reset=reset;
      NSFGFrameResult result{}; const auto before=std::chrono::steady_clock::now();
      status=NSFGSessionEvaluate(owner.session,&frame,&result);
      const double wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-before).count();
      if(status!=NSFG_STATUS_OK) throw std::runtime_error(std::string(NSFGStatusName(status))+": "+NSFGSessionLastError(owner.session));
      if(result.generated_output_valid==reset || result.selected_launch_count!=(reset ? 9u : 76u)) throw std::runtime_error("native generation/reset contract failed");
      if(colorBefore!=Read(current.color) || motionBefore!=Read(current.motion) || depthBefore!=Read(current.depth)) throw std::runtime_error("native inputs mutated");
      if(previousColor && previousBefore!=Read(previousColor)) throw std::runtime_error("previous color input mutated");
      if(reset && Read(target)!=poison) throw std::runtime_error("reset touched generated output");
      const auto modelOutput=reset ? colorBefore : Read(target);
      auto actual=separateHUD ? Read(renderer.composite(reset ? current.color : target,current.hud)) : modelOutput;
      if(Read(current.hud)!=hudBefore || (reset ? Read(current.color) : Read(target))!=modelOutput) throw std::runtime_error("HUD compositor mutated its inputs");
      auto hudMetrics=HUDMetrics(actual,modelOutput,hudBefore,separateHUD);
      // HUD updates remain at current UI time, independent of the scene midpoint.
      auto expected=renderer.render(reset ? time : time-0.5f,reset ? time : time-0.5f,profile,false,true,time);
      auto ideal=Read(expected.color); auto metrics=ErrorMetrics(actual,ideal);
      bool quality=[metrics[@"edge_mae"] doubleValue]<=0.06;
      NSDictionary* occlusionMetrics=profile>=7 ? ErrorMetrics(actual,ideal,index,profile==8) : nil;
      if(occlusionMetrics && [occlusionMetrics[@"edge_mae"] doubleValue]>0.06) quality=false;
      if(profile==6 && [hudMetrics[@"opaque_error_pixels"] unsignedLongValue]!=0) quality=false;
      if(!quality) failures++; if(!reset) generated++;
      std::string name="frame-"+std::to_string(index);
      PNG(output/(name+"-actual.png"),actual); PNG(output/(name+"-reference.png"),ideal);
      PNG(output/(name+"-current.png"),colorBefore);
      if(exportInputs) {
        Binary(output/(name+"-native.rgba8"),actual);
        Binary(output/(name+"-model.rgba8"),modelOutput);
        Binary(output/(name+"-reference.rgba8"),ideal);
      }
      NSMutableDictionary* row=[metrics mutableCopy];
      [row addEntriesFromDictionary:motionDiagnostic];
      row[@"current_frame_baseline"]=ErrorMetrics(separateHUD ? CompositeBaseline(colorBefore,hudBefore) : colorBefore,ideal);
      row[@"hud"]=hudMetrics;
      if(occlusionMetrics) row[@"occlusion"]=occlusionMetrics;
      row[@"native_model_output_sha256"]=Str(Hash(modelOutput));
      row[@"input_hud_sha256"]=Str(Hash(hudBefore));
      if(!reset) {
        Bytes blended(colorBefore.size());
        for(size_t i=0;i<blended.size();i++) blended[i]=(unsigned(colorBefore[i])+unsigned(previousBefore[i])+1)/2;
        row[@"previous_frame_baseline"]=ErrorMetrics(separateHUD ? CompositeBaseline(previousBefore,hudBefore) : previousBefore,ideal);
        row[@"frame_blend_baseline"]=ErrorMetrics(separateHUD ? CompositeBaseline(blended,hudBefore) : blended,ideal);
      }
      row[@"frame_id"]=@(index); row[@"reset"]=@(reset); row[@"generated"]=@(result.generated_output_valid);
      row[@"selected_launch_count"]=@(result.selected_launch_count); row[@"quality_target_passed"]=@(quality);
      row[@"metal_allocations_during_evaluate"]=@(result.resource_allocations_during_evaluate);
      row[@"evaluate_wall_ms_audit_enabled"]=@(wall); row[@"output_sha256"]=Str(Hash(actual));
      row[@"input_color_sha256"]=Str(Hash(colorBefore)); row[@"input_motion_sha256"]=Str(Hash(motionBefore));
      row[@"input_depth_sha256"]=Str(Hash(depthBefore)); row[@"reference_sha256"]=Str(Hash(ideal));
      [rows addObject:row]; previousColor=current.color;
      std::cout<<scene<<" frame="<<index<<" generated="<<result.generated_output_valid<<" edge_mae="<<[metrics[@"edge_mae"] doubleValue]<<std::endl;
    } }
    NSFGSessionStats stats{};
    if(NSFGSessionGetStats(owner.session,&stats)!=NSFG_STATUS_OK) throw std::runtime_error("runtime statistics unavailable");
    if(stats.generated_frame_count!=uint64_t(generated) || stats.reset_count!=uint64_t(frames-generated)
       || stats.last_accepted_frame_id!=uint64_t(frames-1)) throw std::runtime_error("runtime frame counters disagree with delivery");
    if(stats.input_mismatched_bytes || stats.canary_mismatched_bytes) throw std::runtime_error("runtime guard failure");
    if(exportInputs) {
      if(neuralstudio::fg::offline::Decode(exported).size()!=size_t(frames)) throw std::runtime_error("export validation failed");
      Binary(output/"inputs.nfg",exported);
    }
    NSDictionary* report=@{ @"schema":@"netal.native-fg-quality.v1", @"scene":Str(scene), @"device":device.name,
      @"os_version":NSProcessInfo.processInfo.operatingSystemVersionString, @"profile":@"640x360-live-aux-compact-ex2-reference-hmma",
      @"width":@(W), @"height":@(H), @"native_model_executed":@YES, @"apple_fg_used":@NO, @"block_flow_fallback_used":@NO,
      @"game_used":@NO, @"screen_capture_used":@NO, @"presentation_measured":@NO, @"new_fixture_rtx_parity_proven":@NO,
      @"motion_convention":@"fixture: current-to-previous pixels, y down; submitted values follow motion_experiment; pinned runtime scaling",
      @"depth_convention":(profile==4 || profile==5) ? @"translation plane depth0.5; embedded HUD depth0" : profile>=7 ? @"foreground depth0.2, background0.8; embedded HUD depth0" : @"forward device depth, near0.1 far1000, FOV60; embedded HUD near depth",
      @"camera_metadata":@"runtime retains its pinned identity-transform profile; scene motion supplied in vectors",
      @"motion_experiment":Str(experiment), @"motion_input_perturbed":@(experiment!="none"),
      @"experiment_promoted_to_runtime":@NO,
      @"hud_mode":Str(hudMode), @"hud_source":@"independent renderer layer; never inferred from captured RGB",
      @"hud_presentation":@"current UI time over interpolated scene", @"hud_compositor_used":@(separateHUD),
      @"baseline_hud_policy":separateHUD ? @"same current UI layer applied to held/blended scene baselines" : @"original embedded-color baselines",
      @"input_sequence_exported":@(exportInputs), @"input_sequence_sha256":exportInputs ? Str(Hash(exported)) : @"not-exported",
      @"stationary_motion_max_abs_px":@(stationaryMax), @"generated_frames":@(generated), @"quality_failures":@(failures),
      @"input_mismatched_bytes":@(stats.input_mismatched_bytes), @"canary_mismatched_bytes":@(stats.canary_mismatched_bytes),
      @"runtime_contracts_passed":@YES, @"timing_scope":@"audit-enabled evaluation wall time; not performance or display FPS",
      @"frames":rows };
    NSError* error=nil; auto data=[NSJSONSerialization dataWithJSONObject:report options:NSJSONWritingPrettyPrinted|NSJSONWritingSortedKeys error:&error];
    if(!data || ![data writeToFile:Str((output/"report.json").string()) options:NSDataWritingWithoutOverwriting error:&error]) throw std::runtime_error("receipt write failed");
    return failures ? 2 : 0;
  } catch(const std::exception& e) { std::cerr<<"native-quality-error="<<e.what()<<std::endl; return 1; }
} }
