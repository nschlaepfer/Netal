#import <CommonCrypto/CommonDigest.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

#include "contract.h"

#ifdef FG_CONNECTED_FULL_CHAIN
#include "macos_live_handoff.h"
int FgConnectedFullTailMain(int argc, const char* argv[]);
#endif

namespace fs = std::filesystem;

namespace {

namespace contract = fg_connected_temporal_l3_l87;
constexpr std::size_t kGuardBytes = 256;
constexpr std::size_t kRomBytes = 2097152;
constexpr std::array<std::size_t,5> kPyramidBytes = {
    57600,86400,21600,5520,1440};

struct L3Parameters { std::uint32_t width,height,half_width,half_height; };
struct L4Parameters { std::uint32_t reset_mode,input_width,input_height,
                                    output_width,output_height; };
struct L5Parameters { std::uint32_t width,height; float motion_divisor,
    neighbor_divisor,depth_scale; std::uint32_t invert_depth; };
struct L8Parameters { std::uint32_t width,height,mode; };
struct L9Parameters { std::uint32_t width,height; };
struct StatusParameters { std::uint32_t count,value; };
struct L11Parameters { std::uint32_t input_x,input_y,output_x,output_y,
                                      width,height; };
struct L19Parameters { std::uint32_t input_width,input_height,tensor_width,
                                      tensor_height; std::int32_t input_y_offset; };
struct L20Parameters { std::uint32_t width,height,input_channels,
                                      output_channels,
                                      active_weight_half_count,
                                      column_stage_schedule; };

std::vector<std::uint8_t> ReadFile(const fs::path& path) {
  std::ifstream stream(path,std::ios::binary|std::ios::ate);
  if(!stream) throw std::runtime_error("unable to read "+path.string());
  const auto size=stream.tellg();
  std::vector<std::uint8_t> result(static_cast<std::size_t>(size));
  stream.seekg(0);stream.read(reinterpret_cast<char*>(result.data()),size);
  if(!stream) throw std::runtime_error("short read "+path.string());
  return result;
}

void Write(const fs::path& path,const std::vector<std::uint8_t>& bytes) {
  std::ofstream stream(path,std::ios::binary|std::ios::trunc);
  if(!stream) throw std::runtime_error("unable to write "+path.string());
  stream.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
}

std::string Sha256(const std::vector<std::uint8_t>& bytes) {
  std::array<unsigned char,CC_SHA256_DIGEST_LENGTH> digest{};
  CC_SHA256(bytes.data(),static_cast<CC_LONG>(bytes.size()),digest.data());
  std::ostringstream text;text<<std::hex<<std::setfill('0');
  for(const auto byte:digest)text<<std::setw(2)<<unsigned(byte);
  return text.str();
}

std::size_t Mismatches(const std::vector<std::uint8_t>& a,
                       const std::vector<std::uint8_t>& b) {
  if(a.size()!=b.size()) throw std::runtime_error("comparison size drift");
  std::size_t result=0;
  for(std::size_t i=0;i<a.size();++i)result+=a[i]!=b[i];
  return result;
}

id<MTLBuffer> Buffer(id<MTLDevice> device,const std::vector<std::uint8_t>& data) {
  id<MTLBuffer> result=[device newBufferWithLength:data.size()+2*kGuardBytes
                                           options:MTLResourceStorageModeShared];
  if(result==nil)throw std::runtime_error("buffer allocation failed");
  std::memset(result.contents,contract::kCanaryValue,result.length);
  std::memcpy(static_cast<std::uint8_t*>(result.contents)+kGuardBytes,
              data.data(),data.size());
  return result;
}

std::vector<std::uint8_t> Payload(id<MTLBuffer> buffer,std::size_t bytes) {
  const auto* begin=static_cast<const std::uint8_t*>(buffer.contents)+kGuardBytes;
  return {begin,begin+bytes};
}

std::size_t CanaryMismatches(id<MTLBuffer> buffer,std::size_t bytes) {
  const auto* data=static_cast<const std::uint8_t*>(buffer.contents);
  std::size_t result=0;
  for(std::size_t i=0;i<kGuardBytes;++i){
    result+=data[i]!=contract::kCanaryValue;
    result+=data[kGuardBytes+bytes+i]!=contract::kCanaryValue;
  }
  return result;
}

id<MTLTexture> Texture(id<MTLDevice> device,MTLPixelFormat format,
                       std::uint32_t width,std::uint32_t height,
                       MTLTextureUsage usage,
                       const std::vector<std::uint8_t>& initial,
                       std::size_t bytes_per_pixel) {
  if(initial.size()!=std::size_t{width}*height*bytes_per_pixel)
    throw std::runtime_error("texture initialization size drift");
  auto* descriptor=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
      width:width height:height mipmapped:NO];
  descriptor.storageMode=MTLStorageModeShared;descriptor.usage=usage;
  id<MTLTexture> result=[device newTextureWithDescriptor:descriptor];
  if(result==nil)throw std::runtime_error("texture allocation failed");
  [result replaceRegion:MTLRegionMake2D(0,0,width,height) mipmapLevel:0
             withBytes:initial.data() bytesPerRow:width*bytes_per_pixel];
  return result;
}

std::vector<std::uint8_t> TextureBytes(id<MTLTexture> texture,
                                       std::size_t bytes_per_pixel) {
  std::vector<std::uint8_t> result(texture.width*texture.height*bytes_per_pixel);
  [texture getBytes:result.data() bytesPerRow:texture.width*bytes_per_pixel
          fromRegion:MTLRegionMake2D(0,0,texture.width,texture.height)
         mipmapLevel:0];
  return result;
}

id<MTLComputePipelineState> Pipeline(id<MTLDevice> device,
                                     const fs::path& library_path,
                                     const char* name) {
  NSError* error=nil;
  id<MTLLibrary> library=[device newLibraryWithURL:
      [NSURL fileURLWithPath:[NSString stringWithUTF8String:
          library_path.string().c_str()]] error:&error];
  if(library==nil)throw std::runtime_error("metallib load failed");
  id<MTLFunction> function=[library newFunctionWithName:
      [NSString stringWithUTF8String:name]];
  id<MTLComputePipelineState> result=
      [device newComputePipelineStateWithFunction:function error:&error];
  if(function==nil||result==nil)throw std::runtime_error("pipeline failed");
  return result;
}

void Wait(id<MTLCommandBuffer> command) {
  [command commit];[command waitUntilCompleted];
  if(command.status!=MTLCommandBufferStatusCompleted)
    throw std::runtime_error(command.error.localizedDescription.UTF8String);
}

template<std::size_t N>void Store32(std::array<std::uint8_t,N>* bytes,
  std::size_t offset,std::uint32_t value){
  std::memcpy(bytes->data()+offset,&value,sizeof(value));
}
template<std::size_t N>void Store64(std::array<std::uint8_t,N>* bytes,
  std::size_t offset,std::uint64_t value){
  std::memcpy(bytes->data()+offset,&value,sizeof(value));
}

std::vector<std::uint8_t> Join(
    std::initializer_list<std::vector<std::uint8_t>> values) {
  std::vector<std::uint8_t> result;
  for(const auto& value:values)result.insert(result.end(),value.begin(),value.end());
  return result;
}

void BlitTextureToBuffer(id<MTLCommandQueue> queue,id<MTLTexture> texture,
                         id<MTLBuffer> buffer,std::size_t bytes_per_pixel) {
  id<MTLCommandBuffer> command=[queue commandBuffer];
  id<MTLBlitCommandEncoder> encoder=[command blitCommandEncoder];
  [encoder copyFromTexture:texture sourceSlice:0 sourceLevel:0
               sourceOrigin:MTLOriginMake(0,0,0)
                 sourceSize:MTLSizeMake(texture.width,texture.height,1)
                   toBuffer:buffer destinationOffset:kGuardBytes
        destinationBytesPerRow:texture.width*bytes_per_pixel
      destinationBytesPerImage:texture.width*texture.height*bytes_per_pixel];
  [encoder endEncoding];Wait(command);
}

void BlitBufferToTexture(id<MTLCommandQueue> queue,id<MTLBuffer> buffer,
                         id<MTLTexture> texture,std::size_t bytes_per_pixel) {
  id<MTLCommandBuffer> command=[queue commandBuffer];
  id<MTLBlitCommandEncoder> encoder=[command blitCommandEncoder];
  [encoder copyFromBuffer:buffer sourceOffset:kGuardBytes
        sourceBytesPerRow:texture.width*bytes_per_pixel
      sourceBytesPerImage:texture.width*texture.height*bytes_per_pixel
               sourceSize:MTLSizeMake(texture.width,texture.height,1)
                toTexture:texture destinationSlice:0 destinationLevel:0
        destinationOrigin:MTLOriginMake(0,0,0)];
  [encoder endEncoding];Wait(command);
}

}  // namespace

int main(int argc,const char* argv[]) {
  @autoreleasepool {
#ifdef FG_CONNECTED_FULL_CHAIN
    if(argc!=8){std::cerr<<"usage=fg-connected-l3-l87-mac <libs> <model> <rom> <sigmoid> <output> <run-id> <poison>\n";return 2;}
#else
    if(argc!=7){std::cerr<<"usage=fg-connected-l3-l19-mac <libs> <model> <rom> <output> <run-id> <poison>\n";return 2;}
#endif
    try {
#ifdef FG_CONNECTED_FULL_CHAIN
      const fs::path libs=argv[1],model=argv[2],output_root=argv[5];
      constexpr int kRunArgument=6;
      constexpr int kPoisonArgument=7;
#else
      const fs::path libs=argv[1],model=argv[2],output_root=argv[4];
      constexpr int kRunArgument=5;
      constexpr int kPoisonArgument=6;
#endif
      fs::create_directories(output_root);
      const auto poison_value=std::stoul(argv[kPoisonArgument],nullptr,16);
      if(poison_value>255u)throw std::runtime_error("invalid poison");
      const auto poison=static_cast<std::uint8_t>(poison_value);
      const auto rom=ReadFile(argv[3]);
      if(rom.size()!=kRomBytes||Sha256(rom)!="7cf4f8cece6b18b07605edc9cb601a153346a79b10bf3b93c7ec058d57d29750")
        throw std::runtime_error("RCP ROM identity mismatch");
      const auto l20_weight_bytes=ReadFile(model/"default-token-000063.bin");
      if(l20_weight_bytes.size()!=56320||Sha256(l20_weight_bytes)!=
         "44595b02307a6b934a124abe476cfd327440e2acba7f5ed69f73195ca743e3d9")
        throw std::runtime_error("L20 weight identity mismatch");
      const auto external=contract::Generate();
      id<MTLDevice> device=MTLCreateSystemDefaultDevice();
      id<MTLCommandQueue> queue=[device newCommandQueue];
      if(device==nil||queue==nil)throw std::runtime_error("Metal unavailable");
      auto l3=Pipeline(device,libs/"l3_l13.metallib","fg_early_l3_l13_mode2");
      auto l4=Pipeline(device,libs/"l4_l14.metallib","fg_early_l4_l14");
      auto l5_high=Pipeline(device,libs/"l5_l15.metallib","fg_early_l5_l15_high");
      auto l5_low=Pipeline(device,libs/"l5_l15.metallib","fg_early_l5_l15_low");
      auto l5_pack=Pipeline(device,libs/"l5_l15.metallib","fg_early_l5_l15_pack");
      auto l6=Pipeline(device,libs/"l6_l16.metallib","fg_early_l6_l16");
      auto l7=Pipeline(device,libs/"l7_l17.metallib","fg_early_l7_l17");
      auto l8=Pipeline(device,libs/"l8_l18.metallib","fg_early_l8_l18");
      auto l9=Pipeline(device,libs/"l9.metallib","fg_early_l9");
      auto status=Pipeline(device,libs/"l10_l12.metallib","fg_early_status_write");
      auto l11=Pipeline(device,libs/"l11.metallib","fg_early_l11");
      auto l19_surfaces=Pipeline(device,libs/"l19.metallib","fg_early_l19_surfaces");
      auto l19_tensor=Pipeline(device,libs/"l19.metallib","fg_early_l19_tensor");
#ifndef FG_CONNECTED_FULL_CHAIN
      auto l20=Pipeline(device,libs/"l20.metallib","fg_conv_3x3");
#endif
      id<MTLBuffer> rom_buffer=Buffer(device,rom);
      id<MTLBuffer> l20_weight=Buffer(device,l20_weight_bytes);

      const auto zeros_history=contract::InitialHistory();
      auto source_reset=Texture(device,MTLPixelFormatRGBA8Unorm,640,360,
          MTLTextureUsageShaderRead,external.source_reset,4);
      auto source_generated=Texture(device,MTLPixelFormatRGBA8Unorm,640,360,
          MTLTextureUsageShaderRead,external.source_generated,4);
      auto source_generated2=Texture(device,MTLPixelFormatRGBA8Unorm,640,360,
          MTLTextureUsageShaderRead,external.source_generated2,4);
      auto history_a=Texture(device,MTLPixelFormatRGBA16Float,640,360,
          MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite,zeros_history,8);
      auto history_b=Texture(device,MTLPixelFormatRGBA16Float,640,360,
          MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite,
          std::vector<std::uint8_t>(contract::kHistoryBytes,poison),8);
      auto reset_feature=Texture(device,MTLPixelFormatRGBA16Float,320,180,
          MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite,
          std::vector<std::uint8_t>(contract::kHalfColorBytes,poison),8);
      auto generated_feature=Texture(device,MTLPixelFormatRGBA16Float,320,180,
          MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite,
          std::vector<std::uint8_t>(contract::kHalfColorBytes,poison),8);
      auto shared_metric=Texture(device,MTLPixelFormatR16Float,320,180,
          MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite,
          std::vector<std::uint8_t>(contract::kHalfScalarBytes,poison),2);
      auto vector_reset=Texture(device,MTLPixelFormatRG16Float,640,360,
          MTLTextureUsageShaderRead,external.vector_reset,4);
      auto vector_generated=Texture(device,MTLPixelFormatRG16Float,640,360,
          MTLTextureUsageShaderRead,external.vector_generated,4);
      auto vector_generated2=Texture(device,MTLPixelFormatRG16Float,640,360,
          MTLTextureUsageShaderRead,external.vector_generated2,4);
      auto scalar_reset=Texture(device,MTLPixelFormatR32Float,640,360,
          MTLTextureUsageShaderRead,external.scalar_reset,4);
      auto scalar_generated=Texture(device,MTLPixelFormatR32Float,640,360,
          MTLTextureUsageShaderRead,external.scalar_generated,4);
      auto scalar_generated2=Texture(device,MTLPixelFormatR32Float,640,360,
          MTLTextureUsageShaderRead,external.scalar_generated2,4);
      auto flow=Texture(device,MTLPixelFormatRG32Float,320,180,
          MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite,
          std::vector<std::uint8_t>(contract::kDepthBytes*2u,poison),8);
      auto reset_depth=Texture(device,MTLPixelFormatR32Uint,320,180,
          MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite,
          std::vector<std::uint8_t>(contract::kDepthBytes,poison),4);
      auto generated_depth=Texture(device,MTLPixelFormatR32Uint,320,180,
          MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite,
          contract::InitialGeneratedDepth(),4);
      auto reset_mask=Texture(device,MTLPixelFormatR8Unorm,320,180,
          MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite,
          std::vector<std::uint8_t>(contract::kMaskBytes,poison),1);
      auto generated_mask=Texture(device,MTLPixelFormatR8Unorm,320,180,
          MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite,
          contract::InitialGeneratedMask(),1);
      auto working_vector=Texture(device,MTLPixelFormatRG16Float,320,180,
          MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite,
          std::vector<std::uint8_t>(contract::kVectorBytes,poison),4);
      auto working_mask=Texture(device,MTLPixelFormatR8Unorm,320,180,
          MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite,
          std::vector<std::uint8_t>(contract::kMaskBytes,poison),1);
      auto scatter_reset=Texture(device,MTLPixelFormatR8Unorm,320,180,
          MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite,
          std::vector<std::uint8_t>(contract::kMaskBytes,255),1);
      auto scatter_generated=Texture(device,MTLPixelFormatR8Unorm,320,180,
          MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite,
          std::vector<std::uint8_t>(contract::kMaskBytes,255),1);
      auto reset_projection=Texture(device,MTLPixelFormatRG16Float,320,180,
          MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite,
          std::vector<std::uint8_t>(contract::kVectorBytes,poison),4);
      auto allocation29=Texture(device,MTLPixelFormatR16Float,320,180,
          MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite,
          contract::InitialAllocation29(),2);
      auto auxiliary_mask=Texture(device,MTLPixelFormatR8Unorm,320,180,
          MTLTextureUsageShaderWrite,
          std::vector<std::uint8_t>(contract::kAuxMaskBytes,poison),1);
      auto auxiliary_vector=Texture(device,MTLPixelFormatRG16Float,320,180,
          MTLTextureUsageShaderWrite,
          std::vector<std::uint8_t>(contract::kAuxVectorBytes,poison),4);
      id<MTLBuffer> model_input=Buffer(
          device,std::vector<std::uint8_t>(contract::kModelInputBytes,poison));
#ifndef FG_CONNECTED_FULL_CHAIN
      id<MTLBuffer> l20_output=Buffer(
          device,std::vector<std::uint8_t>(contract::kL20OutputBytes,poison));
#endif
      id<MTLBuffer> atomic=Buffer(device,contract::InitialAtomic());
      std::array<id<MTLBuffer>,5> pyramid;
      for(std::size_t i=0;i<pyramid.size();++i)
        pyramid[i]=Buffer(device,std::vector<std::uint8_t>(kPyramidBytes[i],poison));
      id<MTLBuffer> vector_buffer=Buffer(
          device,std::vector<std::uint8_t>(contract::kVectorBytes,poison));
      id<MTLBuffer> status_buffer=Buffer(device,std::vector<std::uint8_t>(4,poison));

#ifdef FG_CONNECTED_FULL_CHAIN
      std::array<std::string,25> checkpoints{};
#else
      std::array<std::string,18> checkpoints{};
#endif
      const L3Parameters l3_parameters{640,360,320,180};
      auto encode_l3=[&](id<MTLTexture> source,id<MTLTexture> history,
                         id<MTLTexture> feature,id<MTLTexture> next_history){
        id<MTLCommandBuffer> command=[queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
        [encoder setComputePipelineState:l3];
        [encoder setBytes:&l3_parameters length:sizeof(l3_parameters) atIndex:0];
        [encoder setTexture:source atIndex:0];[encoder setTexture:history atIndex:1];
        [encoder setTexture:feature atIndex:2];[encoder setTexture:shared_metric atIndex:3];
        [encoder setTexture:next_history atIndex:4];
        [encoder dispatchThreads:MTLSizeMake(320,180,1)
            threadsPerThreadgroup:MTLSizeMake(16,8,1)];
        [encoder endEncoding];Wait(command);
      };
      encode_l3(source_reset,history_a,reset_feature,history_b);
      checkpoints[0]=Sha256(Join({TextureBytes(reset_feature,8),
          TextureBytes(shared_metric,2),TextureBytes(history_b,8)}));

      auto encode_l4=[&](bool reset,id<MTLTexture> input_vector,
                         id<MTLTexture> input_scalar,id<MTLTexture> depth){
        const L4Parameters parameters{reset?127u:0u,640,360,320,180};
        id<MTLCommandBuffer> command=[queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
        [encoder setComputePipelineState:l4];[encoder setTexture:input_vector atIndex:0];
        [encoder setTexture:input_scalar atIndex:1];[encoder setTexture:flow atIndex:2];
        [encoder setTexture:depth atIndex:3];
        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:0];
        [encoder dispatchThreads:MTLSizeMake(320,180,1)
            threadsPerThreadgroup:MTLSizeMake(16,16,1)];
        [encoder endEncoding];Wait(command);
      };
      encode_l4(true,vector_reset,scalar_reset,reset_depth);
      checkpoints[1]=Sha256(Join({TextureBytes(flow,8),TextureBytes(reset_depth,4)}));

      std::size_t transient_canary_mismatches=0;
      auto encode_l5=[&](id<MTLTexture> depth,id<MTLTexture> mask){
        const std::vector<std::uint8_t> words(contract::kDepthBytes,0xff);
        id<MTLBuffer> high=Buffer(device,words),low=Buffer(device,words);
        const L5Parameters parameters{320,180,144.0f,12.96f,1.0f,0};
        for(id<MTLComputePipelineState> pipeline in @[l5_high,l5_low]){
          id<MTLCommandBuffer> command=[queue commandBuffer];
          id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
          [encoder setComputePipelineState:pipeline];[encoder setTexture:flow atIndex:0];
          [encoder setTexture:depth atIndex:1];[encoder setTexture:mask atIndex:2];
          [encoder setBuffer:high offset:kGuardBytes atIndex:0];
          [encoder setBuffer:low offset:kGuardBytes atIndex:1];
          [encoder setBuffer:rom_buffer offset:kGuardBytes atIndex:2];
          [encoder setBytes:&parameters length:sizeof(parameters) atIndex:3];
          [encoder dispatchThreads:MTLSizeMake(320,180,1)
              threadsPerThreadgroup:MTLSizeMake(16,16,1)];
          [encoder endEncoding];Wait(command);
        }
        id<MTLCommandBuffer> command=[queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
        [encoder setComputePipelineState:l5_pack];
        [encoder setBuffer:high offset:kGuardBytes atIndex:0];
        [encoder setBuffer:low offset:kGuardBytes atIndex:1];
        [encoder setBuffer:atomic offset:kGuardBytes atIndex:2];
        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(320u*180u,1,1)
            threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [encoder endEncoding];Wait(command);
        transient_canary_mismatches+=CanaryMismatches(high,words.size());
        transient_canary_mismatches+=CanaryMismatches(low,words.size());
      };
      encode_l5(reset_depth,reset_mask);
      checkpoints[2]=Sha256(Join({Payload(atomic,contract::kAtomicBytes),
                                  TextureBytes(reset_mask,1)}));

      auto encode_l6=[&](id<MTLTexture> opposite_depth,
                         id<MTLTexture> opposite_mask){
        std::array<std::uint8_t,224> parameters{};
        Store64(&parameters,24,atomic.gpuAddress+kGuardBytes);
        for(std::size_t i=0;i<pyramid.size();++i)
          Store64(&parameters,48+i*8,pyramid[i].gpuAddress+kGuardBytes);
        Store32(&parameters,128,std::bit_cast<std::uint32_t>(1.0f/640.0f));
        Store32(&parameters,132,std::bit_cast<std::uint32_t>(1.0f/360.0f));
        Store32(&parameters,136,320);Store32(&parameters,140,180);
        Store32(&parameters,144,std::bit_cast<std::uint32_t>(40.0f));
        Store32(&parameters,148,std::bit_cast<std::uint32_t>(1.0f));
        Store32(&parameters,152,std::bit_cast<std::uint32_t>(600.0f));
        const std::array<float,16> identity={1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
        std::memcpy(parameters.data()+160,identity.data(),64);
        id<MTLCommandBuffer> command=[queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
        [encoder setComputePipelineState:l6];
        [encoder setBytes:parameters.data() length:parameters.size() atIndex:0];
        [encoder setBuffer:rom_buffer offset:kGuardBytes atIndex:1];
        [encoder setTexture:flow atIndex:0];[encoder setTexture:opposite_depth atIndex:1];
        [encoder setTexture:opposite_mask atIndex:2];
        [encoder setTexture:working_vector atIndex:3];[encoder setTexture:working_mask atIndex:4];
        [encoder useResource:atomic usage:MTLResourceUsageRead];
        for(id<MTLBuffer> item:pyramid)[encoder useResource:item usage:MTLResourceUsageWrite];
        [encoder dispatchThreadgroups:MTLSizeMake(20,12,1)
                threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [encoder endEncoding];Wait(command);
      };
      encode_l6(generated_depth,generated_mask);
      checkpoints[3]=Sha256(Join({TextureBytes(working_vector,4),
          TextureBytes(working_mask,1),Payload(pyramid[0],kPyramidBytes[0]),
          Payload(pyramid[1],kPyramidBytes[1]),Payload(pyramid[2],kPyramidBytes[2]),
          Payload(pyramid[3],kPyramidBytes[3]),Payload(pyramid[4],kPyramidBytes[4])}));

      auto encode_l7=[&]{
        BlitTextureToBuffer(queue,working_vector,vector_buffer,4);
        std::array<std::uint8_t,72> parameters{};
        Store64(&parameters,8,pyramid[1].gpuAddress+kGuardBytes);
        Store64(&parameters,16,pyramid[2].gpuAddress+kGuardBytes);
        Store64(&parameters,24,pyramid[3].gpuAddress+kGuardBytes);
        Store64(&parameters,32,pyramid[4].gpuAddress+kGuardBytes);
        Store32(&parameters,40,320);Store32(&parameters,44,180);
        Store32(&parameters,48,1);
        Store64(&parameters,56,vector_buffer.gpuAddress+kGuardBytes);
        Store64(&parameters,64,pyramid[0].gpuAddress+kGuardBytes);
        id<MTLCommandBuffer> command=[queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
        [encoder setComputePipelineState:l7];
        [encoder setBytes:parameters.data() length:parameters.size() atIndex:0];
        [encoder setBuffer:rom_buffer offset:kGuardBytes atIndex:1];
        for(id<MTLBuffer> item:pyramid)[encoder useResource:item usage:MTLResourceUsageRead];
        [encoder useResource:vector_buffer usage:MTLResourceUsageWrite];
        [encoder dispatchThreadgroups:MTLSizeMake(20,12,1)
                threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [encoder endEncoding];Wait(command);
        BlitBufferToTexture(queue,vector_buffer,working_vector,4);
      };
      encode_l7();
      checkpoints[4]=Sha256(TextureBytes(working_vector,4));

      auto encode_l8=[&](id<MTLTexture> scatter){
        const L8Parameters parameters{320,180,0};
        id<MTLCommandBuffer> command=[queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
        [encoder setComputePipelineState:l8];[encoder setTexture:working_vector atIndex:0];
        [encoder setTexture:scatter atIndex:1];
        [encoder setBuffer:rom_buffer offset:kGuardBytes atIndex:0];
        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:1];
        [encoder dispatchThreads:MTLSizeMake(320,180,1)
            threadsPerThreadgroup:MTLSizeMake(16,16,1)];
        [encoder endEncoding];Wait(command);
      };
      encode_l8(scatter_reset);
      checkpoints[5]=Sha256(TextureBytes(scatter_reset,1));

      const L9Parameters l9_parameters{320,180};
      id<MTLCommandBuffer> l9_command=[queue commandBuffer];
      id<MTLComputeCommandEncoder> l9_encoder=[l9_command computeCommandEncoder];
      [l9_encoder setComputePipelineState:l9];[l9_encoder setTexture:reset_feature atIndex:0];
      [l9_encoder setTexture:reset_projection atIndex:1];
      [l9_encoder setBuffer:rom_buffer offset:kGuardBytes atIndex:0];
      [l9_encoder setBytes:&l9_parameters length:sizeof(l9_parameters) atIndex:1];
      [l9_encoder dispatchThreads:MTLSizeMake(320,180,1)
          threadsPerThreadgroup:MTLSizeMake(16,16,1)];
      [l9_encoder endEncoding];Wait(l9_command);
      checkpoints[6]=Sha256(TextureBytes(reset_projection,4));

      auto encode_status=[&](std::uint32_t value){
        const StatusParameters parameters{1,value};
        id<MTLCommandBuffer> command=[queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
        [encoder setComputePipelineState:status];
        [encoder setBuffer:status_buffer offset:kGuardBytes atIndex:0];
        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:1];
        [encoder dispatchThreads:MTLSizeMake(1,1,1)
            threadsPerThreadgroup:MTLSizeMake(1,1,1)];
        [encoder endEncoding];Wait(command);
      };
      encode_status(1);
      checkpoints[7]=Sha256(Payload(status_buffer,4));

      auto reset_present=Texture(device,MTLPixelFormatRGBA8Unorm,640,360,
          MTLTextureUsageShaderWrite,
          std::vector<std::uint8_t>(contract::kSourceBytes,poison),4);
      const L11Parameters l11_parameters{0,0,0,0,640,360};
      id<MTLCommandBuffer> l11_command=[queue commandBuffer];
      id<MTLComputeCommandEncoder> l11_encoder=[l11_command computeCommandEncoder];
      [l11_encoder setComputePipelineState:l11];[l11_encoder setTexture:source_reset atIndex:0];
      [l11_encoder setTexture:reset_present atIndex:1];
      [l11_encoder setBytes:&l11_parameters length:sizeof(l11_parameters) atIndex:0];
      [l11_encoder dispatchThreads:MTLSizeMake(640,360,1)
          threadsPerThreadgroup:MTLSizeMake(16,16,1)];
      [l11_encoder endEncoding];Wait(l11_command);
      checkpoints[8]=Sha256(TextureBytes(reset_present,4));

      encode_status(0);
      checkpoints[9]=Sha256(Payload(status_buffer,4));

      encode_l3(source_generated,history_b,generated_feature,history_a);
      checkpoints[10]=Sha256(Join({TextureBytes(generated_feature,8),
          TextureBytes(shared_metric,2),TextureBytes(history_a,8)}));
      encode_l4(false,vector_generated,scalar_generated,generated_depth);
      checkpoints[11]=Sha256(Join({TextureBytes(flow,8),TextureBytes(generated_depth,4)}));
      encode_l5(generated_depth,generated_mask);
      checkpoints[12]=Sha256(Join({Payload(atomic,contract::kAtomicBytes),
                                   TextureBytes(generated_mask,1)}));
      encode_l6(reset_depth,reset_mask);
      checkpoints[13]=Sha256(Join({TextureBytes(working_vector,4),
          TextureBytes(working_mask,1),Payload(pyramid[0],kPyramidBytes[0]),
          Payload(pyramid[1],kPyramidBytes[1]),Payload(pyramid[2],kPyramidBytes[2]),
          Payload(pyramid[3],kPyramidBytes[3]),Payload(pyramid[4],kPyramidBytes[4])}));
      encode_l7();
      checkpoints[14]=Sha256(TextureBytes(working_vector,4));
      encode_l8(scatter_generated);
      checkpoints[15]=Sha256(TextureBytes(scatter_generated,1));

      const L19Parameters l19_parameters{320,180,160,96,-6};
      auto encode_l19=[&](const std::array<id<MTLTexture>,10>& inputs,
                          id<MTLTexture> vector_output,
                          std::size_t checkpoint_index){
        id<MTLCommandBuffer> command=[queue commandBuffer];
        id<MTLComputeCommandEncoder> surface_encoder=
            [command computeCommandEncoder];
        [surface_encoder setComputePipelineState:l19_surfaces];
        for(NSUInteger i=0;i<inputs.size();++i)
          [surface_encoder setTexture:inputs[i] atIndex:i];
        [surface_encoder setTexture:auxiliary_mask atIndex:10];
        [surface_encoder setTexture:vector_output atIndex:11];
        [surface_encoder setBytes:&l19_parameters length:sizeof(l19_parameters)
                          atIndex:0];
        [surface_encoder dispatchThreads:MTLSizeMake(320,180,1)
                     threadsPerThreadgroup:MTLSizeMake(16,16,1)];
        [surface_encoder endEncoding];
        id<MTLComputeCommandEncoder> tensor_encoder=
            [command computeCommandEncoder];
        [tensor_encoder setComputePipelineState:l19_tensor];
        for(NSUInteger i=0;i<inputs.size();++i)
          [tensor_encoder setTexture:inputs[i] atIndex:i];
        [tensor_encoder setBuffer:model_input offset:kGuardBytes atIndex:0];
        [tensor_encoder setBytes:&l19_parameters length:sizeof(l19_parameters)
                         atIndex:1];
        [tensor_encoder dispatchThreads:MTLSizeMake(160,96,1)
                    threadsPerThreadgroup:MTLSizeMake(16,16,1)];
        [tensor_encoder endEncoding];Wait(command);
        checkpoints[checkpoint_index]=Sha256(Join({
            Payload(model_input,contract::kModelInputBytes),
            TextureBytes(auxiliary_mask,1),TextureBytes(vector_output,4)}));
      };
      const std::array<id<MTLTexture>,10> l19_inputs_frame1={
          reset_feature,generated_feature,scatter_generated,scatter_reset,
          reset_projection,shared_metric,allocation29,flow,working_vector,
          working_mask};
      encode_l19(l19_inputs_frame1,auxiliary_vector,16);
      const auto model_input_frame1_bytes=
          Payload(model_input,contract::kModelInputBytes);
      const auto auxiliary_mask_frame1_bytes=TextureBytes(auxiliary_mask,1);
      const auto auxiliary_vector_frame1_bytes=TextureBytes(auxiliary_vector,4);

#ifdef FG_CONNECTED_FULL_CHAIN
      std::string tail_program="fg-connected-l20-l87-live";
      std::string libs_argument=libs.string();
      std::string model_argument=model.string();
      std::string rom_argument=argv[3];
      std::string sigmoid_argument=argv[4];
      std::string run_argument=argv[kRunArgument];
      std::string poison_argument=argv[kPoisonArgument];
      auto run_tail=[&](const char* frame_name){
        const fs::path tail_output=output_root/frame_name/"l20-l87";
        std::string tail_output_argument=tail_output.string();
        std::string frame_run_argument=run_argument+"-"+frame_name;
        std::array<const char*,8> tail_argv={
            tail_program.c_str(),libs_argument.c_str(),model_argument.c_str(),
            rom_argument.c_str(),sigmoid_argument.c_str(),
            tail_output_argument.c_str(),frame_run_argument.c_str(),
            poison_argument.c_str()};
        FgConnectedSetMacLivePrefixInput(model_input);
        FgConnectedSetMacAllocation29(allocation29);
        const int status=FgConnectedFullTailMain(
            static_cast<int>(tail_argv.size()),tail_argv.data());
        FgConnectedSetMacLivePrefixInput(nil);
        FgConnectedSetMacAllocation29(nil);
        if(status!=0)throw std::runtime_error(
            std::string("connected L20-L87 tail failed: ")+frame_name);
      };
      run_tail("generated-1");
      const auto allocation29_after_frame1=TextureBytes(allocation29,2);

      encode_status(0);
      checkpoints[17]=Sha256(Payload(status_buffer,4));
      encode_l3(source_generated2,history_a,reset_feature,history_b);
      checkpoints[18]=Sha256(Join({TextureBytes(reset_feature,8),
          TextureBytes(shared_metric,2),TextureBytes(history_b,8)}));
      encode_l4(false,vector_generated2,scalar_generated2,reset_depth);
      checkpoints[19]=Sha256(Join({TextureBytes(flow,8),
                                   TextureBytes(reset_depth,4)}));
      encode_l5(reset_depth,reset_mask);
      checkpoints[20]=Sha256(Join({Payload(atomic,contract::kAtomicBytes),
                                   TextureBytes(reset_mask,1)}));
      encode_l6(generated_depth,generated_mask);
      checkpoints[21]=Sha256(Join({TextureBytes(working_vector,4),
          TextureBytes(working_mask,1),Payload(pyramid[0],kPyramidBytes[0]),
          Payload(pyramid[1],kPyramidBytes[1]),Payload(pyramid[2],kPyramidBytes[2]),
          Payload(pyramid[3],kPyramidBytes[3]),Payload(pyramid[4],kPyramidBytes[4])}));
      encode_l7();
      checkpoints[22]=Sha256(TextureBytes(working_vector,4));
      encode_l8(scatter_reset);
      checkpoints[23]=Sha256(TextureBytes(scatter_reset,1));
      const std::array<id<MTLTexture>,10> l19_inputs_frame2={
          generated_feature,reset_feature,scatter_reset,scatter_generated,
          auxiliary_vector,shared_metric,allocation29,flow,working_vector,
          working_mask};
      encode_l19(l19_inputs_frame2,reset_projection,24);
      const auto model_input_frame2_bytes=
          Payload(model_input,contract::kModelInputBytes);
      const auto auxiliary_mask_frame2_bytes=TextureBytes(auxiliary_mask,1);
      const auto auxiliary_vector_frame2_bytes=TextureBytes(reset_projection,4);
      run_tail("generated-2");
      const auto allocation29_after_frame2=TextureBytes(allocation29,2);
      const auto model_input_after_tail=Payload(
          model_input,contract::kModelInputBytes);
#else
      const L20Parameters l20_parameters{160,96,48,32,48u*32u*9u,0};
      id<MTLCommandBuffer> l20_command=[queue commandBuffer];
      id<MTLComputeCommandEncoder> l20_encoder=
          [l20_command computeCommandEncoder];
      [l20_encoder setComputePipelineState:l20];
      [l20_encoder setBuffer:l20_weight offset:kGuardBytes atIndex:0];
      [l20_encoder setBuffer:model_input offset:kGuardBytes atIndex:1];
      [l20_encoder setBuffer:l20_output offset:kGuardBytes atIndex:2];
      [l20_encoder setBytes:&l20_parameters length:sizeof(l20_parameters)
                     atIndex:3];
      [l20_encoder dispatchThreads:MTLSizeMake(160,96,32)
                threadsPerThreadgroup:MTLSizeMake(8,4,1)];
      [l20_encoder endEncoding];Wait(l20_command);
      const auto l20_output_bytes=Payload(l20_output,contract::kL20OutputBytes);
      const auto model_input_after_l20=Payload(model_input,
                                                contract::kModelInputBytes);
      const auto l20_weight_after=Payload(l20_weight,l20_weight_bytes.size());
      checkpoints[17]=Sha256(l20_output_bytes);
#endif

      std::size_t input_mismatches=0;
      input_mismatches+=Mismatches(external.source_reset,TextureBytes(source_reset,4));
      input_mismatches+=Mismatches(external.source_generated,TextureBytes(source_generated,4));
      input_mismatches+=Mismatches(external.source_generated2,TextureBytes(source_generated2,4));
      input_mismatches+=Mismatches(external.vector_reset,TextureBytes(vector_reset,4));
      input_mismatches+=Mismatches(external.vector_generated,TextureBytes(vector_generated,4));
      input_mismatches+=Mismatches(external.vector_generated2,TextureBytes(vector_generated2,4));
      input_mismatches+=Mismatches(external.scalar_reset,TextureBytes(scalar_reset,4));
      input_mismatches+=Mismatches(external.scalar_generated,TextureBytes(scalar_generated,4));
      input_mismatches+=Mismatches(external.scalar_generated2,TextureBytes(scalar_generated2,4));
      input_mismatches+=Mismatches(rom,Payload(rom_buffer,kRomBytes));
#ifdef FG_CONNECTED_FULL_CHAIN
      input_mismatches+=Mismatches(model_input_frame2_bytes,model_input_after_tail);
#else
      input_mismatches+=Mismatches(model_input_bytes,model_input_after_l20);
      input_mismatches+=Mismatches(l20_weight_bytes,l20_weight_after);
#endif
      std::size_t canary_mismatches=transient_canary_mismatches;
      canary_mismatches+=CanaryMismatches(rom_buffer,kRomBytes);
      canary_mismatches+=CanaryMismatches(atomic,contract::kAtomicBytes);
      canary_mismatches+=CanaryMismatches(vector_buffer,contract::kVectorBytes);
      canary_mismatches+=CanaryMismatches(status_buffer,4);
      canary_mismatches+=CanaryMismatches(model_input,contract::kModelInputBytes);
      canary_mismatches+=CanaryMismatches(l20_weight,l20_weight_bytes.size());
#ifndef FG_CONNECTED_FULL_CHAIN
      canary_mismatches+=CanaryMismatches(l20_output,contract::kL20OutputBytes);
#endif
      for(std::size_t i=0;i<pyramid.size();++i)
        canary_mismatches+=CanaryMismatches(pyramid[i],kPyramidBytes[i]);
      std::size_t poison_halfwords=0;
      for(const auto* bytes:{&model_input_frame1_bytes,&model_input_frame2_bytes})
        for(std::size_t i=0;i<bytes->size();i+=2)
          poison_halfwords+=(*bytes)[i]==poison&&(*bytes)[i+1]==poison;
      std::size_t allocation29_poison_halfwords=0;
      for(const auto* bytes:{&allocation29_after_frame1,&allocation29_after_frame2})
        for(std::size_t i=0;i<bytes->size();i+=2)
          allocation29_poison_halfwords+=(*bytes)[i]==poison&&
                                             (*bytes)[i+1]==poison;
#ifndef FG_CONNECTED_FULL_CHAIN
      std::size_t l20_poison_halfwords=0;
      for(std::size_t i=0;i<l20_output_bytes.size();i+=2)
        l20_poison_halfwords+=l20_output_bytes[i]==poison&&
                              l20_output_bytes[i+1]==poison;
#endif
#ifdef FG_CONNECTED_FULL_CHAIN
      const bool pass=input_mismatches==0&&canary_mismatches==0&&
                      poison_halfwords==0&&allocation29_poison_halfwords==0&&
                      allocation29_after_frame1!=contract::InitialAllocation29();
#else
      const bool pass=input_mismatches==0&&canary_mismatches==0&&
                      poison_halfwords==0&&l20_poison_halfwords==0;
#endif
      Write(output_root/"generated-1"/"allocation37.fp16",model_input_frame1_bytes);
      Write(output_root/"generated-1"/"allocation33.r8",auxiliary_mask_frame1_bytes);
      Write(output_root/"generated-1"/"allocation27.rg16f",auxiliary_vector_frame1_bytes);
      Write(output_root/"generated-1"/"allocation29-post-l52.r16f",allocation29_after_frame1);
      Write(output_root/"generated-2"/"allocation37.fp16",model_input_frame2_bytes);
      Write(output_root/"generated-2"/"allocation33.r8",auxiliary_mask_frame2_bytes);
      Write(output_root/"generated-2"/"allocation26.rg16f",auxiliary_vector_frame2_bytes);
      Write(output_root/"generated-2"/"allocation29-post-l52.r16f",allocation29_after_frame2);
#ifndef FG_CONNECTED_FULL_CHAIN
      Write(output_root/"l20-output.fp16",l20_output_bytes);
#endif
      std::ofstream receipt(output_root/"run-receipt.json");
      receipt<<"{\n  \"schema\": \"neuralstudio.fg.connected-temporal-early-run.v1\",\n"
             <<"  \"backend\": \"mac-metal\",\n"
             <<"  \"run_id\": \""<<argv[kRunArgument]<<"\",\n"
             <<"  \"process_id\": "<<getpid()<<",\n"
             <<"  \"poison_byte\": "<<unsigned(poison)<<",\n"
#ifdef FG_CONNECTED_FULL_CHAIN
             <<"  \"phase_launch_counts\": [9, 8, 8],\n"
             <<"  \"launch_count\": 25,\n"
#else
             <<"  \"launch_range\": [3, 20],\n"
             <<"  \"launch_count\": 18,\n"
#endif
             <<"  \"checkpoint_sha256\": {";
      for(std::size_t i=0;i<9;++i)
        receipt<<(i?",":"")<<"\n    \"reset-launch-"<<std::setfill('0')
               <<std::setw(6)<<(3+i)<<"\": \""<<checkpoints[i]<<"\"";
      for(std::size_t i=0;i<8;++i)
        receipt<<",\n    \"generated-1-launch-"<<std::setfill('0')
               <<std::setw(6)<<(12+i)<<"\": \""<<checkpoints[9+i]<<"\"";
      for(std::size_t i=0;i<8;++i)
        receipt<<",\n    \"generated-2-launch-"<<std::setfill('0')
               <<std::setw(6)<<(12+i)<<"\": \""<<checkpoints[17+i]<<"\"";
      receipt<<"\n  },\n  \"allocation29_runtime_source\": \"generated-half-0.5-then-live-L52\",\n"
             <<"  \"allocation29_capture_bytes_consumed\": false,\n"
             <<"  \"input_mismatched_bytes\": "<<input_mismatches<<",\n"
             <<"  \"canary_mismatched_bytes\": "<<canary_mismatches<<",\n"
             <<"  \"allocation37_poison_halfwords\": "<<poison_halfwords<<",\n"
             <<"  \"allocation29_poison_halfwords\": "<<allocation29_poison_halfwords<<",\n"
#ifndef FG_CONNECTED_FULL_CHAIN
             <<"  \"l20_output_poison_halfwords\": "<<l20_poison_halfwords<<",\n"
#endif
             <<"  \"expected_or_archived_intermediate_ingress\": false,\n"
             <<"  \"address_fields_recorded\": false,\n"
             <<"  \"allocation37_frame1_sha256\": \""<<Sha256(model_input_frame1_bytes)<<"\",\n"
             <<"  \"allocation37_frame2_sha256\": \""<<Sha256(model_input_frame2_bytes)<<"\",\n"
             <<"  \"allocation29_frame1_sha256\": \""<<Sha256(allocation29_after_frame1)<<"\",\n"
             <<"  \"allocation29_frame2_sha256\": \""<<Sha256(allocation29_after_frame2)<<"\",\n"
             <<"  \"l20_live_handoff\": true,\n"
#ifdef FG_CONNECTED_FULL_CHAIN
             <<"  \"connected_tail_receipts\": [\"generated-1/l20-l87/run-receipt.json\", \"generated-2/l20-l87/run-receipt.json\"],\n"
#else
             <<"  \"l20_output_bytes\": "<<l20_output_bytes.size()<<",\n"
             <<"  \"l20_output_sha256\": \""<<Sha256(l20_output_bytes)<<"\",\n"
#endif
             <<"  \"result\": \""<<(pass?"pass":"fail")<<"\"\n}\n";
      std::cout<<"allocation37_frame1_sha256="<<Sha256(model_input_frame1_bytes)<<'\n'
               <<"allocation37_frame2_sha256="<<Sha256(model_input_frame2_bytes)<<'\n'
               <<"allocation29_frame1_sha256="<<Sha256(allocation29_after_frame1)<<'\n'
               <<"allocation29_frame2_sha256="<<Sha256(allocation29_after_frame2)<<'\n'
#ifndef FG_CONNECTED_FULL_CHAIN
               <<"l20_output_sha256="<<Sha256(l20_output_bytes)<<'\n'
#endif
               <<"result="<<(pass?"pass":"fail")<<'\n';
      return pass?0:3;
    } catch(const std::exception& error){
      std::cerr<<"error="<<error.what()<<'\n';return 1;
    }
  }
}
