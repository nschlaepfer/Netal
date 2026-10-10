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
#include "../fg_distortion_path/connected_contract.h"

#ifdef FG_CONNECTED_FULL_CHAIN
#include "macos_live_handoff.h"
int FgConnectedFullTailMain(int argc, const char* argv[]);
#endif

namespace fs = std::filesystem;

namespace {

namespace contract = fg_connected_distortion_l3_l87;
namespace distortion = fg_distortion_connected;
constexpr std::size_t kGuardBytes = 256;
constexpr std::size_t kRomBytes = 2097152;
constexpr std::array<std::size_t,5> kPyramidBytes = {
    57600,86400,21600,5520,1440};

struct L3Parameters { std::uint32_t width,height,half_width,half_height; };
struct L4Parameters { std::uint32_t reset_mode,input_width,input_height,
                                    output_width,output_height; };
struct Distortion10Parameters {
  std::uint32_t input_origin[2],input_extent[2];
  float inverse_input_extent[2];
  std::uint32_t output_extent[2];
  float output_bias,output_scale;
};
struct Distortion9Parameters { std::uint32_t width,height; };
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
    if(argc!=10){std::cerr<<"usage=fg-connected-distortion-l3-l91-mac <libs> <model> <rom> <sigmoid> <ex2-unit> <ex2-fraction> <output> <run-id> <poison>\n";return 2;}
#else
    if(argc!=7){std::cerr<<"usage=fg-connected-l3-l19-mac <libs> <model> <rom> <output> <run-id> <poison>\n";return 2;}
#endif
    try {
#ifdef FG_CONNECTED_FULL_CHAIN
      const fs::path libs=argv[1],model=argv[2],output_root=argv[7];
      constexpr int kRunArgument=8;
      constexpr int kPoisonArgument=9;
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
      const auto ex2_unit_bytes=ReadFile(argv[5]);
      const auto ex2_fraction_bytes=ReadFile(argv[6]);
      if(ex2_unit_bytes.size()!=67108868u||Sha256(ex2_unit_bytes)!=
         "40088ff59b58c4018114b4a2dd1fb5b124488b91e7cad79bdf3bf143bf267b75")
        throw std::runtime_error("EX2 unit table identity mismatch");
      if(ex2_fraction_bytes.size()!=33554436u||Sha256(ex2_fraction_bytes)!=
         "f7cfb2fd9df97ae01bc747b866e4928a79311e8443700f938859f43772f73d86")
        throw std::runtime_error("EX2 fraction table identity mismatch");
      const auto external=contract::Generate();
      const auto distortion_reset_input=distortion::Distortion(0);
      const auto distortion_previous_input=distortion::Distortion(1);
      const auto distortion_generated_input=distortion::Distortion(2);
      id<MTLDevice> device=MTLCreateSystemDefaultDevice();
      id<MTLCommandQueue> queue=[device newCommandQueue];
      if(device==nil||queue==nil)throw std::runtime_error("Metal unavailable");
      auto l3=Pipeline(device,libs/"l3_l13.metallib","fg_early_l3_l13_mode2");
      auto l4=Pipeline(device,libs/"l4_l14.metallib","fg_early_l4_l14");
      auto distortion10=Pipeline(device,libs/"distortion10.metallib",
                                 "fg_distortion_module10");
      auto distortion9=Pipeline(device,libs/"distortion9.metallib",
                                "fg_connected_distortion_module9");
      auto l5_high=Pipeline(device,libs/"l5_l15.metallib","fg_early_l5_l15_high");
      auto l5_low=Pipeline(device,libs/"l5_l15.metallib","fg_early_l5_l15_low");
      auto l5_pack=Pipeline(device,libs/"l5_l15.metallib","fg_early_l5_l15_pack");
      auto l6=Pipeline(device,libs/"l6_l16.metallib","fg_early_l6_l16");
      auto l7=Pipeline(device,libs/"l7_l17.metallib","fg_early_l7_l17");
      auto l8=Pipeline(device,libs/"l8_l18.metallib","fg_early_l8_l18");
      auto l9=Pipeline(device,libs/"l9.metallib","fg_early_l9");
      auto l11=Pipeline(device,libs/"l11.metallib","fg_early_l11");
      auto l19_surfaces=Pipeline(device,libs/"l19.metallib","fg_early_l19_surfaces");
      auto l19_tensor=Pipeline(device,libs/"l19.metallib","fg_early_l19_tensor");
#ifndef FG_CONNECTED_FULL_CHAIN
      auto l20=Pipeline(device,libs/"l20.metallib","fg_conv_3x3");
#endif
      id<MTLBuffer> rom_buffer=Buffer(device,rom);
      id<MTLBuffer> l20_weight=Buffer(device,l20_weight_bytes);
      id<MTLBuffer> ex2_unit_buffer=Buffer(device,ex2_unit_bytes);
      id<MTLBuffer> ex2_fraction_buffer=Buffer(device,ex2_fraction_bytes);

      const auto zeros_history=contract::InitialHistory();
      auto source_reset=Texture(device,MTLPixelFormatRGBA8Unorm,640,360,
          MTLTextureUsageShaderRead,external.source_reset,4);
      auto source_generated=Texture(device,MTLPixelFormatRGBA8Unorm,640,360,
          MTLTextureUsageShaderRead,external.source_generated,4);
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
      auto scalar_reset=Texture(device,MTLPixelFormatR32Float,640,360,
          MTLTextureUsageShaderRead,external.scalar_reset,4);
      auto scalar_generated=Texture(device,MTLPixelFormatR32Float,640,360,
          MTLTextureUsageShaderRead,external.scalar_generated,4);
      constexpr MTLTextureUsage distortion_read_write=
          MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite;
      auto distortion_reset_source=Texture(
          device,MTLPixelFormatRGBA16Float,320,180,MTLTextureUsageShaderRead,
          distortion_reset_input,8);
      auto distortion_generated_source=Texture(
          device,MTLPixelFormatRGBA16Float,320,180,MTLTextureUsageShaderRead,
          distortion_generated_input,8);
      auto distortion_internal_reset=Texture(
          device,MTLPixelFormatRGBA16Float,320,180,distortion_read_write,
          std::vector<std::uint8_t>(distortion::kRgba16Bytes,poison),8);
      auto distortion_internal_generated=Texture(
          device,MTLPixelFormatRGBA16Float,320,180,distortion_read_write,
          distortion_previous_input,8);
      auto distortion_flow_a=Texture(
          device,MTLPixelFormatRG16Float,320,180,distortion_read_write,
          std::vector<std::uint8_t>(distortion::kRg16Bytes,poison),4);
      auto distortion_flow_b=Texture(
          device,MTLPixelFormatRG16Float,320,180,distortion_read_write,
          std::vector<std::uint8_t>(distortion::kRg16Bytes,poison),4);
      auto distortion_depth_reset_out=Texture(
          device,MTLPixelFormatR32Uint,320,180,distortion_read_write,
          std::vector<std::uint8_t>(distortion::kR32Bytes,poison),4);
      auto distortion_scatter_reset=Texture(
          device,MTLPixelFormatR8Unorm,320,180,distortion_read_write,
          std::vector<std::uint8_t>(distortion::kR8Bytes,255),1);
      auto distortion_scatter_generated=Texture(
          device,MTLPixelFormatR8Unorm,320,180,distortion_read_write,
          std::vector<std::uint8_t>(distortion::kR8Bytes,255),1);
      auto distortion_canary_rgba=Texture(
          device,MTLPixelFormatRGBA16Float,320,180,MTLTextureUsageShaderWrite,
          std::vector<std::uint8_t>(distortion::kRgba16Bytes,poison^0xffu),8);
      auto distortion_canary_rg=Texture(
          device,MTLPixelFormatRG16Float,320,180,MTLTextureUsageShaderWrite,
          std::vector<std::uint8_t>(distortion::kRg16Bytes,poison^0xffu),4);
      auto distortion_canary_r32=Texture(
          device,MTLPixelFormatR32Uint,320,180,MTLTextureUsageShaderWrite,
          std::vector<std::uint8_t>(distortion::kR32Bytes,poison^0xffu),4);
      auto distortion_canary_r8=Texture(
          device,MTLPixelFormatR8Unorm,320,180,MTLTextureUsageShaderWrite,
          std::vector<std::uint8_t>(distortion::kR8Bytes,poison^0xffu),1);
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
      std::array<std::string,21> checkpoints{};
      auto encode_distortion10=[&](id<MTLTexture> input,
                                    id<MTLTexture> output){
        const Distortion10Parameters parameters{
            {0,0},{320,180},
            {std::bit_cast<float>(0x3b4ccccdu),
             std::bit_cast<float>(0x3bb60b61u)},
            {320,180},0.0f,1.0f};
        id<MTLCommandBuffer> command=[queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
        [encoder setComputePipelineState:distortion10];
        [encoder setTexture:input atIndex:0];
        [encoder setTexture:output atIndex:1];
        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:0];
        [encoder dispatchThreads:MTLSizeMake(320,180,1)
            threadsPerThreadgroup:MTLSizeMake(16,16,1)];
        [encoder endEncoding];Wait(command);
      };
      auto encode_distortion9=[&](id<MTLTexture> a,id<MTLTexture> b,
                                   id<MTLTexture> depth,
                                   id<MTLTexture> depth_output){
        const Distortion9Parameters parameters{320,180};
        id<MTLCommandBuffer> command=[queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
        [encoder setComputePipelineState:distortion9];
        [encoder setTexture:a atIndex:0];[encoder setTexture:b atIndex:1];
        [encoder setTexture:flow atIndex:2];
        [encoder setTexture:working_vector atIndex:3];
        [encoder setTexture:depth atIndex:4];
        [encoder setTexture:distortion_flow_a atIndex:5];
        [encoder setTexture:distortion_flow_b atIndex:6];
        [encoder setTexture:depth_output atIndex:7];
        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:0];
        [encoder setBuffer:rom_buffer offset:kGuardBytes atIndex:1];
        [encoder dispatchThreads:MTLSizeMake(320,180,1)
            threadsPerThreadgroup:MTLSizeMake(16,16,1)];
        [encoder endEncoding];Wait(command);
      };
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
      const auto flow_reset_bytes=TextureBytes(flow,8);
      const auto reset_depth_bytes=TextureBytes(reset_depth,4);
      checkpoints[1]=Sha256(Join({flow_reset_bytes,reset_depth_bytes}));
      encode_distortion10(distortion_reset_source,distortion_internal_reset);
      const auto distortion_internal_reset_bytes=
          TextureBytes(distortion_internal_reset,8);
      checkpoints[2]=Sha256(distortion_internal_reset_bytes);

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
      checkpoints[3]=Sha256(Join({Payload(atomic,contract::kAtomicBytes),
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
      checkpoints[4]=Sha256(Join({TextureBytes(working_vector,4),
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
      const auto working_vector_l7_bytes=TextureBytes(working_vector,4);
      checkpoints[5]=Sha256(working_vector_l7_bytes);

      auto encode_l8=[&](id<MTLTexture> input,id<MTLTexture> scatter){
        const L8Parameters parameters{320,180,0};
        id<MTLCommandBuffer> command=[queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
        [encoder setComputePipelineState:l8];[encoder setTexture:input atIndex:0];
        [encoder setTexture:scatter atIndex:1];
        [encoder setBuffer:rom_buffer offset:kGuardBytes atIndex:0];
        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:1];
        [encoder dispatchThreads:MTLSizeMake(320,180,1)
            threadsPerThreadgroup:MTLSizeMake(16,16,1)];
        [encoder endEncoding];Wait(command);
      };
      encode_distortion9(distortion_internal_reset,
                         distortion_internal_generated,reset_depth,
                         distortion_depth_reset_out);
      const auto distortion_flow_a_reset_bytes=TextureBytes(distortion_flow_a,4);
      const auto distortion_flow_b_reset_bytes=TextureBytes(distortion_flow_b,4);
      const auto distortion_depth_reset_bytes=
          TextureBytes(distortion_depth_reset_out,4);
      checkpoints[6]=Sha256(Join({distortion_flow_a_reset_bytes,
          distortion_flow_b_reset_bytes,distortion_depth_reset_bytes}));
      encode_l8(distortion_flow_b,distortion_scatter_reset);
      checkpoints[7]=Sha256(TextureBytes(distortion_scatter_reset,1));
      encode_l8(distortion_flow_a,scatter_reset);
      checkpoints[8]=Sha256(TextureBytes(scatter_reset,1));

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
      checkpoints[9]=Sha256(TextureBytes(reset_projection,4));

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
      checkpoints[10]=Sha256(TextureBytes(reset_present,4));

      encode_l3(source_generated,history_b,generated_feature,history_a);
      checkpoints[11]=Sha256(Join({TextureBytes(generated_feature,8),
          TextureBytes(shared_metric,2),TextureBytes(history_a,8)}));
      encode_l4(false,vector_generated,scalar_generated,generated_depth);
      const auto flow_generated_bytes=TextureBytes(flow,8);
      const auto generated_depth_bytes=TextureBytes(generated_depth,4);
      checkpoints[12]=Sha256(Join({flow_generated_bytes,generated_depth_bytes}));
      encode_distortion10(distortion_generated_source,
                          distortion_internal_generated);
      const auto distortion_internal_generated_bytes=
          TextureBytes(distortion_internal_generated,8);
      checkpoints[13]=Sha256(distortion_internal_generated_bytes);
      encode_l5(generated_depth,generated_mask);
      checkpoints[14]=Sha256(Join({Payload(atomic,contract::kAtomicBytes),
                                   TextureBytes(generated_mask,1)}));
      encode_l6(reset_depth,reset_mask);
      checkpoints[15]=Sha256(Join({TextureBytes(working_vector,4),
          TextureBytes(working_mask,1),Payload(pyramid[0],kPyramidBytes[0]),
          Payload(pyramid[1],kPyramidBytes[1]),Payload(pyramid[2],kPyramidBytes[2]),
          Payload(pyramid[3],kPyramidBytes[3]),Payload(pyramid[4],kPyramidBytes[4])}));
      encode_l7();
      const auto working_vector_l17_bytes=TextureBytes(working_vector,4);
      checkpoints[16]=Sha256(working_vector_l17_bytes);
      encode_distortion9(distortion_internal_generated,
                         distortion_internal_reset,generated_depth,
                         reset_depth);
      const auto distortion_flow_a_generated_bytes=TextureBytes(distortion_flow_a,4);
      const auto distortion_flow_b_generated_bytes=TextureBytes(distortion_flow_b,4);
      const auto distortion_depth_generated_bytes=TextureBytes(reset_depth,4);
      checkpoints[17]=Sha256(Join({distortion_flow_a_generated_bytes,
          distortion_flow_b_generated_bytes,distortion_depth_generated_bytes}));
      encode_l8(distortion_flow_b,distortion_scatter_generated);
      checkpoints[18]=Sha256(TextureBytes(distortion_scatter_generated,1));
      encode_l8(distortion_flow_a,scatter_generated);
      checkpoints[19]=Sha256(TextureBytes(scatter_generated,1));

      const L19Parameters l19_parameters{320,180,160,96,-6};
      const std::array<id<MTLTexture>,10> l19_inputs={
          reset_feature,generated_feature,distortion_scatter_generated,
          distortion_scatter_reset,reset_projection,shared_metric,allocation29,
          distortion_flow_a,distortion_flow_b,scatter_generated};
      id<MTLCommandBuffer> l19_command=[queue commandBuffer];
      id<MTLComputeCommandEncoder> surface_encoder=
          [l19_command computeCommandEncoder];
      [surface_encoder setComputePipelineState:l19_surfaces];
      for(NSUInteger i=0;i<l19_inputs.size();++i)
        [surface_encoder setTexture:l19_inputs[i] atIndex:i];
      [surface_encoder setTexture:auxiliary_mask atIndex:10];
      [surface_encoder setTexture:auxiliary_vector atIndex:11];
      [surface_encoder setBytes:&l19_parameters length:sizeof(l19_parameters)
                        atIndex:0];
      [surface_encoder dispatchThreads:MTLSizeMake(320,180,1)
                   threadsPerThreadgroup:MTLSizeMake(16,16,1)];
      [surface_encoder endEncoding];
      id<MTLComputeCommandEncoder> tensor_encoder=
          [l19_command computeCommandEncoder];
      [tensor_encoder setComputePipelineState:l19_tensor];
      for(NSUInteger i=0;i<l19_inputs.size();++i)
        [tensor_encoder setTexture:l19_inputs[i] atIndex:i];
      [tensor_encoder setBuffer:model_input offset:kGuardBytes atIndex:0];
      [tensor_encoder setBytes:&l19_parameters length:sizeof(l19_parameters)
                       atIndex:1];
      [tensor_encoder dispatchThreads:MTLSizeMake(160,96,1)
                  threadsPerThreadgroup:MTLSizeMake(16,16,1)];
      [tensor_encoder endEncoding];Wait(l19_command);
      const auto model_input_bytes=Payload(model_input,contract::kModelInputBytes);
      const auto auxiliary_mask_bytes=TextureBytes(auxiliary_mask,1);
      const auto auxiliary_vector_bytes=TextureBytes(auxiliary_vector,4);
      checkpoints[20]=Sha256(Join({model_input_bytes,auxiliary_mask_bytes,
                                   auxiliary_vector_bytes}));

#ifdef FG_CONNECTED_FULL_CHAIN
      const fs::path tail_output=output_root/"l20-l87";
      FgConnectedSetMacLivePrefixInput(model_input);
      std::string tail_program="fg-connected-l20-l87-live";
      std::string libs_argument=libs.string();
      std::string model_argument=model.string();
      std::string rom_argument=argv[3];
      std::string sigmoid_argument=argv[4];
      std::string tail_output_argument=tail_output.string();
      std::string run_argument=argv[kRunArgument];
      std::string poison_argument=argv[kPoisonArgument];
      std::array<const char*,8> tail_argv={
          tail_program.c_str(),libs_argument.c_str(),model_argument.c_str(),
          rom_argument.c_str(),sigmoid_argument.c_str(),
          tail_output_argument.c_str(),run_argument.c_str(),
          poison_argument.c_str()};
      FgConnectedSetMacEx2Tables(ex2_unit_buffer,ex2_fraction_buffer);
      const int tail_status=FgConnectedFullTailMain(
          static_cast<int>(tail_argv.size()),tail_argv.data());
      FgConnectedSetMacEx2Tables(nil,nil);
      FgConnectedSetMacLivePrefixInput(nil);
      if(tail_status!=0)throw std::runtime_error("connected L20-L87 tail failed");
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
      input_mismatches+=Mismatches(external.vector_reset,TextureBytes(vector_reset,4));
      input_mismatches+=Mismatches(external.vector_generated,TextureBytes(vector_generated,4));
      input_mismatches+=Mismatches(external.scalar_reset,TextureBytes(scalar_reset,4));
      input_mismatches+=Mismatches(external.scalar_generated,TextureBytes(scalar_generated,4));
      input_mismatches+=Mismatches(distortion_reset_input,
                                   TextureBytes(distortion_reset_source,8));
      input_mismatches+=Mismatches(distortion_generated_input,
                                   TextureBytes(distortion_generated_source,8));
      input_mismatches+=Mismatches(contract::InitialAllocation29(),
                                   TextureBytes(allocation29,2));
      input_mismatches+=Mismatches(rom,Payload(rom_buffer,kRomBytes));
      input_mismatches+=Mismatches(ex2_unit_bytes,
                                   Payload(ex2_unit_buffer,ex2_unit_bytes.size()));
      input_mismatches+=Mismatches(
          ex2_fraction_bytes,Payload(ex2_fraction_buffer,ex2_fraction_bytes.size()));
      const std::size_t distortion_handoff_mismatches=
          Mismatches(distortion_reset_input,distortion_internal_reset_bytes)+
          Mismatches(distortion_generated_input,
                     distortion_internal_generated_bytes);
#ifdef FG_CONNECTED_FULL_CHAIN
      input_mismatches+=Mismatches(model_input_bytes,model_input_after_tail);
#else
      input_mismatches+=Mismatches(model_input_bytes,model_input_after_l20);
      input_mismatches+=Mismatches(l20_weight_bytes,l20_weight_after);
#endif
      std::size_t canary_mismatches=transient_canary_mismatches;
      canary_mismatches+=CanaryMismatches(rom_buffer,kRomBytes);
      canary_mismatches+=CanaryMismatches(ex2_unit_buffer,ex2_unit_bytes.size());
      canary_mismatches+=CanaryMismatches(ex2_fraction_buffer,
                                          ex2_fraction_bytes.size());
      canary_mismatches+=CanaryMismatches(atomic,contract::kAtomicBytes);
      canary_mismatches+=CanaryMismatches(vector_buffer,contract::kVectorBytes);
      canary_mismatches+=CanaryMismatches(model_input,contract::kModelInputBytes);
      canary_mismatches+=CanaryMismatches(l20_weight,l20_weight_bytes.size());
      canary_mismatches+=Mismatches(
          std::vector<std::uint8_t>(distortion::kRgba16Bytes,poison^0xffu),
          TextureBytes(distortion_canary_rgba,8));
      canary_mismatches+=Mismatches(
          std::vector<std::uint8_t>(distortion::kRg16Bytes,poison^0xffu),
          TextureBytes(distortion_canary_rg,4));
      canary_mismatches+=Mismatches(
          std::vector<std::uint8_t>(distortion::kR32Bytes,poison^0xffu),
          TextureBytes(distortion_canary_r32,4));
      canary_mismatches+=Mismatches(
          std::vector<std::uint8_t>(distortion::kR8Bytes,poison^0xffu),
          TextureBytes(distortion_canary_r8,1));
#ifndef FG_CONNECTED_FULL_CHAIN
      canary_mismatches+=CanaryMismatches(l20_output,contract::kL20OutputBytes);
#endif
      for(std::size_t i=0;i<pyramid.size();++i)
        canary_mismatches+=CanaryMismatches(pyramid[i],kPyramidBytes[i]);
      std::size_t poison_halfwords=0;
      for(std::size_t i=0;i<model_input_bytes.size();i+=2)
        poison_halfwords+=model_input_bytes[i]==poison&&
                          model_input_bytes[i+1]==poison;
#ifndef FG_CONNECTED_FULL_CHAIN
      std::size_t l20_poison_halfwords=0;
      for(std::size_t i=0;i<l20_output_bytes.size();i+=2)
        l20_poison_halfwords+=l20_output_bytes[i]==poison&&
                              l20_output_bytes[i+1]==poison;
#endif
#ifdef FG_CONNECTED_FULL_CHAIN
      const bool pass=input_mismatches==0&&distortion_handoff_mismatches==0&&
                      canary_mismatches==0&&
                      poison_halfwords==0;
#else
      const bool pass=input_mismatches==0&&canary_mismatches==0&&
                      poison_halfwords==0&&l20_poison_halfwords==0;
#endif
      Write(output_root/"allocation37.fp16",model_input_bytes);
      Write(output_root/"distortion-reset-a.rg16f",distortion_flow_a_reset_bytes);
      Write(output_root/"distortion-reset-b.rg16f",distortion_flow_b_reset_bytes);
      Write(output_root/"distortion-reset-depth.r32u",distortion_depth_reset_bytes);
      Write(output_root/"distortion-generated-a.rg16f",distortion_flow_a_generated_bytes);
      Write(output_root/"distortion-generated-b.rg16f",distortion_flow_b_generated_bytes);
      Write(output_root/"distortion-generated-depth.r32u",distortion_depth_generated_bytes);
      Write(output_root/"distortion-reset-input-flow.rg32f",flow_reset_bytes);
      Write(output_root/"distortion-reset-input-vector.rg16f",working_vector_l7_bytes);
      Write(output_root/"distortion-reset-input-depth.r32u",reset_depth_bytes);
      Write(output_root/"distortion-generated-input-flow.rg32f",flow_generated_bytes);
      Write(output_root/"distortion-generated-input-vector.rg16f",working_vector_l17_bytes);
      Write(output_root/"distortion-generated-input-depth.r32u",generated_depth_bytes);
      Write(output_root/"allocation33.r8",auxiliary_mask_bytes);
      Write(output_root/"allocation27.rg16f",auxiliary_vector_bytes);
#ifndef FG_CONNECTED_FULL_CHAIN
      Write(output_root/"l20-output.fp16",l20_output_bytes);
#endif
      std::ofstream receipt(output_root/"run-receipt.json");
      receipt<<"{\n  \"schema\": \"neuralstudio.fg.connected-distortion-l3-l23-run.v1\",\n"
             <<"  \"backend\": \"mac-metal\",\n"
             <<"  \"run_id\": \""<<argv[kRunArgument]<<"\",\n"
             <<"  \"process_id\": "<<getpid()<<",\n"
             <<"  \"poison_byte\": "<<unsigned(poison)<<",\n"
#ifdef FG_CONNECTED_FULL_CHAIN
             <<"  \"launch_range\": [3, 23],\n"
             <<"  \"launch_count\": 21,\n"
#else
             <<"  \"launch_range\": [3, 20],\n"
             <<"  \"launch_count\": 18,\n"
#endif
             <<"  \"checkpoint_sha256\": {";
      for(std::size_t i=0;i<checkpoints.size();++i)
        receipt<<(i?",":"")<<"\n    \"launch-"<<std::setfill('0')
               <<std::setw(6)<<(3+i)<<"\": \""<<checkpoints[i]<<"\"";
      receipt<<"\n  },\n  \"allocation29_runtime_source\": \"generated-half-0.5\",\n"
             <<"  \"distortion_component_sha256\": {\n"
             <<"    \"reset\": [\""<<Sha256(distortion_flow_a_reset_bytes)<<"\", \""
             <<Sha256(distortion_flow_b_reset_bytes)<<"\", \""
             <<Sha256(distortion_depth_reset_bytes)<<"\"],\n"
             <<"    \"generated\": [\""<<Sha256(distortion_flow_a_generated_bytes)<<"\", \""
             <<Sha256(distortion_flow_b_generated_bytes)<<"\", \""
             <<Sha256(distortion_depth_generated_bytes)<<"\"]\n  },\n"
             <<"  \"allocation29_capture_bytes_consumed\": false,\n"
             <<"  \"input_mismatched_bytes\": "<<input_mismatches<<",\n"
             <<"  \"distortion_handoff_mismatched_bytes\": "
             <<distortion_handoff_mismatches<<",\n"
             <<"  \"canary_mismatched_bytes\": "<<canary_mismatches<<",\n"
             <<"  \"allocation37_poison_halfwords\": "<<poison_halfwords<<",\n"
#ifndef FG_CONNECTED_FULL_CHAIN
             <<"  \"l20_output_poison_halfwords\": "<<l20_poison_halfwords<<",\n"
#endif
             <<"  \"expected_or_archived_intermediate_ingress\": false,\n"
             <<"  \"distortion_live_device_handoffs\": true,\n"
             <<"  \"host_distortion_checkpoint_ingress\": false,\n"
             <<"  \"address_fields_recorded\": false,\n"
             <<"  \"allocation37_bytes\": "<<model_input_bytes.size()<<",\n"
             <<"  \"allocation37_sha256\": \""<<Sha256(model_input_bytes)<<"\",\n"
             <<"  \"l20_live_handoff\": true,\n"
#ifdef FG_CONNECTED_FULL_CHAIN
             <<"  \"connected_tail_receipt\": \"l20-l87/run-receipt.json\",\n"
#else
             <<"  \"l20_output_bytes\": "<<l20_output_bytes.size()<<",\n"
             <<"  \"l20_output_sha256\": \""<<Sha256(l20_output_bytes)<<"\",\n"
#endif
             <<"  \"result\": \""<<(pass?"pass":"fail")<<"\"\n}\n";
      std::cout<<"allocation37_sha256="<<Sha256(model_input_bytes)<<'\n'
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
