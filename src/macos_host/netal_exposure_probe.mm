#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

struct Resource { uint64_t address; uint32_t width, height, format, channels; };
static_assert(sizeof(Resource)==24);

std::vector<uint16_t> Run(id<MTLDevice> device, NSString* path, const std::vector<uint32_t>& values, bool& intact) {
  NSError* error=nil;
  id<MTLLibrary> library=[device newLibraryWithURL:[NSURL fileURLWithPath:path] error:&error];
  id<MTLFunction> function=[library newFunctionWithName:@"cuda_copy_exposure_kernel"];
  id<MTLComputePipelineState> pipeline=function==nil?nil:[device newComputePipelineStateWithFunction:function error:&error];
  if (pipeline==nil) { std::cerr << "pipeline failure " << (error?error.localizedDescription.UTF8String:"") << '\n'; return {}; }
  const NSUInteger count=values.size(), stride=32;
  id<MTLBuffer> source=[device newBufferWithLength:count*stride options:MTLResourceStorageModeShared];
  id<MTLBuffer> output=[device newBufferWithLength:count*stride options:MTLResourceStorageModeShared];
  id<MTLBuffer> src_desc=[device newBufferWithLength:count*stride options:MTLResourceStorageModeShared];
  id<MTLBuffer> dst_desc=[device newBufferWithLength:count*stride options:MTLResourceStorageModeShared];
  id<MTLBuffer> arguments=[device newBufferWithLength:count*256 options:MTLResourceStorageModeShared];
  if (!source || !output || !src_desc || !dst_desc || !arguments) return {};
  std::memset(source.contents,0xa5,source.length);std::memset(output.contents,0x5a,output.length);
  std::memset(arguments.contents,0,arguments.length);
  for (NSUInteger i=0;i<count;++i) {
    std::memcpy((uint8_t*)source.contents+i*stride+8,&values[i],4);
    const Resource s{source.gpuAddress+i*stride+8,1,1,32,1}, d{output.gpuAddress+i*stride+8,1,1,16,1};
    std::memcpy((uint8_t*)src_desc.contents+i*stride,&s,sizeof(s));
    std::memcpy((uint8_t*)dst_desc.contents+i*stride,&d,sizeof(d));
    const uint64_t args[2]={dst_desc.gpuAddress+i*stride,src_desc.gpuAddress+i*stride};
    std::memcpy((uint8_t*)arguments.contents+i*256,args,sizeof(args));
  }
  id<MTLCommandQueue> queue=[device newCommandQueue];id<MTLCommandBuffer> command=[queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  for (id<MTLBuffer> resource in @[source,src_desc,dst_desc,arguments]) [encoder useResource:resource usage:MTLResourceUsageRead];
  [encoder useResource:output usage:MTLResourceUsageWrite];
  for (NSUInteger i=0;i<count;++i) {
    [encoder setBuffer:arguments offset:i*256 atIndex:0];
    [encoder dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
  }
  [encoder endEncoding];[command commit];[command waitUntilCompleted];
  if (command.status!=MTLCommandBufferStatusCompleted) return {};
  std::vector<uint16_t> result(count);
  for (NSUInteger i=0;i<count;++i) {
    const auto* bytes=(const uint8_t*)output.contents+i*stride;
    std::memcpy(&result[i],bytes+8,2);
    for (NSUInteger j=0;j<stride;++j) if ((j<8 || j>=10) && bytes[j]!=0x5a) intact=false;
  }
  return result;
}

int main(int argc,const char** argv) {
  @autoreleasepool {
    if (argc!=4) return 2;
    std::vector<uint32_t> values={0,0x80000000,0x3f800000,0x3faf5c29,0x401a3d71,0x33000000,0x33800000,0x387fffff,
      0x38800000,0x477fe000,0x477fffff,0x47800000,0x7f7fffff,0xff7fffff,0x7f800000,0xff800000,0x7fc00001,0xffc00001,0x7f800001};
    for (uint32_t exponent=0;exponent<255;++exponent)
      for (uint32_t fraction : {0u,1u,0x3fffffu,0x7fffffu})
        for (uint32_t sign : {0u,0x80000000u}) values.push_back(sign|(exponent<<23)|fraction);
    uint32_t state=0x9e3779b9;
    for (unsigned i=0;i<512;++i) {state=1664525u*state+1013904223u;values.push_back(state);}
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();if (!device) return 3;
    bool intact=true;
    const auto reference=Run(device,@(argv[1]),values,intact), candidate=Run(device,@(argv[2]),values,intact);
    if (reference.size()!=values.size() || candidate.size()!=values.size()) return 4;
    std::ofstream report(argv[3]);unsigned mismatches=0,finite_mismatches=0;
    report << "input_bits,reference_half,candidate_half\n";
    for (size_t i=0;i<values.size();++i) {
      if (reference[i]!=candidate[i]) {++mismatches;if ((values[i]&0x7f800000)!=0x7f800000)++finite_mismatches;}
      report << values[i] << ',' << reference[i] << ',' << candidate[i] << '\n';
    }
    std::cout << "cases=" << values.size() << " mismatches=" << mismatches << " finite_mismatches=" << finite_mismatches << " canaries=" << intact << '\n';
    return mismatches==0 && intact?0:9;
  }
}
