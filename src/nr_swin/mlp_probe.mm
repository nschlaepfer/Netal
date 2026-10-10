#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <CommonCrypto/CommonDigest.h>
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <unistd.h>

namespace fs=std::filesystem;
constexpr size_t Guard=256;
struct Buffer { id<MTLBuffer> metal;size_t bytes; };
std::vector<uint8_t> Read(const fs::path& path) {
  std::ifstream stream(path,std::ios::binary|std::ios::ate);
  if(!stream)throw std::runtime_error("input open failed");
  auto size=stream.tellg();stream.seekg(0);
  std::vector<uint8_t> result(static_cast<size_t>(size));
  stream.read(reinterpret_cast<char*>(result.data()),size);
  if(!stream)throw std::runtime_error("input read failed");return result;
}
std::string Sha(const void* data,size_t bytes) {
  std::array<unsigned char,32> hash{};CC_SHA256(data,static_cast<CC_LONG>(bytes),hash.data());
  std::ostringstream out;out<<std::hex<<std::setfill('0');
  for(auto v:hash)out<<std::setw(2)<<unsigned(v);return out.str();
}
uint8_t* Data(Buffer& b){return static_cast<uint8_t*>(b.metal.contents)+Guard;}
Buffer Make(id<MTLDevice> dev,size_t bytes,uint8_t poison) {
  Buffer b{[dev newBufferWithLength:bytes+2*Guard options:MTLResourceStorageModeShared],bytes};
  if(!b.metal)throw std::runtime_error("buffer allocation failed");
  std::memset(b.metal.contents,0xc3,bytes+2*Guard);std::memset(Data(b),poison,bytes);return b;
}
size_t Canary(Buffer& b) {
  auto* p=static_cast<uint8_t*>(b.metal.contents);size_t errors=0;
  for(size_t i=0;i<Guard;++i)errors+=(p[i]!=0xc3)+(p[Guard+b.bytes+i]!=0xc3);
  return errors;
}
void Write(const fs::path& p,Buffer& b) {
  std::ofstream f(p,std::ios::binary);f.write(reinterpret_cast<char*>(Data(b)),b.bytes);
  if(!f)throw std::runtime_error("output write failed");
}

int main(int argc,const char** argv) {
  @autoreleasepool {try {
    if(argc!=7)throw std::runtime_error("usage: mlp_probe LIB INPUT WEIGHT ROWS POISON OUTPUT");
    const uint32_t rows=static_cast<uint32_t>(std::stoul(argv[4]));
    const uint32_t poison=static_cast<uint32_t>(std::stoul(argv[5],nullptr,16));
    if(rows==0||rows>65536||poison>255)throw std::runtime_error("invalid dimensions/poison");
    fs::path output(argv[6]);if(fs::exists(output))throw std::runtime_error("preserve previous run");
    const auto input=Read(argv[2]),weights=Read(argv[3]);
    if(input.size()!=size_t(rows)*32||weights.size()!=20672)throw std::runtime_error("input extent mismatch");
    if(Sha(weights.data(),weights.size())!="0270fb8f406d3d0a916953ddaf6ec5d29328fe95b9d6cb408ca0c25c91108e3a")
      throw std::runtime_error("pinned weight mismatch");
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();NSError* error=nil;
    id<MTLLibrary> lib=[device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&error];
    if(!lib)throw std::runtime_error(error.localizedDescription.UTF8String);
    id<MTLCommandQueue> queue=[device newCommandQueue];
    std::array<id<MTLComputePipelineState>,3> pipelines;
    NSArray<NSString*>* names=@[@"nr_swin32_expand",@"nr_swin32_activate",@"nr_swin32_contract"];
    for(size_t i=0;i<3;++i) {
      pipelines[i]=[device newComputePipelineStateWithFunction:[lib newFunctionWithName:names[i]] error:&error];
      if(!pipelines[i])throw std::runtime_error(error.localizedDescription.UTF8String);
    }
    Buffer x=Make(device,input.size(),0),w=Make(device,weights.size(),0);
    std::memcpy(Data(x),input.data(),input.size());std::memcpy(Data(w),weights.data(),weights.size());
    std::array<Buffer,3> y={Make(device,size_t(rows)*128*2,poison),Make(device,size_t(rows)*128,poison),Make(device,size_t(rows)*128*2,poison)};
    std::array<Buffer,3> counts={Make(device,size_t(rows)*128*4,0),Make(device,size_t(rows)*128*4,0),Make(device,size_t(rows)*128*4,0)};
    id<MTLCommandBuffer> command=[queue commandBuffer];
    for(size_t i=0;i<3;++i) {
      id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
      [encoder setComputePipelineState:pipelines[i]];
      auto bind=[&](Buffer& b,NSUInteger index){[encoder setBuffer:b.metal offset:Guard atIndex:index];};
      if(i==0){bind(x,0);bind(w,1);bind(y[0],2);bind(counts[0],3);[encoder setBytes:&rows length:4 atIndex:4];}
      if(i==1){bind(y[0],0);bind(y[1],1);bind(counts[1],2);[encoder setBytes:&rows length:4 atIndex:3];}
      if(i==2){bind(x,0);bind(y[1],1);bind(w,2);bind(y[2],3);bind(counts[2],4);[encoder setBytes:&rows length:4 atIndex:5];}
      [encoder dispatchThreads:MTLSizeMake(size_t(rows)*(i==2?32:128),1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
      [encoder endEncoding];
    }
    [command commit];[command waitUntilCompleted];
    if(command.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error(command.error.localizedDescription.UTF8String);
    size_t coverage=0,canary=Canary(x)+Canary(w),immutable=0;
    for(size_t i=0;i<input.size();++i)immutable+=input[i]!=Data(x)[i];
    for(size_t i=0;i<weights.size();++i)immutable+=weights[i]!=Data(w)[i];
    for(size_t i=0;i<3;++i){canary+=Canary(y[i])+Canary(counts[i]);auto* c=reinterpret_cast<uint32_t*>(Data(counts[i]));for(size_t j=0;j<size_t(rows)*128;++j)coverage+=c[j]!=1;}
    fs::create_directories(output);
    const std::array<const char*,3> files={"expansion.f16","activation.e4m3","partial.f16"};
    for(size_t i=0;i<3;++i)Write(output/files[i],y[i]);
    std::ofstream receipt(output/"run.json");
    receipt<<"{\n\"schema\":\"neuralstudio.nr.swin32-mlp-metal.v1\",\n\"pid\":"<<getpid()<<",\n\"rows\":"<<rows<<",\n\"poison\":"<<poison<<",\n"
      <<"\"input_sha256\":\""<<Sha(input.data(),input.size())<<"\",\n\"weight_sha256\":\""<<Sha(weights.data(),weights.size())<<"\",\n"
      <<"\"expected_file_opened\":false,\n\"dispatch_count\":3,\n\"outputs\":{";
    for(size_t i=0;i<3;++i)receipt<<(i?",":"")<<"\""<<files[i]<<"\":\""<<Sha(Data(y[i]),y[i].bytes)<<"\"";
    receipt<<"},\n\"coverage_errors\":"<<coverage<<",\n\"canary_errors\":"<<canary<<",\n\"immutable_errors\":"<<immutable<<",\n\"result\":\""<<((coverage|canary|immutable)?"fail":"pass")<<"\"\n}\n";
    std::cout<<"coverage_errors="<<coverage<<" canary_errors="<<canary<<" immutable_errors="<<immutable<<"\n";
    return (coverage|canary|immutable)?3:0;
  }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
}
