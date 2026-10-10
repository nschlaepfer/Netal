#include "nr_encoder_runtime.h"
#import <Foundation/Foundation.h>
#import <CommonCrypto/CommonDigest.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include <cstring>
#include <array>
#include <iomanip>
#include <sstream>
#include <unistd.h>
namespace fs=std::filesystem;
static std::vector<uint8_t> Read(const fs::path& p) {
  std::ifstream f(p,std::ios::binary|std::ios::ate);if(!f)throw std::runtime_error("file open failed: "+p.string());
  auto n=f.tellg();if(n<0)throw std::runtime_error("file size");f.seekg(0);std::vector<uint8_t> v(static_cast<size_t>(n));f.read(reinterpret_cast<char*>(v.data()),n);if(!f)throw std::runtime_error("file read");return v;
}
static std::string Sha(const void* p,size_t n) {
  unsigned char h[32];CC_SHA256(p,static_cast<CC_LONG>(n),h);std::ostringstream s;s<<std::hex<<std::setfill('0');for(auto b:h)s<<std::setw(2)<<unsigned(b);return s.str();
}
static const uint8_t* Data(const NrEncoderRegion& r) {
  if(!r.buffer||r.offset>r.buffer.length||r.bytes>r.buffer.length-r.offset||!r.buffer.contents)throw std::runtime_error("returned region bounds");
  return static_cast<const uint8_t*>(r.buffer.contents)+r.offset;
}
static std::array<std::pair<std::string,NrEncoderRegion>,6> Ports(const NrEncoderResult& r) {
  return {{{"block0-skip",r.block0_skip},{"block22-middle",r.entry},{"block22-native",r.skips[0]},{"block14-native",r.skips[1]},{"block8-native",r.skips[2]},{"block4-native",r.skips[3]}}};
}
static NSString* S(const std::string& s){return [NSString stringWithUTF8String:s.c_str()];}
static id<MTLTexture> Texture(id<MTLDevice> dev,const std::vector<uint8_t>& bytes) {
  if(bytes.size()!=1920*1080*8)throw std::runtime_error("image size");
  auto d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float width:1920 height:1080 mipmapped:NO];d.storageMode=MTLStorageModeShared;d.usage=MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite;
  auto t=[dev newTextureWithDescriptor:d];if(!t)throw std::runtime_error("texture allocation");[t replaceRegion:MTLRegionMake2D(0,0,1920,1080) mipmapLevel:0 withBytes:bytes.data() bytesPerRow:1920*8];return t;
}
int main(int argc,const char** argv) {@autoreleasepool{try{
  if(argc!=8&&!(argc==9&&std::string(argv[8])=="--profile"))throw std::runtime_error("usage: probe ASSETS IMAGE0 REFERENCE0 IMAGE1 REFERENCE1 POISON OUTPUT [--profile]");
  const bool profile=argc==9;
  unsigned poison=std::stoul(argv[6],nullptr,16);if(poison>255)throw std::runtime_error("poison");
  fs::path out=argv[7];if(fs::exists(out))throw std::runtime_error("preserve previous output");fs::create_directories(out);
  NrEncoderContext ctx;std::string error;if(nr_encoder_prepare(ctx,argv[1],error))throw std::runtime_error(error);
  NSMutableArray* frames=[NSMutableArray array];NSMutableArray* negative=[NSMutableArray array];std::vector<NrEncoderResult> retained;
  auto bytes0=Read(argv[2]);auto tex0=Texture(ctx.device,bytes0);
  for(unsigned test=0;test<6;++test) {
    NrEncoderFrame bad;bad.color=tex0;
    if(test==0)bad.color=nil;
    if(test==1)bad.reset=false;
    if(test==2)bad.frame_index=1;
    if(test==3)bad.producer_completion=[ctx.queue commandBuffer];
    if(test==4||test==5){auto d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:test==5?MTLPixelFormatRGBA8Unorm:MTLPixelFormatRGBA16Float width:test==4?1919:1920 height:1080 mipmapped:NO];d.storageMode=MTLStorageModeShared;bad.color=[ctx.device newTextureWithDescriptor:d];}
    NrEncoderResult rejected;int rc=nr_encoder_run_frame(ctx,bad,uint8_t(poison),rejected);
    if(rc==0||rejected.gpu_complete||rejected.entry.buffer||rejected.block0_skip.buffer||rejected.guarded_outputs_checked)throw std::runtime_error("negative control did not fail closed");
    [negative addObject:@{@"test":@(test),@"code":@(rc),@"error":S(rejected.error)}];
  }
  for(unsigned i=0;i<2;++i) {
    auto bytes=i?Read(argv[4]):bytes0;auto texture=i?Texture(ctx.device,bytes):tex0;
    NrEncoderFrame frame;frame.color=texture;NrEncoderResult result;auto start=[NSDate date];
    const uint8_t current_poison=i?uint8_t(poison^0xff):uint8_t(poison);
    const fs::path diagnostics=out/("diagnostics-frame"+std::to_string(i));
    int code=nr_encoder_run_frame(ctx,frame,current_poison,result,profile?diagnostics.c_str():nullptr);double wall=-[start timeIntervalSinceNow]*1000;
    if(code||!result.gpu_complete||!result.input_immutable||!result.guarded_outputs_checked||!result.coverage_values_checked)throw std::runtime_error(result.error.empty()?"incomplete runtime result":result.error);
    auto imageSha=Sha(bytes.data(),bytes.size());if(result.input_sha256!=imageSha)throw std::runtime_error("actual input image mismatch");
    // Expected artifacts are opened ONLY after native execution has returned.
    fs::path ref=argv[i?5:3],dir=out/("frame"+std::to_string(i));fs::create_directories(dir);NSMutableDictionary* ports=[NSMutableDictionary dictionary];
    size_t mismatches=0;
    for(const auto& [name,region]:Ports(result)) {
      auto data=Data(region);std::string h=Sha(data,region.bytes);if(h!=region.sha256||region.buffer.device!=ctx.device)throw std::runtime_error("returned GPU allocation identity");
      auto expected=Read(ref/(name=="block0-skip"?"block0/skip.e4m3":"encoder/"+name+".e4m3"));
      if(expected.size()!=region.bytes)throw std::runtime_error("comparison extent");size_t bad=0;for(size_t j=0;j<region.bytes;++j)bad+=data[j]!=expected[j];mismatches+=bad;
      std::ofstream f(dir/(name+".e4m3"),std::ios::binary);f.write(reinterpret_cast<const char*>(data),region.bytes);if(!f)throw std::runtime_error("output retention failed");
      ports[S(name)]=@{@"sha256":S(h),@"bytes":@(region.bytes),@"offset":@(region.offset),@"mismatching_bytes":@(bad)};
    }
    retained.push_back(result);
    // Earlier returned buffers remain live and must survive a subsequent frame.
    for(const auto& old:retained)for(const auto& pair:Ports(old))if(Sha(Data(pair.second),pair.second.bytes)!=pair.second.sha256)throw std::runtime_error("previous output lifetime or mutation");
    [frames addObject:@{@"frame":@(i),@"input_sha256":S(imageSha),@"poison":@(current_poison),@"ports":ports,@"coverage_values_checked":@(result.coverage_values_checked),@"guarded_outputs_checked":@(result.guarded_outputs_checked),@"input_immutable":@YES,@"gpu_complete":@YES,@"expected_gpu_input":@NO,@"mismatching_bytes":@(mismatches),@"wall_ms_diagnostic_only":@(wall),@"bundle_root":S(result.bundle_root),@"prior_outputs_still_intact":@YES}];
    NSDictionary* receipt=@{@"schema":@"neuralstudio.nr.encoder-runtime-characterization.v1",@"pid":@(getpid()),@"frames":frames,@"negative_controls":negative,@"context_reused":@YES,@"activation_file_ingress":@NO,@"result":mismatches?@"fail":i==1?@"pass":@"partial"};
    NSError* e=nil;auto data=[NSJSONSerialization dataWithJSONObject:receipt options:NSJSONWritingPrettyPrinted error:&e];
    if(!data||![data writeToFile:S((out/"run.json").string()) atomically:YES])throw std::runtime_error("receipt write");
    std::cout<<"frame="<<i<<" mismatching_bytes="<<mismatches<<" guarded_outputs="<<result.guarded_outputs_checked<<"\n"<<std::flush;
    if(mismatches)return 3;
  }
  return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 2;}}}
