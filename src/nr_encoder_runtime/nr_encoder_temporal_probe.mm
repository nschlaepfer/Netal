#include "nr_encoder_temporal_runtime.h"
#include "nr_body_runtime.h"
#import <Foundation/Foundation.h>
#import <CommonCrypto/CommonDigest.h>
#include <filesystem>
#include <fstream>
#include <vector>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <unistd.h>
namespace fs=std::filesystem;
static std::vector<uint8_t> Read(const fs::path& p){std::ifstream f(p,std::ios::binary|std::ios::ate);if(!f)throw std::runtime_error("input open");auto n=f.tellg();if(n<0)throw std::runtime_error("input extent");f.seekg(0);std::vector<uint8_t>b(static_cast<size_t>(n));f.read(reinterpret_cast<char*>(b.data()),n);if(!f)throw std::runtime_error("input read");return b;}
static std::string Sha(const void* p,size_t n){unsigned char h[32];CC_SHA256(p,CC_LONG(n),h);std::ostringstream s;s<<std::hex<<std::setfill('0');for(auto b:h)s<<std::setw(2)<<unsigned(b);return s.str();}
static NSString* S(const std::string& s){return [NSString stringWithUTF8String:s.c_str()];}
static void Write(const fs::path& p,const void* b,size_t n){std::ofstream f(p,std::ios::binary);f.write(static_cast<const char*>(b),n);if(!f)throw std::runtime_error("write failed");}
static std::vector<uint8_t> TextureBytes(id<MTLTexture> t,size_t bpp){std::vector<uint8_t>b(t.width*t.height*bpp);[t getBytes:b.data() bytesPerRow:t.width*bpp fromRegion:MTLRegionMake2D(0,0,t.width,t.height) mipmapLevel:0];return b;}
static id<MTLTexture> Texture(id<MTLDevice> d,MTLPixelFormat fmt,const std::vector<uint8_t>& b,size_t bpp){auto desc=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:fmt width:1920 height:1080 mipmapped:NO];desc.storageMode=MTLStorageModeShared;desc.usage=MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite;auto t=[d newTextureWithDescriptor:desc];if(!t)throw std::runtime_error("texture allocation");[t replaceRegion:MTLRegionMake2D(0,0,1920,1080) mipmapLevel:0 withBytes:b.data() bytesPerRow:1920*bpp];return t;}
static NrBodyRegion Region(const NrEncoderRegion& r){NrBodyRegion b;b.buffer=r.buffer;b.offset=r.offset;b.bytes=r.bytes;b.provenance_sha256=r.sha256;b.producer="live native encoder";return b;}
int main(int argc,char** argv){@autoreleasepool{try{
  if(argc!=8)throw std::runtime_error("usage: probe ASSETS BODY_DESCRIPTOR IMAGE REFERENCE_MATRIX SAMPLER POISON OUT");
  fs::path out=argv[7];if(fs::exists(out))throw std::runtime_error("preserve output");fs::create_directories(out);
  unsigned sampler=std::stoul(argv[5]),poison=std::stoul(argv[6],nullptr,16);if(sampler>2||poison>255)throw std::runtime_error("configuration");
  auto image=Read(argv[3]);if(image.size()!=16588800||Sha(image.data(),image.size())!="81ec1de5d1e76e6b55c4fd987b3732180c9720919fbd760dd5b4e8aa8cdd0342")throw std::runtime_error("original-image test contract");
  NrEncoderContext enc;std::string error;if(nr_encoder_prepare(enc,argv[1],error))throw std::runtime_error(error);
  auto colour=Texture(enc.device,MTLPixelFormatRGBA16Float,image,8);auto output=Texture(enc.device,MTLPixelFormatRGBA16Float,std::vector<uint8_t>(image.size(),uint8_t(poison)),8);
  NrEncoderFrame first;first.color=colour;NrEncoderResult e0;int ec=nr_encoder_run_frame(enc,first,uint8_t(poison),e0);if(ec)throw std::runtime_error(e0.error);
  auto db=Read(argv[2]);NSDictionary* desc=[NSJSONSerialization JSONObjectWithData:[NSData dataWithBytes:db.data() length:db.size()] options:0 error:nil];if(!desc)throw std::runtime_error("descriptor JSON");
  NrBodyContext body;body.dev=enc.device;body.queue=enc.queue;NrBodyFrameInputs bi;bi.entry=Region(e0.entry);bi.block0_skip=Region(e0.block0_skip);for(int i=0;i<4;++i)bi.skips[i]=Region(e0.skips[i]);bi.input_color=colour;bi.output_surface=output;bi.poison=uint8_t(poison);bi.production=true;NrBodyResult br;
  int bc=nr_body_run_frame(body,desc,bi,nullptr,br);if(bc||!br.check_failures.empty()||br.checks_enforced!=241)throw std::runtime_error("live first-frame body failure: "+br.error);
  auto firstBytes=TextureBytes(output,8);auto firstHash=Sha(firstBytes.data(),firstBytes.size());if(firstHash!="a07835db4be8469dad1e3d3befcc495be396790f89408be62ddb8a5a710585a2"||firstHash!=br.final_surface_sha256)throw std::runtime_error("live first-frame reference mismatch");Write(out/"first-live-output.rgba16le",firstBytes.data(),firstBytes.size());
  NrTemporalState state;if(!nr_temporal_state_create(state,enc.device,1920,1080,uint8_t(poison^255)))throw std::runtime_error("history allocation");
  auto copyHistory=[&](){auto c=[enc.queue commandBuffer];auto bl=[c blitCommandEncoder];[bl copyFromTexture:output sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0) sourceSize:MTLSizeMake(1920,1080,1) toTexture:state.history destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0,0,0)];[bl endEncoding];[c commit];[c waitUntilCompleted];if(c.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error("live history copy failed");};copyHistory();
  auto hb=TextureBytes(state.history,8);if(Sha(hb.data(),hb.size())!=firstHash)throw std::runtime_error("live history copy mismatch");state.history_sha256=state.last_output_sha256=firstHash;state.frames_since_reset=1;
  auto motion=Texture(enc.device,MTLPixelFormatRG16Float,std::vector<uint8_t>(8294400,0),4);
  NrEncoderTemporalFrame next;next.frame.color=colour;next.frame.reset=false;next.frame.frame_index=1;next.temporal=&state;next.motion=motion;next.sampler_variant=sampler;
  NSMutableArray* negatives=[NSMutableArray array];
  for(unsigned n=0;n<9;++n){auto bad=next;NrTemporalState altered=state;
    if(n==0)bad.temporal=nullptr;if(n==1)bad.frame.reset=true;if(n==2)bad.frame.frame_index=0;if(n==3)bad.frame.frame_index=2;if(n==4)bad.sampler_variant=3;if(n==5)bad.motion=nil;
    if(n==6){altered.history_view.linear=false;bad.temporal=&altered;}
    if(n==7){altered.history_sha256="invalid";bad.temporal=&altered;}
    if(n==8){uint16_t junk[4]={0x7c00,0,0,0};[state.history replaceRegion:MTLRegionMake2D(0,0,1,1) mipmapLevel:0 withBytes:junk bytesPerRow:8];}
    NrEncoderResult rejected;int rc=nr_encoder_run_temporal_frame(enc,bad,uint8_t(poison),rejected);if(rc!=2||rejected.entry.buffer||rejected.block0_skip.buffer||rejected.guarded_outputs_checked)throw std::runtime_error("temporal negative control failed");[negatives addObject:@{@"test":@(n),@"code":@(rc),@"error":S(rejected.error)}];if(n==8)copyHistory();
  }
  std::vector<uint8_t> poisoned(image.size(),uint8_t(poison^255));[output replaceRegion:MTLRegionMake2D(0,0,1920,1080) mipmapLevel:0 withBytes:poisoned.data() bytesPerRow:1920*8];
  NrEncoderResult e1;fs::path diagnostics=out/"temporal-diagnostics";int tc=nr_encoder_run_temporal_frame(enc,next,uint8_t(poison^255),e1,diagnostics.c_str());
  NSMutableDictionary* ports=[NSMutableDictionary dictionary];bool match=tc==0;
  // Reference data is opened ONLY after live temporal encoder execution returns.
  auto rb=Read(argv[4]);NSDictionary* reference=[NSJSONSerialization JSONObjectWithData:[NSData dataWithBytes:rb.data() length:rb.size()] options:0 error:nil];NSDictionary* wanted=reference[@"boundary_ports"][3];
  if(tc==0){std::array<std::pair<const char*,NrEncoderRegion>,6> regions={{{"block0-skip",e1.block0_skip},{"block22-middle",e1.entry},{"block22-native",e1.skips[0]},{"block14-native",e1.skips[1]},{"block8-native",e1.skips[2]},{"block4-native",e1.skips[3]}}};
    for(auto& item:regions){const auto& r=item.second;auto p=static_cast<const uint8_t*>(r.buffer.contents)+r.offset;auto h=Sha(p,r.bytes);Write(out/(std::string(item.first)+".e4m3"),p,r.bytes);bool equal=[wanted[S(item.first)] isEqualToString:S(h)];match&=equal;ports[S(item.first)]=@{@"sha256":S(h),@"bytes":@(r.bytes),@"offset":@(r.offset),@"matches_rtx":@(equal)};}
    match&=e1.guarded_outputs_checked==1040&&e1.coverage_values_checked==5935497216ull;
  }
  hb=TextureBytes(state.history,8);bool immutable=Sha(hb.data(),hb.size())==firstHash;match&=immutable;
  NSDictionary* receipt=@{@"schema":@"neuralstudio.nr.live-temporal-encoder-probe.v1",@"pid":@(getpid()),@"sampler_variant":@(sampler),@"poison_first":@(poison),@"poison_second":@(poison^255),@"first_output_sha256":S(firstHash),@"first_encoder_code":@(ec),@"first_body_code":@(bc),@"first_body_checks":@(br.checks_enforced),@"history_origin":@"GPU copy of this process's live first native generated frame",@"archived_history_input":@NO,@"history_immutable":@(immutable),@"negative_controls":negatives,@"temporal_encoder_code":@(tc),@"error":S(e1.error),@"guarded_outputs_checked":@(e1.guarded_outputs_checked),@"coverage_values_checked":@(e1.coverage_values_checked),@"feature_sha256":S(e1.feature_sha256),@"ports":ports,@"result":match?@"pass_temporal_encoder_boundaries":@"counterexample",@"full_temporal_model":@NO};
  NSMutableDictionary* full=[receipt mutableCopy];NSMutableDictionary* loaded=[NSMutableDictionary dictionary];for(const auto& pair:br.provenance)loaded[S(pair.first)]=S(pair.second);full[@"first_body_loaded"]=loaded;full[@"first_body_constants_unchanged"]=@(br.constants_unchanged_after_frame);full[@"first_body_constants_rehashed"]=@(br.constants_rehashed);full[@"first_encoder_guarded_outputs"]=@(e0.guarded_outputs_checked);full[@"first_encoder_coverage_values"]=@(e0.coverage_values_checked);
  NSData* json=[NSJSONSerialization dataWithJSONObject:full options:NSJSONWritingPrettyPrinted|NSJSONWritingSortedKeys error:nil];Write(out/"run.json",json.bytes,json.length);std::cout<<"temporal_encoder="<<tc<<" boundary_match="<<match<<"\n";return match?0:3;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 2;}}}
