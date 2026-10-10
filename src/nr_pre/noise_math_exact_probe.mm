#define main nr_mlp_diagnostic_unused_main
#include "../nr_swin/mlp_probe.mm"
#undef main
int main(int argc,const char**argv){@autoreleasepool{try{
  if(argc!=7)throw std::runtime_error("usage: noise_math_exact_probe LIB DELTAS ESCAPES EXPECTED POISON OUTPUT");unsigned poison=std::stoul(argv[5],nullptr,16);if(poison>255)throw std::runtime_error("poison");
  fs::path out(argv[6]);if(fs::exists(out))throw std::runtime_error("preserve run");fs::create_directories(out);
  auto db=Read(argv[2]),eb=Read(argv[3]);
  if(db.size()!=67108864||Sha(db.data(),db.size())!="768da747f3c7c799d768cd4410195934292cc838fbc083ec6a3ab9c04184f37b")throw std::runtime_error("instruction delta identity");
  if(eb.size()!=2780432||Sha(eb.data(),eb.size())!="c2e095109b1f220ed289ea09f5b2d10a4df9f15f5366355e7fdab30b455d40f5")throw std::runtime_error("instruction escape identity");
  auto device=MTLCreateSystemDefaultDevice();auto queue=[device newCommandQueue];NSError*error=nil;
  auto lib=[device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&error];if(!lib)throw std::runtime_error(error.localizedDescription.UTF8String);
  auto state=[device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"nr_noise_math_exact_probe"] error:&error];if(!state)throw std::runtime_error(error.localizedDescription.UTF8String);
  constexpr size_t Count=16777216,Fields=6;Buffer y=Make(device,Count*Fields*4,poison),c=Make(device,Count*Fields*4,0),d=Make(device,db.size(),0),esc=Make(device,eb.size(),0);
  std::memcpy(Data(d),db.data(),db.size());std::memcpy(Data(esc),eb.data(),eb.size());uint32_t escapes=uint32_t(eb.size()/8);
  auto command=[queue commandBuffer];auto e=[command computeCommandEncoder];[e setComputePipelineState:state];std::array<Buffer*,4> buffers={&y,&c,&d,&esc};
  for(size_t i=0;i<4;++i)[e setBuffer:buffers[i]->metal offset:Guard atIndex:i];[e setBytes:&escapes length:4 atIndex:4];
  [e dispatchThreads:MTLSizeMake(Count,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[e endEncoding];[command commit];[command waitUntilCompleted];if(command.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error(command.error.localizedDescription.UTF8String);
  Write(out/"output.bin",y);Write(out/"coverage.bin",c);
  auto expected=Read(argv[4]);if(expected.size()!=y.bytes||Sha(expected.data(),expected.size())!="daa0f10f8a04d625c38b277bfc9fef2d4fd68622fd9f6203024e9d7d0415975a")throw std::runtime_error("comparison-only oracle identity");
  size_t coverage=0,canary=Canary(y)+Canary(c)+Canary(d)+Canary(esc),mismatch=0;std::array<size_t,6> field{};
  auto*counts=reinterpret_cast<uint32_t*>(Data(c));auto*actual=reinterpret_cast<uint32_t*>(Data(y));auto*ref=reinterpret_cast<uint32_t*>(expected.data());
  for(size_t i=0;i<Count*Fields;++i){coverage+=counts[i]!=1;if(actual[i]!=ref[i]){++mismatch;++field[i%6];}}
  bool immutable=std::memcmp(Data(d),db.data(),db.size())==0&&std::memcmp(Data(esc),eb.data(),eb.size())==0;
  std::ofstream r(out/"run.json");r<<"{\"schema\":\"neuralstudio.nr.noise-math-exact-mac.v1\",\"pid\":"<<getpid()<<",\"poison\":"<<poison<<",\"records\":"<<Count
    <<",\"fields\":6,\"output_sha256\":\""<<Sha(Data(y),y.bytes)<<"\",\"mismatching_words\":"<<mismatch<<",\"field_mismatches\":[";
  for(size_t i=0;i<6;++i)r<<(i?",":"")<<field[i];r<<"],\"coverage_errors\":"<<coverage<<",\"canary_errors\":"<<canary<<",\"corrections_immutable\":"<<(immutable?"true":"false")
    <<",\"model_inputs\":false,\"expected_gpu_input\":false,\"radius_lookup\":false,\"result\":\""<<((coverage|canary|mismatch|!immutable)?"fail":"pass")<<"\"}\n";
  if(!r)throw std::runtime_error("receipt write");std::cout<<"mismatch="<<mismatch<<" coverage="<<coverage<<" canary="<<canary<<"\n";return (coverage|canary|mismatch|!immutable)?3:0;
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}}
