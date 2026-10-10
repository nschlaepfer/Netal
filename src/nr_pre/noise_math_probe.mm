#define main nr_mlp_diagnostic_unused_main
#include "../nr_swin/mlp_probe.mm"
#undef main
int main(int argc,const char**argv){@autoreleasepool{try{
  if(argc!=4)throw std::runtime_error("usage: noise_math_probe LIB POISON OUTPUT");unsigned poison=std::stoul(argv[2],nullptr,16);if(poison>255)throw std::runtime_error("poison");
  fs::path out(argv[3]);if(fs::exists(out))throw std::runtime_error("preserve run");fs::create_directories(out);
  auto device=MTLCreateSystemDefaultDevice();auto queue=[device newCommandQueue];NSError* error=nil;
  auto lib=[device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&error];if(!lib)throw std::runtime_error(error.localizedDescription.UTF8String);
  auto state=[device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"nr_noise_math_baseline"] error:&error];if(!state)throw std::runtime_error(error.localizedDescription.UTF8String);
  constexpr size_t Count=16777216,Fields=5;Buffer y=Make(device,Count*Fields*4,poison),c=Make(device,Count*Fields*4,0);
  auto command=[queue commandBuffer];auto e=[command computeCommandEncoder];[e setComputePipelineState:state];[e setBuffer:y.metal offset:Guard atIndex:0];[e setBuffer:c.metal offset:Guard atIndex:1];
  [e dispatchThreads:MTLSizeMake(Count,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[e endEncoding];[command commit];[command waitUntilCompleted];if(command.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error(command.error.localizedDescription.UTF8String);
  size_t coverage=0,canary=Canary(y)+Canary(c);auto*counts=reinterpret_cast<uint32_t*>(Data(c));for(size_t i=0;i<Count*Fields;++i)coverage+=counts[i]!=1;
  Write(out/"baseline.bin",y);Write(out/"coverage.bin",c);std::ofstream r(out/"run.json");r<<"{\"schema\":\"neuralstudio.nr.noise-math-baseline-mac.v1\",\"pid\":"<<getpid()<<",\"poison\":"<<poison
    <<",\"records\":"<<Count<<",\"fields\":5,\"output_sha256\":\""<<Sha(Data(y),y.bytes)<<"\",\"coverage_errors\":"<<coverage<<",\"canary_errors\":"<<canary
    <<",\"model_inputs\":false,\"expected_gpu_input\":false,\"result\":\""<<((coverage|canary)?"fail":"pass")<<"\"}\n";
  if(!r)throw std::runtime_error("receipt write");std::cout<<"coverage="<<coverage<<" canary="<<canary<<"\n";return (coverage|canary)?3:0;
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}}
