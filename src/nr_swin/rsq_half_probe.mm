#define main nr_mlp_unused_main
#include "mlp_probe.mm"
#undef main

int main(int argc,const char** argv){@autoreleasepool{try{
  if(argc!=4)throw std::runtime_error("usage: rsq_probe LIB POISON OUTPUT");
  fs::path output(argv[3]);if(fs::exists(output))throw std::runtime_error("preserve prior run");
  unsigned poison=std::stoul(argv[2],nullptr,16);if(poison>255)throw std::runtime_error("invalid poison");
  id<MTLDevice> device=MTLCreateSystemDefaultDevice();NSError* error=nil;
  id<MTLLibrary> lib=[device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&error];
  if(!lib)throw std::runtime_error(error.localizedDescription.UTF8String);
  auto pipeline=[device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"nr_rsq_half"] error:&error];
  if(!pipeline)throw std::runtime_error(error.localizedDescription.UTF8String);
  constexpr unsigned count=0x7bff;
  Buffer values=Make(device,count*2,poison),coverage=Make(device,count*4,0);
  id<MTLCommandQueue> queue=[device newCommandQueue];id<MTLCommandBuffer> command=[queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];[encoder setBuffer:values.metal offset:Guard atIndex:0];
  [encoder setBuffer:coverage.metal offset:Guard atIndex:1];
  [encoder dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];[encoder endEncoding];
  [command commit];[command waitUntilCompleted];if(command.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error("GPU failed");
  size_t errors=Canary(values)+Canary(coverage);auto* c=reinterpret_cast<unsigned*>(Data(coverage));
  for(unsigned i=0;i<count;++i)errors+=c[i]!=1;
  fs::create_directories(output);Write(output/"output.bin",values);std::ofstream receipt(output/"run.json");
  receipt<<"{\"schema\":\"neuralstudio.nr.rsq-half-metal.v1\",\"pid\":"<<getpid()<<",\"count\":"<<count
    <<",\"poison\":"<<poison<<",\"errors\":"<<errors<<",\"output_sha256\":\""<<Sha(Data(values),values.bytes)
    <<"\",\"expected_input_opened\":false,\"result\":\""<<(errors?"fail":"pass")<<"\"}\n";
  std::cout<<"count="<<count<<" errors="<<errors<<"\n";return errors?3:0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}}
