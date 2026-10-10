#define main nr_mlp_diagnostic_unused_main
#include "../nr_swin/mlp_probe.mm"
#undef main
int main(int argc,const char**argv){@autoreleasepool{try{
  if(argc!=6)throw std::runtime_error("usage: qmma_fixed_probe LIB OPERANDS EXPECTED POISON OUTPUT");
  unsigned poison=std::stoul(argv[4],nullptr,16);if(poison>255)throw std::runtime_error("poison");fs::path out(argv[5]);if(fs::exists(out))throw std::runtime_error("preserve run");fs::create_directories(out);
  auto input=Read(argv[2]);if(input.size()==0||input.size()%1024||input.size()>1024*1024)throw std::runtime_error("input extent");uint32_t cases=uint32_t(input.size()/1024);
  auto dev=MTLCreateSystemDefaultDevice();auto queue=[dev newCommandQueue];NSError*error=nil;auto lib=[dev newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&error];if(!lib)throw std::runtime_error(error.localizedDescription.UTF8String);
  auto state=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"nr_qmma_fixed_probe"] error:&error];if(!state)throw std::runtime_error(error.localizedDescription.UTF8String);
  Buffer x=Make(dev,input.size(),0),y=Make(dev,size_t(cases)*256,poison),c=Make(dev,size_t(cases)*256,0);std::memcpy(Data(x),input.data(),input.size());auto before=Sha(Data(x),x.bytes);
  auto command=[queue commandBuffer];auto e=[command computeCommandEncoder];[e setComputePipelineState:state];[e setBuffer:x.metal offset:Guard atIndex:0];[e setBuffer:y.metal offset:Guard atIndex:1];[e setBuffer:c.metal offset:Guard atIndex:2];[e setBytes:&cases length:4 atIndex:3];
  [e dispatchThreads:MTLSizeMake(cases*64,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];[e endEncoding];[command commit];[command waitUntilCompleted];if(command.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error(command.error.localizedDescription.UTF8String);
  Write(out/"output.bin",y);Write(out/"coverage.bin",c);auto expected=Read(argv[3]);if(expected.size()!=y.bytes)throw std::runtime_error("expected extent");
  size_t mismatch=0,coverage=0,canary=Canary(x)+Canary(y)+Canary(c);auto*actual=reinterpret_cast<uint32_t*>(Data(y));auto*ref=reinterpret_cast<uint32_t*>(expected.data());auto*cnt=reinterpret_cast<uint32_t*>(Data(c));
  for(size_t i=0;i<y.bytes/4;++i){mismatch+=actual[i]!=ref[i];coverage+=cnt[i]!=1;}bool immutable=before==Sha(Data(x),x.bytes);
  std::ofstream r(out/"run.json");r<<"{\"schema\":\"neuralstudio.nr.handwritten-qmma-fixed-probe.v1\",\"pid\":"<<getpid()<<",\"poison\":"<<poison<<",\"cases\":"<<cases
    <<",\"input_sha256\":\""<<before<<"\",\"expected_sha256\":\""<<Sha(expected.data(),expected.size())<<"\",\"output_sha256\":\""<<Sha(Data(y),y.bytes)<<"\",\"mismatching_words\":"<<mismatch
    <<",\"coverage_errors\":"<<coverage<<",\"canary_errors\":"<<canary<<",\"input_immutable\":"<<(immutable?"true":"false")<<",\"expected_gpu_input\":false,\"result\":\""<<((mismatch|coverage|canary|!immutable)?"fail":"pass")<<"\"}\n";
  if(!r)throw std::runtime_error("receipt write");std::cout<<"mismatch="<<mismatch<<" coverage="<<coverage<<" canary="<<canary<<"\n";return (mismatch|coverage|canary|!immutable)?3:0;
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}}
