#define main nr_mlp_diagnostic_unused_main
#include "../nr_swin/mlp_probe.mm"
#undef main
int main(int argc,const char**argv){@autoreleasepool{try{
  if(argc!=5)throw std::runtime_error("usage: entry_layout_probe LIB NATIVE_PRE_RUN POISON OUTPUT");
  unsigned poison=std::stoul(argv[3],nullptr,16);if(poison>255)throw std::runtime_error("poison");fs::path pre(argv[2]),out(argv[4]);
  if(fs::exists(out))throw std::runtime_error("preserve run");fs::create_directories(out);
  auto receipt=Read(pre/"run.json");NSData*data=[NSData dataWithBytes:receipt.data() length:receipt.size()];NSError*error=nil;
  NSDictionary*producer=[NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
  if(!producer||![producer[@"schema"] isEqual:@"neuralstudio.nr.pre-native-candidate.v1"]||![producer[@"result"] isEqual:@"pass"]||[producer[@"expected_gpu_input"] boolValue]||![producer[@"live_device_handoffs"] boolValue])throw std::runtime_error("accepted live producer receipt required");
  auto input=Read(pre/"encoder.e4m3");auto before=Sha(input.data(),input.size());
  if(input.size()!=17694720||before!="de4789425897a4ffa3d5646670c12385d25109dcad60609c1729dca2da4a33a0"||before!=std::string([producer[@"encoder_sha256"] UTF8String]))throw std::runtime_error("native producer output identity");
  auto device=MTLCreateSystemDefaultDevice();auto queue=[device newCommandQueue];auto lib=[device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&error];if(!lib)throw std::runtime_error(error.localizedDescription.UTF8String);
  auto state=[device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"nr_pre_entry_layout"] error:&error];if(!state)throw std::runtime_error(error.localizedDescription.UTF8String);
  Buffer x=Make(device,input.size(),0),y=Make(device,input.size(),poison),c=Make(device,input.size()*4,0);std::memcpy(Data(x),input.data(),input.size());
  auto command=[queue commandBuffer];auto e=[command computeCommandEncoder];[e setComputePipelineState:state];[e setBuffer:x.metal offset:Guard atIndex:0];[e setBuffer:y.metal offset:Guard atIndex:1];[e setBuffer:c.metal offset:Guard atIndex:2];
  [e dispatchThreads:MTLSizeMake(input.size(),1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[e endEncoding];[command commit];[command waitUntilCompleted];if(command.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error(command.error.localizedDescription.UTF8String);
  Write(out/"encoder-window.e4m3",y);Write(out/"coverage.bin",c);
  auto after=Sha(Data(y),y.bytes);size_t coverage=0,canary=Canary(x)+Canary(y)+Canary(c);auto*counts=reinterpret_cast<uint32_t*>(Data(c));
  for(size_t i=0;i<c.bytes/4;++i)coverage+=counts[i]!=1;bool immutable=before==Sha(Data(x),x.bytes);
  bool exact=after=="176544cf02971c274f9f8d5de8a0436d5e71945ce4e333dc2ef169b48b10e554";
  std::ofstream r(out/"run.json");r<<"{\"schema\":\"neuralstudio.nr.block0-encoder-entry-layout.v1\",\"pid\":"<<getpid()<<",\"poison\":"<<poison
    <<",\"producer_pid\":"<<[producer[@"pid"] unsignedIntValue]<<",\"producer_receipt_sha256\":\""<<Sha(receipt.data(),receipt.size())<<"\",\"input_sha256\":\""<<before<<"\",\"output_sha256\":\""<<after
    <<"\",\"coverage_errors\":"<<coverage<<",\"canary_errors\":"<<canary<<",\"input_immutable\":"<<(immutable?"true":"false")<<",\"expected_gpu_input\":false,\"result\":\""<<((coverage|canary|!immutable|!exact)?"fail":"pass")<<"\"}\n";
  if(!r)throw std::runtime_error("receipt write");std::cout<<"output_sha256="<<after<<"\n";return (coverage|canary|!immutable|!exact)?3:0;
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}}
