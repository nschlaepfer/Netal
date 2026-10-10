#define main nr_mlp_diagnostic_unused_main
#include "../nr_swin/mlp_probe.mm"
#undef main

struct PreShape {uint32_t width,height,padded_height,frame,phase_math;};
int main(int argc,const char**argv){@autoreleasepool{try{
  if(argc!=7)throw std::runtime_error("usage: pre_features_probe LIB IMAGE EXPECTED POISON PHASE_MATH OUTPUT");
  unsigned poison=std::stoul(argv[4],nullptr,16),phase=std::stoul(argv[5]);if(poison>255||phase>2)throw std::runtime_error("configuration");
  fs::path out(argv[6]);if(fs::exists(out))throw std::runtime_error("preserve run");fs::create_directories(out);
  auto source=Read(argv[2]);if(source.size()!=1920*1080*8||Sha(source.data(),source.size())!="81ec1de5d1e76e6b55c4fd987b3732180c9720919fbd760dd5b4e8aa8cdd0342")throw std::runtime_error("pinned image mismatch");
  id<MTLDevice> device=MTLCreateSystemDefaultDevice();auto queue=[device newCommandQueue];NSError*error=nil;
  auto lib=[device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&error];if(!lib)throw std::runtime_error(error.localizedDescription.UTF8String);
  auto pipeline=[device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"nr_pre_features"] error:&error];if(!pipeline)throw std::runtime_error(error.localizedDescription.UTF8String);
  Buffer input=Make(device,source.size(),0),output=Make(device,70778880,poison),counts=Make(device,70778880*2,0);
  std::memcpy(Data(input),source.data(),source.size());auto before=Sha(Data(input),input.bytes);PreShape shape{1920,1080,1152,0,phase};
  auto command=[queue commandBuffer];auto encoder=[command computeCommandEncoder];[encoder setComputePipelineState:pipeline];
  [encoder setBuffer:input.metal offset:Guard atIndex:0];[encoder setBuffer:output.metal offset:Guard atIndex:1];[encoder setBuffer:counts.metal offset:Guard atIndex:2];[encoder setBytes:&shape length:sizeof(shape) atIndex:3];
  [encoder dispatchThreads:MTLSizeMake(1920*1152,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[encoder endEncoding];[command commit];[command waitUntilCompleted];
  if(command.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error(command.error.localizedDescription.UTF8String);
  Write(out/"features-shared.f16",output);Write(out/"coverage.bin",counts);
  // Expected bytes are first opened after native execution. Never GPU inputs.
  auto expected=Read(argv[3]);if(expected.size()!=output.bytes)throw std::runtime_error("expected extent");
  size_t mismatch=0,coverage=0,canary=Canary(input)+Canary(output)+Canary(counts);std::array<size_t,16> channel{};
  auto*actual=reinterpret_cast<const uint16_t*>(Data(output));auto*oracle=reinterpret_cast<const uint16_t*>(expected.data());auto*c=reinterpret_cast<const uint32_t*>(Data(counts));
  for(size_t i=0;i<output.bytes/2;++i){if(actual[i]!=oracle[i]){++mismatch;++channel[((i%1024)/512)*8+i%8];}coverage+=c[i]!=1;}
  bool immutable=before==Sha(Data(input),input.bytes);std::ofstream receipt(out/"run.json");
  receipt<<"{\"schema\":\"neuralstudio.nr.pre-features-candidate.v1\",\"pid\":"<<getpid()<<",\"poison\":"<<poison<<",\"phase_math\":"<<phase
    <<",\"input_sha256\":\""<<before<<"\",\"expected_sha256\":\""<<Sha(expected.data(),expected.size())<<"\",\"output_sha256\":\""<<Sha(Data(output),output.bytes)
    <<"\",\"mismatching_half_values\":"<<mismatch<<",\"channel_mismatches\":[";
  for(size_t i=0;i<16;++i)receipt<<(i?",":"")<<channel[i];
  receipt<<"],\"coverage_errors\":"<<coverage<<",\"canary_errors\":"<<canary<<",\"input_immutable\":"<<(immutable?"true":"false")
    <<",\"expected_gpu_input\":false,\"candidate_only\":true,\"result\":\""<<((mismatch|coverage|canary|!immutable)?"fail":"pass")<<"\"}\n";
  if(!receipt)throw std::runtime_error("receipt write");std::cout<<"mismatch="<<mismatch<<" coverage="<<coverage<<" canary="<<canary<<"\n";return (mismatch|coverage|canary|!immutable)?3:0;
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}}
