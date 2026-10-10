#define main nr_mlp_unused_main
#include "mlp_probe.mm"
#undef main

int main(int argc,const char** argv){@autoreleasepool{try{
  if(argc!=7)throw std::runtime_error("usage: wide_probe LIB INPUT WEIGHT ROWS POISON OUTPUT");
  const uint32_t rows=std::stoul(argv[4]);const unsigned poison=std::stoul(argv[5],nullptr,16);
  if(rows==0||rows%64||rows>138240||poison>255)throw std::runtime_error("shape/poison unsupported");
  fs::path output(argv[6]);if(fs::exists(output))throw std::runtime_error("preserve run");fs::create_directories(output);
  auto input=Read(argv[2]),weights=Read(argv[3]);if(input.size()!=size_t(rows)*64||weights.size()!=61760)throw std::runtime_error("input extent");
  if(Sha(weights.data(),weights.size())!="1b905577285978107b21b5730aac2b0105d45f757b5e78334291d9b4ab5646bb")throw std::runtime_error("Block 5 weight identity");
  auto device=MTLCreateSystemDefaultDevice();auto queue=[device newCommandQueue];NSError* error=nil;
  auto lib=[device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&error];if(!lib)throw std::runtime_error(error.localizedDescription.UTF8String);
  NSArray<NSString*>* functions=@[@"nr_wide_expand",@"nr_wide_activate",@"nr_wide_mid",@"nr_wide_mix",@"nr_wide_qkv",@"nr_wide_normalize",@"nr_wide_scores",@"nr_swin32_probabilities",@"nr_wide_values",@"nr_wide_project",@"nr_wide_pack"];
  const std::array<unsigned,11> elements={256,256,64,64,192,128,128,128,64,64,64};
  const std::array<unsigned,11> bytes={2,1,2,2,2,1,2,1,2,2,1};
  const std::array<unsigned,11> dispatch={256,256,64,64,192,4,128,2,64,64,64};
  const std::array<const char*,11> names={"expansion.f16","activation.e4m3","mid.f16","ffn.f16","qkv.f16","normalized.e4m3","scores.f16","probabilities.e4m3","values.f16","projection.f16","output.e4m3"};
  std::array<id<MTLComputePipelineState>,11> pipelines;std::vector<Buffer> y,counts;
  for(size_t i=0;i<11;++i){pipelines[i]=[device newComputePipelineStateWithFunction:[lib newFunctionWithName:functions[i]] error:&error];if(!pipelines[i])throw std::runtime_error("pipeline creation");
    y.push_back(Make(device,size_t(rows)*elements[i]*bytes[i],poison));counts.push_back(Make(device,size_t(rows)*elements[i]*4,0));}
  Buffer x=Make(device,input.size(),0),w=Make(device,weights.size(),0);std::memcpy(Data(x),input.data(),input.size());std::memcpy(Data(w),weights.data(),weights.size());
  auto command=[queue commandBuffer];
  for(size_t i=0;i<11;++i){auto encoder=[command computeCommandEncoder];[encoder setComputePipelineState:pipelines[i]];
    auto bind=[&](Buffer& b,NSUInteger j){[encoder setBuffer:b.metal offset:Guard atIndex:j];};
    auto shape=[&](NSUInteger j){uint32_t value=rows*(i==7?2:1);[encoder setBytes:&value length:4 atIndex:j];};
    if(i==0){bind(x,0);bind(w,1);bind(y[0],2);bind(counts[0],3);shape(4);}
    if(i==1){bind(y[0],0);bind(y[1],1);bind(counts[1],2);shape(3);}
    if(i==2){bind(y[1],0);bind(w,1);bind(y[2],2);bind(counts[2],3);shape(4);}
    if(i==3){bind(x,0);bind(y[2],1);bind(w,2);bind(y[3],3);bind(counts[3],4);shape(5);}
    if(i==4){bind(y[3],0);bind(w,1);bind(y[4],2);bind(counts[4],3);shape(4);}
    if(i==5){bind(y[4],0);bind(w,1);bind(y[5],2);bind(counts[5],3);shape(4);}
    if(i==6){bind(y[5],0);bind(w,1);bind(y[6],2);bind(counts[6],3);shape(4);}
    if(i==7){bind(y[6],0);bind(y[7],1);bind(counts[7],2);shape(3);}
    if(i==8){bind(y[7],0);bind(y[4],1);bind(y[8],2);bind(counts[8],3);shape(4);}
    if(i==9){bind(y[8],0);bind(y[3],1);bind(w,2);bind(y[9],3);bind(counts[9],4);shape(5);}
    if(i==10){bind(y[9],0);bind(y[10],1);bind(counts[10],2);shape(3);}
    [encoder dispatchThreads:MTLSizeMake(size_t(rows)*dispatch[i],1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];[encoder endEncoding];
  }
  [command commit];[command waitUntilCompleted];if(command.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error(command.error.localizedDescription.UTF8String);
  size_t coverage=0,canary=Canary(x)+Canary(w),immutable=(Sha(input.data(),input.size())!=Sha(Data(x),x.bytes))+(Sha(weights.data(),weights.size())!=Sha(Data(w),w.bytes));
  for(size_t i=0;i<11;++i){canary+=Canary(y[i])+Canary(counts[i]);auto* c=reinterpret_cast<uint32_t*>(Data(counts[i]));for(size_t j=0;j<size_t(rows)*elements[i];++j)coverage+=c[j]!=1;Write(output/names[i],y[i]);}
  std::ofstream receipt(output/"run.json");receipt<<"{\"schema\":\"neuralstudio.nr.swin-wide-block5-metal.v1\",\"pid\":"<<getpid()<<",\"rows\":"<<rows<<",\"poison\":"<<poison
    <<",\"input_sha256\":\""<<Sha(input.data(),input.size())<<"\",\"weight_sha256\":\""<<Sha(weights.data(),weights.size())<<"\",\"expected_input_opened\":false,\"outputs\":{";
  for(size_t i=0;i<11;++i)receipt<<(i?",":"")<<"\""<<names[i]<<"\":\""<<Sha(Data(y[i]),y[i].bytes)<<"\"";
  receipt<<"},\"coverage_errors\":"<<coverage<<",\"canary_errors\":"<<canary<<",\"immutable_errors\":"<<immutable<<",\"result\":\""<<((coverage|canary|immutable)?"fail":"pass")<<"\"}\n";
  std::cout<<"output_sha256="<<Sha(Data(y[10]),y[10].bytes)<<" coverage="<<coverage<<" canary="<<canary<<" immutable="<<immutable<<"\n";return (coverage|canary|immutable)?3:0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}}
