// Included after the guarded Buffer helpers. No file input is opened here.
Buffer ExecuteWide(id<MTLDevice> device,id<MTLCommandQueue> queue,id<MTLLibrary> lib,
                   Buffer& x,Buffer& w,uint32_t rows,uint8_t poison,const fs::path& output,Buffer* projection=nullptr,uint32_t channels=64) {
  const bool weight_extent=channels==64?(w.bytes==61760||w.bytes==69936):channels==128?(w.bytes==197184||w.bytes==229936):channels==256?(w.bytes==689232||w.bytes==820288):false;
  if(!weight_extent||x.bytes!=size_t(rows)*channels||!rows||rows%64)throw std::runtime_error("wide extent mismatch");
  const uint32_t heads=channels/32,hidden=channels*4;
  const auto input_before=Sha(Data(x),x.bytes),weight_before=Sha(Data(w),w.bytes);
  NSArray<NSString*>* functions=@[@"nr_wide_expand",@"nr_wide_activate",@"nr_wide_mid",@"nr_wide_mix",@"nr_wide_qkv",@"nr_wide_normalize",@"nr_wide_scores",@"nr_swin32_probabilities",@"nr_wide_values",@"nr_wide_project",@"nr_wide_pack"];
  const std::array<unsigned,11> elements={hidden,hidden,channels,channels,channels*3,channels*2,heads*64,heads*64,channels,channels,channels};
  const std::array<unsigned,11> bytes={2,1,2,2,2,1,2,1,2,2,1};
  const std::array<unsigned,11> dispatch={hidden,hidden,channels,channels,channels*3,heads*2,heads*64,heads,channels,channels,channels};
  const std::array<const char*,11> names={"expansion.f16","activation.e4m3","mid.f16","ffn.f16","qkv.f16","normalized.e4m3","scores.f16","probabilities.e4m3","values.f16","projection.f16","output.e4m3"};
  std::array<id<MTLComputePipelineState>,11> pipelines;std::vector<Buffer> y,counts;NSError* error=nil;
  for(size_t i=0;i<11;++i){pipelines[i]=[device newComputePipelineStateWithFunction:[lib newFunctionWithName:functions[i]] error:&error];if(!pipelines[i])throw std::runtime_error("wide pipeline creation");
    y.push_back(Make(device,size_t(rows)*elements[i]*bytes[i],poison));counts.push_back(Make(device,size_t(rows)*elements[i]*4,0));}
  auto command=[queue commandBuffer];
  for(size_t i=0;i<11;++i){auto encoder=[command computeCommandEncoder];[encoder setComputePipelineState:pipelines[i]];
    auto bind=[&](Buffer& b,NSUInteger j){[encoder setBuffer:b.metal offset:Guard atIndex:j];};
    auto shape=[&](NSUInteger j){uint32_t value=rows*(i==7?heads:1);[encoder setBytes:&value length:4 atIndex:j];};
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
  Complete(command);fs::create_directories(output);
  for(size_t i=0;i<11;++i){GuardOutput(y[i],counts[i]);
    std::ofstream sample(output/(std::string("first-window-")+names[i]),std::ios::binary);
    sample.write(reinterpret_cast<char*>(Data(y[i])),size_t(64)*elements[i]*bytes[i]);if(!sample)throw std::runtime_error("sample write");}
  if(input_before!=Sha(Data(x),x.bytes)||weight_before!=Sha(Data(w),w.bytes)||Canary(x)||Canary(w))throw std::runtime_error("wide input mutation");
  if(projection)*projection=y[9];
  return y[10];
}
