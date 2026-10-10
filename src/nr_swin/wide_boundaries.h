Buffer WideRewindow(Pipelines& p,id<MTLLibrary> lib,Buffer& source,Grid from,Grid to,uint8_t poison,uint32_t channels=64,uint32_t width=480,uint32_t height=288) {
  WindowMap shape{width,height,from.width,from.height,to.width,to.height,from.x,from.y,to.x,to.y};
  if((channels!=64&&channels!=128&&channels!=256)||source.bytes!=size_t(from.width)*from.height*64*channels||from.x>0||from.y>0||from.width*8<uint32_t(int(width)-from.x)||from.height*8<uint32_t(int(height)-from.y))throw std::runtime_error("wide source extent");
  size_t size=size_t(to.width)*to.height*64*channels;Buffer result=Make(p.device,size,poison),counts=Make(p.device,size*4,0);
  const auto before=Sha(Data(source),source.bytes);NSError* error=nil;
  auto state=[p.device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"nr_wide_rewindow"] error:&error];if(!state)throw std::runtime_error("rewindow pipeline");
  auto command=[p.queue commandBuffer];auto encoder=[command computeCommandEncoder];[encoder setComputePipelineState:state];
  [encoder setBuffer:source.metal offset:Guard atIndex:0];[encoder setBuffer:result.metal offset:Guard atIndex:1];[encoder setBuffer:counts.metal offset:Guard atIndex:2];
  [encoder setBytes:&shape length:sizeof(shape) atIndex:3];[encoder dispatchThreads:MTLSizeMake(size,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];[encoder endEncoding];
  Complete(command);GuardOutput(result,counts);if(before!=Sha(Data(source),source.bytes)||Canary(source))throw std::runtime_error("wide rewindow mutation");return result;
}
Buffer WideDownsample(Pipelines& p,id<MTLLibrary> lib,Buffer& source,Buffer& weight,Grid grid,uint8_t poison,const fs::path& output,uint32_t block=8,uint32_t channels=64,uint32_t width=480,uint32_t height=288) {
  const uint32_t destination_width=(width/2+7)/8,destination_height=(height/2+7)/8;
  const uint32_t rows=destination_width*destination_height*64;
  const bool selected=(block==8&&channels==64&&width==480&&height==288&&grid.width==60&&grid.height==37&&grid.x==0&&grid.y==-4&&weight.bytes==69936)||
    (block==14&&channels==128&&width==240&&height==144&&grid.width==31&&grid.height==19&&grid.x==-4&&grid.y==-4&&weight.bytes==229936)||
    (block==22&&channels==256&&width==120&&height==72&&grid.width==15&&grid.height==10&&grid.x==0&&grid.y==-4&&weight.bytes==820288);
  if(!selected||source.bytes!=size_t(grid.width)*grid.height*64*channels*2)throw std::runtime_error("selected downsample extent");
  WindowMap shape{width,height,grid.width,grid.height,destination_width,destination_height,grid.x,grid.y,0,0};
  Buffer pooled=Make(p.device,size_t(rows)*channels,poison),result=Make(p.device,size_t(rows)*channels*2,poison);
  Buffer pc=Make(p.device,size_t(rows)*channels*4,0),rc=Make(p.device,size_t(rows)*channels*2*4,0);
  auto before=Sha(Data(source),source.bytes),weight_before=Sha(Data(weight),weight.bytes);NSError* error=nil;auto command=[p.queue commandBuffer];
  for(unsigned i=0;i<2;++i){auto state=[p.device newComputePipelineStateWithFunction:[lib newFunctionWithName:i?@"nr_wide_down_project":@"nr_wide_pool"] error:&error];if(!state)throw std::runtime_error("downsample pipeline");
    auto encoder=[command computeCommandEncoder];[encoder setComputePipelineState:state];auto bind=[&](Buffer& b,NSUInteger j){[encoder setBuffer:b.metal offset:Guard atIndex:j];};
    if(i==0){bind(source,0);bind(pooled,1);bind(pc,2);[encoder setBytes:&shape length:sizeof(shape) atIndex:3];}
    else{bind(pooled,0);bind(weight,1);bind(result,2);bind(rc,3);[encoder setBytes:&rows length:4 atIndex:4];}
    [encoder dispatchThreads:MTLSizeMake(size_t(rows)*channels*(i?2:1),1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];[encoder endEncoding];
  }Complete(command);GuardOutput(pooled,pc);GuardOutput(result,rc);
  if(before!=Sha(Data(source),source.bytes)||weight_before!=Sha(Data(weight),weight.bytes)||Canary(source)||Canary(weight))throw std::runtime_error("wide downsample mutation");
  Write(output/("block"+std::to_string(block)+"-pooled.e4m3"),pooled);Write(output/("block"+std::to_string(block)+"-downsample.e4m3"),result);
  std::ofstream receipt(output/("downsample"+std::to_string(block)+".json"));receipt<<"{\"pid\":"<<getpid()<<",\"poison\":"<<unsigned(poison)<<",\"input_sha256\":\""<<before<<"\",\"weight_sha256\":\""<<weight_before
    <<"\",\"pooled_sha256\":\""<<Sha(Data(pooled),pooled.bytes)<<"\",\"output_sha256\":\""<<Sha(Data(result),result.bytes)
    <<"\",\"expected_input_opened\":false,\"live_projection_input\":true,\"coverage_errors\":0,\"canary_errors\":0,\"immutable_errors\":0,\"result\":\"pass\"}\n";
  std::cout<<"downsample"<<block<<"_sha256="<<Sha(Data(result),result.bytes)<<"\n"<<std::flush;return result;
}
