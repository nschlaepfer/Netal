#define main nr_mlp_diagnostic_unused_main
#include "mlp_probe.mm"
#undef main

struct Pipelines {
  id<MTLDevice> device;id<MTLCommandQueue> queue;
  std::array<id<MTLComputePipelineState>,11> states;
};
struct WindowMap {
  uint32_t width,height,source_grid_width,source_grid_height,destination_grid_width,destination_grid_height;
  int32_t source_x,source_y,destination_x,destination_y;
};
struct Grid {uint32_t width,height;int32_t x,y;};

void Complete(id<MTLCommandBuffer> command) {
  [command commit];[command waitUntilCompleted];
  if(command.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error(command.error.localizedDescription.UTF8String);
}
void GuardOutput(Buffer& data,Buffer& counts) {
  size_t errors=Canary(data)+Canary(counts);auto* c=reinterpret_cast<uint32_t*>(Data(counts));
  for(size_t i=0;i<counts.bytes/4;++i)errors+=c[i]!=1;
  if(errors)throw std::runtime_error("output write coverage/canary mismatch");
}
Buffer Rewindow(Pipelines& p,Buffer& input,Grid from,Grid to,uint8_t poison) {
  WindowMap shape{960,576,from.width,from.height,to.width,to.height,from.x,from.y,to.x,to.y};
  if(from.x>0||from.y>0||from.width*8<uint32_t(960-from.x)||from.height*8<uint32_t(576-from.y))throw std::runtime_error("source grid does not cover image");
  size_t size=size_t(to.width)*to.height*64*32;
  Buffer output=Make(p.device,size,poison),counts=Make(p.device,size*4,0);
  const auto before=Sha(Data(input),input.bytes);
  auto command=[p.queue commandBuffer];auto encoder=[command computeCommandEncoder];
  [encoder setComputePipelineState:p.states[10]];
  [encoder setBuffer:input.metal offset:Guard atIndex:0];[encoder setBuffer:output.metal offset:Guard atIndex:1];
  [encoder setBuffer:counts.metal offset:Guard atIndex:2];[encoder setBytes:&shape length:sizeof(shape) atIndex:3];
  [encoder dispatchThreads:MTLSizeMake(size,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];[encoder endEncoding];Complete(command);
  GuardOutput(output,counts);if(before!=Sha(Data(input),input.bytes)||Canary(input))throw std::runtime_error("rewindow mutated source");
  return output;
}
Buffer Block(Pipelines& p,Buffer& input,Buffer& weight,uint32_t rows,uint8_t poison,const fs::path& output) {
  const std::array<unsigned,10> elements={128,128,128,96,64,64,64,64,32,32};
  const std::array<unsigned,10> bytes={2,1,2,2,1,2,1,2,2,1};
  const std::array<unsigned,10> dispatch={128,128,32,96,2,64,1,32,32,32};
  const std::array<const char*,10> names={"expansion.f16","activation.e4m3","partial.f16","qkv.f16","normalized.e4m3","scores.f16","probabilities.e4m3","value_partial.f16","projection.f16","output.e4m3"};
  std::vector<Buffer> y,counts;
  for(size_t i=0;i<10;++i){y.push_back(Make(p.device,size_t(rows)*elements[i]*bytes[i],poison));counts.push_back(Make(p.device,size_t(rows)*elements[i]*4,0));}
  const auto before=Sha(Data(input),input.bytes),weight_before=Sha(Data(weight),weight.bytes);
  auto command=[p.queue commandBuffer];
  for(size_t i=0;i<10;++i) {
    auto encoder=[command computeCommandEncoder];[encoder setComputePipelineState:p.states[i]];
    auto bind=[&](Buffer& b,NSUInteger j){[encoder setBuffer:b.metal offset:Guard atIndex:j];};
    auto shape=[&](NSUInteger j){[encoder setBytes:&rows length:4 atIndex:j];};
    if(i==0){bind(input,0);bind(weight,1);bind(y[0],2);bind(counts[0],3);shape(4);}
    if(i==1){bind(y[0],0);bind(y[1],1);bind(counts[1],2);shape(3);}
    if(i==2){bind(input,0);bind(y[1],1);bind(weight,2);bind(y[2],3);bind(counts[2],4);shape(5);}
    if(i==3){bind(y[2],0);bind(weight,1);bind(y[3],2);bind(counts[3],3);shape(4);}
    if(i==4){bind(y[3],0);bind(weight,1);bind(y[4],2);bind(counts[4],3);shape(4);}
    if(i==5){bind(y[4],0);bind(weight,1);bind(y[5],2);bind(counts[5],3);shape(4);}
    if(i==6){bind(y[5],0);bind(y[6],1);bind(counts[6],2);shape(3);}
    if(i==7){bind(y[6],0);bind(y[3],1);bind(y[7],2);bind(counts[7],3);shape(4);}
    if(i==8){bind(y[7],0);bind(y[2],1);bind(weight,2);bind(y[8],3);bind(counts[8],4);shape(5);}
    if(i==9){bind(y[8],0);bind(y[9],1);bind(counts[9],2);shape(3);}
    [encoder dispatchThreads:MTLSizeMake(size_t(rows)*dispatch[i],1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];[encoder endEncoding];
  }
  Complete(command);fs::create_directories(output);
  for(size_t i=0;i<10;++i) {
    GuardOutput(y[i],counts[i]);
    std::ofstream f(output/(std::string("first-window-")+names[i]),std::ios::binary);
    const unsigned chunks=i==2?4:i==7?2:1;
    const size_t stride=size_t(rows)*elements[i]*bytes[i]/chunks;
    const size_t span=size_t(64)*elements[i]*bytes[i]/chunks;
    for(unsigned chunk=0;chunk<chunks;++chunk)f.write(reinterpret_cast<char*>(Data(y[i])+chunk*stride),span);
  }
  if(before!=Sha(Data(input),input.bytes)||weight_before!=Sha(Data(weight),weight.bytes)||Canary(input)||Canary(weight))throw std::runtime_error("block input mutated");
  return y[9];
}

int main(int argc,const char** argv){@autoreleasepool{try{
  if(argc!=7)throw std::runtime_error("usage: chain_probe LIB INPUT WEIGHTS POISON OUTPUT BLOCK_COUNT");
  const uint32_t count=std::stoul(argv[6]);if(count<1||count>3)throw std::runtime_error("block count unsupported");
  const unsigned poison=std::stoul(argv[4],nullptr,16);if(poison>255)throw std::runtime_error("invalid poison");
  fs::path output(argv[5]);if(fs::exists(output))throw std::runtime_error("preserve run");fs::create_directories(output);
  Pipelines p;p.device=MTLCreateSystemDefaultDevice();p.queue=[p.device newCommandQueue];NSError* error=nil;
  auto lib=[p.device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&error];
  if(!lib)throw std::runtime_error(error.localizedDescription.UTF8String);
  NSArray<NSString*>* names=@[@"nr_swin32_expand",@"nr_swin32_activate",@"nr_swin32_contract",@"nr_swin32_qkv",@"nr_swin32_normalize",@"nr_swin32_scores",@"nr_swin32_probabilities",@"nr_swin32_values",@"nr_swin32_project",@"nr_swin32_pack",@"nr_swin32_rewindow"];
  for(size_t i=0;i<11;++i){p.states[i]=[p.device newComputePipelineStateWithFunction:[lib newFunctionWithName:names[i]] error:&error];if(!p.states[i])throw std::runtime_error("pipeline failed");}
  const auto initial=Read(argv[2]);if(initial.size()!=17694720)throw std::runtime_error("input extent");
  Buffer input=Make(p.device,initial.size(),0);std::memcpy(Data(input),initial.data(),initial.size());
  const std::array<const char*,3> hashes={"0270fb8f406d3d0a916953ddaf6ec5d29328fe95b9d6cb408ca0c25c91108e3a","379da9fb2027273f2dffa52a970a3950e97ca2d720fdf089fe4a1af409bf81c8","8f26be9b75d09960141bbf561a70e9628c269cc901934efdce362426ebbe1589"};
  const std::array<Grid,3> grids={Grid{120,72,0,0},Grid{121,73,-4,-4},Grid{121,72,-4,0}};
  std::ofstream receipt(output/"run.json");receipt<<"{\"schema\":\"neuralstudio.nr.swin-c32-live-chain.v1\",\"pid\":"<<getpid()<<",\"poison\":"<<poison<<",\"input_sha256\":\""<<Sha(initial.data(),initial.size())<<"\",\"blocks\":[";
  for(uint32_t i=0;i<count;++i){@autoreleasepool{
    auto bytes=Read(fs::path(argv[3])/("block"+std::to_string(i+1)+".bin"));
    if(bytes.size()!=20672||Sha(bytes.data(),bytes.size())!=hashes[i])throw std::runtime_error("layer weight identity");
    Buffer weight=Make(p.device,bytes.size(),0);std::memcpy(Data(weight),bytes.data(),bytes.size());
    Buffer prepared=i==0?input:Rewindow(p,input,grids[0],grids[i],poison);
    Buffer result=Block(p,prepared,weight,grids[i].width*grids[i].height*64,poison,output/("block"+std::to_string(i+1)));
    input=i==0?result:Rewindow(p,result,grids[i],grids[0],poison);
    Write(output/("block"+std::to_string(i+1)+".e4m3"),input);
    receipt<<(i?",":"")<<"{\"block\":"<<i+1<<",\"output_sha256\":\""<<Sha(Data(input),input.bytes)<<"\"}";receipt.flush();
    std::cout<<"block="<<i+1<<" output_sha256="<<Sha(Data(input),input.bytes)<<"\n"<<std::flush;
  }}
  receipt<<"],\"expected_intermediate_opened\":false,\"live_device_handoffs\":true,\"coverage_errors\":0,\"canary_errors\":0,\"immutable_errors\":0,\"result\":\"pass\"}\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}}
