#define main nr_mlp_diagnostic_unused_main
#include "../nr_swin/mlp_probe.mm"
#undef main

struct PreShape {uint32_t width,height,padded_height,frame,phase_math;};
struct PreChunk {uint32_t rows,base_window;};
void Complete(id<MTLCommandBuffer> command){[command commit];[command waitUntilCompleted];if(command.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error(command.error.localizedDescription.UTF8String);}
void GuardOutput(Buffer& data,Buffer& counts){size_t errors=Canary(data)+Canary(counts);auto*c=reinterpret_cast<uint32_t*>(Data(counts));for(size_t i=0;i<counts.bytes/4;++i)errors+=c[i]!=1;if(errors)throw std::runtime_error("write coverage/canary failure");}
int main(int argc,const char**argv){@autoreleasepool{try{
  if(argc!=7&&argc!=9)throw std::runtime_error("usage: pre_probe LIB IMAGE WEIGHTS EXPECTED_POOL POISON OUTPUT [MATH_DELTAS MATH_ESCAPES]");
  unsigned poison=std::stoul(argv[5],nullptr,16);if(poison>255)throw std::runtime_error("poison");fs::path out(argv[6]);
  if(fs::exists(out))throw std::runtime_error("preserve run");fs::create_directories(out);
  auto image=Read(argv[2]),weights=Read(argv[3]);
  if(image.size()!=1920*1080*8||Sha(image.data(),image.size())!="81ec1de5d1e76e6b55c4fd987b3732180c9720919fbd760dd5b4e8aa8cdd0342")throw std::runtime_error("pinned image");
  if(weights.size()!=21696||Sha(weights.data(),weights.size())!="5fe2ab86289f7b21253d00368ce7a95f005ec3c8476abd7db22550b918962a5c")throw std::runtime_error("pinned weight");
  id<MTLDevice> device=MTLCreateSystemDefaultDevice();auto queue=[device newCommandQueue];NSError*error=nil;
  auto lib=[device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&error];if(!lib)throw std::runtime_error(error.localizedDescription.UTF8String);
  NSArray<NSString*>* names=@[@"nr_pre_features",@"nr_pre_adapter",@"nr_swin32_expand",@"nr_swin32_activate",@"nr_swin32_contract",@"nr_swin32_qkv",@"nr_swin32_normalize",@"nr_swin32_scores",@"nr_swin32_probabilities",@"nr_swin32_values",@"nr_swin32_project",@"nr_pre_export"];
  std::array<id<MTLComputePipelineState>,12> pipeline;
  for(size_t i=0;i<12;++i){pipeline[i]=[device newComputePipelineStateWithFunction:[lib newFunctionWithName:names[i]] error:&error];if(!pipeline[i])throw std::runtime_error(error.localizedDescription.UTF8String);}
  Buffer input=Make(device,image.size(),0),weight=Make(device,weights.size(),0),features=Make(device,70778880,poison),fc=Make(device,141557760,0);
  std::memcpy(Data(input),image.data(),image.size());std::memcpy(Data(weight),weights.data(),weights.size());
  auto inputHash=Sha(Data(input),input.bytes),weightHash=Sha(Data(weight),weight.bytes);PreShape shape{1920,1080,1152,0,1};
  Buffer deltas{},escapes{};uint32_t escapeCount=0;std::string deltaHash,escapeHash;
  if(argc==9){auto d=Read(argv[7]),e=Read(argv[8]);deltaHash=Sha(d.data(),d.size());escapeHash=Sha(e.data(),e.size());
    if(d.size()!=67108864||deltaHash!="768da747f3c7c799d768cd4410195934292cc838fbc083ec6a3ab9c04184f37b"||e.size()!=2780432||escapeHash!="c2e095109b1f220ed289ea09f5b2d10a4df9f15f5366355e7fdab30b455d40f5")throw std::runtime_error("math correction identity");
    deltas=Make(device,d.size(),0);escapes=Make(device,e.size(),0);std::memcpy(Data(deltas),d.data(),d.size());std::memcpy(Data(escapes),e.data(),e.size());escapeCount=uint32_t(e.size()/8);}
  auto command=[queue commandBuffer];auto enc=[command computeCommandEncoder];[enc setComputePipelineState:pipeline[0]];
  [enc setBuffer:input.metal offset:Guard atIndex:0];[enc setBuffer:features.metal offset:Guard atIndex:1];[enc setBuffer:fc.metal offset:Guard atIndex:2];[enc setBytes:&shape length:sizeof(shape) atIndex:3];
  if(argc==9){[enc setBuffer:deltas.metal offset:Guard atIndex:4];[enc setBuffer:escapes.metal offset:Guard atIndex:5];[enc setBytes:&escapeCount length:4 atIndex:6];}
  [enc dispatchThreads:MTLSizeMake(1920*1152,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[enc endEncoding];Complete(command);GuardOutput(features,fc);
  auto featureHash=Sha(Data(features),features.bytes);Write(out/"features-shared.f16",features);fc.metal=nil;
  Buffer skip=Make(device,70778880,poison),encoder=Make(device,17694720,poison),sc=Make(device,283115520,0),ec=Make(device,70778880,0);
  constexpr uint32_t Windows=240*144,Chunk=512;
  const std::array<unsigned,9> elements={128,128,128,96,64,64,64,64,32};
  const std::array<unsigned,9> sizes={2,1,2,2,1,2,1,2,2};
  const std::array<unsigned,9> dispatch={128,128,32,96,2,64,1,32,32};
  const std::array<const char*,9> stageNames={"expand.f16","activation.e4m3","ffn-partial.f16","qkv.f16","normalized.e4m3","scores.f16","probabilities.e4m3","value-partial.f16","projection.f16"};
  for(uint32_t base=0;base<Windows;base+=Chunk){@autoreleasepool{
    PreChunk s{std::min(Chunk,Windows-base)*64,base};uint32_t rows=s.rows;
    Buffer raw=Make(device,size_t(rows)*64,poison),packed=Make(device,size_t(rows)*32,poison),rc=Make(device,size_t(rows)*128,0),pc=Make(device,size_t(rows)*128,0);
    std::vector<Buffer> y,c;for(size_t i=0;i<9;++i){y.push_back(Make(device,size_t(rows)*elements[i]*sizes[i],poison));c.push_back(Make(device,size_t(rows)*elements[i]*4,0));}
    auto cmd=[queue commandBuffer];auto adapter=[cmd computeCommandEncoder];[adapter setComputePipelineState:pipeline[1]];
    std::array<Buffer*,6> ab={&features,&weight,&raw,&packed,&rc,&pc};for(size_t i=0;i<ab.size();++i)[adapter setBuffer:ab[i]->metal offset:Guard atIndex:i];
    [adapter setBytes:&s length:sizeof(s) atIndex:6];[adapter dispatchThreads:MTLSizeMake(size_t(rows)*32,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];[adapter endEncoding];
    for(size_t i=0;i<9;++i){auto e=[cmd computeCommandEncoder];[e setComputePipelineState:pipeline[i+2]];
      auto bind=[&](Buffer&b,NSUInteger j){[e setBuffer:b.metal offset:Guard atIndex:j];};auto bindShape=[&](NSUInteger j){[e setBytes:&rows length:4 atIndex:j];};
      if(i==0){bind(packed,0);bind(weight,1);bind(y[0],2);bind(c[0],3);bindShape(4);}
      if(i==1){bind(y[0],0);bind(y[1],1);bind(c[1],2);bindShape(3);}
      if(i==2){bind(raw,0);bind(y[1],1);bind(weight,2);bind(y[2],3);bind(c[2],4);bindShape(5);}
      if(i==3){bind(y[2],0);bind(weight,1);bind(y[3],2);bind(c[3],3);bindShape(4);}
      if(i==4){bind(y[3],0);bind(weight,1);bind(y[4],2);bind(c[4],3);bindShape(4);}
      if(i==5){bind(y[4],0);bind(weight,1);bind(y[5],2);bind(c[5],3);bindShape(4);}
      if(i==6){bind(y[5],0);bind(y[6],1);bind(c[6],2);bindShape(3);}
      if(i==7){bind(y[6],0);bind(y[3],1);bind(y[7],2);bind(c[7],3);bindShape(4);}
      if(i==8){bind(y[7],0);bind(y[2],1);bind(weight,2);bind(y[8],3);bind(c[8],4);bindShape(5);}
      [e dispatchThreads:MTLSizeMake(size_t(rows)*dispatch[i],1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];[e endEncoding];
    }
    auto e=[cmd computeCommandEncoder];[e setComputePipelineState:pipeline[11]];std::array<Buffer*,5> eb={&y[8],&skip,&encoder,&sc,&ec};
    for(size_t i=0;i<eb.size();++i)[e setBuffer:eb[i]->metal offset:Guard atIndex:i];[e setBytes:&s length:sizeof(s) atIndex:5];
    [e dispatchThreads:MTLSizeMake(size_t(rows)*32,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];[e endEncoding];Complete(cmd);
    GuardOutput(raw,rc);GuardOutput(packed,pc);for(size_t i=0;i<9;++i)GuardOutput(y[i],c[i]);
    if(base==0){
      auto first=[&](const std::string&name,Buffer&b,size_t span,unsigned chunks){std::ofstream f(out/name,std::ios::binary);for(unsigned j=0;j<chunks;++j)f.write(reinterpret_cast<char*>(Data(b)+j*b.bytes/chunks),span);if(!f)throw std::runtime_error("first window write");};
      first("first-window-adapter.f16",raw,64*32*2,1);first("first-window-adapter.e4m3",packed,64*32,1);
      for(size_t i=0;i<9;++i){unsigned chunks=i==2?4:i==7?2:1;first(std::string("first-window-")+stageNames[i],y[i],size_t(64)*elements[i]*sizes[i]/chunks,chunks);}
    }
    std::cout<<"windows="<<base+rows/64<<"/"<<Windows<<"\n"<<std::flush;
  }}
  GuardOutput(skip,sc);GuardOutput(encoder,ec);Write(out/"skip.e4m3",skip);Write(out/"encoder.e4m3",encoder);Write(out/"skip.coverage.bin",sc);Write(out/"encoder.coverage.bin",ec);
  size_t immutable=(inputHash!=Sha(Data(input),input.bytes))+(weightHash!=Sha(Data(weight),weight.bytes))+(featureHash!=Sha(Data(features),features.bytes));
  size_t canary=Canary(input)+Canary(weight)+Canary(features);
  if(argc==9){canary+=Canary(deltas)+Canary(escapes);immutable+=(deltaHash!=Sha(Data(deltas),deltas.bytes))+(escapeHash!=Sha(Data(escapes),escapes.bytes));}
  // Comparison-only oracle is opened after every live native producer completes.
  auto expected=Read(argv[4]);if(expected.size()!=212519936||Sha(expected.data(),expected.size())!="1e340ab4b04af0107bba1a62e1667e304f90d62123651f019efbf1abc77a01db")throw std::runtime_error("comparison pool identity");
  size_t sm=0,em=0;for(size_t i=0;i<skip.bytes;++i)sm+=Data(skip)[i]!=expected[388096+i];for(size_t i=0;i<encoder.bytes;++i)em+=Data(encoder)[i]!=expected[71166976+i];
  std::ofstream receipt(out/"run.json");receipt<<"{\"schema\":\"neuralstudio.nr.pre-native-candidate.v1\",\"pid\":"<<getpid()<<",\"poison\":"<<poison
    <<",\"input_sha256\":\""<<inputHash<<"\",\"weight_sha256\":\""<<weightHash<<"\",\"feature_sha256\":\""<<featureHash<<"\",\"skip_sha256\":\""<<Sha(Data(skip),skip.bytes)
    <<"\",\"encoder_sha256\":\""<<Sha(Data(encoder),encoder.bytes)<<"\",\"skip_mismatching_bytes\":"<<sm<<",\"encoder_mismatching_bytes\":"<<em
    <<",\"coverage_errors\":0,\"canary_errors\":"<<canary<<",\"immutable_errors\":"<<immutable<<",\"expected_gpu_input\":false,\"live_device_handoffs\":true,\"instruction_corrections\":"<<(argc==9?"true":"false")
    <<",\"delta_sha256\":\""<<deltaHash<<"\",\"escape_sha256\":\""<<escapeHash<<"\",\"candidate_only\":true,\"result\":\""<<((sm|em|immutable|canary)?"fail":"pass")<<"\"}\n";
  if(!receipt)throw std::runtime_error("receipt write");std::cout<<"skip_mismatch="<<sm<<" encoder_mismatch="<<em<<"\n";return (sm|em|immutable|canary)?3:0;
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}}
