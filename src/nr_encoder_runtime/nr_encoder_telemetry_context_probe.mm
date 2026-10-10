#define main nr_unused_legacy_probe_main
#include "nr_encoder_runtime_probe.mm"
#undef main
#include "nr_encoder_telemetry.h"

static void Require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
int main(int argc,const char**argv){@autoreleasepool{try{
  Require(argc==7,"usage: probe ASSETS IMAGE0 REFERENCE0 IMAGE1 REFERENCE1 OUTPUT");
  fs::path out=argv[6];Require(!fs::exists(out),"preserve context test");fs::create_directories(out);
  NrEncoderContext a,b;std::string error,payload;
  Require(nr_encoder_prepare(a,argv[1],error)==0,"prepare A");
  b.device=a.device;b.queue=a.queue;Require(nr_encoder_prepare(b,argv[1],error)==0,"prepare independent B");
  Require(a.state!=b.state,"contexts unexpectedly alias");
  Require(nr_encoder_enable_telemetry(a,true,error)==0&&nr_encoder_enable_telemetry(b,true,error)==0,"enable both contexts");
  Require(nr_encoder_copy_telemetry(a,payload,error)==2&&nr_encoder_copy_telemetry(b,payload,error)==2,"empty prepared context exposed telemetry");
  std::array<NrEncoderResult,2> results;std::array<std::string,2> snapshots;
  NSMutableArray* rows=[NSMutableArray array];
  for(unsigned i=0;i<2;++i){
    auto& ctx=i?b:a;auto bytes=Read(argv[i?4:2]);auto texture=Texture(ctx.device,bytes);NrEncoderFrame frame;frame.color=texture;
    Require(nr_encoder_run_frame(ctx,frame,i?0x5a:0xa5,results[i],nullptr)==0,"checked producer failed");
    Require(results[i].guarded_outputs_checked==1039&&results[i].coverage_values_checked==5900107776ull,"coverage changed");
    Require(nr_encoder_copy_telemetry(ctx,snapshots[i],error)==0,"successful context lacks snapshot");
    NSData* data=[NSData dataWithBytes:snapshots[i].data() length:snapshots[i].size()];NSDictionary* profile=[NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
    auto input=Sha(bytes.data(),bytes.size());
    Require([profile[@"generation"] unsignedLongLongValue]==1&&[profile[@"input_sha256"] isEqualToString:S(input)],"wrong context generation/input");
    fs::path ref=argv[i?5:3];
    for(const auto& [name,region]:Ports(results[i])){
      auto expected=Read(ref/(name=="block0-skip"?"block0/skip.e4m3":"encoder/"+name+".e4m3"));
      Require(expected.size()==region.bytes&&!std::memcmp(expected.data(),Data(region),region.bytes),"context producer differs from RTX reference");
    }
    std::ofstream f(out/("context"+std::to_string(i)+".json"));f<<snapshots[i];Require(bool(f),"snapshot write");
    [rows addObject:@{@"context":@(i),@"input_sha256":S(input),@"generation":@1,@"ports_match_rtx":@YES}];
    if(i==0)Require(nr_encoder_copy_telemetry(b,payload,error)==2&&payload.empty(),"prepared B borrowed A's snapshot");
  }
  Require(nr_encoder_copy_telemetry(a,payload,error)==0&&payload==snapshots[0],"B execution replaced A's snapshot");
  Require(nr_encoder_enable_telemetry(b,false,error)==0&&nr_encoder_copy_telemetry(b,payload,error)==2,"disable B did not clear B");
  Require(nr_encoder_copy_telemetry(a,payload,error)==0&&payload==snapshots[0],"disable B cleared A");
  NrEncoderFrame bad;NrEncoderResult rejected;Require(nr_encoder_run_frame(a,bad,0xa5,rejected,nullptr)==2,"bad frame admitted");
  Require(nr_encoder_copy_telemetry(a,payload,error)==2&&payload.empty(),"admitted rejection retained a stale snapshot");
  for(const auto& r:results)for(const auto& pair:Ports(r))Require(Sha(Data(pair.second),pair.second.bytes)==pair.second.sha256,"retained result lifetime changed");
  NSDictionary* report=@{@"schema":@"neuralstudio.nr.encoder-telemetry-context-test.v1",@"pid":@(getpid()),@"frames":rows,@"two_prepared_states":@YES,@"cross_context_isolation":@YES,@"disable_is_context_local":@YES,@"rejection_clears_snapshot":@YES,@"prior_results_intact":@YES,@"expected_gpu_input":@NO,@"result":@"pass"};
  NSData* data=[NSJSONSerialization dataWithJSONObject:report options:NSJSONWritingPrettyPrinted error:nil];Require(data&&[data writeToFile:S((out/"run.json").string()) atomically:YES],"receipt write");
  std::cout<<"context isolation and exact producer ports passed\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 2;}}}
