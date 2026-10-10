// ANE-FEAS-001: one synthetic fixed-weight64x64 convolution, not a model port.
// MIL/API conventions derived from maderix/ANE commit
// d91c9845c0784dec7753048954fc6d0e8411fe29 (MIT, adjacent license).
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>
#import <objc/message.h>
#include <dlfcn.h>
#include <chrono>
#include <algorithm>
#include <string>
#include <vector>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <unistd.h>
using Clock=std::chrono::steady_clock;
static double Ms(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
static SEL S(const char* n){return sel_registerName(n);}
static void Save(NSString* dir,NSString* name,NSData* data){NSError* e=nil;if(![data writeToFile:[dir stringByAppendingPathComponent:name] options:NSDataWritingWithoutOverwriting error:&e])throw std::runtime_error(e.description.UTF8String);}
static void SaveCache(NSString* dir,NSString* name,NSData* data){NSString* p=[dir stringByAppendingPathComponent:name];if([[NSFileManager defaultManager] fileExistsAtPath:p]){if(![[NSData dataWithContentsOfFile:p] isEqualToData:data])throw std::runtime_error("different existing cache data; preserved");}else Save(dir,name,data);}
static IOSurfaceRef Surface(size_t bytes){return IOSurfaceCreate((__bridge CFDictionaryRef)@{(id)kIOSurfaceWidth:@(bytes),(id)kIOSurfaceHeight:@1,(id)kIOSurfaceBytesPerElement:@1,(id)kIOSurfaceBytesPerRow:@(bytes),(id)kIOSurfaceAllocSize:@(bytes),(id)kIOSurfacePixelFormat:@0});}
static id SafeValue(id object,NSString* key){@try{return [object valueForKey:key];}@catch(NSException*){return nil;}}
static NSString* PropertyType(Class cls,const char* name){objc_property_t p=class_getProperty(cls,name);return p?[NSString stringWithUTF8String:property_getAttributes(p)]:@"unavailable";}
static uint64_t HardwareTime(id request){
  uint64_t value=0;id one=SafeValue(request,@"perfStats");id array=SafeValue(request,@"perfStatsArray");
  NSMutableArray* stats=[NSMutableArray array];if([one isKindOfClass:[NSArray class]])[stats addObjectsFromArray:one];else if(one)[stats addObject:one];if([array isKindOfClass:[NSArray class]])[stats addObjectsFromArray:array];
  for(id wrapper in [stats copy])if([wrapper respondsToSelector:S("stats")]){id decoded=SafeValue(wrapper,@"stats");if(decoded)[stats addObject:decoded];}
  for(id stat in stats)if([stat respondsToSelector:S("hwExecutionTime")]){id n=SafeValue(stat,@"hwExecutionTime");if([n isKindOfClass:[NSNumber class]])value=std::max(value,uint64_t([n unsignedLongLongValue]));}
  return value;
}
int main(int argc,const char** argv){@autoreleasepool{
  if(argc!=4){fprintf(stderr,"usage: conv_probe NEW_OUTPUT fp16|fp32 POISON\n");return 2;}
  NSString* out=[NSString stringWithUTF8String:argv[1]];NSString* mode=[NSString stringWithUTF8String:argv[2]];
  if([[NSFileManager defaultManager] fileExistsAtPath:out])return 3;
  [[NSFileManager defaultManager] createDirectoryAtPath:out withIntermediateDirectories:YES attributes:nil error:nullptr];
  NSMutableDictionary* r=[@{@"schema":@"neuralstudio.nr.ane-conv-feasibility.v1",@"pid":@(getpid()),@"mode":mode,@"ane_execution_proven":@NO,@"model_ported":@NO} mutableCopy];
  IOSurfaceRef input=nullptr,output=nullptr,statsSurface=nullptr;id model=nil;int status=1;
  try {@try{
    const bool fp32=[mode isEqualToString:@"fp32"];if(!fp32 && ![mode isEqualToString:@"fp16"])throw std::runtime_error("mode");
    unsigned poison=std::stoul(argv[3],nullptr,16);if(poison!=0xa5 && poison!=0x5a)throw std::runtime_error("poison");r[@"poison"]=@(poison);
    if(!dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",RTLD_NOW|RTLD_LOCAL))throw std::runtime_error("ANE framework missing");
    Class descClass=NSClassFromString(@"_ANEInMemoryModelDescriptor"),modelClass=NSClassFromString(@"_ANEInMemoryModel"),reqClass=NSClassFromString(@"_ANERequest"),ioClass=NSClassFromString(@"_ANEIOSurfaceObject");
    if(!descClass||!modelClass||!reqClass||!ioClass)throw std::runtime_error("ANE classes missing");
    constexpr unsigned C=64,P=64,N=C*P;std::vector<float>w(C*C),x(N),y(N);
    for(unsigned n=0;n<C;++n)for(unsigned k=0;k<C;++k)w[n*C+k]=float(int((n*7+k*11)%5)-2);
    std::fill(w.begin(),w.begin()+3*C,0);w[0]=16;w[1]=1;w[C]=448;w[C+2]=-448;w[C+3]=1;w[2*C+5]=1;
    for(unsigned k=0;k<C;++k)for(unsigned p=0;p<P;++p)x[k*P+p]=float(int((k*13+p*3)%15)-7);
    for(unsigned p=0;p<P;++p){x[p]=p%2?448:128;x[P+p]=1;x[2*P+p]=x[p];x[3*P+p]=1;x[5*P+p]=p%2?std::ldexp(1.0f,-24):0;}
    Save(out,@"input.f32",[NSData dataWithBytes:x.data() length:x.size()*4]);Save(out,@"weights.f32",[NSData dataWithBytes:w.data() length:w.size()*4]);
    NSMutableString* mil=[NSMutableString stringWithString:@"program(1.3)\n[buildInfo = dict<string,string>({{\"coremlc-component-MIL\",\"3510.2.1\"},{\"coremlc-version\",\"3505.4.1\"},{\"coremltools-component-milinternal\",\"\"},{\"coremltools-version\",\"9.0\"}})]\n{\nfunc main<ios18>(tensor<fp32,[1,64,1,64]> x) {\n"];
    [mil appendString:@"string pt=const()[name=string(\"pt\"),val=string(\"valid\")];\ntensor<int32,[2]> st=const()[name=string(\"st\"),val=tensor<int32,[2]>([1,1])];\ntensor<int32,[4]> pd=const()[name=string(\"pd\"),val=tensor<int32,[4]>([0,0,0,0])];\ntensor<int32,[2]> dl=const()[name=string(\"dl\"),val=tensor<int32,[2]>([1,1])];\nint32 gr=const()[name=string(\"gr\"),val=int32(1)];\n"];
    NSMutableDictionary* weightFiles=[NSMutableDictionary dictionary];
    if(fp32){
      // Apple's MILBlob enum defines Float32=2; metadata fields are64-bit.
      std::vector<uint8_t> blob(128+C*C*4,0);blob[0]=1;blob[4]=2;blob[64]=0xef;blob[65]=0xbe;blob[66]=0xad;blob[67]=0xde;blob[68]=2;
      uint64_t size=C*C*4,offset=128;std::memcpy(blob.data()+72,&size,8);std::memcpy(blob.data()+80,&offset,8);std::memcpy(blob.data()+128,w.data(),size);
      NSData* data=[NSData dataWithBytes:blob.data() length:blob.size()];weightFiles[@"@model_path/weights/weight.bin"]=@{@"offset":@0,@"data":data};Save(out,@"weight.bin",data);
      [mil appendString:@"tensor<fp32,[64,64,1,1]> W=const()[name=string(\"W\"),val=tensor<fp32,[64,64,1,1]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"),offset=uint64(64)))];\ntensor<fp32,[1,64,1,64]> y=conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W,x=x)[name=string(\"conv_fp32\")];\n"];
    }else{
      std::vector<uint8_t> blob(128+C*C*2,0);blob[0]=1;blob[4]=2;blob[64]=0xef;blob[65]=0xbe;blob[66]=0xad;blob[67]=0xde;blob[68]=1;
      uint32_t size=C*C*2,offset=128;std::memcpy(blob.data()+72,&size,4);std::memcpy(blob.data()+80,&offset,4);
      for(unsigned i=0;i<C*C;++i){_Float16 v=(_Float16)w[i];std::memcpy(blob.data()+128+i*2,&v,2);}
      NSData* data=[NSData dataWithBytes:blob.data() length:blob.size()];weightFiles[@"@model_path/weights/weight.bin"]=@{@"offset":@0,@"data":data};Save(out,@"weight.bin",data);
      [mil appendString:@"string t16=const()[name=string(\"t16\"),val=string(\"fp16\")];\ntensor<fp16,[1,64,1,64]> x16=cast(dtype=t16,x=x)[name=string(\"cin\")];\ntensor<fp16,[64,64,1,1]> W=const()[name=string(\"W\"),val=tensor<fp16,[64,64,1,1]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"),offset=uint64(64)))];\ntensor<fp16,[1,64,1,64]> y16=conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W,x=x16)[name=string(\"conv_fp16\")];\nstring t32=const()[name=string(\"t32\"),val=string(\"fp32\")];\ntensor<fp32,[1,64,1,64]> y=cast(dtype=t32,x=y16)[name=string(\"cout\")];\n"];
    }
    [mil appendString:@"} -> (y);\n}\n"];NSData* text=[mil dataUsingEncoding:NSUTF8StringEncoding];Save(out,@"model.mil",text);
    id descriptor=((id(*)(Class,SEL,id,id,id))objc_msgSend)(descClass,S("modelWithMILText:weights:optionsPlist:"),text,weightFiles,nil);
    if(!descriptor)throw std::runtime_error("model descriptor failed");
    model=((id(*)(Class,SEL,id))objc_msgSend)(modelClass,S("inMemoryModelWithDescriptor:"),descriptor);if(!model)throw std::runtime_error("model creation failed");
    NSString* identifier=SafeValue(model,@"hexStringIdentifier");if(![identifier isKindOfClass:[NSString class]])throw std::runtime_error("model identifier missing");
    NSString* cache=[NSTemporaryDirectory() stringByAppendingPathComponent:identifier];r[@"model_cache"]=cache;
    [[NSFileManager defaultManager] createDirectoryAtPath:[cache stringByAppendingPathComponent:@"weights"] withIntermediateDirectories:YES attributes:nil error:nullptr];
    r[@"cache_existed"]=@([[NSFileManager defaultManager] fileExistsAtPath:[cache stringByAppendingPathComponent:@"model.mil"]]);
    SaveCache(cache,@"model.mil",text);SaveCache(cache,@"weights/weight.bin",weightFiles[@"@model_path/weights/weight.bin"][@"data"]);
    NSError* error=nil;auto start=Clock::now();BOOL compiled=((BOOL(*)(id,SEL,unsigned,id,NSError**))objc_msgSend)(model,S("compileWithQoS:options:error:"),21,@{},&error);
    r[@"compile_ms"]=@(Ms(start));r[@"compiled"]=@(compiled);if(!compiled)throw std::runtime_error(error?error.description.UTF8String:"compile failed without error");
    if([model respondsToSelector:S("setPerfStatsMask:")])((void(*)(id,SEL,uint32_t))objc_msgSend)(model,S("setPerfStatsMask:"),1);
    start=Clock::now();BOOL loaded=((BOOL(*)(id,SEL,unsigned,id,NSError**))objc_msgSend)(model,S("loadWithQoS:options:error:"),21,@{},&error);r[@"load_ms"]=@(Ms(start));r[@"loaded"]=@(loaded);
    if(!loaded)throw std::runtime_error(error?error.description.UTF8String:"load failed without error");
    r[@"program_loaded"]=@([SafeValue(model,@"programHandle") unsignedLongLongValue]!=0);
    input=Surface(N*4);output=Surface(N*4);if(!input||!output)throw std::runtime_error("IOSurface allocation failed");
    id in=((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(ioClass,S("objectWithIOSurface:"),input),outSurface=((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(ioClass,S("objectWithIOSurface:"),output);
    id statsObject=nil;Class statsClass=NSClassFromString(@"_ANEPerformanceStatsIOSurface");SEL statsSelector=S("objectWithIOSurface:statType:");Method statsMethod=class_getClassMethod(statsClass,statsSelector);
    if(statsMethod){char type[128]{};method_getArgumentType(statsMethod,3,type,sizeof(type));r[@"stats_factory_encoding"]=[NSString stringWithUTF8String:method_getTypeEncoding(statsMethod)];
      if(type[0]=='I'||type[0]=='Q'||type[0]=='i'||type[0]=='q'){statsSurface=Surface(65536);if(!statsSurface)throw std::runtime_error("stats surface allocation failed");IOSurfaceLock(statsSurface,0,nullptr);std::memset(IOSurfaceGetBaseAddress(statsSurface),0,65536);IOSurfaceUnlock(statsSurface,0,nullptr);
        NSString* property=PropertyType(statsClass,"stats");r[@"stats_property_type"]=property;r[@"request_perf_stats_type"]=PropertyType(reqClass,"perfStats");
        if([property containsString:@"_ANEIOSurfaceObject"]){id wrapped=((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(ioClass,S("objectWithIOSurface:"),statsSurface);statsObject=((id(*)(Class,SEL,id,long))objc_msgSend)(statsClass,statsSelector,wrapped,0);}
        else throw std::runtime_error("unconfirmed statistics surface property type");}}
    r[@"stats_object_created"]=@(statsObject!=nil);r[@"stats_type_hypothesis"]=@0;
    id request=((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(reqClass,S("requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:"),@[in],@[@0],@[outSurface],@[@0],nil,statsObject?@[statsObject]:nil,@0);
    if(!request)throw std::runtime_error("request creation failed");
    // statsWithRequestPerformanceBuffer:statsBufferSize: takes void**/uint32_t*
    // OUTPUT parameters, not an existing byte buffer and scalar length. Never
    // call it as a decoder. Use the request wrapper's observed `stats` getter.
    auto hardware=[&](){return HardwareTime(request);};
    r[@"hardware_time_before"]=[NSNull null];r[@"stats_buffer_initial_zero"]=@(statsSurface!=nullptr);NSMutableArray* timings=[NSMutableArray array];bool repeat=true,inputOK=true;std::vector<float> first;
    for(unsigned it=0;it<8;++it){
      auto total=Clock::now();auto t=Clock::now();IOSurfaceLock(input,0,nullptr);std::memcpy(IOSurfaceGetBaseAddress(input),x.data(),N*4);IOSurfaceUnlock(input,0,nullptr);
      IOSurfaceLock(output,0,nullptr);std::memset(IOSurfaceGetBaseAddress(output),int(poison),N*4);IOSurfaceUnlock(output,0,nullptr);double prepare=Ms(t);
      if(statsSurface){IOSurfaceLock(statsSurface,0,nullptr);std::memset(IOSurfaceGetBaseAddress(statsSurface),0,65536);IOSurfaceUnlock(statsSurface,0,nullptr);}prepare=Ms(t);
      t=Clock::now();BOOL ok=((BOOL(*)(id,SEL,unsigned,id,id,NSError**))objc_msgSend)(model,S("evaluateWithQoS:options:request:error:"),21,@{},request,&error);double eval=Ms(t);if(!ok)throw std::runtime_error(error?error.description.UTF8String:"evaluate failed");
      t=Clock::now();IOSurfaceLock(output,kIOSurfaceLockReadOnly,nullptr);std::memcpy(y.data(),IOSurfaceGetBaseAddress(output),N*4);IOSurfaceUnlock(output,kIOSurfaceLockReadOnly,nullptr);double read=Ms(t),roundtrip=Ms(total);
      IOSurfaceLock(input,kIOSurfaceLockReadOnly,nullptr);inputOK&=std::memcmp(x.data(),IOSurfaceGetBaseAddress(input),N*4)==0;IOSurfaceUnlock(input,kIOSurfaceLockReadOnly,nullptr);
      if(it==0)first=y;else repeat&=std::memcmp(first.data(),y.data(),N*4)==0;
      [timings addObject:@{@"iteration":@(it),@"warmup":@(it<2),@"prepare_ms":@(prepare),@"evaluate_wait_ms":@(eval),@"read_ms":@(read),@"roundtrip_ms":@(roundtrip),@"hardware_time":@(hardware())}];
    }
    Save(out,@"output.f32",[NSData dataWithBytes:y.data() length:N*4]);r[@"timings"]=timings;r[@"input_immutable"]=@(inputOK);r[@"repeat_equal"]=@(repeat);r[@"evaluated"]=@YES;
    NSUInteger mismatch=0,roundedMismatch=0;std::vector<float> expected(N);
    for(unsigned n=0;n<C;++n)for(unsigned p=0;p<P;++p){double sum=0;for(unsigned k=0;k<C;++k)sum+=double(w[n*C+k])*x[k*P+p];expected[n*P+p]=float(sum);mismatch+=y[n*P+p]!=float(sum);roundedMismatch+=y[n*P+p]!=float((_Float16)sum);}
    Save(out,@"expected-comparison-only.f32",[NSData dataWithBytes:expected.data() length:N*4]);r[@"required_fp32_mismatches"]=@(mismatch);r[@"half_rounded_mismatches"]=@(roundedMismatch);r[@"witness_expected"]=@(expected[0]);r[@"witness_observed"]=std::isfinite(y[0])?(id)@(y[0]):(id)@"nonfinite";
    r[@"hardware_time_after"]=@(hardware());r[@"perf_stats_class"]=SafeValue(request,@"perfStats")?NSStringFromClass([SafeValue(request,@"perfStats") class]):@"nil";
    bool statsChanged=false;if(statsSurface){IOSurfaceLock(statsSurface,kIOSurfaceLockReadOnly,nullptr);auto* data=static_cast<const uint8_t*>(IOSurfaceGetBaseAddress(statsSurface));for(size_t i=0;i<65536;++i)statsChanged|=data[i]!=0;Save(out,@"request-perf-stats.bin",[NSData dataWithBytes:data length:65536]);IOSurfaceUnlock(statsSurface,kIOSurfaceLockReadOnly,nullptr);}r[@"stats_buffer_changed"]=@(statsChanged);
    // Deliberately conservative: nonzero request-generated hardware timing is
    // necessary, not merely successful compilation or a changed output.
    r[@"ane_execution_proven"]=@(hardware()>0 && statsChanged);
    r[@"result"]=@"observed";status=0;
  }@catch(NSException* ex){r[@"error"]=ex.reason?:@"Objective-C exception";r[@"result"]=@"failed";}}
  catch(const std::exception& ex){r[@"error"]=[NSString stringWithUTF8String:ex.what()];r[@"result"]=@"failed";}
  if(model && [r[@"loaded"] boolValue]){NSError* e=nil;((BOOL(*)(id,SEL,unsigned,NSError**))objc_msgSend)(model,S("unloadWithQoS:error:"),21,&e);}
  if(input)CFRelease(input);if(output)CFRelease(output);if(statsSurface)CFRelease(statsSurface);
  r[@"scope"]=@"Synthetic operation only; no model weights/port. Timings exclude comparison and include host copies and synchronous eval. Hardware placement/precision must be judged from evidence, not result=observed.";
  NSError* error=nil;NSData* json=[NSJSONSerialization dataWithJSONObject:r options:NSJSONWritingPrettyPrinted error:&error];if(json)[json writeToFile:[out stringByAppendingPathComponent:@"run.json"] options:NSDataWritingWithoutOverwriting error:&error];
  return status;
}}
