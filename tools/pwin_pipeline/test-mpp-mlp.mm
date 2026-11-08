// Weight-free MLP admission test on the local Mac. No model/device FPS claim.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

static float value(size_t i, uint seed) {
  uint32_t x=uint32_t(i)+seed*0x9e3779b9u;
  x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;x^=x>>16;
  return float(int(x%1021u)-510)/2048.0f;
}
static _Float16 gelu(_Float16 h) {
  _Float16 t=std::min(std::max(h,(_Float16)-2),(_Float16)2);
  _Float16 u=(_Float16)std::fma(float(-(_Float16)0x1.4c1bacp-4f),std::abs(float(t)),float((_Float16)0x1.a60dd6p-2f));
  _Float16 s=(_Float16)std::fma(float(t),float(u),0.5f);
  return (_Float16)(float(h)*float(s));
}
static id<MTLComputePipelineState> pipeline(id<MTLDevice> dev,id<MTLLibrary> lib,const uint* c) {
  MTLFunctionConstantValues* constants=[MTLFunctionConstantValues new];
  const uint indices[]={0,1,2,3,7,14,15};
  for(int i=0;i<7;++i)[constants setConstantValue:&c[i] type:MTLDataTypeUInt atIndex:indices[i]];
  NSError* error=nil;
  id<MTLFunction> f=[lib newFunctionWithName:@"pwin_mlp2_tokens" constantValues:constants error:&error];
  if(!f){fprintf(stderr,"function: %s\n",error.description.UTF8String);exit(2);}
  id<MTLComputePipelineState> out=[dev newComputePipelineStateWithFunction:f error:&error];
  if(!out){fprintf(stderr,"pipeline: %s\n",error.description.UTF8String);exit(2);}
  return out;
}
static void run(id<MTLCommandQueue> q,id<MTLComputePipelineState> pso,
                 const std::vector<id<MTLBuffer>>& buffers,const uint* p,bool candidate) {
  id<MTLCommandBuffer> command=[q commandBuffer];
  id<MTLComputeCommandEncoder> enc=[command computeCommandEncoder];
  [enc setComputePipelineState:pso];
  for(int i=0;i<9;++i)if(i!=7)[enc setBuffer:buffers[i] offset:0 atIndex:i];
  [enc setBytes:p length:32 atIndex:7];
  uint tile=candidate?64:128;
  [enc dispatchThreadgroups:MTLSizeMake((p[0]+tile-1)/tile,1,1)
       threadsPerThreadgroup:MTLSizeMake(candidate?128:256,1,1)];
  [enc endEncoding];[command commit];[command waitUntilCompleted];
  if(command.status==MTLCommandBufferStatusError){fprintf(stderr,"GPU: %s\n",command.error.description.UTF8String);exit(2);}
}
int main(int argc,const char**argv){@autoreleasepool{
  if(argc!=4){fprintf(stderr,"usage: test-mpp-mlp candidate.metallib original.metallib cases.json\n");return 2;}
  id<MTLDevice> dev=MTLCreateSystemDefaultDevice();if(!dev)return 2;
  NSError* error=nil;
  id<MTLLibrary> candidate=[dev newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&error];
  id<MTLLibrary> original=[dev newLibraryWithURL:[NSURL fileURLWithPath:@(argv[2])] error:&error];
  if(!candidate||!original){fprintf(stderr,"library: %s\n",error.description.UTF8String);return 2;}
  NSArray* cases=[NSJSONSerialization JSONObjectWithData:[NSData dataWithContentsOfFile:@(argv[3])] options:0 error:&error];
  if(!cases)return 2;
  id<MTLCommandQueue> queue=[dev newCommandQueue];
  NSMutableArray* results=[NSMutableArray new];bool passed=true;
  for(NSDictionary* item in cases){
    uint p[8],c[7];for(int i=0;i<8;++i)p[i]=[item[@"params"][i] unsignedIntValue];
    for(int i=0;i<7;++i)c[i]=[item[@"constants"][i] unsignedIntValue];
    uint tokens=p[0],k=c[0],n=c[1],xs=c[2],cs=c[3],ys=c[4],hd=c[5],ws=c[6]?c[6]:n;
    size_t rows=((tokens+127)/128)*128,ycount=rows*ys+128;
    auto buf=[&](size_t elements,size_t bytes){return [dev newBufferWithLength:elements*bytes options:MTLResourceStorageModeShared];};
    auto x=buf(rows*xs,2),we=buf(k*hd,2),ce=buf(16*hd,4),wc=buf(hd*ws,2),ci=buf(rows*cs,2),cb=buf(n,2),scale=buf(k,2);
    auto yc=buf(ycount,2),yr=buf(ycount,2);
    _Float16*xp=(_Float16*)x.contents,*ep=(_Float16*)we.contents,*wp=(_Float16*)wc.contents;
    float*bp=(float*)ce.contents;
    _Float16*ip=(_Float16*)ci.contents,*cbp=(_Float16*)cb.contents,*sp=(_Float16*)scale.contents;
    float factor=item[@"stress"]?[item[@"stress"] floatValue]:1.0f;
    for(size_t i=0;i<x.length/2;++i)xp[i]=(_Float16)(value(i,1)*factor);
    for(size_t i=0;i<we.length/2;++i)ep[i]=(_Float16)(value(i,2)*0.5f*factor);
    for(size_t i=0;i<ce.length/4;++i)bp[i]=value(i,3)*0.5f;
    for(size_t i=0;i<wc.length/2;++i)wp[i]=(_Float16)(value(i,4)*0.5f*factor);
    for(size_t i=0;i<ci.length/2;++i)ip[i]=(_Float16)value(i,5);
    for(size_t i=0;i<cb.length/2;++i)cbp[i]=(_Float16)value(i,6);
    for(size_t i=0;i<scale.length/2;++i)sp[i]=(_Float16)(1.0f+value(i,7));
    for(size_t i=0;i<ycount;++i)((_Float16*)yc.contents)[i]=((_Float16*)yr.contents)[i]=(_Float16)23;
    std::vector<id<MTLBuffer>> buffers={x,we,ce,wc,ci,cb,yc,nil,scale};
    run(queue,pipeline(dev,candidate,c),buffers,p,true);buffers[6]=yr;
    run(queue,pipeline(dev,original,c),buffers,p,false);
    size_t count=0,different=0,canary=0,nonfinite=0;double maxref=0,maxcpu=0,sqref=0,sqcpu=0;
    std::vector<_Float16> a(k),hidden(hd);
    for(size_t r=0;r<rows;++r){
      if(r<tokens){
        for(uint z=0;z<k;++z)a[z]=p[4]?(_Float16)(float(xp[r*xs+z])*float(sp[z])):xp[r*xs+z];
        uint blk=(((uint(r)/p[2])+p[3])&7u)>>1u;
        uint rounds=p[5]?p[5]:k/16-1;
        for(uint h=0;h<hd;++h){
          float acc=bp[(r&15)*hd+h];
          for(uint kt=0;kt<k/16;++kt){
            for(uint z=0;z<16;++z)acc=std::fma(float(a[kt*16+z]),float(ep[(kt*16+z)*hd+h]),acc);
            bool s2=p[1]==2&&kt==0&&h<16&&k/16>1&&blk!=3;
            bool s56=(p[1]==5||p[1]==6)&&h<16&&kt<rounds&&(p[1]==6||(blk&1)==0);
            if(s2||s56)acc=float((_Float16)acc);
          }
          hidden[h]=gelu((_Float16)acc);
        }
      }
      for(size_t col=0;col<ys;++col){
        float actual=float(((_Float16*)yc.contents)[r*ys+col]);
        if(r>=tokens||col>=n){if(actual!=23)++canary;continue;}
        float ref=float(((_Float16*)yr.contents)[r*ys+col]);
        float acc=float((_Float16)(float(ip[r*cs+col])+float(cbp[col])));
        for(uint g=0;g<hd;g+=32){
          for(uint z=0;z<32;++z)acc=std::fma(float(hidden[g+z]),float(wp[(g+z)*ws+col]),acc);
          acc=float((_Float16)acc);
        }
        if(!std::isfinite(actual)||!std::isfinite(ref))++nonfinite;
        double er=std::abs(actual-ref),ec=std::abs(actual-acc);
        maxref=std::max(maxref,er);maxcpu=std::max(maxcpu,ec);sqref+=er*er;sqcpu+=ec*ec;
        ++count;if(actual!=ref)++different;
      }
    }
    for(size_t i=rows*ys;i<ycount;++i)if(((_Float16*)yc.contents)[i]!=(_Float16)23)++canary;
    bool ok=canary==0&&nonfinite==0&&std::isfinite(maxref)&&std::isfinite(maxcpu)&&maxref<=0.01&&maxcpu<=0.01;
    passed=passed&&ok;
    [results addObject:@{@"name":item[@"name"],@"tokens":@(tokens),@"K":@(k),@"N":@(n),@"hidden":@(hd),
      @"max_abs_original_mlp2":@(maxref),@"rms_original_mlp2":@(std::sqrt(sqref/count)),
      @"max_abs_scalar_fp32":@(maxcpu),@"rms_scalar_fp32":@(std::sqrt(sqcpu/count)),
      @"different_original_elements":@(different),@"elements":@(count),@"canary_violations":@(canary),
      @"nonfinite_elements":@(nonfinite),@"pass":@(ok)}];
  }
  NSDictionary* receipt=@{@"schema":@"netal.sr.mpp-mlp-synthetic.v1",@"device":dev.name,@"cases":results,@"pass":@(passed),
    @"mlp_backend":@"mpp-half",@"fp8":@NO,@"reference_exact":@NO,@"physical_iphone_test":@NO,
    @"model_inference_validated":@NO,@"acceleration_verified":@NO,@"max_absolute_tolerance":@0.01};
  NSData* encoded=[NSJSONSerialization dataWithJSONObject:receipt options:NSJSONWritingPrettyPrinted|NSJSONWritingSortedKeys error:&error];
  fwrite(encoded.bytes,1,encoded.length,stdout);putchar('\n');return passed?0:1;
}}
