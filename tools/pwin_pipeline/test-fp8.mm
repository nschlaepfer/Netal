// Independent synthetic FP8 quantization/matmul oracle. No model or game data.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

static float value(size_t i,uint seed){uint32_t x=uint32_t(i)+seed*0x9e3779b9u;x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;x^=x>>16;return float(int(x%1021u)-510)/2048.0f;}
static float decode(uint8_t b){if((b&127)==127)return NAN;uint e=(b>>3)&15,m=b&7;float v=e?std::ldexp(1.0f+float(m)/8,int(e)-7):std::ldexp(float(m),-9);return (b&128)?-v:v;}
static uint8_t encode(float v){
  if(!std::isfinite(v))return 127;
  uint sign=std::signbit(v)?128:0;float a=std::abs(v);uint best=0;float error=INFINITY;
  for(uint b=0;b<127;++b){float d=std::abs(a-decode(b));if(d<error||(d==error&&(b&1)==0)){error=d;best=b;}}
  return uint8_t(sign|best);
}
static _Float16 gelu(_Float16 h){_Float16 t=std::min(std::max(h,(_Float16)-2),(_Float16)2);_Float16 u=(_Float16)std::fma(float(-(_Float16)0x1.4c1bacp-4f),std::abs(float(t)),float((_Float16)0x1.a60dd6p-2f));_Float16 s=(_Float16)std::fma(float(t),float(u),0.5f);return (_Float16)(float(h)*float(s));}
static id<MTLComputePipelineState> pipeline(id<MTLDevice>d,id<MTLLibrary>l,NSString*name,const uint*p){
  NSError*error=nil;id<MTLFunction>f=nil;
  if([name isEqualToString:@"pwin_quantize_rows_e4m3"])f=[l newFunctionWithName:name];
  else{
    MTLFunctionConstantValues*cv=[MTLFunctionConstantValues new];
    bool native=[name isEqualToString:@"pwin_gemm_fp8_e4m3"];
    const uint indices[]={0,1,2,3,7,8,9,16},vals[]={p[1],p[2],p[3],p[5],p[4],p[11],p[13],1};
    for(int i=0;i<(native?2:8);++i)[cv setConstantValue:&vals[i] type:MTLDataTypeUInt atIndex:indices[i]];
    f=[l newFunctionWithName:name constantValues:cv error:&error];
  }
  if(!f){fprintf(stderr,"function: %s\n",error.description.UTF8String);exit(2);}
  id<MTLComputePipelineState>o=[d newComputePipelineStateWithFunction:f error:&error];
  if(!o){fprintf(stderr,"pipeline: %s\n",error.description.UTF8String);exit(2);}return o;
}
static void execute(id<MTLCommandQueue>q,id<MTLComputePipelineState>pso,const std::vector<id<MTLBuffer>>&b,
                      const uint*p,bool quant,bool native){
  id<MTLCommandBuffer>cmd=[q commandBuffer];id<MTLComputeCommandEncoder>e=[cmd computeCommandEncoder];[e setComputePipelineState:pso];
  uint skip=quant?3:4;for(uint i=0;i<b.size();++i)if(i!=skip&&b[i])[e setBuffer:b[i] offset:0 atIndex:i];
  uint ref[8]={};if(!quant){uint words[]={p[0],p[6],p[7],p[8],p[9],p[10],p[12],p[14]};std::copy(words,words+8,ref);}
  [e setBytes:quant?p:(native?p:ref) length:quant?32:(native?64:32) atIndex:skip];
  uint cols=quant?0:(p[12]?p[12]:p[2]);
  MTLSize grid=quant?MTLSizeMake(p[5]/4,1,1):(native?MTLSizeMake((p[0]+63)/64,(cols+31)/32,1):MTLSizeMake(((p[0]+15)/16+7)/8,1,1));
  [e dispatchThreadgroups:grid threadsPerThreadgroup:MTLSizeMake(quant||native?128:256,1,1)];
  [e endEncoding];[cmd commit];[cmd waitUntilCompleted];
  if(cmd.status==MTLCommandBufferStatusError){fprintf(stderr,"GPU: %s\n",cmd.error.description.UTF8String);exit(2);}
}
int main(int argc,const char**argv){@autoreleasepool{
  if(argc!=3){fprintf(stderr,"usage: test-fp8 candidate.metallib cases.json\n");return 2;}
  id<MTLDevice>d=MTLCreateSystemDefaultDevice();if(!d)return 2;NSError*error=nil;
  id<MTLLibrary>l=[d newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&error];if(!l){fprintf(stderr,"library: %s\n",error.description.UTF8String);return 2;}
  NSArray*cases=[NSJSONSerialization JSONObjectWithData:[NSData dataWithContentsOfFile:@(argv[2])] options:0 error:&error];if(!cases)return 2;
  id<MTLCommandQueue>queue=[d newCommandQueue];NSMutableArray*results=[NSMutableArray new];bool passed=true;
  for(NSDictionary*item in cases){
    uint p[16];for(int i=0;i<16;++i)p[i]=[item[@"params"][i] unsignedIntValue];
    uint t=p[0],k=p[1],n=p[2],kp=(k+31)/32*32,np=(n+31)/32*32,as=(k+127)/128*128,rows=(t+127)/128*128,qr=(t+63)/64*64,ws=p[11]?p[11]:n,limit=p[12]?p[12]:n;
    size_t ycount=size_t(rows)*p[4]+128;
    auto buf=[&](size_t bytes){return[d newBufferWithLength:bytes options:MTLResourceStorageModeShared];};
    auto x=buf(size_t(rows)*p[3]*2),w=buf(size_t(k)*ws*2),c=buf(16*std::max(p[5],n)*4),yc=buf(ycount*2),yr=buf(ycount*2);
    auto ci=buf(size_t(rows)*std::max({p[5],p[13],n})*2),cb=buf(n*2),aq=buf(size_t(qr)*as+128),sa=buf(qr*4+128),wq=buf(size_t(np)*as),sw=buf(np*4);
    _Float16*xp=(_Float16*)x.contents,*wp=(_Float16*)w.contents,*ip=(_Float16*)ci.contents,*bp=(_Float16*)cb.contents;float*cp=(float*)c.contents;
    for(size_t i=0;i<x.length/2;++i)xp[i]=(_Float16)value(i,1);
    for(size_t i=0;i<w.length/2;++i)wp[i]=(_Float16)value(i,2);
    for(size_t i=0;i<c.length/4;++i)cp[i]=value(i,3);
    for(size_t i=0;i<ci.length/2;++i)ip[i]=(_Float16)value(i,4);
    for(size_t i=0;i<cb.length/2;++i)bp[i]=(_Float16)value(i,5);
    bool codec=[item[@"codec"] boolValue];
    if(codec){for(uint r=0;r<t;++r)for(uint z=0;z<k;++z){uint b=(r*k+z)%126;float v=z==0?448.0f:((r&1)?-decode(b):decode(b));if((r&2)&&z!=0)v=(decode(b)+decode(b+1))*0.5f;xp[r*p[3]+z]=(_Float16)v;}}
    else for(uint r=0;r<t;r+=7)for(uint z=0;z<k;++z)xp[r*p[3]+z]=(_Float16)0;
    memset(aq.contents,0xa5,aq.length);memset(sa.contents,0xa5,sa.length);memset(wq.contents,0,wq.length);
    float*swp=(float*)sw.contents;uint8_t*wqp=(uint8_t*)wq.contents;
    for(uint col=0;col<np;++col){float amax=0;for(uint z=0;z<k&&col<n;++z)amax=std::max(amax,std::abs(float(wp[z*ws+col])));swp[col]=amax>0?amax/448.0f:1.0f;if(col<n)for(uint z=0;z<k;++z)wqp[col*as+z]=encode(float(wp[z*ws+col])/swp[col]);}
    uint qp[]={t,k,p[3],as,kp,qr,0,0};std::vector<id<MTLBuffer>>qb={x,aq,sa};
    execute(queue,pipeline(d,l,@"pwin_quantize_rows_e4m3",p),qb,qp,true,false);
    uint8_t*ap=(uint8_t*)aq.contents;float*sap=(float*)sa.contents;
    size_t qdiff=0,scaleerr=0,qcanary=0;
    for(uint r=0;r<qr;++r){float amax=0;if(r<t)for(uint z=0;z<k;++z)amax=std::max(amax,std::abs(float(xp[r*p[3]+z])));float scale=amax>0?amax/448.0f:1.0f;if(sap[r]!=scale)++scaleerr;
      for(uint z=0;z<as;++z){uint8_t expected=r<t&&z<k?encode(float(xp[r*p[3]+z])/scale):0;if(ap[r*as+z]!=expected)++qdiff;}}
    for(size_t i=size_t(qr)*as;i<aq.length;++i)if(ap[i]!=0xa5)++qcanary;
    for(size_t i=size_t(qr)*4;i<sa.length;++i)if(((uint8_t*)sa.contents)[i]!=0xa5)++qcanary;
    if(codec){bool ok=qdiff==0&&scaleerr==0&&qcanary==0;passed=passed&&ok;[results addObject:@{@"name":item[@"name"],@"codec_only":@YES,@"quantized_byte_mismatches":@(qdiff),@"scale_mismatches":@(scaleerr),@"canary_violations":@(qcanary),@"pass":@(ok)}];continue;}
    for(size_t i=0;i<ycount;++i)((_Float16*)yc.contents)[i]=((_Float16*)yr.contents)[i]=(_Float16)23;
    uint fp[16];std::copy(p,p+16,fp);fp[3]=as;fp[11]=as;fp[15]=kp;
    std::vector<id<MTLBuffer>>fb={aq,wq,c,yc,nil,ci,cb,sa,sw},rb={x,w,c,yr,nil,ci,cb};
    execute(queue,pipeline(d,l,@"pwin_gemm_fp8_e4m3",fp),fb,fp,false,true);
    execute(queue,pipeline(d,l,@"pwin_gemm3_tokens",p),rb,p,false,false);
    double maxcpu=0,maxref=0,sqcpu=0,sqref=0;size_t count=0,canary=qcanary,nonfinite=0;
    for(uint r=0;r<rows;++r)for(uint col=0;col<p[4];++col){
      uint blk=(((r/p[9])+p[10])&7u)>>1u;bool writes=r<t&&col<limit&&!(p[14]&&(blk&1));float actual=float(((_Float16*)yc.contents)[r*p[4]+col]);
      if(!writes){if(actual!=23)++canary;continue;}
      float raw=0;for(uint z=0;z<kp;++z)raw=std::fma(decode(ap[r*as+z]),decode(wqp[col*as+z]),raw);
      float bias=p[7]?float((_Float16)(float(ip[r*p[5]+col])+float(bp[col]))):cp[(r&15)*p[5]+col];
      _Float16 h=(_Float16)((raw*sap[r])*swp[col]+bias);
      if(p[6]==1)h=std::max(h,(_Float16)0);else if(p[6]==2)h=gelu(h);else if(p[6]==3)h=(_Float16)(float(h)+float(ip[r*p[13]+col]));
      float ref=float(((_Float16*)yr.contents)[r*p[4]+col]);if(!std::isfinite(actual)||!std::isfinite(ref))++nonfinite;
      double ec=std::abs(actual-float(h)),er=std::abs(actual-ref);maxcpu=std::max(maxcpu,ec);maxref=std::max(maxref,er);sqcpu+=ec*ec;sqref+=er*er;++count;
    }
    for(size_t i=size_t(rows)*p[4];i<ycount;++i)if(((_Float16*)yc.contents)[i]!=(_Float16)23)++canary;
    bool ok=qdiff==0&&scaleerr==0&&canary==0&&nonfinite==0&&std::isfinite(maxcpu)&&maxcpu<=0.01;passed=passed&&ok;
    [results addObject:@{@"name":item[@"name"],@"K":@(k),@"N":@(n),@"tokens":@(t),@"max_abs_real_quant_oracle":@(maxcpu),@"rms_real_quant_oracle":@(std::sqrt(sqcpu/count)),
      @"max_abs_original_fp16":@(maxref),@"rms_original_fp16":@(std::sqrt(sqref/count)),@"original_fp16_error_is_reported_not_admitted":@YES,
      @"quantized_byte_mismatches":@(qdiff),@"scale_mismatches":@(scaleerr),@"canary_violations":@(canary),@"nonfinite_elements":@(nonfinite),@"elements":@(count),@"pass":@(ok)}];
  }
  NSDictionary*r=@{@"schema":@"netal.sr.fp8-synthetic.v1",@"device":d.name,@"cases":results,@"pass":@(passed),@"fp8":@YES,@"format":@"E4M3FN",
    @"reference_exact":@NO,@"intermediate_k16_handoffs":@"deliberately omitted",@"real_quant_oracle_tolerance":@0.01,
    @"model_quality_admitted":@NO,@"physical_iphone_test":@NO,@"native_instruction_path_verified":@NO};
  NSData*out=[NSJSONSerialization dataWithJSONObject:r options:NSJSONWritingPrettyPrinted|NSJSONWritingSortedKeys error:&error];fwrite(out.bytes,1,out.length,stdout);putchar('\n');return passed?0:1;
}}
