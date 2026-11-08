// Full-K relaxed-mode numerical/ABI test on the local Mac, not iPhone.
// Original Reference/Batched test and its strict admission gates are unchanged.
// Synthetic bounded values cannot establish full-model DLSS image quality.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

static id<MTLBuffer> buffer(id<MTLDevice> dev, size_t count, size_t bytes) {
  return [dev newBufferWithLength:count * bytes options:MTLResourceStorageModeShared];
}
static float value(size_t i, uint seed) {
  uint32_t x = uint32_t(i) + seed * 0x9e3779b9u;
  x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
  return float(int(x % 1021u) - 510) / 2048.0f;
}
static id<MTLComputePipelineState> pipeline(id<MTLDevice> dev, id<MTLLibrary> lib,
                                           NSString* name, const uint* p) {
  MTLFunctionConstantValues* cv = [MTLFunctionConstantValues new];
  const uint indices[] = {0,1,2,3,7,8,9,16};
  const uint vals[] = {p[1],p[2],p[3],p[5],p[4],p[11],p[13],1};
  bool mpp = [name isEqualToString:@"pwin_gemm_mpp"];
  for (int i = 0; i < (mpp ? 2 : 8); ++i)
    [cv setConstantValue:&vals[i] type:MTLDataTypeUInt atIndex:indices[i]];
  NSError* error = nil;
  id<MTLFunction> f = [lib newFunctionWithName:name constantValues:cv error:&error];
  if (!f) { fprintf(stderr,"function: %s\n",error.description.UTF8String); exit(2); }
  id<MTLComputePipelineState> out = [dev newComputePipelineStateWithFunction:f error:&error];
  if (!out) { fprintf(stderr,"pipeline: %s\n",error.description.UTF8String); exit(2); }
  return out;
}
static void run(id<MTLCommandQueue> queue, id<MTLComputePipelineState> pso,
                const std::vector<id<MTLBuffer>>& bufs, const uint* params, bool mpp) {
  id<MTLCommandBuffer> command = [queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
  [enc setComputePipelineState:pso];
  for (int i = 0; i < 7; ++i) if (i != 4) [enc setBuffer:bufs[i] offset:0 atIndex:i];
  uint reference[] = {params[0],params[6],params[7],params[8],params[9],params[10],params[12],params[14]};
  [enc setBytes:mpp ? params : reference length:mpp ? 64 : 32 atIndex:4];
  uint columns = params[12] ? params[12] : params[2];
  MTLSize grid = mpp ? MTLSizeMake((params[0]+63)/64,(columns+31)/32,1)
                     : MTLSizeMake(((params[0]+15)/16+7)/8,1,1);
  [enc dispatchThreadgroups:grid threadsPerThreadgroup:MTLSizeMake(mpp ? 128 : 256,1,1)];
  [enc endEncoding]; [command commit]; [command waitUntilCompleted];
  if (command.status == MTLCommandBufferStatusError) {
    fprintf(stderr,"GPU: %s\n",command.error.description.UTF8String); exit(2);
  }
}
int main(int argc, const char** argv) { @autoreleasepool {
  if (argc != 3) { fprintf(stderr,"usage: test-fast-half library.metallib cases.json\n"); return 2; }
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  if (!dev) return 2;
  NSError* error = nil;
  id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&error];
  if (!lib) { fprintf(stderr,"library: %s\n",error.description.UTF8String); return 2; }
  NSArray* cases = [NSJSONSerialization JSONObjectWithData:[NSData dataWithContentsOfFile:@(argv[2])] options:0 error:&error];
  if (!cases) return 2;
  id<MTLCommandQueue> queue = [dev newCommandQueue];
  NSMutableArray* results = [NSMutableArray new];
  bool passed = true;
  for (NSDictionary* item in cases) {
    uint p[16]; NSArray* words = item[@"params"];
    if (words.count != 16) return 2;
    for (int i = 0; i < 16; ++i) p[i] = [words[i] unsignedIntValue];
    uint tokens=p[0], k=p[1], n=p[2], limit=p[12]?p[12]:n, ws=p[11]?p[11]:n;
    size_t rows=((tokens+127)/128)*128, ycount=rows*p[4]+128;
    auto x=buffer(dev,rows*p[3],2), w=buffer(dev,k*ws,2), c=buffer(dev,16*std::max(p[5],n),4);
    auto ym=buffer(dev,ycount,2), yr=buffer(dev,ycount,2);
    auto cin=buffer(dev,rows*std::max({p[5],p[13],n}),2), cb=buffer(dev,n,2);
    _Float16* xp=(_Float16*)x.contents; _Float16* wp=(_Float16*)w.contents;
    float* cp=(float*)c.contents; _Float16* ip=(_Float16*)cin.contents; _Float16* bp=(_Float16*)cb.contents;
    for (size_t i=0;i<x.length/2;++i) xp[i]=(_Float16)value(i,1);
    for (size_t i=0;i<w.length/2;++i) wp[i]=(_Float16)value(i,2);
    for (size_t i=0;i<c.length/4;++i) cp[i]=value(i,3);
    for (size_t i=0;i<cin.length/2;++i) ip[i]=(_Float16)value(i,4);
    for (size_t i=0;i<cb.length/2;++i) bp[i]=(_Float16)value(i,5);
    for (size_t i=0;i<ycount;++i) ((_Float16*)ym.contents)[i]=((_Float16*)yr.contents)[i]=(_Float16)23;
    std::vector<id<MTLBuffer>> buffers = {x,w,c,ym,nil,cin,cb};
    run(queue,pipeline(dev,lib,@"pwin_gemm_mpp",p),buffers,p,true);
    buffers[3]=yr; run(queue,pipeline(dev,lib,@"pwin_gemm3_tokens",p),buffers,p,false);
    double maxcpu=0,maxref=0,sq=0,sqref=0; size_t count=0,diff=0,canary=0,nonfinite=0;
    for (size_t r=0;r<rows;++r) for(size_t col=0;col<p[4];++col) {
      uint blk=(((uint(r)/p[9])+p[10])&7u)>>1u;
      bool writes=r<tokens && col<limit && !(p[14] && (blk&1));
      float actual=float(((_Float16*)ym.contents)[r*p[4]+col]);
      if (!writes) { if(actual!=23) ++canary; continue; }
      const float bias=p[7] ? float((_Float16)(ip[r*p[5]+col]+bp[col])) : cp[(r&15)*p[5]+col];
      float acc=0.0f;
      // Independent full-K scalar FP32 oracle over the actual FP16 inputs.
      // The relaxed candidate deliberately has no intermediate FP16 handoffs.
      for(uint z=0;z<k;++z)
        acc=std::fma(float(xp[r*p[3]+z]),float(wp[z*ws+col]),acc);
      acc+=bias;
      _Float16 h=(_Float16)acc;
      if(p[6]==1)h=std::max(h,(_Float16)0);
      else if(p[6]==2) {
        _Float16 t=std::min(std::max(h,(_Float16)-2),(_Float16)2);
        _Float16 u=(_Float16)std::fma(float(-(_Float16)0x1.4c1bacp-4f),float(std::abs(float(t))),float((_Float16)0x1.a60dd6p-2f));
        _Float16 s=(_Float16)std::fma(float(t),float(u),0.5f); h=(_Float16)(h*s);
      } else if(p[6]==3)h=(_Float16)(h+ip[r*p[13]+col]);
      float reference=float(((_Float16*)yr.contents)[r*p[4]+col]);
      if (!std::isfinite(actual) || !std::isfinite(reference)) ++nonfinite;
      double err=std::abs(actual-float(h));
      maxcpu=std::max(maxcpu,err); maxref=std::max(maxref,double(std::abs(actual-reference)));
      sq+=err*err; sqref+=double(actual-reference)*double(actual-reference); ++count; if(actual!=reference)++diff;
    }
    for(size_t i=rows*p[4];i<ycount;++i)if(((_Float16*)ym.contents)[i]!=(_Float16)23)++canary;
    bool ok=canary==0 && nonfinite==0 && std::isfinite(maxcpu) && maxcpu<=0.01;
    passed=passed&&ok;
    [results addObject:@{@"name":item[@"name"],@"K":@(k),@"N":@(n),@"tokens":@(tokens),
      @"max_abs_full_fp32_oracle":@(maxcpu),@"rms_full_fp32_oracle":@(std::sqrt(sq/count)),
      @"max_abs_original_gemm3":@(maxref),@"rms_original_gemm3":@(std::sqrt(sqref/count)),@"original_gemm3_error_is_reported_not_admitted":@YES,@"different_gemm3_elements":@(diff),@"elements":@(count),
      @"canary_violations":@(canary),@"nonfinite_elements":@(nonfinite),@"pass":@(ok)}];
  }
  NSDictionary* receipt=@{@"schema":@"netal.sr.fast-half-synthetic.v1",@"device":dev.name,@"cases":results,
    @"pass":@(passed),@"reference_exact":@NO,@"physical_iphone_test":@NO,
    @"model_inference_validated":@NO,@"intermediate_k16_handoffs":@"deliberately omitted",@"bias_order":@"dot product then real-unit bias",@"acceleration_verified":@NO,
    @"scalar_absolute_tolerance":@0.01,@"input_range":@"approximately [-0.25, 0.25]"};
  NSData* encoded=[NSJSONSerialization dataWithJSONObject:receipt options:NSJSONWritingPrettyPrinted|NSJSONWritingSortedKeys error:&error];
  fwrite(encoded.bytes,1,encoded.length,stdout);putchar('\n');return passed?0:1;
} }
