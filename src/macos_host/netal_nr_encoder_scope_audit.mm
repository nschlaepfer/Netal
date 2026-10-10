// CPU/mock audit only; no Metal device, model or game execution.
#include "../nr_experimental/nr_encoder_scope.h"
#include <cstdio>
#include <stdexcept>
#include <utility>
using namespace netal_nr_experimental;
@interface EncoderFixture : NSObject
@property(nonatomic) unsigned ends;
@property(nonatomic) BOOL failEnd;
@property(nonatomic) BOOL failEndCpp;
- (void)endEncoding;
@end
@implementation EncoderFixture
- (void)endEncoding { ++_ends; if(_failEnd) @throw [NSException exceptionWithName:@"MockEndFailure" reason:@"injected" userInfo:nil]; if(_failEndCpp)throw std::runtime_error("C++ end failure"); }
@end
int main() { @autoreleasepool {
  unsigned cases=0;auto need=[&](bool b,const char* why){++cases;if(!b)throw std::runtime_error(why);};
  try {
    FrameEncoderState s;auto e=[EncoderFixture new];
    { EncoderScope guard(s,(id<MTLCommandEncoder>)e);
      need(s.started==1,"count begins immediately"); need(guard.valid(),"valid encoder");
      need(guard.close(),"explicit close");need(guard.close(),"repeat close"); }
    need(e.ends==1,"exactly one end");
    auto cpp=[EncoderFixture new];
    try { EncoderScope guard(s,(id<MTLCommandEncoder>)cpp); throw std::runtime_error("stage error"); }
    catch(const std::runtime_error&) {}
    need(cpp.ends==1,"C++ exception closes encoder");
    auto objc=[EncoderFixture new];
    @try { EncoderScope guard(s,(id<MTLCommandEncoder>)objc); @throw [NSException exceptionWithName:@"StageError" reason:nil userInfo:nil]; }
    @catch(NSException* exception) { (void)exception; }
    need(objc.ends==1,"Objective-C exception closes encoder");
    auto returned=[EncoderFixture new];
    auto body=[&](){try {EncoderScope guard(s,(id<MTLCommandEncoder>)returned);throw std::runtime_error("caught inside body");}catch(...){return 2;}};
    need(body()==2 && returned.ends==1,"body converts exception to return after cleanup");
    auto early=[EncoderFixture new];auto normal=[&](){EncoderScope guard(s,(id<MTLCommandEncoder>)early);return false;};
    need(!normal() && early.ends==1,"ordinary early return closes encoder");
    const auto count=s.started;{EncoderScope guard(s,nil);need(!guard.valid(),"nil encoder");}
    need(s.started==count,"nil does not count work");
    FrameEncoderState other;auto a=[EncoderFixture new],b=[EncoderFixture new];
    {EncoderScope ga(s,(id<MTLCommandEncoder>)a);{EncoderScope gb(other,(id<MTLCommandEncoder>)b);}
      need(b.ends==1 && a.ends==0,"independent invocation cannot close another encoder");}
    need(a.ends==1 && other.started==1 && !other.cleanup_failed,"independent state");
    FrameEncoderState bad;auto fail=[EncoderFixture new];fail.failEnd=YES;
    {EncoderScope guard(bad,(id<MTLCommandEncoder>)fail);need(!guard.close(),"failed close signalled");}
    need(bad.cleanup_failed && fail.ends==1,"failed close not retried or hidden");
    FrameEncoderState badCpp;auto failCpp=[EncoderFixture new];failCpp.failEndCpp=YES;
    {EncoderScope guard(badCpp,(id<MTLCommandEncoder>)failCpp);}
    need(badCpp.cleanup_failed && failCpp.ends==1,"C++ cleanup exception contained and reported");
    FrameEncoderState moved;auto moveFixture=[EncoderFixture new];
    {EncoderScope first(moved,(id<MTLCommandEncoder>)moveFixture);EncoderScope second(std::move(first));
      need(!first.valid() && second.get()==(id<MTLCommandEncoder>)moveFixture,"factory ownership transfer");}
    need(moved.started==1 && moveFixture.ends==1,"move counts and closes once");
    std::printf("result=PASS cases=%u GPU_execution=0 runtime_adoption_not_proven=1\n",cases);return 0;
  }catch(const std::exception& e){std::fprintf(stderr,"result=FAIL detail=%s\n",e.what());return 1;}
} }
