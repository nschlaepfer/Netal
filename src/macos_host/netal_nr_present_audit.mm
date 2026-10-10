// CPU mocks: validate observer forwarding/ownership without presenting anything.
#define NETAL_NR_OBSERVE_ONLY 1
#include "netal_nr_experimental_bridge.mm"
#include <stdexcept>
@interface PresentDrawableFixture : NSObject
@property(nonatomic) NSUInteger drawableID;
@property(nonatomic) CFTimeInterval presentedTime;
@property(nonatomic,copy) MTLDrawablePresentedHandler handler;
- (void)addPresentedHandler:(MTLDrawablePresentedHandler)handler;
- (void)fire;
@end
@implementation PresentDrawableFixture
- (void)addPresentedHandler:(MTLDrawablePresentedHandler)handler {_handler=[handler copy];}
- (void)fire {if(_handler)_handler((id<MTLDrawable>)self);}
@end
@interface PresentBufferFixture : NSObject
@property(nonatomic) NSUInteger calls,lastID;
@property(nonatomic) double time;
- (void)presentDrawable:(id<MTLDrawable>)d;
- (void)presentDrawable:(id<MTLDrawable>)d atTime:(CFTimeInterval)t;
- (void)presentDrawable:(id<MTLDrawable>)d afterMinimumDuration:(CFTimeInterval)t;
@end
@implementation PresentBufferFixture
- (void)presentDrawable:(id<MTLDrawable>)d {++_calls;_lastID=d.drawableID;[(PresentDrawableFixture*)d fire];}
- (void)presentDrawable:(id<MTLDrawable>)d atTime:(CFTimeInterval)t {_time=t;++_calls;_lastID=d.drawableID;[(PresentDrawableFixture*)d fire];}
- (void)presentDrawable:(id<MTLDrawable>)d afterMinimumDuration:(CFTimeInterval)t {_time=t;++_calls;_lastID=d.drawableID;[(PresentDrawableFixture*)d fire];}
@end
@interface PresentChildFixture : PresentBufferFixture @end
@implementation PresentChildFixture @end
@interface PresentOverrideFixture : PresentBufferFixture @end
@implementation PresentOverrideFixture
- (void)presentDrawable:(id<MTLDrawable>)d { [super presentDrawable:d]; }
@end
int main(){@autoreleasepool{unsigned cases=0;auto need=[&](bool ok,const char*message){++cases;if(!ok)throw std::runtime_error(message);};
try{
  auto b=[PresentBufferFixture new];InstallNRPresentObserver(b);InstallNRPresentObserver(b);
  MarkNRCommand(b,42,7);need(NRObjectID(b)==nr_trace_command.load(),"stable command identity");
  auto d=[PresentDrawableFixture new];d.drawableID=17;d.presentedTime=123.5;
  [b presentDrawable:(id<MTLDrawable>)d];need(b.calls==1 && b.lastID==17,"plain present forwarded once");
  [b presentDrawable:(id<MTLDrawable>)d atTime:12.25];need(b.calls==2 && b.time==12.25,"atTime preserved");
  [b presentDrawable:(id<MTLDrawable>)d afterMinimumDuration:0.125];need(b.calls==3 && b.time==0.125,"minimum duration preserved");
  need(nr_presented_logs.load()==3,"presented handlers observed");
  auto child=[PresentChildFixture new];InstallNRPresentObserver(child);
  [child presentDrawable:(id<MTLDrawable>)d];need(child.calls==1,"inherited hook forwards without recursion");
  auto overridden=[PresentOverrideFixture new];InstallNRPresentObserver(overridden);
  const auto observed=nr_presented_logs.load();[overridden presentDrawable:(id<MTLDrawable>)d];
  need(overridden.calls==1,"overridden method calling super forwards once");
  need(nr_presented_logs.load()==observed+1,"super forwarding does not duplicate observation");
  __weak PresentDrawableFixture* weak=nil;
  @autoreleasepool{auto transient=[PresentDrawableFixture new];weak=transient;[b presentDrawable:(id<MTLDrawable>)transient];}
  need(weak==nil,"no drawable-handler retain cycle");
  const auto before=b.calls;
  for(unsigned i=0;i<70;++i){auto next=[PresentDrawableFixture new];next.drawableID=i;[b presentDrawable:(id<MTLDrawable>)next];}
  need(b.calls==before+70,"log limit never suppresses original presentation");
  std::printf("result=PASS cases=%u GPU_execution=0 actual_presentation=0\n",cases);return 0;
}catch(const std::exception&e){std::fprintf(stderr,"result=FAIL detail=%s\n",e.what());return 1;}}}
