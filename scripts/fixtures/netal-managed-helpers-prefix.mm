#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import "dlss_sr_session.h"
#import "dlss_sr_texture_adapter.h"
#import "dlss_sr_managed_internal.h"
#include "dlss_sr_frame_ledger.h"
#include "dlss_sr_neural_backend.h"
#include <atomic>
#include <memory>
#include <mutex>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#define CHECK(...) do { if(!(__VA_ARGS__)) { fprintf(stderr,"FAIL %d %s\n",__LINE__,#__VA_ARGS__); abort(); } } while(0)
static BOOL FailOptions,FailHandler,FailHandlerCPP,PrematureHandler;
@interface FakeOptions:NSObject
@property(nonatomic,strong) NSMutableArray* handlers;
- (void)addFeedbackHandler:(MTL4CommitFeedbackHandler)handler;
@end
@interface FakeFeedback:NSObject
@property(nonatomic,strong) NSError* error;
@end
@implementation FakeFeedback @end
@implementation FakeOptions
- (instancetype)init { if(FailOptions)return nil;self=[super init];if(self)_handlers=[NSMutableArray array];return self; }
- (void)addFeedbackHandler:(MTL4CommitFeedbackHandler)b {
 if(FailHandler)@throw [NSException exceptionWithName:@"setup" reason:@"handler setup" userInfo:nil];
 if(FailHandlerCPP)throw std::runtime_error("CPU handler preparation");
 [self.handlers addObject:[b copy]];if(PrematureHandler)b((id)[FakeFeedback new]);
}
@end
@interface FakeDescriptor:NSObject
@property(nonatomic) NSUInteger initialCapacity;
@end
@implementation FakeDescriptor @end
@interface FakeSet:NSObject
@property(nonatomic,strong) NSMutableArray* allocations;
- (void)addAllocation:(id)x;
- (void)removeAllAllocations;
- (void)commit;
@end
@implementation FakeSet
- (instancetype)init {self=[super init];if(self)_allocations=[NSMutableArray array];return self;}
- (void)addAllocation:(id)x {CHECK(x);[self.allocations addObject:x];}
- (void)removeAllAllocations {[self.allocations removeAllObjects];}
- (void)commit {}
@end
@interface FakeBuffer:NSObject
@property(nonatomic,strong) NSMutableData* data;
@property(nonatomic,strong) id device;
- (void*)contents;
- (NSUInteger)length;
@end
@implementation FakeBuffer
- (void*)contents {return self.data.mutableBytes;}
- (NSUInteger)length {return self.data.length;}
- (BOOL)conformsToProtocol:(Protocol*)p {return p==@protocol(MTLAllocation)||p==@protocol(MTLResource)||[super conformsToProtocol:p];}
@end
@interface FakeDevice:NSObject
- (id)newBufferWithLength:(NSUInteger)length options:(MTLResourceOptions)options;
- (id)newResidencySetWithDescriptor:(id)descriptor error:(NSError**)error;
@end
@implementation FakeDevice
- (id)newBufferWithLength:(NSUInteger)length options:(MTLResourceOptions)options {(void)options;FakeBuffer*b=[FakeBuffer new];b.data=[NSMutableData dataWithLength:length];b.device=self;return b;}
- (id)newResidencySetWithDescriptor:(id)descriptor error:(NSError**)error {(void)descriptor;(void)error;return [FakeSet new];}
@end
@interface FakeEncoder:NSObject
- (void)barrierAfterQueueStages:(MTLStages)a beforeStages:(MTLStages)b visibilityOptions:(MTL4VisibilityOptions)c;
- (void)copyFromBuffer:(id)a sourceOffset:(NSUInteger)ao toBuffer:(id)b destinationOffset:(NSUInteger)bo size:(NSUInteger)n;
- (void)endEncoding;
@end
@implementation FakeEncoder
- (void)barrierAfterQueueStages:(MTLStages)a beforeStages:(MTLStages)b visibilityOptions:(MTL4VisibilityOptions)c {(void)a;(void)b;(void)c;}
- (void)copyFromBuffer:(FakeBuffer*)a sourceOffset:(NSUInteger)ao toBuffer:(FakeBuffer*)b destinationOffset:(NSUInteger)bo size:(NSUInteger)n {memcpy((char*)b.contents+bo,(char*)a.contents+ao,n);}
- (void)endEncoding {}
@end
@interface FakeCommand:NSObject
@property(nonatomic,strong) id device;
- (void)useResidencySet:(id)set;
- (id)computeCommandEncoder;
@end
@implementation FakeCommand
- (void)useResidencySet:(id)set {(void)set;}
- (id)computeCommandEncoder {return [FakeEncoder new];}
@end
@interface FakeQueue:NSObject
@property(nonatomic,strong) id device;
@property(nonatomic,strong) NSMutableArray* options;
@property(nonatomic) BOOL deferFeedback,throwBeforeFeedback,throwCppBeforeFeedback,throwAfterFeedback;
@property(nonatomic) NSUInteger commits;
- (void)commit:(const id<MTL4CommandBuffer>*)commands count:(NSUInteger)count options:(id)options;
- (void)emitLast;
@end
@implementation FakeQueue
- (instancetype)init {self=[super init];if(self)_options=[NSMutableArray array];return self;}
- (void)emitLast {FakeFeedback*f=[FakeFeedback new];for(MTL4CommitFeedbackHandler b in ((FakeOptions*)self.options.lastObject).handlers)b((id)f);}
- (void)commit:(const id<MTL4CommandBuffer>*)commands count:(NSUInteger)count options:(FakeOptions*)options {
 CHECK(count&&commands[0]);++self.commits;[self.options addObject:options];
 if(self.throwBeforeFeedback)@throw [NSException exceptionWithName:@"commit" reason:@"unknown" userInfo:nil];
 if(self.throwCppBeforeFeedback)throw std::runtime_error("CPU unknown commit");
 if(!self.deferFeedback)[self emitLast];
 if(self.throwAfterFeedback)@throw [NSException exceptionWithName:@"after" reason:@"after exact feedback" userInfo:nil];
}
@end
struct FakeState {
 std::mutex mutex; id device,residency; id<MTL4CommandQueue> managed_queue;
 BOOL managed_enabled=false,temporal_state_initialized=true;NSUInteger frames_started=0;
 std::atomic_bool consumed{false};netal::sr::FrameLedger managed; FakeState();NSString*backend_status=@"native-default";
};
@interface NeuralStudioDLSSSRSession () {
@public FakeState* _state;NSMutableDictionary* _managedLeases;dispatch_queue_t _managedFeedbackQueue;
}
- (instancetype)initCPUFixture:(id)device;
@end
static void SetError(NSError**error,int code,NSString*reason){if(error)*error=[NSError errorWithDomain:@"CPUFixture" code:code userInfo:@{NSLocalizedDescriptionKey:reason}];}
