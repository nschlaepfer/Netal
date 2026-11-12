static NeuralStudioDLSSSRSession* Session(FakeQueue*q) {auto s=[[NeuralStudioDLSSSRSession alloc]initCPUFixture:q.device];CHECK([s enableManagedMTL4FramesWithError:nullptr]);return s;}
static NeuralStudioDLSSSRFrameLease* Begin(NeuralStudioDLSSSRSession*s,FakeQueue*q,BOOL reset,FakeCommand**command=nullptr) {
 FakeCommand*c=[FakeCommand new];c.device=q.device;if(command)*command=c;
 // This CPU endpoint represents a caller allocator; no real allocator or Metal object is created.
 auto l=[s newManagedFrameLeaseForCommandBuffer:(id)c allocator:(id)q queue:(id)q reset:reset error:nullptr];CHECK(l);return l;
}
static NeuralStudioDLSSSRFrameLease* Ready(NeuralStudioDLSSSRSession*s,FakeQueue*q,BOOL reset,FakeCommand**command=nullptr) {
 auto l=Begin(s,q,reset,command);CHECK([s neuralStudioBeginManagedLease:l error:nullptr]);
 CHECK(s->_state->managed.encoded(l->_token)==netal::sr::FrameResult::Accepted);l->_cachedStatus.store(NeuralStudioDLSSSRManagedFrameCoreEncoded);
 CHECK([s neuralStudioFinishManagedFrameLease:l error:nullptr]);return l;
}
static void Drain(NeuralStudioDLSSSRSession*s){dispatch_sync(s->_managedFeedbackQueue,^{});}
static BOOL Submit(NeuralStudioDLSSSRSession*s,FakeQueue*q,NeuralStudioDLSSSRFrameLease*l) {
 return [s commitManagedCommandBuffers:@[l->_command] queue:(id)q leases:@[l] feedback:nil error:nullptr];
}
static int Reentries,ReentrantSubmits;
@interface Probe:NSObject
@property(nonatomic,weak) NeuralStudioDLSSSRSession* session;
@property(nonatomic,strong) FakeQueue* queue;
@property(nonatomic) BOOL submitInDealloc;
@end
@implementation Probe
- (void)dealloc {
 CHECK(self.session.managedMTL4FramesEnabled);++Reentries;
 if(self.submitInDealloc){auto next=Ready(self.session,self.queue,NO);CHECK(Submit(self.session,self.queue,next));++ReentrantSubmits;}
}
@end
@interface Invocation:NSObject<NeuralStudioDLSSSRPreparedNeuralCoreInvocation>
@property(nonatomic,strong) NSArray* base;
@end
@implementation Invocation
- (NSArray*)retainedResources{return [self.base arrayByAddingObject:self];}
- (BOOL)encodeNeuralCoreWithContext:(NeuralStudioDLSSSRNeuralCoreEncodingContext*)c error:(NSError**)e {CHECK([c claimCommandBufferForInvocation:self error:e]);@throw [NSException exceptionWithName:@"provider" reason:@"CPU encode failure" userInfo:nil];}
@end
int main(){@autoreleasepool{
 FakeDevice*device=[FakeDevice new];FakeQueue*q=[FakeQueue new];q.device=device;
 for(int kind=0;kind<3;++kind){
  auto s=Session(q);auto l=Ready(s,q,YES);NSUInteger before=q.commits;FailOptions=kind==0;FailHandler=kind==1;FailHandlerCPP=kind==2;
  CHECK(!Submit(s,q,l));FailOptions=FailHandler=FailHandlerCPP=NO;
  CHECK(q.commits==before&&l.status==NeuralStudioDLSSSRManagedFrameEncoded&&!s->_state->managed.needsReset());
  CHECK([s discardUnsubmittedManagedFrameLease:l error:nullptr]);CHECK(l.isRetired&&!l->_owner&&!l->_command&&!l->_queue&&!l->_markerResidency&&!l->_resources);
 }puts("D1 setup failures: no issue/commit, still discardable, terminal handle empty");
 {
  auto s=Session(q);auto l=Ready(s,q,YES);q.throwCppBeforeFeedback=YES;CHECK(!Submit(s,q,l));q.throwCppBeforeFeedback=NO;
  CHECK(l.status==NeuralStudioDLSSSRManagedFrameSubmitUnknown&&!l.isRetired&&s->_state->managed.needsReset());
  CHECK(![s discardUnsubmittedManagedFrameLease:l error:nullptr]);[q emitLast];CHECK(l.isRetired&&s->_state->managed.drained()&&s->_state->managed.needsReset());
  [q.options removeAllObjects];auto next=Ready(s,q,YES);CHECK(Submit(s,q,next));CHECK(next.isRetired);
 }puts("D1 unknown + exact late completion: ownership retires, content stays cut until reset");
 {
  auto s=Session(q);auto l=Ready(s,q,YES);q.throwAfterFeedback=YES;CHECK(!Submit(s,q,l));q.throwAfterFeedback=NO;
  CHECK(l.isRetired&&s->_state->managed.drained()&&!s->_state->managed.needsReset());[q.options removeAllObjects];
  auto next=Ready(s,q,NO);PrematureHandler=YES;CHECK(Submit(s,q,next));PrematureHandler=NO;CHECK(next.isRetired);
 }puts("early callback wins return/exception; preparation-time callback is inert");
 {
  auto a=Session(q),b=Session(q);FakeCommand*ca;auto la=Ready(a,q,YES,&ca);
  CHECK(![b newManagedFrameLeaseForCommandBuffer:(id)ca allocator:(id)q queue:(id)q reset:YES error:nullptr]);
  FakeCommand*cb;auto lb=Ready(b,q,YES,&cb);NSUInteger before=q.commits;
  CHECK(![a commitManagedCommandBuffers:@[(id)ca,(id)cb] queue:(id)q leases:@[la] feedback:nil error:nullptr]);CHECK(q.commits==before);
  CHECK([b discardUnsubmittedManagedFrameLease:lb error:nullptr]);CHECK(![a commitManagedCommandBuffers:@[(id)ca,(id)cb] queue:(id)q leases:@[la] feedback:nil error:nullptr]);
  CHECK(![a newManagedFrameLeaseForCommandBuffer:(id)cb allocator:(id)q queue:(id)q reset:YES error:nullptr]);
  CHECK(Submit(a,q,la));CHECK(![a newManagedFrameLeaseForCommandBuffer:(id)ca allocator:(id)q queue:(id)q reset:YES error:nullptr]);[q.options removeAllObjects];
 }puts("D2 global active CB rejection and permanent live-object tombstones prevent helper smuggling/reuse");
 {
  auto s=Session(q);auto l=Begin(s,q,YES);CHECK([s neuralStudioBeginManagedLease:l error:nullptr]);
  __weak Probe*weak;
  @autoreleasepool {Probe*p=[Probe new];p.session=s;weak=p;CHECK([s neuralStudioRetainManagedResources:@[p] lease:l error:nullptr]);p=nil;}
  CHECK(![s discardUnsubmittedManagedFrameLease:l error:nullptr]);[s neuralStudioFailManagedLease:l];CHECK([s discardUnsubmittedManagedFrameLease:l error:nullptr]);CHECK(!weak&&Reentries>0);
  CHECK(l.status==NeuralStudioDLSSSRManagedFrameRetired&&!l->_owner&&!l->_markerSource&&!l->_markerDestination&&!l->_command&&!l->_queue&&!l->_resources&&!l->_markerResidency);
  auto next=Ready(s,q,YES);CHECK(l.status==NeuralStudioDLSSSRManagedFrameRetired&&Submit(s,q,next));[q.options removeAllObjects];
 }puts("D3/D4 discard drops all terminal payload; resource deallocator safely reenters actual protected getter");
 {
  __weak NeuralStudioDLSSSRSession*weakSession; NeuralStudioDLSSSRFrameLease*l;
  @autoreleasepool {
  NeuralStudioDLSSSRSession*s=Session(q);weakSession=s;
  l=Begin(s,q,YES);CHECK([s neuralStudioBeginManagedLease:l error:nullptr]);
  __weak Probe*weak;
  @autoreleasepool {Probe*p=[Probe new];p.session=s;weak=p;CHECK([s neuralStudioRetainManagedResources:@[p]lease:l error:nullptr]);p=nil;}
  CHECK(s->_state->managed.encoded(l->_token)==netal::sr::FrameResult::Accepted);l->_cachedStatus.store(NeuralStudioDLSSSRManagedFrameCoreEncoded);CHECK([s neuralStudioFinishManagedFrameLease:l error:nullptr]);
  CHECK(Submit(s,q,l));Drain(s);CHECK(!weak&&l.isRetired);s=nil;
  }
  CHECK(!weakSession&&l.status==NeuralStudioDLSSSRManagedFrameRetired);[q.options removeAllObjects];
 }puts("D3/D4 completion release reenters; retained terminal handle no longer pins its session");
 {
  auto s=Session(q);auto l=Ready(s,q,YES);CHECK(s->_state->temporal_state_initialized&&!s->_state->managed.needsReset());
  [s neuralStudioFailManagedLease:l];CHECK(s->_state->temporal_state_initialized&&!s->_state->managed.needsReset());
  q.deferFeedback=YES;CHECK(Submit(s,q,l));q.deferFeedback=NO;
  CHECK(![s neuralStudioEncodeManagedFrameWithLease:l colorBuffer:nil colorOffset:0 motionBuffer:nil motionOffset:0 depthBuffer:nil depthOffset:0 exposureBuffer:nil exposureOffset:0 outputBuffer:nil outputOffset:0 jitterOffsetX:0 jitterOffsetY:0 motionVectorScaleX:1 motionVectorScaleY:1 depthReversed:YES error:nullptr]);
  CHECK(s->_state->temporal_state_initialized&&!s->_state->managed.needsReset());[q emitLast];
  FakeOptions*old=q.options.lastObject;auto next=Ready(s,q,YES);for(MTL4CommitFeedbackHandler cb in old.handlers)cb((id)[FakeFeedback new]);
  CHECK(!s->_state->managed.needsReset()&&next.status==NeuralStudioDLSSSRManagedFrameEncoded&&l.isRetired);CHECK(Submit(s,q,next));[q.options removeAllObjects];
 }puts("D5/D6 stale/wrong-state failure is inert; duplicate callback cannot cut a new epoch");
 {
  auto s=Session(q);auto l=Ready(s,q,YES);memset(l->_markerDestination.contents,0,sizeof(ManagedMarker));CHECK(Submit(s,q,l));CHECK(l.status==NeuralStudioDLSSSRManagedFrameQuarantined&&!l.isRetired);
  memcpy(l->_markerDestination.contents,l->_markerSource.contents,sizeof(ManagedMarker));[q emitLast];CHECK(!l.isRetired&&!s->_state->managed.drained());
 }puts("wrong marker remains quarantined despite repeated later callback; no timeout/blanket release");
 {
  FakeQueue*old=[FakeQueue new];old.device=device;__weak FakeQueue*weak=old;auto s=Session(old);auto l=Ready(s,old,YES);CHECK(Submit(s,old,l));Drain(s);[old.options removeAllObjects];old=nil;CHECK(weak);
  FakeQueue*replacement=[FakeQueue new];replacement.device=device;auto next=Ready(s,replacement,YES);CHECK(!weak);CHECK(Submit(s,replacement,next));[replacement.options removeAllObjects];
 }puts("D6 queue strongly bound while active; drained reset rebind releases prior queue outside locks");
 {
  auto s=Session(q);auto l=Begin(s,q,YES);CHECK([s neuralStudioBeginManagedLease:l error:nullptr]);
  Invocation*inv=[Invocation new];__weak Probe*weak;
  @autoreleasepool {Probe*p=[Probe new];p.session=s;weak=p;inv.base=@[p];CHECK([s neuralStudioRetainManagedResources:inv.retainedResources lease:l error:nullptr]);p=nil;}
  netal::sr::NativeFeatureLayouts views;CHECK(netal::sr::NativeHWCFeatureLayouts(4,2,0,640,0,views));
  id buffer=[device newBufferWithLength:640 options:MTLResourceStorageModeShared];
  auto ctx=[[NeuralStudioDLSSSRNeuralCoreEncodingContext alloc]initWithLease:l invocation:inv input:buffer output:buffer inputLayout:views.input outputLayout:views.output];CHECK(ctx);
  CHECK(![ctx claimCommandBufferForInvocation:(id)[NSObject new] error:nullptr]);
  @try {[inv encodeNeuralCoreWithContext:ctx error:nullptr];} @catch(NSException*exception){(void)exception;[s neuralStudioFailManagedLease:l];} @finally {[ctx closeEncodingContext];}
  CHECK(![ctx claimCommandBufferForInvocation:inv error:nullptr]);inv=nil;CHECK(weak);CHECK([s discardUnsubmittedManagedFrameLease:l error:nullptr]);CHECK(!weak);
 }puts("prepared resources retained before encode exception; authentic context exact/one-shot/closed");
 {
  auto s=Session(q);auto l=Begin(s,q,YES);CHECK([s neuralStudioBeginManagedLease:l error:nullptr]);
  @autoreleasepool {Probe*p=[Probe new];p.session=s;p.queue=q;p.submitInDealloc=YES;CHECK([s neuralStudioRetainManagedResources:@[p]lease:l error:nullptr]);p=nil;}
  CHECK(s->_state->managed.encoded(l->_token)==netal::sr::FrameResult::Accepted);l->_cachedStatus.store(NeuralStudioDLSSSRManagedFrameCoreEncoded);CHECK([s neuralStudioFinishManagedFrameLease:l error:nullptr]);
  CHECK(Submit(s,q,l));Drain(s);CHECK(ReentrantSubmits>0);Drain(s);[q.options removeAllObjects];
 }puts("D4 asynchronous reaping permits last-resource deallocator to reenter actual submit helper");
 {
  [q.options removeAllObjects];auto s=Session(q);NSMutableArray*pending=[NSMutableArray array];q.deferFeedback=YES;
  for(int i=0;i<6;++i){auto l=Ready(s,q,i==0);[pending addObject:l];CHECK(Submit(s,q,l));}
  FakeCommand*blocked=[FakeCommand new];blocked.device=device;
  CHECK(![s newManagedFrameLeaseForCommandBuffer:(id)blocked allocator:(id)q queue:(id)q reset:NO error:nullptr]);
  for(FakeOptions*o in [[q.options copy] reverseObjectEnumerator])for(MTL4CommitFeedbackHandler h in o.handlers)h((id)[FakeFeedback new]);
  q.deferFeedback=NO;Drain(s);CHECK(s->_state->managed.drained());
  for(NeuralStudioDLSSSRFrameLease*l in pending)CHECK(l.isRetired);
  [q.options removeAllObjects];auto reset=Ready(s,q,YES);CHECK(Submit(s,q,reset));Drain(s);[q.options removeAllObjects];
 }puts("six exact slots retire out of callback arrival order; proven drain admits fresh reset");
 {
  auto s=Session(q);auto adapter=[[NeuralStudioDLSSTextureAdapter alloc]initCPUFixture:s];
  FakeDevice*foreign=[FakeDevice new];id bad=[foreign newBufferWithLength:8 options:MTLResourceStorageModeShared];
  FakeCommand*command=[FakeCommand new];command.device=device;NeuralStudioDLSSSRFrameLease*l=nil;
  CHECK(![adapter encodeManagedTemporalFrameToMTL4CommandBuffer:(id)command allocator:(id)q queue:(id)q
    colorTexture:bad motionTexture:bad depthTexture:bad exposureTexture:nil outputTexture:bad
    inputContentWidth:1 inputContentHeight:1 jitterOffsetX:0 jitterOffsetY:0 motionVectorScaleX:1 motionVectorScaleY:1
    preExposure:1 reset:YES depthReversed:YES lease:&l error:nullptr]);
  CHECK(l&&l.status==NeuralStudioDLSSSRManagedFrameFailedUnsubmitted&&adapter->_cpuImportCalls==0&&s->_state->managed.needsReset());
  CHECK([s discardUnsubmittedManagedFrameLease:l error:nullptr]);
 }puts("D6 actual adapter retention failure cuts the authentic lease before any import mutation");
 puts("PASS V2 actual helper/context/guard methods; fake CPU endpoints only; NO Metal/GPU/model/app execution");
}return 0;}
