@interface B1Session:NeuralStudioDLSSSRSession
@property(nonatomic) int reviewMode;
@property(nonatomic,strong) FakeQueue*reviewQueue;
@end
@implementation B1Session
- (BOOL)neuralStudioEncodeManagedFrameWithLease:(NeuralStudioDLSSSRFrameLease*)lease
                        colorBuffer:(id<MTLBuffer>)colorBuffer
                        colorOffset:(NSUInteger)colorOffset
                       motionBuffer:(id<MTLBuffer>)motionBuffer
                       motionOffset:(NSUInteger)motionOffset
                        depthBuffer:(id<MTLBuffer>)depthBuffer
                        depthOffset:(NSUInteger)depthOffset
                     exposureBuffer:(id<MTLBuffer>)exposureBuffer
                     exposureOffset:(NSUInteger)exposureOffset
                       outputBuffer:(id<MTLBuffer>)outputBuffer
                       outputOffset:(NSUInteger)outputOffset
                      jitterOffsetX:(float)jitterOffsetX
                      jitterOffsetY:(float)jitterOffsetY
                 motionVectorScaleX:(float)motionVectorScaleX
                 motionVectorScaleY:(float)motionVectorScaleY
                      depthReversed:(BOOL)depthReversed
                              error:(NSError**)error {
 if(self.reviewMode==4)throw std::runtime_error("CPU public wrapper exception");
 if(self.reviewMode==3)@throw [NSException exceptionWithName:@"B1 CPU fault" reason:@"ordinary internal exception" userInfo:nil];
 auto originalThread=std::this_thread::get_id();
 std::thread cutter([self,originalThread]{@autoreleasepool{
  CHECK(std::this_thread::get_id()!=originalThread);
  if(self.reviewMode==1){
   FakeFeedback*f=[FakeFeedback new];f.error=[NSError errorWithDomain:@"old workload" code:1 userInfo:nil];
   for(MTL4CommitFeedbackHandler h in ((FakeOptions*)self.reviewQueue.options.firstObject).handlers)h((id)f);
  }else [self invalidateManagedTemporalChain];
 }});cutter.join();
 return [super neuralStudioEncodeManagedFrameWithLease:lease colorBuffer:colorBuffer colorOffset:colorOffset
  motionBuffer:motionBuffer motionOffset:motionOffset depthBuffer:depthBuffer depthOffset:depthOffset
  exposureBuffer:exposureBuffer exposureOffset:exposureOffset outputBuffer:outputBuffer outputOffset:outputOffset
  jitterOffsetX:jitterOffsetX jitterOffsetY:jitterOffsetY motionVectorScaleX:motionVectorScaleX motionVectorScaleY:motionVectorScaleY
  depthReversed:depthReversed error:error];
}

@end
static BOOL PublicEncode(NeuralStudioDLSSSRSession*s,NeuralStudioDLSSSRFrameLease*l,id b){
 return [s encodeManagedFrameWithLease:l colorBuffer:b colorOffset:0 motionBuffer:b motionOffset:0
  depthBuffer:b depthOffset:0 exposureBuffer:b exposureOffset:0 outputBuffer:b outputOffset:0
  jitterOffsetX:0 jitterOffsetY:0 motionVectorScaleX:1 motionVectorScaleY:1 depthReversed:YES error:nullptr];
}
int main(){@autoreleasepool{
 FakeDevice*d=[FakeDevice new];FakeQueue*q=[FakeQueue new];q.device=d;
 for(int mode=0;mode<2;mode++){
  B1Session*s=[[B1Session alloc]initCPUFixture:d];s.reviewMode=mode;s.reviewQueue=q;
  CHECK([s enableManagedMTL4FramesWithError:nullptr]);
  if(mode==1){auto a=Ready(s,q,YES);q.deferFeedback=YES;CHECK(Submit(s,q,a));q.deferFeedback=NO;}
  auto l=Begin(s,q,mode==0);id b=[d newBufferWithLength:8192 options:MTLResourceStorageModeShared];
  NSUInteger before=q.commits;CHECK(!PublicEncode(s,l,b));CHECK(q.commits==before);
  CHECK(s->_state->managed.needsReset());
  BOOL discard=[s discardUnsubmittedManagedFrameLease:l error:nullptr];
  FakeCommand*c=[FakeCommand new];c.device=d;
  auto reset=[s newManagedFrameLeaseForCommandBuffer:(id)c allocator:(id)q queue:(id)q reset:YES error:nullptr];
#if EXPECT_FIXED
  CHECK(discard&&l.isRetired);
  if(mode==0){CHECK(reset);CHECK([s discardUnsubmittedManagedFrameLease:reset error:nullptr]);}
  else CHECK(!reset); // Old error workload remains legitimately quarantined.
#else
  CHECK(!discard&&!reset&&l.status==NeuralStudioDLSSSRManagedFrameEncoding);
  CHECK(s->_state->managed.ownership(l->_token)==netal::sr::FrameOwnership::Encoding);
  CHECK(![s neuralStudioFinishManagedFrameLease:l error:nullptr]);CHECK(!Submit(s,q,l));
#endif
  printf("B1 mode=%s expect_fixed=%d public_result=NO own_commit_calls=0 discard=%d terminal=%d reset_admitted=%d\n",
   mode==0?"distinct-thread explicit cut":"older registered error callback",EXPECT_FIXED,discard,l.isRetired,reset!=nil);
 }
 // Existing internal nil-input failure already performs authenticated Fail.
 {auto s=Session(q);auto l=Begin(s,q,YES);CHECK(!PublicEncode(s,l,nil));CHECK(l.status==NeuralStudioDLSSSRManagedFrameFailedUnsubmitted);
  CHECK([s discardUnsubmittedManagedFrameLease:l error:nullptr]);puts("normal internal nil-input failure is already FailedUnsubmitted/discardable");}
 // Existing public exception footer cleans the successful Begin.
 {B1Session*s=[[B1Session alloc]initCPUFixture:d];s.reviewMode=3;s.reviewQueue=q;CHECK([s enableManagedMTL4FramesWithError:nullptr]);
  auto l=Begin(s,q,YES);id b=[d newBufferWithLength:8192 options:MTLResourceStorageModeShared];CHECK(!PublicEncode(s,l,b));
  CHECK(l.status==NeuralStudioDLSSSRManagedFrameFailedUnsubmitted&&[s discardUnsubmittedManagedFrameLease:l error:nullptr]);
  puts("normal internal exception runs public Fail footer and remains discardable");}
 {auto s=Session(q);auto old=Ready(s,q,YES);CHECK(Submit(s,q,old));Drain(s);
  CHECK(old.isRetired&&!s->_state->managed.needsReset());CHECK(!PublicEncode(s,old,nil));CHECK(!s->_state->managed.needsReset());
  auto next=Ready(s,q,NO);CHECK(Submit(s,q,next));Drain(s);
  puts("stale public wrapper request rejects at Begin without cutting healthy successor");}

 // Current authentic foreign-device rejection remains recoverable, without entering the native encoder.
 {auto s=Session(q);auto l=Begin(s,q,YES);FakeDevice*other=[FakeDevice new];id bad=[other newBufferWithLength:8192 options:MTLResourceStorageModeShared];
  CHECK(!PublicEncode(s,l,bad));CHECK(l.status==NeuralStudioDLSSSRManagedFrameFailedUnsubmitted);
  CHECK([s discardUnsubmittedManagedFrameLease:l error:nullptr]);puts("current foreign-device failure stays owned then explicitly discardable");}
 // Foreign handles are rejected before successful Begin; cleanup must not mutate either owner.
 {auto a=Session(q);auto la=Ready(a,q,YES);auto b=Session(q);auto prior=Ready(b,q,YES);CHECK(Submit(b,q,prior));Drain(b);
  auto lb=Begin(b,q,NO);CHECK([b neuralStudioBeginManagedLease:lb error:nullptr]);
  CHECK(!PublicEncode(a,lb,nil)&&!PublicEncode(b,la,nil));
  CHECK(!a->_state->managed.needsReset()&&!b->_state->managed.needsReset());
  CHECK(la.status==NeuralStudioDLSSSRManagedFrameEncoded&&b->_state->managed.encoding(lb->_token));
  CHECK([a discardUnsubmittedManagedFrameLease:la error:nullptr]);[b neuralStudioFailManagedLease:lb];CHECK([b discardUnsubmittedManagedFrameLease:lb error:nullptr]);
  puts("current foreign-session handles reject before Begin without cutting either healthy owner");}
 {B1Session*s=[[B1Session alloc]initCPUFixture:d];s.reviewMode=4;s.reviewQueue=q;CHECK([s enableManagedMTL4FramesWithError:nullptr]);
  auto l=Begin(s,q,YES);id buf=[d newBufferWithLength:8192 options:MTLResourceStorageModeShared];CHECK(!PublicEncode(s,l,buf));
  CHECK(l.status==NeuralStudioDLSSSRManagedFrameFailedUnsubmitted&&[s discardUnsubmittedManagedFrameLease:l error:nullptr]);
  puts("foreign C++ exception reaches existing authenticated public failure footer");}
 puts("CPU public-wrapper controls PASS; no native encoder/Metal/model/GPU calls");
}return 0;}
