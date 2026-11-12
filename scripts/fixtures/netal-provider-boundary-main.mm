static __weak NeuralStudioDLSSSRSession* ReviewSession;
static NeuralStudioDLSSSRFrameLease* ReviewLease;
static FakeQueue* ReviewQueue;
static int ReviewMode,PrepareReentries,GetterReentries,DeallocReentries,ProviderEmissions,NativeInputRecords;
static bool ConcurrentRejected;
@interface BoundaryInvocation:NSObject<NeuralStudioDLSSSRPreparedNeuralCoreInvocation>@end
@implementation BoundaryInvocation
- (NSArray*)retainedResources {
 if(ReviewMode==2){CHECK(ReviewSession.managedMTL4FramesEnabled);++GetterReentries;}
 if(ReviewMode<=3)return nil;
 return @[];
}
- (BOOL)encodeNeuralCoreWithContext:(NeuralStudioDLSSSRNeuralCoreEncodingContext*)context error:(NSError**)error {
 if(ReviewMode==7)[ReviewSession invalidateManagedTemporalChain];
 id command=[context claimCommandBufferForInvocation:self error:error];
 if(command==nil)return NO;
 if(ReviewMode==8)[ReviewSession invalidateManagedTemporalChain];
 ++ProviderEmissions;return YES; // CPU counter only; no driver/GPU commands
}
- (void)dealloc {if(ReviewMode==3){CHECK(ReviewSession.managedMTL4FramesEnabled);++DeallocReentries;}}
@end
@interface BoundaryBackend:NSObject<NeuralStudioDLSSSRNeuralCoreBackend>@end
@implementation BoundaryBackend
- (netal::sr::NeuralCoreContract)neuralCoreContract{return {};}
- (id<NeuralStudioDLSSSRPreparedNeuralCoreInvocation>)prepareNeuralCoreForCommandBuffer:(id<MTL4CommandBuffer>)command inputBuffer:(id<MTLBuffer>)input inputLayout:(netal::sr::FeatureLayout)inLayout outputBuffer:(id<MTLBuffer>)output outputLayout:(netal::sr::FeatureLayout)outLayout phase:(NSUInteger)phase reset:(BOOL)reset error:(NSError**)error {
 (void)command;(void)input;(void)output;(void)inLayout;(void)outLayout;(void)phase;(void)reset;(void)error;
 if(ReviewMode==1){CHECK(ReviewSession.managedMTL4FramesEnabled);++PrepareReentries;}
 if(ReviewMode==4){
  FakeFeedback*f=[FakeFeedback new];f.error=[NSError errorWithDomain:@"CPU old-frame failure" code:1 userInfo:nil];
  for(MTL4CommitFeedbackHandler callback in ((FakeOptions*)ReviewQueue.options.firstObject).handlers)callback((id)f);
 }
 if(ReviewMode==5)throw std::runtime_error("CPU provider prepare exception");
 if(ReviewMode==6){
  dispatch_sync(dispatch_get_global_queue(QOS_CLASS_DEFAULT,0),^{
  FakeCommand*c=[FakeCommand new];c.device=ReviewQueue.device;
  CHECK(![ReviewSession newManagedFrameLeaseForCommandBuffer:(id)c allocator:(id)ReviewQueue queue:(id)ReviewQueue reset:NO error:nullptr]);
  CHECK(![ReviewSession neuralStudioBeginManagedLease:ReviewLease error:nullptr]);
  NeuralStudioDLSSSRSession*active=ReviewSession;CHECK(active->_state->managed.encoding(ReviewLease->_token));ConcurrentRejected=true;
  });
 }
 return [BoundaryInvocation new];
}
@end
int main(int argc,char**argv){@autoreleasepool{
 CHECK(argc==3);ReviewMode=atoi(argv[1]);bool expectV3=atoi(argv[2]);
 FakeDevice*d=[FakeDevice new];FakeQueue*q=[FakeQueue new];q.device=d;ReviewQueue=q;
 auto s=Session(q);ReviewSession=s;NeuralStudioDLSSSRFrameLease*prior=nil;
 if(ReviewMode==4){prior=Ready(s,q,YES);q.deferFeedback=YES;CHECK(Submit(s,q,prior));q.deferFeedback=NO;}
 auto l=Begin(s,q,ReviewMode!=4);ReviewLease=l;CHECK([s neuralStudioBeginManagedLease:l error:nullptr]);
 s->_state->arena6.buffer=[d newBufferWithLength:640 options:MTLResourceStorageModeShared];s->_state->qualified_backend=[BoundaryBackend new];
 BOOL prepared=NO;
 @autoreleasepool {prepared=[s reviewerActualProviderBoundary:l];}
 BOOL coreAccepted=prepared&&s->_state->managed.encoded(l->_token)==netal::sr::FrameResult::Accepted;
 BOOL finalized=NO;
 if(coreAccepted){l->_cachedStatus.store(NeuralStudioDLSSSRManagedFrameCoreEncoded);finalized=[s neuralStudioFinishManagedFrameLease:l error:nullptr];}
 else {[s neuralStudioFailManagedLease:l];CHECK(![s neuralStudioFinishManagedFrameLease:l error:nullptr]);}
 NSUInteger commitsBefore=q.commits;
 BOOL submitted=Submit(s,q,l);CHECK(submitted==finalized);
 if(ReviewMode<=3){CHECK(!prepared&&!ProviderEmissions&&!NativeInputRecords);CHECK(PrepareReentries+GetterReentries+DeallocReentries>0);}
 if(ReviewMode==4){CHECK(!finalized&&!submitted&&q.commits==commitsBefore);CHECK(prior.status==NeuralStudioDLSSSRManagedFrameQuarantined);
  CHECK(ProviderEmissions==(expectV3?0:1)&&NativeInputRecords==(expectV3?0:1));}
 if(ReviewMode==5)CHECK(!prepared&&!ProviderEmissions&&!NativeInputRecords&&!submitted);
 if(ReviewMode==6)CHECK(ConcurrentRejected&&prepared&&finalized&&submitted&&ProviderEmissions==1);
 if(ReviewMode==7){CHECK(!finalized&&!submitted&&q.commits==commitsBefore);CHECK(ProviderEmissions==(expectV3?0:1));}
 if(ReviewMode==8){CHECK(!finalized&&!submitted&&q.commits==commitsBefore&&ProviderEmissions==1);}
 if(!finalized){CHECK(l.status==NeuralStudioDLSSSRManagedFrameFailedUnsubmitted&&l->_owner&&l->_resources);CHECK([s discardUnsubmittedManagedFrameLease:l error:nullptr]);CHECK(l.isRetired);}
 Drain(s);
 printf("{\"mode\":%d,\"expect_v3\":%d,\"prepared\":%d,\"native_input_records\":%d,\"provider_emissions\":%d,\"finalized\":%d,\"submitted\":%d,\"prepare_reentries\":%d,\"getter_reentries\":%d,\"dealloc_reentries\":%d,\"concurrent_rejected\":%d,\"gpu_execution\":false}\n",ReviewMode,expectV3,prepared,NativeInputRecords,ProviderEmissions,finalized,submitted,PrepareReentries,GetterReentries,DeallocReentries,ConcurrentRejected);
 ReviewLease=nil;ReviewQueue=nil;
}return 0;}
