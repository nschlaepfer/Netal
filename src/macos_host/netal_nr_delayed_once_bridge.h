#pragma once
// Included inside the bridge namespace, only in the explicitly linked delayed
// build. Game commands contain copies only, never the NR graph or GPU waits.
struct DelayedNRJob {
  std::string cancel_path;
  std::string compute_path, publish_path; // Empty keeps the existing immediate delayed behavior.
  std::chrono::steady_clock::time_point compute_gate_deadline;
  uint32_t compute_gate_polls=0; // Owned host queue only, bounded even if a marker never arrives.
  std::atomic<bool> cancel_seen{false};
  netal_nr_adapter::DelayedOnceState state;
  nr::ResetWorkerPtr worker;
  nr::ResetJobPtr reset_job; // Accessed only on serial host_queue.
  dispatch_queue_t host_queue = dispatch_queue_create("netal.nr.delayed-once", DISPATCH_QUEUE_SERIAL);
  id<MTLCommandQueue> gpu_queue = nil;
  id<MTLTexture> source = nil, fitted = nil, bounded = nil, publication = nil, model_output = nil;
  id<MTLBuffer> export_copy = nil, rejected = nil;
  netal_nr_adapter::AspectLayout layout;
  std::shared_ptr<netal_nr_adapter::FrameCapture> capture;
  uint64_t capture_command = 0, publication_command = 0;
};
std::mutex delayed_job_mutex;
std::shared_ptr<DelayedNRJob> delayed_job;
std::shared_ptr<DelayedNRJob> DelayedJob() {
  std::lock_guard<std::mutex> lock(delayed_job_mutex); return delayed_job;
}
void DelayedFail(std::shared_ptr<DelayedNRJob> job, const char* reason) {
  job->state.fail();const auto status=job->state.snapshot();
  Log("event=nr_delayed_failed detail=%s original_target_unmodified=%d publication_may_be_encoded=%d subsequent_frames_original=1",
    reason,!status.publication_may_be_encoded,status.publication_may_be_encoded);
}
template<class Action> void DelayedSafe(std::shared_ptr<DelayedNRJob> job, Action action) {
  @try { try { action(); }
    catch(const std::exception& e){DelayedFail(job,e.what());}
    catch(...){DelayedFail(job,"unknown worker exception");} }
  @catch(NSException* e){DelayedFail(job,e.reason.UTF8String);}
}
void DelayedRequire(bool value, const char* message) { if(!value)throw std::runtime_error(message); }
// These asynchronous helpers deliberately take shared ownership BY VALUE.
// Objective-C blocks capture a C++ reference parameter as a reference; retaining
// the pointee elsewhere does not keep the referenced shared_ptr object alive.
void DelayedWrite(std::shared_ptr<DelayedNRJob> job, const char* stage, bool published) {
  // Freeze metadata/copy flags before handing the writer to another thread.
  auto record=std::make_shared<netal_nr_adapter::FrameCapture>(*job->capture);
  record->root+="/";record->root+=stage;
  dispatch_async(dispatch_get_global_queue(QOS_CLASS_UTILITY,0), ^{
    @autoreleasepool { DelayedSafe(job,[&]{
      NSString* error=nil;
      const bool ok=netal_nr_adapter::WriteFrameCapture(record,true,published,0,&error);
      Log("event=nr_delayed_capture stage=%s source_frame=%llu success=%d detail=%s",
        stage,(unsigned long long)record->frame,ok,error?error.UTF8String:"");
    }); }
  });
}
void DelayedPoll(std::shared_ptr<DelayedNRJob> job);
#ifndef NR_DELAYED_HAS_SUBMISSION_LABELS
#define NR_DELAYED_HAS_SUBMISSION_LABELS 0
#endif
// Diagnostic only, on our host queue after terminal worker state. No driver
// callback allocation and no addresses or expected outputs in these receipts.
void DelayedWorkerEvidence(std::shared_ptr<DelayedNRJob> job) {
#if NR_DELAYED_HAS_SUBMISSION_LABELS
  const auto records=nr::reset_worker_submissions(job->reset_job);
  const auto first=nr::reset_worker_first_failure(job->reset_job);
  const auto dropped=nr::reset_worker_dropped_duration_samples(job->reset_job);
  Log("event=nr_delayed_worker_receipt source_frame=%llu records=%zu dropped=%u first_failure_known=%d",
    (unsigned long long)job->capture->frame,records.size(),dropped,first<records.size());
  for(const auto& r:records) {
    const auto label=nr::reset_worker_submission_label(r);
    Log("event=nr_delayed_submission source_frame=%llu index=%u gpu_ms=%.6f status=%d over_budget=%d encoders=%u encoder_count_known=%d label=%s",
      (unsigned long long)job->capture->frame,r.index,r.gpu_ms,r.status,r.over_budget,r.encoders,r.encoders>0,label.c_str());
  }
  if(first<records.size()) {
    const auto& r=records[first];const auto label=nr::reset_worker_submission_label(r);
    Log("event=nr_delayed_first_failed_submission index=%u gpu_ms=%.6f label=%s",r.index,r.gpu_ms,label.c_str());
  }
  DelayedRequire(dropped==0,"worker evidence dropped submission records; result withheld");
#else
  (void)job;
#endif
}
void DelayedExport(std::shared_ptr<DelayedNRJob> job) {
  auto result=nr::reset_worker_output(job->reset_job);
  DelayedRequire(result && result.device==job->source.device && result.pixelFormat==MTLPixelFormatRGBA16Float &&
    result.width==1920 && result.height==1080 && result.sampleCount==1 && result.textureType==MTLTextureType2D &&
    !result.framebufferOnly && result.storageMode!=MTLStorageModeMemoryless &&
    result.hazardTrackingMode==MTLHazardTrackingModeTracked,"reset worker output contract");
  job->model_output=result;
  auto cb=[job->gpu_queue commandBuffer];DelayedRequire(cb!=nil,"export command unavailable");
  nr::FrameEncoderState ownership;
  auto b=[cb blitCommandEncoder];nr::EncoderScope bs(ownership,b);DelayedRequire(b!=nil,"export copy encoder");
  [b copyFromTexture:result sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0)
    sourceSize:MTLSizeMake(1920,1080,1) toBuffer:job->export_copy destinationOffset:0
    destinationBytesPerRow:1920*8 destinationBytesPerImage:1920*1080*8];
  netal_nr_adapter::CopyCapture(b,job->capture,2,result);
  DelayedRequire(bs.close(),"export copy close");
  const uint32_t dims[4]={1920,1080,1,0};
  const id<MTLComputePipelineState> passes[]={color_inverse,scan_finite,fit_publish};
  for(unsigned i=0;i<3;++i) {
    auto c=[cb computeCommandEncoder];nr::EncoderScope cs(ownership,c);DelayedRequire(c!=nil,"export compute encoder");
    [c setComputePipelineState:passes[i]];[c setBuffer:job->export_copy offset:0 atIndex:0];
    [c setBuffer:job->rejected offset:0 atIndex:1];
    if(i==2){[c setBytes:&job->layout length:sizeof(job->layout) atIndex:2];[c setTexture:job->publication atIndex:0];}
    else [c setBytes:dims length:(i==0?8:16) atIndex:2];
    [c dispatchThreads:MTLSizeMake(1920,i==2?1200:1080,1) threadsPerThreadgroup:MTLSizeMake(8,8,1)];
    DelayedRequire(cs.close(),"export compute close");
  }
  [cb addCompletedHandler:^(id<MTLCommandBuffer> done){
    dispatch_async(job->host_queue, ^{ @autoreleasepool { DelayedSafe(job,[&]{
      DelayedRequire(done.status==MTLCommandBufferStatusCompleted,"export GPU failure");
      const uint32_t flags=*(const uint32_t*)job->rejected.contents;
      DelayedWrite(job,"computed",false);
      DelayedRequire(flags==0,"color/export rejection: no publication");
      if(job->state.compute_completed(true))Log("event=nr_delayed_ready source_frame=%llu model_computed=1 screen_presented=0 publication_gate_required=%d",
        (unsigned long long)job->capture->frame,!job->publish_path.empty());
    }); } });
  }];
  [cb commit]; // OUR queue only; never a game command.
}
void DelayedPoll(std::shared_ptr<DelayedNRJob> job) {
  DelayedSafe(job,[&]{
    if(!job->cancel_path.empty() && access(job->cancel_path.c_str(),F_OK)==0 && !job->cancel_seen.exchange(true))
      DelayedFail(job,"manual cancellation requested");
    if(job->state.snapshot().phase==netal_nr_adapter::DelayedOnceState::Phase::Failed)
      nr::reset_worker_cancel(job->reset_job);
    const auto progress=nr::reset_worker_poll(job->reset_job);
    if(progress.state==nr::ResetState::Completed) {
      DelayedWorkerEvidence(job);
      DelayedRequire(progress.output_publishable,"completed reset worker refused output");
      job->state.note_model_computed();
      Log("event=nr_delayed_computed source_frame=%llu chunks=%u total_gpu_ms=%.6f worst_chunk_gpu_ms=%.6f screen_presented=0",
        (unsigned long long)job->capture->frame,progress.chunks_done,progress.total_gpu_ms,progress.worst_chunk_gpu_ms);
      if(job->state.snapshot().phase==netal_nr_adapter::DelayedOnceState::Phase::Computing)DelayedExport(job);
      return;
    }
    if(progress.state==nr::ResetState::Failed || progress.state==nr::ResetState::Cancelled) {
      DelayedWorkerEvidence(job);
      DelayedFail(job,progress.error.empty()?"reset worker cancelled":progress.error.c_str());return;
    }
    // Materialize a local owner before creating an ObjC block INSIDE this C++
    // lambda. Referencing its capture directly would retain the lambda's `this`
    // pointer, not a standalone owner, after DelayedSafe returns.
    const auto keep=job;
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW,10*NSEC_PER_MSEC),keep->host_queue,^{DelayedPoll(keep);});
  });
}
// Runs on the owned serial host queue, after input GPU completion. No game
// texture or command buffer is kept alive while the operator pauses the game.
void DelayedBeginCompute(std::shared_ptr<DelayedNRJob> job) {
  DelayedSafe(job,[&]{
    if(!job->cancel_path.empty() && access(job->cancel_path.c_str(),F_OK)==0 && !job->cancel_seen.exchange(true))
      DelayedFail(job,"manual cancellation before compute");
    using Phase=netal_nr_adapter::DelayedOnceState::Phase;
    if(job->state.snapshot().phase==Phase::WaitingForCompute) {
      const auto now=std::chrono::steady_clock::now();
      if(job->compute_gate_polls==0)job->compute_gate_deadline=now+std::chrono::minutes(5);
      if(job->compute_gate_polls++>=6000 || now>=job->compute_gate_deadline) {
        DelayedFail(job,"manual compute gate expired after bounded wait; no NR admitted");return;
      }
      if(!job->state.release_compute(access(job->compute_path.c_str(),F_OK)==0)) {
        if(job->state.snapshot().phase!=Phase::WaitingForCompute)return;
        const auto keep=job;
        dispatch_after(dispatch_time(DISPATCH_TIME_NOW,50*NSEC_PER_MSEC),keep->host_queue,^{DelayedBeginCompute(keep);});
        return;
      }
      Log("event=nr_delayed_compute_released source_frame=%llu manual_gate=1",
        (unsigned long long)job->capture->frame);
    }
    if(job->state.snapshot().phase!=Phase::Computing)return;
    nr::Report report;
    const bool begun=nr::reset_worker_begin(job->worker,job->bounded,job->reset_job,report);
    DelayedRequire(begun,report.error.c_str());
    DelayedPoll(job);
  });
}
void DelayedPrepareInput(std::shared_ptr<DelayedNRJob> job) {
  auto cb=[job->gpu_queue commandBuffer];DelayedRequire(cb!=nil,"input command unavailable");
  nr::FrameEncoderState ownership;
  auto b=[cb blitCommandEncoder];nr::EncoderScope bs(ownership,b);DelayedRequire(b!=nil,"input clear encoder");
  [b fillBuffer:job->rejected range:NSMakeRange(0,4) value:0];
  [b copyFromTexture:job->source toTexture:job->publication];
  DelayedRequire(bs.close(),"input clear close");
  for(unsigned i=0;i<2;++i) {
    auto c=[cb computeCommandEncoder];nr::EncoderScope cs(ownership,c);DelayedRequire(c!=nil,"input compute encoder");
    [c setComputePipelineState:i==0?fit_input:color_forward];
    [c setTexture:i==0?job->source:job->fitted atIndex:0];
    [c setTexture:i==0?job->fitted:job->bounded atIndex:1];
    if(i==0)[c setBytes:&job->layout length:sizeof(job->layout) atIndex:0];
    else [c setBuffer:job->rejected offset:0 atIndex:0];
    [c dispatchThreads:MTLSizeMake(1920,1080,1) threadsPerThreadgroup:MTLSizeMake(8,8,1)];
    DelayedRequire(cs.close(),"input compute close");
  }
  b=[cb blitCommandEncoder];nr::EncoderScope capture_scope(ownership,b);DelayedRequire(b!=nil,"input capture encoder");
  netal_nr_adapter::CopyCapture(b,job->capture,1,job->bounded);DelayedRequire(capture_scope.close(),"input capture close");
  [cb addCompletedHandler:^(id<MTLCommandBuffer> done){
    dispatch_async(job->host_queue, ^{ @autoreleasepool { DelayedSafe(job,[&]{
      DelayedRequire(done.status==MTLCommandBufferStatusCompleted,"input conversion GPU failure");
      DelayedWrite(job,"input",false);
      DelayedRequire(*(const uint32_t*)job->rejected.contents==0,"input color rejection");
      if(job->state.snapshot().phase==netal_nr_adapter::DelayedOnceState::Phase::WaitingForCompute)
        Log("event=nr_delayed_waiting_compute source_frame=%llu input_gpu_completed=1 worker_started=0 original_renderer_retained=1",
          (unsigned long long)job->capture->frame);
      DelayedBeginCompute(job);
    }); } });
  }];
  [cb commit];
}
void NRDelayedPresented(uint64_t command) {
  auto job=DelayedJob();if(!job)return;
  const auto s=job->state.snapshot();
  if(command==job->publication_command && job->state.note_presented(s.publication_frame,s.stream))
    Log("event=nr_delayed_drawable_presented source_frame=%llu publication_frame=%llu command_id=%llu pixel_content_verified=0",
      (unsigned long long)s.source_frame,(unsigned long long)s.publication_frame,(unsigned long long)command);
}
void AppendDelayedOnce(id<MTLFXTemporalScaler> scaler,id<MTLCommandBuffer> cb,uint64_t frame,uint64_t stream) {
  using Phase=netal_nr_adapter::DelayedOnceState::Phase;
  if(!delayed_once_mode || !stream || ![(id)cb respondsToSelector:@selector(addCompletedHandler:)] ||
     ![(id)cb respondsToSelector:@selector(commandQueue)] ||
     ![(id)cb respondsToSelector:@selector(status)] || cb.status!=MTLCommandBufferStatusNotEnqueued)return;
  auto output=scaler.outputTexture;
  auto job=DelayedJob();
  if(job && !job->cancel_path.empty() && access(job->cancel_path.c_str(),F_OK)==0 && !job->cancel_seen.exchange(true))
    DelayedFail(job,"manual cancellation requested");
  if(job)job->state.observe_geometry(stream,uint32_t(output.width),uint32_t(output.height));
  if(!output || output.width!=1920 || output.height!=1200 || output.pixelFormat!=MTLPixelFormatRG11B10Float ||
     !WholeFrameProfile(scaler,1920,1200) || output.device!=cb.device || output.framebufferOnly ||
     output.textureType!=MTLTextureType2D || output.sampleCount!=1 || output.storageMode==MTLStorageModeMemoryless ||
     (output.hazardTrackingMode==MTLHazardTrackingModeUntracked && !scaler.fence))return;
  if(job) {
    const auto current=job->state.snapshot();
    if((current.phase==Phase::Recovering || (current.phase==Phase::Failed && current.publication_may_be_encoded)) &&
       stream==current.stream && frame>current.publication_frame) {
      [cb addCompletedHandler:^(id<MTLCommandBuffer> done){
        DelayedSafe(job,[&]{
          if(job->state.original_completed(frame,stream,done.status==MTLCommandBufferStatusCompleted))
            Log("event=nr_delayed_original_recovered frame=%llu stream=%llu continuous_nr=0",
              (unsigned long long)frame,(unsigned long long)stream);
        });
      }];
    }
    const bool released=job->publish_path.empty() || access(job->publish_path.c_str(),F_OK)==0;
    if(attempted.load() || !job->state.begin_publication(frame,stream,1920,1200,released))return;
    job->publication_command=NRObjectID(cb);
    job->capture->publication_frame=frame;job->capture->publication_command_id=job->publication_command;
    MarkNRCommand(cb,frame,stream);
    auto fence=scaler.fence;
    [cb addCompletedHandler:^(id<MTLCommandBuffer> done){
      (void)output;(void)fence; // Retain game resources through unretained CB completion only.
      DelayedSafe(job,[&]{
        const bool ok=done.status==MTLCommandBufferStatusCompleted;
        if(job->state.publication_completed(ok)) {
          nr_published.fetch_add(1);
          Log("event=nr_delayed_export_completed source_frame=%llu publication_frame=%llu screen_pixels_not_yet_verified=1",
            (unsigned long long)job->capture->frame,(unsigned long long)frame);
          const auto keep=job;
          dispatch_async(keep->host_queue,^{DelayedSafe(keep,[&]{DelayedWrite(keep,"published",true);});});
        }
      });
    }];
    nr::FrameEncoderState ownership;
    auto b=[cb blitCommandEncoder];nr::EncoderScope bs(ownership,b);DelayedRequire(b!=nil,"publication encoder");
    if(fence)[b waitForFence:fence];
    if(!job->state.commit_publication()){DelayedRequire(bs.close(),"cancelled publication close");return;}
    [b copyFromTexture:job->publication toTexture:output];
    netal_nr_adapter::CopyCapture(b,job->capture,3,output);
    if(fence)[b updateFence:fence];
    DelayedRequire(bs.close(),"publication close");
    Log("event=nr_delayed_publication_scheduled source_frame=%llu publication_frame=%llu frozen_reset_diagnostic=1 continuous_nr=0",
      (unsigned long long)job->capture->frame,(unsigned long long)frame);
    return;
  }
  if(attempted.load())return;
  if(!arm_file.empty() && access((arm_file+".cancel").c_str(),F_OK)==0) {
    attempted.store(true);Log("event=nr_delayed_cancelled_before_capture original_renderer_retained=1");return;
  }
  StartPrepare(cb.device);
  if(arm_file.empty() || access(arm_file.c_str(),F_OK)!=0)return;
  if(delayed_phase_gates && (access((arm_file+".compute").c_str(),F_OK)==0 || access((arm_file+".publish").c_str(),F_OK)==0)) {
    attempted.store(true);Log("event=nr_delayed_declined reason=release_marker_predates_capture original_renderer_retained=1");return;
  }
  nr::ResetWorkerPtr worker;nr::Report prepared;
  {std::lock_guard<std::mutex> lock(state_mutex);if(ready!=Ready::Ready || !reset_worker)return;worker=reset_worker;prepared=reset_prepare_report;}
  auto candidate=std::make_shared<DelayedNRJob>();candidate->worker=worker;candidate->cancel_path=arm_file+".cancel";
  if(delayed_phase_gates){candidate->compute_path=arm_file+".compute";candidate->publish_path=arm_file+".publish";}
  DelayedRequire(netal_nr_adapter::MakeAspectLayout(1920,1200,candidate->layout),"delayed aspect");
  candidate->capture=netal_nr_adapter::MakeFrameCapture(cb.device,capture_root.c_str(),candidate->layout,output.pixelFormat,frame,stream);
  DelayedRequire(bool(candidate->capture),"delayed capture allocation");
  candidate->capture->color_mode="bounded-reinhard-srgb";candidate->capture->color_reconstruction="inverse-reinhard-srgb";
  candidate->capture->execution_mode="delayed-reset-once";candidate->capture->model_identity=prepared.model_identity;
  candidate->capture->arithmetic=prepared.arithmetic_label;candidate->capture->command_id=NRObjectID(cb);
  candidate->capture->game_texture_id=NRObjectID(output);candidate->capture_command=NRObjectID(cb);
  auto d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRG11B10Float width:1920 height:1200 mipmapped:NO];
  d.storageMode=MTLStorageModePrivate;d.hazardTrackingMode=MTLHazardTrackingModeTracked;d.usage=MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite;
  candidate->source=[cb.device newTextureWithDescriptor:d];candidate->publication=[cb.device newTextureWithDescriptor:d];
  d.pixelFormat=MTLPixelFormatRGBA16Float;d.height=1080;
  candidate->fitted=[cb.device newTextureWithDescriptor:d];candidate->bounded=[cb.device newTextureWithDescriptor:d];
  candidate->export_copy=[cb.device newBufferWithLength:1920*1080*8 options:MTLResourceStorageModePrivate|MTLResourceHazardTrackingModeTracked];
  candidate->rejected=[cb.device newBufferWithLength:4 options:MTLResourceStorageModeShared|MTLResourceHazardTrackingModeTracked];
  candidate->gpu_queue=[cb.device newCommandQueue];
  DelayedRequire(candidate->source && candidate->publication && candidate->fitted && candidate->bounded &&
    candidate->export_copy && candidate->rejected && candidate->gpu_queue,"delayed resources");
  {std::lock_guard<std::mutex> lock(delayed_job_mutex);if(delayed_job)return;
    DelayedRequire(candidate->state.begin_capture(frame,stream,1920,1200),"delayed capture admission");delayed_job=candidate;}
  job=candidate;auto fence=scaler.fence;
  [cb addCompletedHandler:^(id<MTLCommandBuffer> done){
    (void)output;(void)fence;
    const bool ok=done.status==MTLCommandBufferStatusCompleted;
    if(!job->state.capture_completed(ok,!job->compute_path.empty()))return;
    dispatch_async(job->host_queue,^{@autoreleasepool{DelayedSafe(job,[&]{DelayedWrite(job,"source",false);DelayedPrepareInput(job);});}});
  }];
  nr::FrameEncoderState ownership;
  auto b=[cb blitCommandEncoder];nr::EncoderScope bs(ownership,b);DelayedRequire(b!=nil,"source capture encoder");
  if(fence)[b waitForFence:fence];
  [b copyFromTexture:output toTexture:job->source];netal_nr_adapter::CopyCapture(b,job->capture,0,output);
  if(fence)[b updateFence:fence];
  DelayedRequire(bs.close(),"source capture close");
  Log("event=nr_delayed_capture_scheduled source_frame=%llu stream=%llu command_id=%llu model_not_on_game_command=1",
    (unsigned long long)frame,(unsigned long long)stream,(unsigned long long)job->capture_command);
}
