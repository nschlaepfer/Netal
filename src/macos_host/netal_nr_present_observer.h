#pragma once
// Included inside the bridge's private namespace AFTER Log(). Opt-in diagnostics
// only. No drawable retention cycles, scheduling changes, vsync or layer mutation.
struct NRPresentHook { Class type; SEL selector; IMP original, replacement; };
std::mutex nr_present_mutex, nr_object_id_mutex;
std::vector<NRPresentHook> nr_present_hooks;
char nr_object_id_key;
std::atomic<uint64_t> nr_object_serial{0}, nr_trace_command{0}, nr_trace_frame{0}, nr_trace_stream{0};
std::atomic<unsigned> nr_present_scheduled_logs{0}, nr_presented_logs{0};
thread_local void* nr_present_current=nullptr;
struct NRPresentCall {
  void* previous;
  bool nested;
  explicit NRPresentCall(id object):previous(nr_present_current),nested(previous==(__bridge void*)object){nr_present_current=(__bridge void*)object;}
  ~NRPresentCall(){nr_present_current=previous;}
};
uint64_t NRObjectID(id object) {
  if(!object)return 0;
  std::lock_guard<std::mutex> lock(nr_object_id_mutex);
  NSNumber* n=objc_getAssociatedObject(object,&nr_object_id_key);
  if(!n){n=@(++nr_object_serial);objc_setAssociatedObject(object,&nr_object_id_key,n,OBJC_ASSOCIATION_RETAIN_NONATOMIC);}
  return n.unsignedLongLongValue;
}
void NRObservePresent(id cb,id<MTLDrawable> drawable,SEL selector) {
  if(!drawable || (nr_present_scheduled_logs.load()>=64 && nr_presented_logs.load()>=64))return;
  const uint64_t command=NRObjectID(cb), object=NRObjectID(drawable);
  const uint64_t drawable_id=drawable.drawableID;
  id layer=nil;id<MTLTexture> texture=nil;
  if([(id)drawable respondsToSelector:@selector(layer)])layer=reinterpret_cast<id(*)(id,SEL)>(objc_msgSend)(drawable,@selector(layer));
  if([(id)drawable respondsToSelector:@selector(texture)])texture=reinterpret_cast<id(*)(id,SEL)>(objc_msgSend)(drawable,@selector(texture));
  const uint64_t layer_id=NRObjectID(layer),texture_id=NRObjectID(texture);
  if(nr_trace_command.load() && nr_present_scheduled_logs.fetch_add(1)<64)
    Log("event=drawable_present_scheduled command_id=%llu drawable_object_id=%llu drawable_id=%llu layer_id=%llu texture_id=%llu method=%s same_nr_command=%d",
      (unsigned long long)command,(unsigned long long)object,(unsigned long long)drawable_id,(unsigned long long)layer_id,(unsigned long long)texture_id,sel_getName(selector),command==nr_trace_command.load());
  if(![(id)drawable respondsToSelector:@selector(addPresentedHandler:)])return;
  // Capture only numeric metadata. Capturing the same drawable/CB strongly here
  // could create a cycle or exhaust the layer's finite drawable pool.
  [drawable addPresentedHandler:^(id<MTLDrawable> shown) {
    const uint64_t marked=nr_trace_command.load();
    if(!marked || nr_presented_logs.fetch_add(1)>=64)return;
    const bool same=command==marked;
    Log("event=drawable_presented command_id=%llu drawable_object_id=%llu drawable_id=%llu layer_id=%llu texture_id=%llu presented_time=%.9f same_nr_command=%d nr_frame=%llu nr_stream=%llu pixel_content_verified=0",
      (unsigned long long)command,(unsigned long long)object,(unsigned long long)drawable_id,(unsigned long long)layer_id,(unsigned long long)texture_id,shown.presentedTime,same,
      (unsigned long long)(same?nr_trace_frame.load():0),(unsigned long long)(same?nr_trace_stream.load():0));
#if NETAL_NR_DELAYED_ONCE && !NETAL_NR_OBSERVE_ONLY
    if(same && shown.presentedTime>0)NRDelayedPresented(command);
#endif
  }];
}
void InstallNRPresentObserver(id cb) {
  Class cls=object_getClass(cb);
  const SEL selectors[]={@selector(presentDrawable:),@selector(presentDrawable:atTime:),@selector(presentDrawable:afterMinimumDuration:)};
  std::lock_guard<std::mutex> lock(nr_present_mutex);
  for(unsigned i=0;i<3;++i){
    SEL sel=selectors[i];bool installed=false;
    for(const auto& h:nr_present_hooks)if(h.type==cls && h.selector==sel)installed=true;
    if(installed)continue;
    Method method=class_getInstanceMethod(cls,sel);if(!method)continue;
    IMP original=method_getImplementation(method);
    for(const auto& h:nr_present_hooks)if(h.selector==sel && original==h.replacement){original=h.original;break;}
    // Bind each replacement to ITS original implementation. Looking up by the
    // object's dynamic class would recurse if an overridden method calls super.
    IMP replacement;
    if(i==0)replacement=imp_implementationWithBlock(^(id object,id<MTLDrawable> drawable){
      NRPresentCall call(object);
      if(!call.nested){@try{try{NRObservePresent(object,drawable,sel);}catch(...){}}@catch(id exception){(void)exception;}}
      reinterpret_cast<void(*)(id,SEL,id<MTLDrawable>)>(original)(object,sel,drawable);
    });
    else replacement=imp_implementationWithBlock(^(id object,id<MTLDrawable> drawable,CFTimeInterval time){
      NRPresentCall call(object);
      if(!call.nested){@try{try{NRObservePresent(object,drawable,sel);}catch(...){}}@catch(id exception){(void)exception;}}
      reinterpret_cast<void(*)(id,SEL,id<MTLDrawable>,CFTimeInterval)>(original)(object,sel,drawable,time);
    });
    if(!replacement)continue;
    try{nr_present_hooks.push_back({cls,sel,original,replacement});}catch(...){imp_removeBlock(replacement);throw;}
    if(!class_addMethod(cls,sel,replacement,method_getTypeEncoding(method)))
      method_setImplementation(class_getInstanceMethod(cls,sel),replacement);
  }
}
[[maybe_unused]] void MarkNRCommand(id cb,uint64_t frame,uint64_t stream) {
  nr_trace_frame.store(frame);nr_trace_stream.store(stream);
  nr_trace_command.store(NRObjectID(cb));
  Log("event=nr_command_identity command_id=%llu frame=%llu stream=%llu process_local_ids=1",
    (unsigned long long)nr_trace_command.load(),(unsigned long long)frame,(unsigned long long)stream);
}
