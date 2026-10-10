#pragma once
#include <chrono>
#include <cmath>
#include <map>

// Host-only timing. No shader, buffer binding, dispatch, or validation changes.
struct NrTimingRecord {std::string label;double gpu_ms,submit_wait_ms,host_scope_ms;};
static std::vector<NrTimingRecord> nr_timings;
static std::vector<std::pair<std::string,double>> nr_phases;
using NrClock=std::chrono::steady_clock;
inline double NrElapsed(NrClock::time_point start){return std::chrono::duration<double,std::milli>(NrClock::now()-start).count();}
// Mutually exclusive phases within one invocation. These refine (overlap) the
// enclosing host scope; never add them to that scope or to GPU duration.
struct NrPhaseClock {
  std::string prefix,phase;NrClock::time_point start;
  NrPhaseClock(std::string p,std::string first):prefix(std::move(p)),phase(std::move(first)),start(NrClock::now()){}
  void next(std::string value){nr_phases.emplace_back(prefix+"/"+phase,NrElapsed(start));phase=std::move(value);start=NrClock::now();}
  ~NrPhaseClock(){nr_phases.emplace_back(prefix+"/"+phase,NrElapsed(start));}
};
struct NrHostScope {
  std::string label;NrClock::time_point start;
  explicit NrHostScope(std::string value):label(std::move(value)),start(NrClock::now()){}
  ~NrHostScope(){nr_timings.push_back({label,0,0,NrElapsed(start)});}
};
inline void NrComplete(id<MTLCommandBuffer> command){
  auto start=NrClock::now();[command commit];[command waitUntilCompleted];
  if(command.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error(command.error.localizedDescription.UTF8String);
  double gpu=(command.GPUEndTime-command.GPUStartTime)*1000.0;
  if(!std::isfinite(gpu)||gpu<0)throw std::runtime_error("invalid GPU timestamp");
  nr_timings.push_back({command.label?std::string(command.label.UTF8String):"unlabelled",gpu,NrElapsed(start),0});
}
inline void NrDumpProfile(const fs::path& directory){
  NSMutableArray* rows=[NSMutableArray array];
  for(const auto&r:nr_timings)[rows addObject:@{@"label":[NSString stringWithUTF8String:r.label.c_str()],@"gpu_ms":@(r.gpu_ms),@"submit_wait_ms":@(r.submit_wait_ms),@"host_scope_ms":@(r.host_scope_ms)}];
  NSMutableArray* phases=[NSMutableArray array];
  for(const auto&r:nr_phases)[phases addObject:@{@"label":[NSString stringWithUTF8String:r.first.c_str()],@"wall_ms":@(r.second)}];
  NSDictionary* result=@{@"schema":@"neuralstudio.nr.encoder-timing.v1",@"pid":@(getpid()),@"records":rows,@"phases":phases,@"shader_changes":@NO,@"scope":@"Instrumented acceptance harness; phases refine and overlap host scopes, which include GPU wait. Not production frame time."};
  NSError*error=nil;NSData*data=[NSJSONSerialization dataWithJSONObject:result options:NSJSONWritingPrettyPrinted error:&error];
  if(!data)throw std::runtime_error(error.localizedDescription.UTF8String);
  std::ofstream f(directory/"profile.json",std::ios::binary);f.write(static_cast<const char*>(data.bytes),static_cast<std::streamsize>(data.length));if(!f)throw std::runtime_error("profile write");
}
