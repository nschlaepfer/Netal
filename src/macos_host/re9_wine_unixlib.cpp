#include "netal_launch_config.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <unistd.h>

using NTSTATUS = std::int32_t;
using UnixEntry = NTSTATUS (*)(void *);

static void append_log(const std::string& path, const char* event, const std::string& detail) {
  if (FILE* stream=std::fopen(path.c_str(),"a")) {
    std::fprintf(stream,"event=%s detail=%s\n",event,detail.c_str());std::fclose(stream);
  }
}
static void optional_env(const char* name,const std::string& value) {
  if (value.empty()) unsetenv(name);else setenv(name,value.c_str(),1);
}
static void flag_env(const char* name,const std::string& value) {
  if (value=="1") setenv(name,"1",1);else unsetenv(name);
}

static NTSTATUS load_re9_bridge(void*) {
  try {
    // The launcher writes a private process-lifetime configuration. No checkout path or
    // user name is compiled into this library, and a recycled PID cannot reuse stale state.
    const auto process=netal::Identity(getpid());
    const auto config=netal::ReadConfig(process);
    const auto directory=std::filesystem::path(config.at("log_dir"));
    const auto unix_log=(directory/"re9-live-unixlib.log").string();
    const auto metal_log=(directory/"re9-live-metalfx.log").string();
    if (!std::filesystem::is_regular_file(config.at("bridge"))) throw std::runtime_error("Configured bridge is missing");
    setenv("NEURALSTUDIO_RE9_METALFX_NOOP","1",1);
    setenv("NEURALSTUDIO_RE9_METALFX_LOG",metal_log.c_str(),1);
    setenv("NEURALSTUDIO_RE9_SR_BUNDLE",config.at("sr_bundle").c_str(),1);
    setenv("NEURALSTUDIO_RE9_FG_BUNDLE",config.at("fg_bundle").c_str(),1);
    flag_env("NEURALSTUDIO_RE9_SR_ARM",config.at("arm_sr"));
    flag_env("NEURALSTUDIO_RE9_FG_ARM",config.at("arm_fg"));
    flag_env("NEURALSTUDIO_RE9_SR_SUSTAINED",config.at("sustained"));
    optional_env("NEURALSTUDIO_SR_PWIN_SUFFIX",config.at("pwin_suffix"));
    setenv("NEURALSTUDIO_SR_PWIN_PIPELINE",config.at("pipeline").c_str(),1);
    const std::pair<const char*,const char*> options[]={
      {"target_exe","NEURALSTUDIO_TARGET_EXE"}, {"commit_hook","NEURALSTUDIO_SR_METAL4_COMMIT_HOOK"},
      {"dump_spec","NEURALSTUDIO_RE9_SR_DUMP"}, {"jitter_sign","NEURALSTUDIO_RE9_JITTER_SIGN"},
      {"mv_sign","NEURALSTUDIO_RE9_MV_SIGN"}, {"fg_dump","NEURALSTUDIO_FG_DUMP_DIR"}};
    for (const auto& option:options) optional_env(option.second,config.at(option.first));
    flag_env("NEURALSTUDIO_RE9_FORCE_VSYNC",config.at("force_vsync"));
    append_log(unix_log,"unix_entry",config.at("bridge"));
    dlerror();
    void* module=dlopen(config.at("bridge").c_str(),RTLD_NOW|RTLD_LOCAL);
    if (!module) {
      const char* error=dlerror();append_log(unix_log,"bridge_failed",error?error:"unknown");
      return static_cast<NTSTATUS>(0xc0000135u);
    }
    append_log(unix_log,"bridge_loaded",config.at("bridge"));return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr,"netal: launch configuration refused: %s\n",error.what());
    return static_cast<NTSTATUS>(0xc000000du);
  }
}
extern "C" __attribute__((visibility("default"),used))
const UnixEntry __wine_unix_call_funcs[]={load_re9_bridge, [](void*) -> NTSTATUS { return getpid(); }};
