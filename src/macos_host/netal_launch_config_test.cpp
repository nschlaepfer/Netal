#include "netal_launch_config.h"
#include <iostream>
#include <dlfcn.h>
#include <functional>

unsigned checks=0;
void Check(bool value,const char* message) {if (!value) throw std::runtime_error(message);++checks;}
void Reject(const std::function<void()>& operation,const char* message) {
  try {operation();} catch (const std::exception&) {++checks;return;}throw std::runtime_error(message);
}
int main(int argc,char** argv) {
  std::filesystem::path directory;
  bool owned=false;
  try {
    if (argc!=3) throw std::runtime_error("expected unix helper and fixture bridge paths");
    const auto process=netal::Identity(getpid());directory=netal::SessionDirectory(process);
    Check(!std::filesystem::exists(directory),"test session unexpectedly exists");
    netal::Config config={{"pid",std::to_string(process.pid)},{"uid",std::to_string(process.uid)},
      {"start_sec",std::to_string(process.seconds)},{"start_usec",std::to_string(process.microseconds)},
      {"bridge",std::filesystem::absolute(argv[2]).string()},{"sr_bundle","/test assets/SR=bundle"},{"fg_bundle","/test assets/FG"},
      {"log_dir",directory.string()},{"target_exe","Test Game.exe"},{"arm_sr","1"},{"arm_fg","0"},
      {"sustained","1"},{"pipeline","1"},{"pwin_suffix","_hw7"},{"commit_hook",""},{"dump_spec",""},
      {"jitter_sign",""},{"mv_sign",""},{"force_vsync","0"},{"fg_dump",""}};
    const auto serialized=netal::Serialize(config,process);
    Check(netal::Parse(serialized,process)==config,"round trip failed for spaces/equal signs");
    auto stale=process;++stale.microseconds;
    Reject([&]{netal::Parse(serialized,stale);},"stale PID lifetime accepted");
    Reject([&]{netal::Parse(serialized+"arm_sr=1\n",process);},"duplicate field accepted");
    Reject([&]{netal::Parse(serialized+"unknown=1\n",process);},"unknown field accepted");
    auto invalid=config;invalid["target_exe"]="../other.exe";
    Reject([&]{netal::Serialize(invalid,process);},"path target accepted");
    invalid=config;invalid["bridge"]="relative.dylib";
    Reject([&]{netal::Serialize(invalid,process);},"relative bridge accepted");
    invalid=config;invalid["bridge"]="/tmp/line\nbreak";
    Reject([&]{netal::Serialize(invalid,process);},"newline accepted");
    invalid=config;invalid["arm_sr"]="0";
    Reject([&]{netal::Serialize(invalid,process);},"sustained without SR accepted");
    owned=true;
    netal::WriteConfig(config,process);
    Check(netal::ReadConfig(process)==config,"private config read failed");
    const auto path=directory/"launch.conf";
    chmod(path.c_str(),0644);
    Reject([&]{netal::ReadConfig(process);},"public-readable config accepted");
    chmod(path.c_str(),0600);
    std::filesystem::rename(path,directory/"saved.conf");
    std::filesystem::create_symlink(directory/"saved.conf",path);
    Reject([&]{netal::ReadConfig(process);},"linked config accepted");
    std::filesystem::remove(path);std::filesystem::rename(directory/"saved.conf",path);
    const auto fixture_receipt=directory/"fixture.txt";
    setenv("NETAL_TEST_RECEIPT",fixture_receipt.c_str(),1);
    void* library=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);
    if (!library) throw std::runtime_error(dlerror());
    using Entry=int32_t(*)(void*);
    auto functions=static_cast<const Entry*>(dlsym(library,"__wine_unix_call_funcs"));
    Check(functions && functions[0](nullptr)==0,"portable helper failed to load configured fixture");
    std::ifstream receipt(fixture_receipt);std::string body((std::istreambuf_iterator<char>(receipt)),{});
    Check(body.find("target=Test Game.exe")!=std::string::npos,"target environment missing");
    Check(body.find("sr=/test assets/SR=bundle")!=std::string::npos,"resource path changed");
    Check(body.find("arm_sr=1")!=std::string::npos && body.find("arm_fg=<unset>")!=std::string::npos,"feature flags wrong");
    std::filesystem::remove(path);
    Check(functions[1](nullptr)==getpid(),"process discovery needs configuration or returned the wrong PID");
    Check(functions[0](nullptr)!=0,"missing configuration did not fail closed");
    std::filesystem::remove_all(directory);directory.clear();
    std::cout << checks << " launch configuration checks passed\n";return 0;
  } catch (const std::exception& error) {
    if (owned && !directory.empty()) {std::error_code ignored;std::filesystem::remove_all(directory,ignored);}
    std::cerr << "failure: " << error.what() << '\n';return 1;
  }
}
