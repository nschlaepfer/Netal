#include "netal_launch_config.h"
#include <iostream>

int main(int argc,char** argv) {
  try {
    if (argc<3) throw std::runtime_error("usage: netal-launch-config info|prepare <pid> [--key value ...]");
    size_t consumed=0;const auto number=std::stol(argv[2],&consumed);
    if (consumed!=std::string(argv[2]).size() || number<=0 || number>INT32_MAX) throw std::runtime_error("Invalid PID");
    const auto process=netal::Identity(static_cast<pid_t>(number));
    const auto directory=netal::SessionDirectory(process);
    if (!netal::PlainValue(directory.string())) throw std::runtime_error("Unsupported session path");
    if (std::string(argv[1])=="info") {
      if (argc!=3) throw std::runtime_error("Unexpected info arguments");
      std::cout << "pid=" << process.pid << "\nuid=" << process.uid << "\nstart_sec=" << process.seconds
                << "\nstart_usec=" << process.microseconds << "\nsession_dir=" << directory.string()
                << "\nlog=" << (directory/"re9-live-metalfx.log").string() << '\n';return 0;
    }
    if (std::string(argv[1])!="prepare") throw std::runtime_error("Unknown action");
    netal::Config config={{"pid",std::to_string(process.pid)},{"uid",std::to_string(process.uid)},
      {"start_sec",std::to_string(process.seconds)},{"start_usec",std::to_string(process.microseconds)},
      {"log_dir",directory.string()},{"arm_sr","0"},{"arm_fg","0"},{"sustained","0"},{"pipeline","0"},
      {"pwin_suffix",""},{"commit_hook",""},{"dump_spec",""},{"jitter_sign",""},{"mv_sign",""},{"force_vsync","0"},{"fg_dump",""}};
    std::vector<std::string> supplied;
    for (int i=3;i<argc;i+=2) {
      if (i+1>=argc || std::string(argv[i]).rfind("--",0)!=0) throw std::runtime_error("Expected --key value pairs");
      const std::string key=std::string(argv[i]).substr(2);
      if (key=="pid" || key=="uid" || key=="start_sec" || key=="start_usec" || key=="log_dir" ||
          std::find(supplied.begin(),supplied.end(),key)!=supplied.end()) throw std::runtime_error("Reserved or duplicate option");
      supplied.push_back(key);config[key]=argv[i+1];
    }
    netal::Validate(config,process);
    if (!std::filesystem::is_regular_file(config.at("bridge"))) throw std::runtime_error("Bridge file is missing");
    for (const auto& feature: {std::pair{"arm_sr","sr_bundle"},std::pair{"arm_fg","fg_bundle"}})
      if (config.at(feature.first)=="1" && !std::filesystem::is_regular_file(std::filesystem::path(config.at(feature.second))/"manifest.json"))
        throw std::runtime_error(std::string("Runtime manifest missing: ")+feature.second);
    netal::WriteConfig(config,process);
    std::cout << "config=" << (directory/"launch.conf").string() << "\nlog=" << (directory/"re9-live-metalfx.log").string() << "\nresult=pass\n";
    return 0;
  } catch (const std::exception& error) {std::cerr << "result=fail " << error.what() << '\n';return 1;}
}
