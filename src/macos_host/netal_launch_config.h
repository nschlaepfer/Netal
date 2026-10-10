#pragma once

#include <sys/stat.h>
#include <sys/types.h>
#include <libproc.h>
#include <pwd.h>
#include <unistd.h>
#include <fcntl.h>
#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace netal {
struct ProcessIdentity {
  pid_t pid;
  uid_t uid;
  unsigned long long seconds;
  unsigned long long microseconds;
};

inline ProcessIdentity Identity(pid_t pid) {
  proc_bsdinfo info{};
  if (pid <= 0 || proc_pidinfo(pid, PROC_PIDTBSDINFO, 0, &info, sizeof(info)) != sizeof(info))
    throw std::runtime_error("Cannot read target process identity");
  if (info.pbi_uid != getuid()) throw std::runtime_error("Target process belongs to another user");
  return {pid, info.pbi_uid, info.pbi_start_tvsec, info.pbi_start_tvusec};
}
inline std::filesystem::path SessionRoot() {
  passwd user{};passwd* found=nullptr;
  std::vector<char> buffer(16384);
  if (getpwuid_r(getuid(), &user, buffer.data(), buffer.size(), &found)!=0 || !found || !user.pw_dir || !*user.pw_dir)
    throw std::runtime_error("Cannot resolve the current user's home");
  return std::filesystem::path(user.pw_dir) / "Library/Application Support/Netal/Sessions";
}

inline std::filesystem::path SessionDirectory(const ProcessIdentity& process) {
  return SessionRoot() / (std::to_string(process.pid) + "-" + std::to_string(process.seconds) + "-" + std::to_string(process.microseconds));
}
using Config = std::map<std::string, std::string>;
inline bool PlainValue(const std::string& value) {
  return value.size() <= 4096 && value.find_first_of("\r\n") == std::string::npos && value.find('\0') == std::string::npos;
}
inline std::string Required(const Config& config, const std::string& key) {
  const auto found=config.find(key);
  if (found==config.end() || found->second.empty()) throw std::runtime_error("Missing configuration field: "+key);
  return found->second;
}
inline void Validate(const Config& config, const ProcessIdentity& process) {
  static const std::vector<std::string> keys={"pid","uid","start_sec","start_usec","bridge","sr_bundle","fg_bundle",
    "log_dir","target_exe","arm_sr","arm_fg","sustained","pwin_suffix","pipeline","commit_hook","dump_spec","jitter_sign","mv_sign","force_vsync","fg_dump"};
  for (const auto& item:config) {
    if (std::find(keys.begin(),keys.end(),item.first)==keys.end() || !PlainValue(item.second))
      throw std::runtime_error("Unknown or invalid configuration field: "+item.first);
  }
  for (const auto& key:keys) if (config.find(key)==config.end()) throw std::runtime_error("Missing configuration field: "+key);
  if (Required(config,"pid")!=std::to_string(process.pid) || Required(config,"uid")!=std::to_string(process.uid) ||
      Required(config,"start_sec")!=std::to_string(process.seconds) || Required(config,"start_usec")!=std::to_string(process.microseconds))
    throw std::runtime_error("Configuration belongs to a different process lifetime");
  for (const char* key:{"bridge","sr_bundle","fg_bundle","log_dir"})
    if (!std::filesystem::path(Required(config,key)).is_absolute()) throw std::runtime_error(std::string("Path must be absolute: ")+key);
  if (config.at("log_dir") != SessionDirectory(process).string()) throw std::runtime_error("Logs must stay in the process session directory");
  for (const char* key:{"arm_sr","arm_fg","sustained","pipeline","force_vsync"}) {
    const auto value=Required(config,key);
    if (value!="0" && value!="1") throw std::runtime_error(std::string("Expected boolean: ")+key);
  }
  if (config.at("sustained")=="1" && config.at("arm_sr")!="1") throw std::runtime_error("Sustained SR requires SR arming");
  const auto target=Required(config,"target_exe");
  if (target.size()>255 || target.find_first_of("/\\")!=std::string::npos || target=="." || target=="..")
    throw std::runtime_error("Expected a bare game executable name");
  const auto suffix=config.at("pwin_suffix");
  if (suffix.size()>64 || suffix.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")!=std::string::npos)
    throw std::runtime_error("Invalid kernel suffix");
  const auto hook=config.at("commit_hook");
  if (!hook.empty() && hook!="observe" && hook!="feedback") throw std::runtime_error("Invalid commit-hook mode");
}
inline Config Parse(const std::string& text, const ProcessIdentity& process) {
  if (text.size()>65536 || text.find('\0')!=std::string::npos) throw std::runtime_error("Configuration is oversized or contains NUL");
  std::istringstream stream(text);std::string line;
  if (!std::getline(stream,line) || line!="netal.runtime.launch.v1") throw std::runtime_error("Unsupported launch configuration version");
  Config config;
  while (std::getline(stream,line)) {
    const auto split=line.find('=');
    if (split==std::string::npos || split==0 || !config.emplace(line.substr(0,split),line.substr(split+1)).second)
      throw std::runtime_error("Malformed or duplicate launch field");
  }
  Validate(config,process);return config;
}
inline std::string Serialize(const Config& config, const ProcessIdentity& process) {
  Validate(config,process);std::string text="netal.runtime.launch.v1\n";
  for (const auto& item:config) text+=item.first+"="+item.second+"\n";
  return text;
}
inline Config ReadConfig(const ProcessIdentity& process) {
  const auto path=SessionDirectory(process)/"launch.conf";
  const int fd=open(path.c_str(),O_RDONLY|O_NOFOLLOW|O_CLOEXEC);
  if (fd<0) throw std::runtime_error("No private launch configuration for this process");
  struct stat info{};
  if (fstat(fd,&info)!=0 || !S_ISREG(info.st_mode) || info.st_uid!=process.uid || (info.st_mode&077)!=0 || info.st_size>65536) {
    close(fd);throw std::runtime_error("Unsafe launch configuration ownership, permissions or size");
  }
  std::string text;char buffer[4096];ssize_t count;
  while ((count=read(fd,buffer,sizeof(buffer)))>0) {
    text.append(buffer,static_cast<size_t>(count));
    if (text.size()>65536) {close(fd);throw std::runtime_error("Launch configuration grew beyond its limit");}
  }
  close(fd);if (count<0) throw std::runtime_error("Cannot read launch configuration");
  return Parse(text,process);
}
inline void PrivateDirectory(const std::filesystem::path& path) {
  std::filesystem::create_directories(path);
  struct stat info{};
  if (lstat(path.c_str(),&info)!=0 || !S_ISDIR(info.st_mode) || info.st_uid!=getuid())
    throw std::runtime_error("Launch directory is not an owned real directory");
  if (chmod(path.c_str(),0700)!=0) throw std::runtime_error("Cannot protect launch directory");
}
inline void WriteConfig(const Config& config, const ProcessIdentity& process) {
  const auto text=Serialize(config,process);
  const auto root=SessionRoot(), directory=SessionDirectory(process);
  PrivateDirectory(root);PrivateDirectory(directory);
  const auto path=directory/"launch.conf";
  // Use an exclusive temporary file and atomic rename; never follow an existing config link.
  const auto temporary=directory/(".launch-"+std::to_string(getpid())+".tmp");
  const int fd=open(temporary.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC,0600);
  if (fd<0) throw std::runtime_error("Cannot create private launch configuration");
  size_t offset=0;
  while (offset<text.size()) {
    const auto written=write(fd,text.data()+offset,text.size()-offset);
    if (written<=0) {close(fd);unlink(temporary.c_str());throw std::runtime_error("Cannot write launch configuration");}
    offset+=static_cast<size_t>(written);
  }
  const bool flushed=fsync(fd)==0;close(fd);
  if (!flushed || rename(temporary.c_str(),path.c_str())!=0) {unlink(temporary.c_str());throw std::runtime_error("Cannot publish launch configuration");}
}
}  // namespace netal
