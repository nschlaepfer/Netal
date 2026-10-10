/* NeuralStudio Runtime v0 host - SPDX-License-Identifier: MIT */

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX

#include "../../runtime/runtime_host/runtime_engine.h"
#include "../../runtime/control/local_api_server.h"

#include <windows.h>

#include <charconv>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>

namespace {

bool ParseUnsigned(std::string_view text, std::uint32_t& value) {
  const char* begin = text.data();
  const char* end = begin + text.size();
  const auto parsed = std::from_chars(begin, end, value);
  return parsed.ec == std::errc{} && parsed.ptr == end;
}

std::filesystem::path ExecutableDirectory() {
  std::wstring storage(32768, L'\0');
  const DWORD size = GetModuleFileNameW(nullptr, storage.data(),
                                        static_cast<DWORD>(storage.size()));
  if (size == 0 || size >= storage.size()) return {};
  storage.resize(size);
  return std::filesystem::path(storage).parent_path();
}

void Usage() {
  std::cerr
      << "Usage: neuralstudio-runtime.exe --diagnostic --arm-inference "
         "[--package PATH] [--output PATH] [--width N] [--height N] "
         "[--frames N]\n"
      << "   or: neuralstudio-runtime.exe --service [--arm-inference] "
         "[--package PATH] [--output PATH]\n";
}

}  // namespace

int main(int argc, char** argv) {
  neuralstudio::DiagnosticOptions options;
  const std::filesystem::path root = ExecutableDirectory();
  options.package_root = root / "packages" / "com.nvidia.ngx.dlss-nr";
  options.output_path = root / "output" / "diagnostic.bmp";
  bool diagnostic = false;
  bool service = false;

  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    auto take = [&](const char* name) -> const char* {
      if (index + 1 >= argc) {
        std::cerr << name << " requires a value\n";
        return nullptr;
      }
      return argv[++index];
    };
    if (argument == "--diagnostic") diagnostic = true;
    else if (argument == "--service") service = true;
    else if (argument == "--arm-inference") options.inference_armed = true;
    else if (argument == "--package") {
      const char* value = take("--package");
      if (value == nullptr) return 2;
      options.package_root = value;
    } else if (argument == "--output") {
      const char* value = take("--output");
      if (value == nullptr) return 2;
      options.output_path = value;
    } else if (argument == "--width") {
      const char* value = take("--width");
      if (value == nullptr || !ParseUnsigned(value, options.width)) return 2;
    } else if (argument == "--height") {
      const char* value = take("--height");
      if (value == nullptr || !ParseUnsigned(value, options.height)) return 2;
    } else if (argument == "--frames") {
      const char* value = take("--frames");
      if (value == nullptr || !ParseUnsigned(value, options.evaluation_frames)) return 2;
    } else {
      Usage();
      return 2;
    }
  }
  if (diagnostic == service) {
    Usage();
    return 2;
  }
  std::error_code error;
  std::filesystem::create_directories(options.output_path.parent_path(), error);
  if (error) {
    std::cerr << "{\"ok\":false,\"detail\":\"Unable to create output directory\"}\n";
    return 1;
  }
  if (service) {
    neuralstudio::LocalApiOptions server;
    server.default_package = options.package_root;
    server.default_output = options.output_path;
    server.inference_armed = options.inference_armed;
    return neuralstudio::RunLocalApiServer(server);
  }
  const neuralstudio::DiagnosticResult result =
      neuralstudio::RunDiagnostic(options);
  std::cout << neuralstudio::DiagnosticResultJson(result) << '\n';
  return result.ok ? 0 : 1;
}
