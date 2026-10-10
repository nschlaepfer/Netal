/* NeuralStudio Local API v1 named-pipe server - SPDX-License-Identifier: MIT */

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX

#include "local_api_server.h"

#include "../manifest/json.h"
#include "../package/package_loader.h"
#include "../runtime_host/runtime_engine.h"

#include <windows.h>
#include <sddl.h>
#include <bcrypt.h>

#include <array>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace neuralstudio {
namespace {

constexpr wchar_t kPipeName[] = L"\\\\.\\pipe\\neuralstudio.v1";
constexpr std::uint32_t kMaximumMessageBytes = 1024u * 1024u;
constexpr DWORD kIoTimeoutMs = 30000;

class Handle {
 public:
  Handle() = default;
  explicit Handle(HANDLE value) : value_(value) {}
  ~Handle() { reset(); }
  Handle(const Handle&) = delete;
  Handle& operator=(const Handle&) = delete;
  HANDLE get() const noexcept { return value_; }
  explicit operator bool() const noexcept {
    return value_ != nullptr && value_ != INVALID_HANDLE_VALUE;
  }
  void reset(HANDLE value = nullptr) noexcept {
    if (*this) CloseHandle(value_);
    value_ = value;
  }

 private:
  HANDLE value_ = nullptr;
};

struct LocalMemory {
  HLOCAL value = nullptr;
  ~LocalMemory() { if (value != nullptr) LocalFree(value); }
};

std::vector<std::uint8_t> UserSidForToken(HANDLE token) {
  DWORD bytes = 0;
  (void)GetTokenInformation(token, TokenUser, nullptr, 0, &bytes);
  if (bytes == 0) return {};
  std::vector<std::uint8_t> storage(bytes);
  if (!GetTokenInformation(token, TokenUser, storage.data(), bytes, &bytes)) {
    return {};
  }
  return storage;
}

bool SameInteractiveUser(HANDLE pipe) {
  ULONG client_pid = 0;
  if (!GetNamedPipeClientProcessId(pipe, &client_pid) || client_pid == 0) {
    return false;
  }
  Handle client(OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE,
                            client_pid));
  if (!client) return false;
  HANDLE raw_client_token = nullptr;
  HANDLE raw_server_token = nullptr;
  if (!OpenProcessToken(client.get(), TOKEN_QUERY, &raw_client_token) ||
      !OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &raw_server_token)) {
    if (raw_client_token != nullptr) CloseHandle(raw_client_token);
    if (raw_server_token != nullptr) CloseHandle(raw_server_token);
    return false;
  }
  Handle client_token(raw_client_token);
  Handle server_token(raw_server_token);
  const auto client_sid = UserSidForToken(client_token.get());
  const auto server_sid = UserSidForToken(server_token.get());
  if (client_sid.empty() || server_sid.empty()) return false;
  const auto* client_user = reinterpret_cast<const TOKEN_USER*>(client_sid.data());
  const auto* server_user = reinterpret_cast<const TOKEN_USER*>(server_sid.data());
  return EqualSid(client_user->User.Sid, server_user->User.Sid) != FALSE;
}

bool MakePipeSecurity(SECURITY_ATTRIBUTES& attributes, LocalMemory& owner) {
  HANDLE raw_token = nullptr;
  if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &raw_token)) return false;
  Handle token(raw_token);
  const auto storage = UserSidForToken(token.get());
  if (storage.empty()) return false;
  const auto* user = reinterpret_cast<const TOKEN_USER*>(storage.data());
  LPWSTR sid_text = nullptr;
  if (!ConvertSidToStringSidW(user->User.Sid, &sid_text)) return false;
  LocalMemory sid_owner{sid_text};
  const std::wstring sddl =
      L"D:P(A;;GA;;;SY)(A;;GA;;;" + std::wstring(sid_text) + L")";
  PSECURITY_DESCRIPTOR descriptor = nullptr;
  if (!ConvertStringSecurityDescriptorToSecurityDescriptorW(
          sddl.c_str(), SDDL_REVISION_1, &descriptor, nullptr)) {
    return false;
  }
  owner.value = descriptor;
  attributes = {sizeof(attributes), descriptor, FALSE};
  return true;
}

bool Transfer(HANDLE pipe, void* buffer, std::uint32_t bytes, bool write) {
  std::uint8_t* cursor = static_cast<std::uint8_t*>(buffer);
  std::uint32_t remaining = bytes;
  while (remaining != 0) {
    Handle event(CreateEventW(nullptr, TRUE, FALSE, nullptr));
    if (!event) return false;
    OVERLAPPED overlapped{};
    overlapped.hEvent = event.get();
    DWORD completed = 0;
    const BOOL immediate = write
        ? WriteFile(pipe, cursor, remaining, &completed, &overlapped)
        : ReadFile(pipe, cursor, remaining, &completed, &overlapped);
    if (!immediate) {
      if (GetLastError() != ERROR_IO_PENDING ||
          WaitForSingleObject(event.get(), kIoTimeoutMs) != WAIT_OBJECT_0 ||
          !GetOverlappedResult(pipe, &overlapped, &completed, FALSE)) {
        (void)CancelIoEx(pipe, &overlapped);
        return false;
      }
    }
    if (completed == 0) return false;
    cursor += completed;
    remaining -= completed;
  }
  return true;
}

bool ReadMessage(HANDLE pipe, std::string& message) {
  std::array<std::uint8_t, 4> header{};
  if (!Transfer(pipe, header.data(), static_cast<std::uint32_t>(header.size()),
                false)) return false;
  const std::uint32_t length =
      static_cast<std::uint32_t>(header[0]) |
      (static_cast<std::uint32_t>(header[1]) << 8) |
      (static_cast<std::uint32_t>(header[2]) << 16) |
      (static_cast<std::uint32_t>(header[3]) << 24);
  if (length == 0 || length > kMaximumMessageBytes) return false;
  message.assign(length, '\0');
  return Transfer(pipe, message.data(), length, false);
}

bool WriteMessage(HANDLE pipe, const std::string& message) {
  if (message.empty() || message.size() > kMaximumMessageBytes) return false;
  const std::uint32_t length = static_cast<std::uint32_t>(message.size());
  std::array<std::uint8_t, 4> header{
      static_cast<std::uint8_t>(length & 0xffu),
      static_cast<std::uint8_t>((length >> 8) & 0xffu),
      static_cast<std::uint8_t>((length >> 16) & 0xffu),
      static_cast<std::uint8_t>((length >> 24) & 0xffu)};
  return Transfer(pipe, header.data(), static_cast<std::uint32_t>(header.size()),
                  true) &&
         Transfer(pipe, const_cast<char*>(message.data()), length, true);
}

std::string Id(const json::Value& request) {
  const json::Value& id = request["id"];
  if (id.is_string()) return "\"" + JsonEscape(id.AsString()) + "\"";
  if (id.is_number()) {
    std::ostringstream value;
    value << std::setprecision(17) << id.AsNumber();
    return value.str();
  }
  return "null";
}

std::string Result(const std::string& id, const std::string& value) {
  return "{\"jsonrpc\":\"2.0\",\"id\":" + id +
         ",\"result\":" + value + "}";
}

std::string Error(const std::string& id, int code, const std::string& message) {
  return "{\"jsonrpc\":\"2.0\",\"id\":" + id +
         ",\"error\":{\"code\":" + std::to_string(code) +
         ",\"message\":\"" + JsonEscape(message) + "\"}}";
}

std::uint32_t Uint(const json::Value& value, std::uint32_t fallback) {
  const double number = value.AsNumber(static_cast<double>(fallback));
  if (number < 0.0 || number > 4294967295.0) return fallback;
  return static_cast<std::uint32_t>(number);
}

bool TestModeEnabled() {
  char value[8]{};
  return GetEnvironmentVariableA("NEURALSTUDIO_TEST_MODE", value,
                                 static_cast<DWORD>(sizeof(value))) != 0 &&
         std::strcmp(value, "1") == 0;
}

struct ConnectionState {
  bool handshake = false;
  bool source_open = false;
  bool pipeline_created = false;
  bool pipeline_running = false;
  bool shutdown = false;
  DiagnosticOptions diagnostic;
  DiagnosticResult last;
  std::string session_id;
};

std::string RandomSessionId() {
  std::array<std::uint8_t, 16> bytes{};
  if (BCryptGenRandom(nullptr, bytes.data(),
                      static_cast<ULONG>(bytes.size()),
                      BCRYPT_USE_SYSTEM_PREFERRED_RNG) < 0) {
    return {};
  }
  std::ostringstream text;
  text << std::hex << std::setfill('0');
  for (const std::uint8_t byte : bytes) {
    text << std::setw(2) << static_cast<unsigned>(byte);
  }
  return text.str();
}

std::string HandleRequest(const json::Value& request,
                          const LocalApiOptions& server,
                          ConnectionState& connection) {
  const std::string id = Id(request);
  if (!request.is_object() || request["jsonrpc"].AsString() != "2.0" ||
      !request["method"].is_string()) {
    return Error(id, -32600, "Invalid JSON-RPC request");
  }
  const std::string method = request["method"].AsString();
  const json::Value& params = request["params"];
  if (method == "rpc.handshake") {
    const std::string version = params["apiVersion"].AsString();
    if (version != "1.0") {
      return Error(id, -32001, "Unsupported API major version");
    }
    connection.handshake = true;
    connection.session_id = RandomSessionId();
    if (connection.session_id.empty()) {
      connection.handshake = false;
      return Error(id, -32603, "Unable to create a cryptographic session id");
    }
    return Result(id,
        "{\"apiVersion\":\"1.0\",\"runtimeVersion\":\"0.1.0\","
        "\"capabilities\":[\"d3d12\",\"isolated-worker\","
        "\"diagnostic-source\"],\"sessionId\":\"" +
        connection.session_id + "\"}");
  }
  if (!connection.handshake) {
    return Error(id, -32000, "rpc.handshake is required first");
  }
  if (method == "runtime.getInfo") {
    return Result(id,
        std::string("{\"runtimeVersion\":\"0.1.0\",\"pipe\":") +
        "\"neuralstudio.v1\",\"inferenceArmed\":" +
        (server.inference_armed ? "true" : "false") + "}");
  }
  if (method == "runtime.getHealth") {
    return Result(id, DiagnosticResultJson(connection.last));
  }
  if (method == "packages.validate") {
    std::filesystem::path path = server.default_package;
    if (params["path"].is_string()) path = params["path"].AsString();
    VerifiedPackage package;
    std::string error;
    if (!VerifyPackage(path, package, error)) return Error(id, -32010, error);
    return Result(id, "{\"valid\":true,\"packageId\":\"" +
        JsonEscape(package.manifest.summary.package_id) +
        "\",\"rulesPassed\":" + std::to_string(package.rules.size()) + "}");
  }
  if (method == "sources.open") {
    if (params["kind"].AsString() != "diagnostic") {
      return Error(id, -32021, "Runtime v0 exposes only the diagnostic source");
    }
    connection.source_open = true;
    return Result(id, "{\"sourceId\":\"diagnostic-1\","
                      "\"guides\":[\"color\",\"depth\",\"motionVectors\"]}");
  }
  if (method == "pipelines.create") {
    if (!connection.source_open) return Error(id, -32021, "Open a source first");
    connection.diagnostic.package_root = server.default_package;
    connection.diagnostic.output_path = server.default_output;
    if (params["packagePath"].is_string())
      connection.diagnostic.package_root = params["packagePath"].AsString();
    if (params["outputPath"].is_string())
      connection.diagnostic.output_path = params["outputPath"].AsString();
    connection.diagnostic.width = Uint(params["width"], 640);
    connection.diagnostic.height = Uint(params["height"], 360);
    connection.diagnostic.evaluation_frames = Uint(params["frames"], 1);
    connection.diagnostic.inference_armed = server.inference_armed;
    connection.diagnostic.test_failure = DiagnosticTestFailure::None;
    const std::string test_failure = params["_testFailure"].AsString();
    if (!test_failure.empty()) {
      if (!TestModeEnabled()) {
        return Error(id, -32602, "Test fault injection is disabled");
      }
      if (test_failure == "crash")
        connection.diagnostic.test_failure = DiagnosticTestFailure::Crash;
      else if (test_failure == "timeout")
        connection.diagnostic.test_failure = DiagnosticTestFailure::Timeout;
      else if (test_failure == "device-loss")
        connection.diagnostic.test_failure = DiagnosticTestFailure::DeviceLoss;
      else
        return Error(id, -32602, "Unknown test failure mode");
      if (connection.diagnostic.evaluation_frames < 2)
        connection.diagnostic.evaluation_frames = 2;
    }
    connection.pipeline_created = true;
    return Result(id, "{\"pipelineId\":\"pipeline-1\",\"state\":\"Standby\"}");
  }
  if (method == "pipelines.start") {
    if (!connection.pipeline_created) return Error(id, -32022, "Create a pipeline first");
    if (!server.inference_armed) return Error(id, -32023, "Inference is disarmed at process start");
    connection.pipeline_running = true;
    connection.last = RunDiagnostic(connection.diagnostic);
    connection.pipeline_running = false;
    if (!connection.last.ok) return Error(id, -32024, connection.last.detail);
    return Result(id, DiagnosticResultJson(connection.last));
  }
  if (method == "pipelines.getState") {
    return Result(id, DiagnosticResultJson(connection.last));
  }
  if (method == "pipelines.stop") {
    connection.pipeline_running = false;
    connection.pipeline_created = false;
    return Result(id, "{\"state\":\"Standby\"}");
  }
  if (method == "runtime.shutdown") {
    if (connection.pipeline_running && !params["force"].AsBool(false)) {
      return Error(id, -32025, "An active pipeline must be stopped first");
    }
    connection.shutdown = true;
    return Result(id, "{\"shuttingDown\":true}");
  }
  return Error(id, -32601, "Method not found");
}

bool Connect(HANDLE pipe) {
  Handle event(CreateEventW(nullptr, TRUE, FALSE, nullptr));
  if (!event) return false;
  OVERLAPPED overlapped{};
  overlapped.hEvent = event.get();
  if (ConnectNamedPipe(pipe, &overlapped)) return true;
  const DWORD error = GetLastError();
  if (error == ERROR_PIPE_CONNECTED) return true;
  if (error != ERROR_IO_PENDING) return false;
  if (WaitForSingleObject(event.get(), INFINITE) != WAIT_OBJECT_0) return false;
  DWORD ignored = 0;
  return GetOverlappedResult(pipe, &overlapped, &ignored, FALSE) != FALSE;
}

}  // namespace

int RunLocalApiServer(const LocalApiOptions& options) {
  SECURITY_ATTRIBUTES attributes{};
  LocalMemory descriptor;
  if (!MakePipeSecurity(attributes, descriptor)) {
    std::fprintf(stderr, "Unable to create current-user pipe ACL\n");
    return 1;
  }
  bool shutdown = false;
  while (!shutdown) {
    Handle pipe(CreateNamedPipeW(
        kPipeName, PIPE_ACCESS_DUPLEX | FILE_FLAG_OVERLAPPED |
                       FILE_FLAG_FIRST_PIPE_INSTANCE,
        PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT |
            PIPE_REJECT_REMOTE_CLIENTS,
        1, kMaximumMessageBytes, kMaximumMessageBytes, 0, &attributes));
    if (!pipe) {
      std::fprintf(stderr, "CreateNamedPipe failed: %lu\n",
                   static_cast<unsigned long>(GetLastError()));
      return 1;
    }
    if (!Connect(pipe.get()) || !SameInteractiveUser(pipe.get())) {
      (void)DisconnectNamedPipe(pipe.get());
      continue;
    }
    ConnectionState connection;
    while (!connection.shutdown) {
      std::string message;
      if (!ReadMessage(pipe.get(), message)) break;
      json::Value request;
      json::ParseError parse;
      std::string response;
      if (!json::Parse(message, request, parse)) {
        response = Error("null", -32700, "Parse error");
      } else if (request.is_array()) {
        response = Error("null", -32600, "Batch requests are not implemented in Runtime v0");
      } else {
        response = HandleRequest(request, options, connection);
      }
      if (!WriteMessage(pipe.get(), response)) break;
    }
    shutdown = connection.shutdown;
    FlushFileBuffers(pipe.get());
    (void)DisconnectNamedPipe(pipe.get());
  }
  return 0;
}

}  // namespace neuralstudio
