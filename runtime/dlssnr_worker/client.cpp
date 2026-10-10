/* NeuralStudio DLSSNR isolated-worker client - SPDX-License-Identifier: MIT */

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX

#include "client.h"

#include <windows.h>
#include <sddl.h>
#include <bcrypt.h>
#include <wrl/client.h>

#include <array>
#include <chrono>
#include <cstring>
#include <utility>
#include <vector>

#pragma comment(lib, "bcrypt.lib")

using Microsoft::WRL::ComPtr;

namespace neuralstudio::dlssnr_worker {
namespace {

constexpr DWORD kDefaultControlTimeoutMs = 15000;

class Handle {
 public:
  Handle() = default;
  explicit Handle(HANDLE value) : value_(value) {}
  ~Handle() { reset(); }
  Handle(const Handle&) = delete;
  Handle& operator=(const Handle&) = delete;
  Handle(Handle&& other) noexcept : value_(other.release()) {}
  Handle& operator=(Handle&& other) noexcept {
    if (this != &other) reset(other.release());
    return *this;
  }
  [[nodiscard]] HANDLE get() const noexcept { return value_; }
  [[nodiscard]] explicit operator bool() const noexcept {
    return value_ != nullptr && value_ != INVALID_HANDLE_VALUE;
  }
  HANDLE* address() noexcept {
    reset();
    return &value_;
  }
  HANDLE release() noexcept {
    HANDLE result = value_;
    value_ = nullptr;
    return result;
  }
  void reset(HANDLE value = nullptr) noexcept {
    if (*this) CloseHandle(value_);
    value_ = value;
  }

 private:
  HANDLE value_ = nullptr;
};

class LocalMemory {
 public:
  ~LocalMemory() { if (value_ != nullptr) LocalFree(value_); }
  void** address() noexcept { return &value_; }
  void* get() const noexcept { return value_; }

 private:
  void* value_ = nullptr;
};

std::wstring CurrentUserSid() {
  HANDLE raw = nullptr;
  if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &raw)) return {};
  Handle token(raw);
  DWORD size = 0;
  GetTokenInformation(token.get(), TokenUser, nullptr, 0, &size);
  if (GetLastError() != ERROR_INSUFFICIENT_BUFFER || size == 0) return {};
  std::vector<std::uint8_t> storage(size);
  if (!GetTokenInformation(token.get(), TokenUser, storage.data(), size,
                           &size)) return {};
  const auto* user = reinterpret_cast<const TOKEN_USER*>(storage.data());
  wchar_t* sid = nullptr;
  if (!ConvertSidToStringSidW(user->User.Sid, &sid)) return {};
  std::wstring result(sid);
  LocalFree(sid);
  return result;
}

struct PipeSecurity {
  LocalMemory descriptor;
  SECURITY_ATTRIBUTES attributes{};
  bool Initialize() {
    const std::wstring sid = CurrentUserSid();
    if (sid.empty()) return false;
    const std::wstring sddl = L"D:P(A;;GA;;;" + sid + L")";
    if (!ConvertStringSecurityDescriptorToSecurityDescriptorW(
            sddl.c_str(), SDDL_REVISION_1, descriptor.address(), nullptr)) {
      return false;
    }
    attributes.nLength = sizeof(attributes);
    attributes.lpSecurityDescriptor = descriptor.get();
    attributes.bInheritHandle = FALSE;
    return true;
  }
};

enum class TransferResult { complete, disconnected, timeout, failed };

TransferResult Transfer(HANDLE pipe, bool write, void* buffer, DWORD size,
                        DWORD timeout_ms) {
  Handle event(CreateEventW(nullptr, TRUE, FALSE, nullptr));
  if (!event) return TransferResult::failed;
  OVERLAPPED overlap{};
  overlap.hEvent = event.get();
  DWORD transferred = 0;
  const BOOL immediate = write
      ? WriteFile(pipe, buffer, size, &transferred, &overlap)
      : ReadFile(pipe, buffer, size, &transferred, &overlap);
  if (!immediate) {
    const DWORD code = GetLastError();
    if (code == ERROR_BROKEN_PIPE || code == ERROR_NO_DATA) {
      return TransferResult::disconnected;
    }
    if (code != ERROR_IO_PENDING) return TransferResult::failed;
    const DWORD waited = WaitForSingleObject(event.get(), timeout_ms);
    if (waited == WAIT_TIMEOUT) {
      CancelIoEx(pipe, &overlap);
      WaitForSingleObject(event.get(), INFINITE);
      return TransferResult::timeout;
    }
    if (waited != WAIT_OBJECT_0 ||
        !GetOverlappedResult(pipe, &overlap, &transferred, FALSE)) {
      const DWORD completion = GetLastError();
      return completion == ERROR_BROKEN_PIPE || completion == ERROR_NO_DATA ||
                     completion == ERROR_OPERATION_ABORTED
                 ? TransferResult::disconnected
                 : TransferResult::failed;
    }
  }
  return transferred == size ? TransferResult::complete
                             : TransferResult::failed;
}

std::uint64_t RandomNonce() {
  std::uint64_t value = 0;
  if (BCryptGenRandom(nullptr, reinterpret_cast<PUCHAR>(&value), sizeof(value),
                      BCRYPT_USE_SYSTEM_PREFERRED_RNG) < 0 || value == 0) {
    value = (static_cast<std::uint64_t>(GetTickCount64()) << 17) ^
            (static_cast<std::uint64_t>(GetCurrentProcessId()) << 3) ^
            reinterpret_cast<std::uintptr_t>(&value);
  }
  return value == 0 ? 1 : value;
}

std::uint64_t LuidValue(LUID luid) {
  std::uint64_t value = 0;
  static_assert(sizeof(value) == sizeof(luid));
  std::memcpy(&value, &luid, sizeof(value));
  return value;
}

}  // namespace

class Client::Impl {
 public:
  ~Impl() {
    try {
      std::string ignored;
      (void)Shutdown(1000, ignored);
    } catch (...) {
      Abort();
    }
  }

  bool Start(const std::filesystem::path& worker_binary,
             const std::filesystem::path& runtime_directory,
             std::uint32_t generation, std::string& error) {
    Abort();
    if (!std::filesystem::is_regular_file(worker_binary) ||
        !std::filesystem::is_regular_file(runtime_directory /
                                          L"sl.interposer.dll")) {
      error = "Worker binary or runtime directory is incomplete";
      return false;
    }

    nonce_ = RandomNonce() ^ generation;
    if (nonce_ == 0) nonce_ = 1;
    sequence_ = 1;
    supervisor_id_ = GetCurrentProcessId();
    const std::wstring pipe_name =
        L"\\\\.\\pipe\\NeuralStudio.DlssNrWorker." +
        std::to_wstring(supervisor_id_) + L"." + std::to_wstring(nonce_);
    PipeSecurity security;
    if (!security.Initialize()) {
      error = "Unable to create a current-user-only pipe ACL";
      return false;
    }
    pipe_.reset(CreateNamedPipeW(
        pipe_name.c_str(), PIPE_ACCESS_DUPLEX | FILE_FLAG_OVERLAPPED,
        PIPE_TYPE_MESSAGE | PIPE_READMODE_MESSAGE | PIPE_WAIT |
            PIPE_REJECT_REMOTE_CLIENTS,
        1, protocol::kMaximumPacketBytes * 2,
        protocol::kMaximumPacketBytes * 2, kDefaultControlTimeoutMs,
        &security.attributes));
    if (!pipe_) {
      error = "CreateNamedPipe failed";
      return false;
    }

    Handle connected(CreateEventW(nullptr, TRUE, FALSE, nullptr));
    if (!connected) {
      error = "CreateEvent(connect) failed";
      Abort();
      return false;
    }
    OVERLAPPED overlap{};
    overlap.hEvent = connected.get();
    const BOOL immediate = ConnectNamedPipe(pipe_.get(), &overlap);
    const DWORD connect_error = immediate ? ERROR_SUCCESS : GetLastError();
    if (!immediate && connect_error != ERROR_IO_PENDING &&
        connect_error != ERROR_PIPE_CONNECTED) {
      error = "ConnectNamedPipe failed";
      Abort();
      return false;
    }

    job_.reset(CreateJobObjectW(nullptr, nullptr));
    if (!job_) {
      error = "CreateJobObject failed";
      Abort();
      return false;
    }
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION job_limits{};
    job_limits.BasicLimitInformation.LimitFlags =
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_ACTIVE_PROCESS;
    job_limits.BasicLimitInformation.ActiveProcessLimit = 1;
    if (!SetInformationJobObject(job_.get(), JobObjectExtendedLimitInformation,
                                 &job_limits, sizeof(job_limits))) {
      error = "SetInformationJobObject failed";
      Abort();
      return false;
    }

    std::wstring command = L"\"" + worker_binary.wstring() +
                           L"\" --pipe \"" + pipe_name +
                           L"\" --streamline-dir \"" +
                           runtime_directory.wstring() + L"\"";
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION created{};
    if (!CreateProcessW(worker_binary.c_str(), command.data(), nullptr, nullptr,
                        FALSE, CREATE_NO_WINDOW | CREATE_SUSPENDED, nullptr,
                        runtime_directory.c_str(), &startup, &created)) {
      error = "CreateProcess(worker) failed";
      Abort();
      return false;
    }
    process_.reset(created.hProcess);
    thread_.reset(created.hThread);
    process_id_ = created.dwProcessId;
    if (!AssignProcessToJobObject(job_.get(), process_.get()) ||
        ResumeThread(thread_.get()) == static_cast<DWORD>(-1)) {
      error = "Unable to contain and resume the worker";
      Abort();
      return false;
    }

    if (connect_error == ERROR_IO_PENDING) {
      if (WaitForSingleObject(connected.get(), kDefaultControlTimeoutMs) !=
          WAIT_OBJECT_0) {
        error = "Worker connection timed out";
        Abort();
        return false;
      }
      DWORD transferred = 0;
      if (!GetOverlappedResult(pipe_.get(), &overlap, &transferred, FALSE) &&
          GetLastError() != ERROR_PIPE_CONNECTED) {
        error = "Worker connection failed";
        Abort();
        return false;
      }
    }

    ULONG client_id = 0;
    if (!GetNamedPipeClientProcessId(pipe_.get(), &client_id) ||
        client_id != process_id_) {
      error = "Worker pipe identity mismatch";
      Abort();
      return false;
    }
    protocol::Packet hello = Request(protocol::MessageType::hello);
    protocol::Packet reply{};
    if (!RoundTrip(hello, reply, kDefaultControlTimeoutMs, error) ||
        reply.type != protocol::MessageType::hello_ack ||
        reply.status != protocol::Status::ok) {
      if (error.empty()) error = "Worker handshake was rejected";
      Abort();
      return false;
    }
    return true;
  }

  bool Configure(ID3D12Device* device, std::uint32_t width,
                 std::uint32_t height,
                 const std::array<HANDLE, 4>& shared_resources,
                 const protocol::Options& options, std::string& error) {
    if (!alive() || device == nullptr || width < 64 || height < 64 ||
        !protocol::ValidOptions(options)) {
      error = "Invalid worker configuration";
      return false;
    }
    for (HANDLE handle : shared_resources) {
      if (handle == nullptr || handle == INVALID_HANDLE_VALUE) {
        error = "A shared resource handle is missing";
        return false;
      }
    }
    fence_.Reset();
    shared_fence_.reset();
    if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_SHARED,
                                   IID_PPV_ARGS(&fence_))) ||
        FAILED(device->CreateSharedHandle(fence_.Get(), nullptr,
                                          GENERIC_ALL, nullptr,
                                          shared_fence_.address()))) {
      error = "Unable to create the worker timeline fence";
      return false;
    }

    protocol::Packet request = Request(protocol::MessageType::configure);
    request.width = width;
    request.height = height;
    request.adapter_luid = LuidValue(device->GetAdapterLuid());
    request.options = options;
    std::array<HANDLE, protocol::kSharedHandleCount> local_handles{};
    local_handles[0] = shared_fence_.get();
    for (std::uint32_t slot = 0; slot < protocol::kSlotCount; ++slot) {
      for (std::uint32_t resource = 0; resource < 4; ++resource) {
        local_handles[protocol::ResourceHandleIndex(slot, resource)] =
            shared_resources[resource];
      }
    }
    for (std::size_t index = 0; index < local_handles.size(); ++index) {
      HANDLE remote = nullptr;
      if (!DuplicateHandle(GetCurrentProcess(), local_handles[index],
                           process_.get(), &remote, 0, FALSE,
                           DUPLICATE_SAME_ACCESS)) {
        error = "Duplicating a shared handle into the worker failed";
        Abort();
        return false;
      }
      request.shared_handles[index] =
          reinterpret_cast<std::uint64_t>(remote);
    }
    protocol::Packet reply{};
    if (!RoundTrip(request, reply, kDefaultControlTimeoutMs, error) ||
        reply.type != protocol::MessageType::configured ||
        reply.status != protocol::Status::ok) {
      if (error.empty()) error = "Worker rejected shared resources";
      Abort();
      return false;
    }
    configured_ = true;
    width_ = width;
    height_ = height;
    last_retired_ = 0;
    options_ = options;
    return true;
  }

  Evaluation Evaluate(std::uint64_t frame_id, std::uint32_t slot, bool reset,
                      const protocol::FrameConstants& constants,
                      std::uint32_t timeout_ms, std::string& error) {
    Evaluation result;
    result.worker_alive = alive();
    if (!configured_ || !result.worker_alive || frame_id == 0 ||
        slot >= protocol::kSlotCount || fence_ == nullptr ||
        !protocol::ValidConstants(constants)) {
      error = "Worker is not configured for evaluation";
      result.status = protocol::Status::invalid_state;
      return result;
    }
    protocol::Packet request = Request(protocol::MessageType::evaluate);
    request.width = width_;
    request.height = height_;
    request.slot = slot;
    request.frame_id = frame_id;
    request.reset = reset ? 1u : 0u;
    request.options = options_;
    request.constants = constants;
    request.input_ready = last_retired_ + 1;
    request.model_complete = last_retired_ + 2;
    request.copyback_complete = last_retired_ + 3;
    request.retired = last_retired_ + 4;
    if (FAILED(fence_->Signal(request.input_ready))) {
      error = "Signaling input readiness failed";
      result.status = protocol::Status::device_lost;
      Abort();
      return result;
    }
    protocol::Packet reply{};
    if (!RoundTrip(request, reply, timeout_ms, error)) {
      result.status = protocol::Status::gpu_timeout;
      Abort();
      return result;
    }
    result.status = reply.status;
    result.feature_creations = reply.feature_creations;
    result.successful_evaluations = reply.successful_evaluations;
    result.debug_severe_messages = reply.debug_severe_messages;
    if (reply.type != protocol::MessageType::evaluated ||
        reply.status != protocol::Status::ok) {
      if (error.empty()) error = "Worker evaluation failed";
      if (reply.status == protocol::Status::gpu_timeout ||
          reply.status == protocol::Status::device_lost) Abort();
      result.worker_alive = alive();
      return result;
    }
    if (FAILED(fence_->Signal(request.copyback_complete)) ||
        FAILED(fence_->Signal(request.retired))) {
      error = "Retiring the worker timeline failed";
      result.status = protocol::Status::device_lost;
      Abort();
      return result;
    }
    last_retired_ = request.retired;
    result.worker_alive = alive();
    return result;
  }

  bool SetOptions(const protocol::Options& options, std::uint32_t timeout_ms,
                  std::string& error) {
    if (!configured_ || !protocol::ValidOptions(options)) {
      error = "Worker options are invalid or no session is configured";
      return false;
    }
    protocol::Packet request = Request(protocol::MessageType::set_options);
    request.options = options;
    protocol::Packet reply{};
    if (!RoundTrip(request, reply, timeout_ms, error) ||
        reply.type != protocol::MessageType::options_set ||
        reply.status != protocol::Status::ok) {
      if (error.empty()) error = "Worker rejected options";
      return false;
    }
    options_ = options;
    return true;
  }

  bool Release(std::uint32_t timeout_ms, std::string& error) {
    if (!alive()) {
      configured_ = false;
      fence_.Reset();
      shared_fence_.reset();
      return true;
    }
    protocol::Packet request = Request(protocol::MessageType::release);
    protocol::Packet reply{};
    const bool ok = RoundTrip(request, reply, timeout_ms, error) &&
                    reply.type == protocol::MessageType::released &&
                    reply.status == protocol::Status::ok;
    configured_ = false;
    fence_.Reset();
    shared_fence_.reset();
    return ok;
  }

  bool InjectFailure(FailureInjection failure, std::uint32_t timeout_ms,
                     std::string& error) {
    if (!alive()) {
      error = "Worker is not alive";
      return false;
    }
    protocol::MessageType type = protocol::MessageType::inject_crash;
    if (failure == FailureInjection::timeout) {
      type = protocol::MessageType::inject_timeout;
    } else if (failure == FailureInjection::device_loss) {
      type = protocol::MessageType::inject_device_loss;
    }
    protocol::Packet request = Request(type);
    if (Transfer(pipe_.get(), true, &request, sizeof(request), timeout_ms) !=
        TransferResult::complete) {
      error = "Sending failure injection failed";
      Abort();
      return false;
    }
    protocol::Packet reply{};
    const TransferResult transfer =
        Transfer(pipe_.get(), false, &reply, sizeof(reply), timeout_ms);
    const bool expected =
        (failure == FailureInjection::crash &&
         transfer == TransferResult::disconnected) ||
        (failure == FailureInjection::timeout &&
         transfer == TransferResult::timeout) ||
        (failure == FailureInjection::device_loss &&
         transfer == TransferResult::complete &&
         protocol::Valid(reply) &&
         reply.status == protocol::Status::device_lost);
    Abort();
    if (!expected) error = "Failure injection did not produce the expected isolation event";
    return expected;
  }

  bool Shutdown(std::uint32_t timeout_ms, std::string& error) {
    bool clean = true;
    if (alive()) {
      protocol::Packet request = Request(protocol::MessageType::shutdown);
      protocol::Packet reply{};
      clean = RoundTrip(request, reply, timeout_ms, error) &&
              reply.type == protocol::MessageType::shutdown_ack &&
              reply.status == protocol::Status::ok &&
              WaitForSingleObject(process_.get(), timeout_ms) == WAIT_OBJECT_0;
    }
    if (!clean) Abort();
    ResetHandles();
    return clean;
  }

  void Abort() noexcept {
    if (process_ && WaitForSingleObject(process_.get(), 0) == WAIT_TIMEOUT) {
      if (job_) {
        TerminateJobObject(job_.get(), 0xE121u);
      } else {
        TerminateProcess(process_.get(), 0xE121u);
      }
      WaitForSingleObject(process_.get(), 5000);
    }
    ResetHandles();
  }

  [[nodiscard]] bool alive() const noexcept {
    return process_ && WaitForSingleObject(process_.get(), 0) == WAIT_TIMEOUT;
  }
  [[nodiscard]] bool configured() const noexcept { return configured_; }
  [[nodiscard]] std::uint32_t process_id() const noexcept { return process_id_; }

 private:
  protocol::Packet Request(protocol::MessageType type) {
    protocol::Packet packet;
    packet.type = type;
    packet.nonce = nonce_;
    packet.sequence = sequence_++;
    packet.process_id = supervisor_id_;
    return packet;
  }

  bool RoundTrip(protocol::Packet& request, protocol::Packet& reply,
                 std::uint32_t timeout_ms, std::string& error) {
    const TransferResult written =
        Transfer(pipe_.get(), true, &request, sizeof(request), timeout_ms);
    if (written != TransferResult::complete) {
      error = written == TransferResult::timeout ? "Worker write timed out"
                                                 : "Worker pipe disconnected";
      return false;
    }
    const TransferResult read =
        Transfer(pipe_.get(), false, &reply, sizeof(reply), timeout_ms);
    if (read != TransferResult::complete) {
      error = read == TransferResult::timeout ? "Worker response timed out"
                                              : "Worker pipe disconnected";
      return false;
    }
    if (!protocol::Valid(reply) || reply.nonce != nonce_ ||
        reply.sequence != request.sequence ||
        reply.process_id != process_id_) {
      error = "Worker response identity mismatch";
      return false;
    }
    return true;
  }

  void ResetHandles() noexcept {
    configured_ = false;
    width_ = 0;
    height_ = 0;
    last_retired_ = 0;
    fence_.Reset();
    shared_fence_.reset();
    pipe_.reset();
    thread_.reset();
    process_.reset();
    job_.reset();
    process_id_ = 0;
  }

  Handle job_;
  Handle process_;
  Handle thread_;
  Handle pipe_;
  Handle shared_fence_;
  ComPtr<ID3D12Fence> fence_;
  std::uint64_t nonce_ = 0;
  std::uint64_t sequence_ = 1;
  std::uint64_t last_retired_ = 0;
  std::uint32_t supervisor_id_ = 0;
  std::uint32_t process_id_ = 0;
  std::uint32_t width_ = 0;
  std::uint32_t height_ = 0;
  protocol::Options options_{};
  bool configured_ = false;
};

Client::Client() : impl_(std::make_unique<Impl>()) {}
Client::~Client() = default;
Client::Client(Client&&) noexcept = default;
Client& Client::operator=(Client&&) noexcept = default;

bool Client::Start(const std::filesystem::path& worker_binary,
                   const std::filesystem::path& runtime_directory,
                   std::uint32_t generation, std::string& error) {
  return impl_->Start(worker_binary, runtime_directory, generation, error);
}
bool Client::Configure(ID3D12Device* source_device, std::uint32_t width,
                       std::uint32_t height,
                       const std::array<HANDLE, 4>& shared_resources,
                       const protocol::Options& options, std::string& error) {
  return impl_->Configure(source_device, width, height, shared_resources,
                          options, error);
}
Evaluation Client::Evaluate(std::uint64_t frame_id, std::uint32_t slot,
                            bool reset,
                            const protocol::FrameConstants& constants,
                            std::uint32_t timeout_ms,
                            std::string& error) {
  return impl_->Evaluate(frame_id, slot, reset, constants, timeout_ms, error);
}
bool Client::SetOptions(const protocol::Options& options,
                        std::uint32_t timeout_ms, std::string& error) {
  return impl_->SetOptions(options, timeout_ms, error);
}
bool Client::Release(std::uint32_t timeout_ms, std::string& error) {
  return impl_->Release(timeout_ms, error);
}
bool Client::InjectFailure(FailureInjection failure, std::uint32_t timeout_ms,
                           std::string& error) {
  return impl_->InjectFailure(failure, timeout_ms, error);
}
bool Client::Shutdown(std::uint32_t timeout_ms, std::string& error) {
  return impl_->Shutdown(timeout_ms, error);
}
void Client::Abort() noexcept { impl_->Abort(); }
bool Client::alive() const noexcept { return impl_->alive(); }
bool Client::configured() const noexcept { return impl_->configured(); }
std::uint32_t Client::process_id() const noexcept { return impl_->process_id(); }

}  // namespace neuralstudio::dlssnr_worker
