/* NeuralStudio DLSSNR worker shared-resource probe - SPDX-License-Identifier: MIT */

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX

#include <windows.h>
#include <sddl.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include "bridge_core.h"
#include "dlssnr_worker_protocol.h"
#include "vulkan_external_transport.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using Microsoft::WRL::ComPtr;
namespace bridge = neuralstudio::vulkan_bridge;
namespace protocol = neuralstudio::vulkan_bridge::dlssnr_worker;

namespace {

constexpr DWORD kControlTimeoutMs = 15000;
constexpr DWORD kShortTimeoutMs = 250;
constexpr std::uint64_t kFnvOffset = 14695981039346656037ull;
constexpr std::uint64_t kFnvPrime = 1099511628211ull;

[[noreturn]] void Fail(std::string message) {
  throw std::runtime_error(std::move(message));
}

void RequireHr(HRESULT result, std::string_view operation) {
  if (FAILED(result)) {
    char code[16]{};
    std::snprintf(code, sizeof(code), "%08X",
                  static_cast<unsigned>(result));
    Fail(std::string(operation) + " failed HRESULT=0x" + code);
  }
}

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
  LocalMemory(const LocalMemory&) = delete;
  LocalMemory& operator=(const LocalMemory&) = delete;
  LocalMemory() = default;
  void** address() noexcept { return &value_; }
  void* get() const noexcept { return value_; }

 private:
  void* value_ = nullptr;
};

std::wstring CurrentUserSid() {
  HANDLE raw = nullptr;
  if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &raw)) {
    Fail("OpenProcessToken failed");
  }
  Handle token(raw);
  DWORD size = 0;
  GetTokenInformation(token.get(), TokenUser, nullptr, 0, &size);
  if (GetLastError() != ERROR_INSUFFICIENT_BUFFER || size == 0) {
    Fail("TokenUser size query failed");
  }
  std::vector<std::uint8_t> storage(size);
  if (!GetTokenInformation(token.get(), TokenUser, storage.data(), size,
                           &size)) {
    Fail("TokenUser query failed");
  }
  const auto* user = reinterpret_cast<const TOKEN_USER*>(storage.data());
  wchar_t* sid = nullptr;
  if (!ConvertSidToStringSidW(user->User.Sid, &sid)) {
    Fail("ConvertSidToStringSid failed");
  }
  std::wstring result(sid);
  LocalFree(sid);
  return result;
}

struct PipeSecurity {
  LocalMemory descriptor;
  SECURITY_ATTRIBUTES attributes{};
  PipeSecurity() {
    const std::wstring sddl = L"D:P(A;;GA;;;" + CurrentUserSid() + L")";
    if (!ConvertStringSecurityDescriptorToSecurityDescriptorW(
            sddl.c_str(), SDDL_REVISION_1, descriptor.address(), nullptr)) {
      Fail("Pipe security descriptor creation failed");
    }
    attributes.nLength = sizeof(attributes);
    attributes.lpSecurityDescriptor = descriptor.get();
    attributes.bInheritHandle = FALSE;
  }
};

enum class TransferResult { complete, disconnected, timeout };

TransferResult OverlappedTransfer(HANDLE pipe, bool write, void* buffer,
                                  DWORD size, DWORD timeout_ms) {
  Handle event(CreateEventW(nullptr, TRUE, FALSE, nullptr));
  if (!event) Fail("CreateEvent(I/O) failed");
  OVERLAPPED overlap{};
  overlap.hEvent = event.get();
  DWORD transferred = 0;
  const BOOL immediate = write
      ? WriteFile(pipe, buffer, size, &transferred, &overlap)
      : ReadFile(pipe, buffer, size, &transferred, &overlap);
  if (!immediate) {
    const DWORD error = GetLastError();
    if (error == ERROR_BROKEN_PIPE || error == ERROR_NO_DATA) {
      return TransferResult::disconnected;
    }
    if (error != ERROR_IO_PENDING) Fail("Named-pipe I/O failed");
    const DWORD waited = WaitForSingleObject(event.get(), timeout_ms);
    if (waited == WAIT_TIMEOUT) {
      CancelIoEx(pipe, &overlap);
      WaitForSingleObject(event.get(), INFINITE);
      return TransferResult::timeout;
    }
    if (waited != WAIT_OBJECT_0 ||
        !GetOverlappedResult(pipe, &overlap, &transferred, FALSE)) {
      const DWORD completion_error = GetLastError();
      if (completion_error == ERROR_BROKEN_PIPE ||
          completion_error == ERROR_NO_DATA ||
          completion_error == ERROR_OPERATION_ABORTED) {
        return TransferResult::disconnected;
      }
      Fail("Named-pipe I/O completion failed");
    }
  }
  if (transferred != size) Fail("Named-pipe packet was truncated");
  return TransferResult::complete;
}

struct Worker {
  Handle process;
  Handle thread;
  Handle pipe;
  std::uint64_t nonce = 0;
  std::uint64_t sequence = 1;
  DWORD process_id = 0;
  DWORD supervisor_id = GetCurrentProcessId();
};

protocol::Packet MakeRequest(Worker& worker, protocol::MessageType type) {
  protocol::Packet packet;
  packet.type = type;
  packet.nonce = worker.nonce;
  packet.sequence = worker.sequence++;
  packet.process_id = worker.supervisor_id;
  return packet;
}

void WriteRequest(Worker& worker, protocol::Packet& packet,
                  DWORD timeout_ms = kControlTimeoutMs) {
  if (OverlappedTransfer(worker.pipe.get(), true, &packet, sizeof(packet),
                         timeout_ms) != TransferResult::complete) {
    Fail("Writing worker request failed");
  }
}

TransferResult ReadResponse(Worker& worker, const protocol::Packet& request,
                            protocol::Packet& response,
                            DWORD timeout_ms = kControlTimeoutMs) {
  const TransferResult result = OverlappedTransfer(
      worker.pipe.get(), false, &response, sizeof(response), timeout_ms);
  if (result != TransferResult::complete) return result;
  if (!protocol::Valid(response) || response.nonce != worker.nonce ||
      response.sequence != request.sequence ||
      response.process_id != worker.process_id) {
    Fail("Worker response identity mismatch");
  }
  return result;
}

Worker StartWorker(const std::filesystem::path& binary,
                   const std::filesystem::path& runtime,
                   std::uint32_t generation) {
  Worker worker;
  worker.nonce = (static_cast<std::uint64_t>(GetTickCount64()) << 20) ^
                 (static_cast<std::uint64_t>(GetCurrentProcessId()) << 3) ^
                 generation;
  const std::wstring pipe_name =
      L"\\\\.\\pipe\\NeuralStudio.DlssNrWorker." +
      std::to_wstring(GetCurrentProcessId()) + L"." +
      std::to_wstring(worker.nonce);
  PipeSecurity security;
  HANDLE raw_pipe = CreateNamedPipeW(
      pipe_name.c_str(), PIPE_ACCESS_DUPLEX | FILE_FLAG_OVERLAPPED,
      PIPE_TYPE_MESSAGE | PIPE_READMODE_MESSAGE | PIPE_WAIT |
          PIPE_REJECT_REMOTE_CLIENTS,
      1, protocol::kMaximumPacketBytes * 2,
      protocol::kMaximumPacketBytes * 2, kControlTimeoutMs,
      &security.attributes);
  if (raw_pipe == INVALID_HANDLE_VALUE) Fail("CreateNamedPipe failed");
  worker.pipe.reset(raw_pipe);

  Handle connected(CreateEventW(nullptr, TRUE, FALSE, nullptr));
  if (!connected) Fail("CreateEvent(connect) failed");
  OVERLAPPED overlap{};
  overlap.hEvent = connected.get();
  const BOOL immediate = ConnectNamedPipe(worker.pipe.get(), &overlap);
  const DWORD connect_error = immediate ? ERROR_SUCCESS : GetLastError();
  if (!immediate && connect_error != ERROR_IO_PENDING &&
      connect_error != ERROR_PIPE_CONNECTED) {
    Fail("ConnectNamedPipe failed");
  }

  std::wstring command = L"\"" + binary.wstring() + L"\" --pipe \"" +
                         pipe_name + L"\" --streamline-dir \"" +
                         runtime.wstring() + L"\"";
  STARTUPINFOW startup{};
  startup.cb = sizeof(startup);
  PROCESS_INFORMATION process{};
  if (!CreateProcessW(binary.c_str(), command.data(), nullptr, nullptr, FALSE,
                      CREATE_NO_WINDOW, nullptr, runtime.c_str(), &startup,
                      &process)) {
    Fail("CreateProcess(worker) failed");
  }
  worker.process.reset(process.hProcess);
  worker.thread.reset(process.hThread);
  worker.process_id = process.dwProcessId;

  if (connect_error == ERROR_IO_PENDING) {
    if (WaitForSingleObject(connected.get(), kControlTimeoutMs) !=
        WAIT_OBJECT_0) {
      TerminateProcess(worker.process.get(), 0xE110u);
      Fail("Worker connection timed out");
    }
    DWORD transferred = 0;
    if (!GetOverlappedResult(worker.pipe.get(), &overlap, &transferred, FALSE) &&
        GetLastError() != ERROR_PIPE_CONNECTED) {
      Fail("Worker connection failed");
    }
  }

  ULONG pipe_client_id = 0;
  if (!GetNamedPipeClientProcessId(worker.pipe.get(), &pipe_client_id) ||
      pipe_client_id != worker.process_id) {
    TerminateProcess(worker.process.get(), 0xE112u);
    WaitForSingleObject(worker.process.get(), 5000);
    Fail("Named-pipe client identity mismatch");
  }

  auto hello = MakeRequest(worker, protocol::MessageType::hello);
  WriteRequest(worker, hello);
  protocol::Packet response;
  if (ReadResponse(worker, hello, response) != TransferResult::complete ||
      response.type != protocol::MessageType::hello_ack ||
      response.status != protocol::Status::ok) {
    Fail("Worker handshake failed");
  }
  return worker;
}

std::uint64_t DuplicateIntoWorker(const Worker& worker, HANDLE handle) {
  HANDLE remote = nullptr;
  if (!DuplicateHandle(GetCurrentProcess(), handle, worker.process.get(),
                       &remote, 0, FALSE, DUPLICATE_SAME_ACCESS)) {
    Fail("DuplicateHandle into worker failed");
  }
  return reinterpret_cast<std::uint64_t>(remote);
}

void StopWorker(Worker& worker) noexcept {
  if (!worker.process) return;
  if (WaitForSingleObject(worker.process.get(), 0) == WAIT_TIMEOUT) {
    TerminateProcess(worker.process.get(), 0xE111u);
    WaitForSingleObject(worker.process.get(), 5000);
  }
}

std::uint16_t FloatToHalf(float value) {
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  const std::uint32_t sign = (bits >> 16) & 0x8000u;
  int exponent = static_cast<int>((bits >> 23) & 0xffu) - 127 + 15;
  std::uint32_t mantissa = bits & 0x7fffffu;
  if (exponent <= 0) {
    if (exponent < -10) return static_cast<std::uint16_t>(sign);
    mantissa = (mantissa | 0x800000u) >> (1 - exponent);
    return static_cast<std::uint16_t>(sign | ((mantissa + 0x1000u) >> 13));
  }
  if (exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
  return static_cast<std::uint16_t>(
      sign | (static_cast<std::uint32_t>(exponent) << 10) |
      ((mantissa + 0x1000u) >> 13));
}

std::vector<std::uint16_t> MakeColor(std::uint32_t width,
                                     std::uint32_t height,
                                     std::uint32_t seed) {
  std::vector<std::uint16_t> result(
      static_cast<std::size_t>(width) * height * 4);
  for (std::uint32_t y = 0; y < height; ++y) {
    for (std::uint32_t x = 0; x < width; ++x) {
      const std::size_t at = (static_cast<std::size_t>(y) * width + x) * 4;
      const float fx = static_cast<float>(x) / static_cast<float>(width - 1);
      const float fy = static_cast<float>(y) / static_cast<float>(height - 1);
      result[at + 0] = FloatToHalf(std::clamp(fx + seed * 0.001f, 0.0f, 1.0f));
      result[at + 1] = FloatToHalf(fy);
      result[at + 2] = FloatToHalf(0.15f + 0.7f * (1.0f - fx) * fy);
      result[at + 3] = FloatToHalf(1.0f);
    }
  }
  return result;
}

std::uint64_t Fnv1a(std::span<const std::uint8_t> bytes) {
  std::uint64_t hash = kFnvOffset;
  for (const std::uint8_t value : bytes) {
    hash ^= value;
    hash *= kFnvPrime;
  }
  return hash;
}

struct D3DContext {
  ComPtr<IDXGIFactory6> factory;
  ComPtr<IDXGIAdapter4> adapter;
  ComPtr<ID3D12Device> device;
  ComPtr<ID3D12CommandQueue> queue;
  ComPtr<ID3D12CommandAllocator> allocator;
  ComPtr<ID3D12GraphicsCommandList> list;
  ComPtr<ID3D12Fence> timeline;
  ComPtr<ID3D12Fence> local_fence;
  Handle timeline_handle;
  Handle event;
  Handle local_event;
  DXGI_ADAPTER_DESC3 description{};
  std::uint64_t luid = 0;
  std::uint64_t local_value = 0;
};

D3DContext CreateD3DContext() {
  D3DContext context;
  ComPtr<ID3D12Debug> debug;
  if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) {
    debug->EnableDebugLayer();
  }
  ComPtr<ID3D12DeviceRemovedExtendedDataSettings1> dred;
  if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&dred)))) {
    dred->SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
    dred->SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
  }
  RequireHr(CreateDXGIFactory2(0, IID_PPV_ARGS(&context.factory)),
            "CreateDXGIFactory2");
  for (UINT index = 0;; ++index) {
    ComPtr<IDXGIAdapter1> candidate;
    const HRESULT found = context.factory->EnumAdapterByGpuPreference(
        index, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
        IID_PPV_ARGS(&candidate));
    if (found == DXGI_ERROR_NOT_FOUND) break;
    RequireHr(found, "EnumAdapterByGpuPreference");
    DXGI_ADAPTER_DESC1 description{};
    if (FAILED(candidate->GetDesc1(&description)) ||
        (description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0 ||
        description.VendorId != 0x10DEu) {
      continue;
    }
    ComPtr<ID3D12Device> device;
    if (SUCCEEDED(D3D12CreateDevice(candidate.Get(), D3D_FEATURE_LEVEL_12_0,
                                    IID_PPV_ARGS(&device)))) {
      RequireHr(candidate.As(&context.adapter), "Query IDXGIAdapter4");
      context.device = std::move(device);
      break;
    }
  }
  if (context.device == nullptr) Fail("No NVIDIA D3D12 device found");
  RequireHr(context.adapter->GetDesc3(&context.description),
            "IDXGIAdapter4::GetDesc3");
  std::memcpy(&context.luid, &context.description.AdapterLuid,
              sizeof(context.luid));
  D3D12_COMMAND_QUEUE_DESC queue{};
  queue.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
  RequireHr(context.device->CreateCommandQueue(&queue,
                                               IID_PPV_ARGS(&context.queue)),
            "CreateCommandQueue");
  RequireHr(context.device->CreateCommandAllocator(
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                IID_PPV_ARGS(&context.allocator)),
            "CreateCommandAllocator");
  RequireHr(context.device->CreateCommandList(
                0, D3D12_COMMAND_LIST_TYPE_DIRECT, context.allocator.Get(),
                nullptr, IID_PPV_ARGS(&context.list)),
            "CreateCommandList");
  RequireHr(context.list->Close(), "Initial command list close");
  RequireHr(context.device->CreateFence(
                0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&context.timeline)),
            "CreateFence(shared)");
  RequireHr(context.device->CreateFence(
                0, D3D12_FENCE_FLAG_NONE,
                IID_PPV_ARGS(&context.local_fence)),
            "CreateFence(local)");
  HANDLE timeline_handle = nullptr;
  RequireHr(context.device->CreateSharedHandle(
                context.timeline.Get(), nullptr, GENERIC_ALL, nullptr,
                &timeline_handle),
            "CreateSharedHandle(fence)");
  context.timeline_handle.reset(timeline_handle);
  context.event.reset(CreateEventW(nullptr, FALSE, FALSE, nullptr));
  if (!context.event) Fail("CreateEvent(timeline) failed");
  context.local_event.reset(CreateEventW(nullptr, FALSE, FALSE, nullptr));
  if (!context.local_event) Fail("CreateEvent(local queue) failed");
  return context;
}

void RebuildSharedTimeline(D3DContext& context) {
  context.timeline_handle.reset();
  context.timeline.Reset();
  RequireHr(context.device->CreateFence(
                0, D3D12_FENCE_FLAG_SHARED,
                IID_PPV_ARGS(&context.timeline)),
            "CreateFence(recovery generation)");
  HANDLE timeline_handle = nullptr;
  RequireHr(context.device->CreateSharedHandle(
                context.timeline.Get(), nullptr, GENERIC_ALL, nullptr,
                &timeline_handle),
            "CreateSharedHandle(recovery fence)");
  context.timeline_handle.reset(timeline_handle);
}

void ValidateParentDebugMessages(D3DContext& d3d) {
  ComPtr<ID3D12InfoQueue> info;
  if (FAILED(d3d.device.As(&info))) {
    Fail("Parent D3D12 debug queue is unavailable");
  }
  std::uint32_t severe = 0;
  const std::uint64_t count =
      info->GetNumStoredMessagesAllowedByRetrievalFilter();
  for (std::uint64_t index = 0; index < count; ++index) {
    SIZE_T size = 0;
    if (FAILED(info->GetMessage(index, nullptr, &size)) || size == 0) continue;
    std::vector<std::uint8_t> bytes(size);
    auto* message = reinterpret_cast<D3D12_MESSAGE*>(bytes.data());
    if (FAILED(info->GetMessage(index, message, &size))) continue;
    if (message->Severity == D3D12_MESSAGE_SEVERITY_CORRUPTION ||
        message->Severity == D3D12_MESSAGE_SEVERITY_ERROR) {
      ++severe;
      std::cerr << "parent.d3d12_error=\""
                << (message->pDescription == nullptr
                        ? "unknown"
                        : message->pDescription)
                << "\"\n";
    }
  }
  std::cout << "model.parent_d3d12_debug_available=true\n";
  std::cout << "model.parent_d3d12_severe_messages=" << severe << '\n';
  if (severe != 0) Fail("Parent D3D12 debug layer reported severe messages");
}

ComPtr<ID3D12Resource> CreateBuffer(ID3D12Device* device, std::uint64_t size,
                                    D3D12_HEAP_TYPE type,
                                    D3D12_RESOURCE_STATES state) {
  D3D12_HEAP_PROPERTIES heap{};
  heap.Type = type;
  D3D12_RESOURCE_DESC desc{};
  desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  desc.Width = size;
  desc.Height = 1;
  desc.DepthOrArraySize = 1;
  desc.MipLevels = 1;
  desc.SampleDesc.Count = 1;
  desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  ComPtr<ID3D12Resource> resource;
  RequireHr(device->CreateCommittedResource(
                &heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                IID_PPV_ARGS(&resource)),
            "CreateCommittedResource(buffer)");
  return resource;
}

struct SharedTexture {
  ComPtr<ID3D12Resource> resource;
  Handle handle;
  ComPtr<ID3D12Resource> upload;
  ComPtr<ID3D12Resource> readback;
  D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
  std::uint64_t footprint_size = 0;
  std::uint32_t bytes_per_pixel = 0;
};

SharedTexture CreateSharedTexture(D3DContext& d3d, std::uint32_t width,
                                  std::uint32_t height, DXGI_FORMAT format,
                                  std::uint32_t bytes_per_pixel, bool output) {
  SharedTexture texture;
  texture.bytes_per_pixel = bytes_per_pixel;
  D3D12_HEAP_PROPERTIES heap{};
  heap.Type = D3D12_HEAP_TYPE_DEFAULT;
  D3D12_RESOURCE_DESC desc{};
  desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
  desc.Width = width;
  desc.Height = height;
  desc.DepthOrArraySize = 1;
  desc.MipLevels = 1;
  desc.Format = format;
  desc.SampleDesc.Count = 1;
  desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
  desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS |
               (output ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
                       : D3D12_RESOURCE_FLAG_NONE);
  RequireHr(d3d.device->CreateCommittedResource(
                &heap, D3D12_HEAP_FLAG_SHARED, &desc,
                D3D12_RESOURCE_STATE_COMMON, nullptr,
                IID_PPV_ARGS(&texture.resource)),
            "CreateCommittedResource(shared texture)");
  HANDLE shared = nullptr;
  RequireHr(d3d.device->CreateSharedHandle(texture.resource.Get(), nullptr,
                                           GENERIC_ALL, nullptr, &shared),
            "CreateSharedHandle(texture)");
  texture.handle.reset(shared);
  UINT rows = 0;
  UINT64 row_size = 0;
  d3d.device->GetCopyableFootprints(&desc, 0, 1, 0, &texture.footprint,
                                    &rows, &row_size,
                                    &texture.footprint_size);
  if (rows != height || row_size !=
                            static_cast<std::uint64_t>(width) * bytes_per_pixel) {
    Fail("Unexpected texture footprint");
  }
  texture.upload = CreateBuffer(d3d.device.Get(), texture.footprint_size,
                                D3D12_HEAP_TYPE_UPLOAD,
                                D3D12_RESOURCE_STATE_GENERIC_READ);
  if (output) {
    texture.readback = CreateBuffer(d3d.device.Get(), texture.footprint_size,
                                    D3D12_HEAP_TYPE_READBACK,
                                    D3D12_RESOURCE_STATE_COPY_DEST);
  }
  return texture;
}

struct SharedSlot {
  SharedTexture input;
  SharedTexture output;
  SharedTexture depth;
  SharedTexture motion;
  std::uint64_t last_output_hash = 0;
};

class ResourceSet {
 public:
  ResourceSet(D3DContext& d3d, std::uint32_t width, std::uint32_t height)
      : width_(width), height_(height) {
    for (SharedSlot& slot : slots_) {
      slot.input = CreateSharedTexture(
          d3d, width, height, DXGI_FORMAT_R16G16B16A16_FLOAT, 8, false);
      slot.output = CreateSharedTexture(
          d3d, width, height, DXGI_FORMAT_R16G16B16A16_FLOAT, 8, true);
      slot.depth = CreateSharedTexture(d3d, width, height,
                                       DXGI_FORMAT_R32_FLOAT, 4, false);
      slot.motion = CreateSharedTexture(d3d, width, height,
                                        DXGI_FORMAT_R16G16_FLOAT, 4, false);
    }
  }
  SharedSlot& operator[](std::size_t index) { return slots_.at(index); }
  [[nodiscard]] std::uint32_t width() const noexcept { return width_; }
  [[nodiscard]] std::uint32_t height() const noexcept { return height_; }

 private:
  std::array<SharedSlot, protocol::kSlotCount> slots_{};
  std::uint32_t width_ = 0;
  std::uint32_t height_ = 0;
};

std::unique_ptr<bridge::VulkanExternalTransport> CreateVulkanTransport(
    D3DContext& d3d, ResourceSet& resources) {
  bridge::ExternalTransportDescription description;
  description.adapter_luid = d3d.luid;
  description.timeline_fence =
      reinterpret_cast<bridge::HandleToken>(d3d.timeline_handle.get());
  description.width = resources.width();
  description.height = resources.height();
  description.bytes_per_pixel = 8;
  for (std::size_t index = 0; index < protocol::kSlotCount; ++index) {
    description.slots[index].input =
        reinterpret_cast<bridge::HandleToken>(resources[index].input.handle.get());
    description.slots[index].output =
        reinterpret_cast<bridge::HandleToken>(resources[index].output.handle.get());
  }
  std::string error;
  auto transport = bridge::VulkanExternalTransport::Create(description, error);
  if (!transport) Fail("Vulkan transport creation failed: " + error);
  if (!transport->Healthy(error) || transport->adapter_luid() != d3d.luid) {
    Fail("Vulkan transport health check failed: " + error);
  }
  return transport;
}

void Begin(D3DContext& d3d) {
  RequireHr(d3d.allocator->Reset(), "CommandAllocator::Reset");
  RequireHr(d3d.list->Reset(d3d.allocator.Get(), nullptr),
            "CommandList::Reset");
}

void Execute(D3DContext& d3d) {
  RequireHr(d3d.list->Close(), "CommandList::Close");
  ID3D12CommandList* lists[]{d3d.list.Get()};
  d3d.queue->ExecuteCommandLists(1, lists);
}

bool WaitTimeline(D3DContext& d3d, std::uint64_t value,
                  DWORD timeout_ms = kControlTimeoutMs) {
  if (d3d.timeline->GetCompletedValue() >= value) return true;
  RequireHr(d3d.timeline->SetEventOnCompletion(value, d3d.event.get()),
            "Fence::SetEventOnCompletion");
  return WaitForSingleObject(d3d.event.get(), timeout_ms) == WAIT_OBJECT_0;
}

void WaitParentQueue(D3DContext& d3d) {
  const std::uint64_t value = ++d3d.local_value;
  RequireHr(d3d.queue->Signal(d3d.local_fence.Get(), value),
            "Signal(local queue fence)");
  RequireHr(d3d.local_fence->SetEventOnCompletion(value,
                                                  d3d.local_event.get()),
            "Fence::SetEventOnCompletion(local)");
  if (WaitForSingleObject(d3d.local_event.get(), kControlTimeoutMs) !=
      WAIT_OBJECT_0) {
    Fail("Parent D3D12 queue timed out");
  }
}

void Upload(SharedTexture& texture, std::span<const std::uint8_t> bytes,
            std::uint32_t width, std::uint32_t height) {
  if (bytes.size() != static_cast<std::size_t>(width) * height *
                          texture.bytes_per_pixel) {
    Fail("Upload byte count mismatch");
  }
  void* mapped = nullptr;
  const D3D12_RANGE no_read{0, 0};
  RequireHr(texture.upload->Map(0, &no_read, &mapped), "Upload::Map");
  for (std::uint32_t row = 0; row < height; ++row) {
    std::memcpy(static_cast<std::uint8_t*>(mapped) + texture.footprint.Offset +
                    static_cast<std::size_t>(row) *
                        texture.footprint.Footprint.RowPitch,
                bytes.data() + static_cast<std::size_t>(row) * width *
                                   texture.bytes_per_pixel,
                static_cast<std::size_t>(width) * texture.bytes_per_pixel);
  }
  texture.upload->Unmap(0, nullptr);
}

void CopyUpload(ID3D12GraphicsCommandList* list, SharedTexture& texture) {
  D3D12_RESOURCE_BARRIER before{};
  before.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  before.Transition.pResource = texture.resource.Get();
  before.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
  before.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
  before.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
  list->ResourceBarrier(1, &before);
  D3D12_TEXTURE_COPY_LOCATION destination{};
  destination.pResource = texture.resource.Get();
  destination.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
  D3D12_TEXTURE_COPY_LOCATION source{};
  source.pResource = texture.upload.Get();
  source.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
  source.PlacedFootprint = texture.footprint;
  list->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
  std::swap(before.Transition.StateBefore, before.Transition.StateAfter);
  list->ResourceBarrier(1, &before);
}

void WriteInputs(D3DContext& d3d, SharedSlot& slot,
                 std::span<const std::uint16_t> color,
                 std::span<const std::uint16_t> output_canary,
                 std::uint32_t width, std::uint32_t height) {
  if (output_canary.size() != color.size()) {
    Fail("Output canary size mismatch");
  }
  const auto color_bytes = std::as_bytes(color);
  const auto canary_bytes = std::as_bytes(output_canary);
  std::vector<float> depth(static_cast<std::size_t>(width) * height, 0.5f);
  std::vector<std::uint16_t> motion(
      static_cast<std::size_t>(width) * height * 2, 0);
  (void)color_bytes;
  Upload(slot.output,
         {reinterpret_cast<const std::uint8_t*>(canary_bytes.data()),
          canary_bytes.size()},
         width, height);
  Upload(slot.depth,
         {reinterpret_cast<const std::uint8_t*>(depth.data()),
          depth.size() * sizeof(depth[0])},
         width, height);
  Upload(slot.motion,
         {reinterpret_cast<const std::uint8_t*>(motion.data()),
          motion.size() * sizeof(motion[0])},
         width, height);
  Begin(d3d);
  CopyUpload(d3d.list.Get(), slot.output);
  CopyUpload(d3d.list.Get(), slot.depth);
  CopyUpload(d3d.list.Get(), slot.motion);
  Execute(d3d);
  WaitParentQueue(d3d);
}

[[maybe_unused]]
std::vector<std::uint8_t> ReadOutput(D3DContext& d3d, SharedSlot& slot,
                                     std::uint32_t width,
                                     std::uint32_t height,
                                     std::uint64_t wait_value,
                                     std::uint64_t signal_value) {
  RequireHr(d3d.queue->Wait(d3d.timeline.Get(), wait_value),
            "Wait(model_complete)");
  Begin(d3d);
  D3D12_RESOURCE_BARRIER before{};
  before.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  before.Transition.pResource = slot.output.resource.Get();
  before.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
  before.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
  before.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
  d3d.list->ResourceBarrier(1, &before);
  D3D12_TEXTURE_COPY_LOCATION source{};
  source.pResource = slot.output.resource.Get();
  source.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
  D3D12_TEXTURE_COPY_LOCATION destination{};
  destination.pResource = slot.output.readback.Get();
  destination.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
  destination.PlacedFootprint = slot.output.footprint;
  d3d.list->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
  std::swap(before.Transition.StateBefore, before.Transition.StateAfter);
  d3d.list->ResourceBarrier(1, &before);
  Execute(d3d);
  RequireHr(d3d.queue->Signal(d3d.timeline.Get(), signal_value),
            "Signal(copyback_complete)");
  if (!WaitTimeline(d3d, signal_value)) Fail("Output copyback timed out");

  const D3D12_RANGE read_range{
      0, static_cast<SIZE_T>(slot.output.footprint_size)};
  const std::uint8_t* mapped = nullptr;
  RequireHr(slot.output.readback->Map(
                0, &read_range,
                reinterpret_cast<void**>(const_cast<std::uint8_t**>(&mapped))),
            "Readback::Map");
  std::vector<std::uint8_t> result(
      static_cast<std::size_t>(width) * height * 8);
  for (std::uint32_t row = 0; row < height; ++row) {
    std::memcpy(result.data() + static_cast<std::size_t>(row) * width * 8,
                mapped + slot.output.footprint.Offset +
                    static_cast<std::size_t>(row) *
                        slot.output.footprint.Footprint.RowPitch,
                static_cast<std::size_t>(width) * 8);
  }
  const D3D12_RANGE no_write{0, 0};
  slot.output.readback->Unmap(0, &no_write);
  return result;
}

protocol::Packet ConfigureWorker(Worker& worker, D3DContext& d3d,
                                 ResourceSet& resources) {
  auto request = MakeRequest(worker, protocol::MessageType::configure);
  request.width = resources.width();
  request.height = resources.height();
  request.constants.camera_aspect =
      static_cast<float>(resources.width()) /
      static_cast<float>(resources.height());
  request.adapter_luid = d3d.luid;
  request.shared_handles[0] =
      DuplicateIntoWorker(worker, d3d.timeline_handle.get());
  for (std::uint32_t index = 0; index < protocol::kSlotCount; ++index) {
    SharedSlot& slot = resources[index];
    request.shared_handles[protocol::ResourceHandleIndex(index, 0)] =
        DuplicateIntoWorker(worker, slot.input.handle.get());
    request.shared_handles[protocol::ResourceHandleIndex(index, 1)] =
        DuplicateIntoWorker(worker, slot.output.handle.get());
    request.shared_handles[protocol::ResourceHandleIndex(index, 2)] =
        DuplicateIntoWorker(worker, slot.depth.handle.get());
    request.shared_handles[protocol::ResourceHandleIndex(index, 3)] =
        DuplicateIntoWorker(worker, slot.motion.handle.get());
  }
  WriteRequest(worker, request);
  protocol::Packet response;
  if (ReadResponse(worker, request, response) != TransferResult::complete ||
      response.type != protocol::MessageType::configured ||
      response.status != protocol::Status::ok ||
      response.debug_layer_available != 1 ||
      response.debug_severe_messages != 0) {
    Fail("Worker configuration failed status=" +
         std::to_string(static_cast<unsigned>(response.status)));
  }
  return response;
}

void VerifyFirstFrameResetGuard(Worker& worker, D3DContext& d3d,
                                ResourceSet& resources) {
  auto request = MakeRequest(worker, protocol::MessageType::evaluate);
  request.slot = 0;
  request.width = resources.width();
  request.height = resources.height();
  request.reset = 0;
  request.adapter_luid = d3d.luid;
  request.frame_id = 1;
  request.input_ready = 1;
  request.model_complete = 2;
  request.copyback_complete = 3;
  request.retired = 4;
  WriteRequest(worker, request);
  protocol::Packet response;
  if (ReadResponse(worker, request, response) != TransferResult::complete ||
      response.type != protocol::MessageType::evaluated ||
      response.status != protocol::Status::invalid_state ||
      response.successful_evaluations != 0) {
    Fail("Fresh worker accepted an evaluation without Reset=true");
  }
}

void ReleaseWorker(Worker& worker) {
  auto request = MakeRequest(worker, protocol::MessageType::release);
  WriteRequest(worker, request);
  protocol::Packet response;
  if (ReadResponse(worker, request, response) != TransferResult::complete ||
      response.type != protocol::MessageType::released ||
      response.status != protocol::Status::ok) {
    Fail("Worker release failed");
  }
}

void ShutdownWorker(Worker& worker) {
  auto request = MakeRequest(worker, protocol::MessageType::shutdown);
  WriteRequest(worker, request);
  protocol::Packet response;
  if (ReadResponse(worker, request, response) != TransferResult::complete ||
      response.type != protocol::MessageType::shutdown_ack ||
      response.status != protocol::Status::ok) {
    Fail("Worker shutdown response failed");
  }
  if (WaitForSingleObject(worker.process.get(), 5000) != WAIT_OBJECT_0) {
    StopWorker(worker);
    Fail("Worker clean shutdown timed out");
  }
  DWORD code = 1;
  if (!GetExitCodeProcess(worker.process.get(), &code) || code != 0) {
    Fail("Worker clean shutdown returned failure");
  }
}

struct FrameEvidence {
  std::uint64_t input_hash = 0;
  std::uint64_t canary_hash = 0;
  std::uint64_t output_hash = 0;
  std::size_t nonzero_bytes = 0;
  std::size_t changed_bytes = 0;
  std::size_t changed_canary_bytes = 0;
  std::uint32_t feature_creations = 0;
};

FrameEvidence RunFrame(Worker& worker, D3DContext& d3d,
                       ResourceSet& resources,
                       bridge::VulkanExternalTransport& transport,
                       bridge::FrameRing& ring,
                       std::uint64_t completed, std::uint32_t seed) {
  const bridge::FrameGeometry geometry{
      resources.width(), resources.height(),
      static_cast<std::uint32_t>(DXGI_FORMAT_R16G16B16A16_FLOAT), 1};
  const auto begun = ring.Begin(geometry, completed);
  if (begun.status != bridge::BeginStatus::ready || !begun.ticket) {
    Fail("Frame ring refused model frame");
  }
  const bridge::FrameTicket ticket = *begun.ticket;
  SharedSlot& slot = resources[ticket.slot];
  const auto color = MakeColor(resources.width(), resources.height(), seed);
  std::vector<std::uint16_t> output_canary(
      color.size(), static_cast<std::uint16_t>(0x3000u + (seed & 0x03ffu)));
  if (!ring.Transition(ticket, bridge::SlotState::recording,
                       bridge::SlotState::awaiting_submit)) {
    Fail("Frame lifecycle rejected recording");
  }
  WriteInputs(d3d, slot, color, output_canary, resources.width(),
              resources.height());
  const auto input_bytes = std::span<const std::uint8_t>(
      reinterpret_cast<const std::uint8_t*>(color.data()),
      color.size() * sizeof(color[0]));
  std::string transport_error;
  if (!transport.UploadInput(ticket.slot, input_bytes,
                             ticket.timeline.input_ready,
                             kControlTimeoutMs, transport_error)) {
    (void)ring.MarkFailure(ticket, true,
                           "Vulkan input transport failed");
    Fail("Vulkan input transport failed: " + transport_error);
  }
  if (!ring.Transition(ticket, bridge::SlotState::awaiting_submit,
                       bridge::SlotState::input_ready) ||
      !ring.Transition(ticket, bridge::SlotState::input_ready,
                       bridge::SlotState::model_running)) {
    Fail("Frame lifecycle rejected input/model transition");
  }

  auto request = MakeRequest(worker, protocol::MessageType::evaluate);
  request.slot = static_cast<std::uint32_t>(ticket.slot);
  request.width = resources.width();
  request.height = resources.height();
  request.reset = ticket.reset ? 1u : 0u;
  request.constants.camera_aspect =
      static_cast<float>(resources.width()) /
      static_cast<float>(resources.height());
  request.adapter_luid = d3d.luid;
  request.frame_id = ticket.frame_id;
  request.input_ready = ticket.timeline.input_ready;
  request.model_complete = ticket.timeline.model_complete;
  request.copyback_complete = ticket.timeline.copyback_complete;
  request.retired = ticket.timeline.retired;
  WriteRequest(worker, request);
  protocol::Packet response;
  if (ReadResponse(worker, request, response) != TransferResult::complete ||
      response.type != protocol::MessageType::evaluated ||
      response.status != protocol::Status::ok) {
    (void)ring.MarkFailure(ticket, true, "DLSSNR worker evaluation failed");
    Fail("Worker evaluation failed status=" +
         std::to_string(static_cast<unsigned>(response.status)));
  }
  if (!WaitTimeline(d3d, ticket.timeline.model_complete)) {
    (void)ring.MarkFailure(ticket, true, "DLSSNR model timeline timeout");
    Fail("Model-complete timeline timed out");
  }
  if (!ring.Transition(ticket, bridge::SlotState::model_running,
                       bridge::SlotState::output_ready)) {
    Fail("Frame lifecycle rejected output_ready");
  }
  std::vector<std::uint8_t> output;
  if (!transport.ReadOutput(ticket.slot, ticket.timeline.model_complete,
                            ticket.timeline.copyback_complete,
                            kControlTimeoutMs, output, transport_error)) {
    (void)ring.MarkFailure(ticket, true,
                           "Vulkan output transport failed");
    Fail("Vulkan output transport failed: " + transport_error);
  }
  if (!ring.Transition(ticket, bridge::SlotState::output_ready,
                       bridge::SlotState::copyback_pending)) {
    Fail("Frame lifecycle rejected copyback");
  }
  if (ticket.reset && !ring.AcknowledgeReset(ticket)) {
    Fail("Frame lifecycle rejected temporal-reset acknowledgement");
  }
  RequireHr(d3d.queue->Wait(d3d.timeline.Get(),
                            ticket.timeline.copyback_complete),
            "Wait(Vulkan copyback_complete)");
  RequireHr(d3d.queue->Signal(d3d.timeline.Get(), ticket.timeline.retired),
            "Signal(retired)");
  if (!WaitTimeline(d3d, ticket.timeline.retired) ||
      !ring.MarkRetired(ticket)) {
    Fail("Frame retirement failed");
  }
  ring.Reclaim(ticket.timeline.retired);

  const auto canary_bytes = std::span<const std::uint8_t>(
      reinterpret_cast<const std::uint8_t*>(output_canary.data()),
      output_canary.size() * sizeof(output_canary[0]));
  FrameEvidence evidence;
  evidence.input_hash = Fnv1a(input_bytes);
  evidence.canary_hash = Fnv1a(canary_bytes);
  evidence.output_hash = Fnv1a(output);
  for (std::size_t index = 0; index < output.size(); ++index) {
    if (output[index] != 0) ++evidence.nonzero_bytes;
    if (output[index] != input_bytes[index]) ++evidence.changed_bytes;
    if (output[index] != canary_bytes[index]) {
      ++evidence.changed_canary_bytes;
    }
  }
  evidence.feature_creations = response.feature_creations;
  if (response.debug_layer_available != 1 ||
      response.debug_severe_messages != 0 ||
      response.successful_evaluations == 0) {
    Fail("DLSSNR worker debug/evaluation evidence is invalid");
  }
  if (evidence.nonzero_bytes == 0 || evidence.changed_bytes == 0 ||
      evidence.changed_canary_bytes == 0 ||
      evidence.input_hash == evidence.output_hash ||
      evidence.canary_hash == evidence.output_hash ||
      (slot.last_output_hash != 0 &&
       slot.last_output_hash == evidence.output_hash)) {
    Fail("DLSSNR worker did not produce fresh changed output");
  }
  slot.last_output_hash = evidence.output_hash;
  std::cout << "model.frame=" << ticket.frame_id
            << " slot=" << ticket.slot
            << " reset=" << (ticket.reset ? "true" : "false")
            << " input_fnv64=0x" << std::hex << std::uppercase
            << evidence.input_hash << " output_fnv64=0x"
            << evidence.output_hash << std::dec
            << " changed_bytes=" << evidence.changed_bytes
            << " changed_canary_bytes=" << evidence.changed_canary_bytes
            << " timeline_retired=" << ticket.timeline.retired << '\n';
  return evidence;
}

void RunFrames(Worker& worker, D3DContext& d3d, ResourceSet& resources,
               bridge::VulkanExternalTransport& transport,
               bridge::FrameRing& ring, std::uint64_t& completed,
               std::uint32_t count, std::uint32_t seed_base) {
  for (std::uint32_t index = 0; index < count; ++index) {
    const FrameEvidence evidence = RunFrame(
        worker, d3d, resources, transport, ring, completed,
        seed_base + index);
    completed = d3d.timeline->GetCompletedValue();
    if (evidence.feature_creations == 0) {
      Fail("Worker did not report DLSSNR feature creation");
    }
  }
}

void InjectCrash(Worker& worker) {
  auto request = MakeRequest(worker, protocol::MessageType::inject_crash);
  WriteRequest(worker, request);
  if (WaitForSingleObject(worker.process.get(), 5000) != WAIT_OBJECT_0) {
    StopWorker(worker);
    Fail("Injected crash timed out");
  }
  protocol::Packet response;
  if (ReadResponse(worker, request, response, kShortTimeoutMs) !=
      TransferResult::disconnected) {
    Fail("Injected crash did not break the worker pipe");
  }
}

void InjectTimeout(Worker& worker) {
  auto request = MakeRequest(worker, protocol::MessageType::inject_timeout);
  WriteRequest(worker, request);
  protocol::Packet response;
  if (ReadResponse(worker, request, response, kShortTimeoutMs) !=
      TransferResult::timeout) {
    Fail("Injected timeout did not hit the supervisor deadline");
  }
  StopWorker(worker);
}

void InjectDeviceLoss(Worker& worker) {
  auto request = MakeRequest(worker, protocol::MessageType::inject_device_loss);
  WriteRequest(worker, request);
  protocol::Packet response;
  if (ReadResponse(worker, request, response) != TransferResult::complete ||
      response.type != protocol::MessageType::device_lost ||
      response.status != protocol::Status::device_lost) {
    Fail("Worker did not report an induced D3D12 device loss");
  }
  if (WaitForSingleObject(worker.process.get(), 5000) != WAIT_OBJECT_0) {
    StopWorker(worker);
    Fail("Device-loss worker did not exit");
  }
}

}  // namespace

int wmain(int argc, wchar_t** argv) {
  try {
    if (argc != 3) {
      std::wcerr << L"Usage: worker-probe.exe <worker.exe> <runtime-dir>\n";
      return 2;
    }
    const std::filesystem::path worker_binary =
        std::filesystem::weakly_canonical(argv[1]);
    const std::filesystem::path runtime =
        std::filesystem::weakly_canonical(argv[2]);
    if (!std::filesystem::is_regular_file(worker_binary) ||
        !std::filesystem::is_regular_file(runtime / L"sl.interposer.dll")) {
      Fail("Worker or Streamline runtime is missing");
    }

    D3DContext d3d = CreateD3DContext();
    std::wcout << L"model.adapter=\"" << d3d.description.Description << L"\"\n";
    std::cout << "model.shared_gpu_resources=true\n";
    std::cout << "model.slot_count=" << protocol::kSlotCount << '\n';
    std::cout << "model.vulkan_input_transport=true\n";
    std::cout << "model.vulkan_output_transport=true\n";
    bridge::FrameRing ring(protocol::kSlotCount);
    std::uint64_t completed = 0;

    auto full_hd = std::make_unique<ResourceSet>(d3d, 1920, 1080);
    auto full_hd_transport = CreateVulkanTransport(d3d, *full_hd);
    Worker primary = StartWorker(worker_binary, runtime, 1);
    ConfigureWorker(primary, d3d, *full_hd);
    VerifyFirstFrameResetGuard(primary, d3d, *full_hd);
    std::cout << "model.first_frame_reset_enforced=true\n";
    RunFrames(primary, d3d, *full_hd, *full_hd_transport, ring, completed, 6,
              1);
    ReleaseWorker(primary);
    full_hd_transport.reset();
    full_hd.reset();

    auto resized = std::make_unique<ResourceSet>(d3d, 1280, 720);
    auto resized_transport = CreateVulkanTransport(d3d, *resized);
    const auto resize_gate = ring.Begin(
        {1280, 720,
         static_cast<std::uint32_t>(DXGI_FORMAT_R16G16B16A16_FLOAT), 1},
        completed);
    if (resize_gate.status != bridge::BeginStatus::ready ||
        !resize_gate.ticket || !resize_gate.ticket->reset ||
        !ring.MarkFailure(*resize_gate.ticket, false, "resize preflight")) {
      Fail("Resize did not request temporal reset");
    }
    ConfigureWorker(primary, d3d, *resized);
    RunFrames(primary, d3d, *resized, *resized_transport, ring, completed, 3,
              100);
    InjectCrash(primary);
    std::cout << "model.resize_rebuild=true\n";
    std::cout << "model.temporal_reset_after_resize=true\n";
    std::cout << "model.crash_isolated=true\n";
    std::cout << "pipeline_state=bypassed\n";

    resized_transport.reset();
    resized.reset();
    RebuildSharedTimeline(d3d);
    auto after_crash = std::make_unique<ResourceSet>(d3d, 1280, 720);
    auto after_crash_transport = CreateVulkanTransport(d3d, *after_crash);
    bridge::FrameRing after_crash_ring(protocol::kSlotCount);
    std::uint64_t after_crash_completed = 0;
    Worker after_crash_worker = StartWorker(worker_binary, runtime, 2);
    ConfigureWorker(after_crash_worker, d3d, *after_crash);
    const FrameEvidence crash_recovery = RunFrame(
        after_crash_worker, d3d, *after_crash, *after_crash_transport,
        after_crash_ring, after_crash_completed, 200);
    if (crash_recovery.feature_creations != 1) {
      Fail("Crash recovery did not create a fresh model feature");
    }
    std::cout << "model.restart_after_crash=true\n";
    InjectTimeout(after_crash_worker);
    std::cout << "model.timeout_bounded=true\n";
    std::cout << "pipeline_state=bypassed\n";

    after_crash_transport.reset();
    after_crash.reset();
    RebuildSharedTimeline(d3d);
    auto after_timeout = std::make_unique<ResourceSet>(d3d, 1280, 720);
    auto after_timeout_transport = CreateVulkanTransport(d3d, *after_timeout);
    bridge::FrameRing after_timeout_ring(protocol::kSlotCount);
    std::uint64_t after_timeout_completed = 0;
    Worker after_timeout_worker = StartWorker(worker_binary, runtime, 3);
    ConfigureWorker(after_timeout_worker, d3d, *after_timeout);
    const FrameEvidence timeout_recovery = RunFrame(
        after_timeout_worker, d3d, *after_timeout, *after_timeout_transport,
        after_timeout_ring, after_timeout_completed, 300);
    if (timeout_recovery.feature_creations != 1) {
      Fail("Timeout recovery did not create a fresh model feature");
    }
    std::cout << "model.restart_after_timeout=true\n";
    InjectDeviceLoss(after_timeout_worker);
    RequireHr(d3d.device->GetDeviceRemovedReason(),
              "Parent device after worker device loss");
    std::cout << "model.device_loss_isolated=true\n";
    std::cout << "pipeline_state=bypassed\n";

    after_timeout_transport.reset();
    after_timeout.reset();
    RebuildSharedTimeline(d3d);
    auto recovered_resources =
        std::make_unique<ResourceSet>(d3d, 1280, 720);
    auto recovered_transport =
        CreateVulkanTransport(d3d, *recovered_resources);
    bridge::FrameRing recovered_ring(protocol::kSlotCount);
    std::uint64_t recovered_completed = 0;
    Worker recovered = StartWorker(worker_binary, runtime, 4);
    ConfigureWorker(recovered, d3d, *recovered_resources);
    const FrameEvidence recovery =
        RunFrame(recovered, d3d, *recovered_resources, *recovered_transport,
                 recovered_ring, recovered_completed, 500);
    if (recovery.feature_creations != 1) {
      Fail("Restarted worker did not create exactly one fresh model feature");
    }
    recovered_completed = d3d.timeline->GetCompletedValue();
    ShutdownWorker(recovered);
    recovered_transport.reset();
    ValidateParentDebugMessages(d3d);
    std::cout << "model.worker_d3d12_debug_available=true\n";
    std::cout << "model.worker_d3d12_severe_messages=0\n";
    std::cout << "model.resource_generation_rebuilt=true\n";
    std::cout << "model.restart_after_device_loss=true\n";
    std::cout << "model.restart_after_failure=true\n";
    std::cout << "model.shutdown_clean=true\n";
    std::cout << "model.inference_executed=true\n";
    std::cout << "probe=dlssnr_bridge_worker_passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL " << error.what() << '\n';
    std::cerr << "probe=dlssnr_bridge_worker_failed\n";
    return 1;
  }
}
