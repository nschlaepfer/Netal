/* NeuralStudio DLSSNR isolated-worker client - SPDX-License-Identifier: MIT */

#pragma once

#include <d3d12.h>

#include "dlssnr_worker_protocol.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace neuralstudio::dlssnr_worker {

namespace protocol = vulkan_bridge::dlssnr_worker;

enum class FailureInjection { crash, timeout, device_loss };

struct Evaluation {
  protocol::Status status = protocol::Status::internal_error;
  bool worker_alive = false;
  std::uint32_t feature_creations = 0;
  std::uint32_t successful_evaluations = 0;
  std::uint32_t debug_severe_messages = 0;
};

class Client {
 public:
  Client();
  ~Client();
  Client(const Client&) = delete;
  Client& operator=(const Client&) = delete;
  Client(Client&&) noexcept;
  Client& operator=(Client&&) noexcept;

  bool Start(const std::filesystem::path& worker_binary,
             const std::filesystem::path& runtime_directory,
             std::uint32_t generation, std::string& error);

  bool Configure(ID3D12Device* source_device, std::uint32_t width,
                 std::uint32_t height,
                 const std::array<HANDLE, 4>& shared_resources,
                 const protocol::Options& options, std::string& error);

  Evaluation Evaluate(std::uint64_t frame_id, std::uint32_t slot, bool reset,
                      const protocol::FrameConstants& constants,
                      std::uint32_t timeout_ms, std::string& error);

  bool SetOptions(const protocol::Options& options, std::uint32_t timeout_ms,
                  std::string& error);
  bool Release(std::uint32_t timeout_ms, std::string& error);
  bool InjectFailure(FailureInjection failure, std::uint32_t timeout_ms,
                     std::string& error);
  bool Shutdown(std::uint32_t timeout_ms, std::string& error);
  void Abort() noexcept;

  [[nodiscard]] bool alive() const noexcept;
  [[nodiscard]] bool configured() const noexcept;
  [[nodiscard]] std::uint32_t process_id() const noexcept;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace neuralstudio::dlssnr_worker
