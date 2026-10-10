/* NeuralStudio Local API v1 named-pipe server - SPDX-License-Identifier: MIT */

#pragma once

#include <filesystem>

namespace neuralstudio {

struct LocalApiOptions {
  std::filesystem::path default_package;
  std::filesystem::path default_output;
  bool inference_armed = false;
};

int RunLocalApiServer(const LocalApiOptions& options);

}  // namespace neuralstudio
