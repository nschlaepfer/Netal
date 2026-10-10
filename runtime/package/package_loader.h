/* NeuralStudio package verifier/loader - SPDX-License-Identifier: MIT */

#pragma once

#include "../manifest/json.h"
#include "../manifest/manifest_reader.h"

#include <filesystem>
#include <string>
#include <vector>

namespace neuralstudio {

struct VerificationRule {
  std::string rule;
  bool passed = false;
  std::string detail;
};

struct VerifiedPackage {
  std::filesystem::path root;
  std::filesystem::path adapter_library;
  std::filesystem::path runtime_directory;
  /* Every executable/runtime file whose contents were verified. The loader
     locks these read-only and re-verifies while locked before executing code,
     closing the verify-to-load replacement window. */
  std::vector<std::filesystem::path> verified_files;
  json::Value document;
  ManifestReadResult manifest;
  std::vector<VerificationRule> rules;
};

bool VerifyPackage(const std::filesystem::path& package_root,
                   VerifiedPackage& result, std::string& error);

std::string Sha256File(const std::filesystem::path& path,
                       std::string& error);

bool VerifyAuthenticode(const std::filesystem::path& path,
                        const std::string& expected_signer,
                        std::string& actual_signer,
                        std::string& error);

}  // namespace neuralstudio
