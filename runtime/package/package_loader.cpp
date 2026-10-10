/* NeuralStudio package verifier/loader - SPDX-License-Identifier: MIT */

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX

#include "package_loader.h"

#include <windows.h>
#include <bcrypt.h>
#include <wincrypt.h>
#include <wintrust.h>
#include <softpub.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cwctype>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <vector>

#pragma comment(lib, "bcrypt.lib")
#pragma comment(lib, "crypt32.lib")
#pragma comment(lib, "wintrust.lib")

namespace neuralstudio {
namespace {

std::filesystem::path Utf8Path(const std::string& text) {
  if (text.empty()) return {};
  const int size = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                                       text.data(),
                                       static_cast<int>(text.size()),
                                       nullptr, 0);
  if (size <= 0) return {};
  std::wstring wide(static_cast<std::size_t>(size), L'\0');
  if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(),
                          static_cast<int>(text.size()), wide.data(), size) != size) {
    return {};
  }
  return std::filesystem::path(wide);
}

std::string Lower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

bool Inside(const std::filesystem::path& root,
            const std::filesystem::path& candidate) {
  std::wstring base = std::filesystem::weakly_canonical(root).wstring();
  std::wstring child = std::filesystem::weakly_canonical(candidate).wstring();
  std::transform(base.begin(), base.end(), base.begin(),
                 [](wchar_t c) { return static_cast<wchar_t>(std::towlower(c)); });
  std::transform(child.begin(), child.end(), child.begin(),
                 [](wchar_t c) { return static_cast<wchar_t>(std::towlower(c)); });
  while (!base.empty() && (base.back() == L'\\' || base.back() == L'/')) {
    base.pop_back();
  }
  return child.size() > base.size() &&
         child.compare(0, base.size(), base) == 0 &&
         (child[base.size()] == L'\\' || child[base.size()] == L'/');
}

bool SafeChild(const std::filesystem::path& root, const std::string& relative,
               std::filesystem::path& result) {
  if (relative.empty()) return false;
  const std::filesystem::path requested = Utf8Path(relative);
  if (requested.is_absolute()) return false;
  result = std::filesystem::weakly_canonical(root / requested);
  return Inside(root, result);
}

std::string ReadText(const std::filesystem::path& path, std::string& error) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) { error = "Opening package manifest failed"; return {}; }
  std::ostringstream text;
  text << stream.rdbuf();
  if (!stream.good() && !stream.eof()) {
    error = "Reading package manifest failed";
    return {};
  }
  return text.str();
}

std::filesystem::path ExpandSearchPath(const std::filesystem::path& root,
                                       std::string value) {
  const std::string token = "${packageRoot}";
  const std::size_t at = value.find(token);
  if (at != std::string::npos) {
    value.replace(at, token.size(), root.string());
  }
  const std::wstring wide = Utf8Path(value).wstring();
  const DWORD size = ExpandEnvironmentStringsW(wide.c_str(), nullptr, 0);
  if (size == 0) return {};
  std::wstring expanded(size, L'\0');
  if (ExpandEnvironmentStringsW(wide.c_str(), expanded.data(), size) == 0) {
    return {};
  }
  if (!expanded.empty() && expanded.back() == L'\0') expanded.pop_back();
  return std::filesystem::path(expanded);
}

std::string CertificateSigner(const std::filesystem::path& path,
                              std::string& error) {
  HCERTSTORE store = nullptr;
  HCRYPTMSG message = nullptr;
  DWORD encoding = 0;
  DWORD content = 0;
  DWORD format = 0;
  if (!CryptQueryObject(CERT_QUERY_OBJECT_FILE, path.c_str(),
                        CERT_QUERY_CONTENT_FLAG_PKCS7_SIGNED_EMBED,
                        CERT_QUERY_FORMAT_FLAG_BINARY, 0, &encoding, &content,
                        &format, &store, &message, nullptr)) {
    error = "Reading Authenticode signer failed";
    return {};
  }
  struct Cleanup {
    HCERTSTORE store;
    HCRYPTMSG message;
    ~Cleanup() {
      if (message) CryptMsgClose(message);
      if (store) CertCloseStore(store, 0);
    }
  } cleanup{store, message};
  DWORD size = 0;
  if (!CryptMsgGetParam(message, CMSG_SIGNER_INFO_PARAM, 0, nullptr, &size) ||
      size == 0) {
    error = "Authenticode signer metadata is missing";
    return {};
  }
  std::vector<std::uint8_t> storage(size);
  if (!CryptMsgGetParam(message, CMSG_SIGNER_INFO_PARAM, 0, storage.data(),
                        &size)) {
    error = "Reading Authenticode signer metadata failed";
    return {};
  }
  const auto* signer = reinterpret_cast<const CMSG_SIGNER_INFO*>(storage.data());
  CERT_INFO info{};
  info.Issuer = signer->Issuer;
  info.SerialNumber = signer->SerialNumber;
  PCCERT_CONTEXT certificate = CertFindCertificateInStore(
      store, encoding, 0, CERT_FIND_SUBJECT_CERT, &info, nullptr);
  if (certificate == nullptr) {
    error = "Authenticode signing certificate was not found";
    return {};
  }
  struct CertCleanup {
    PCCERT_CONTEXT value;
    ~CertCleanup() { CertFreeCertificateContext(value); }
  } cert_cleanup{certificate};
  const DWORD chars = CertGetNameStringA(
      certificate, CERT_NAME_SIMPLE_DISPLAY_TYPE, 0, nullptr, nullptr, 0);
  if (chars <= 1) { error = "Authenticode signer has no display name"; return {}; }
  std::string result(chars, '\0');
  CertGetNameStringA(certificate, CERT_NAME_SIMPLE_DISPLAY_TYPE, 0, nullptr,
                     result.data(), chars);
  if (!result.empty() && result.back() == '\0') result.pop_back();
  return result;
}

void Rule(VerifiedPackage& package, std::string name, bool passed,
          std::string detail) {
  package.rules.push_back({std::move(name), passed, std::move(detail)});
}

}  // namespace

std::string Sha256File(const std::filesystem::path& path,
                       std::string& error) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) { error = "Opening artifact for hashing failed"; return {}; }
  BCRYPT_ALG_HANDLE algorithm = nullptr;
  BCRYPT_HASH_HANDLE hash = nullptr;
  if (BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr,
                                  0) < 0 ||
      BCryptCreateHash(algorithm, &hash, nullptr, 0, nullptr, 0, 0) < 0) {
    if (algorithm) BCryptCloseAlgorithmProvider(algorithm, 0);
    error = "Creating SHA-256 provider failed";
    return {};
  }
  struct Cleanup {
    BCRYPT_ALG_HANDLE algorithm;
    BCRYPT_HASH_HANDLE hash;
    ~Cleanup() {
      if (hash) BCryptDestroyHash(hash);
      if (algorithm) BCryptCloseAlgorithmProvider(algorithm, 0);
    }
  } cleanup{algorithm, hash};
  std::array<std::uint8_t, 1 << 16> buffer{};
  while (stream) {
    stream.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
    const std::streamsize count = stream.gcount();
    if (count > 0 && BCryptHashData(hash, buffer.data(),
                                   static_cast<ULONG>(count), 0) < 0) {
      error = "Hashing artifact failed";
      return {};
    }
  }
  if (!stream.eof()) { error = "Reading artifact for hashing failed"; return {}; }
  std::array<std::uint8_t, 32> digest{};
  if (BCryptFinishHash(hash, digest.data(),
                       static_cast<ULONG>(digest.size()), 0) < 0) {
    error = "Finishing SHA-256 failed";
    return {};
  }
  std::ostringstream encoded;
  encoded << std::uppercase << std::hex << std::setfill('0');
  for (std::uint8_t byte : digest) encoded << std::setw(2) << static_cast<unsigned>(byte);
  return encoded.str();
}

bool VerifyAuthenticode(const std::filesystem::path& path,
                        const std::string& expected_signer,
                        std::string& actual_signer,
                        std::string& error) {
  WINTRUST_FILE_INFO file{};
  file.cbStruct = sizeof(file);
  file.pcwszFilePath = path.c_str();
  GUID policy = WINTRUST_ACTION_GENERIC_VERIFY_V2;
  WINTRUST_DATA trust{};
  trust.cbStruct = sizeof(trust);
  trust.dwUIChoice = WTD_UI_NONE;
  trust.fdwRevocationChecks = WTD_REVOKE_NONE;
  trust.dwUnionChoice = WTD_CHOICE_FILE;
  trust.pFile = &file;
  trust.dwStateAction = WTD_STATEACTION_VERIFY;
  /* Trust the normal Windows Authenticode policy. Hash pinning is checked
     separately, so this call's job is certificate-chain and publisher trust.
     WTD_SAFER_FLAG changes policy semantics and incorrectly rejected a valid
     NVIDIA chain on a clean Windows 11 host. */
  trust.dwProvFlags = 0;
  const LONG status = WinVerifyTrust(nullptr, &policy, &trust);
  trust.dwStateAction = WTD_STATEACTION_CLOSE;
  (void)WinVerifyTrust(nullptr, &policy, &trust);
  if (status != ERROR_SUCCESS) {
    std::ostringstream detail;
    detail << "WinVerifyTrust rejected artifact: 0x" << std::hex
           << static_cast<unsigned long>(status);
    error = detail.str();
    return false;
  }
  actual_signer = CertificateSigner(path, error);
  if (actual_signer.empty()) return false;
  if (!expected_signer.empty() &&
      Lower(actual_signer).find(Lower(expected_signer)) == std::string::npos) {
    error = "Authenticode signer mismatch: " + actual_signer;
    return false;
  }
  return true;
}

bool VerifyPackage(const std::filesystem::path& requested_root,
                   VerifiedPackage& result, std::string& error) {
  try {
    result = {};
    error.clear();
    result = {};
    result.root = std::filesystem::weakly_canonical(requested_root);
    if (!std::filesystem::is_directory(result.root)) {
      error = "Package root is not a directory";
      return false;
    }
    const std::string manifest_text = ReadText(result.root / L"package.json", error);
    if (!error.empty()) return false;
    json::ParseError parse;
    if (!json::Parse(manifest_text, result.document, parse)) {
      error = "Manifest JSON parse failed at byte " + std::to_string(parse.position) +
              ": " + parse.message;
      return false;
    }
    result.manifest = ReadManifest(result.document, NS_ADAPTER_ABI_VERSION_MAJOR);
    if (!result.manifest.ok) {
      error = result.manifest.errors.empty() ? "Manifest reader rejected package"
                                             : result.manifest.errors.front();
      return false;
    }

    const json::Value& provenance = result.document["provenance"];
    const json::Value& backend = result.document["backend"];
    const std::string adapter_relative = backend["adapter"]["library"].AsString();
    if (!SafeChild(result.root, adapter_relative, result.adapter_library)) {
      error = "Adapter path escapes the package root";
      Rule(result, "backend.adapter.library", false, error);
      return false;
    }
    bool adapter_declared = false;
    for (const json::Value& artifact : provenance["artifacts"].items()) {
      const std::string relative = artifact["path"].AsString();
      std::filesystem::path path;
      if (!SafeChild(result.root, relative, path) ||
          !std::filesystem::is_regular_file(path)) {
        error = "Package artifact is missing or escapes the package root: " + relative;
        Rule(result, "artifact.path", false, error);
        return false;
      }
      std::string hash_error;
      const std::string actual = Sha256File(path, hash_error);
      const std::string expected = artifact["sha256"].AsString();
      if (!hash_error.empty() || Lower(actual) != Lower(expected)) {
        error = hash_error.empty() ? "Package artifact hash mismatch: " + relative
                                   : hash_error;
        Rule(result, "artifact.sha256", false, error);
        return false;
      }
      Rule(result, "artifact.sha256", true, relative);
      result.verified_files.push_back(path);
      const bool signature_required = artifact["signature"]["required"].AsBool(false);
      if (signature_required) {
        std::string signer;
        if (!VerifyAuthenticode(path,
                                artifact["signature"]["expectedSigner"].AsString(),
                                signer, error)) {
          Rule(result, "artifact.signature", false, error);
          return false;
        }
        Rule(result, "artifact.signature", true, signer);
      }
      if (artifact["role"].AsString() == "adapter" &&
          relative == adapter_relative) adapter_declared = true;
    }
    if (!adapter_declared) {
      error = "Adapter library is not bound to an exact artifact hash";
      Rule(result, "backend.adapter.library", false, error);
      return false;
    }

    const json::Value& acquisition = backend["runtimeAcquisition"];
    const auto& runtime_files = acquisition["requiredFiles"].items();
    if (!runtime_files.empty()) {
      std::string last_candidate_error;
      for (const json::Value& raw_path : acquisition["searchPaths"].items()) {
        const std::filesystem::path directory =
            ExpandSearchPath(result.root, raw_path.AsString());
        if (directory.empty() || !std::filesystem::is_directory(directory)) continue;
        bool valid_directory = true;
        std::string candidate_error;
        for (const json::Value& runtime_file : runtime_files) {
          const std::filesystem::path file = directory /
              Utf8Path(runtime_file["name"].AsString());
          if (!std::filesystem::is_regular_file(file)) {
            valid_directory = false;
            candidate_error = "Located runtime file is missing";
            break;
          }
          const std::string expected_hash = runtime_file["sha256"].AsString();
          if (!expected_hash.empty()) {
            std::string hash_error;
            if (Lower(Sha256File(file, hash_error)) != Lower(expected_hash)) {
              valid_directory = false;
              candidate_error = hash_error.empty() ? "Located runtime hash mismatch"
                                                    : hash_error;
              break;
            }
          }
          if (runtime_file["signature"]["required"].AsBool(false)) {
            std::string signer;
            if (!VerifyAuthenticode(
                    file, runtime_file["signature"]["expectedSigner"].AsString(),
                    signer, candidate_error)) {
              valid_directory = false;
              break;
            }
          }
          result.verified_files.push_back(
              std::filesystem::weakly_canonical(file));
        }
        if (valid_directory) {
          result.runtime_directory = std::filesystem::weakly_canonical(directory);
          Rule(result, "runtimeAcquisition", true,
               result.runtime_directory.string());
          return true;
        }
        if (!candidate_error.empty()) {
          last_candidate_error = directory.string() + ": " + candidate_error;
        }
      }
      error = "No runtime search path contained the exact signed provider files";
      if (!last_candidate_error.empty()) error += "; " + last_candidate_error;
      Rule(result, "runtimeAcquisition", false, error);
      return false;
    }
    result.runtime_directory = result.root / L"runtime";
    return true;
  } catch (const std::exception& exception) {
    error = exception.what();
    return false;
  }
}

}  // namespace neuralstudio
