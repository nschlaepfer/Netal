/*
 * NeuralStudio runtime - manifest reader
 * SPDX-License-Identifier: MIT
 *
 * Turns a parsed package.json into the PackageRequirements the compatibility
 * engine consumes. This is the connective tissue between the ecosystem contract
 * (the manifest schema) and the runtime's decision-making.
 *
 * It reads only the compatibility-relevant subset. Full schema validation lives
 * in tools/validate-package.ps1; this reader assumes a manifest that has passed
 * validation and focuses on mapping it faithfully. Where an enum string is
 * unrecognized it reports a named issue rather than guessing, because a silently
 * mis-mapped convention is exactly the class of bug the manifest exists to
 * prevent.
 */

#ifndef NEURALSTUDIO_RUNTIME_MANIFEST_READER_H
#define NEURALSTUDIO_RUNTIME_MANIFEST_READER_H

#include <string>
#include <vector>

#include "json.h"
#include "../broker/compatibility.h"

namespace neuralstudio {

struct ManifestSummary {
    std::string package_id;
    std::string name;
    std::string version;
    std::string publisher;
    std::string license_id;
    std::string backend_type;
    std::string distribution;
    bool redistribution_allowed = false;
    bool adapter_signature_required = false;
    bool runtime_signature_required = false;
    bool stateful = false;
};

struct ManifestReadResult {
    bool ok = false;
    ManifestSummary summary;
    PackageRequirements requirements;

    /* Non-fatal mapping notes, e.g. an enum value this runtime build does not
       recognize. Surfaced so the catalog can warn instead of hiding a model. */
    std::vector<std::string> warnings;

    /* Fatal problems that prevented a usable requirements structure. */
    std::vector<std::string> errors;
};

/* Maps enum strings from the manifest to runtime enums. Return false on an
   unrecognized value so callers can report it by name. */
bool ParseGpuVendor(const std::string& text, GpuVendor& out_vendor);
bool ParseGraphicsApi(const std::string& text, ns_graphics_api& out_api);
bool ParseGuideKind(const std::string& text, ns_guide_kind& out_kind);
bool ParseOsPlatform(const std::string& text, OsPlatform& out_platform);

/* Reads a parsed manifest document. */
ManifestReadResult ReadManifest(const json::Value& document, std::uint32_t host_abi_major);

/* Convenience: parse text then read it. */
ManifestReadResult ReadManifestText(const std::string& text, std::uint32_t host_abi_major);

}  // namespace neuralstudio

#endif  /* NEURALSTUDIO_RUNTIME_MANIFEST_READER_H */
