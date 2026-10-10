/*
 * NeuralStudio runtime - compatibility engine
 * SPDX-License-Identifier: MIT
 */

#include "compatibility.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <sstream>

namespace neuralstudio {
namespace {

bool Contains(const std::vector<ns_guide_kind>& list, ns_guide_kind kind) {
    return std::find(list.begin(), list.end(), kind) != list.end();
}

bool Contains(const std::vector<ns_graphics_api>& list, ns_graphics_api api) {
    return std::find(list.begin(), list.end(), api) != list.end();
}

std::string ToLower(const std::string& text) {
    std::string result = text;
    std::transform(result.begin(), result.end(), result.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return result;
}

std::vector<std::uint64_t> SplitVersion(const std::string& version) {
    std::vector<std::uint64_t> components;
    std::uint64_t current = 0;
    bool has_digit = false;

    for (const char c : version) {
        if (c >= '0' && c <= '9') {
            current = current * 10u + static_cast<std::uint64_t>(c - '0');
            has_digit = true;
        } else if (c == '.') {
            components.push_back(has_digit ? current : 0u);
            current = 0;
            has_digit = false;
        } else {
            /* Stop at the first non-numeric, non-separator character so build
               suffixes like "616.56-beta" compare on their numeric part. */
            break;
        }
    }
    if (has_digit || components.empty()) {
        components.push_back(current);
    }
    return components;
}

void AddReason(CompatibilityVerdict& verdict, std::string rule, std::string detail) {
    verdict.reasons.push_back(Incompatibility{std::move(rule), std::move(detail)});
}

std::string Describe(const Extent& extent) {
    std::ostringstream stream;
    stream << extent.width << "x" << extent.height;
    return stream.str();
}

}  // namespace

const char* ToString(GpuVendor vendor) {
    switch (vendor) {
    case GpuVendor::Nvidia:  return "NVIDIA";
    case GpuVendor::Amd:     return "AMD";
    case GpuVendor::Intel:   return "Intel";
    case GpuVendor::Any:     return "any";
    case GpuVendor::Unknown: return "unknown";
    }
    return "unknown";
}

const char* ToString(ns_guide_kind kind) {
    switch (kind) {
    case ns_guide_color:          return "color";
    case ns_guide_depth:          return "depth";
    case ns_guide_motion_vectors: return "motionVectors";
    case ns_guide_exposure:       return "exposure";
    case ns_guide_mask:           return "mask";
    case ns_guide_ui_mask:        return "uiMask";
    case ns_guide_albedo:         return "albedo";
    case ns_guide_normal:         return "normal";
    case ns_guide_roughness:      return "roughness";
    case ns_guide_specular:       return "specular";
    case ns_guide_transparency:   return "transparency";
    case ns_guide_optical_flow:   return "opticalFlow";
    }
    return "unknown";
}

const char* ToString(ns_graphics_api api) {
    switch (api) {
    case ns_api_none:   return "none";
    case ns_api_d3d12:  return "d3d12";
    case ns_api_d3d11:  return "d3d11";
    case ns_api_vulkan: return "vulkan";
    case ns_api_cuda:   return "cuda";
    case ns_api_cpu:    return "cpu";
    }
    return "unknown";
}

int CompareVersions(const std::string& left, const std::string& right) {
    const std::vector<std::uint64_t> a = SplitVersion(left);
    const std::vector<std::uint64_t> b = SplitVersion(right);
    const std::size_t count = std::max(a.size(), b.size());

    for (std::size_t i = 0; i < count; ++i) {
        const std::uint64_t lhs = (i < a.size()) ? a[i] : 0u;
        const std::uint64_t rhs = (i < b.size()) ? b[i] : 0u;
        if (lhs < rhs) { return -1; }
        if (lhs > rhs) { return 1; }
    }
    return 0;
}

int RankArchitecture(GpuVendor vendor, const std::string& architecture) {
    const std::string name = ToLower(architecture);
    if (name.empty()) { return -1; }

    switch (vendor) {
    case GpuVendor::Nvidia:
        if (name == "pascal")    { return 10; }
        if (name == "turing")    { return 20; }
        if (name == "ampere")    { return 30; }
        if (name == "ada")       { return 40; }
        if (name == "blackwell") { return 50; }
        return -1;
    case GpuVendor::Amd:
        if (name == "rdna")  { return 10; }
        if (name == "rdna2") { return 20; }
        if (name == "rdna3") { return 30; }
        if (name == "rdna4") { return 40; }
        return -1;
    case GpuVendor::Intel:
        if (name == "xe")      { return 10; }
        if (name == "xe2")     { return 20; }
        if (name == "battlemage") { return 20; }
        return -1;
    default:
        return -1;
    }
}

CompatibilityVerdict CompatibilityEngine::EvaluateHardware(
    const PackageRequirements& requirements,
    const HardwareProfile& hardware) const {

    CompatibilityVerdict verdict;

    if (requirements.adapter_abi_major != hardware.host_adapter_abi_major) {
        std::ostringstream detail;
        detail << "Package targets adapter ABI " << requirements.adapter_abi_major
               << "; this runtime speaks " << hardware.host_adapter_abi_major << ".";
        AddReason(verdict, "backend.adapter.abiVersion", detail.str());
    }

    if (requirements.platform != hardware.platform) {
        AddReason(verdict, "compatibility.os.platform", "Package does not support this operating system.");
    } else if (!requirements.min_os_version.empty() && !hardware.os_version.empty() &&
               CompareVersions(hardware.os_version, requirements.min_os_version) < 0) {
        AddReason(verdict, "compatibility.os.minVersion",
                  "Requires OS " + requirements.min_os_version + "; found " + hardware.os_version + ".");
    }

    if (!requirements.vendors.empty()) {
        const bool accepts_any =
            std::find(requirements.vendors.begin(), requirements.vendors.end(), GpuVendor::Any) !=
            requirements.vendors.end();
        const bool matches_vendor =
            std::find(requirements.vendors.begin(), requirements.vendors.end(), hardware.vendor) !=
            requirements.vendors.end();
        if (!accepts_any && !matches_vendor) {
            AddReason(verdict, "compatibility.gpu.vendors",
                      std::string("Requires a different GPU vendor; found ") + ToString(hardware.vendor) + ".");
        }
    }

    if (!requirements.min_architecture.empty()) {
        const int required_rank = RankArchitecture(hardware.vendor, requirements.min_architecture);
        const int actual_rank = RankArchitecture(hardware.vendor, hardware.architecture);
        if (required_rank < 0 || actual_rank < 0) {
            /* An unknown architecture name is reported, not used to refuse. A
               table that has not been updated should not block newer hardware. */
            verdict.warnings.push_back(
                "Could not rank GPU architecture '" + hardware.architecture + "' against required '" +
                requirements.min_architecture + "'; compatibility was not verified.");
        } else if (actual_rank < required_rank) {
            AddReason(verdict, "compatibility.gpu.minArchitecture",
                      "Requires " + requirements.min_architecture + " or newer; found " +
                          hardware.architecture + ".");
        }
    }

    if (requirements.min_vram_bytes > 0 && hardware.vram_bytes > 0 &&
        hardware.vram_bytes < requirements.min_vram_bytes) {
        std::ostringstream detail;
        detail << "Requires " << (requirements.min_vram_bytes / (1024u * 1024u))
               << " MB of VRAM; found " << (hardware.vram_bytes / (1024u * 1024u)) << " MB.";
        AddReason(verdict, "compatibility.gpu.minVramMB", detail.str());
    }

    for (const std::string& feature : requirements.required_gpu_features) {
        if (std::find(hardware.gpu_features.begin(), hardware.gpu_features.end(), feature) ==
            hardware.gpu_features.end()) {
            AddReason(verdict, "compatibility.gpu.requiredFeatures",
                      "GPU does not report required feature '" + feature + "'.");
        }
    }

    if (!hardware.driver_version.empty()) {
        if (!requirements.driver_min_version.empty() &&
            CompareVersions(hardware.driver_version, requirements.driver_min_version) < 0) {
            AddReason(verdict, "compatibility.driver.minVersion",
                      "Requires driver " + requirements.driver_min_version + " or newer; found " +
                          hardware.driver_version + ".");
        }
        if (!requirements.driver_max_version.empty() &&
            CompareVersions(hardware.driver_version, requirements.driver_max_version) > 0) {
            AddReason(verdict, "compatibility.driver.maxVersion",
                      "Requires driver " + requirements.driver_max_version + " or older; found " +
                          hardware.driver_version + ".");
        }
        for (const std::string& bad : requirements.known_bad_driver_versions) {
            if (CompareVersions(hardware.driver_version, bad) == 0) {
                AddReason(verdict, "compatibility.driver.knownBadVersions",
                          "Driver " + bad + " is known to be incompatible with this package.");
            }
        }
    } else if (!requirements.driver_min_version.empty()) {
        verdict.warnings.push_back("Driver version could not be detected; the minimum requirement was not verified.");
    }

    if (!requirements.graphics_apis.empty()) {
        const bool shared = std::any_of(
            requirements.graphics_apis.begin(), requirements.graphics_apis.end(),
            [&hardware](ns_graphics_api api) { return Contains(hardware.graphics_apis, api); });
        if (!shared) {
            AddReason(verdict, "compatibility.graphicsApis",
                      "This runtime does not provide any graphics API the package supports.");
        }
    }

    verdict.compatible = verdict.reasons.empty();
    return verdict;
}

CompatibilityVerdict CompatibilityEngine::Evaluate(
    const PackageRequirements& requirements,
    const HardwareProfile& hardware,
    const SourceProfile& source,
    const GuideLayerCapabilities& guide_layer) const {

    CompatibilityVerdict verdict = EvaluateHardware(requirements, hardware);

    if (!requirements.graphics_apis.empty() && source.api != ns_api_none &&
        !Contains(requirements.graphics_apis, source.api)) {
        AddReason(verdict, "source.graphicsApi",
                  std::string("Source provides ") + ToString(source.api) +
                      " resources, which this package does not accept.");
    }

    if (!source.extent.empty()) {
        if (!requirements.min_resolution.empty() &&
            (source.extent.width < requirements.min_resolution.width ||
             source.extent.height < requirements.min_resolution.height)) {
            AddReason(verdict, "compatibility.resolution.min",
                      "Source " + Describe(source.extent) + " is below the minimum " +
                          Describe(requirements.min_resolution) + ".");
        }
        if (!requirements.max_resolution.empty() &&
            (source.extent.width > requirements.max_resolution.width ||
             source.extent.height > requirements.max_resolution.height)) {
            AddReason(verdict, "compatibility.resolution.max",
                      "Source " + Describe(source.extent) + " exceeds the maximum " +
                          Describe(requirements.max_resolution) + ".");
        }
        if (requirements.resolution_multiple.width > 1 &&
            source.extent.width % requirements.resolution_multiple.width != 0) {
            AddReason(verdict, "compatibility.resolution.multipleOf",
                      "Source width " + std::to_string(source.extent.width) +
                          " is not a multiple of " +
                          std::to_string(requirements.resolution_multiple.width) + ".");
        }
        if (requirements.resolution_multiple.height > 1 &&
            source.extent.height % requirements.resolution_multiple.height != 0) {
            AddReason(verdict, "compatibility.resolution.multipleOf",
                      "Source height " + std::to_string(source.extent.height) +
                          " is not a multiple of " +
                          std::to_string(requirements.resolution_multiple.height) + ".");
        }
    }

    if (source.resolution_varies && !requirements.supports_dynamic_resolution) {
        /* Not fatal: the broker recreates the session on each change. The user
           is told because it costs a history reset every time. */
        verdict.warnings.push_back(
            "Source resolution varies and this package does not support dynamic resolution; "
            "the session will be recreated on every change.");
    }

    if (requirements.hdr_required && !source.hdr) {
        AddReason(verdict, "conventions.hdr.required",
                  "Package requires an HDR source; this source is SDR.");
    }

    for (const GuideRequirement& guide : requirements.required_guides) {
        if (Contains(source.native_guides, guide.kind)) {
            continue;
        }
        if (guide.synthesizable && Contains(guide_layer.synthesizable, guide.kind)) {
            verdict.synthesized_guides.push_back(guide.kind);
            verdict.warnings.push_back(
                std::string("Guide '") + ToString(guide.kind) +
                "' will be generated by the guide layer because the source cannot supply it. "
                "Synthesized guides are not equivalent to native ones.");
            continue;
        }

        std::string detail = std::string("Source cannot supply the required '") +
                             ToString(guide.kind) + "' guide";
        if (!guide.synthesizable) {
            detail += ", and this package does not permit synthesizing it.";
        } else {
            detail += ", and the guide layer cannot generate it.";
        }
        AddReason(verdict, "inputs.required", detail);
    }

    for (const GuideRequirement& guide : requirements.optional_guides) {
        if (!Contains(source.native_guides, guide.kind) &&
            !(guide.synthesizable && Contains(guide_layer.synthesizable, guide.kind))) {
            verdict.warnings.push_back(
                std::string("Optional guide '") + ToString(guide.kind) +
                "' is unavailable; output quality may be reduced.");
        }
    }

    verdict.compatible = verdict.reasons.empty();
    return verdict;
}

}  // namespace neuralstudio
