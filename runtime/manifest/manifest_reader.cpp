/*
 * NeuralStudio runtime - manifest reader
 * SPDX-License-Identifier: MIT
 */

#include "manifest_reader.h"

#include <algorithm>
#include <cstdint>

namespace neuralstudio {
namespace {

std::uint64_t MegabytesToBytes(double megabytes) {
    if (megabytes <= 0.0) { return 0; }
    return static_cast<std::uint64_t>(megabytes) * 1024ull * 1024ull;
}

Extent ReadExtent(const json::Value& value) {
    Extent extent;
    extent.width = static_cast<std::uint32_t>(value["width"].AsNumber(0.0));
    extent.height = static_cast<std::uint32_t>(value["height"].AsNumber(0.0));
    return extent;
}

}  // namespace

bool ParseGpuVendor(const std::string& text, GpuVendor& out_vendor) {
    if (text == "nvidia") { out_vendor = GpuVendor::Nvidia; return true; }
    if (text == "amd")    { out_vendor = GpuVendor::Amd; return true; }
    if (text == "intel")  { out_vendor = GpuVendor::Intel; return true; }
    if (text == "any")    { out_vendor = GpuVendor::Any; return true; }
    return false;
}

bool ParseGraphicsApi(const std::string& text, ns_graphics_api& out_api) {
    if (text == "d3d12")  { out_api = ns_api_d3d12; return true; }
    if (text == "d3d11")  { out_api = ns_api_d3d11; return true; }
    if (text == "vulkan") { out_api = ns_api_vulkan; return true; }
    if (text == "cuda")   { out_api = ns_api_cuda; return true; }
    if (text == "cpu")    { out_api = ns_api_cpu; return true; }
    return false;
}

bool ParseGuideKind(const std::string& text, ns_guide_kind& out_kind) {
    if (text == "color")         { out_kind = ns_guide_color; return true; }
    if (text == "depth")         { out_kind = ns_guide_depth; return true; }
    if (text == "motionVectors") { out_kind = ns_guide_motion_vectors; return true; }
    if (text == "exposure")      { out_kind = ns_guide_exposure; return true; }
    if (text == "jitter")        { out_kind = ns_guide_color; return false; }  /* jitter is not a texture guide */
    if (text == "camera")        { out_kind = ns_guide_color; return false; }
    if (text == "mask")          { out_kind = ns_guide_mask; return true; }
    if (text == "uiMask")        { out_kind = ns_guide_ui_mask; return true; }
    if (text == "albedo")        { out_kind = ns_guide_albedo; return true; }
    if (text == "normal")        { out_kind = ns_guide_normal; return true; }
    if (text == "roughness")     { out_kind = ns_guide_roughness; return true; }
    if (text == "specular")      { out_kind = ns_guide_specular; return true; }
    if (text == "transparency")  { out_kind = ns_guide_transparency; return true; }
    if (text == "opticalFlow")   { out_kind = ns_guide_optical_flow; return true; }
    return false;
}

bool ParseOsPlatform(const std::string& text, OsPlatform& out_platform) {
    if (text == "windows") { out_platform = OsPlatform::Windows; return true; }
    if (text == "linux")   { out_platform = OsPlatform::Linux; return true; }
    return false;
}

ManifestReadResult ReadManifest(const json::Value& document, std::uint32_t host_abi_major) {
    /* The host ABI is compared inside the compatibility engine, not here; the
       reader only records what the manifest declares. The parameter is kept for
       API symmetry and future use. */
    (void)host_abi_major;

    ManifestReadResult result;
    PackageRequirements& requirements = result.requirements;

    if (!document.is_object()) {
        result.errors.push_back("Manifest root is not a JSON object.");
        return result;
    }

    const std::string schema_version = document["schemaVersion"].AsString();
    if (schema_version.rfind("1.", 0) != 0) {
        result.errors.push_back("Unsupported schemaVersion '" + schema_version +
                                "'. This runtime speaks major version 1.");
        return result;
    }

    /* --- summary --- */
    const json::Value& package = document["package"];
    result.summary.package_id = package["id"].AsString();
    result.summary.name = package["name"].AsString();
    result.summary.version = package["version"].AsString();
    result.summary.publisher = package["publisher"].AsString();
    result.summary.license_id = package["license"]["id"].AsString();
    requirements.package_id = result.summary.package_id;

    if (result.summary.package_id.empty()) {
        result.errors.push_back("package.id is missing.");
    }

    const json::Value& backend = document["backend"];
    result.summary.backend_type = backend["type"].AsString();
    requirements.adapter_abi_major =
        static_cast<std::uint32_t>(backend["adapter"]["abiVersion"].AsNumber(0.0));
    if (requirements.adapter_abi_major == 0) {
        result.errors.push_back("backend.adapter.abiVersion is missing or zero.");
    }

    const json::Value& provenance = document["provenance"];
    result.summary.distribution = provenance["distribution"].AsString();
    result.summary.redistribution_allowed =
        provenance["redistribution"]["allowed"].AsBool(false);
    const std::string adapter_library = backend["adapter"]["library"].AsString();
    for (const json::Value& artifact : provenance["artifacts"].items()) {
        if (artifact["role"].AsString() == "adapter" &&
            artifact["path"].AsString() == adapter_library) {
            result.summary.adapter_signature_required =
                artifact["signature"]["required"].AsBool(false);
            break;
        }
    }
    result.summary.runtime_signature_required =
        backend["runtimeAcquisition"]["signature"]["required"].AsBool(false);

    /* --- compatibility.os --- */
    const json::Value& os_list = document["compatibility"]["os"];
    if (os_list.size() > 0) {
        const json::Value& os = os_list.items().front();
        const std::string platform_text = os["platform"].AsString();
        if (!ParseOsPlatform(platform_text, requirements.platform)) {
            result.errors.push_back("Unrecognized compatibility.os.platform '" + platform_text + "'.");
        }
        requirements.min_os_version = os["minVersion"].AsString();
    } else {
        result.errors.push_back("compatibility.os is empty.");
    }

    /* --- compatibility.gpu --- */
    const json::Value& gpu = document["compatibility"]["gpu"];
    for (const json::Value& vendor_value : gpu["vendors"].items()) {
        const std::string vendor_text = vendor_value.AsString();
        GpuVendor vendor = GpuVendor::Unknown;
        if (ParseGpuVendor(vendor_text, vendor)) {
            requirements.vendors.push_back(vendor);
        } else {
            result.warnings.push_back("Unrecognized GPU vendor '" + vendor_text + "' was ignored.");
        }
    }
    requirements.min_architecture = gpu["minArchitecture"].AsString();
    requirements.min_vram_bytes = MegabytesToBytes(gpu["minVramMB"].AsNumber(0.0));
    for (const json::Value& feature : gpu["requiredFeatures"].items()) {
        requirements.required_gpu_features.push_back(feature.AsString());
    }

    /* --- compatibility.driver --- */
    const json::Value& driver = document["compatibility"]["driver"];
    requirements.driver_min_version = driver["minVersion"].AsString();
    requirements.driver_max_version = driver["maxVersion"].AsString();
    for (const json::Value& bad : driver["knownBadVersions"].items()) {
        requirements.known_bad_driver_versions.push_back(bad.AsString());
    }

    /* --- compatibility.graphicsApis --- */
    for (const json::Value& api_value : document["compatibility"]["graphicsApis"].items()) {
        const std::string api_text = api_value.AsString();
        ns_graphics_api api = ns_api_none;
        if (ParseGraphicsApi(api_text, api)) {
            requirements.graphics_apis.push_back(api);
        } else {
            result.warnings.push_back("Unrecognized graphics API '" + api_text + "' was ignored.");
        }
    }

    /* --- compatibility.resolution --- */
    const json::Value& resolution = document["compatibility"]["resolution"];
    if (resolution.Has("min")) { requirements.min_resolution = ReadExtent(resolution["min"]); }
    if (resolution.Has("max")) { requirements.max_resolution = ReadExtent(resolution["max"]); }
    if (resolution.Has("multipleOf")) {
        requirements.resolution_multiple = ReadExtent(resolution["multipleOf"]);
    }
    requirements.supports_dynamic_resolution = resolution["dynamicResolution"].AsBool(false);

    /* --- inputs --- */
    auto read_guides = [&result](const json::Value& list, std::vector<GuideRequirement>& out) {
        for (const json::Value& guide_value : list.items()) {
            const std::string kind_text = guide_value["kind"].AsString();
            ns_guide_kind kind = ns_guide_color;
            if (!ParseGuideKind(kind_text, kind)) {
                /* jitter and camera are declared in the manifest's guide list
                   but are not texture guides; they are handled through the
                   frame's camera and jitter fields, so they are skipped here
                   rather than reported as errors. */
                if (kind_text != "jitter" && kind_text != "camera") {
                    result.warnings.push_back("Unrecognized guide kind '" + kind_text + "' was ignored.");
                }
                continue;
            }
            GuideRequirement requirement;
            requirement.kind = kind;
            requirement.synthesizable = guide_value["synthesizable"].AsBool(false);
            out.push_back(requirement);
        }
    };
    read_guides(document["inputs"]["required"], requirements.required_guides);
    read_guides(document["inputs"]["optional"], requirements.optional_guides);

    const bool has_color = std::any_of(
        requirements.required_guides.begin(), requirements.required_guides.end(),
        [](const GuideRequirement& guide) { return guide.kind == ns_guide_color; });
    if (!has_color) {
        result.errors.push_back("inputs.required does not contain a color guide.");
    }

    /* --- conventions.hdr --- */
    requirements.hdr_required = document["conventions"]["hdr"]["required"].AsBool(false);

    /* --- temporal --- */
    result.summary.stateful = document["temporal"]["stateful"].AsBool(false);
    if (result.summary.stateful && !document["temporal"].Has("reset")) {
        result.errors.push_back("A stateful package must declare temporal.reset.");
    }

    result.ok = result.errors.empty();
    return result;
}

ManifestReadResult ReadManifestText(const std::string& text, std::uint32_t host_abi_major) {
    json::Value document;
    json::ParseError error;
    if (!json::Parse(text, document, error)) {
        ManifestReadResult result;
        result.errors.push_back("JSON parse error at line " + std::to_string(error.line) +
                                ", column " + std::to_string(error.column) + ": " + error.message);
        return result;
    }
    return ReadManifest(document, host_abi_major);
}

}  // namespace neuralstudio
