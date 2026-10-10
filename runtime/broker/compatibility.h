/*
 * NeuralStudio runtime - compatibility engine
 * SPDX-License-Identifier: MIT
 *
 * Decides whether a package can run on this machine against this source, and
 * when it cannot, says exactly why.
 *
 * The vision requires NeuralStudio to "recommend only compatible models" and to
 * "identify missing guides". Both reduce to this engine. It is deliberately
 * free of GPU, adapter, and IPC dependencies so every rule can be unit tested,
 * because a compatibility bug shows up to the user as a model that silently
 * never appears in the catalog.
 *
 * A verdict is never a bare boolean. Every negative carries a rule name and a
 * user-presentable detail, and every satisfied-by-substitution case carries a
 * warning, because a synthesized guide is not equivalent to a native one.
 */

#ifndef NEURALSTUDIO_RUNTIME_COMPATIBILITY_H
#define NEURALSTUDIO_RUNTIME_COMPATIBILITY_H

#include <cstdint>
#include <string>
#include <vector>

#include "../../spec/adapter/v1/neuralstudio_adapter.h"

namespace neuralstudio {

enum class GpuVendor { Unknown, Nvidia, Amd, Intel, Any };
enum class OsPlatform { Windows, Linux };

const char* ToString(GpuVendor vendor);
const char* ToString(ns_guide_kind kind);
const char* ToString(ns_graphics_api api);

struct Extent {
    std::uint32_t width = 0;
    std::uint32_t height = 0;

    bool empty() const { return width == 0 || height == 0; }
};

struct GuideRequirement {
    ns_guide_kind kind = ns_guide_color;

    /* Whether the manifest permits the host guide layer to generate this input
       when the source cannot supply it natively. */
    bool synthesizable = false;
};

/* The compatibility-relevant subset of a package manifest. */
struct PackageRequirements {
    std::string package_id;

    OsPlatform platform = OsPlatform::Windows;
    std::string min_os_version;

    std::vector<GpuVendor> vendors;
    std::string min_architecture;
    std::uint64_t min_vram_bytes = 0;
    std::vector<std::string> required_gpu_features;

    std::string driver_min_version;
    std::string driver_max_version;
    std::vector<std::string> known_bad_driver_versions;

    std::vector<ns_graphics_api> graphics_apis;

    Extent min_resolution;
    Extent max_resolution;
    Extent resolution_multiple;
    bool supports_dynamic_resolution = false;

    std::vector<GuideRequirement> required_guides;
    std::vector<GuideRequirement> optional_guides;

    bool hdr_required = false;
    std::uint32_t adapter_abi_major = 1;
};

struct HardwareProfile {
    OsPlatform platform = OsPlatform::Windows;
    std::string os_version;

    GpuVendor vendor = GpuVendor::Unknown;
    std::string architecture;
    std::uint64_t vram_bytes = 0;
    std::vector<std::string> gpu_features;
    std::string driver_version;

    std::vector<ns_graphics_api> graphics_apis;
    std::uint32_t host_adapter_abi_major = NS_ADAPTER_ABI_VERSION_MAJOR;
};

struct SourceProfile {
    Extent extent;
    ns_graphics_api api = ns_api_cpu;
    bool hdr = false;
    bool resolution_varies = false;
    std::vector<ns_guide_kind> native_guides;
};

struct GuideLayerCapabilities {
    std::vector<ns_guide_kind> synthesizable;
};

struct Incompatibility {
    std::string rule;
    std::string detail;
};

struct CompatibilityVerdict {
    bool compatible = false;
    std::vector<Incompatibility> reasons;
    std::vector<std::string> warnings;

    /* Guides that will be supplied by the guide layer rather than the source.
       Surfaced so the UI can say so rather than implying native quality. */
    std::vector<ns_guide_kind> synthesized_guides;
};

/*
 * Compares dotted numeric versions such as "616.56" or "10.0.19041".
 * Returns -1 when left < right, 0 when equal, 1 when left > right.
 * Missing components are treated as zero, so "616" == "616.0".
 */
int CompareVersions(const std::string& left, const std::string& right);

/*
 * Ranks a GPU architecture name for minimum-architecture checks. Returns -1 for
 * an unrecognized name, which the engine reports as a warning rather than a
 * hard failure - refusing to run on an architecture the table simply has not
 * heard of yet would age badly.
 */
int RankArchitecture(GpuVendor vendor, const std::string& architecture);

class CompatibilityEngine {
public:
    /* Hardware-only verdict, used to populate the catalog before a source is
       chosen. */
    CompatibilityVerdict EvaluateHardware(const PackageRequirements& requirements,
                                          const HardwareProfile& hardware) const;

    /* Full verdict for a specific pairing. Includes every hardware rule plus
       source resolution, HDR, and guide availability. */
    CompatibilityVerdict Evaluate(const PackageRequirements& requirements,
                                  const HardwareProfile& hardware,
                                  const SourceProfile& source,
                                  const GuideLayerCapabilities& guide_layer) const;
};

}  // namespace neuralstudio

#endif  /* NEURALSTUDIO_RUNTIME_COMPATIBILITY_H */
