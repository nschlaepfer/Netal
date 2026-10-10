/*
 * NeuralStudio runtime - compatibility engine tests
 * SPDX-License-Identifier: MIT
 *
 * The rules here decide which models a user is even offered. A false negative
 * makes a working model invisible; a false positive sends the user into a
 * failure they were told would work. Both are worth testing exhaustively.
 */

#include "compatibility.h"
#include "test_support.h"

#include <algorithm>

namespace {

using namespace neuralstudio;
using nstest::Check;
using nstest::Group;

bool HasRule(const CompatibilityVerdict& verdict, const std::string& rule) {
    return std::any_of(verdict.reasons.begin(), verdict.reasons.end(),
                       [&rule](const Incompatibility& reason) { return reason.rule == rule; });
}

/* Modeled on the real com.nvidia.ngx.dlss-nr manifest. */
PackageRequirements NgxRequirements() {
    PackageRequirements requirements;
    requirements.package_id = "com.nvidia.ngx.dlss-nr";
    requirements.platform = OsPlatform::Windows;
    requirements.min_os_version = "10.0.19041";
    requirements.vendors = {GpuVendor::Nvidia};
    requirements.min_architecture = "blackwell";
    requirements.min_vram_bytes = 12288ull * 1024ull * 1024ull;
    requirements.required_gpu_features = {"tensor-cores", "fp16"};
    requirements.driver_min_version = "616.56";
    requirements.graphics_apis = {ns_api_d3d12};
    requirements.min_resolution = Extent{640, 360};
    requirements.max_resolution = Extent{7680, 4320};
    requirements.resolution_multiple = Extent{8, 8};
    requirements.supports_dynamic_resolution = false;
    requirements.required_guides = {
        GuideRequirement{ns_guide_color, false},
        GuideRequirement{ns_guide_depth, false},
        GuideRequirement{ns_guide_motion_vectors, false},
    };
    requirements.optional_guides = {
        GuideRequirement{ns_guide_ui_mask, false},
    };
    requirements.adapter_abi_major = 1;
    return requirements;
}

HardwareProfile Rtx5090() {
    HardwareProfile hardware;
    hardware.platform = OsPlatform::Windows;
    hardware.os_version = "10.0.26200";
    hardware.vendor = GpuVendor::Nvidia;
    hardware.architecture = "blackwell";
    hardware.vram_bytes = 32ull * 1024ull * 1024ull * 1024ull;
    hardware.gpu_features = {"tensor-cores", "fp16", "cooperative-vectors"};
    hardware.driver_version = "616.56";
    hardware.graphics_apis = {ns_api_d3d12, ns_api_cpu};
    hardware.host_adapter_abi_major = 1;
    return hardware;
}

SourceProfile GameSource() {
    SourceProfile source;
    source.extent = Extent{3840, 2160};
    source.api = ns_api_d3d12;
    source.hdr = true;
    source.native_guides = {ns_guide_color, ns_guide_depth, ns_guide_motion_vectors};
    return source;
}

SourceProfile VideoSource() {
    SourceProfile source;
    source.extent = Extent{3840, 2160};
    source.api = ns_api_d3d12;
    source.hdr = false;
    source.native_guides = {ns_guide_color};
    return source;
}

void TestVersionComparison() {
    Group("version comparison");

    Check(CompareVersions("616.56", "616.55") > 0, "a higher minor version compares greater");
    Check(CompareVersions("616.56", "616.56") == 0, "equal versions compare equal");
    Check(CompareVersions("616", "616.0") == 0, "missing components are treated as zero");
    Check(CompareVersions("10.0.19041", "10.0.22000") < 0, "build components are compared");
    Check(CompareVersions("616.56-beta", "616.56") == 0, "a build suffix does not affect ordering");
    Check(CompareVersions("9.9", "10.0") < 0, "components compare numerically, not lexically");
}

void TestArchitectureRanking() {
    Group("architecture ranking");

    Check(RankArchitecture(GpuVendor::Nvidia, "blackwell") >
              RankArchitecture(GpuVendor::Nvidia, "ada"),
          "newer NVIDIA architectures rank higher");
    Check(RankArchitecture(GpuVendor::Nvidia, "BLACKWELL") ==
              RankArchitecture(GpuVendor::Nvidia, "blackwell"),
          "architecture names are case-insensitive");
    Check(RankArchitecture(GpuVendor::Nvidia, "rubin") < 0,
          "an unrecognized architecture is unrankable rather than lowest");
}

void TestHardwareRules() {
    Group("hardware compatibility");

    const CompatibilityEngine engine;
    const PackageRequirements requirements = NgxRequirements();

    {
        const CompatibilityVerdict verdict = engine.EvaluateHardware(requirements, Rtx5090());
        Check(verdict.compatible, "a validated machine is compatible",
              verdict.reasons.empty() ? "" : verdict.reasons.front().detail);
    }

    {
        HardwareProfile hardware = Rtx5090();
        hardware.driver_version = "610.12";
        const CompatibilityVerdict verdict = engine.EvaluateHardware(requirements, hardware);
        Check(!verdict.compatible, "an older driver is rejected");
        Check(HasRule(verdict, "compatibility.driver.minVersion"), "the driver rule is named");
    }

    {
        HardwareProfile hardware = Rtx5090();
        hardware.vendor = GpuVendor::Amd;
        hardware.architecture = "rdna4";
        const CompatibilityVerdict verdict = engine.EvaluateHardware(requirements, hardware);
        Check(HasRule(verdict, "compatibility.gpu.vendors"), "a different GPU vendor is rejected");
    }

    {
        HardwareProfile hardware = Rtx5090();
        hardware.vram_bytes = 8ull * 1024ull * 1024ull * 1024ull;
        const CompatibilityVerdict verdict = engine.EvaluateHardware(requirements, hardware);
        Check(HasRule(verdict, "compatibility.gpu.minVramMB"), "insufficient VRAM is rejected");
    }

    {
        HardwareProfile hardware = Rtx5090();
        hardware.gpu_features = {"fp16"};
        const CompatibilityVerdict verdict = engine.EvaluateHardware(requirements, hardware);
        Check(HasRule(verdict, "compatibility.gpu.requiredFeatures"),
              "a missing GPU feature is rejected");
    }

    {
        HardwareProfile hardware = Rtx5090();
        hardware.architecture = "ada";
        const CompatibilityVerdict verdict = engine.EvaluateHardware(requirements, hardware);
        Check(HasRule(verdict, "compatibility.gpu.minArchitecture"),
              "an older architecture is rejected");
    }

    {
        /* Future hardware the rank table has never heard of must warn, not
           block. A stale table should not gate newer GPUs. */
        HardwareProfile hardware = Rtx5090();
        hardware.architecture = "rubin";
        const CompatibilityVerdict verdict = engine.EvaluateHardware(requirements, hardware);
        Check(verdict.compatible, "an unrecognized newer architecture is not blocked");
        Check(!verdict.warnings.empty(), "an unrecognized architecture produces a warning");
    }

    {
        HardwareProfile hardware = Rtx5090();
        hardware.driver_version.clear();
        const CompatibilityVerdict verdict = engine.EvaluateHardware(requirements, hardware);
        Check(verdict.compatible, "an undetectable driver version does not hard-fail");
        Check(!verdict.warnings.empty(), "an undetectable driver version warns");
    }

    {
        PackageRequirements bad_abi = requirements;
        bad_abi.adapter_abi_major = 2;
        const CompatibilityVerdict verdict = engine.EvaluateHardware(bad_abi, Rtx5090());
        Check(HasRule(verdict, "backend.adapter.abiVersion"),
              "an adapter ABI the runtime does not speak is rejected");
    }

    {
        PackageRequirements known_bad = requirements;
        known_bad.known_bad_driver_versions = {"616.56"};
        const CompatibilityVerdict verdict = engine.EvaluateHardware(known_bad, Rtx5090());
        Check(HasRule(verdict, "compatibility.driver.knownBadVersions"),
              "a known-bad driver is rejected even when it satisfies the minimum");
    }
}

void TestSourceRules() {
    Group("source compatibility");

    const CompatibilityEngine engine;
    const PackageRequirements requirements = NgxRequirements();
    const HardwareProfile hardware = Rtx5090();
    const GuideLayerCapabilities no_synthesis;

    {
        const CompatibilityVerdict verdict =
            engine.Evaluate(requirements, hardware, GameSource(), no_synthesis);
        Check(verdict.compatible, "a source supplying every required guide is compatible",
              verdict.reasons.empty() ? "" : verdict.reasons.front().detail);
    }

    {
        /* The headline case: decoded video has no native motion vectors. */
        const CompatibilityVerdict verdict =
            engine.Evaluate(requirements, hardware, VideoSource(), no_synthesis);
        Check(!verdict.compatible, "a source missing a required guide is incompatible");
        Check(HasRule(verdict, "inputs.required"), "the missing-guide rule is named");

        const bool names_guide = std::any_of(
            verdict.reasons.begin(), verdict.reasons.end(), [](const Incompatibility& reason) {
                return reason.detail.find("motionVectors") != std::string::npos;
            });
        Check(names_guide, "the verdict names the specific missing guide");
    }

    {
        PackageRequirements synthesizable = requirements;
        synthesizable.required_guides = {
            GuideRequirement{ns_guide_color, false},
            GuideRequirement{ns_guide_depth, true},
            GuideRequirement{ns_guide_motion_vectors, true},
        };
        GuideLayerCapabilities guide_layer;
        guide_layer.synthesizable = {ns_guide_depth, ns_guide_motion_vectors};

        const CompatibilityVerdict verdict =
            engine.Evaluate(synthesizable, hardware, VideoSource(), guide_layer);
        Check(verdict.compatible, "a synthesizable guide the guide layer can generate is accepted");
        Check(verdict.synthesized_guides.size() == 2, "synthesized guides are reported");
        Check(!verdict.warnings.empty(),
              "synthesis produces a warning rather than implying native quality");
    }

    {
        PackageRequirements synthesizable = requirements;
        synthesizable.required_guides = {
            GuideRequirement{ns_guide_color, false},
            GuideRequirement{ns_guide_motion_vectors, true},
        };
        /* Manifest permits synthesis, but this guide layer cannot do it. */
        const CompatibilityVerdict verdict =
            engine.Evaluate(synthesizable, hardware, VideoSource(), no_synthesis);
        Check(!verdict.compatible,
              "a synthesizable guide the guide layer cannot generate is still incompatible");
    }

    {
        SourceProfile source = GameSource();
        source.extent = Extent{320, 180};
        const CompatibilityVerdict verdict =
            engine.Evaluate(requirements, hardware, source, no_synthesis);
        Check(HasRule(verdict, "compatibility.resolution.min"),
              "a source below the minimum resolution is rejected");
    }

    {
        SourceProfile source = GameSource();
        source.extent = Extent{1921, 1080};
        const CompatibilityVerdict verdict =
            engine.Evaluate(requirements, hardware, source, no_synthesis);
        Check(HasRule(verdict, "compatibility.resolution.multipleOf"),
              "a resolution violating the multiple-of constraint is rejected");
    }

    {
        PackageRequirements hdr_only = requirements;
        hdr_only.hdr_required = true;
        SourceProfile source = GameSource();
        source.hdr = false;
        const CompatibilityVerdict verdict =
            engine.Evaluate(hdr_only, hardware, source, no_synthesis);
        Check(HasRule(verdict, "conventions.hdr.required"),
              "an SDR source is rejected by an HDR-only package");
    }

    {
        SourceProfile source = GameSource();
        source.resolution_varies = true;
        const CompatibilityVerdict verdict =
            engine.Evaluate(requirements, hardware, source, no_synthesis);
        Check(verdict.compatible, "a varying resolution does not block a static-resolution package");
        Check(!verdict.warnings.empty(), "a varying resolution warns about repeated resets");
    }

    {
        /* uiMask is optional in the NGX manifest; the game source does not
           supply it. That must warn, never block. */
        const CompatibilityVerdict verdict =
            engine.Evaluate(requirements, hardware, GameSource(), no_synthesis);
        const bool warns_about_optional = std::any_of(
            verdict.warnings.begin(), verdict.warnings.end(), [](const std::string& warning) {
                return warning.find("uiMask") != std::string::npos;
            });
        Check(warns_about_optional, "a missing optional guide warns but does not block");
    }

    {
        SourceProfile source = GameSource();
        source.api = ns_api_cpu;
        const CompatibilityVerdict verdict =
            engine.Evaluate(requirements, hardware, source, no_synthesis);
        Check(HasRule(verdict, "source.graphicsApi"),
              "a source whose resources the package cannot accept is rejected");
    }
}

}  // namespace

void TestCompatibility() {
    TestVersionComparison();
    TestArchitectureRanking();
    TestHardwareRules();
    TestSourceRules();
}
