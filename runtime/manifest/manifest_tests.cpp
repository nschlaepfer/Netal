/*
 * NeuralStudio runtime - JSON and manifest reader tests
 * SPDX-License-Identifier: MIT
 *
 * Parses the two real reference manifests from the repository and runs them
 * through the compatibility engine. This is the integration proof that the
 * ecosystem contract (the schema) actually connects to the runtime's
 * decision-making, not just that each half works alone.
 */

#include "json.h"
#include "manifest_reader.h"
#include "../broker/compatibility.h"
#include "../broker/test_support.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>

namespace {

using namespace neuralstudio;
using nstest::Check;
using nstest::Group;

std::string ReadFile(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) { return std::string(); }
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    return buffer.str();
}

/* The tests are launched with the repository root as the working directory. */
std::string PackagePath(const std::string& id) {
    return "packages/" + id + "/package.json";
}

void TestJsonParser() {
    Group("json parser");

    {
        json::Value value;
        json::ParseError error;
        Check(json::Parse("{}", value, error) && value.is_object(), "parses an empty object");
        Check(json::Parse("[]", value, error) && value.is_array(), "parses an empty array");
    }

    {
        json::Value value;
        json::ParseError error;
        json::Parse(R"({"a":1,"b":true,"c":null,"d":"x","e":[1,2,3]})", value, error);
        Check(value["a"].AsNumber() == 1.0, "reads a number member");
        Check(value["b"].AsBool() == true, "reads a boolean member");
        Check(value["c"].is_null(), "reads a null member");
        Check(value["d"].AsString() == "x", "reads a string member");
        Check(value["e"].size() == 3, "reads an array member");
    }

    {
        json::Value value;
        json::ParseError error;
        json::Parse(R"({"s":"line\nbreak\t\"quoted\""})", value, error);
        Check(value["s"].AsString() == "line\nbreak\t\"quoted\"", "decodes string escapes");
    }

    {
        json::Value value;
        json::ParseError error;
        json::Parse(R"({"u":"A\u00e9\u000a"})", value, error);
        /* \u00e9 is 'e' with acute accent: two UTF-8 bytes. */
        Check(value["u"].AsString().size() == 4, "decodes \\u escapes to UTF-8");
    }

    {
        json::Value value;
        json::ParseError error;
        json::Parse("-1.25e3", value, error);
        Check(value.AsNumber() == -1250.0, "parses negative and exponent numbers");
    }

    {
        json::Value value;
        json::ParseError error;
        Check(!json::Parse(R"({"a":})", value, error), "rejects a missing value");
        Check(!error.ok && error.line >= 1, "reports an error position");
    }

    {
        json::Value value;
        json::ParseError error;
        Check(!json::Parse(R"({"a":1} trailing)", value, error), "rejects trailing characters");
    }

    {
        json::Value value;
        json::ParseError error;
        Check(!json::Parse(R"({"a":1,)", value, error), "rejects an unterminated object");
    }

    {
        /* Duplicate keys keep the last value. */
        json::Value value;
        json::ParseError error;
        json::Parse(R"({"a":1,"a":2})", value, error);
        Check(value["a"].AsNumber() == 2.0, "duplicate keys keep the last value");
    }
}

void TestEnumMapping() {
    Group("manifest enum mapping");

    GpuVendor vendor = GpuVendor::Unknown;
    Check(ParseGpuVendor("nvidia", vendor) && vendor == GpuVendor::Nvidia, "maps a known vendor");
    Check(!ParseGpuVendor("matrox", vendor), "reports an unknown vendor");

    ns_graphics_api api = ns_api_none;
    Check(ParseGraphicsApi("d3d12", api) && api == ns_api_d3d12, "maps a known graphics API");
    Check(!ParseGraphicsApi("metal", api), "reports an unknown graphics API");

    ns_guide_kind kind = ns_guide_color;
    Check(ParseGuideKind("motionVectors", kind) && kind == ns_guide_motion_vectors, "maps a known guide");
    Check(!ParseGuideKind("holograms", kind), "reports an unknown guide");
}

void TestTestPackage() {
    Group("read test package manifest");

    const std::string text = ReadFile(PackagePath("org.neuralstudio.test.passthrough"));
    Check(!text.empty(), "test package manifest is present on disk");
    if (text.empty()) { return; }

    const ManifestReadResult result = ReadManifestText(text, 1);
    Check(result.ok, "reads without fatal errors",
          result.errors.empty() ? "" : result.errors.front());
    Check(result.summary.package_id == "org.neuralstudio.test.passthrough", "reads the package id");
    Check(result.summary.backend_type == "test", "reads the backend type");
    Check(result.summary.redistribution_allowed, "reads redistribution as allowed");
    Check(!result.summary.stateful, "reads the package as stateless");

    const bool has_color = std::any_of(
        result.requirements.required_guides.begin(), result.requirements.required_guides.end(),
        [](const GuideRequirement& guide) { return guide.kind == ns_guide_color; });
    Check(has_color, "requires a color guide");
}

void TestNgxPackage() {
    Group("read NGX package manifest");

    const std::string text = ReadFile(PackagePath("com.nvidia.ngx.dlss-nr"));
    Check(!text.empty(), "NGX package manifest is present on disk");
    if (text.empty()) { return; }

    const ManifestReadResult result = ReadManifestText(text, 1);
    Check(result.ok, "reads without fatal errors",
          result.errors.empty() ? "" : result.errors.front());
    Check(result.summary.package_id == "com.nvidia.ngx.dlss-nr", "reads the package id");
    Check(result.summary.backend_type == "ngx", "reads the backend type");
    Check(!result.summary.redistribution_allowed, "reads redistribution as not allowed");
    Check(!result.summary.adapter_signature_required,
          "reads the development adapter as hash-pinned and unsigned");
    Check(result.summary.runtime_signature_required,
          "reads the separately located NVIDIA runtime signature as required");
    Check(result.summary.stateful, "reads the package as stateful");

    Check(result.requirements.min_architecture == "blackwell", "reads the minimum architecture");
    Check(result.requirements.driver_min_version == "616.56", "reads the minimum driver version");
    Check(result.requirements.min_vram_bytes == 12288ull * 1024ull * 1024ull, "reads the VRAM minimum");
    Check(result.requirements.required_guides.size() == 3,
          "reads three required texture guides (color, depth, motion vectors)");

    auto requires_guide = [&result](ns_guide_kind kind) {
        return std::any_of(result.requirements.required_guides.begin(),
                           result.requirements.required_guides.end(),
                           [kind](const GuideRequirement& guide) { return guide.kind == kind; });
    };
    Check(requires_guide(ns_guide_color) && requires_guide(ns_guide_depth) &&
              requires_guide(ns_guide_motion_vectors),
          "the three required guides are color, depth, and motion vectors");
}

void TestEndToEnd() {
    Group("manifest to compatibility, end to end");

    const std::string text = ReadFile(PackagePath("com.nvidia.ngx.dlss-nr"));
    if (text.empty()) {
        Check(false, "NGX manifest present for end-to-end test");
        return;
    }

    const ManifestReadResult manifest = ReadManifestText(text, 1);
    Check(manifest.ok, "manifest read succeeded");

    HardwareProfile rtx5090;
    rtx5090.platform = OsPlatform::Windows;
    rtx5090.os_version = "10.0.26200";
    rtx5090.vendor = GpuVendor::Nvidia;
    rtx5090.architecture = "blackwell";
    rtx5090.vram_bytes = 32ull * 1024ull * 1024ull * 1024ull;
    rtx5090.gpu_features = {"tensor-cores", "fp16"};
    rtx5090.driver_version = "616.56";
    rtx5090.graphics_apis = {ns_api_d3d12, ns_api_cpu};
    rtx5090.host_adapter_abi_major = 1;

    SourceProfile game;
    game.extent = Extent{3840, 2160};
    game.api = ns_api_d3d12;
    game.hdr = true;
    game.native_guides = {ns_guide_color, ns_guide_depth, ns_guide_motion_vectors};

    const CompatibilityEngine engine;
    const GuideLayerCapabilities no_synthesis;

    const CompatibilityVerdict game_verdict =
        engine.Evaluate(manifest.requirements, rtx5090, game, no_synthesis);
    Check(game_verdict.compatible,
          "the real NGX manifest is compatible with the validated machine and a game source",
          game_verdict.reasons.empty() ? "" : game_verdict.reasons.front().detail);

    SourceProfile video;
    video.extent = Extent{3840, 2160};
    video.api = ns_api_d3d12;
    video.native_guides = {ns_guide_color};

    const CompatibilityVerdict video_verdict =
        engine.Evaluate(manifest.requirements, rtx5090, video, no_synthesis);
    Check(!video_verdict.compatible,
          "the real NGX manifest is incompatible with a plain video source (no motion vectors)");
}

}  // namespace

void TestManifest() {
    TestJsonParser();
    TestEnumMapping();
    TestTestPackage();
    TestNgxPackage();
    TestEndToEnd();
}
