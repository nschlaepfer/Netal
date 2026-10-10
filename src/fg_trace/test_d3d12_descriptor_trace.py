#!/usr/bin/env python3
"""Static safety contract for the optional D3D12 descriptor tracer."""

from __future__ import annotations

import pathlib
import re
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = pathlib.Path(__file__).with_name("d3d12_descriptor_trace.cpp")
FORWARD = pathlib.Path(__file__).with_name("nvapi_forward_trace.cpp")
HOST = ROOT / "src" / "fg_host" / "main.cpp"
BUILD = ROOT / "scripts" / "build-fg-nvapi-trace.ps1"


class DescriptorTraceContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SOURCE.read_text(encoding="utf-8")
        cls.forward = FORWARD.read_text(encoding="utf-8")
        cls.host = HOST.read_text(encoding="utf-8")
        cls.build = BUILD.read_text(encoding="utf-8")

    def test_capture_is_explicitly_path_gated_and_create_new(self) -> None:
        for name in (
            "NEURALSTUDIO_FG_D3D12_DESCRIPTOR_TRACE",
            "NEURALSTUDIO_FG_D3D12_DESCRIPTOR_VOLATILE_TRACE",
        ):
            self.assertIn(name, self.source)
        self.assertIn("CREATE_NEW", self.source)
        self.assertIn("IsAbsoluteWindowsPath", self.source)
        self.assertIn("g_semantic_path.empty() && g_volatile_path.empty()", self.source)

    def test_no_capture_environment_is_a_noop(self) -> None:
        self.assertRegex(
            self.source,
            r"g_semantic_path\.empty\(\)\s*&&\s*g_volatile_path\.empty\(\)\)\s*return TRUE",
        )
        self.assertIn("if (!g_capture_enabled) return true;", self.source)

    def test_host_attach_is_optional_without_a_proxy(self) -> None:
        self.assertIn("FgNvapiAttachD3D12Device attach_descriptor_trace = nullptr", self.host)
        self.assertIn("if (attach_descriptor_trace != nullptr)", self.host)
        self.assertIn("descriptor_trace_attach=not_available", self.host)
        self.assertIn("ScopedFgDescriptorTraceAttachment", self.host)

    def test_early_attach_precedes_runtime_allocation_capture(self) -> None:
        attach = self.host.index("attach_descriptor_trace(gpu.device.Get())")
        allocation = self.host.index("runtime_allocation_capture.emplace(")
        requirements = self.host.index('set_trace_phase("requirements")')
        self.assertLess(attach, allocation)
        self.assertLess(attach, requirements)

    def test_required_device_methods_are_hooked_and_restored(self) -> None:
        for marker in (
            "HookCreateDescriptorHeap",
            "HookCreateShaderResourceView",
            "HookCreateUnorderedAccessView",
            "HookCreateSampler",
            "HookCopyDescriptors",
            "HookCopyDescriptorsSimple",
            "HookCreateCommittedResource",
            "HookCreatePlacedResource",
            "HookCreateReservedResource",
            "RestoreHooks",
            "InterlockedCompareExchangePointer",
        ):
            self.assertIn(marker, self.source)

    def test_semantic_trace_explicitly_excludes_addresses(self) -> None:
        self.assertIn('AppendBoolean(ready, "addresses_recorded", false)', self.source)
        self.assertIn(
            'AppendBoolean(ready, "gpu_virtual_addresses_recorded", false)',
            self.source,
        )
        self.assertIn(
            'AppendBoolean(event, "gpu_virtual_address_recorded", false)',
            self.source,
        )
        semantic_writers = re.findall(r"WriteSemantic\(([^)]+)\)", self.source)
        self.assertNotIn("raw", semantic_writers)

    def test_raw_cpu_handles_are_confined_to_volatile_events(self) -> None:
        for field in (
            "resource_identity",
            "heap_identity",
            "cpu_descriptor_start",
            "cpu_descriptor_handle",
            "source_cpu_descriptor_handle",
            "sampler_cpu_descriptor_handle",
            "counter_resource_identity",
        ):
            self.assertIn(field, self.source)
        self.assertIn("WriteVolatile(raw)", self.source)
        self.assertNotIn('AppendHex(event, "cpu_descriptor_handle"', self.source)
        self.assertNotIn('AppendHex(event, "resource_identity"', self.source)

    def test_nvapi_requests_are_bound_after_real_success(self) -> None:
        self.assertEqual(
            self.forward.count("DescriptorTraceSnapshotResourceObject("), 4
        )
        self.assertEqual(
            self.forward.count("DescriptorTraceRecordResourceObject("), 4
        )
        for kind in (
            "kMergedTextureSampler",
            "kIndependentSurface",
            "kTexture",
            "kSurface",
        ):
            self.assertIn(kind, self.forward)
        self.assertIn("mapping_resolved", self.source)
        self.assertIn('"snapshot_timing", "pre_real_nvapi_call"', self.source)
        self.assertIn("unresolved_successful_request_count", self.source)

    def test_view_subresource_and_sampler_fields_are_captured(self) -> None:
        for field in (
            "most_detailed_mip",
            "mip_count",
            "mip_slice",
            "first_array_slice",
            "array_size",
            "plane_slice",
            "first_depth_slice",
            "depth_size",
            "filter",
            "address_u",
            "min_lod_bits",
            "max_lod_bits",
        ):
            self.assertIn(f'"{field}"', self.source)

    def test_build_links_both_translation_units_and_checks_exports(self) -> None:
        self.assertIn("d3d12_descriptor_trace.cpp", self.build)
        self.assertIn("d3d12_descriptor_trace.obj", self.build)
        for export in (
            "NeuralStudioFgNvapiTraceAttachD3D12Device",
            "NeuralStudioFgNvapiTraceDetachD3D12Device",
        ):
            self.assertIn(export, self.build)


if __name__ == "__main__":
    unittest.main()
