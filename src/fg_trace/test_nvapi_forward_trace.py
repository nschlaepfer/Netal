#!/usr/bin/env python3
"""Static contract checks that run on macOS before the Windows build gate."""

from __future__ import annotations

import pathlib
import re
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = pathlib.Path(__file__).with_name("nvapi_forward_trace.cpp")
BUILD = ROOT / "scripts" / "build-fg-nvapi-trace.ps1"


EXPECTED_IDS = {
    "kIdCreateCubinExV2": "299f5fdc",
    "kIdCreateCubin": "2a2c79e8",
    "kIdCreateCubinEx": "3151211b",
    "kIdCreateCubinWithName": "1dc7261f",
    "kIdLaunchCubin": "5c52bb86",
    "kIdDestroyCubin": "7fb785ba",
    "kIdGetMergedTextureSampler": "329fe6e0",
    "kIdGetIndependentDescriptor": "0ddac234",
    "kIdGetTextureObject": "80403fc9",
    "kIdGetSurfaceObject": "48f5b2ee",
    "kIdIsFatbinPtxSupported": "70c07832",
    "kIdCreateCuModule": "ad1a677d",
    "kIdEnumFunctions": "7ab88d88",
    "kIdCreateCuFunction": "e2436e22",
    "kIdLaunchKernelChain": "24973538",
    "kIdLaunchKernelChainEx": "846a9bf0",
    "kIdDestroyCuModule": "41c65285",
    "kIdDestroyCuFunction": "df295ea6",
    "kIdCaptureUavInfo": "6e5ea9db",
}


class ForwardTraceContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SOURCE.read_text(encoding="utf-8")
        cls.build = BUILD.read_text(encoding="utf-8")

    def test_all_pinned_interface_ids_are_exact(self) -> None:
        for name, value in EXPECTED_IDS.items():
            pattern = rf"constexpr\s+std::uint32_t\s+{name}\s*=\s*0x{value}\s*;"
            self.assertRegex(self.source, pattern, name)
        self.assertEqual(
            self.source.count("INTERFACE_BINDING("), len(EXPECTED_IDS) + 1
        )

    def test_both_dispatchers_are_exported_and_forwarded(self) -> None:
        for export in ("nvapi_QueryInterface", "nvapi_Direct_GetMethod"):
            self.assertIn(f'__declspec(dllexport) void* __cdecl {export}', self.source)
            self.assertIn(export, self.build)
        self.assertIn("return returned;", self.source)

    def test_opaque_interface_call_sentinel_preserves_unknown_status(self) -> None:
        self.assertRegex(
            self.source,
            r"constexpr\s+std::uint32_t\s+kIdOpaqueAd298d3f\s*=\s*0xad298d3f\s*;",
        )
        self.assertIn('"opaque_call_sentinel"', self.source)
        decoded_literals = self.source.replace('\\"', '"')
        self.assertIn('"opaque_interface_call_summary"', decoded_literals)
        self.assertIn('"opaque_interface_call_raw"', decoded_literals)
        self.assertIn("signature_inferred", self.source)
        self.assertIn("tail_jump_preserved_arguments", self.source)
        self.assertIn("PAGE_EXECUTE_READ", self.source)
        self.assertNotIn("PAGE_EXECUTE_READWRITE", self.source)

    def test_phase_marker_export_is_bounded_and_serialized(self) -> None:
        export = "NeuralStudioFgNvapiTraceSetPhase"
        self.assertIn(f"void WINAPI {export}", self.source)
        self.assertIn(export, self.build)
        self.assertIn("phase_marker", self.source)
        self.assertIn("kMaxPhaseBytes = 128", self.source)
        self.assertIn("SanitizePhase", self.source)
        self.assertIn("WriteTrace(event)", self.source)

    def test_capture_is_hash_and_path_gated(self) -> None:
        for name in (
            "NEURALSTUDIO_FG_REAL_NVAPI64",
            "NEURALSTUDIO_FG_REAL_NVAPI64_SHA256",
            "NEURALSTUDIO_FG_NVAPI_TRACE",
            "NEURALSTUDIO_FG_NVAPI_VOLATILE_TRACE",
            "NEURALSTUDIO_FG_NVAPI_DUMP_DIR",
        ):
            self.assertIn(name, self.source)
        self.assertIn("ExpectedHashMatches", self.source)
        self.assertIn("CREATE_NEW", self.source)
        self.assertIn("Get-AuthenticodeSignature", self.build)
        self.assertIn("System32", self.build)

    def test_uav_raw_values_are_isolated_in_volatile_trace(self) -> None:
        self.assertIn("struct UavInfoV2", self.source)
        self.assertIn("static_assert(sizeof(UavInfoV2) == 32)", self.source)
        self.assertIn('BeginCall("NvAPI_D3D12_CaptureUAVInfo")', self.source)
        self.assertIn('AppendString(event, "phase", phase)', self.source)
        self.assertIn('AppendToken(event, "uav_info_token"', self.source)
        self.assertIn('AppendUnsigned(event, "version", version)', self.source)
        self.assertIn('AppendUnsigned(event, "gpu_va_size", gpu_va_size)', self.source)
        self.assertIn('AppendUnsigned(event, "out_flags", out_flags)', self.source)
        self.assertNotIn('AppendHex(event, "gpu_va_start"', self.source)
        self.assertNotIn('AppendHex(event, "surface_handle"', self.source)
        self.assertIn('AppendHex(volatile_event, "gpu_va_start"', self.source)
        self.assertIn('AppendUnsigned(volatile_event, "gpu_va_size"', self.source)
        self.assertIn('AppendHex(volatile_event, "surface_handle"', self.source)
        self.assertIn("WriteVolatileTrace(volatile_event)", self.source)

    def test_resource_raw_values_are_isolated_in_volatile_trace(self) -> None:
        self.assertIn('"resource_object_raw"', self.source.replace('\\"', '"'))
        self.assertEqual(self.source.count("WriteRawResourceObject("), 5)
        self.assertIn('AppendHex(volatile_event, "cuda_object_handle"', self.source)
        self.assertIn('"texture_descriptor_ptr"', self.source)
        self.assertIn('"sampler_descriptor_ptr"', self.source)
        self.assertIn('"descriptor_ptr"', self.source)
        self.assertIn('"uav_descriptor_ptr"', self.source)
        self.assertIn('AppendString(volatile_event, "byte_order", "little")', self.source)
        self.assertIn('AppendUnsigned(volatile_event, "cuda_object_handle_bytes"', self.source)
        self.assertNotIn('AppendHex(event, "cuda_object_handle"', self.source)
        self.assertNotIn('AppendHex(event, "texture_descriptor_ptr"', self.source)
        self.assertNotIn('AppendHex(event, "sampler_descriptor_ptr"', self.source)
        self.assertNotIn('AppendHex(event, "descriptor_ptr"', self.source)
        self.assertNotIn('AppendHex(event, "uav_descriptor_ptr"', self.source)
        for call in (
            "NvAPI_D3D12_GetCudaMergedTextureSamplerObject",
            "NvAPI_D3D12_GetCudaIndependentDescriptorObject",
            "NvAPI_D3D12_GetCudaTextureObject",
            "NvAPI_D3D12_GetCudaSurfaceObject",
        ):
            self.assertIn(call, self.source)

    def test_known_passthroughs_are_named_but_not_wrapped(self) -> None:
        for interface_id, name in (
            ("0150e828", "NvAPI_Initialize"),
            ("e5ac921f", "NvAPI_EnumPhysicalGPUs"),
            ("d8265d24", "NvAPI_GPU_GetArchInfo"),
            ("0694d52e", "NvAPI_DRS_CreateSession"),
            ("73bf8338", "NvAPI_DRS_GetSetting"),
            ("dad9cff8", "NvAPI_DRS_DestroySession"),
        ):
            self.assertIn(f'{{0x{interface_id}, "{name}"}}', self.source)
            self.assertNotIn(f'INTERFACE_BINDING(0x{interface_id}', self.source)
        self.assertIn('AppendString(event, "classification", "known_passthrough")', self.source)
        self.assertIn('AppendString(event, "classification", "unknown")', self.source)

    def test_whcp_signature_requires_system32_and_nvidia_identity(self) -> None:
        self.assertIn("Microsoft Windows Hardware Compatibility Publisher", self.build)
        self.assertIn("$isWhcpSigner", self.build)
        self.assertIn("$isExactSystemPath", self.build)
        self.assertIn("$identifiesNvidia", self.build)
        self.assertIn("VersionInfo", self.build)
        self.assertIn("CompanyName", self.build)
        self.assertIn("ProductName", self.build)
        self.assertRegex(
            self.build,
            r"\$isWhcpSigner\s+-and\s+\$isExactSystemPath\s+-and\s+\$identifiesNvidia",
        )
        self.assertNotRegex(self.build, r"(?i)-match\s+['\"]Microsoft['\"]")

    def test_normalized_trace_has_no_pointer_formatter(self) -> None:
        self.assertNotIn("%p", self.source)
        self.assertNotRegex(self.source, r'AppendUnsigned\([^\n]+descriptor')
        self.assertNotIn('AppendUnsigned(event, "gpu_va_start"', self.source)
        self.assertNotIn('AppendUnsigned(event, "surface_handle"', self.source)

    def test_module_and_parameter_bytes_are_dumped(self) -> None:
        for marker in (
            '"legacy-cubin"',
            '"cu-module"',
            ".params.bin",
            'AppendDump(event, "module_blob"',
            'AppendDump(event, "parameter"',
        ):
            self.assertIn(marker, self.source)

    def test_build_is_reproducible_x64_and_checks_exports(self) -> None:
        for marker in ("vcvars64.bat", "/Brepro", "/WX", "dumpbin", "machine \\(x64\\)"):
            self.assertIn(marker, self.build)


if __name__ == "__main__":
    unittest.main()
