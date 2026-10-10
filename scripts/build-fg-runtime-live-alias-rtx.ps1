[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string] $JobRoot,
    [Parameter(Mandatory = $true)] [string] $AttemptId,
    [string] $RepositoryRoot
)

$ErrorActionPreference = 'Stop'
if (-not $RepositoryRoot) {
    $RepositoryRoot = Join-Path $JobRoot 'input\full-source'
}
$attempt = Join-Path $JobRoot (Join-Path 'output' $AttemptId)
if (Test-Path -LiteralPath $attempt) {
    throw "Attempt exists; preserve it: $attempt"
}
$modules = Join-Path $JobRoot 'input\full-modules'
$model = Join-Path $JobRoot 'input\full-model'
$earlyRoot = Join-Path $RepositoryRoot 'src\fg_production_contract_l3_l87'
$tailRoot = Join-Path $RepositoryRoot 'src\fg_connected_l20_l87'
$earlySource = Join-Path $earlyRoot 'rtx_temporal.cpp'
$runtimeRoot = Join-Path $RepositoryRoot 'src\fg_runtime'
$liveAux = Join-Path $runtimeRoot 'rtx_live_aux_bridge.cpp'
$livePrefix = Join-Path $earlyRoot 'rtx_live_prefix_bridge.cpp'
$temporalHandoff = Join-Path $earlyRoot 'rtx_temporal_handoff.cpp'
$tailSource = Join-Path $tailRoot 'rtx_mid_l51_l56_smoke.cpp'
$liveCore = Join-Path $tailRoot 'rtx_live_core_bridge.cpp'
foreach ($required in @($earlySource, $livePrefix, $temporalHandoff, $liveAux,
                         $tailSource, $liveCore,
                         $modules, $model)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required connected source/input missing: $required"
    }
}

$cuda = $env:CUDA_PATH
if (-not $cuda) {
    $cuda = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3'
}
$vswhere = Join-Path ${env:ProgramFiles(x86)} `
    'Microsoft Visual Studio\Installer\vswhere.exe'
$installation = & $vswhere -latest -products '*' `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath
if (-not $installation) { throw 'Visual Studio C++ x64 tools unavailable' }
$vcvars = Join-Path $installation 'VC\Auxiliary\Build\vcvars64.bat'
$build = Join-Path $attempt 'build'
New-Item -ItemType Directory -Path $build | Out-Null
$early = Join-Path $build 'rtx_live_alias_temporal.cpp'
function Replace-ExactlyOnce([string] $Text, [string] $Old, [string] $New,
                             [string] $Label) {
    if (($Text.Split(@($Old), [StringSplitOptions]::None).Count - 1) -ne 1) {
        throw "Live-alias source anchor drift: $Label"
    }
    return $Text.Replace($Old, $New)
}
$earlyText = [IO.File]::ReadAllText($earlySource)
$earlyText = Replace-ExactlyOnce $earlyText `
    '#include "rtx_temporal_handoff.h"' `
    "#include `"rtx_temporal_handoff.h`"`r`n#include `"../fg_runtime/rtx_live_aux_bridge.h`"" `
    'early include'
$textureAnchor = @'
    auto source_reset=UploadTypedTexture(gpu,DXGI_FORMAT_R8G8B8A8_UNORM,640,360,4,
'@
$sharedHelper = @'
    auto make_shared_texture=[&](DXGI_FORMAT format,std::uint32_t w,
                                 std::uint32_t h,std::uint32_t bpp,
                                 D3D12_RESOURCE_STATES state,
                                 const std::vector<std::uint8_t>& bytes){
      return connected::UploadSharedTexture(gpu,format,w,h,bpp,state,bytes);
    };
    auto source_reset=UploadTypedTexture(gpu,DXGI_FORMAT_R8G8B8A8_UNORM,640,360,4,
'@
$earlyText = Replace-ExactlyOnce $earlyText $textureAnchor $sharedHelper 'shared helper'
$earlyPairs = @(
    @('    auto history_a=make_texture(DXGI_FORMAT_R16G16B16A16_FLOAT,640,360,8,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,connected::InitialHistory());',
      "    auto history_a_shared=make_shared_texture(DXGI_FORMAT_R16G16B16A16_FLOAT,640,360,8,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,connected::InitialHistory());`r`n    GpuTexture& history_a=history_a_shared.d3d;"),
    @('    auto history_b=make_texture(DXGI_FORMAT_R16G16B16A16_FLOAT,640,360,8,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,std::vector<std::uint8_t>(connected::kHistoryBytes,poison));',
      "    auto history_b_shared=make_shared_texture(DXGI_FORMAT_R16G16B16A16_FLOAT,640,360,8,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,std::vector<std::uint8_t>(connected::kHistoryBytes,poison));`r`n    GpuTexture& history_b=history_b_shared.d3d;"),
    @('    auto flow=make_texture(DXGI_FORMAT_R32G32_FLOAT,320,180,8,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,std::vector<std::uint8_t>(connected::kDepthBytes*2u,poison));',
      "    auto flow_shared=make_shared_texture(DXGI_FORMAT_R32G32_FLOAT,320,180,8,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,std::vector<std::uint8_t>(connected::kDepthBytes*2u,poison));`r`n    GpuTexture& flow=flow_shared.d3d;"),
    @('    auto reset_depth=make_texture(DXGI_FORMAT_R32_UINT,320,180,4,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,std::vector<std::uint8_t>(connected::kDepthBytes,poison));',
      "    auto reset_depth_shared=make_shared_texture(DXGI_FORMAT_R32_UINT,320,180,4,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,std::vector<std::uint8_t>(connected::kDepthBytes,poison));`r`n    GpuTexture& reset_depth=reset_depth_shared.d3d;"),
    @('    auto generated_depth=make_texture(DXGI_FORMAT_R32_UINT,320,180,4,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,connected::InitialGeneratedDepth());',
      "    auto generated_depth_shared=make_shared_texture(DXGI_FORMAT_R32_UINT,320,180,4,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,connected::InitialGeneratedDepth());`r`n    GpuTexture& generated_depth=generated_depth_shared.d3d;"),
    @('    auto working_vector=make_texture(DXGI_FORMAT_R16G16_FLOAT,320,180,4,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,std::vector<std::uint8_t>(connected::kVectorBytes,poison));',
      "    auto working_vector_shared=make_shared_texture(DXGI_FORMAT_R16G16_FLOAT,320,180,4,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,std::vector<std::uint8_t>(connected::kVectorBytes,poison));`r`n    GpuTexture& working_vector=working_vector_shared.d3d;"),
    @('    auto reset_projection=make_texture(DXGI_FORMAT_R16G16_FLOAT,320,180,4,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,std::vector<std::uint8_t>(connected::kVectorBytes,poison));',
      "    auto reset_projection_shared=make_shared_texture(DXGI_FORMAT_R16G16_FLOAT,320,180,4,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,std::vector<std::uint8_t>(connected::kVectorBytes,poison));`r`n    GpuTexture& reset_projection=reset_projection_shared.d3d;"),
    @('    auto auxiliary_mask=make_texture(DXGI_FORMAT_R8_UNORM,320,180,1,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,std::vector<std::uint8_t>(connected::kAuxMaskBytes,poison));',
      "    auto auxiliary_mask_shared=make_shared_texture(DXGI_FORMAT_R8_UNORM,320,180,1,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,std::vector<std::uint8_t>(connected::kAuxMaskBytes,poison));`r`n    GpuTexture& auxiliary_mask=auxiliary_mask_shared.d3d;"),
    @('    auto auxiliary_vector=make_texture(DXGI_FORMAT_R16G16_FLOAT,320,180,4,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,std::vector<std::uint8_t>(connected::kAuxVectorBytes,poison));',
      "    auto auxiliary_vector_shared=make_shared_texture(DXGI_FORMAT_R16G16_FLOAT,320,180,4,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,std::vector<std::uint8_t>(connected::kAuxVectorBytes,poison));`r`n    GpuTexture& auxiliary_vector=auxiliary_vector_shared.d3d;")
)
for ($index = 0; $index -lt $earlyPairs.Count; $index++) {
    $earlyText = Replace-ExactlyOnce $earlyText $earlyPairs[$index][0] `
        $earlyPairs[$index][1] "early shared texture $index"
}
$earlyText = $earlyText.Replace(
    'auto auxiliary_mask_frame1_bytes=ReadTexture(gpu,auxiliary_mask);',
    'auto auxiliary_mask_frame1_bytes=snapshot_texture(auxiliary_mask,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);')
$earlyText = $earlyText.Replace(
    'auto auxiliary_vector_frame1_bytes=ReadTexture(gpu,auxiliary_vector);',
    'auto auxiliary_vector_frame1_bytes=snapshot_texture(auxiliary_vector,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);')
$earlyText = $earlyText.Replace(
    'auto auxiliary_mask_frame2_bytes=ReadTexture(gpu,auxiliary_mask);',
    'auto auxiliary_mask_frame2_bytes=snapshot_texture(auxiliary_mask,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);')
$earlyText = $earlyText.Replace(
    'auto auxiliary_vector_frame2_bytes=ReadTexture(gpu,reset_projection);',
    'auto auxiliary_vector_frame2_bytes=snapshot_texture(reset_projection,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);')
$earlyText = $earlyText.Replace(
    'auto aux_mask_bytes=ReadTexture(gpu,auxiliary_mask);',
    'auto aux_mask_bytes=snapshot_texture(auxiliary_mask,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);')
$earlyText = $earlyText.Replace(
    'auto projection_bytes=ReadTexture(gpu,current_projection);',
    'auto projection_bytes=snapshot_texture(current_projection,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);')
$earlyText = Replace-ExactlyOnce $earlyText `
    '    auto run_tail=[&](const std::string& frame_name){' `
    @'
    auto run_tail=[&](const std::string& frame_name,
                       HANDLE current_depth_handle,
                       HANDLE previous_depth_handle,
                       HANDLE previous_history_handle,
                       HANDLE next_history_handle,
                       HANDLE current_projection_handle){
'@ `
    'run-tail signature'
$allocationSetter = @'
      FgConnectedSetRtxAllocation29Handle(reinterpret_cast<std::uint64_t>(
          allocation29_shared.shared_handle));
'@
$auxiliarySetter = @'
      FgConnectedSetRtxAllocation29Handle(reinterpret_cast<std::uint64_t>(
          allocation29_shared.shared_handle));
      const NSFGRtxLiveAuxiliaryHandles live_handles{
          reinterpret_cast<std::uint64_t>(auxiliary_mask_shared.shared_handle),
          reinterpret_cast<std::uint64_t>(flow_shared.shared_handle),
          reinterpret_cast<std::uint64_t>(working_vector_shared.shared_handle),
          reinterpret_cast<std::uint64_t>(current_depth_handle),
          reinterpret_cast<std::uint64_t>(previous_depth_handle),
          reinterpret_cast<std::uint64_t>(previous_history_handle),
          reinterpret_cast<std::uint64_t>(next_history_handle),
          reinterpret_cast<std::uint64_t>(current_projection_handle)};
      NSFGSetRtxLiveAuxiliaryHandles(&live_handles);
'@
$earlyText = Replace-ExactlyOnce $earlyText $allocationSetter $auxiliarySetter `
    'live auxiliary setter'
$earlyText = Replace-ExactlyOnce $earlyText `
    '      FgConnectedSetRtxAllocation29Handle(0);' `
    "      FgConnectedSetRtxAllocation29Handle(0);`r`n      NSFGSetRtxLiveAuxiliaryHandles(nullptr);" `
    'live auxiliary clear'
$earlyText = Replace-ExactlyOnce $earlyText `
    '    run_tail("generated-1");' `
    @'
    run_tail("generated-1", generated_depth_shared.shared_handle,
             reset_depth_shared.shared_handle, history_b_shared.shared_handle,
             history_a_shared.shared_handle, auxiliary_vector_shared.shared_handle);
'@ `
    'frame1 live call'
$earlyText = Replace-ExactlyOnce $earlyText `
    '    run_tail("generated-2");' `
    @'
    run_tail("generated-2", reset_depth_shared.shared_handle,
             generated_depth_shared.shared_handle, history_a_shared.shared_handle,
             history_b_shared.shared_handle, reset_projection_shared.shared_handle);
'@ `
    'frame2 live call'
$earlyText = Replace-ExactlyOnce $earlyText `
    '      run_tail("generated-"+std::to_string(frame));' `
    @'
      run_tail("generated-"+std::to_string(frame),
               odd?generated_depth_shared.shared_handle:reset_depth_shared.shared_handle,
               odd?reset_depth_shared.shared_handle:generated_depth_shared.shared_handle,
               odd?history_b_shared.shared_handle:history_a_shared.shared_handle,
               odd?history_a_shared.shared_handle:history_b_shared.shared_handle,
               odd?auxiliary_vector_shared.shared_handle:reset_projection_shared.shared_handle);
'@ `
    'long live call'
$releaseAnchor = @'
    model_input_shared.Release();
    allocation29_shared.Release();
'@
$releaseReplacement = @'
    model_input_shared.Release();
    allocation29_shared.Release();
    history_a_shared.Release(); history_b_shared.Release();
    flow_shared.Release(); reset_depth_shared.Release();
    generated_depth_shared.Release(); working_vector_shared.Release();
    reset_projection_shared.Release(); auxiliary_mask_shared.Release();
    auxiliary_vector_shared.Release();
'@
$earlyText = Replace-ExactlyOnce $earlyText $releaseAnchor $releaseReplacement `
    'shared handle release'
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[IO.File]::WriteAllText($early, $earlyText, $utf8NoBom)
$generatedTail = Join-Path $build 'rtx_full_tail_live_main.cpp'
$tailText = [IO.File]::ReadAllText($tailSource)
$needle = 'int main(int argc, char** argv) {'
if (($tailText.Split(@($needle), [StringSplitOptions]::None).Count - 1) -ne 1) {
    throw 'Accepted full-tail main signature drifted'
}
$tailText = $tailText.Replace(
    $needle, 'int FgConnectedFullTailMain(int argc, char** argv) {')
$tailText = "#include `"rtx_temporal_handoff.h`"`r`n#include `"../fg_runtime/rtx_live_aux_bridge.cpp`"`r`n" + $tailText
$fixtureBlock = @'
    const GeneratedInputs l55_external;
    const auto l52_depth = BinaryDepth();
    const std::array<std::vector<std::uint8_t>, 4> l53_external = {
        l53::CurrToPrevFlow(), l53::PrevToCurrFlow(false),
        l53::CurrDepth(), l53::PrevDepth()};

'@
$tailText = Replace-ExactlyOnce $tailText $fixtureBlock '' 'tail fixture removal'
$gpuAnchor = @'
    GpuContext gpu = CreateGpuContext();
'@
$gpuReplacement = @'
    GpuContext gpu = CreateGpuContext();
    const auto* live = NSFGGetRtxLiveAuxiliaryHandles();
    if (live == nullptr) throw std::runtime_error("live auxiliary handles missing");
    auto open_texture = [&](std::uint64_t raw_handle, DXGI_FORMAT format,
                            std::uint32_t width, std::uint32_t height,
                            std::uint32_t bytes_per_pixel,
                            D3D12_RESOURCE_STATES state) {
      if (raw_handle == 0) throw std::runtime_error("null live auxiliary handle");
      GpuTexture texture;
      texture.width = width; texture.height = height;
      texture.bytes_per_pixel = bytes_per_pixel;
      texture.format = format; texture.state = state;
      RequireHr(gpu.device->OpenSharedHandle(reinterpret_cast<HANDLE>(raw_handle),
                  IID_PPV_ARGS(&texture.resource)),
                "OpenSharedHandle(live auxiliary)");
      return texture;
    };
'@
$tailText = Replace-ExactlyOnce $tailText $gpuAnchor $gpuReplacement `
    'tail live open helper'
$tailPairs = @(
    @(@'
    GpuTexture l52_depth_texture = UploadTexture(
        gpu, DXGI_FORMAT_R8_UNORM, 320, 180, 1,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, l52_depth);
'@, @'
    GpuTexture l52_depth_texture = open_texture(
        live->auxiliary_mask, DXGI_FORMAT_R8_UNORM, 320, 180, 1,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
'@),
    @(@'
    GpuTexture l53_flow_a = UploadTexture(
        gpu, DXGI_FORMAT_R32G32_FLOAT, 320, 180, 8,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, l53_external[0]);
    GpuTexture l53_flow_b = UploadTexture(
        gpu, DXGI_FORMAT_R16G16_FLOAT, 320, 180, 4,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, l53_external[1]);
    GpuTexture l53_depth_a = UploadTexture(
        gpu, DXGI_FORMAT_R32_UINT, 320, 180, 4,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, l53_external[2]);
    GpuTexture l53_depth_b = UploadTexture(
        gpu, DXGI_FORMAT_R32_UINT, 320, 180, 4,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, l53_external[3]);
'@, @'
    GpuTexture l53_flow_a = open_texture(
        live->flow, DXGI_FORMAT_R32G32_FLOAT, 320, 180, 8,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    GpuTexture l53_flow_b = open_texture(
        live->working_vector, DXGI_FORMAT_R16G16_FLOAT, 320, 180, 4,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    GpuTexture l53_depth_a = open_texture(
        live->current_depth, DXGI_FORMAT_R32_UINT, 320, 180, 4,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    GpuTexture l53_depth_b = open_texture(
        live->previous_depth, DXGI_FORMAT_R32_UINT, 320, 180, 4,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
'@),
    @(@'
    GpuTexture l54_blend = UploadTexture(
        gpu, DXGI_FORMAT_R16G16_FLOAT, 320, 180, 4,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        std::vector<std::uint8_t>(kRgTextureBytes, poison));
'@, @'
    GpuTexture l54_blend = open_texture(
        live->working_vector, DXGI_FORMAT_R16G16_FLOAT, 320, 180, 4,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
'@),
    @(@'
    GpuTexture rgba_a = UploadTexture(
        gpu, DXGI_FORMAT_R16G16B16A16_FLOAT, kActiveWidth, kActiveHeight, 8,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        l55_external.rgba_a);
    GpuTexture rgba_b = UploadTexture(
        gpu, DXGI_FORMAT_R16G16B16A16_FLOAT, kActiveWidth, kActiveHeight, 8,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        l55_external.rgba_b);
    GpuTexture motion = UploadTexture(
        gpu, DXGI_FORMAT_R16G16_FLOAT, kCandidateWidth, kCandidateHeight, 4,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        l55_external.motion);
'@, @'
    GpuTexture rgba_a = open_texture(
        live->previous_history, DXGI_FORMAT_R16G16B16A16_FLOAT,
        kActiveWidth, kActiveHeight, 8,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    GpuTexture rgba_b = open_texture(
        live->next_history, DXGI_FORMAT_R16G16B16A16_FLOAT,
        kActiveWidth, kActiveHeight, 8,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    GpuTexture motion = open_texture(
        live->current_projection, DXGI_FORMAT_R16G16_FLOAT,
        kCandidateWidth, kCandidateHeight, 4,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
'@)
)
for ($index = 0; $index -lt $tailPairs.Count; $index++) {
    $tailText = Replace-ExactlyOnce $tailText $tailPairs[$index][0] `
        $tailPairs[$index][1] "tail live texture $index"
}
$l54Anchor = @'
    Restore(gpu, l53_b, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    ExecuteAndReset(gpu);

    std::array<std::uint8_t, 88> p54{};
'@
$l54Replacement = @'
    Restore(gpu, l53_b, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    Restore(gpu, l54_blend, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    ExecuteAndReset(gpu);

    std::array<std::uint8_t, 88> p54{};
'@
$tailText = Replace-ExactlyOnce $tailText $l54Anchor $l54Replacement `
    'L53 read before L54 overwrite'
$removeMismatch = @(
    '    input_mismatches += Mismatches(l52_depth, ReadTexture(gpu, l52_depth_texture));',
    '    input_mismatches += Mismatches(l53_external[0], ReadTexture(gpu,l53_flow_a));',
    '    input_mismatches += Mismatches(l53_external[1], ReadTexture(gpu,l53_flow_b));',
    '    input_mismatches += Mismatches(l53_external[2], ReadTexture(gpu,l53_depth_a));',
    '    input_mismatches += Mismatches(l53_external[3], ReadTexture(gpu,l53_depth_b));',
    '    input_mismatches += Mismatches(l54_bytes, ReadTexture(gpu, l54_blend));',
    '    input_mismatches += Mismatches(l55_external.rgba_a, ReadTexture(gpu, rgba_a));',
    '    input_mismatches += Mismatches(l55_external.rgba_b, ReadTexture(gpu, rgba_b));',
    '    input_mismatches += Mismatches(l55_external.motion, ReadTexture(gpu, motion));'
)
foreach ($line in $removeMismatch) {
    $tailText = Replace-ExactlyOnce $tailText $line '' `
        ('remove fixture mismatch: ' + $line)
}
$allocationNeedle = @'
    GpuTexture l52_confidence = UploadTexture(
        gpu, DXGI_FORMAT_R16_FLOAT, 320, 180, 2,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        std::vector<std::uint8_t>(kL52ScalarBytes, poison));
'@
$allocationReplacement = @'
    GpuTexture l52_confidence;
    l52_confidence.width = 320;
    l52_confidence.height = 180;
    l52_confidence.bytes_per_pixel = 2;
    l52_confidence.format = DXGI_FORMAT_R16_FLOAT;
    l52_confidence.state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    const auto allocation29_handle = reinterpret_cast<HANDLE>(
        FgConnectedRtxAllocation29Handle());
    if (allocation29_handle == nullptr)
      throw std::runtime_error("live allocation29 binding missing");
    RequireHr(gpu.device->OpenSharedHandle(
                  allocation29_handle,
                  IID_PPV_ARGS(&l52_confidence.resource)),
              "OpenSharedHandle(allocation29)");
    Transition(gpu.list.Get(), l52_confidence.resource.Get(),
               l52_confidence.state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    l52_confidence.state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
'@
if (($tailText.Split(@($allocationNeedle),
                     [StringSplitOptions]::None).Count - 1) -ne 1) {
    throw 'Accepted allocation29 tail anchor drifted'
}
$tailText = $tailText.Replace($allocationNeedle, $allocationReplacement)
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[IO.File]::WriteAllText($generatedTail, $tailText, $utf8NoBom)

$earlyObject = Join-Path $build 'early.obj'
$prefixObject = Join-Path $build 'live-prefix.obj'
$tailObject = Join-Path $build 'full-tail.obj'
$coreObject = Join-Path $build 'live-core.obj'
$handoffObject = Join-Path $build 'temporal-handoff.obj'
$binary = Join-Path $build 'fg-connected-l3-l87.exe'
$compileLog = Join-Path $build 'compile.log'
$command = '"{0}" >nul && cl.exe /nologo /std:c++20 /EHsc /MD /W4 /WX /permissive- /Zc:__cplusplus /utf-8 /DFG_CONNECTED_FULL_CHAIN /I"{1}" /I"{13}" /I"{16}" /Fo"{2}" /c "{3}" >"{4}" 2>&1 && cl.exe /nologo /std:c++20 /EHsc /MD /W4 /WX /permissive- /Zc:__cplusplus /utf-8 /I"{1}" /I"{13}" /I"{16}" /Fo"{5}" /c "{6}" >>"{4}" 2>&1 && cl.exe /nologo /std:c++20 /EHsc /MD /W4 /WX /permissive- /Zc:__cplusplus /utf-8 /I"{1}" /I"{13}" /I"{16}" /Fo"{7}" /c "{8}" >>"{4}" 2>&1 && cl.exe /nologo /std:c++20 /EHsc /MD /W4 /WX /permissive- /Zc:__cplusplus /utf-8 /I"{1}" /I"{13}" /I"{16}" /Fo"{9}" /c "{10}" >>"{4}" 2>&1 && cl.exe /nologo /std:c++20 /EHsc /MD /W4 /WX /permissive- /Zc:__cplusplus /utf-8 /I"{1}" /I"{13}" /I"{16}" /Fo"{14}" /c "{15}" >>"{4}" 2>&1 && link.exe /nologo /OUT:"{11}" "{2}" "{5}" "{7}" "{9}" "{14}" "{12}" bcrypt.lib d3d12.lib dxgi.lib >>"{4}" 2>&1' -f `
    $vcvars, (Join-Path $cuda 'include'), $earlyObject, $early,
    $compileLog, $prefixObject, $livePrefix, $tailObject, $generatedTail,
    $coreObject, $liveCore, $binary, (Join-Path $cuda 'lib\x64\cuda.lib'),
    $tailRoot, $handoffObject, $temporalHandoff, $earlyRoot
& $env:ComSpec /d /s /c $command
if ($LASTEXITCODE -ne 0) { throw "Build failed; see $compileLog" }

$cases = @(
    [ordered]@{ id = 'fg-production-contract-l3-l87-poison-a5-process-a'; poison = 'a5' },
    [ordered]@{ id = 'fg-production-contract-l3-l87-poison-a5-process-b'; poison = 'a5' },
    [ordered]@{ id = 'fg-production-contract-l3-l87-poison-5a-process-a'; poison = '5a' },
    [ordered]@{ id = 'fg-production-contract-l3-l87-poison-5a-process-b'; poison = '5a' }
)
$processes = @()
foreach ($case in $cases) {
    $run = Join-Path $attempt $case.id
    New-Item -ItemType Directory -Path $run | Out-Null
    $process = Start-Process -FilePath $binary `
        -ArgumentList @($modules, $model, $run, $case.id, $case.poison) `
        -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput (Join-Path $run 'stdout.log') `
        -RedirectStandardError (Join-Path $run 'stderr.log')
    $processes += [ordered]@{run_id=$case.id;process_id=$process.Id;exit_code=$process.ExitCode}
    if ($process.ExitCode -ne 0) {
        [ordered]@{schema='neuralstudio.fg.production-contract-l3-l87-rig-failure.v1';attempt_id=$AttemptId;failed_run=$case.id;process_id=$process.Id;exit_code=$process.ExitCode;remote_job_retained=$JobRoot;result='fail'} |
            ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $attempt 'rig-failure.json') -Encoding ascii
        throw "Temporal run failed: $($case.id)"
    }
}

$allCheckpoints=@();$finalHashes=@();$runPids=@()
foreach($case in $cases){
    $run=Join-Path $attempt $case.id
    $earlyReceipt=Get-Content -LiteralPath (Join-Path $run 'run-receipt.json') -Raw|ConvertFrom-Json
    $tails=@(1..15|ForEach-Object{Get-Content -LiteralPath (Join-Path $run ("generated-$_\l20-l87\run-receipt.json")) -Raw|ConvertFrom-Json})
    $processId=($processes|Where-Object run_id -eq $case.id).process_id
    if(@($tails|Where-Object process_id -ne $earlyReceipt.process_id).Count -ne 0 -or $earlyReceipt.process_id -ne $processId){throw "Same-process proof failed: $($case.id)"}
    if($earlyReceipt.result -ne 'pass' -or @($tails|Where-Object result -ne 'pass').Count -ne 0){throw "Receipt failed: $($case.id)"}
    foreach($key in @('input_mismatched_bytes','canary_mismatched_bytes')){if($earlyReceipt.$key -ne 0){throw "Early integrity: $key"}}
    foreach($tail in $tails){foreach($key in @('input_model_mismatched_bytes','canary_mismatched_bytes','readonly_tail_mismatched_bytes')){if($tail.$key -ne 0){throw "Tail integrity: $key"}}}
    if(-not $earlyReceipt.l20_live_handoff -or $earlyReceipt.allocation37_poison_halfwords -ne 0 -or $earlyReceipt.allocation29_poison_halfwords -ne 0){throw 'Temporal coverage/handoff failed'}
    $combined=[ordered]@{}
    foreach($p in $earlyReceipt.checkpoint_sha256.PSObject.Properties){$combined[$p.Name]=$p.Value}
    for($index=0;$index -lt $tails.Count;$index++){foreach($p in $tails[$index].checkpoint_sha256.PSObject.Properties){$combined['generated-'+($index+1)+'-'+$p.Name]=$p.Value}}
    if(@($combined.Keys).Count -ne 1149){throw 'Temporal checkpoint count failed'}
    $receipt=[ordered]@{schema='neuralstudio.fg.production-contract-l3-l87-run.v1';backend='rtx-cuda-external-memory-d3d12-nvapi';run_id=$case.id;process_id=$earlyReceipt.process_id;poison_byte=$earlyReceipt.poison_byte;rendered_frame_count=16;phase_launch_counts=@(9,1140);launch_count=1149;checkpoint_sha256=$combined;allocation37_sha256=@($earlyReceipt.allocation37_sha256);allocation29_sha256=@($earlyReceipt.allocation29_sha256);live_device_handoff_l19_l20=$true;host_intermediate_ingress=$false;expected_or_archived_tensor_ingress=$false;address_fields_recorded=$false;final_output_bytes=@($tails|ForEach-Object final_output_bytes);final_output_sha256=@($tails|ForEach-Object final_output_sha256);result='pass'}
    [IO.File]::WriteAllText((Join-Path $run 'temporal-receipt.json'),(($receipt|ConvertTo-Json -Depth 8)+"`n"),$utf8NoBom)
    $allCheckpoints+=($combined|ConvertTo-Json -Compress)
    $finalHashes+=(($receipt.final_output_sha256|ConvertTo-Json -Compress))
    $runPids+=$earlyReceipt.process_id
}
if(@($runPids|Sort-Object -Unique).Count -ne 4 -or @($allCheckpoints|Sort-Object -Unique).Count -ne 1 -or @($finalHashes|Sort-Object -Unique).Count -ne 1){throw 'Fresh-process/two-poison invariance failed'}
$first=Get-Content -LiteralPath (Join-Path $attempt ($cases[0].id+'\temporal-receipt.json')) -Raw|ConvertFrom-Json
$receipt=[ordered]@{schema='neuralstudio.fg.production-contract-l3-l87-rig.v1';job_id=(Split-Path $JobRoot -Leaf);remote_job_retained=$JobRoot;attempt_id=$AttemptId;gpu=(& nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits|Select-Object -First 1).Trim();early_source_sha256=(Get-FileHash $early -Algorithm SHA256).Hash.ToLowerInvariant();live_prefix_source_sha256=(Get-FileHash $livePrefix -Algorithm SHA256).Hash.ToLowerInvariant();temporal_handoff_source_sha256=(Get-FileHash $temporalHandoff -Algorithm SHA256).Hash.ToLowerInvariant();accepted_tail_source_sha256=(Get-FileHash $tailSource -Algorithm SHA256).Hash.ToLowerInvariant();generated_tail_source_sha256=(Get-FileHash $generatedTail -Algorithm SHA256).Hash.ToLowerInvariant();live_core_source_sha256=(Get-FileHash $liveCore -Algorithm SHA256).Hash.ToLowerInvariant();binary_sha256=(Get-FileHash $binary -Algorithm SHA256).Hash.ToLowerInvariant();processes=$processes;fresh_process_count=4;poison_bytes=@(90,165);rendered_frame_count=16;phase_launch_counts=@(9,1140);launch_count=1149;allocation37_sha256=$first.allocation37_sha256;allocation29_sha256=$first.allocation29_sha256;invariant_output_sha256=$first.final_output_sha256;all_checkpoint_two_poison_invariant=$true;full_write_coverage_from_two_poison_invariance=$true;live_allocation29_temporal_reuse=$true;host_intermediate_ingress=$false;expected_or_archived_tensor_ingress=$false;address_fields_recorded=$false;result='pass'}
[IO.File]::WriteAllText((Join-Path $attempt 'backend-receipt.json'),(($receipt|ConvertTo-Json -Depth 8)+"`n"),$utf8NoBom)
Write-Output "attempt_root=$attempt"
Write-Output "invariant_output_sha256=$($first.final_output_sha256 -join ',')"
Write-Output 'result=pass'
