<# Build the DLSSNR worker supervisor and shared-resource probe. #>
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$repository = Split-Path -Parent $PSScriptRoot
$source = Join-Path $repository 'src\dlssnr_bridge_worker_probe\main.cpp'
$bridgeSource = Join-Path $repository 'runtime\vulkan_bridge\bridge_core.cpp'
$transportSource = Join-Path $repository 'runtime\vulkan_bridge\vulkan_external_transport.cpp'
$bridgeInclude = Join-Path $repository 'runtime\vulkan_bridge'
$vulkanHeaders = Join-Path $repository 'build\dependencies\Vulkan-Headers-1.3.280'
$vulkanInclude = Join-Path $vulkanHeaders 'include'
$vulkanHeadersCommit = '577baa05033cf1d9236b3d078ca4b3269ed87a2b'
$build = Join-Path $repository 'build\dlssnr_bridge_worker'
$binary = Join-Path $build 'neuralstudio-dlssnr-bridge-worker-probe.exe'
$sourceObject = Join-Path $build 'worker-probe.obj'
$bridgeObject = Join-Path $build 'worker-probe-bridge-core.obj'
$transportObject = Join-Path $build 'worker-probe-vulkan-transport.obj'

if (-not (Test-Path -LiteralPath (Join-Path $vulkanInclude 'vulkan\vulkan.h') -PathType Leaf)) {
    New-Item -ItemType Directory -Path (Split-Path -Parent $vulkanHeaders) -Force | Out-Null
    & git clone --quiet --depth 1 --branch v1.3.280 `
        https://github.com/KhronosGroup/Vulkan-Headers.git $vulkanHeaders
    if ($LASTEXITCODE -ne 0) { throw 'Unable to acquire pinned Vulkan 1.3.280 headers.' }
}
$resolvedVulkanHeadersCommit = [string](& git -C $vulkanHeaders rev-parse HEAD 2>$null)
$resolvedVulkanHeadersCommit = $resolvedVulkanHeadersCommit.Trim()
if ($LASTEXITCODE -ne 0 -or $resolvedVulkanHeadersCommit -ne $vulkanHeadersCommit) {
    throw "Vulkan-Headers provenance check failed. Expected $vulkanHeadersCommit, got '$resolvedVulkanHeadersCommit'."
}
$vulkanHeadersChanges = @(& git -C $vulkanHeaders status --porcelain)
if ($LASTEXITCODE -ne 0 -or $vulkanHeadersChanges.Count -ne 0) {
    throw 'Pinned Vulkan-Headers checkout has local changes.'
}

$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$installation = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $installation) { throw 'Visual Studio C++ x64 build tools were not found.' }
$vcvars = Join-Path $installation 'VC\Auxiliary\Build\vcvars64.bat'
New-Item -ItemType Directory -Path $build -Force | Out-Null

$command = '"{0}" >nul && cl.exe /nologo /std:c++20 /EHsc /W4 /WX /O2 /MD /DUNICODE /D_UNICODE /I"{1}" /I"{2}" /Fo"{3}" /c "{4}" && cl.exe /nologo /std:c++20 /EHsc /W4 /WX /O2 /MD /DUNICODE /D_UNICODE /I"{1}" /I"{2}" /Fo"{5}" /c "{6}" && cl.exe /nologo /std:c++20 /EHsc /W4 /WX /O2 /MD /DUNICODE /D_UNICODE /I"{1}" /I"{2}" /Fo"{7}" /c "{8}" && link.exe /nologo /Brepro /OUT:"{9}" "{3}" "{5}" "{7}" d3d12.lib dxgi.lib advapi32.lib' -f `
    $vcvars, $bridgeInclude, $vulkanInclude, $sourceObject, $source, $bridgeObject, $bridgeSource, $transportObject, $transportSource, $binary
& $env:ComSpec /d /s /c $command
if ($LASTEXITCODE -ne 0) { throw "DLSSNR bridge worker probe build failed with exit code $LASTEXITCODE." }

Get-FileHash -LiteralPath $binary -Algorithm SHA256
Write-Host "Built: $binary"
