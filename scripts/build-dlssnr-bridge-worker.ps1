<# Build the crash-isolated D3D12 DLSSNR worker. #>
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$repository = Split-Path -Parent $PSScriptRoot
$source = Join-Path $repository 'src\dlssnr_bridge_worker\main.cpp'
$bridgeInclude = Join-Path $repository 'runtime\vulkan_bridge'
$privateInclude = Join-Path $repository 'src\media_host'
$streamlineCommit = 'e8aaa6eaac968711fb62473d4ae8256dde20919b'
$preferredStreamline = Join-Path (Split-Path -Parent $repository) 'source\renodx\external\Streamline'
$streamline = if (Test-Path -LiteralPath (Join-Path $preferredStreamline 'include\sl.h') -PathType Leaf) {
    $preferredStreamline
} else {
    Join-Path $repository 'build\dependencies\Streamline-e8aaa6e'
}
$dependencyRoot = Split-Path -Parent $streamline
if (-not (Test-Path -LiteralPath (Join-Path $streamline 'include\sl.h') -PathType Leaf)) {
    New-Item -ItemType Directory -Path $dependencyRoot -Force | Out-Null
    & git clone --quiet --filter=blob:none --no-checkout https://github.com/clshortfuse/Streamline.git $streamline
    if ($LASTEXITCODE -ne 0) { throw 'Unable to acquire the pinned Streamline source.' }
    & git -C $streamline fetch --quiet --depth 1 origin $streamlineCommit
    if ($LASTEXITCODE -ne 0) { throw 'Unable to fetch the pinned Streamline commit.' }
    & git -C $streamline checkout --quiet --detach $streamlineCommit
    if ($LASTEXITCODE -ne 0) { throw 'Unable to check out the pinned Streamline commit.' }
}
$streamlineInclude = Join-Path $streamline 'include'
$build = Join-Path $repository 'build\dlssnr_bridge_worker'
$binary = Join-Path $build 'neuralstudio-dlssnr-bridge-worker.exe'
$object = Join-Path $build 'worker.obj'

if (-not (Test-Path -LiteralPath (Join-Path $streamlineInclude 'sl.h') -PathType Leaf)) {
    throw "Pinned Streamline headers were not found: $streamlineInclude"
}
$resolvedStreamlineCommit = [string](& git -C $streamline rev-parse HEAD 2>$null)
if ($LASTEXITCODE -ne 0 -or $resolvedStreamlineCommit.Trim() -ne $streamlineCommit) {
    throw "Streamline provenance check failed. Expected $streamlineCommit."
}
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$installation = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $installation) { throw 'Visual Studio C++ x64 build tools were not found.' }
$vcvars = Join-Path $installation 'VC\Auxiliary\Build\vcvars64.bat'
New-Item -ItemType Directory -Path $build -Force | Out-Null

$command = '"{0}" >nul && cl.exe /nologo /std:c++20 /EHsc /W4 /WX /O2 /MD /DUNICODE /D_UNICODE /I"{1}" /I"{2}" /I"{3}" /Fo"{4}" "{5}" /link /Brepro /OUT:"{6}" d3d12.lib dxgi.lib' -f `
    $vcvars, $streamlineInclude, $bridgeInclude, $privateInclude, $object, $source, $binary
& $env:ComSpec /d /s /c $command
if ($LASTEXITCODE -ne 0) { throw "DLSSNR bridge worker build failed with exit code $LASTEXITCODE." }

Get-FileHash -LiteralPath $binary -Algorithm SHA256
Write-Host "Built: $binary"
