<#
.SYNOPSIS
    Build the NeuralStudio ABI v1 adapter for the isolated DLSSNR worker.
#>
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$repository = Split-Path -Parent $PSScriptRoot
$build = Join-Path $repository 'build\adapters\ngx'
$binary = Join-Path $build 'ns_adapter_ngx.dll'
$packageBinary = Join-Path $repository 'packages\com.nvidia.ngx.dlss-nr\ns_adapter_ngx.dll'
$sources = @(
    (Join-Path $repository 'adapters\ngx\ns_adapter_ngx.cpp'),
    (Join-Path $repository 'runtime\dlssnr_worker\client.cpp')
)

$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path -LiteralPath $vswhere -PathType Leaf)) { throw 'Visual Studio Installer vswhere.exe was not found.' }
$installation = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $installation) { throw 'Visual Studio C++ x64 build tools were not found.' }
$vcvars = Join-Path $installation 'VC\Auxiliary\Build\vcvars64.bat'
New-Item -ItemType Directory -Path $build -Force | Out-Null

$includes = @(
    (Join-Path $repository 'spec\adapter\v1'),
    (Join-Path $repository 'runtime\dlssnr_worker'),
    (Join-Path $repository 'runtime\vulkan_bridge')
) | ForEach-Object { '/I"{0}"' -f $_ }
$objects = @(
    (Join-Path $build 'ns_adapter_ngx.obj'),
    (Join-Path $build 'dlssnr_worker_client.obj')
)
$compile = for ($index = 0; $index -lt $sources.Count; $index++) {
    'cl.exe /nologo /std:c++20 /EHsc /W4 /WX /O2 /MT {0} /Fo"{1}" /c "{2}"' -f ($includes -join ' '), $objects[$index], $sources[$index]
}
$link = 'link.exe /nologo /DLL /Brepro /OUT:"{0}" {1} d3d12.lib dxgi.lib advapi32.lib bcrypt.lib' -f $binary, (($objects | ForEach-Object { '"{0}"' -f $_ }) -join ' ')
$command = '"{0}" >nul && {1} && {2}' -f $vcvars, ($compile -join ' && '), $link
& $env:ComSpec /d /s /c $command
if ($LASTEXITCODE -ne 0) { throw "NGX adapter build failed with exit code $LASTEXITCODE." }

Copy-Item -LiteralPath $binary -Destination $packageBinary -Force
$hash = (Get-FileHash -LiteralPath $binary -Algorithm SHA256).Hash
Write-Host "Built: $binary"
Write-Host "SHA256: $hash"
Write-Host "Packaged: $packageBinary"
