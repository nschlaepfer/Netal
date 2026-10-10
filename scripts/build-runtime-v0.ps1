<# Build the NeuralStudio Runtime v0 native host. #>
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$repository = Split-Path -Parent $PSScriptRoot
$build = Join-Path $repository 'build\runtime-v0'
$binary = Join-Path $build 'neuralstudio-runtime.exe'
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path -LiteralPath $vswhere -PathType Leaf)) { throw 'Visual Studio Installer vswhere.exe was not found.' }
$installation = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $installation) { throw 'Visual Studio C++ x64 build tools were not found.' }
$vcvars = Join-Path $installation 'VC\Auxiliary\Build\vcvars64.bat'
New-Item -ItemType Directory -Path $build -Force | Out-Null

$sources = @(
    'src\runtime_host\main.cpp',
    'runtime\control\local_api_server.cpp',
    'runtime\runtime_host\runtime_engine.cpp',
    'runtime\frame_pipeline\diagnostic_d3d12.cpp',
    'runtime\package\package_loader.cpp',
    'runtime\manifest\json.cpp',
    'runtime\manifest\manifest_reader.cpp',
    'runtime\broker\compatibility.cpp',
    'runtime\broker\pipeline_state.cpp'
)
$objects = @()
$compile = @()
foreach ($source in $sources) {
    $name = ($source -replace '[\\/.]', '_') + '.obj'
    $object = Join-Path $build $name
    $objects += $object
    $compile += 'cl.exe /nologo /std:c++20 /EHsc /W4 /WX /O2 /MT /DUNICODE /D_UNICODE /I"{0}" /I"{1}" /Fo"{2}" /c "{3}"' -f `
        $repository, (Join-Path $repository 'spec\adapter\v1'), $object, (Join-Path $repository $source)
}
$quotedObjects = ($objects | ForEach-Object { '"{0}"' -f $_ }) -join ' '
$link = 'link.exe /nologo /Brepro /OUT:"{0}" {1} d3d12.lib dxgi.lib bcrypt.lib wintrust.lib crypt32.lib advapi32.lib' -f $binary, $quotedObjects
$command = '"{0}" >nul && {1} && {2}' -f $vcvars, ($compile -join ' && '), $link
& $env:ComSpec /d /s /c $command
if ($LASTEXITCODE -ne 0) { throw "Runtime v0 build failed with exit code $LASTEXITCODE." }

Write-Host "Built: $binary"
Get-FileHash -LiteralPath $binary -Algorithm SHA256
