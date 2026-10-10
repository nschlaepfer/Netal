<#
.SYNOPSIS
    Build and run the runtime broker core tests.

.DESCRIPTION
    The broker's state machine, reset policy, and compatibility engine carry the
    rules the product's honesty depends on: Active is only reported with
    observed evaluations and fresh output, temporal history is discarded exactly
    when it must be, and an incompatible package is refused with a named reason
    rather than silently hidden.

    All three are free of GPU, adapter, and IPC dependencies so they can be
    tested exhaustively on any machine with the MSVC toolchain.

.EXAMPLE
    .\scripts\test-broker.ps1
#>
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

$repository = Split-Path -Parent $PSScriptRoot
$brokerDirectory = Join-Path $repository 'runtime\broker'
$manifestDirectory = Join-Path $repository 'runtime\manifest'
$build = Join-Path $repository 'build\runtime\broker'
$binary = Join-Path $build 'broker_tests.exe'

$sources = @(
    Join-Path $brokerDirectory 'pipeline_state.cpp'
    Join-Path $brokerDirectory 'reset_policy.cpp'
    Join-Path $brokerDirectory 'compatibility.cpp'
    Join-Path $brokerDirectory 'test_support.cpp'
    Join-Path $brokerDirectory 'compatibility_tests.cpp'
    Join-Path $manifestDirectory 'json.cpp'
    Join-Path $manifestDirectory 'manifest_reader.cpp'
    Join-Path $manifestDirectory 'manifest_tests.cpp'
    Join-Path $brokerDirectory 'broker_tests.cpp'
)

foreach ($source in $sources) {
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "Source is missing: $source"
    }
}

$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path -LiteralPath $vswhere -PathType Leaf)) {
    throw 'Visual Studio Installer vswhere.exe was not found.'
}

$installation = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $installation) {
    throw 'Visual Studio C++ x64 build tools were not found.'
}

$vcvars = Join-Path $installation 'VC\Auxiliary\Build\vcvars64.bat'
New-Item -ItemType Directory -Path $build -Force | Out-Null

$quotedSources = ($sources | ForEach-Object { '"{0}"' -f $_ }) -join ' '

$command = '"{0}" >nul && cl.exe /nologo /std:c++20 /EHsc /W4 /WX /O2 /MT /Fo"{1}\\" {2} /link /OUT:"{3}"' -f `
    $vcvars,
    $build,
    $quotedSources,
    $binary

& $env:ComSpec /d /s /c $command
if ($LASTEXITCODE -ne 0) {
    throw "Broker test build failed with exit code $LASTEXITCODE."
}

Write-Host ''
# The manifest tests read packages\*\package.json relative to the repository
# root, so run from there rather than from the scripts directory.
Push-Location $repository
try {
    & $binary
    $testExit = $LASTEXITCODE
} finally {
    Pop-Location
}

Write-Host ''
if ($testExit -ne 0) {
    throw "Broker core tests failed with exit code $testExit."
}
Write-Host 'Broker core tests passed.'
