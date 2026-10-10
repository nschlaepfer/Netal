<#
.SYNOPSIS
    Build and run the adapter ABI conformance suite.

.DESCRIPTION
    Builds the conformance runner, builds the test reference adapter if it is
    missing, and asserts the adapter contract the host depends on.

    Any adapter that ships must pass this suite. Point -AdapterPath at another
    adapter to check it against the same contract.

.EXAMPLE
    .\scripts\test-adapter-conformance.ps1

.EXAMPLE
    .\scripts\test-adapter-conformance.ps1 -AdapterPath .\build\adapters\ngx\ns_adapter_ngx.dll -ExpectedPackageId com.nvidia.ngx.dlss-nr
#>
[CmdletBinding()]
param(
    [string]$AdapterPath,
    [string]$ExpectedPackageId = 'org.neuralstudio.test.passthrough',
    [switch]$MetadataOnly
)

$ErrorActionPreference = 'Stop'

$repository = Split-Path -Parent $PSScriptRoot
$source = Join-Path $repository 'tools\conformance\ns_conformance.c'
$build = Join-Path $repository ("build\conformance\run-{0}" -f $PID)
$binary = Join-Path $build 'ns_conformance.exe'
$object = Join-Path $build 'ns_conformance.obj'

if (-not $AdapterPath) {
    $AdapterPath = Join-Path $repository 'build\adapters\test\ns_adapter_test.dll'
    if (-not (Test-Path -LiteralPath $AdapterPath -PathType Leaf)) {
        Write-Host 'Test adapter not built yet; building it first.'
        & (Join-Path $PSScriptRoot 'build-test-adapter.ps1')
    }
}

if (-not (Test-Path -LiteralPath $AdapterPath -PathType Leaf)) {
    throw "Adapter not found: $AdapterPath"
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

$command = '"{0}" >nul && cl.exe /nologo /W4 /WX /O2 /MT /Fo"{1}" "{2}" /link /OUT:"{3}"' -f `
    $vcvars,
    $object,
    $source,
    $binary

& $env:ComSpec /d /s /c $command
if ($LASTEXITCODE -ne 0) {
    throw "Conformance runner build failed with exit code $LASTEXITCODE."
}

Write-Host ''
$arguments = @($AdapterPath, $ExpectedPackageId)
if ($MetadataOnly) { $arguments += '--metadata-only' }
& $binary @arguments
$conformanceExit = $LASTEXITCODE

Write-Host ''
if ($conformanceExit -ne 0) {
    throw "Adapter conformance failed with exit code $conformanceExit."
}
Write-Host 'Adapter conformance passed.'
