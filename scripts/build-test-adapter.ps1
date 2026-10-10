[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

$repository = Split-Path -Parent $PSScriptRoot
$source = Join-Path $repository 'adapters\test\ns_adapter_test.c'
$build = Join-Path $repository 'build\adapters\test'
$binary = Join-Path $build 'ns_adapter_test.dll'
$object = Join-Path $build 'ns_adapter_test.obj'

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

# /W4 /WX because the adapter boundary is the contract everything else trusts.
$command = '"{0}" >nul && cl.exe /nologo /W4 /WX /O2 /MT /Fo"{1}" "{2}" /LD /link /Brepro /OUT:"{3}"' -f `
    $vcvars,
    $object,
    $source,
    $binary

& $env:ComSpec /d /s /c $command
if ($LASTEXITCODE -ne 0) {
    throw "Test adapter build failed with exit code $LASTEXITCODE."
}

$hash = (Get-FileHash -LiteralPath $binary -Algorithm SHA256).Hash
$packageBinary = Join-Path $repository 'packages\org.neuralstudio.test.passthrough\ns_adapter_test.dll'
Copy-Item -LiteralPath $binary -Destination $packageBinary -Force
Write-Host "Built: $binary"
Write-Host "SHA256: $hash"
Write-Host "Packaged: $packageBinary"
