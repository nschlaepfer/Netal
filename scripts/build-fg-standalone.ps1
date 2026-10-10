[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$NgxInclude,
    [string]$RuntimePath,
    [string]$BuildDirectory
)

$ErrorActionPreference = 'Stop'
$expectedRuntimeSha256 = '5D5CBF14D2727D47F93FD10BF77BD91708AE122482A6F86FD564971641EBD47B'
$repository = Split-Path -Parent $PSScriptRoot
$sourceDirectory = Join-Path $repository 'src\fg_host'
if (-not $RuntimePath) {
    $RuntimePath = Join-Path $repository 'runtime\streamline-2.13\nvngx_dlssg.dll'
}
if (-not $BuildDirectory) {
    $BuildDirectory = Join-Path $repository 'build\fg_host'
}
$NgxInclude = [IO.Path]::GetFullPath($NgxInclude)
$RuntimePath = [IO.Path]::GetFullPath($RuntimePath)
$BuildDirectory = [IO.Path]::GetFullPath($BuildDirectory)

$requiredInputs = @(
    (Join-Path $NgxInclude 'nvsdk_ngx.h'),
    (Join-Path $NgxInclude 'nvsdk_ngx_defs_dlssg.h'),
    (Join-Path $NgxInclude 'nvsdk_ngx_params.h'),
    (Join-Path $sourceDirectory 'main.cpp'),
    (Join-Path $sourceDirectory 'ngx_d3d12_thunk.cpp'),
    (Join-Path $sourceDirectory 'ngx_parameter_store.h'),
    (Join-Path $sourceDirectory 'fg_parameter_contract.h'),
    (Join-Path $sourceDirectory 'runtime_allocation_capture.h'),
    (Join-Path $sourceDirectory 'runtime_allocation_capture.cpp'),
    (Join-Path $sourceDirectory 'synthetic_contract.h'),
    (Join-Path $sourceDirectory 'offline_sequence.h'),
    (Join-Path $sourceDirectory 'synthetic_contract_test.cpp'),
    (Join-Path $sourceDirectory 'parameter_store_contract_test.cpp'),
    (Join-Path $sourceDirectory 'runtime_upload_capture_contract_test.cpp'),
    $RuntimePath
)
foreach ($required in $requiredInputs) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "FG standalone build input was not found: $required"
    }
}

$runtimeHash = (Get-FileHash -LiteralPath $RuntimePath -Algorithm SHA256).Hash
if ($runtimeHash -ne $expectedRuntimeSha256) {
    throw "Exact nvngx_dlssg.dll hash mismatch: $runtimeHash"
}
$signature = Get-AuthenticodeSignature -LiteralPath $RuntimePath
if ($signature.Status -ne 'Valid' -or
    -not $signature.SignerCertificate -or
    $signature.SignerCertificate.Subject -notmatch 'NVIDIA') {
    throw "Pinned FG runtime does not have a valid NVIDIA Authenticode signature: status=$($signature.Status) signer=$($signature.SignerCertificate.Subject)"
}

$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path -LiteralPath $vswhere -PathType Leaf)) {
    throw 'Visual Studio Installer vswhere.exe was not found.'
}
$installation = & $vswhere -latest -products '*' `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath
if (-not $installation) {
    throw 'Visual Studio C++ x64 build tools were not found.'
}
$vcvars = Join-Path $installation 'VC\Auxiliary\Build\vcvars64.bat'
if (-not (Test-Path -LiteralPath $vcvars -PathType Leaf)) {
    throw "vcvars64.bat was not found: $vcvars"
}

New-Item -ItemType Directory -Path $BuildDirectory -Force | Out-Null
$hostObject = Join-Path $BuildDirectory 'main.obj'
$thunkObject = Join-Path $BuildDirectory 'ngx_d3d12_thunk.obj'
$testObject = Join-Path $BuildDirectory 'synthetic_contract_test.obj'
$parameterTestObject = Join-Path $BuildDirectory 'parameter_store_contract_test.obj'
$allocationCaptureObject = Join-Path $BuildDirectory 'runtime_allocation_capture.obj'
$uploadCaptureTestObject = Join-Path $BuildDirectory 'runtime_upload_capture_contract_test.obj'
$hostBinary = Join-Path $BuildDirectory 'neuralstudio-fg-standalone.exe'
$thunkBinary = Join-Path $BuildDirectory 'nvngx.dll'
$testBinary = Join-Path $BuildDirectory 'synthetic-contract-test.exe'
$parameterTestBinary = Join-Path $BuildDirectory 'parameter-store-contract-test.exe'
$uploadCaptureTestBinary = Join-Path $BuildDirectory 'runtime-upload-capture-contract-test.exe'
$stagedRuntime = Join-Path $BuildDirectory 'nvngx_dlssg.dll'
$manifestPath = Join-Path $BuildDirectory 'build-manifest.json'

$common = '/nologo /std:c++20 /EHsc /MD /W4 /WX /permissive- /Zc:__cplusplus /utf-8 /DUNICODE /D_UNICODE'
$testCommand = '"{0}" >nul && cl.exe {1} /Fo"{2}" /Fe"{3}" "{4}"' -f `
    $vcvars, $common, $testObject, $testBinary,
    (Join-Path $sourceDirectory 'synthetic_contract_test.cpp')
& $env:ComSpec /d /s /c $testCommand
if ($LASTEXITCODE -ne 0) {
    throw "Synthetic-contract test build failed with exit code $LASTEXITCODE."
}
& $testBinary
if ($LASTEXITCODE -ne 0) {
    throw "Synthetic-contract test failed with exit code $LASTEXITCODE."
}

$parameterTestCommand = '"{0}" >nul && cl.exe {1} /I"{2}" /I"{3}" /Fo"{4}" /Fe"{5}" "{6}"' -f `
    $vcvars, $common, $NgxInclude, $sourceDirectory, $parameterTestObject,
    $parameterTestBinary,
    (Join-Path $sourceDirectory 'parameter_store_contract_test.cpp')
& $env:ComSpec /d /s /c $parameterTestCommand
if ($LASTEXITCODE -ne 0) {
    throw "Parameter-store contract test build failed with exit code $LASTEXITCODE."
}
& $parameterTestBinary
if ($LASTEXITCODE -ne 0) {
    throw "Parameter-store contract test failed with exit code $LASTEXITCODE."
}

$uploadCaptureTestCommand = '"{0}" >nul && cl.exe {1} /I"{2}" /I"{3}" /Fo"{4}" /Fe"{5}" "{6}"' -f `
    $vcvars, $common, $NgxInclude, $sourceDirectory, $uploadCaptureTestObject,
    $uploadCaptureTestBinary,
    (Join-Path $sourceDirectory 'runtime_upload_capture_contract_test.cpp')
& $env:ComSpec /d /s /c $uploadCaptureTestCommand
if ($LASTEXITCODE -ne 0) {
    throw "Runtime-upload capture contract test build failed with exit code $LASTEXITCODE."
}
& $uploadCaptureTestBinary
if ($LASTEXITCODE -ne 0) {
    throw "Runtime-upload capture contract test failed with exit code $LASTEXITCODE."
}

$thunkCommand = '"{0}" >nul && cl.exe {1} /I"{2}" /Fo"{3}" /c "{4}" && link.exe /nologo /DLL /OUT:"{5}" "{3}"' -f `
    $vcvars, $common, $NgxInclude, $thunkObject,
    (Join-Path $sourceDirectory 'ngx_d3d12_thunk.cpp'), $thunkBinary
& $env:ComSpec /d /s /c $thunkCommand
if ($LASTEXITCODE -ne 0) {
    throw "Typed NGX D3D12 thunk build failed with exit code $LASTEXITCODE."
}

$hostCommand = '"{0}" >nul && cl.exe {1} /I"{2}" /I"{3}" /Fo"{4}" /c "{5}" && cl.exe {1} /I"{2}" /I"{3}" /Fo"{6}" /c "{7}" && link.exe /nologo /OUT:"{8}" "{4}" "{6}" d3d12.lib dxgi.lib bcrypt.lib' -f `
    $vcvars, $common, $NgxInclude, $sourceDirectory, $hostObject,
    (Join-Path $sourceDirectory 'main.cpp'), $allocationCaptureObject,
    (Join-Path $sourceDirectory 'runtime_allocation_capture.cpp'), $hostBinary
& $env:ComSpec /d /s /c $hostCommand
if ($LASTEXITCODE -ne 0) {
    throw "FG standalone host build failed with exit code $LASTEXITCODE."
}

Copy-Item -LiteralPath $RuntimePath -Destination $stagedRuntime -Force
$stagedRuntimeHash = (Get-FileHash -LiteralPath $stagedRuntime -Algorithm SHA256).Hash
if ($stagedRuntimeHash -ne $expectedRuntimeSha256) {
    throw "Staged FG runtime hash changed: $stagedRuntimeHash"
}

$artifacts = @(
    $hostBinary,
    $thunkBinary,
    $testBinary,
    $parameterTestBinary,
    $uploadCaptureTestBinary,
    $stagedRuntime
) | ForEach-Object {
    [ordered]@{
        name = [IO.Path]::GetFileName($_)
        bytes = (Get-Item -LiteralPath $_).Length
        sha256 = (Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}
$manifest = [ordered]@{
    schema = 'neuralstudio.dlss_fg.standalone_build.v1'
    built_at = (Get-Date).ToUniversalTime().ToString('o')
    source_root = $sourceDirectory
    ngx_include = $NgxInclude
    runtime_source = $RuntimePath
    runtime_expected_sha256 = $expectedRuntimeSha256.ToLowerInvariant()
    runtime_signature_status = [string]$signature.Status
    runtime_signer = $signature.SignerCertificate.Subject
    compiler_environment = $vcvars
    feature = 'DLSS Frame Generation'
    feature_id = 11
    default_scale = '2x'
    claim = 'build_receipt_only_no_execution_or_parity_claim'
    artifacts = $artifacts
}
$manifest | ConvertTo-Json -Depth 6 |
    Set-Content -LiteralPath $manifestPath -Encoding utf8

Get-Content -LiteralPath $manifestPath
Write-Host "Built FG standalone host: $hostBinary"
