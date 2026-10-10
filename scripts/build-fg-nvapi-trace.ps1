<#
Build the FG NvAPI forwarding trace DLL and stage a hash-identical, renamed
copy of the authenticated System32 NvAPI beside it.
#>
[CmdletBinding()]
param(
    [string]$OutputDirectory,
    [string]$SystemNvapiPath
)

$ErrorActionPreference = 'Stop'
$repository = Split-Path -Parent $PSScriptRoot
$source = Join-Path $repository 'src\fg_trace\nvapi_forward_trace.cpp'
$descriptorSource = Join-Path $repository 'src\fg_trace\d3d12_descriptor_trace.cpp'
if (-not $OutputDirectory) {
    $OutputDirectory = Join-Path $repository 'build\fg_nvapi_trace'
}
if (-not $SystemNvapiPath) {
    $SystemNvapiPath = Join-Path $env:WINDIR 'System32\nvapi64.dll'
}

$jobRoot = [IO.Path]::GetFullPath('D:\CodexLab').TrimEnd('\') + '\'
$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
if (-not $OutputDirectory.StartsWith($jobRoot, [StringComparison]::OrdinalIgnoreCase)) {
    throw "FG NvAPI trace output must stay in a job beneath D:\CodexLab: $OutputDirectory"
}

$systemDirectory = [IO.Path]::GetFullPath((Join-Path $env:WINDIR 'System32'))
$systemNvapi = [IO.Path]::GetFullPath($SystemNvapiPath)
$expectedSystemNvapi = [IO.Path]::GetFullPath((Join-Path $systemDirectory 'nvapi64.dll'))
$isExactSystemPath = [StringComparer]::OrdinalIgnoreCase.Equals(
    $systemNvapi,
    $expectedSystemNvapi
)
if (-not $isExactSystemPath) {
    throw "Only the exact System32 NvAPI may be staged: $expectedSystemNvapi"
}
if (-not (Test-Path -LiteralPath $systemNvapi -PathType Leaf)) {
    throw "System32 NvAPI was not found: $systemNvapi"
}
foreach ($requiredSource in @($source, $descriptorSource)) {
    if (-not (Test-Path -LiteralPath $requiredSource -PathType Leaf)) {
        throw "FG NvAPI trace source was not found: $requiredSource"
    }
}

$signature = Get-AuthenticodeSignature -LiteralPath $systemNvapi
if ($signature.Status -ne [System.Management.Automation.SignatureStatus]::Valid) {
    throw "System32 NvAPI signature validation failed ($($signature.Status))."
}
if (-not $signature.SignerCertificate) {
    throw 'System32 NvAPI has no signer certificate.'
}

$signerName = $signature.SignerCertificate.GetNameInfo(
    [Security.Cryptography.X509Certificates.X509NameType]::SimpleName,
    $false
)
$signerSubject = [string]$signature.SignerCertificate.Subject
$isNvidiaSigner = (
    $signerName -match '(?i)^NVIDIA(?: Corporation)?(?: [A-Za-z0-9._-]+)*$' -and
    $signerSubject -match '(?i)(?:^|,\s*)(?:CN|O)=NVIDIA(?: Corporation)?(?:,|$)'
)
$isWhcpSigner = [StringComparer]::OrdinalIgnoreCase.Equals(
    $signerName,
    'Microsoft Windows Hardware Compatibility Publisher'
)

$versionInfo = (Get-Item -LiteralPath $systemNvapi).VersionInfo
$companyName = ([string]$versionInfo.CompanyName).Trim()
$productName = ([string]$versionInfo.ProductName).Trim()
$identifiesNvidia = (
    $companyName -match '(?i)^NVIDIA Corporation$' -and
    $productName -match '(?i)(?:^|\s)NVIDIA(?:\s|$)'
)
$isAuthenticatedNvidiaDriver = (
    $isNvidiaSigner -or
    ($isWhcpSigner -and $isExactSystemPath -and $identifiesNvidia)
)
if (-not $isAuthenticatedNvidiaDriver) {
    throw "System32 NvAPI identity policy rejected signer '$signerName', company '$companyName', product '$productName'."
}

$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path -LiteralPath $vswhere -PathType Leaf)) {
    throw "Visual Studio locator was not found: $vswhere"
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

New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$output = Join-Path $OutputDirectory 'nvapi64.dll'
$object = Join-Path $OutputDirectory 'nvapi_forward_trace.obj'
$descriptorObject = Join-Path $OutputDirectory 'd3d12_descriptor_trace.obj'
$realCopy = Join-Path $OutputDirectory 'nvapi64.real.dll'

$command = '"{0}" >nul && cl.exe /nologo /std:c++20 /EHsc /W4 /WX /O2 /MT /DUNICODE /D_UNICODE /D_WIN32_WINNT=0x0A00 /c /Fo"{1}" "{2}" && cl.exe /nologo /std:c++20 /EHsc /W4 /WX /O2 /MT /DUNICODE /D_UNICODE /D_WIN32_WINNT=0x0A00 /c /Fo"{3}" "{4}" && link.exe /nologo /DLL /Brepro /OUT:"{5}" "{1}" "{3}" bcrypt.lib' -f `
    $vcvars, $object, $source, $descriptorObject, $descriptorSource, $output
& $env:ComSpec /d /s /c $command
if ($LASTEXITCODE -ne 0) {
    throw "FG NvAPI trace build failed with exit code $LASTEXITCODE."
}
if (-not (Test-Path -LiteralPath $output -PathType Leaf)) {
    throw "FG NvAPI trace DLL was not produced: $output"
}

Copy-Item -LiteralPath $systemNvapi -Destination $realCopy -Force
$sourceHash = (Get-FileHash -LiteralPath $systemNvapi -Algorithm SHA256).Hash
$copyHash = (Get-FileHash -LiteralPath $realCopy -Algorithm SHA256).Hash
if ($copyHash -ne $sourceHash) {
    throw 'The renamed real NvAPI copy is not hash-identical to System32.'
}
$copySignature = Get-AuthenticodeSignature -LiteralPath $realCopy
if ($copySignature.Status -ne [System.Management.Automation.SignatureStatus]::Valid) {
    throw "Renamed NvAPI signature validation failed ($($copySignature.Status))."
}

$dumpbin = Get-ChildItem -LiteralPath (Join-Path $installation 'VC\Tools\MSVC') `
    -Filter dumpbin.exe -File -Recurse |
    Where-Object FullName -Match '\\Hostx64\\x64\\dumpbin.exe$' |
    Sort-Object FullName -Descending |
    Select-Object -First 1 -ExpandProperty FullName
if (-not $dumpbin) {
    throw 'The x64 dumpbin executable was not found.'
}
$exports = (& $dumpbin /exports $output) -join "`n"
foreach ($name in @(
    'nvapi_QueryInterface',
    'nvapi_Direct_GetMethod',
    'NeuralStudioFgNvapiTraceSetPhase',
    'NeuralStudioFgNvapiTraceAttachD3D12Device',
    'NeuralStudioFgNvapiTraceDetachD3D12Device'
)) {
    if ($exports -notmatch "(?m)\b$([regex]::Escape($name))\b") {
        throw "FG NvAPI trace export is missing: $name"
    }
}
$headers = (& $dumpbin /headers $output) -join "`n"
if ($headers -notmatch '(?im)machine \(x64\)') {
    throw 'FG NvAPI trace DLL is not an x64 image.'
}

$proxyHash = (Get-FileHash -LiteralPath $output -Algorithm SHA256).Hash
Write-Host "proxy.path=$output"
Write-Host "proxy.sha256=$proxyHash"
Write-Host "real_nvapi.path=$realCopy"
Write-Host "real_nvapi.sha256=$copyHash"
Write-Host 'required.env.NEURALSTUDIO_FG_REAL_NVAPI64=real_nvapi.path'
Write-Host 'required.env.NEURALSTUDIO_FG_REAL_NVAPI64_SHA256=real_nvapi.sha256'
Write-Host 'required.env.NEURALSTUDIO_FG_NVAPI_TRACE=new absolute JSONL path'
Write-Host 'required.env.NEURALSTUDIO_FG_NVAPI_VOLATILE_TRACE=new absolute volatile JSONL path'
Write-Host 'required.env.NEURALSTUDIO_FG_NVAPI_DUMP_DIR=existing absolute directory'
Write-Host 'probe=fg_nvapi_forward_trace_built'
