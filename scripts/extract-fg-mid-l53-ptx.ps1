[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$JobRoot
)

$ErrorActionPreference = 'Stop'
$JobRoot = [IO.Path]::GetFullPath($JobRoot)
if (-not $JobRoot.StartsWith('D:\CodexLab\')) {
    throw 'JobRoot must be beneath D:\CodexLab'
}
$module = Join-Path $JobRoot 'inputs\cu-module-000012.bin'
$output = Join-Path $JobRoot 'analysis\ptx'
if (-not (Test-Path -LiteralPath $module -PathType Leaf)) {
    throw "Missing selected module: $module"
}
if ((Get-Item -LiteralPath $module).Length -ne 98408 -or
    (Get-FileHash -LiteralPath $module -Algorithm SHA256).Hash -ne
        '5A8E0284AAB8AC14FC82B0504BBEEF25D2FCE1D13A1C11D8BDB3F91FEE8145FC') {
    throw 'Selected L53 module identity mismatch'
}
New-Item -ItemType Directory -Force -Path $output | Out-Null
$cudaPath = $env:CUDA_PATH
if (-not $cudaPath) {
    $cudaPath = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3'
}
$cuobjdump = Join-Path $cudaPath 'bin\cuobjdump.exe'
Push-Location $output
try {
    & $cuobjdump -xptx all $module *> (Join-Path $output 'cuobjdump.log')
    if ($LASTEXITCODE -ne 0) { throw "cuobjdump exited $LASTEXITCODE" }
} finally {
    Pop-Location
}
$artifacts = @()
Get-ChildItem -LiteralPath $output -File | Sort-Object Name | ForEach-Object {
    $artifacts += [ordered]@{
        name = $_.Name
        bytes = $_.Length
        sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}
$receipt = [ordered]@{
    schema = 'neuralstudio.fg.mid-l53-ptx-extraction.v1'
    module_sha256 = (Get-FileHash -LiteralPath $module -Algorithm SHA256).Hash.ToLowerInvariant()
    artifacts = $artifacts
    result = 'pass'
}
$utf8 = New-Object System.Text.UTF8Encoding($false)
$receiptPath = Join-Path $JobRoot 'analysis\ptx-extraction.json'
[IO.File]::WriteAllText($receiptPath, (($receipt | ConvertTo-Json -Depth 5) + "`n"), $utf8)
Write-Host "receipt=$receiptPath"
Write-Host "receipt_sha256=$((Get-FileHash -LiteralPath $receiptPath -Algorithm SHA256).Hash.ToLowerInvariant())"
Write-Host 'result=pass'
