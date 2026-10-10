param(
    [Parameter(Mandatory = $true)][string]$JobRoot
)

$ErrorActionPreference = 'Stop'
$module = Join-Path $JobRoot 'input\cu-module-000017.bin'
$output = Join-Path $JobRoot 'analysis\ptx'
$receipt = Join-Path $JobRoot 'analysis\ptx-extraction.json'

if (-not $JobRoot.StartsWith('D:\CodexLab\')) {
    throw 'JobRoot must be beneath D:\CodexLab'
}
if (-not (Test-Path $module)) {
    throw "Missing selected module: $module"
}
New-Item -ItemType Directory -Force -Path $output | Out-Null

$expected = '3d8c6aeba3a82fdeb99672243a3d392776bd1742576ab44e8ef42af105c2fd76'
$actual = (Get-FileHash $module -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actual -ne $expected) {
    throw "Selected module hash mismatch: $actual"
}

$cudaBin = Split-Path -Parent (Get-Command nvcc.exe).Source
$cuobjdump = Join-Path $cudaBin 'cuobjdump.exe'
if (-not (Test-Path $cuobjdump)) {
    throw "Missing cuobjdump: $cuobjdump"
}

$started = (Get-Date).ToUniversalTime().ToString('o')
Push-Location $output
try {
    & $cuobjdump -xptx all $module 2>&1 | Tee-Object -FilePath (Join-Path $output 'cuobjdump.log')
    if ($LASTEXITCODE -ne 0) { throw "cuobjdump exited $LASTEXITCODE" }
} finally {
    Pop-Location
}

$artifacts = @()
Get-ChildItem $output -File | Sort-Object Name | ForEach-Object {
    $artifacts += [ordered]@{
        relative_path = ('analysis/ptx/' + $_.Name)
        bytes = $_.Length
        sha256 = (Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}

$result = [ordered]@{
    schema = 'neuralstudio.fg.postprocess-l84-ptx-extraction.v1'
    started_utc = $started
    ended_utc = (Get-Date).ToUniversalTime().ToString('o')
    module = [ordered]@{ bytes = (Get-Item $module).Length; sha256 = $actual }
    command = 'cuobjdump.exe -xptx all input\cu-module-000017.bin'
    artifacts = $artifacts
    exit_code = 0
}
$result | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 $receipt
Write-Output $receipt
