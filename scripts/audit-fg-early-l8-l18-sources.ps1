[CmdletBinding()]
param([Parameter(Mandatory = $true)][string]$JobRoot)
$ErrorActionPreference = 'Stop'
$JobRoot = [IO.Path]::GetFullPath($JobRoot)
if (-not $JobRoot.StartsWith('D:\CodexLab\', [StringComparison]::OrdinalIgnoreCase)) {
    throw "JobRoot must be beneath D:\CodexLab: $JobRoot"
}
$outputRoot = Join-Path $JobRoot 'output\source-audit-v1'
if (Test-Path -LiteralPath $outputRoot) { throw "Source audit exists: $outputRoot" }
New-Item -ItemType Directory -Path $outputRoot | Out-Null
$paths = [ordered]@{
    unit_source = Join-Path $JobRoot 'repo\src\fg_early_l8_l18\rtx_oracle.cpp'
    contract = Join-Path $JobRoot 'repo\src\fg_early_l8_l18\contract.h'
    shared_cuda_helper = Join-Path $JobRoot 'repo\src\fg_conv\rtx_oracle.cpp'
    build_harness = Join-Path $JobRoot 'repo\scripts\build-fg-early-l8-l18-rtx.ps1'
    module = Join-Path $JobRoot 'inputs\cu-module-000025.bin'
    binary = Join-Path $JobRoot 'output\oracle-v1\build\fg-early-l8-l18-rtx.exe'
}
$files = [ordered]@{}
foreach ($entry in $paths.GetEnumerator()) {
    if (-not (Test-Path -LiteralPath $entry.Value -PathType Leaf)) {
        throw "Missing source-audit input: $($entry.Value)"
    }
    $item = Get-Item -LiteralPath $entry.Value
    $files[$entry.Key] = [ordered]@{
        bytes = [int64]$item.Length
        sha256 = (Get-FileHash -LiteralPath $entry.Value -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}
$expected = [ordered]@{
    unit_source = 'b863420c2ea99837aaec31923d49e783f3fe177a4430ec528de3812f8483d40d'
    contract = '7247dd17060e7dd91d47742d777dca39619d8ef54b5565452edd510f1cff636b'
    shared_cuda_helper = 'a83039f524ae70f2c6bc86190c6c8634a2670a86ecbcd5525a055aade533a1b6'
    build_harness = '0ed1fa7be61a97331263b91ec6bda69057b93790d0bf9f8cd3413a2d438788db'
    module = '28fe0e153d1ffc80b54995c022f18395c5556f3c041cc9d724bfe7e70dc4e824'
    binary = '4207e6b84d8b458ae6b50336a5d335a4f28578e1ff0b5a7c2eb2283f90db27a7'
}
foreach ($entry in $expected.GetEnumerator()) {
    if ($files[$entry.Key].sha256 -ne $entry.Value) {
        throw "Source-audit hash mismatch: $($entry.Key)"
    }
}
$receipt = [ordered]@{
    schema = 'neuralstudio.fg.early-l8-l18-source-audit.v1'
    job_id = Split-Path -Leaf $JobRoot
    audited_attempt = 'oracle-v1'
    captured_utc = (Get-Date).ToUniversalTime().ToString('o')
    gpu = (& nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | Select-Object -First 1).Trim()
    files = $files
    shared_cuda_helper_omitted_from_original_rig_receipt = $true
    shared_cuda_helper_live_hash_recovered = $true
    local_match_required_by_final_verifier = $true
    address_fields_recorded = $false
    result = 'pass'
}
$utf8 = New-Object System.Text.UTF8Encoding($false)
$receiptPath = Join-Path $outputRoot 'source-audit.json'
[IO.File]::WriteAllText($receiptPath, (($receipt | ConvertTo-Json -Depth 6) + "`n"), $utf8)
Write-Host "source_audit=$receiptPath"
Write-Host "source_audit_sha256=$((Get-FileHash -LiteralPath $receiptPath -Algorithm SHA256).Hash.ToLowerInvariant())"
Write-Host 'result=pass'
