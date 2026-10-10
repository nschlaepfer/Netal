[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string] $SourceJobRoot,
    [Parameter(Mandatory = $true)] [string] $TargetJobRoot
)

$ErrorActionPreference = 'Stop'
$sourceRoot = Join-Path $SourceJobRoot 'input'
$targetRoot = Join-Path $TargetJobRoot 'input'
$copies = @(
    [ordered]@{ source = 'full-source'; target = 'full-source-v2' },
    [ordered]@{ source = 'full-modules'; target = 'full-modules-v2' },
    [ordered]@{ source = 'full-model'; target = 'full-model-v2' }
)
foreach ($copy in $copies) {
    $source = Join-Path $sourceRoot $copy.source
    $target = Join-Path $targetRoot $copy.target
    if (-not (Test-Path -LiteralPath $source -PathType Container)) {
        throw "Source snapshot missing: $source"
    }
    if (Test-Path -LiteralPath $target) {
        throw "Target snapshot exists; preserve it: $target"
    }
    Copy-Item -LiteralPath $source -Destination $target -Recurse
}

$rows = @()
foreach ($copy in $copies) {
    $root = Join-Path $targetRoot $copy.target
    $files = @(Get-ChildItem -LiteralPath $root -Recurse -File |
               Sort-Object FullName)
    $rows += [ordered]@{
        name = $copy.target
        file_count = $files.Count
        total_bytes = [int64](($files | Measure-Object -Property Length -Sum).Sum)
        file_hash_root_sha256 = (& {
            $lines = @($files | ForEach-Object {
                $relative = $_.FullName.Substring($root.Length).TrimStart('\')
                $digest = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).
                    Hash.ToLowerInvariant()
                '{0}|{1}|{2}' -f $relative.Replace('\','/'),$_.Length,$digest
            })
            $bytes = [Text.Encoding]::UTF8.GetBytes(($lines -join "`n") + "`n")
            $sha = [Security.Cryptography.SHA256]::Create()
            try {
                -join ($sha.ComputeHash($bytes) |
                    ForEach-Object { $_.ToString('x2') })
            } finally { $sha.Dispose() }
        })
    }
}
$receipt = [ordered]@{
    schema = 'neuralstudio.fg.connected-subrect-tail-job-setup.v1'
    source_job = Split-Path $SourceJobRoot -Leaf
    target_job = Split-Path $TargetJobRoot -Leaf
    snapshots = $rows
    copied_expected_or_intermediate_outputs = $false
    result = 'pass'
}
$receiptPath = Join-Path $targetRoot 'tail-setup-receipt.json'
if (Test-Path -LiteralPath $receiptPath) {
    throw "Setup receipt exists; preserve it: $receiptPath"
}
$utf8NoBom = New-Object Text.UTF8Encoding($false)
[IO.File]::WriteAllText($receiptPath,
    (($receipt | ConvertTo-Json -Depth 6) + "`n"), $utf8NoBom)
Write-Output "setup_receipt=$receiptPath"
Write-Output 'result=pass'
