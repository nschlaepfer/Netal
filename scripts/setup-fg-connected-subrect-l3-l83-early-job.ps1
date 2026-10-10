[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string] $JobRoot,
    [string] $ParentVersion = 'v2',
    [string] $Version = 'v3',
    [string] $StageSuffix = ''
)

$ErrorActionPreference = 'Stop'
$normalizedSuffix = if ($StageSuffix) { "-$StageSuffix" } else { '' }
$source = Join-Path $JobRoot ("input\full-source-{0}" -f $ParentVersion)
$destination = Join-Path $JobRoot ("input\full-source-{0}" -f $Version)
$staged = [ordered]@{
    contract = Join-Path $JobRoot `
        ("stage-fg-connected-subrect-contract{0}.h" -f $normalizedSuffix)
    generator = Join-Path $JobRoot `
        ("stage-generate-fg-connected-subrect-l3-l83-rtx-early{0}.py" -f `
            $normalizedSuffix)
    build = Join-Path $JobRoot `
        ("stage-build-fg-connected-subrect-l3-l83-early-rtx{0}.ps1" -f `
            $normalizedSuffix)
    live_header = Join-Path $JobRoot `
        ("stage-rtx-subrect-live-handoff{0}.h" -f $normalizedSuffix)
    full_early_generator = Join-Path $JobRoot `
        ("stage-generate-fg-connected-subrect-l3-l83-rtx-full-early{0}.py" -f `
            $normalizedSuffix)
    live_prefix_generator = Join-Path $JobRoot `
        ("stage-generate-fg-connected-subrect-l3-l83-rtx-live-prefix{0}.py" -f `
            $normalizedSuffix)
    full_tail_generator = Join-Path $JobRoot `
        ("stage-generate-fg-connected-subrect-l3-l83-rtx-full-tail{0}.py" -f `
            $normalizedSuffix)
    full_build = Join-Path $JobRoot `
        ("stage-build-fg-connected-subrect-l3-l83-full-rtx{0}.ps1" -f `
            $normalizedSuffix)
}
if (-not (Test-Path -LiteralPath $source -PathType Container)) {
    throw "Frozen v2 source is missing: $source"
}
if (Test-Path -LiteralPath $destination) {
    throw "Frozen v3 source already exists; preserve it: $destination"
}
foreach ($entry in $staged.GetEnumerator()) {
    if (-not (Test-Path -LiteralPath $entry.Value -PathType Leaf)) {
        throw "Staged input is missing: $($entry.Value)"
    }
}
Copy-Item -LiteralPath $source -Destination $destination -Recurse
$sourceDirectory = Join-Path $destination 'src\fg_connected_subrect_l3_l83'
$scriptDirectory = Join-Path $destination 'scripts'
New-Item -ItemType Directory -Path $sourceDirectory -Force | Out-Null
New-Item -ItemType Directory -Path $scriptDirectory -Force | Out-Null
Copy-Item -LiteralPath $staged.contract `
    -Destination (Join-Path $sourceDirectory 'contract.h')
Copy-Item -LiteralPath $staged.generator `
    -Destination (Join-Path $scriptDirectory `
        'generate-fg-connected-subrect-l3-l83-rtx-early.py')
Copy-Item -LiteralPath $staged.build `
    -Destination (Join-Path $scriptDirectory `
        'build-fg-connected-subrect-l3-l83-early-rtx.ps1')
Copy-Item -LiteralPath $staged.live_header `
    -Destination (Join-Path $sourceDirectory 'rtx_subrect_live_handoff.h')
Copy-Item -LiteralPath $staged.full_early_generator `
    -Destination (Join-Path $scriptDirectory `
        'generate-fg-connected-subrect-l3-l83-rtx-full-early.py')
Copy-Item -LiteralPath $staged.live_prefix_generator `
    -Destination (Join-Path $scriptDirectory `
        'generate-fg-connected-subrect-l3-l83-rtx-live-prefix.py')
Copy-Item -LiteralPath $staged.full_tail_generator `
    -Destination (Join-Path $scriptDirectory `
        'generate-fg-connected-subrect-l3-l83-rtx-full-tail.py')
Copy-Item -LiteralPath $staged.full_build `
    -Destination (Join-Path $scriptDirectory `
        'build-fg-connected-subrect-l3-l83-full-rtx.ps1')
$pins = [ordered]@{
    schema = 'neuralstudio.fg.connected-subrect-l3-l83-source-snapshot.v1'
    version = $Version
    parent_source = $source
    frozen_source = $destination
    files = [ordered]@{}
    created_utc = (Get-Date).ToUniversalTime().ToString('o')
    result = 'pass'
}
foreach ($relative in @(
    'src\fg_connected_subrect_l3_l83\contract.h',
    'scripts\generate-fg-connected-subrect-l3-l83-rtx-early.py',
    'scripts\build-fg-connected-subrect-l3-l83-early-rtx.ps1',
    'src\fg_connected_subrect_l3_l83\rtx_subrect_live_handoff.h',
    'scripts\generate-fg-connected-subrect-l3-l83-rtx-full-early.py',
    'scripts\generate-fg-connected-subrect-l3-l83-rtx-live-prefix.py',
    'scripts\generate-fg-connected-subrect-l3-l83-rtx-full-tail.py',
    'scripts\build-fg-connected-subrect-l3-l83-full-rtx.ps1')) {
    $pins.files[$relative] =
        (Get-FileHash (Join-Path $destination $relative) -Algorithm SHA256).Hash.ToLowerInvariant()
}
$utf8NoBom = New-Object Text.UTF8Encoding($false)
[IO.File]::WriteAllText(
    (Join-Path $JobRoot ("input\full-source-{0}-receipt.json" -f $Version)),
    (($pins | ConvertTo-Json -Depth 6) + "`n"), $utf8NoBom)
Write-Output "source_root=$destination"
Write-Output 'result=pass'
