[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string] $JobRoot,
    [Parameter(Mandatory)] [string] $NgxInclude,
    [Parameter(Mandatory)] [string] $RuntimePath
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$JobRoot = [IO.Path]::GetFullPath($JobRoot)
if ($JobRoot -notmatch '^D:\\CodexLab\\[A-Za-z0-9_-]+$') {
    throw 'Reference jobs must use a new job directory under D:\CodexLab.'
}
$build = Join-Path $JobRoot 'build'
$output = Join-Path $JobRoot 'output'
if ((Test-Path -LiteralPath $build) -or (Test-Path -LiteralPath $output)) { throw 'Preserve existing build/output; use a new job.' }
$manifestPath = Join-Path $JobRoot 'job.json'
$manifestHash = (Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if ($manifest.schema -ne 'netal.fg.offline-reference-job.v1') { throw 'Unsupported job manifest.' }
if ($manifest.cases.Count -lt 1 -or $manifest.cases.Count -gt 12) { throw 'Reference job must contain 1..12 cases.' }
function CheckedPath([string] $relative) {
    if ($relative -notmatch '^[A-Za-z0-9_./-]+$' -or $relative.Contains('..') -or $relative.StartsWith('/')) { throw 'Unsafe input path.' }
    return Join-Path $JobRoot ($relative.Replace('/', '\'))
}
function VerifyInputs {
    foreach ($entry in $manifest.files) {
        $path = CheckedPath $entry.path
        if ((Get-Item -LiteralPath $path).Length -ne $entry.bytes -or
            (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.sha256) {
            throw "Input changed: $($entry.path)"
        }
    }
    if ((Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $manifestHash) { throw 'Manifest changed.' }
}
VerifyInputs
New-Item -ItemType Directory -Path $output | Out-Null
& nvidia-smi -q | Set-Content -LiteralPath (Join-Path $output 'gpu-before.txt')
$started = (Get-Date).ToUniversalTime().ToString('o')
& (Join-Path $JobRoot 'source\scripts\build-fg-standalone.ps1') -NgxInclude $NgxInclude -RuntimePath $RuntimePath -BuildDirectory $build *> (Join-Path $output 'build.log')
if (-not (Test-Path -LiteralPath (Join-Path $build 'neuralstudio-fg-standalone.exe'))) { throw 'Reference build failed.' }
$results = @()
foreach ($case in $manifest.cases) {
    if ($case.name -notmatch '^[a-z0-9-]+$' -or $case.frames -lt 2 -or $case.frames -gt 16 -or
        $case.motion_scale -notin @('recorded', 'one')) { throw 'Invalid reference case.' }
    $sequence = CheckedPath $case.sequence
    if ((Get-FileHash -LiteralPath $sequence -Algorithm SHA256).Hash.ToLowerInvariant() -ne $case.sha256) { throw 'Sequence hash mismatch.' }
    foreach ($poison in @(165, 90)) {
        foreach ($repeat in @(0, 1)) {
            $name = "$($case.name)-p$poison-r$repeat"
            $directory = Join-Path $output $name
            $arguments = @('--arm-execution', '--runtime', (Join-Path $build 'nvngx_dlssg.dll'),
                '--thunk', (Join-Path $build 'nvngx.dll'), '--output-dir', $directory,
                '--width', '640', '--height', '360', '--rendered-frames', [string]$case.frames,
                '--disable-interpolation-output', '--output-poison-byte', [string]$poison,
                '--input-sequence', $sequence, '--input-sequence-sha256', $case.sha256,
                '--reference-motion-scale', $case.motion_scale)
            # All arguments are fixed tokens or whitespace-free validated paths.
            # Own only this newly started child; never stop another GPU job.
            $process = Start-Process -FilePath (Join-Path $build 'neuralstudio-fg-standalone.exe') -ArgumentList $arguments `
                -NoNewWindow -PassThru -RedirectStandardOutput (Join-Path $output "$name.stdout") -RedirectStandardError (Join-Path $output "$name.stderr")
            $null = $process.Handle
            [ordered]@{name=$name;pid=$process.Id;started=(Get-Date).ToUniversalTime().ToString('o')} |
                ConvertTo-Json | Set-Content -LiteralPath (Join-Path $output 'active-child.json') -Encoding utf8
            $completed = $process.WaitForExit(120000)
            if (-not $completed) { Stop-Process -Id $process.Id -Force; $process.WaitForExit(); $code = 124 }
            else { $code = $process.ExitCode }
            $results += [ordered]@{name=$name;case=$case.name;poison=$poison;repeat=$repeat;exit_code=$code;timed_out=(-not $completed);arguments=$arguments}
            $results | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $output 'runs.json') -Encoding utf8
            if ($code -ne 0) { throw "Reference execution failed: $name exit $code" }
        }
    }
}
VerifyInputs
& nvidia-smi -q | Set-Content -LiteralPath (Join-Path $output 'gpu-after.txt')
[ordered]@{schema='netal.fg.offline-reference-execution.v1';started=$started;finished=(Get-Date).ToUniversalTime().ToString('o');
    manifest_sha256=$manifestHash;runs=$results;inputs_unchanged=$true;parity_claim=$false;game_used=$false} |
    ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $output 'execution.json') -Encoding utf8
