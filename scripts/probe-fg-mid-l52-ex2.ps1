param(
    [Parameter(Mandatory = $true)][string]$JobRoot,
    [Parameter(Mandatory = $true)][string]$RepositoryRoot,
    [Parameter(Mandatory = $true)][string]$AttemptId
)
$ErrorActionPreference = 'Stop'
if (-not $JobRoot.StartsWith('D:\CodexLab\')) { throw 'JobRoot must be beneath D:\CodexLab' }
$source = Join-Path $RepositoryRoot 'src\fg_mid_l52\rtx_ex2_probe.cu'
$attemptRoot = Join-Path $JobRoot (Join-Path 'output\fg-mid-l52-ex2-v1' $AttemptId)
if (Test-Path $attemptRoot) { throw "Attempt exists: $attemptRoot" }
New-Item -ItemType Directory -Path (Join-Path $attemptRoot 'build') | Out-Null
$cudaPath = if ($env:CUDA_PATH) { $env:CUDA_PATH } else { 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3' }
$nvcc = Join-Path $cudaPath 'bin\nvcc.exe'
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$installation = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$vcvars = Join-Path $installation 'VC\Auxiliary\Build\vcvars64.bat'
$binary = Join-Path $attemptRoot 'build\fg-mid-l52-ex2-probe.exe'
$compileLog = Join-Path $attemptRoot 'build\compile.log'
$command = '"{0}" >nul && "{1}" -std=c++20 -arch=sm_120 -O2 -Xcompiler="/W4 /WX /wd4211 /EHsc /MD /permissive-" "{2}" bcrypt.lib -o "{3}" >"{4}" 2>&1' -f $vcvars,$nvcc,$source,$binary,$compileLog
& $env:ComSpec /d /s /c $command
if ($LASTEXITCODE -ne 0) { throw "EX2 probe build failed; see $compileLog" }
$runs = @()
foreach ($ordinal in @('a','b')) {
    $runId = "process-$ordinal"
    $runRoot = Join-Path $attemptRoot $runId
    $process = Start-Process -FilePath $binary -ArgumentList @($runRoot,$runId) -NoNewWindow -Wait -PassThru -RedirectStandardOutput (Join-Path $attemptRoot "$runId.stdout.log") -RedirectStandardError (Join-Path $attemptRoot "$runId.stderr.log")
    if ($process.ExitCode -ne 0) { throw "EX2 probe failed: $runId" }
    $receipt = Get-Content (Join-Path $runRoot 'ex2-probe-receipt.json') -Raw | ConvertFrom-Json
    $runs += $receipt
}
if (@($runs.process_id | Sort-Object -Unique).Count -ne 2) { throw 'EX2 fresh process gate failed' }
foreach ($field in @('dense_sha256','random_sha256')) {
    if ($runs[0].$field -ne $runs[1].$field) { throw "EX2 repeat mismatch: $field" }
}
$rig = [ordered]@{
    schema='neuralstudio.fg.mid-l52-ex2-rig.v1'
    job_id=(Split-Path -Leaf $JobRoot)
    attempt_id=$AttemptId
    gpu=(& nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader,nounits | Select-Object -First 1).Trim()
    cuda_toolkit=(& $nvcc --version | Select-Object -Last 1).Trim()
    source_sha256=(Get-FileHash $source -Algorithm SHA256).Hash.ToLowerInvariant()
    binary_sha256=(Get-FileHash $binary -Algorithm SHA256).Hash.ToLowerInvariant()
    process_ids=@($runs.process_id)
    dense=[ordered]@{ records=$runs[0].dense_records; bytes=16777220; sha256=$runs[0].dense_sha256 }
    random=[ordered]@{ records=$runs[0].random_records; bytes=12582912; sha256=$runs[0].random_sha256 }
    repeat_equal=$true
    expected_output_fed_to_execution=$false
    archived_intermediates_consumed=$false
    address_fields_recorded=$false
    result='pass'
}
$utf8=New-Object System.Text.UTF8Encoding($false)
$path=Join-Path $attemptRoot 'rig-receipt.json'
[IO.File]::WriteAllText($path,(($rig|ConvertTo-Json -Depth 8)+"`n"),$utf8)
Write-Host "attempt_root=$attemptRoot"
Write-Host "dense_sha256=$($runs[0].dense_sha256)"
Write-Host "random_sha256=$($runs[0].random_sha256)"
Write-Host 'result=pass'
