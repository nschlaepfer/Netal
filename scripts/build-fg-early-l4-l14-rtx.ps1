param(
    [Parameter(Mandatory = $true)][string]$JobRoot,
    [Parameter(Mandatory = $true)][string]$RepositoryRoot,
    [Parameter(Mandatory = $true)][string]$AttemptId,
    [string]$BaseCommit = 'working-tree-source-pins'
)
$ErrorActionPreference = 'Stop'
$started = (Get-Date).ToUniversalTime().ToString('o')
$JobRoot = [IO.Path]::GetFullPath($JobRoot)
$RepositoryRoot = [IO.Path]::GetFullPath($RepositoryRoot)
if (-not $JobRoot.StartsWith('D:\CodexLab\')) { throw 'JobRoot must be beneath D:\CodexLab' }
$source = Join-Path $RepositoryRoot 'src\fg_early_l4_l14\rtx_oracle.cpp'
$contract = Join-Path $RepositoryRoot 'src\fg_early_l4_l14\contract.h'
$harness = Join-Path $RepositoryRoot 'scripts\build-fg-early-l4-l14-rtx.ps1'
$verifier = Join-Path $RepositoryRoot 'scripts\verify-fg-early-l4-l14.py'
$module = Join-Path $JobRoot 'input\cu-module-000015.bin'
$attemptRoot = Join-Path $JobRoot (Join-Path 'output' $AttemptId)
foreach ($required in @($source,$contract,$harness,$verifier,$module)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "Missing L4/L14 input: $required" }
}
if (Test-Path -LiteralPath $attemptRoot) { throw "Attempt exists: $attemptRoot" }
if ((Get-Item $module).Length -ne 12632 -or (Get-FileHash $module -Algorithm SHA256).Hash -ne '9E772DF29E9434E4EFF1EDF1CDE32706EA641C2B0156A459A080565A7483F381') { throw 'module15 identity mismatch' }
$cudaPath = if ($env:CUDA_PATH) { $env:CUDA_PATH } else { 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3' }
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$installation = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$vcvars = Join-Path $installation 'VC\Auxiliary\Build\vcvars64.bat'
$buildRoot = Join-Path $attemptRoot 'build'
New-Item -ItemType Directory -Path $buildRoot | Out-Null
$binary = Join-Path $buildRoot 'fg-early-l4-l14-rtx.exe'
$object = Join-Path $buildRoot 'fg-early-l4-l14-rtx.obj'
$compileLog = Join-Path $buildRoot 'compile.log'
$command = '"{0}" >nul && cl.exe /nologo /std:c++20 /EHsc /MD /W4 /WX /permissive- /Zc:__cplusplus /utf-8 /I"{1}" /Fo"{2}" /c "{3}" >"{4}" 2>&1 && link.exe /nologo /OUT:"{5}" "{2}" "{6}" bcrypt.lib >>"{4}" 2>&1' -f $vcvars,(Join-Path $cudaPath 'include'),$object,$source,$compileLog,$binary,(Join-Path $cudaPath 'lib\x64\cuda.lib')
& $env:ComSpec /d /s /c $command
if ($LASTEXITCODE -ne 0) { throw "L4/L14 build failed; see $compileLog" }

$receipts = @()
$processResults = @()
foreach ($fixture in @('branch-grid','center-tie')) {
  foreach ($phase in @('reset','generated')) {
    foreach ($poison in @('a5','5a')) {
      foreach ($ordinal in @('a','b')) {
        $launch = if ($phase -eq 'reset') { '4' } else { '14' }
        $runId = "l$launch-$fixture-poison-$poison-process-$ordinal"
        $runRoot = Join-Path $attemptRoot $runId
        New-Item -ItemType Directory -Path $runRoot | Out-Null
        $process = Start-Process -FilePath $binary -ArgumentList @($module,$runRoot,$runId,$poison,$phase,$fixture) -NoNewWindow -Wait -PassThru -RedirectStandardOutput (Join-Path $runRoot 'stdout.log') -RedirectStandardError (Join-Path $runRoot 'stderr.log')
        if ($process.ExitCode -ne 0) { throw "L4/L14 run failed: $runId" }
        $receipt = Get-Content (Join-Path $runRoot 'run-receipt.json') -Raw | ConvertFrom-Json
        if ($receipt.result -ne 'pass') { throw "L4/L14 receipt failed: $runId" }
        $receipts += $receipt
        $processResults += [ordered]@{run_id=$runId;process_id=$process.Id;exit_code=$process.ExitCode}
      }
    }
  }
}
$pids = @($receipts.process_id | Sort-Object -Unique)
if ($pids.Count -ne 16) { throw 'L4/L14 fresh-process gate failed' }
$scenarios = [ordered]@{}
foreach ($fixture in @('branch-grid','center-tie')) {
  foreach ($phase in @('reset','generated')) {
    $key = "$phase/$fixture"
    $selected = @($receipts | Where-Object {$_.phase -eq $phase -and $_.fixture -eq $fixture})
    $vectorHashes = @($selected.vector_output_sha256 | Sort-Object -Unique)
    $depthHashes = @($selected.depth_output_sha256 | Sort-Object -Unique)
    if ($selected.Count -ne 4 -or $vectorHashes.Count -ne 1 -or $depthHashes.Count -ne 1) { throw "L4/L14 repeat gate failed: $key" }
    $scenarios[$key] = [ordered]@{selected_launch=[int]$selected[0].selected_launch;vector_output_sha256=$vectorHashes[0];depth_output_sha256=$depthHashes[0];depth_unique_u32_values=[int]$selected[0].depth_unique_u32_values}
  }
}

$negativeRoot = Join-Path $attemptRoot 'unsupported-negative'
New-Item -ItemType Directory -Path $negativeRoot | Out-Null
$negativePhase = Start-Process -FilePath $binary -ArgumentList @($module,$negativeRoot,'unsupported-phase','a5','unsupported','branch-grid') -NoNewWindow -Wait -PassThru -RedirectStandardOutput (Join-Path $negativeRoot 'phase.stdout.log') -RedirectStandardError (Join-Path $negativeRoot 'phase.stderr.log')
$negativeFixture = Start-Process -FilePath $binary -ArgumentList @($module,$negativeRoot,'unsupported-fixture','a5','reset','unsupported') -NoNewWindow -Wait -PassThru -RedirectStandardOutput (Join-Path $negativeRoot 'fixture.stdout.log') -RedirectStandardError (Join-Path $negativeRoot 'fixture.stderr.log')
if ($negativePhase.ExitCode -ne 1 -or $negativeFixture.ExitCode -ne 1 -or (Test-Path (Join-Path $negativeRoot 'run-receipt.json'))) { throw 'L4/L14 unsupported domain did not fail closed' }

$rig = [ordered]@{
 schema='neuralstudio.fg.early-l4-l14-rig.v1';job_id=(Split-Path -Leaf $JobRoot);attempt_id=$AttemptId;base_commit=$BaseCommit;started_utc=$started;ended_utc=(Get-Date).ToUniversalTime().ToString('o');gpu=(& nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader,nounits|Select-Object -First 1).Trim();cuda_toolkit=(&(Join-Path $cudaPath 'bin\nvcc.exe') --version|Select-Object -Last 1).Trim();module_sha256=(Get-FileHash $module -Algorithm SHA256).Hash.ToLowerInvariant();source_sha256=(Get-FileHash $source -Algorithm SHA256).Hash.ToLowerInvariant();contract_sha256=(Get-FileHash $contract -Algorithm SHA256).Hash.ToLowerInvariant();harness_sha256=(Get-FileHash $harness -Algorithm SHA256).Hash.ToLowerInvariant();verifier_sha256=(Get-FileHash $verifier -Algorithm SHA256).Hash.ToLowerInvariant();binary_sha256=(Get-FileHash $binary -Algorithm SHA256).Hash.ToLowerInvariant();run_count=$receipts.Count;fresh_process_count=$pids.Count;process_results=$processResults;scenarios=$scenarios;unsupported_phase_negative=[ordered]@{process_id=$negativePhase.Id;exit_code=$negativePhase.ExitCode;receipt_created=$false;result='rejected-before-gpu-execution'};unsupported_fixture_negative=[ordered]@{process_id=$negativeFixture.Id;exit_code=$negativeFixture.ExitCode;receipt_created=$false;result='rejected-before-gpu-execution'};input_generator='fg-early-l4-l14-independent-inputs-v1';expected_output_fed_to_execution=$false;archived_intermediates_consumed=$false;address_fields_recorded=$false;result='pass'
}
$utf8 = New-Object System.Text.UTF8Encoding($false)
$rigPath = Join-Path $attemptRoot 'rig-receipt.json'
[IO.File]::WriteAllText($rigPath,(($rig|ConvertTo-Json -Depth 8)+"`n"),$utf8)
$backendPath = Join-Path $attemptRoot 'backend-receipt.json'
& python $verifier backend --backend rtx --root $attemptRoot --output $backendPath *> (Join-Path $attemptRoot 'backend-verification.log')
if ($LASTEXITCODE -ne 0) { throw 'L4/L14 RTX backend verification failed' }
$backendHash = (Get-FileHash $backendPath -Algorithm SHA256).Hash.ToLowerInvariant()
[IO.File]::WriteAllText((Join-Path $attemptRoot 'backend-receipt.sha256'),($backendHash+"  backend-receipt.json`n"),$utf8)
Write-Host "attempt_root=$attemptRoot"
Write-Host "rig_receipt_sha256=$((Get-FileHash $rigPath -Algorithm SHA256).Hash.ToLowerInvariant())"
Write-Host "backend_receipt_sha256=$backendHash"
Write-Host 'result=pass'
