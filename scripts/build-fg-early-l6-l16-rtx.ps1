[CmdletBinding()]
param(
 [Parameter(Mandatory=$true)][string]$JobRoot,
 [string]$AttemptId='oracle-v1',[string]$RepositoryRoot,
 [string]$BaseCommit='6ec4e9c'
)
$ErrorActionPreference='Stop'
if(-not$RepositoryRoot){$RepositoryRoot=Join-Path $JobRoot 'repo'}
$JobRoot=[IO.Path]::GetFullPath($JobRoot);$RepositoryRoot=[IO.Path]::GetFullPath($RepositoryRoot)
if(-not$JobRoot.StartsWith('D:\CodexLab\',[StringComparison]::OrdinalIgnoreCase)){throw"JobRoot outside D:\CodexLab: $JobRoot"}
$source=Join-Path $RepositoryRoot 'src\fg_early_l6_l16\rtx_nvapi_oracle.cpp'
$contract=Join-Path $RepositoryRoot 'src\fg_early_l6_l16\contract.h'
$baseSource=Join-Path $RepositoryRoot 'src\fg_mid_l52\rtx_oracle.cpp'
$harness=Join-Path $RepositoryRoot 'scripts\build-fg-early-l6-l16-rtx.ps1'
$module=Join-Path $JobRoot 'inputs\cu-module-000021.bin'
$attempt=Join-Path $JobRoot "output\$AttemptId";$build=Join-Path $attempt 'build'
foreach($f in @($source,$contract,$baseSource,$harness,$module)){if(-not(Test-Path -LiteralPath $f -PathType Leaf)){throw"Missing $f"}}
if(Test-Path -LiteralPath $attempt){throw"Attempt exists: $attempt"}
if((Get-Item -LiteralPath $module).Length-ne38472-or(Get-FileHash -LiteralPath $module -Algorithm SHA256).Hash-ne'11DF926493FCA0A981D560427B36C4DCF489CCB84D794501A70BDA8D414EA8A2'){throw'module21 identity'}
$vswhere=Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe';$install=&$vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath;$vcvars=Join-Path $install 'VC\Auxiliary\Build\vcvars64.bat'
New-Item -ItemType Directory -Path $build|Out-Null;$env:TEMP=$build;$env:TMP=$build
$exe=Join-Path $build 'fg-early-l6-l16-rtx.exe';$obj=Join-Path $build 'probe.obj';$log=Join-Path $build 'compile.log';$started=(Get-Date).ToUniversalTime().ToString('o')
$cmd='"{0}" >nul && cl.exe /nologo /std:c++20 /EHsc /MD /W4 /WX /permissive- /Zc:__cplusplus /utf-8 /Fo"{1}" /c "{2}" >"{3}" 2>&1 && link.exe /nologo /OUT:"{4}" "{1}" bcrypt.lib d3d12.lib dxgi.lib >>"{3}" 2>&1'-f$vcvars,$obj,$source,$log,$exe
&$env:ComSpec /d /s /c $cmd;if($LASTEXITCODE-ne0){throw"build failed: $log"}
$receipts=@();foreach($phase in @('reset','generated')){$launch=if($phase-eq'reset'){'l6'}else{'l16'};foreach($poison in @('a5','5a')){foreach($ordinal in @('a','b')){$id="$launch-$phase-poison-$poison-process-$ordinal";$root=Join-Path $attempt $id;New-Item -ItemType Directory -Path $root|Out-Null;$p=Start-Process -FilePath $exe -ArgumentList @($module,$root,$id,$poison,$phase) -NoNewWindow -Wait -PassThru -RedirectStandardOutput (Join-Path $root 'stdout.log') -RedirectStandardError (Join-Path $root 'stderr.log');if($p.ExitCode-ne0){throw"run $id exit=$($p.ExitCode)"};$r=Get-Content (Join-Path $root 'run-receipt.json') -Raw|ConvertFrom-Json;if($r.result-ne'pass'-or$r.input_mismatched_bytes-ne0-or$r.buffer_canary_mismatched_bytes-ne0){throw"receipt rejected $id"};$receipts+=$r}}}
$pids=@($receipts|%{[int64]$_.process_id}|sort -Unique);if($pids.Count-ne8){throw'fresh process gate'}
$completed=(Get-Date).ToUniversalTime().ToString('o')
$receipt=[ordered]@{schema='neuralstudio.fg.early-l6-l16-rig.v1';job_id=Split-Path -Leaf $JobRoot;attempt_id=$AttemptId;local_base_commit=$BaseCommit;started_utc=$started;completed_utc=$completed;gpu=(&nvidia-smi --query-gpu=name,driver_version,memory.total,pci.bus_id --format=csv,noheader,nounits|select -First 1).Trim();module_sha256=(Get-FileHash $module -Algorithm SHA256).Hash.ToLowerInvariant();source_sha256=(Get-FileHash $source -Algorithm SHA256).Hash.ToLowerInvariant();contract_sha256=(Get-FileHash $contract -Algorithm SHA256).Hash.ToLowerInvariant();base_source_sha256=(Get-FileHash $baseSource -Algorithm SHA256).Hash.ToLowerInvariant();harness_sha256=(Get-FileHash $harness -Algorithm SHA256).Hash.ToLowerInvariant();binary_sha256=(Get-FileHash $exe -Algorithm SHA256).Hash.ToLowerInvariant();run_count=8;fresh_process_count=8;poison_bytes=@(165,90);true_d3d12_nvapi_resources=$true;input_immutable=$true;buffer_canaries_intact=$true;expected_or_captured_output_ingress=$false;address_fields_recorded=$false;result='pass'}
$utf8=New-Object System.Text.UTF8Encoding($false);$path=Join-Path $attempt 'rig-receipt.json';[IO.File]::WriteAllText($path,(($receipt|ConvertTo-Json -Depth 6)+"`n"),$utf8)
Write-Host "attempt_root=$attempt";Write-Host "rig_receipt_sha256=$((Get-FileHash $path -Algorithm SHA256).Hash.ToLowerInvariant())";Write-Host result=pass
