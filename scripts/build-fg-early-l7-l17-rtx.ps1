[CmdletBinding()]
param(
 [Parameter(Mandatory=$true)][string]$JobRoot,
 [string]$AttemptId='oracle-v1',[string]$RepositoryRoot,
 [string]$BaseCommit='384913a'
)
$ErrorActionPreference='Stop'
if(-not $RepositoryRoot){$RepositoryRoot=Split-Path -Parent $PSScriptRoot}
$JobRoot=[IO.Path]::GetFullPath($JobRoot);$RepositoryRoot=[IO.Path]::GetFullPath($RepositoryRoot)
$source=Join-Path $RepositoryRoot 'src\fg_early_l7_l17\rtx_oracle.cpp';$contract=Join-Path $RepositoryRoot 'src\fg_early_l7_l17\contract.h';$base=Join-Path $RepositoryRoot 'src\fg_conv\rtx_oracle.cpp';$harness=Join-Path $RepositoryRoot 'scripts\build-fg-early-l7-l17-rtx.ps1';$module=Join-Path $JobRoot 'inputs\cu-module-000023.bin';$attempt=Join-Path $JobRoot "output\$AttemptId";$build=Join-Path $attempt 'build'
foreach($f in @($source,$contract,$base,$harness,$module)){if(-not(Test-Path -LiteralPath $f -PathType Leaf)){throw "Missing $f"}}
if(Test-Path $attempt){throw "Attempt exists: $attempt"}
if((Get-Item $module).Length-ne56480-or(Get-FileHash $module -Algorithm SHA256).Hash-ne'D787CF7B861295D1B4C6A80CB374230DE793AA69F288A7E724BF59278BC4A62D'){throw'module23 identity'}
$cuda=$env:CUDA_PATH;if(-not$cuda){$cuda='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3'};$vswhere=Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe';$install=&$vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath;$vcvars=Join-Path $install 'VC\Auxiliary\Build\vcvars64.bat';New-Item -ItemType Directory -Path $build|Out-Null
$exe=Join-Path $build 'fg-early-l7-l17-rtx.exe';$obj=Join-Path $build 'probe.obj';$log=Join-Path $build 'compile.log';$cmd='"{0}" >nul && cl.exe /nologo /std:c++20 /EHsc /MD /W4 /WX /permissive- /Zc:__cplusplus /utf-8 /I"{1}" /Fo"{2}" /c "{3}" >"{4}" 2>&1 && link.exe /nologo /OUT:"{5}" "{2}" "{6}" bcrypt.lib >>"{4}" 2>&1'-f$vcvars,(Join-Path $cuda 'include'),$obj,$source,$log,$exe,(Join-Path $cuda 'lib\x64\cuda.lib');&$env:ComSpec /d /s /c $cmd;if($LASTEXITCODE-ne0){throw"build failed: $log"}
$receipts=@();foreach($phase in @('reset','generated')){$launch=if($phase-eq'reset'){'l7'}else{'l17'};foreach($fixture in @('dense','sparse')){foreach($poison in @('a5','5a')){foreach($ordinal in @('a','b')){$id="$launch-$phase-$fixture-poison-$poison-process-$ordinal";$root=Join-Path $attempt $id;New-Item -ItemType Directory -Path $root|Out-Null;$p=Start-Process -FilePath $exe -ArgumentList @($module,$root,$id,$poison,$phase,$fixture) -NoNewWindow -Wait -PassThru -RedirectStandardOutput (Join-Path $root 'stdout.log') -RedirectStandardError (Join-Path $root 'stderr.log');if($p.ExitCode-ne0){throw"run $id exit=$($p.ExitCode)"};$receipts+=Get-Content (Join-Path $root 'run-receipt.json') -Raw|ConvertFrom-Json}}}}
$pids=@($receipts|%{[int64]$_.process_id}|sort -Unique);if($pids.Count-ne16){throw 'process gate'};$outputs=[ordered]@{};foreach($fixture in @('dense','sparse')){$s=@($receipts|?{$_.fixture-eq$fixture});$h=@($s|%{$_.canonical_output_sha256}|sort -Unique);if($s.Count-ne8-or$h.Count-ne1){throw "output gate $fixture"};$outputs[$fixture]=$h[0]}
$receipt=[ordered]@{schema='neuralstudio.fg.early-l7-l17-rig.v1';job_id=Split-Path -Leaf $JobRoot;attempt_id=$AttemptId;local_base_commit=$BaseCommit;gpu=(&nvidia-smi --query-gpu=name,driver_version,memory.total,pci.bus_id --format=csv,noheader,nounits|select -First 1).Trim();cuda_toolkit=(&(Join-Path $cuda 'bin\nvcc.exe')--version|select -Last 1).Trim();module_sha256=(Get-FileHash $module -Algorithm SHA256).Hash.ToLowerInvariant();source_sha256=(Get-FileHash $source -Algorithm SHA256).Hash.ToLowerInvariant();contract_sha256=(Get-FileHash $contract -Algorithm SHA256).Hash.ToLowerInvariant();harness_sha256=(Get-FileHash $harness -Algorithm SHA256).Hash.ToLowerInvariant();binary_sha256=(Get-FileHash $exe -Algorithm SHA256).Hash.ToLowerInvariant();fresh_process_count=16;canonical_outputs=$outputs;result='pass'}
$utf8=New-Object System.Text.UTF8Encoding($false)
$rp=Join-Path $attempt 'rig-receipt.json'
[IO.File]::WriteAllText($rp,(($receipt|ConvertTo-Json -Depth 5)+"`n"),$utf8)
Write-Host "attempt_root=$attempt"
Write-Host "receipt_sha256=$((Get-FileHash $rp -Algorithm SHA256).Hash.ToLowerInvariant())"
Write-Host "dense=$($outputs.dense)"
Write-Host "sparse=$($outputs.sparse)"
Write-Host 'result=pass'
