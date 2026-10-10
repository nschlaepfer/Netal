[CmdletBinding()]
param(
  [Parameter(Mandatory=$true)][string]$JobRoot,
  [string]$AttemptId='oracle-v1',
  [string]$RepositoryRoot,
  [string]$BaseCommit='6070a2782ebb800c7711d3c48fcf545bc1d57326'
)
$ErrorActionPreference='Stop'
if(-not $RepositoryRoot){$RepositoryRoot=Join-Path $JobRoot 'repo'}
$JobRoot=[IO.Path]::GetFullPath($JobRoot)
$RepositoryRoot=[IO.Path]::GetFullPath($RepositoryRoot)
if(-not $JobRoot.StartsWith('D:\CodexLab\',[StringComparison]::OrdinalIgnoreCase)){
  throw "JobRoot outside D:\CodexLab: $JobRoot"
}
$source=Join-Path $RepositoryRoot 'src\fg_dynamic_module15\rtx_oracle.cpp'
$contract=Join-Path $RepositoryRoot 'src\fg_dynamic_module15\contract.h'
$baseSource=Join-Path $RepositoryRoot 'src\fg_early_l4_l14\rtx_oracle.cpp'
$baseContract=Join-Path $RepositoryRoot 'src\fg_early_l4_l14\contract.h'
$driverSource=Join-Path $RepositoryRoot 'src\fg_conv\rtx_oracle.cpp'
$harness=Join-Path $RepositoryRoot 'scripts\build-fg-dynamic-module15-rtx.ps1'
$module=Join-Path $JobRoot 'inputs\cu-module-000015.bin'
$attempt=Join-Path $JobRoot "output\$AttemptId"
$build=Join-Path $attempt 'build'
foreach($file in @($source,$contract,$baseSource,$baseContract,$driverSource,
                    $harness,$module)){
  if(-not(Test-Path -LiteralPath $file -PathType Leaf)){throw "Missing $file"}
}
if(Test-Path -LiteralPath $attempt){throw "Attempt exists: $attempt"}
if((Get-Item -LiteralPath $module).Length -ne 12632 -or
   (Get-FileHash -LiteralPath $module -Algorithm SHA256).Hash -ne
     '9E772DF29E9434E4EFF1EDF1CDE32706EA641C2B0156A459A080565A7483F381'){
  throw 'module15 identity mismatch'
}
$cudaPath=if($env:CUDA_PATH){$env:CUDA_PATH}else{'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3'}
$vswhere=Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$install=& $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$vcvars=Join-Path $install 'VC\Auxiliary\Build\vcvars64.bat'
New-Item -ItemType Directory -Path $build | Out-Null
$env:TEMP=$build;$env:TMP=$build
$exe=Join-Path $build 'fg-dynamic-module15-rtx.exe'
$obj=Join-Path $build 'probe.obj'
$log=Join-Path $build 'compile.log'
$started=(Get-Date).ToUniversalTime().ToString('o')
$command='"{0}" >nul && cl.exe /nologo /std:c++20 /EHsc /MD /W4 /WX /permissive- /Zc:__cplusplus /utf-8 /I"{1}" /Fo"{2}" /c "{3}" >"{4}" 2>&1 && link.exe /nologo /OUT:"{5}" "{2}" "{6}" bcrypt.lib >>"{4}" 2>&1' -f $vcvars,(Join-Path $cudaPath 'include'),$obj,$source,$log,$exe,(Join-Path $cudaPath 'lib\x64\cuda.lib')
& $env:ComSpec /d /s /c $command
if($LASTEXITCODE -ne 0){throw "build failed: $log"}
$receipts=@()
foreach($resolution in @('720p','1080p','1440p','4k')){
  foreach($phase in @('reset','generated')){
    foreach($poison in @('a5','5a')){
      foreach($ordinal in @('a','b')){
        $id="$resolution-$phase-poison-$poison-process-$ordinal"
        $root=Join-Path $attempt $id
        New-Item -ItemType Directory -Path $root | Out-Null
        $artifactMode=if($poison -eq 'a5' -and $ordinal -eq 'a'){'full'}else{'hash-only'}
        $process=Start-Process -FilePath $exe -ArgumentList @(
          $module,$root,$id,$poison,$resolution,$phase,$artifactMode
        ) -NoNewWindow -Wait -PassThru `
          -RedirectStandardOutput (Join-Path $root 'stdout.log') `
          -RedirectStandardError (Join-Path $root 'stderr.log')
        if($process.ExitCode -ne 0){throw "run $id exit=$($process.ExitCode)"}
        $receipt=Get-Content (Join-Path $root 'run-receipt.json') -Raw | ConvertFrom-Json
        if($receipt.result -ne 'pass' -or $receipt.input_mismatched_bytes -ne 0 -or
           $receipt.vector_canary_mismatched_bytes -ne 0 -or
           $receipt.depth_canary_mismatched_bytes -ne 0 -or
           $receipt.vector_untouched_pixels -ne 0 -or
           $receipt.depth_untouched_pixels -ne 0 -or
           $receipt.expected_or_captured_output_ingress -ne $false -or
           $receipt.archived_intermediate_ingress -ne $false){
          throw "receipt rejected $id"
        }
        $receipts += $receipt
      }
    }
  }
}
$pids=@($receipts | ForEach-Object {[int64]$_.process_id} | Sort-Object -Unique)
if($pids.Count -ne 32){throw "fresh process gate: $($pids.Count)/32"}
foreach($resolution in @('720p','1080p','1440p','4k')){
  foreach($phase in @('reset','generated')){
    $scenario=@($receipts | Where-Object {
      $_.resolution -eq $resolution -and $_.phase -eq $phase
    })
    if($scenario.Count -ne 4 -or
       ($scenario.vector_output_sha256 | Sort-Object -Unique).Count -ne 1 -or
       ($scenario.depth_output_sha256 | Sort-Object -Unique).Count -ne 1){
      throw "repeat/poison mismatch: $resolution $phase"
    }
  }
}
$rig=[ordered]@{
  schema='neuralstudio.fg.dynamic-module15-rig.v1'
  job_id=Split-Path -Leaf $JobRoot
  attempt_id=$AttemptId
  local_base_commit=$BaseCommit
  started_utc=$started
  completed_utc=(Get-Date).ToUniversalTime().ToString('o')
  gpu=(& nvidia-smi --query-gpu=name,driver_version,memory.total,pci.bus_id --format=csv,noheader,nounits | Select-Object -First 1).Trim()
  module_sha256=(Get-FileHash $module -Algorithm SHA256).Hash.ToLowerInvariant()
  source_sha256=(Get-FileHash $source -Algorithm SHA256).Hash.ToLowerInvariant()
  contract_sha256=(Get-FileHash $contract -Algorithm SHA256).Hash.ToLowerInvariant()
  base_source_sha256=(Get-FileHash $baseSource -Algorithm SHA256).Hash.ToLowerInvariant()
  harness_sha256=(Get-FileHash $harness -Algorithm SHA256).Hash.ToLowerInvariant()
  binary_sha256=(Get-FileHash $exe -Algorithm SHA256).Hash.ToLowerInvariant()
  resolutions=@('720p','1080p','1440p','4k')
  phases=@('reset','generated')
  run_count=32
  fresh_process_count=32
  poison_bytes=@(165,90)
  repeat_and_poison_exact=$true
  true_typed_cuda_arrays=$true
  input_immutable=$true
  canary_rows_intact=$true
  full_write_coverage=$true
  expected_or_captured_output_ingress=$false
  archived_intermediate_ingress=$false
  address_fields_recorded=$false
  result='pass'
}
$utf8=New-Object System.Text.UTF8Encoding($false)
$receiptPath=Join-Path $attempt 'rig-receipt.json'
[IO.File]::WriteAllText($receiptPath,(($rig | ConvertTo-Json -Depth 8)+"`n"),$utf8)
Write-Host "attempt_root=$attempt"
Write-Host "rig_receipt_sha256=$((Get-FileHash $receiptPath -Algorithm SHA256).Hash.ToLowerInvariant())"
Write-Host 'result=pass'
