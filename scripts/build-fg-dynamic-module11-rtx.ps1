[CmdletBinding()]
param(
  [Parameter(Mandatory=$true)][string]$JobRoot,
  [string]$AttemptId='oracle-v1',
  [string]$RepositoryRoot,
  [string]$BaseCommit='4af4b042530b721ca69ec6e9a82a6a0b81e1ff41'
)
$ErrorActionPreference='Stop'
if(-not $RepositoryRoot){$RepositoryRoot=Join-Path $JobRoot 'repo'}
$JobRoot=[IO.Path]::GetFullPath($JobRoot)
$RepositoryRoot=[IO.Path]::GetFullPath($RepositoryRoot)
if(-not $JobRoot.StartsWith('D:\CodexLab\',[StringComparison]::OrdinalIgnoreCase)){
  throw "JobRoot outside D:\CodexLab: $JobRoot"
}
$source=Join-Path $RepositoryRoot 'src\fg_dynamic_module11\rtx_oracle.cpp'
$contract=Join-Path $RepositoryRoot 'src\fg_dynamic_module11\contract.h'
$baseSource=Join-Path $RepositoryRoot 'src\fg_mid_l55\rtx_oracle.cpp'
$baseContract=Join-Path $RepositoryRoot 'src\fg_mid_l55\fg_mid_l55_contract.h'
$baseInputs=Join-Path $RepositoryRoot 'src\fg_mid_l55\fg_mid_l55_inputs.h'
$harness=Join-Path $RepositoryRoot 'scripts\build-fg-dynamic-module11-rtx.ps1'
$module=Join-Path $JobRoot 'inputs\cu-module-000011.bin'
$attempt=Join-Path $JobRoot "output\$AttemptId"
$build=Join-Path $attempt 'build'
foreach($file in @($source,$contract,$baseSource,
                    $baseContract,$baseInputs,$harness,$module)){
  if(-not(Test-Path -LiteralPath $file -PathType Leaf)){throw "Missing $file"}
}
if(Test-Path -LiteralPath $attempt){throw "Attempt exists: $attempt"}
if((Get-Item -LiteralPath $module).Length -ne 53920 -or
   (Get-FileHash -LiteralPath $module -Algorithm SHA256).Hash -ne
     '06EEE29EC33202996F97C063A08CCDC767D0167252AD03B5ED457C941CB85206'){
  throw 'module11 identity mismatch'
}
$vswhere=Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$install=& $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$vcvars=Join-Path $install 'VC\Auxiliary\Build\vcvars64.bat'
New-Item -ItemType Directory -Path $build | Out-Null
$env:TEMP=$build;$env:TMP=$build
$exe=Join-Path $build 'fg-dynamic-module11-rtx.exe'
$obj=Join-Path $build 'probe.obj'
$log=Join-Path $build 'compile.log'
$started=(Get-Date).ToUniversalTime().ToString('o')
$command='"{0}" >nul && cl.exe /nologo /std:c++20 /EHsc /MD /W4 /WX /permissive- /Zc:__cplusplus /utf-8 /Fo"{1}" /c "{2}" >"{3}" 2>&1 && link.exe /nologo /OUT:"{4}" "{1}" bcrypt.lib d3d12.lib dxgi.lib >>"{3}" 2>&1' -f $vcvars,$obj,$source,$log,$exe
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
           $receipt.canary_mismatched_bytes -ne 0 -or
           $receipt.typed_resource_backend -ne $true -or
           $receipt.expected_or_captured_output_ingress -ne $false -or
           $receipt.archived_intermediate_ingress -ne $false -or
           $receipt.address_fields_recorded -ne $false){
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
    if($scenario.Count -ne 4){throw "scenario count: $resolution $phase"}
    if(($scenario.combined_output_sha256 | Sort-Object -Unique).Count -ne 1){
      throw "repeat/poison mismatch: $resolution $phase"
    }
    if(@($scenario | Where-Object {$_.artifacts_written}).Count -ne 1){
      throw "artifact cardinality: $resolution $phase"
    }
  }
}
$completed=(Get-Date).ToUniversalTime().ToString('o')
$rig=[ordered]@{
  schema='neuralstudio.fg.dynamic-module11-rig.v1'
  job_id=Split-Path -Leaf $JobRoot
  attempt_id=$AttemptId
  local_base_commit=$BaseCommit
  started_utc=$started
  completed_utc=$completed
  gpu=(& nvidia-smi --query-gpu=name,driver_version,memory.total,pci.bus_id --format=csv,noheader,nounits | Select-Object -First 1).Trim()
  module_sha256=(Get-FileHash $module -Algorithm SHA256).Hash.ToLowerInvariant()
  source_sha256=(Get-FileHash $source -Algorithm SHA256).Hash.ToLowerInvariant()
  contract_sha256=(Get-FileHash $contract -Algorithm SHA256).Hash.ToLowerInvariant()
  base_source_sha256=(Get-FileHash $baseSource -Algorithm SHA256).Hash.ToLowerInvariant()
  harness_sha256=(Get-FileHash $harness -Algorithm SHA256).Hash.ToLowerInvariant()
  binary_sha256=(Get-FileHash $exe -Algorithm SHA256).Hash.ToLowerInvariant()
  resolutions=@('720p','1080p','1440p','4k')
  modes=@(2,4)
  phases=@('reset','generated')
  run_count=32
  fresh_process_count=32
  poison_bytes=@(165,90)
  full_artifact_run_count=8
  repeat_and_poison_exact=$true
  true_d3d12_nvapi_resources=$true
  input_immutable=$true
  canaries_intact=$true
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
