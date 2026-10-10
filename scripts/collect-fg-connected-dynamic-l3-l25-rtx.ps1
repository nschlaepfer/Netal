[CmdletBinding()]
param(
  [Parameter(Mandatory=$true)][string]$JobRoot,
  [string]$AttemptId='oracle-v1',
  [string]$CollectionId='retrieval-v1'
)
$ErrorActionPreference='Stop'
$JobRoot=[IO.Path]::GetFullPath($JobRoot)
if(-not $JobRoot.StartsWith('D:\CodexLab\',[StringComparison]::OrdinalIgnoreCase)){
  throw 'D: job required'
}
$source=Join-Path $JobRoot "output\$AttemptId"
$target=Join-Path $JobRoot "analysis\$CollectionId"
if(-not(Test-Path -LiteralPath $source -PathType Container)){throw 'attempt missing'}
if(Test-Path -LiteralPath $target){throw 'collection exists'}
New-Item -ItemType Directory -Path $target|Out-Null
Copy-Item -LiteralPath (Join-Path $source 'rig-receipt.json') -Destination $target
$generator_paths=[ordered]@{
  'generator21-receipt.json'='build\base21\generator-receipt.json'
  'generator23-receipt.json'='build\base23\generator-receipt.json'
  'generator25-receipt.json'='build\generated\generator-receipt.json'
}
foreach($entry in $generator_paths.GetEnumerator()){
  Copy-Item -LiteralPath (Join-Path $source $entry.Value) -Destination (Join-Path $target $entry.Key)
}
foreach($resolution in @('720p','1080p','1440p','4k')){
  foreach($poison in @('a5','5a')){
    foreach($ordinal in @('a','b')){
      $run="$resolution-poison-$poison-process-$ordinal"
      $run_source=Join-Path $source $run
      $run_target=Join-Path $target $run
      New-Item -ItemType Directory -Path $run_target|Out-Null
      Copy-Item -LiteralPath (Join-Path $run_source 'run-receipt.json') -Destination $run_target
      Copy-Item -LiteralPath (Join-Path $run_source 'l24-l25-receipt.json') -Destination $run_target
      if($poison-eq'a5'-and$ordinal-eq'a'){
        Copy-Item -LiteralPath (Join-Path $run_source 'l24-output.fp16') -Destination $run_target
        Copy-Item -LiteralPath (Join-Path $run_source 'l25-output.fp16') -Destination $run_target
      }
    }
  }
}
$files=@()
foreach($file in Get-ChildItem -LiteralPath $target -Recurse -File | Sort-Object FullName){
  $relative=$file.FullName.Substring($target.Length+1).Replace('\','/')
  $files+=[ordered]@{
    path=$relative
    bytes=$file.Length
    sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
  }
}
$receipt=[ordered]@{
  schema='neuralstudio.fg.connected-dynamic-l3-l25-collection.v1'
  source_attempt_id=$AttemptId
  collection_id=$CollectionId
  collector_sha256=(Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
  source_rig_receipt_sha256=(Get-FileHash -LiteralPath (Join-Path $source 'rig-receipt.json') -Algorithm SHA256).Hash.ToLowerInvariant()
  run_receipt_count=16
  group_receipt_count=16
  full_artifact_resolution_count=4
  files=$files
  absolute_addresses_recorded=$false
  expected_or_archived_intermediate_included=$false
  result='pass'
}
$utf8=New-Object System.Text.UTF8Encoding($false)
$receipt_path=Join-Path $target 'collection-receipt.json'
[IO.File]::WriteAllText($receipt_path,(($receipt|ConvertTo-Json -Depth 6)+"`n"),$utf8)
Write-Host "collection_root=$target"
Write-Host "collection_receipt_sha256=$((Get-FileHash $receipt_path -Algorithm SHA256).Hash.ToLowerInvariant())"
Write-Host result=pass
