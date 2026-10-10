$ErrorActionPreference = 'Stop'
$job = 'D:\CodexLab\nr-sampler-edge-20260906-v1'
Set-Location -LiteralPath $job
if (Test-Path -LiteralPath '.\job-receipt.json') { throw 'Preserve completed job' }
if (Test-Path -LiteralPath '.\compile.log') { throw 'Preserve earlier compile attempt; use a fresh job directory' }
$start = (Get-Date).ToUniversalTime().ToString('o')
$expected = @{
  'nr_sampler_edge_oracle.cu'='018524e2cc4f108b5b2bceb4a282fc804d3817b745bb43c14d1e936505f56b8a'
  'nr_sampler_edge_inputs.h'='827da78ed8f5a980c7544124ca65238b0b1517c6ff820c8584791174c025f256'
  'nr_finite_half_texture_oracle.cu'='19aeb1a1339636acd57513e9ae7371c03a255f0c7e57ae2c0eb667f9b1dc7e5f'
}
foreach ($name in $expected.Keys) {
  if ((Get-FileHash -Algorithm SHA256 -LiteralPath $name).Hash.ToLower() -ne $expected[$name]) { throw "Staged source identity mismatch: $name" }
}
$sources = @('nr_sampler_edge_oracle.cu','nr_sampler_edge_inputs.h','nr_finite_half_texture_oracle.cu','run-nr-sampler-edge-rtx.ps1')
$before = @{}
foreach ($s in $sources) { $before[$s] = (Get-FileHash -Algorithm SHA256 -LiteralPath $s).Hash.ToLower() }
$vswhere = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
$vs = & $vswhere -latest -products '*' -property installationPath
$vc = Join-Path $vs 'VC\Auxiliary\Build\vcvars64.bat'
$compile = '"' + $vc + '" && nvcc -std=c++17 -arch=sm_120 -O2 -lineinfo -Xcompiler /MD nr_sampler_edge_oracle.cu -o nr_sampler_edge_oracle.exe'
& cmd.exe /d /s /c $compile 2>&1 | Out-File -Encoding utf8 compile.log
if ($LASTEXITCODE -ne 0) { throw "compile exit $LASTEXITCODE" }
$results = @()
foreach ($pair in @(@('a5-a','a5'),@('a5-b','a5'),@('5a-a','5a'),@('5a-b','5a'))) {
  $label=$pair[0]; $poison=$pair[1]
  if (Test-Path -LiteralPath $label) { throw "Preserve prior $label" }
  & .\nr_sampler_edge_oracle.exe $poison $label 2>&1 | Out-File -Encoding utf8 ($label+'.log')
  $rc=$LASTEXITCODE
  $results += [ordered]@{label=$label;poison=$poison;exit_code=$rc}
  if ($rc -ne 0) { break }
}
$after = @{}
foreach ($s in $sources) { $after[$s] = (Get-FileHash -Algorithm SHA256 -LiteralPath $s).Hash.ToLower() }
$artifacts=@{}
Get-ChildItem -LiteralPath $job -File -Recurse | ForEach-Object {
  $rel=$_.FullName.Substring($job.Length+1).Replace('\','/')
  $artifacts[$rel]=[ordered]@{bytes=$_.Length;sha256=(Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLower()}
}
[ordered]@{schema='neuralstudio.nr.sampler-edge-job.v1';start_utc=$start;end_utc=(Get-Date).ToUniversalTime().ToString('o');host=$env:COMPUTERNAME;gpu=(& nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader);nvcc=(& nvcc --version);compile_command=$compile;sources_before=$before;sources_after=$after;runs=$results;artifacts=$artifacts} | ConvertTo-Json -Depth 7 | Set-Content -Encoding utf8 -LiteralPath '.\job-receipt.json'
if ($results.Count -ne 4 -or @($results | Where-Object { $_.exit_code -ne 0 }).Count -ne 0) { exit 3 }
Write-Output 'FOUR_FRESH_CAPTURES_COMPLETE'
