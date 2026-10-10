$ErrorActionPreference = 'Stop'
$job = $args[0]
$dir = "D:\CodexLab\$job"
Set-Location $dir
$vs = & 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe' -latest -products * -property installationPath
$started = (Get-Date).ToUniversalTime().ToString('o')
# nvcc shells out to cl.exe and will not find it unless the MSVC toolchain is on PATH.
$cl = Get-ChildItem -Path "$vs\VC\Tools\MSVC" -Recurse -Filter cl.exe -ErrorAction SilentlyContinue |
      Where-Object { $_.FullName -like '*Hostx64\x64*' } | Select-Object -First 1
if (-not $cl) { [ordered]@{ stage='locate-cl'; vs=$vs } | ConvertTo-Json; exit 1 }
$env:PATH = "$($cl.DirectoryName);$env:PATH"
$build = & nvcc -O3 -arch=sm_120 -ccbin "$($cl.DirectoryName)" lg2_audit.cu -o lg2_audit.exe 2>&1
$buildCode = $LASTEXITCODE
if ($buildCode -ne 0) {
  [ordered]@{ stage='build'; exit_code=$buildCode; output=($build | Out-String) } | ConvertTo-Json -Depth 4
  exit 1
}
$out = & .\lg2_audit.exe binade.bin 2>&1
$runCode = $LASTEXITCODE
$ended = (Get-Date).ToUniversalTime().ToString('o')
$hash = (Get-FileHash -Algorithm SHA256 binade.bin).Hash.ToLower()
[ordered]@{
  schema        = 'neuralstudio.sr.lg2-mantissa-audit.v1'
  job_id        = $job
  started_utc   = $started
  ended_utc     = $ended
  build_exit    = $buildCode
  run_exit      = $runCode
  gpu           = (nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader)
  cuda_toolkit  = ((& nvcc --version | Select-Object -Last 1))
  msvc          = $cl.FullName
  instruction   = 'lg2.approx.ftz.f32'
  source_sha256 = (Get-FileHash -Algorithm SHA256 lg2_audit.cu).Hash.ToLower()
  output_sha256 = $hash
  output_bytes  = (Get-Item binade.bin).Length
  result        = ($out | Out-String).Trim()
} | ConvertTo-Json -Depth 4
