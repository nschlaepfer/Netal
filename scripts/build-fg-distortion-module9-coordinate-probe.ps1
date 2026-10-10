param(
    [Parameter(Mandatory = $true)][string]$JobRoot,
    [Parameter(Mandatory = $true)][string]$RepositoryRoot,
    [Parameter(Mandatory = $true)][string]$Attempt
)
$ErrorActionPreference = 'Stop'
$JobRoot = [IO.Path]::GetFullPath($JobRoot)
$RepositoryRoot = [IO.Path]::GetFullPath($RepositoryRoot)
if (-not $JobRoot.StartsWith('D:\CodexLab\')) {
    throw 'JobRoot must be beneath D:\CodexLab'
}
$attemptRoot = Join-Path $JobRoot "output\$Attempt"
if (Test-Path -LiteralPath $attemptRoot) {
    throw "attempt already exists: $attemptRoot"
}
New-Item -ItemType Directory -Path $attemptRoot | Out-Null
$source = Join-Path $RepositoryRoot 'src\fg_distortion_path\module9_coordinate_probe.cu'
$binary = Join-Path $attemptRoot 'module9-coordinate-probe.exe'
$samples = Join-Path $attemptRoot 'module9-coordinate-probe.bin'
$compileLog = Join-Path $attemptRoot 'compile.log'
$runLog = Join-Path $attemptRoot 'run.log'
$cudaPath = if ($env:CUDA_PATH) {
    $env:CUDA_PATH
} else {
    'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3'
}
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$installation = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$vcvars = Join-Path $installation 'VC\Auxiliary\Build\vcvars64.bat'
$command = '"{0}" >nul && "{1}" -std=c++20 -arch=sm_120 -I"{2}" -o "{3}" "{4}" >"{5}" 2>&1' -f $vcvars, (Join-Path $cudaPath 'bin\nvcc.exe'), (Join-Path $RepositoryRoot 'src\fg_distortion_path'), $binary, $source, $compileLog
& $env:ComSpec /d /s /c $command
if ($LASTEXITCODE -ne 0) {
    throw "module9 coordinate probe compile failed: $compileLog"
}
& $binary $samples *> $runLog
if ($LASTEXITCODE -ne 0) {
    throw "module9 coordinate probe execution failed: $runLog"
}
$receipt = [ordered]@{
    schema = 'neuralstudio.fg.distortion-module9-coordinate-probe.v1'
    attempt = $Attempt
    source_sha256 = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant()
    binary_sha256 = (Get-FileHash -LiteralPath $binary -Algorithm SHA256).Hash.ToLowerInvariant()
    samples_sha256 = (Get-FileHash -LiteralPath $samples -Algorithm SHA256).Hash.ToLowerInvariant()
    samples_bytes = (Get-Item -LiteralPath $samples).Length
    sample_record_bytes = 32
    sample_count = 320 * 180
    fixture_generator = 'fg-distortion-module9-independent-nonzero-v1'
    probe_fed_to_model_execution = $false
    result = 'pass'
}
$receipt | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $attemptRoot 'receipt.json') -Encoding utf8
$receipt | ConvertTo-Json -Depth 4
