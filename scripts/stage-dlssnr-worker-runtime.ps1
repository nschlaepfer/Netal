<# Compose the minimal, hash-pinned DLSSNR worker runtime in build/. #>
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$repository = Split-Path -Parent $PSScriptRoot
$buildRoot = [System.IO.Path]::GetFullPath((Join-Path $repository 'build'))
$target = [System.IO.Path]::GetFullPath((Join-Path $buildRoot 'dlssnr_bridge_worker\runtime'))
if (-not $target.StartsWith($buildRoot + [System.IO.Path]::DirectorySeparatorChar,
                            [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'Resolved runtime staging path escaped the repository build directory.'
}

$artifacts = @(
    @{
        Source = Join-Path $repository 'runtime\neuralstudio-dlssnr-worker\sl.interposer.dll'
        Relative = 'sl.interposer.dll'
        SHA256 = 'D7138D0D345569CA40FB36FADEAABB2269041BC85049D76D718328431E7EFE05'
        Signed = $false
    },
    @{
        Source = Join-Path $repository 'runtime\neuralstudio-dlssnr-worker\sl.common.dll'
        Relative = 'sl.common.dll'
        SHA256 = '5C144677DE65CCD9759292C6F73196E8C19F4BE423698DB2D73705DEED9F73BA'
        Signed = $false
    },
    @{
        Source = Join-Path $repository 'runtime\neuralstudio-dlssnr-worker\neuralstudio\nvngx.dll'
        Relative = 'neuralstudio\nvngx.dll'
        SHA256 = 'EF0F22BCDF0D8667CA762944B3AC4637B1B969DBF5F47D8671A9B085F064DF40'
        Signed = $false
    },
    @{
        Source = Join-Path $repository 'runtime\streamline-2.13\sl.dlss_nr.dll'
        Relative = 'sl.dlss_nr.dll'
        SHA256 = '9F6672E5E0170DC118A3188D21BDA187E1FC1AA3502895B21AB846D23165C11D'
        Signed = $true
    },
    @{
        Source = Join-Path $repository 'models\nvidia\nvngx_dlssnr.dll'
        Relative = 'nvngx_dlssnr.dll'
        SHA256 = 'E16BCF15E16E13F527491CDF7845B2FE6521A738D8F7C9C721866A8496E1FC8E'
        Signed = $true
        HardLink = $true
    }
)

foreach ($artifact in $artifacts) {
    if (-not (Test-Path -LiteralPath $artifact.Source -PathType Leaf)) {
        throw "Required worker runtime artifact is missing: $($artifact.Source)"
    }
    $hash = (Get-FileHash -LiteralPath $artifact.Source -Algorithm SHA256).Hash
    if ($hash -ne $artifact.SHA256) {
        throw "Worker runtime artifact hash mismatch: $($artifact.Source)"
    }
    if ($artifact.Signed) {
        $signature = Get-AuthenticodeSignature -LiteralPath $artifact.Source
        if ($signature.Status -ne 'Valid' -or
            $signature.SignerCertificate.Subject -notmatch 'NVIDIA') {
            throw "NVIDIA signature validation failed: $($artifact.Source)"
        }
    }
}

if (Test-Path -LiteralPath $target) {
    Remove-Item -LiteralPath $target -Recurse -Force
}
New-Item -ItemType Directory -Path (Join-Path $target 'neuralstudio') -Force | Out-Null
foreach ($artifact in $artifacts) {
    $destination = Join-Path $target $artifact.Relative
    if ($artifact.HardLink) {
        New-Item -ItemType HardLink -Path $destination -Target $artifact.Source | Out-Null
    } else {
        Copy-Item -LiteralPath $artifact.Source -Destination $destination
    }
}

Write-Host "WorkerRuntime=$target"
Write-Host 'worker_runtime.hashes_valid=true'
Write-Host 'worker_runtime.nvidia_signatures_valid=true'
Write-Host 'worker_runtime.model_staged_as_hardlink=true'
