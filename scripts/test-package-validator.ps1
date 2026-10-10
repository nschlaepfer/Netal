<#
.SYNOPSIS
    Exercise the package validator's trust and containment boundaries.

.DESCRIPTION
    Creates disposable copies of the open reference package and proves that a
    valid package passes while path traversal, missing licenses, hash changes,
    invalid signatures, ambiguous adapter declarations, untrusted located
    runtimes, and unauthorized release contents are rejected.
#>
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$repository = Split-Path -Parent $PSScriptRoot
$validator = Join-Path $repository 'tools\validate-package.ps1'
$reference = Join-Path $repository 'packages\org.neuralstudio.test.passthrough'
$pwsh = Join-Path $PSHOME 'pwsh.exe'
if (-not (Test-Path -LiteralPath $pwsh -PathType Leaf)) {
    $pwsh = (Get-Process -Id $PID).Path
}

$testRoot = Join-Path ([IO.Path]::GetTempPath()) ('neuralstudio-validator-' + [guid]::NewGuid().ToString('N'))
$script:checks = 0

function New-TestPackage {
    param([Parameter(Mandatory = $true)][string]$Name)
    $path = Join-Path $testRoot $Name
    Copy-Item -LiteralPath $reference -Destination $path -Recurse
    return $path
}

function Edit-Manifest {
    param(
        [Parameter(Mandatory = $true)][string]$PackagePath,
        [Parameter(Mandatory = $true)][scriptblock]$Edit
    )
    $path = Join-Path $PackagePath 'package.json'
    $manifest = Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
    & $Edit $manifest
    $manifest | ConvertTo-Json -Depth 100 | Set-Content -LiteralPath $path -Encoding utf8NoBOM
}

function Invoke-Validation {
    param(
        [Parameter(Mandatory = $true)][string]$PackagePath,
        [switch]$ReleaseGate
    )
    $arguments = @('-NoProfile', '-File', $validator, '-PackagePath', $PackagePath, '-AsJson')
    if ($ReleaseGate) { $arguments += '-ReleaseGate' }
    $output = & $pwsh @arguments 2>&1
    $exitCode = $LASTEXITCODE
    $text = ($output | Out-String).Trim()
    if ([string]::IsNullOrWhiteSpace($text)) {
        throw "Validator returned no JSON for $PackagePath"
    }
    [pscustomobject]@{
        ExitCode = $exitCode
        Report = $text | ConvertFrom-Json
    }
}

function Assert-Rule {
    param(
        [Parameter(Mandatory = $true)]$Result,
        [Parameter(Mandatory = $true)][string]$Rule,
        [Parameter(Mandatory = $true)][ValidateSet('Pass', 'Fail', 'Warn', 'Skip')][string]$Status
    )
    $match = @($Result.Report.rules | Where-Object { $_.Rule -eq $Rule -and $_.Status -eq $Status })
    if ($match.Count -ne 1) {
        throw "Expected exactly one $Status result for '$Rule'; found $($match.Count)."
    }
    $script:checks++
    Write-Host "  PASS  $Rule -> $Status"
}

try {
    New-Item -ItemType Directory -Path $testRoot | Out-Null
    Write-Host 'NeuralStudio package-validator security tests'

    $valid = Invoke-Validation -PackagePath $reference
    if ($valid.ExitCode -ne 0 -or -not $valid.Report.valid) { throw 'Reference package did not validate.' }
    Assert-Rule $valid 'package.license.file' 'Pass'
    Assert-Rule $valid 'backend.adapter.library' 'Pass'

    $traversal = New-TestPackage 'path-traversal'
    Edit-Manifest $traversal { param($m) $m.provenance.artifacts[0].path = '..\escape.dll'; $m.backend.adapter.library = '..\escape.dll' }
    $result = Invoke-Validation $traversal
    if ($result.ExitCode -eq 0) { throw 'Path traversal was accepted.' }
    Assert-Rule $result 'provenance.artifacts.sha256[..\escape.dll]' 'Fail'
    Assert-Rule $result 'backend.adapter.library' 'Fail'

    $license = New-TestPackage 'missing-license'
    Remove-Item -LiteralPath (Join-Path $license 'LICENSE')
    $result = Invoke-Validation $license
    if ($result.ExitCode -eq 0) { throw 'Missing license was accepted.' }
    Assert-Rule $result 'package.license.file' 'Fail'

    $tampered = New-TestPackage 'tampered-adapter'
    Add-Content -LiteralPath (Join-Path $tampered 'ns_adapter_test.dll') -Value 'tampered'
    $result = Invoke-Validation $tampered
    if ($result.ExitCode -eq 0) { throw 'Tampered adapter was accepted.' }
    Assert-Rule $result 'provenance.artifacts.sha256[ns_adapter_test.dll]' 'Fail'

    $signature = New-TestPackage 'invalid-signature'
    Edit-Manifest $signature {
        param($m)
        $m.provenance.artifacts[0].signature.required = $true
        $m.provenance.artifacts[0].signature.kind = 'authenticode'
        $m.provenance.artifacts[0].signature | Add-Member -NotePropertyName expectedSigner -NotePropertyValue 'NeuralStudio'
    }
    $result = Invoke-Validation $signature
    if ($result.ExitCode -eq 0) { throw 'Invalid signature was accepted.' }
    Assert-Rule $result 'provenance.artifacts.signature[ns_adapter_test.dll]' 'Fail'

    $runtime = New-TestPackage 'untrusted-runtime'
    Edit-Manifest $runtime {
        param($m)
        $m.backend.runtimeAcquisition.method = 'locate'
        $m.backend.runtimeAcquisition.PSObject.Properties.Remove('signature')
    }
    $result = Invoke-Validation $runtime
    if ($result.ExitCode -eq 0) { throw 'Located runtime without trust policy was accepted.' }
    Assert-Rule $result 'backend.runtimeAcquisition.signature' 'Fail'

    $mismatch = New-TestPackage 'adapter-mismatch'
    Edit-Manifest $mismatch { param($m) $m.backend.adapter.library = 'different.dll' }
    $result = Invoke-Validation $mismatch
    if ($result.ExitCode -eq 0) { throw 'Unbound adapter library was accepted.' }
    Assert-Rule $result 'backend.adapter.library' 'Fail'

    $release = New-TestPackage 'release-refusal'
    Edit-Manifest $release { param($m) $m.provenance.redistribution.allowed = $false; $m.provenance.redistribution.rationale = 'Test refusal.' }
    $result = Invoke-Validation $release -ReleaseGate
    if ($result.ExitCode -eq 0) { throw 'Unauthorized release package was accepted.' }
    Assert-Rule $result 'release.redistribution' 'Fail'

    Write-Host "package_validator.checks=$script:checks"
    Write-Host 'probe=package_validator_security_passed'
} finally {
    if (Test-Path -LiteralPath $testRoot) {
        Remove-Item -LiteralPath $testRoot -Recurse -Force
    }
}
