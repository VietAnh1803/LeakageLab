param([string]$Python = 'python', [switch]$Smoke)
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    $env:PYTHONPATH = Join-Path $PSScriptRoot 'src'
    $env:OMP_NUM_THREADS = '2'
    $env:OPENBLAS_NUM_THREADS = '2'
    $env:MKL_NUM_THREADS = '2'
    & $Python -m pytest -q 2>&1 | Tee-Object -FilePath 'artifacts/tests-local.log'
    if ($LASTEXITCODE -ne 0) { throw 'Tests failed' }
    if ($Smoke) {
        & $Python -m leakagelab.run --smoke --output artifacts/smoke
    } else {
        & $Python -m leakagelab.run --output artifacts/local 2>&1 | Tee-Object -FilePath 'artifacts/local-run.log'
    }
    if ($LASTEXITCODE -ne 0) { throw 'Benchmark failed' }
} finally {
    Pop-Location
}
