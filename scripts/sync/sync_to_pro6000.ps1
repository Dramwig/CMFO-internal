param(
    [string]$HostAlias = "pro6000",
    [string]$RemoteRoot = "/root/autodl-tmp/CMFO/CMFO-internal"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path

ssh $HostAlias "mkdir -p '$RemoteRoot'"
& tar --exclude=.git --exclude=.venv --exclude=__pycache__ --exclude=.pytest_cache --exclude=outputs --exclude=logs --exclude=tmp -cf - -C $repoRoot . | & ssh $HostAlias "tar -xf - -C '$RemoteRoot'"

if ($LASTEXITCODE -ne 0) {
    throw "CMFO sync failed with exit code $LASTEXITCODE"
}

ssh $HostAlias "test -f '$RemoteRoot/pyproject.toml' && test -f '$RemoteRoot/src/cmfo/cli.py'"

