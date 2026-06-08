param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]] $Allin1Args
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path

$env:MPLCONFIGDIR = Join-Path $Root ".cache\matplotlib"
$env:HF_HOME = Join-Path $Root ".cache\huggingface"
$env:TORCH_HOME = Join-Path $Root ".cache\torch"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

New-Item -ItemType Directory -Force -Path $env:MPLCONFIGDIR, $env:HF_HOME, $env:TORCH_HOME | Out-Null

$Allin1Exe = Join-Path $Root ".venv\Scripts\allin1.exe"
& $Allin1Exe @Allin1Args
exit $LASTEXITCODE
