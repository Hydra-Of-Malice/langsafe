# Relaunches the full Earth chain after a restart. Finished stages are skipped, so only the unfinished one runs again.
#     powershell -ExecutionPolicy Bypass -File scripts/resume_full_run.ps1
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo '.venv\Scripts\python.exe'
$running = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*run_chain.py*' }
if ($running) {
    Write-Output "The chain is already running (PID $(($running | Select-Object -First 1).ProcessId)). Nothing started."
    exit 0
}
$steps = 'earth-baseline', 'earth-circuit', 'earth-repairs', 'earth-full', 'earth-nf4'
$chain = Start-Process -FilePath $python -ArgumentList (@('-u', 'scripts/run_chain.py') + $steps) -WorkingDirectory $repo `
    -WindowStyle Hidden -RedirectStandardOutput "$repo\runs\chain.out" -RedirectStandardError "$repo\runs\chain.err" -PassThru
Write-Output "Chain started, PID $($chain.Id). Progress: runs/chain-status.json"
