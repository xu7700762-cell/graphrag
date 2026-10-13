param([ValidateRange(1,65535)][int]$Port = 8880)
$ErrorActionPreference = 'Stop'
$projectDir = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectDir '.venv\Scripts\python.exe'
$serviceUrl = 'http://127.0.0.1:' + $Port
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) { throw 'Project Python is unavailable.' }

# Check every execution branch before restarting this project's idle service.
foreach ($route in @('/api/tasks', '/api/brain/sessions', '/api/online/sessions')) {
    $payload = Invoke-RestMethod -Uri ($serviceUrl + $route) -TimeoutSec 10
    $rows = if ($payload -is [array]) { $payload } elseif ($payload.PSObject.Properties.Name -contains 'sessions') { $payload.sessions } else { @($payload) }
    if (@($rows | Where-Object { $_.state -in @('created','running','paused','starting','finalizing','stopping') }).Count -gt 0) {
        throw ('Analysis is active at ' + $route + '; no process has been stopped.')
    }
}
$cloudStatus = Invoke-RestMethod -Uri ($serviceUrl + '/api/cloud-settings') -TimeoutSec 10
if ($cloudStatus.in_use) { throw 'A cloud operation is active; no process has been stopped.' }

$listeners = @(Get-NetTCPConnection -State Listen -LocalPort $Port)
$processIds = @($listeners.OwningProcess | Sort-Object -Unique)
if ($processIds.Count -ne 1) { throw 'Cannot identify a single service process.' }
$serviceProcess = Get-CimInstance Win32_Process -Filter ('ProcessId=' + $processIds[0])
$launcherProcess = Get-CimInstance Win32_Process -Filter ('ProcessId=' + $serviceProcess.ParentProcessId)
$portPattern = '--port\s+' + $Port + '(?:\s|$)'
if ($serviceProcess.Name -ne 'python.exe' -or $serviceProcess.CommandLine -notmatch '-m\s+eeg_agent\s+serve\b' -or
    $serviceProcess.CommandLine -notmatch $portPattern -or -not $launcherProcess -or
    $launcherProcess.CommandLine -notmatch [regex]::Escape($pythonPath) -or
    $launcherProcess.CommandLine -notmatch '-m\s+eeg_agent\s+serve\b') {
    throw 'Port ownership does not match this project; no process has been stopped.'
}

$logDir = Join-Path $projectDir ('outputs\service_restarts\' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
Stop-Process -Id $serviceProcess.ProcessId
for ($attempt = 0; $attempt -lt 30; $attempt++) {
    if (-not (Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue)) { break }
    Start-Sleep -Milliseconds 200
}
if (Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue) { throw 'Service port did not become free.' }
$newLauncher = Start-Process -FilePath $pythonPath -ArgumentList @('-B','-m','eeg_agent','serve','--host','127.0.0.1','--port',([string]$Port)) `
    -WorkingDirectory $projectDir -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logDir 'stdout.log') `
    -RedirectStandardError (Join-Path $logDir 'stderr.log') -PassThru
for ($attempt = 0; $attempt -lt 40; $attempt++) {
    try {
        $health = Invoke-RestMethod -Uri ($serviceUrl + '/api/health') -TimeoutSec 2
        $catalog = Invoke-RestMethod -Uri ($serviceUrl + '/api/online/catalog') -TimeoutSec 5
        $page = Invoke-WebRequest -Uri ($serviceUrl + '/online') -TimeoutSec 5
        if ($health.status -eq 'ok' -and $catalog.pipeline_id -and $page.StatusCode -eq 200) {
            $owner = Get-NetTCPConnection -State Listen -LocalPort $Port | Select-Object -First 1
            $result = [pscustomobject]@{status='passed';url=($serviceUrl + '/online');previous_pid=$serviceProcess.ProcessId;
                service_pid=$owner.OwningProcess;launcher_pid=$newLauncher.Id;health=$health.status;online_http=$page.StatusCode;logs=$logDir}
            $result | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $logDir 'restart.json') -Encoding utf8
            $result | ConvertTo-Json
            exit 0
        }
    } catch {
        if ($newLauncher.HasExited) { break }
    }
    Start-Sleep -Milliseconds 500
}
throw ('Restarted service did not pass health checks. Logs: ' + $logDir)
