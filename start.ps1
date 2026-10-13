[CmdletBinding()]
param([switch]$Restart)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$pythonExe = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$serviceUrl = 'http://127.0.0.1:8880'
$listeners = @(Get-NetTCPConnection -State Listen -LocalPort 8880 -ErrorAction SilentlyContinue)

if ($listeners.Count -gt 0) {
    $ownerIds = @($listeners.OwningProcess | Select-Object -Unique)
    if ($ownerIds.Count -ne 1) {
        throw '端口8880存在多个占用进程，未停止任何进程。'
    }
    $owner = Get-CimInstance Win32_Process -Filter ('ProcessId=' + $ownerIds[0])
    $parent = Get-CimInstance Win32_Process -Filter ('ProcessId=' + $owner.ParentProcessId)
    $servicePattern = '\beeg_agent\s+serve\s+(?:--host\s+127\.0\.0\.1\s+)?--port\s+8880(?:\s|$)'
    $isProjectService = $owner.Name -eq 'python.exe' -and $owner.CommandLine -match $servicePattern -and
        ($owner.ExecutablePath -eq $pythonExe -or
         ($parent -and $parent.ExecutablePath -eq $pythonExe -and $parent.CommandLine -match $servicePattern))
    if (-not $isProjectService) {
        throw ('端口8880由其他程序占用，PID=' + $ownerIds[0] + '。未停止任何进程。')
    }
    if (-not $Restart) {
        Write-Host ('EEG服务已运行：' + $serviceUrl + '，PID=' + $owner.ProcessId)
        Write-Host "需要加载代码更新时运行：& '$PSCommandPath' -Restart"
        return
    }

    $sessions = Invoke-RestMethod -Uri ($serviceUrl + '/api/brain/sessions') -TimeoutSec 5
    $tasks = Invoke-RestMethod -Uri ($serviceUrl + '/api/tasks') -TimeoutSec 5
    $cloud = Invoke-RestMethod -Uri ($serviceUrl + '/api/cloud-settings') -TimeoutSec 5
    if ($null -eq $cloud.in_use -or $cloud.in_use -or
        @($sessions | Where-Object { $_.state -eq 'running' }).Count -gt 0 -or
        @($tasks | Where-Object { $_.state -in @('created', 'queued', 'running', 'finalizing', 'stopping') }).Count -gt 0) {
        throw '存在运行中的分析或云端操作，或状态无法核验；未停止服务。请先结束任务。'
    }

    $stopIds = @($owner.ProcessId)
    if ($parent -and $parent.ExecutablePath -eq $pythonExe -and $parent.CommandLine -match $servicePattern) {
        $stopIds += $parent.ProcessId
    }
    foreach ($serviceProcessId in $stopIds) {
        if (Get-Process -Id $serviceProcessId -ErrorAction SilentlyContinue) {
            Stop-Process -Id $serviceProcessId
        }
    }
    for ($attempt = 0; $attempt -lt 10; $attempt++) {
        if (-not (Get-NetTCPConnection -State Listen -LocalPort 8880 -ErrorAction SilentlyContinue)) { break }
        Start-Sleep -Milliseconds 200
    }
    if (Get-NetTCPConnection -State Listen -LocalPort 8880 -ErrorAction SilentlyContinue) {
        throw '旧服务端口尚未释放，未启动重复进程。'
    }
    Write-Host '旧EEG服务已停止，正在加载最新代码。'
}

& $pythonExe -B -m eeg_agent serve --port 8880
if ($LASTEXITCODE -ne 0) {
    throw ('EEG服务启动失败，退出码=' + $LASTEXITCODE)
}
