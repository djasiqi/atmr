param(
    [int]$DurationMinutes = 30,
    [int]$IntervalSeconds = 180,
    [int]$DriverId = 77777
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

$startTs = Get-Date
$endTs = $startTs.AddMinutes($DurationMinutes)
$stamp = $startTs.ToString("yyyyMMdd-HHmmss")

$outDir = Join-Path $PSScriptRoot "output"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$soakLog = Join-Path $outDir "phase4-stream-c3-soak-$stamp.log"
$eventFile = Join-Path $outDir "phase4-stream-c3-event.jsonl"

function Write-LogLine {
    param([string]$Line)
    $Line | Out-File -FilePath $soakLog -Append -Encoding utf8
}

Write-LogLine "PHASE4_C3_SOAK_START=$($startTs.ToString('o'))"
Write-LogLine "PHASE4_C3_SOAK_END_TARGET=$($endTs.ToString('o'))"
Write-LogLine "DURATION_MINUTES=$DurationMinutes INTERVAL_SECONDS=$IntervalSeconds DRIVER_ID=$DriverId"

$iteration = 0
try {
    while ((Get-Date) -lt $endTs) {
        $iteration += 1
        $iterStart = Get-Date
        $eventId = "ws-c3-soak-$stamp-$iteration"
        $payload = @{
            driver_id = $DriverId
            type = "mission.update"
            payload = @{
                event_id = $eventId
                status = "assigned"
                source = "phase4-c3-soak"
            }
        } | ConvertTo-Json -Compress

        # ASCII avoids UTF-8 BOM so Kafka payload remains strict JSON.
        Set-Content -Path $eventFile -Value $payload -Encoding ascii

        docker cp "$eventFile" atmr-kafka-broker-1:/tmp/phase4-c3-event.jsonl | Out-Null
        docker exec atmr-kafka-broker-1 bash -lc "kafka-console-producer --bootstrap-server kafka-broker-1:29092 --topic mission.events < /tmp/phase4-c3-event.jsonl" | Out-Null

        Start-Sleep -Seconds 3

        $health = docker exec atmr-ws-service python -c "import urllib.request; print(urllib.request.urlopen('http://localhost:8001/health', timeout=10).read().decode())"
        $groupRaw = docker exec atmr-kafka-broker-1 kafka-consumer-groups --bootstrap-server kafka-broker-1:29092 --describe --group ws-service-shared 2>$null
        $groupLines = @($groupRaw | Where-Object { $_ -match "^ws-service-shared\s+" })
        $maxLag = 0
        $activeMember = 0
        foreach ($line in $groupLines) {
            $cols = @($line -split "\s+" | Where-Object { $_ -ne "" })
            if ($cols.Count -ge 5) {
                $lagCandidate = 0
                if ([int]::TryParse($cols[4], [ref]$lagCandidate)) {
                    if ($lagCandidate -gt $maxLag) {
                        $maxLag = $lagCandidate
                    }
                }
            }
            if ($cols.Count -ge 6 -and $cols[5] -ne "-") {
                $activeMember = 1
            }
        }
        $groupSnapshot = "MAX_LAG=$maxLag ACTIVE_MEMBER=$activeMember PARTITIONS=$($groupLines.Count)"
        $restartCount = docker inspect atmr-ws-service --format "{{.RestartCount}}"
        $backendCid = (docker compose -f "docker-compose.production.yml" ps -q "backend" 2>$null | Select-Object -First 1)
        if (-not $backendCid) { $backendCid = "backend" }
        $statsSnapshot = docker stats --no-stream --format "{{.Name}} CPU={{.CPUPerc}} MEM={{.MemUsage}}" atmr-ws-service $backendCid

        $recentLogs = cmd /c "docker logs atmr-ws-service --since ${IntervalSeconds}s 2>&1"
        $errorHits = ($recentLogs | Select-String "GroupCoordinatorNotAvailableError|kafka consumer loop failed").Count
        $processedHits = ($recentLogs | Select-String "kafka event processed.*event_id=$eventId.*relay_mode=redis").Count

        Write-LogLine "ITERATION=$iteration TS=$($iterStart.ToString('o')) EVENT_ID=$eventId"
        Write-LogLine "HEALTH=$health"
        Write-LogLine "GROUP=$groupSnapshot"
        Write-LogLine "RESTART_COUNT=$restartCount"
        Write-LogLine "ERROR_HITS=$errorHits PROCESSED_HITS=$processedHits"
        foreach ($s in $statsSnapshot) {
            Write-LogLine "STATS=$s"
        }

        $elapsed = ((Get-Date) - $iterStart).TotalSeconds
        $sleepFor = [Math]::Max(0, $IntervalSeconds - [int][Math]::Ceiling($elapsed))
        if ($sleepFor -gt 0) {
            Start-Sleep -Seconds $sleepFor
        }
    }
}
finally {
    Write-LogLine "PHASE4_C3_SOAK_END_ACTUAL=$((Get-Date).ToString('o'))"
}

Write-Output "SOAK_LOG=$soakLog"
