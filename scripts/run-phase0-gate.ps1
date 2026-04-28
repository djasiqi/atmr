Param(
  [string]$BaseUrl = "http://localhost:5000",
  [string]$WsUrl = "ws://localhost:5000/socket.io/?EIO=4&transport=websocket",
  [string]$OutDir = "reports/phase0"
)

New-Item -ItemType Directory -Path $OutDir -Force | Out-Null

$env:BASE_URL = $BaseUrl
$env:WS_URL = $WsUrl
$env:ENABLE_WS = "true"
$env:WS_VUS = "500"
$env:WS_DURATION = "20m"

k6 run "tests/load_testing/k6_phase0_gate.js" `
  --summary-export "$OutDir/k6_phase0_summary.json" `
  | Tee-Object -FilePath "$OutDir/k6_phase0_stdout.log"

Write-Host "Rapport genere dans $OutDir"
