$ErrorActionPreference = "Stop"

Write-Host "Smoke test gateway local..." -ForegroundColor Cyan

$checks = @(
    "http://lirie.local/login",
    "http://lirie.local/api/gateway/auth/context",
    "http://lirie.local/api/app/health",
    "http://lirie.local/api/demo/health"
)

foreach ($url in $checks) {
    try {
        $response = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 5
        Write-Host ("OK   {0} -> {1}" -f $url, $response.StatusCode) -ForegroundColor Green
    } catch {
        Write-Host ("FAIL {0} -> {1}" -f $url, $_.Exception.Message) -ForegroundColor Yellow
    }
}
