Param(
  [string]$BaseUrl = "http://localhost:5000"
)

$ErrorActionPreference = "Stop"

function Test-HttpEndpoint {
  param(
    [string]$Url,
    [string]$Label,
    [switch]$AllowHttpsRedirect
  )

  $probe = curl.exe -s -o NUL -w "%{http_code} %{redirect_url}" -I --max-redirs 0 $Url
  if ($LASTEXITCODE -ne 0) {
    Write-Warning "$Label probe HTTP a echoue (curl exit=$LASTEXITCODE)"
    return
  }

  $parts = $probe.Trim() -split "\s+", 2
  $statusCode = if ($parts.Count -ge 1) { $parts[0] } else { "" }
  $redirectUrl = if ($parts.Count -ge 2) { $parts[1] } else { "" }

  if ($statusCode -eq "200") {
    Write-Host "$Label status: 200"
    return
  }

  if ($AllowHttpsRedirect -and ($statusCode -in @("301", "302", "307", "308")) -and (-not [string]::IsNullOrWhiteSpace($redirectUrl)) -and $redirectUrl.StartsWith("https://")) {
    Write-Host "$Label status: $statusCode (redirection HTTPS acceptee -> $redirectUrl)"
    return
  }

  Write-Warning "$Label status inattendu: $statusCode (redirect=$redirectUrl)"
}

Write-Host "== Smoke Scalability Proof =="

Write-Host "`n[1/6] Compose production services"
docker compose -f "docker-compose.production.yml" config --services

Write-Host "`n[2/6] Compose kafka services"
docker compose -f "docker-compose.kafka.yml" config --services

Write-Host "`n[3/6] Runtime status production"
docker compose -f "docker-compose.production.yml" ps

Write-Host "`n[4/6] Runtime status kafka"
docker compose -f "docker-compose.kafka.yml" ps

Write-Host "`n[5/6] Health endpoint"
Test-HttpEndpoint -Url "$BaseUrl/health" -Label "health"

Write-Host "`n[6/6] Realtime canary endpoint"
Test-HttpEndpoint -Url "$BaseUrl/api/v1/realtime-gateway/canary" -Label "canary" -AllowHttpsRedirect

Write-Host "`nSmoke termine."
