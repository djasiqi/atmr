# Script pour déclencher le workflow backend-test via l'API GitHub
# Usage: .\trigger-workflow.ps1 -Token "votre_token_github"

param(
    [Parameter(Mandatory=$true)]
    [string]$Token,
    
    [Parameter(Mandatory=$false)]
    [string]$Repo = "djasiqi/atmr",
    
    [Parameter(Mandatory=$false)]
    [string]$Workflow = "backend-test.yml",
    
    [Parameter(Mandatory=$false)]
    [string]$Ref = "main"
)

$headers = @{
    "Accept" = "application/vnd.github.v3+json"
    "Authorization" = "token $Token"
}

$body = @{
    ref = $Ref
} | ConvertTo-Json

$uri = "https://api.github.com/repos/$Repo/actions/workflows/$Workflow/dispatches"

try {
    $response = Invoke-RestMethod -Uri $uri -Method Post -Headers $headers -Body $body -ContentType "application/json"
    Write-Host "✅ Workflow '$Workflow' déclenché avec succès sur la branche '$Ref'!" -ForegroundColor Green
} catch {
    Write-Host "❌ Erreur lors du déclenchement du workflow:" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    if ($_.ErrorDetails.Message) {
        Write-Host $_.ErrorDetails.Message -ForegroundColor Red
    }
}

