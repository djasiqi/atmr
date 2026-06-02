# Vérification post-déploiement : alignement backend / celery-worker / celery-beat
param(
    [string[]]$Containers = @("atmr-backend-1", "atmr-celery-worker-1", "atmr-celery-beat-1")
)

Write-Host "=== Alignement images Celery ===" -ForegroundColor Cyan
$rows = foreach ($name in $Containers) {
    $imageTag = docker inspect $name --format '{{.Config.Image}}' 2>$null
    $imageId = docker inspect $name --format '{{.Image}}' 2>$null
    $revision = docker inspect $name --format '{{index .Config.Labels "org.opencontainers.image.revision"}}' 2>$null
    [PSCustomObject]@{
        Container = $name
        ImageTag    = $imageTag
        ImageId     = $imageId
        GitRevision = $revision
    }
}

$rows | Format-Table -AutoSize

$uniqueTags = ($rows | Select-Object -ExpandProperty ImageTag -Unique)
$uniqueIds = ($rows | Select-Object -ExpandProperty ImageId -Unique)
$uniqueRevs = ($rows | Where-Object { $_.GitRevision } | Select-Object -ExpandProperty GitRevision -Unique)

if ($uniqueTags.Count -gt 1 -or $uniqueIds.Count -gt 1) {
    Write-Error "Desalignement detecte: tags=$($uniqueTags.Count) ids=$($uniqueIds.Count)"
    exit 1
}

if ($uniqueRevs.Count -gt 1) {
    Write-Error "Desalignement commit SHA detecte: $($uniqueRevs -join ', ')"
    exit 1
}

Write-Host "OK: tag, image id et revision alignes." -ForegroundColor Green
