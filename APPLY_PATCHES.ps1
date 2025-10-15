# Script PowerShell pour appliquer les patches ATMR (Windows)
# Usage: .\APPLY_PATCHES.ps1 [-DryRun] [-CriticalOnly]

param(
    [switch]$DryRun,
    [switch]$CriticalOnly
)

$ErrorActionPreference = "Stop"

# Colors
function Write-Success { Write-Host $args -ForegroundColor Green }
function Write-Warning { Write-Host $args -ForegroundColor Yellow }
function Write-Error { Write-Host $args -ForegroundColor Red }

# Function to apply patch
function Apply-Patch {
    param(
        [string]$PatchFile,
        [string]$Description,
        [bool]$IsCritical
    )
    
    if ($CriticalOnly -and -not $IsCritical) {
        Write-Warning "⏭️  Skipping (non-critical): $Description"
        return $true
    }
    
    Write-Success "📦 Applying: $Description"
    Write-Host "   File: $PatchFile"
    
    if ($DryRun) {
        # Vérifier patch
        $result = git apply --check $PatchFile 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Success "   ✅ Dry-run OK"
        }
        else {
            Write-Error "   ❌ Would fail: $result"
        }
    }
    else {
        # Appliquer patch
        $checkResult = git apply --check $PatchFile 2>&1
        if ($LASTEXITCODE -eq 0) {
            git apply $PatchFile
            Write-Success "   ✅ Applied successfully"
        }
        else {
            Write-Error "   ❌ Failed - apply manually"
            Write-Host "   Error: $checkResult"
            return $false
        }
    }
    Write-Host ""
    return $true
}

# Banner
Write-Host "==================================" -ForegroundColor Cyan
Write-Host "🚀 ATMR Patches Application" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan
Write-Host ""

if ($DryRun) {
    Write-Warning "⚠️  DRY RUN MODE - No changes will be made"
    Write-Host ""
}
if ($CriticalOnly) {
    Write-Warning "⚠️  CRITICAL ONLY MODE - Only applying critical patches"
    Write-Host ""
}

# Check patches directory
if (-not (Test-Path "patches")) {
    Write-Error "❌ Error: patches/ directory not found"
    exit 1
}

# Backup warning
if (-not $DryRun) {
    Write-Warning "⚠️  WARNING: This will modify your codebase"
    Write-Warning "   Recommended: Create backup branch first"
    Write-Host ""
    $response = Read-Host "Continue? (y/N)"
    if ($response -ne 'y' -and $response -ne 'Y') {
        Write-Host "Aborted."
        exit 0
    }
    Write-Host ""
}

# ========== PHASE 1: CRITICAL BACKEND ==========
Write-Host "========== PHASE 1: CRITICAL BACKEND ==========" -ForegroundColor Cyan
Write-Host ""

Apply-Patch "patches/backend_timezone_fix.patch" "Timezone fixes (datetime.utcnow)" $true
Apply-Patch "patches/backend_celery_config.patch" "Celery acks_late + timeouts" $true
Apply-Patch "patches/backend_validation_fixes.patch" "Validation constraints" $true
Apply-Patch "patches/backend_socketio_validation.patch" "SocketIO validation" $true

# ========== PHASE 2: PERFORMANCE ==========
Write-Host "========== PHASE 2: PERFORMANCE ==========" -ForegroundColor Cyan
Write-Host ""

Apply-Patch "patches/backend_n+1_queries.patch" "N+1 queries fix" $true
Apply-Patch "patches/backend_pdf_config.patch" "PDF URLs config" $true

# ========== PHASE 3: FRONTEND ==========
Write-Host "========== PHASE 3: FRONTEND ==========" -ForegroundColor Cyan
Write-Host ""

Apply-Patch "patches/frontend_jwt_refresh.patch" "JWT auto-refresh" $true
Apply-Patch "patches/frontend_tests_setup.patch" "Jest/RTL setup" $false
Apply-Patch "patches/frontend_e2e_cypress.patch" "Cypress E2E" $false

# ========== PHASE 4: INFRA ==========
Write-Host "========== PHASE 4: INFRA ==========" -ForegroundColor Cyan
Write-Host ""

Apply-Patch "patches/infra_docker_compose_healthchecks.patch" "Docker healthchecks" $true

# ========== PHASE 5: CONFIG ==========
Write-Host "========== PHASE 5: CONFIG & DOCS ==========" -ForegroundColor Cyan
Write-Host ""

Apply-Patch "patches/backend_env_example.patch" ".env.example backend" $false
Apply-Patch "patches/frontend_env_example.patch" ".env.example frontend" $false
Apply-Patch "patches/root_gitignore_improvements.patch" ".gitignore" $false

# ========== PHASE 6: SECURITY ==========
Write-Host "========== PHASE 6: SECURITY (GDPR) ==========" -ForegroundColor Cyan
Write-Host ""

Apply-Patch "patches/backend_pii_logging_fix.patch" "PII masking" $false

# ========== PHASE 7: TESTS ==========
Write-Host "========== PHASE 7: TESTS ==========" -ForegroundColor Cyan
Write-Host ""

Apply-Patch "patches/backend_tests_auth.patch" "Backend tests" $false

# ========== CI/CD ==========
Write-Host "========== CI/CD WORKFLOWS ==========" -ForegroundColor Cyan
Write-Host ""

if (-not $DryRun) {
    if (-not (Test-Path ".github/workflows")) {
        New-Item -ItemType Directory -Path ".github/workflows" -Force | Out-Null
    }
    Copy-Item "ci/*.yml" ".github/workflows/" -Force
    Write-Success "✅ CI/CD workflows copied to .github/workflows/"
}
else {
    Write-Host "Would copy: ci/*.yml → .github/workflows/"
}
Write-Host ""

# ========== SUMMARY ==========
Write-Host "==================================" -ForegroundColor Cyan
Write-Host "✨ SUMMARY" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan
Write-Host ""

if ($DryRun) {
    Write-Host "Dry-run completed. Review output above."
}
else {
    Write-Success "✅ Patches applied successfully"
    Write-Host ""
    Write-Host "📝 Next steps:"
    Write-Host "  1. Review: git status && git diff"
    Write-Host "  2. Tests backend: cd backend && pytest"
    Write-Host "  3. DB migration: cd backend && alembic upgrade head"
    Write-Host "  4. Restart: docker-compose restart"
    Write-Host "  5. Config .env: Add PDF_BASE_URL, MASK_PII_LOGS"
    Write-Host ""
    Write-Warning "⚠️  IMPORTANT: backend_migration_indexes.patch"
    Write-Host "   requires manual Alembic migration (see MIGRATIONS_NOTES.md)"
}
Write-Host ""
Write-Host "📚 Documentation:"
Write-Host "  - REPORT.md"
Write-Host "  - MIGRATIONS_NOTES.md"
Write-Host "  - tests_plan.md"
Write-Host "  - patches/README_PATCHES.md"
Write-Host ""

