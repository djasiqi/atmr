#!/bin/bash
#
# Script d'application automatique des patches ATMR
# Usage: ./APPLY_PATCHES.sh [--dry-run] [--critical-only]
#

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

DRY_RUN=false
CRITICAL_ONLY=false

# Parse args
for arg in "$@"; do
  case $arg in
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --critical-only)
      CRITICAL_ONLY=true
      shift
      ;;
  esac
done

# Function to apply patch
apply_patch() {
  local patch_file=$1
  local description=$2
  local is_critical=$3
  
  if [ "$CRITICAL_ONLY" = true ] && [ "$is_critical" != "true" ]; then
    echo -e "${YELLOW}⏭️  Skipping (non-critical): $description${NC}"
    return 0
  fi
  
  echo -e "${GREEN}📦 Applying: $description${NC}"
  echo "   File: $patch_file"
  
  if [ "$DRY_RUN" = true ]; then
    git apply --check "$patch_file" && echo "   ✅ Dry-run OK" || echo "   ❌ Would fail"
  else
    if git apply --check "$patch_file" 2>/dev/null; then
      git apply "$patch_file"
      echo -e "   ${GREEN}✅ Applied successfully${NC}"
    else
      echo -e "   ${RED}❌ Failed - apply manually${NC}"
      return 1
    fi
  fi
  echo ""
}

# Banner
echo "=================================="
echo "🚀 ATMR Patches Application"
echo "=================================="
echo ""
if [ "$DRY_RUN" = true ]; then
  echo -e "${YELLOW}⚠️  DRY RUN MODE - No changes will be made${NC}"
  echo ""
fi
if [ "$CRITICAL_ONLY" = true ]; then
  echo -e "${YELLOW}⚠️  CRITICAL ONLY MODE - Only applying critical patches${NC}"
  echo ""
fi

# Check if patches directory exists
if [ ! -d "patches" ]; then
  echo -e "${RED}❌ Error: patches/ directory not found${NC}"
  exit 1
fi

# Backup check
if [ "$DRY_RUN" = false ]; then
  echo -e "${YELLOW}⚠️  WARNING: This will modify your codebase${NC}"
  echo -e "${YELLOW}   Recommended: Create backup branch first${NC}"
  echo ""
  read -p "Continue? (y/N): " -n 1 -r
  echo ""
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
  fi
  echo ""
fi

# ========== PHASE 1: CRITICAL BACKEND ==========
echo "========== PHASE 1: CRITICAL BACKEND =========="
echo ""

apply_patch "patches/backend_timezone_fix.patch" "Timezone fixes (datetime.utcnow)" true
apply_patch "patches/backend_celery_config.patch" "Celery acks_late + timeouts" true
apply_patch "patches/backend_validation_fixes.patch" "Validation constraints + dead code" true
apply_patch "patches/backend_socketio_validation.patch" "SocketIO payload validation" true

# ========== PHASE 2: PERFORMANCE ==========
echo "========== PHASE 2: PERFORMANCE =========="
echo ""

apply_patch "patches/backend_n+1_queries.patch" "N+1 queries fix (joinedload)" true
apply_patch "patches/backend_pdf_config.patch" "PDF URLs config" true

# ========== PHASE 3: FRONTEND ==========
echo "========== PHASE 3: FRONTEND =========="
echo ""

apply_patch "patches/frontend_jwt_refresh.patch" "JWT auto-refresh on 401" true
apply_patch "patches/frontend_tests_setup.patch" "Jest/RTL setup + tests" false
apply_patch "patches/frontend_e2e_cypress.patch" "Cypress E2E tests" false

# ========== PHASE 4: INFRA ==========
echo "========== PHASE 4: INFRA =========="
echo ""

apply_patch "patches/infra_docker_compose_healthchecks.patch" "Docker healthchecks" true

# ========== PHASE 5: CONFIG & DOCS ==========
echo "========== PHASE 5: CONFIG & DOCS =========="
echo ""

apply_patch "patches/backend_env_example.patch" ".env.example backend" false
apply_patch "patches/frontend_env_example.patch" ".env.example frontend" false
apply_patch "patches/root_gitignore_improvements.patch" ".gitignore improvements" false

# ========== PHASE 6: SECURITY (GDPR) ==========
echo "========== PHASE 6: SECURITY (GDPR) =========="
echo ""

apply_patch "patches/backend_pii_logging_fix.patch" "PII masking in logs" false

# ========== PHASE 7: TESTS ==========
echo "========== PHASE 7: TESTS =========="
echo ""

apply_patch "patches/backend_tests_auth.patch" "Backend auth tests" false

# ========== CI/CD ==========
echo "========== CI/CD WORKFLOWS =========="
echo ""

if [ "$DRY_RUN" = false ]; then
  mkdir -p .github/workflows
  cp ci/*.yml .github/workflows/ 2>/dev/null || true
  echo -e "${GREEN}✅ CI/CD workflows copied to .github/workflows/${NC}"
else
  echo "Would copy: ci/*.yml → .github/workflows/"
fi
echo ""

# ========== SUMMARY ==========
echo "=================================="
echo "✨ SUMMARY"
echo "=================================="
echo ""
if [ "$DRY_RUN" = true ]; then
  echo "Dry-run completed. Review output above."
else
  echo -e "${GREEN}✅ Patches applied successfully${NC}"
  echo ""
  echo "📝 Next steps:"
  echo "  1. Review changes: git status && git diff"
  echo "  2. Run tests: cd backend && pytest"
  echo "  3. Apply DB migration: cd backend && alembic upgrade head"
  echo "  4. Restart services: docker-compose restart"
  echo "  5. Configure .env: Add PDF_BASE_URL, MASK_PII_LOGS"
  echo ""
  echo "⚠️  IMPORTANT: Migration DB index (backend_migration_indexes.patch)"
  echo "   requires manual Alembic migration creation (see MIGRATIONS_NOTES.md)"
fi
echo ""
echo "📚 Documentation:"
echo "  - REPORT.md - Full audit report"
echo "  - MIGRATIONS_NOTES.md - DB migrations guide"
echo "  - tests_plan.md - Testing strategy"
echo "  - patches/README_PATCHES.md - Detailed patch guide"
echo ""

