#!/usr/bin/env bash
# build-docker.sh
# Script de build et validation Docker pour ATMR

set -euo pipefail

# Configuration
IMAGE_NAME="atmr-backend"
TAG="${1:-latest}"
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
VCS_REF=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
VERSION="${2:-latest}"

# Couleurs pour les logs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Fonction de logging
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✅${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ⚠️${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ❌${NC} $1"
}

# Fonction d'aide
show_help() {
    cat << EOF
Usage: $0 [TAG] [VERSION]

Build et validation Docker pour ATMR Backend

Arguments:
    TAG        Tag de l'image Docker (défaut: latest)
    VERSION    Version de l'application (défaut: latest)

Exemples:
    $0                    # Build avec tag 'latest'
    $0 v1.0.0            # Build avec tag 'v1.0.0'
    $0 v1.0.0 1.0.0      # Build avec tag 'v1.0.0' et version '1.0.0'

Options:
    --help, -h           Afficher cette aide
    --no-test           Ne pas exécuter les tests de smoke
    --no-scan           Ne pas scanner les vulnérabilités
    --push              Pousser l'image vers le registry après build
    --multi-arch        Build multi-architecture (amd64, arm64)

EOF
}

# Variables par défaut
NO_TEST=false
NO_SCAN=false
PUSH=false
MULTI_ARCH=false

# Parsing des arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            show_help
            exit 0
            ;;
        --no-test)
            NO_TEST=true
            shift
            ;;
        --no-scan)
            NO_SCAN=true
            shift
            ;;
        --push)
            PUSH=true
            shift
            ;;
        --multi-arch)
            MULTI_ARCH=true
            shift
            ;;
        *)
            break
            ;;
    esac
done

# Vérification des prérequis
check_prerequisites() {
    log "🔍 Vérification des prérequis..."
    
    # Vérifier Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker n'est pas installé"
        exit 1
    fi
    
    # Vérifier Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_warning "Docker Compose n'est pas installé"
    fi
    
    # Vérifier les outils de scan de sécurité
    if [ "$NO_SCAN" = false ]; then
        if ! command -v trivy &> /dev/null && ! command -v grype &> /dev/null; then
            log_warning "Aucun scanner de vulnérabilités trouvé (trivy/grype)"
            log_warning "Installation recommandée pour la sécurité"
        fi
    fi
    
    log_success "Prérequis vérifiés"
}

# Build de l'image Docker
build_image() {
    log "🔨 Build de l'image Docker..."
    
    # Arguments de build
    BUILD_ARGS="--build-arg BUILD_DATE=${BUILD_DATE}"
    BUILD_ARGS="${BUILD_ARGS} --build-arg VCS_REF=${VCS_REF}"
    BUILD_ARGS="${BUILD_ARGS} --build-arg VERSION=${VERSION}"
    BUILD_ARGS="${BUILD_ARGS} --build-arg WITH_POSTGRES=true"
    
    # Build multi-architecture si demandé
    if [ "$MULTI_ARCH" = true ]; then
        log "🏗️  Build multi-architecture (amd64, arm64)..."
        
        # Créer le builder multi-arch
        docker buildx create --name multiarch-builder --use 2>/dev/null || true
        
        # Build avec buildx
        docker buildx build \
            --platform linux/amd64,linux/arm64 \
            ${BUILD_ARGS} \
            -t "${IMAGE_NAME}:${TAG}" \
            -t "${IMAGE_NAME}:latest" \
            --push \
            ./backend
    else
        # Build standard
        log "🏗️  Build standard..."
        
        docker build \
            ${BUILD_ARGS} \
            -t "${IMAGE_NAME}:${TAG}" \
            -t "${IMAGE_NAME}:latest" \
            -f ./backend/Dockerfile.production \
            ./backend
    fi
    
    log_success "Image Docker buildée: ${IMAGE_NAME}:${TAG}"
}

# Scan de sécurité
scan_security() {
    if [ "$NO_SCAN" = true ]; then
        log_warning "Scan de sécurité désactivé"
        return
    fi
    
    log "🔒 Scan de sécurité de l'image..."
    
    # Scan avec Trivy
    if command -v trivy &> /dev/null; then
        log "🔍 Scan avec Trivy..."
        
        trivy image \
            --severity HIGH,CRITICAL \
            --exit-code 1 \
            --format table \
            "${IMAGE_NAME}:${TAG}" || {
            log_warning "Vulnérabilités détectées par Trivy"
            log_warning "Vérifiez les résultats ci-dessus"
        }
        
        log_success "Scan Trivy terminé"
    fi
    
    # Scan avec Grype
    if command -v grype &> /dev/null; then
        log "🔍 Scan avec Grype..."
        
        grype "${IMAGE_NAME}:${TAG}" \
            --fail-on high,critical \
            --format table || {
            log_warning "Vulnérabilités détectées par Grype"
            log_warning "Vérifiez les résultats ci-dessus"
        }
        
        log_success "Scan Grype terminé"
    fi
}

# Tests de smoke
run_smoke_tests() {
    if [ "$NO_TEST" = true ]; then
        log_warning "Tests de smoke désactivés"
        return
    fi
    
    log "🧪 Exécution des tests de smoke..."
    
    # Vérifier que le script de test existe
    if [ ! -f "./backend/scripts/docker_smoke_tests.py" ]; then
        log_error "Script de tests de smoke non trouvé"
        return
    fi
    
    # Exécuter les tests
    python3 ./backend/scripts/docker_smoke_tests.py \
        --image "${IMAGE_NAME}" \
        --tag "${TAG}" || {
        log_error "Tests de smoke échoués"
        exit 1
    }
    
    log_success "Tests de smoke réussis"
}

# Analyse de la taille de l'image
analyze_image_size() {
    log "📊 Analyse de la taille de l'image..."
    
    # Obtenir la taille de l'image
    IMAGE_SIZE=$(docker images --format "table {{.Size}}" "${IMAGE_NAME}:${TAG}" | tail -n 1)
    
    log "📏 Taille de l'image: ${IMAGE_SIZE}"
    
    # Analyser les couches
    log "🔍 Analyse des couches de l'image..."
    docker history "${IMAGE_NAME}:${TAG}" --format "table {{.CreatedBy}}\t{{.Size}}" | head -10
    
    log_success "Analyse de taille terminée"
}

# Push vers le registry
push_image() {
    if [ "$PUSH" = false ]; then
        log "📤 Push désactivé"
        return
    fi
    
    log "📤 Push de l'image vers le registry..."
    
    # Vérifier si un registry est configuré
    REGISTRY="${DOCKER_REGISTRY:-}"
    
    if [ -n "$REGISTRY" ]; then
        # Tag pour le registry
        docker tag "${IMAGE_NAME}:${TAG}" "${REGISTRY}/${IMAGE_NAME}:${TAG}"
        docker tag "${IMAGE_NAME}:${TAG}" "${REGISTRY}/${IMAGE_NAME}:latest"
        
        # Push
        docker push "${REGISTRY}/${IMAGE_NAME}:${TAG}"
        docker push "${REGISTRY}/${IMAGE_NAME}:latest"
        
        log_success "Image poussée vers ${REGISTRY}"
    else
        log_warning "Variable DOCKER_REGISTRY non définie, push ignoré"
    fi
}

# Génération du rapport
generate_report() {
    log "📋 Génération du rapport de build..."
    
    REPORT_FILE="docker-build-report-${TAG}-$(date +%Y%m%d-%H%M%S).json"
    
    cat > "$REPORT_FILE" << EOF
{
    "build_info": {
        "image_name": "${IMAGE_NAME}",
        "tag": "${TAG}",
        "version": "${VERSION}",
        "build_date": "${BUILD_DATE}",
        "vcs_ref": "${VCS_REF}",
        "multi_arch": ${MULTI_ARCH}
    },
    "build_status": "success",
    "security_scan": ${NO_SCAN},
    "smoke_tests": ${NO_TEST},
    "push_enabled": ${PUSH}
}
EOF
    
    log_success "Rapport généré: ${REPORT_FILE}"
}

# Fonction principale
main() {
    log "🚀 Démarrage du build Docker ATMR Backend"
    log "Image: ${IMAGE_NAME}:${TAG}"
    log "Version: ${VERSION}"
    log "Build Date: ${BUILD_DATE}"
    log "VCS Ref: ${VCS_REF}"
    
    # Exécution des étapes
    check_prerequisites
    build_image
    scan_security
    run_smoke_tests
    analyze_image_size
    push_image
    generate_report
    
    log_success "🎉 Build Docker terminé avec succès!"
    log_success "Image prête: ${IMAGE_NAME}:${TAG}"
}

# Gestion des erreurs
trap 'log_error "Build interrompu par une erreur"; exit 1' ERR

# Exécution
main "$@"
