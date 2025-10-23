#!/bin/bash
###############################################################################
# Script de nettoyage des fichiers morts (basé sur DEAD_FILES.json)
# Usage: bash cleanup_dead_files.sh [--dry-run]
#
# ⚠️ IMPORTANT: Vérifier manuellement avant exécution !
#               Créer un backup git au préalable.
###############################################################################

set -euo pipefail

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo -e "${YELLOW}🧪 MODE DRY-RUN: Aucun fichier ne sera supprimé${NC}\n"
fi

# Vérifier qu'on est dans la racine du projet
if [[ ! -f "docker-compose.yml" ]]; then
    echo -e "${RED}❌ Erreur: Exécuter ce script depuis la racine du projet (où se trouve docker-compose.yml)${NC}"
    exit 1
fi

# Créer un backup git tag
if [[ "$DRY_RUN" == "false" ]]; then
    BACKUP_TAG="backup-cleanup-$(date +%Y%m%d_%H%M%S)"
    echo -e "${GREEN}📦 Création du tag de backup: $BACKUP_TAG${NC}"
    git tag "$BACKUP_TAG" || {
        echo -e "${RED}❌ Erreur lors de la création du tag. Assurez-vous d'avoir commité tous les changements.${NC}"
        exit 1
    }
    echo -e "${GREEN}✅ Tag créé. Pour rollback: git checkout $BACKUP_TAG${NC}\n"
fi

###############################################################################
# LISTE DES FICHIERS À SUPPRIMER (HIGH CONFIDENCE)
###############################################################################

declare -a FILES_TO_DELETE=(
    "backend/check_bookings.py"
    "backend/Classeur1.xlsx"
    "backend/transport.xlsx"
    "backend/node_modules"
    "backend/celerybeat-schedule.bak"
    "backend/development.db"
    "frontend/src/styles/EXEMPLE.md"
    "frontend/src/pages/client/Profile"
)

###############################################################################
# FONCTION: Supprimer fichier/dossier
###############################################################################
delete_item() {
    local item="$1"
    
    if [[ ! -e "$item" ]]; then
        echo -e "${YELLOW}⚠️  Déjà absent: $item${NC}"
        return
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        echo -e "${YELLOW}[DRY-RUN] Supprimerait: $item${NC}"
        if [[ -d "$item" ]]; then
            echo -e "           (dossier, $(du -sh "$item" 2>/dev/null | cut -f1))"
        else
            echo -e "           (fichier, $(ls -lh "$item" 2>/dev/null | awk '{print $5}'))"
        fi
        return
    fi
    
    # Suppression réelle
    if [[ -d "$item" ]]; then
        echo -e "${GREEN}🗑️  Suppression dossier: $item${NC}"
        rm -rf "$item"
    else
        echo -e "${GREEN}🗑️  Suppression fichier: $item${NC}"
        rm -f "$item"
    fi
}

###############################################################################
# EXÉCUTION
###############################################################################

echo -e "${GREEN}🧹 NETTOYAGE DES FICHIERS MORTS${NC}"
echo -e "${GREEN}================================${NC}\n"

for file in "${FILES_TO_DELETE[@]}"; do
    delete_item "$file"
done

###############################################################################
# AJOUT AU .gitignore (fichiers temporaires)
###############################################################################

echo -e "\n${GREEN}📝 Mise à jour .gitignore${NC}"

GITIGNORE_ENTRIES=(
    "*.bak"
    "celerybeat-schedule.*"
    "development.db"
    "transport.xlsx"
    "Classeur*.xlsx"
)

for entry in "${GITIGNORE_ENTRIES[@]}"; do
    if ! grep -qF "$entry" .gitignore 2>/dev/null; then
        if [[ "$DRY_RUN" == "true" ]]; then
            echo -e "${YELLOW}[DRY-RUN] Ajouterait à .gitignore: $entry${NC}"
        else
            echo "$entry" >> .gitignore
            echo -e "${GREEN}✅ Ajouté à .gitignore: $entry${NC}"
        fi
    else
        echo -e "${GREEN}✓ Déjà dans .gitignore: $entry${NC}"
    fi
done

###############################################################################
# ARCHIVE mobile/client-app (au lieu de supprimer)
###############################################################################

echo -e "\n${GREEN}📦 Archivage mobile/client-app${NC}"

if [[ -d "mobile/client-app" ]]; then
    ARCHIVE_NAME="mobile/client-app.archive.$(date +%Y%m%d).tar.gz"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        echo -e "${YELLOW}[DRY-RUN] Créerait archive: $ARCHIVE_NAME${NC}"
        echo -e "${YELLOW}[DRY-RUN] Supprimerait: mobile/client-app/${NC}"
    else
        echo -e "${GREEN}📦 Création archive: $ARCHIVE_NAME${NC}"
        tar -czf "$ARCHIVE_NAME" mobile/client-app/
        
        echo -e "${GREEN}🗑️  Suppression: mobile/client-app/${NC}"
        rm -rf mobile/client-app/
        
        echo -e "${GREEN}✅ Archive créée. Pour restaurer: tar -xzf $ARCHIVE_NAME${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  mobile/client-app déjà absent${NC}"
fi

###############################################################################
# RAPPORT FINAL
###############################################################################

echo -e "\n${GREEN}================================${NC}"
echo -e "${GREEN}✅ NETTOYAGE TERMINÉ${NC}"
echo -e "${GREEN}================================${NC}\n"

if [[ "$DRY_RUN" == "true" ]]; then
    echo -e "${YELLOW}🧪 Mode dry-run: Aucun fichier supprimé.${NC}"
    echo -e "${YELLOW}   Exécuter sans --dry-run pour appliquer les changements.${NC}\n"
else
    echo -e "${GREEN}📊 Fichiers supprimés: ${#FILES_TO_DELETE[@]}${NC}"
    echo -e "${GREEN}📦 Backup tag: $BACKUP_TAG${NC}"
    echo -e "${GREEN}🔄 Rollback: git checkout $BACKUP_TAG${NC}\n"
    
    echo -e "${YELLOW}⚠️  PROCHAINES ÉTAPES:${NC}"
    echo -e "${YELLOW}   1. Vérifier que tout fonctionne (make test)${NC}"
    echo -e "${YELLOW}   2. Commiter les changements:${NC}"
    echo -e "${YELLOW}      git add -A${NC}"
    echo -e "${YELLOW}      git commit -m 'chore: cleanup dead files (audit 2025-10-18)'${NC}"
    echo -e "${YELLOW}   3. Si problème: git checkout $BACKUP_TAG${NC}\n"
fi

###############################################################################
# VÉRIFICATION POST-NETTOYAGE (si pas dry-run)
###############################################################################

if [[ "$DRY_RUN" == "false" ]]; then
    echo -e "${GREEN}🔍 VÉRIFICATION POST-NETTOYAGE${NC}"
    echo -e "${GREEN}===============================${NC}\n"
    
    # Vérifier imports Python cassés
    echo -e "${GREEN}🐍 Vérification imports Python...${NC}"
    if command -v python &> /dev/null; then
        cd backend
        if python -c "import app; print('✅ Backend imports OK')" 2>&1 | grep -q "OK"; then
            echo -e "${GREEN}✅ Backend imports OK${NC}"
        else
            echo -e "${RED}❌ Erreur imports backend ! Vérifier les logs.${NC}"
        fi
        cd ..
    fi
    
    # Vérifier build frontend
    echo -e "\n${GREEN}⚛️  Vérification build frontend...${NC}"
    if command -v npm &> /dev/null; then
        cd frontend
        if npm run build > /tmp/build.log 2>&1; then
            echo -e "${GREEN}✅ Frontend build OK${NC}"
        else
            echo -e "${RED}❌ Erreur build frontend ! Voir /tmp/build.log${NC}"
        fi
        cd ..
    fi
    
    echo -e "\n${GREEN}✅ Vérifications terminées${NC}\n"
fi

echo -e "${GREEN}🎉 Script terminé avec succès !${NC}"

