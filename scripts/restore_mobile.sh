#!/bin/bash
# Script de restauration du dossier mobile pour ATMR
# Usage: ./scripts/restore_mobile.sh <backup_path> [--force]
#
# Le backup peut être:
# - Un dossier: ./scripts/restore_mobile.sh /path/to/mobile_backup
# - Une archive: ./scripts/restore_mobile.sh /path/to/mobile_backup.zip

set -euo pipefail

BACKUP_PATH="${1:-}"
FORCE="${2:-}"

# Couleurs pour output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Vérifier argument
if [ -z "$BACKUP_PATH" ]; then
    echo -e "${RED}❌ Usage: $0 <backup_path> [--force]${NC}"
    echo ""
    echo "Exemples:"
    echo "  $0 /path/to/mobile_backup"
    echo "  $0 /path/to/mobile_backup.zip"
    echo "  $0 /path/to/mobile_backup.tar.gz"
    echo "  $0 backups/mobile_backup_20250127 --force"
    exit 1
fi

# Vérifier que le backup existe
if [ ! -e "$BACKUP_PATH" ]; then
    echo -e "${RED}❌ Erreur: Backup non trouvé: $BACKUP_PATH${NC}"
    exit 1
fi

# Chemin de destination
MOBILE_DIR="./mobile"
BACKUP_TEMP_DIR=""

# Détecter si c'est une archive ou un dossier
if [ -f "$BACKUP_PATH" ]; then
    echo -e "${YELLOW}📦 Détection du type d'archive...${NC}"
    
    # Créer un répertoire temporaire pour l'extraction
    BACKUP_TEMP_DIR=$(mktemp -d)
    trap "rm -rf $BACKUP_TEMP_DIR" EXIT
    
    # Détecter et extraire selon le type
    if [[ "$BACKUP_PATH" == *.zip ]]; then
        echo "   Format: ZIP"
        if ! command -v unzip &> /dev/null; then
            echo -e "${RED}❌ Erreur: unzip non trouvé${NC}"
            exit 1
        fi
        unzip -q "$BACKUP_PATH" -d "$BACKUP_TEMP_DIR"
        EXTRACTED_DIR="$BACKUP_TEMP_DIR"
        
    elif [[ "$BACKUP_PATH" == *.tar.gz ]] || [[ "$BACKUP_PATH" == *.tgz ]]; then
        echo "   Format: TAR.GZ"
        if ! command -v tar &> /dev/null; then
            echo -e "${RED}❌ Erreur: tar non trouvé${NC}"
            exit 1
        fi
        tar -xzf "$BACKUP_PATH" -C "$BACKUP_TEMP_DIR"
        EXTRACTED_DIR="$BACKUP_TEMP_DIR"
        
    elif [[ "$BACKUP_PATH" == *.tar ]]; then
        echo "   Format: TAR"
        if ! command -v tar &> /dev/null; then
            echo -e "${RED}❌ Erreur: tar non trouvé${NC}"
            exit 1
        fi
        tar -xf "$BACKUP_PATH" -C "$BACKUP_TEMP_DIR"
        EXTRACTED_DIR="$BACKUP_TEMP_DIR"
        
    else
        echo -e "${RED}❌ Format d'archive non supporté: $BACKUP_PATH${NC}"
        echo "   Formats supportés: .zip, .tar, .tar.gz, .tgz"
        exit 1
    fi
    
    # Trouver le dossier mobile dans l'extraction
    # Il peut être à la racine ou dans un sous-dossier
    if [ -d "$EXTRACTED_DIR/mobile" ]; then
        SOURCE_DIR="$EXTRACTED_DIR/mobile"
    elif [ -d "$EXTRACTED_DIR" ] && [ "$(ls -A $EXTRACTED_DIR | wc -l)" -eq 1 ] && [ -d "$(ls -d $EXTRACTED_DIR/*)" ]; then
        # Si un seul dossier à la racine, c'est probablement le dossier mobile
        SINGLE_DIR=$(ls -d $EXTRACTED_DIR/* | head -1)
        if [ -d "$SINGLE_DIR" ]; then
            SOURCE_DIR="$SINGLE_DIR"
        else
            SOURCE_DIR="$EXTRACTED_DIR"
        fi
    else
        SOURCE_DIR="$EXTRACTED_DIR"
    fi
    
elif [ -d "$BACKUP_PATH" ]; then
    echo "   Format: Dossier"
    SOURCE_DIR="$BACKUP_PATH"
else
    echo -e "${RED}❌ Erreur: Format non reconnu${NC}"
    exit 1
fi

# Vérifier que le dossier source contient du contenu
if [ ! -d "$SOURCE_DIR" ] || [ -z "$(ls -A $SOURCE_DIR 2>/dev/null)" ]; then
    echo -e "${RED}❌ Erreur: Le backup semble vide${NC}"
    exit 1
fi

echo ""
echo "🔄 Restauration du dossier mobile..."
echo "   Source: $SOURCE_DIR"
echo "   Destination: $MOBILE_DIR"
echo ""

# Afficher le contenu du backup
echo "📋 Contenu du backup:"
ls -la "$SOURCE_DIR" | head -10
echo ""

# Confirmation (sauf si --force)
if [ "$FORCE" != "--force" ]; then
    echo -e "${YELLOW}⚠️  ATTENTION: Cette opération va écraser le dossier mobile actuel!${NC}"
    echo "   Toutes les données non sauvegardées seront perdues."
    echo ""
    read -p "Continuer? (tapez 'yes' pour confirmer): " confirm
    
    if [ "$confirm" != "yes" ]; then
        echo "❌ Opération annulée."
        exit 0
    fi
fi

# Sauvegarder l'ancien dossier mobile s'il existe
if [ -d "$MOBILE_DIR" ] && [ -n "$(ls -A $MOBILE_DIR 2>/dev/null)" ]; then
    BACKUP_OLD_DIR="${MOBILE_DIR}.old.$(date +%Y%m%d_%H%M%S)"
    echo -e "${YELLOW}💾 Sauvegarde de l'ancien dossier mobile vers: $BACKUP_OLD_DIR${NC}"
    mv "$MOBILE_DIR" "$BACKUP_OLD_DIR"
fi

# Créer le répertoire de destination
mkdir -p "$MOBILE_DIR"

# Copier le contenu
echo -e "${GREEN}📂 Copie des fichiers...${NC}"
cp -r "$SOURCE_DIR"/* "$MOBILE_DIR"/

# Vérifier que la copie a réussi
if [ -z "$(ls -A $MOBILE_DIR 2>/dev/null)" ]; then
    echo -e "${RED}❌ Erreur: La restauration semble avoir échoué (dossier vide)${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✅ Restauration terminée avec succès!${NC}"
echo ""
echo "📊 Contenu restauré:"
ls -la "$MOBILE_DIR" | head -10
echo ""

# Afficher la structure
if [ -d "$MOBILE_DIR/unified-app" ]; then
    echo "📱 unified-app trouvé"
fi

echo ""
echo "💡 Prochaines étapes:"
echo "   1. Vérifier le contenu: ls -la $MOBILE_DIR"
echo "   2. Installer les dépendances dans chaque app mobile"
echo "   3. Vérifier les fichiers de configuration (.env, etc.)"

