#!/bin/bash
# migrate-file.sh - Script pour migrer un fichier vers sa nouvelle destination
# Usage: ./migrate-file.sh <ancien_fichier> <nouveau_fichier>
#
# Exemple:
#   ./migrate-file.sh types.py core/types.py
#
# Ce script :
# 1. Utilise git mv pour préserver l'historique
# 2. Met à jour les imports internes du fichier
# 3. Vérifie que le fichier compile
# 4. Propose un commit

set -e

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Fonction d'aide
usage() {
    echo "Usage: $0 <ancien_fichier> <nouveau_fichier>"
    echo ""
    echo "Exemple:"
    echo "  $0 types.py core/types.py"
    echo "  $0 data.py data/loader.py"
    exit 1
}

# Vérifier les arguments
if [ $# -ne 2 ]; then
    usage
fi

OLD_FILE="$1"
NEW_FILE="$2"
BASE_DIR="backend/services/unified_dispatch"

# Vérifier que l'ancien fichier existe
if [ ! -f "$BASE_DIR/$OLD_FILE" ]; then
    echo -e "${RED}❌ Erreur: Le fichier $BASE_DIR/$OLD_FILE n'existe pas${NC}"
    exit 1
fi

# Vérifier que le répertoire de destination existe
NEW_DIR=$(dirname "$BASE_DIR/$NEW_FILE")
if [ ! -d "$NEW_DIR" ]; then
    echo -e "${RED}❌ Erreur: Le répertoire $NEW_DIR n'existe pas${NC}"
    exit 1
fi

echo -e "${BLUE}🔄 Migration de fichier${NC}"
echo -e "${BLUE}  De  : $OLD_FILE${NC}"
echo -e "${BLUE}  Vers: $NEW_FILE${NC}"
echo ""

# Étape 1: git mv (préserve l'historique)
echo -e "${YELLOW}📦 Étape 1/5: Déplacement avec git mv...${NC}"
cd "$BASE_DIR"
git mv "$OLD_FILE" "$NEW_FILE"
echo -e "${GREEN}✅ Fichier déplacé${NC}"

# Étape 2: Mettre à jour les imports dans le fichier migré
echo -e "${YELLOW}🔧 Étape 2/5: Mise à jour des imports internes...${NC}"
# Cette étape nécessiterait un script Python pour analyser et mettre à jour les imports
# Pour l'instant, on affiche juste un warning
echo -e "${YELLOW}⚠️  Vérifier manuellement les imports dans $NEW_FILE${NC}"

# Étape 3: Vérifier que le fichier compile
echo -e "${YELLOW}🧪 Étape 3/5: Vérification de la syntaxe Python...${NC}"
python3 -m py_compile "$NEW_FILE" 2>/dev/null && \
    echo -e "${GREEN}✅ Syntaxe correcte${NC}" || \
    echo -e "${RED}❌ Erreur de syntaxe - Vérifier le fichier${NC}"

# Étape 4: Rechercher les imports de ce fichier dans le codebase
echo -e "${YELLOW}🔍 Étape 4/5: Recherche des imports à mettre à jour...${NC}"
OLD_NAME=$(basename "$OLD_FILE" .py)
echo "Fichiers qui importent '$OLD_NAME':"
cd ../../..
grep -r "from.*unified_dispatch.*import.*$OLD_NAME" backend/ --include="*.py" 2>/dev/null | head -n 10 || echo "Aucun import direct trouvé"

# Étape 5: Proposer un commit
echo ""
echo -e "${YELLOW}📝 Étape 5/5: Commit suggéré${NC}"
echo -e "${BLUE}git commit -m \"refactor(B1): migrate $OLD_FILE → $NEW_FILE\"${NC}"
echo ""
echo -e "${GREEN}✅ Migration terminée !${NC}"
echo ""
echo -e "${YELLOW}Actions à faire manuellement:${NC}"
echo "1. Vérifier les imports dans $NEW_FILE"
echo "2. Ajouter l'export dans le __init__.py du module"
echo "3. Ajouter l'import de compatibilité dans unified_dispatch/__init__.py"
echo "4. Mettre à jour les imports dans les fichiers qui utilisent ce module"
echo "5. Exécuter les tests: pytest backend/tests/ -k '$OLD_NAME' -v"
echo "6. Commit si tout est OK"

