#!/bin/bash
# Script de migration des styles shadow* vers boxShadow (compatibilité web)
# Usage: ./scripts/migrate-shadows.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "🔍 Recherche des fichiers avec styles shadow* deprecated..."

# Liste des fichiers à migrer
FILES=$(grep -rl "shadowColor\|shadowOffset\|shadowOpacity\|shadowRadius" \
    "$PROJECT_DIR/components" \
    "$PROJECT_DIR/app" \
    "$PROJECT_DIR/styles" \
    --include="*.tsx" \
    --include="*.ts" \
    --exclude-dir="node_modules" \
    --exclude="shadowStyles.ts" \
    2>/dev/null || true)

FILE_COUNT=$(echo "$FILES" | grep -c . || echo "0")

echo "📊 $FILE_COUNT fichiers trouvés nécessitant une migration"
echo ""

if [ "$FILE_COUNT" -eq 0 ]; then
    echo "✅ Tous les fichiers sont déjà migrés !"
    exit 0
fi

echo "📝 Fichiers à migrer :"
echo "$FILES" | nl
echo ""

# Demander confirmation
read -p "❓ Voulez-vous voir un exemple de migration ? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    cat << 'EOF'

📚 EXEMPLE DE MIGRATION
=======================

AVANT (deprecated) :
--------------------
import { StyleSheet } from "react-native";

const styles = StyleSheet.create({
  card: {
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
  },
});

APRÈS (compatible web/native) :
--------------------------------
import { StyleSheet } from "react-native";
import { shadowPresets } from "@/styles/shadowStyles";

const styles = StyleSheet.create({
  card: {
    ...shadowPresets.small, // ✅ Ou small/medium/large/accent
  },
});

OU (personnalisé) :
-------------------
import { StyleSheet } from "react-native";
import { createShadow } from "@/styles/shadowStyles";

const styles = StyleSheet.create({
  card: {
    ...createShadow({
      shadowColor: "rgba(15,54,43,0.06)",
      shadowOffset: { width: 0, height: 2 },
      shadowOpacity: 1,
      shadowRadius: 4,
      elevation: 2,
    }),
  },
});

📖 PRESETS DISPONIBLES
-----------------------
- shadowPresets.small  : Ombres légères (boutons, cards simples)
- shadowPresets.medium : Ombres moyennes (modals, cartes importantes)
- shadowPresets.large  : Ombres fortes (dropdowns, popovers)
- shadowPresets.accent : Ombres colorées (éléments accentués)

EOF
fi

echo ""
echo "🚀 Pour migrer un fichier :"
echo "   1. Ouvrir le fichier"
echo "   2. Importer shadowPresets ou createShadow depuis @/styles/shadowStyles"
echo "   3. Remplacer les props shadow* par ...shadowPresets.X ou ...createShadow({...})"
echo "   4. Tester sur web ET natif (iOS/Android)"
echo ""
echo "⚠️  IMPORTANT :"
echo "   - Ne PAS supprimer elevation (nécessaire pour Android)"
echo "   - Tester visuellement après migration"
echo "   - Utiliser les presets en priorité pour cohérence"
echo ""
echo "✅ Fichier déjà migré : components/enterprise/transfers/TransferCard.tsx"
echo "   → Utilisez-le comme référence"
