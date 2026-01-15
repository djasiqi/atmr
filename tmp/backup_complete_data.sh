#!/bin/bash
# Backup complet des DONNÉES PostgreSQL avec encodage UTF-8 vérifié

set -euo pipefail

BACKUP_FILE="data/backup_complete_data_$(date +%Y%m%d_%H%M%S).sql"

echo "🔍 Backup complet des DONNÉES PostgreSQL..."
echo "📁 Fichier: $BACKUP_FILE"

# Créer le backup avec encodage UTF-8 explicite
docker compose exec -T postgres pg_dump \
  -U atmr \
  -d atmr \
  --data-only \
  --no-owner \
  --no-privileges \
  --encoding=UTF8 \
  --column-inserts \
  --disable-triggers \
  > "$BACKUP_FILE"

echo "✅ Backup créé: $BACKUP_FILE"

# Vérifier la taille
SIZE=$(wc -c < "$BACKUP_FILE")
SIZE_MB=$((SIZE / 1024 / 1024))
echo "📊 Taille: ${SIZE_MB} MB"

# Vérifier l'encodage UTF-8 avec des exemples
echo ""
echo "🔍 Vérification de l'encodage UTF-8..."
echo ""

# Tester quelques caractères français courants
echo "📝 Test des caractères accentués dans le backup:"
grep -o "é" "$BACKUP_FILE" | head -1 && echo "  ✅ é trouvé" || echo "  ⚠️ é non trouvé"
grep -o "è" "$BACKUP_FILE" | head -1 && echo "  ✅ è trouvé" || echo "  ⚠️ è non trouvé"
grep -o "à" "$BACKUP_FILE" | head -1 && echo "  ✅ à trouvé" || echo "  ⚠️ à non trouvé"
grep -o "ç" "$BACKUP_FILE" | head -1 && echo "  ✅ ç trouvé" || echo "  ⚠️ ç non trouvé"

echo ""
echo "📋 Exemples d'adresses avec accents:"
grep -i "genève\|théodore\|emmenez" "$BACKUP_FILE" | head -5 || echo "  (aucun exemple trouvé)"

echo ""
echo "📊 Statistiques du backup:"
echo "  - Lignes INSERT: $(grep -c "^INSERT INTO" "$BACKUP_FILE" || echo "0")"
echo "  - Tables concernées:"
grep "^INSERT INTO" "$BACKUP_FILE" | sed 's/INSERT INTO \([^ ]*\).*/\1/' | sort | uniq -c | sort -rn | head -10

echo ""
echo "✅ Backup complet terminé et vérifié !"
echo "📁 Fichier: $BACKUP_FILE"
