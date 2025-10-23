# 🧪 TESTS MANUELS SHADOW MODE - GUIDE RAPIDE

**Date :** 21 Octobre 2025  
**Durée estimée :** 15 minutes  
**Statut :** ✅ **PRÊT POUR TESTS**

---

## 🎯 OBJECTIF

Vérifier que le Shadow Mode fonctionne correctement en production avant de laisser tourner pendant 1 semaine.

---

## ✅ PRÉ-REQUIS

```yaml
✅ Backend redémarré (docker-compose restart api)
✅ Shadow Mode intégré dans dispatch
✅ Routes API enregistrées
✅ 50 tests (100% pass)
✅ Token admin disponible
```

---

## 🧪 TEST 1 : API Shadow Mode (5 min)

### Récupérer un token admin

```bash
# Option 1: Via l'API
curl -X POST http://localhost:5000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@atmr.com","password":"votre_password"}'

# Copier le "access_token" de la réponse
# export TOKEN="votre_token_ici"

# Option 2: Depuis la base de données (dev uniquement)
docker-compose exec api python -c "
from flask_jwt_extended import create_access_token
from models import User
from app import create_app
app = create_app()
with app.app_context():
    admin = User.query.filter_by(role='admin').first()
    if admin:
        token = create_access_token(identity=admin.id)
        print(f'Token: {token}')
    else:
        print('Aucun admin trouvé')
"
```

### Tester le statut

```bash
# Remplacer YOUR_TOKEN par le token obtenu
curl http://localhost:5000/api/shadow-mode/status \
  -H "Authorization: Bearer YOUR_TOKEN" \
  | jq '.'
```

**Résultat attendu :**

```json
{
  "status": "active",
  "model_loaded": true,
  "stats": {
    "predictions_count": 0,
    "comparisons_count": 0,
    "agreements_count": 0,
    "agreement_rate": 0.0,
    "model_path": "data/rl/models/dqn_best.pth",
    "log_dir": "data/rl/shadow_mode"
  }
}
```

✅ **Si vous voyez ce résultat, le Shadow Mode est opérationnel !**

---

## 🧪 TEST 2 : Faire des Réassignations (5 min)

### Option A : Via le Frontend

1. Ouvrir `http://localhost:3000`
2. Se connecter comme admin ou company
3. Aller dans **Dashboard → Bookings**
4. Cliquer sur un booking avec status "assigned"
5. Cliquer sur "Réassigner" → Choisir un autre driver
6. Répéter 3-5 fois avec différents bookings

### Option B : Via l'API directement

```bash
# Lister les assignments actuels
curl http://localhost:5000/api/company_dispatch/assignments \
  -H "Authorization: Bearer YOUR_TOKEN" \
  | jq '.[] | {id, booking_id, driver_id}'

# Récupérer un assignment_id et un nouveau driver_id
# Puis réassigner

curl -X POST http://localhost:5000/api/company_dispatch/assignments/ASSIGNMENT_ID/reassign \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"new_driver_id": NOUVEAU_DRIVER_ID}'

# Répéter 3-5 fois
```

✅ **Si les réassignations fonctionnent normalement (sans erreur), c'est parfait !**

---

## 🧪 TEST 3 : Vérifier les Logs Shadow (5 min)

### Vérifier que les fichiers sont créés

```bash
# Lister les fichiers de shadow mode
docker-compose exec api ls -lh data/rl/shadow_mode/

# Vous devriez voir:
# predictions_20251021.jsonl
# comparisons_20251021.jsonl
# (et potentiellement daily_report_20251021.json)
```

### Examiner les prédictions

```bash
# Regarder les 3 premières prédictions
docker-compose exec api head -3 data/rl/shadow_mode/predictions_*.jsonl

# Exemple de sortie:
# {"booking_id": 123, "predicted_driver_id": 5, "action_type": "assign", "confidence": 0.87, ...}
# {"booking_id": 124, "predicted_driver_id": 3, "action_type": "assign", "confidence": 0.72, ...}
# {"booking_id": 125, "predicted_driver_id": null, "action_type": "wait", "confidence": 0.65, ...}
```

### Examiner les comparaisons

```bash
# Regarder les 3 premières comparaisons
docker-compose exec api head -3 data/rl/shadow_mode/comparisons_*.jsonl

# Exemple de sortie:
# {"booking_id": 123, "predicted_driver_id": 5, "actual_driver_id": 5, "agreement": true, ...}
# {"booking_id": 124, "predicted_driver_id": 3, "actual_driver_id": 2, "agreement": false, ...}
```

### Vérifier via API

```bash
# Dernières prédictions (5 dernières)
curl "http://localhost:5000/api/shadow-mode/predictions?limit=5" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  | jq '.predictions[] | {booking_id, action_type, confidence}'

# Dernières comparaisons
curl "http://localhost:5000/api/shadow-mode/comparisons?limit=5" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  | jq '.comparisons[] | {booking_id, agreement, confidence}'
```

✅ **Si vous voyez des données dans les fichiers/API, le logging fonctionne !**

---

## 🧪 TEST 4 : Stats Temps Réel (2 min)

```bash
# Stats complètes
curl http://localhost:5000/api/shadow-mode/stats \
  -H "Authorization: Bearer YOUR_TOKEN" \
  | jq '.'

# Exemple de sortie:
# {
#   "period": "today",
#   "session_stats": {
#     "predictions_count": 5,
#     "comparisons_count": 5,
#     "agreements_count": 4,
#     "agreement_rate": 0.8,  # 80% d'accord!
#     ...
#   },
#   "daily_report": {
#     "summary": {
#       "total_predictions": 5,
#       "agreement_rate": 0.8
#     },
#     ...
#   }
# }
```

✅ **Si vous voyez des stats > 0, tout fonctionne parfaitement !**

---

## 📊 RÉSULTATS ATTENDUS

### Après 5 Réassignations

```yaml
Prédictions:
  Count: 5
  Types: mix de "assign" et "wait"
  Confidence: 0.6-0.9 (variable)

Comparaisons:
  Count: 5
  Agreements: 3-4 (60-80%) ✅
  Disagreements: 1-2 (20-40%)

Taux d'accord:
  Normal: 60-80% sur petit échantillon
  Variance élevée avec peu de données
  Se stabilise après 100+ réassignations
```

### Si Taux d'Accord Faible (<50%)

```yaml
Ne PAS paniquer !
  → Normal sur petit échantillon (5-10 tests)
  → Variance très élevée au début
  → Se stabilise sur 100+ réassignations

Actions:
  1. Continuer à faire des réassignations
  2. Attendre 50-100 échantillons
  3. Analyser les patterns de désaccord
  4. Vérifier que c'est pas un bug (ex: toujours driver null)
```

---

## ⚠️ PROBLÈMES POTENTIELS

### Problème 1 : model_loaded = false

**Symptôme :**

```json
{
  "model_loaded": false,
  ...
}
```

**Solution :**

```bash
# Vérifier que le modèle existe
docker-compose exec api ls -lh data/rl/models/dqn_best.pth

# Si manquant:
docker-compose exec api cp data/rl/models/dqn_ep0600_r672.pth \
                              data/rl/models/dqn_best.pth

# Recharger
curl -X POST http://localhost:5000/api/shadow-mode/reload-model \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json"
```

### Problème 2 : Fichiers logs non créés

**Symptôme :**

```bash
ls backend/data/rl/shadow_mode/
# Vide ou n'existe pas
```

**Solution :**

```bash
# Créer le répertoire manuellement
docker-compose exec api mkdir -p data/rl/shadow_mode

# Vérifier permissions
docker-compose exec api chmod 755 data/rl/shadow_mode

# Redémarrer
docker-compose restart api
```

### Problème 3 : Erreur lors des réassignations

**Symptôme :**
Erreur 500 lors des réassignations via frontend/API

**Solution :**

```bash
# Vérifier les logs
docker-compose logs api --tail=50

# Si erreur "Shadow mode error":
#   → C'est OK! (non-bloquant)
#   → L'assignation devrait quand même fonctionner
#   → Investiguer le détail de l'erreur

# Si erreur "reassign failed":
#   → C'est un problème du système de base
#   → PAS lié au shadow mode
#   → Vérifier la logique de réassignation normale
```

---

## ✅ CHECKLIST VALIDATION

### Après Tests Manuels

- [ ] API `/status` retourne `model_loaded: true`
- [ ] 5+ réassignations effectuées sans erreur
- [ ] Fichiers logs créés dans `data/rl/shadow_mode/`
- [ ] Prédictions visibles (fichiers + API)
- [ ] Comparaisons visibles (fichiers + API)
- [ ] Stats temps réel fonctionnelles
- [ ] Taux d'accord calculé (peu importe la valeur)
- [ ] Aucune erreur critique dans logs

**Si toutes les cases sont cochées : ✅ SHADOW MODE OPÉRATIONNEL !**

---

## 📈 MONITORING QUOTIDIEN (CETTE SEMAINE)

### Routine Matin (5 min)

```bash
# 1. Rapport du jour précédent
date_hier=$(date -d 'yesterday' +%Y%m%d 2>/dev/null || date -v-1d +%Y%m%d)
curl "http://localhost:5000/api/shadow-mode/report/$date_hier" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  | jq '.summary'

# 2. Taux d'accord (devrait être >75% après quelques jours)
```

### Routine Soir (5 min)

```bash
# Stats temps réel
curl http://localhost:5000/api/shadow-mode/stats \
  -H "Authorization: Bearer YOUR_TOKEN" \
  | jq '.session_stats | {predictions_count, comparisons_count, agreement_rate}'

# Dernières prédictions
curl "http://localhost:5000/api/shadow-mode/predictions?limit=5" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  | jq '.predictions[-5:] | .[] | {booking_id, action_type, confidence}'
```

### Routine Vendredi (30 min)

```bash
# Analyse complète semaine
docker-compose exec api python scripts/rl/shadow_mode_analysis.py \
  --start-date 20251021 \
  --end-date 20251027 \
  --output-dir data/rl/shadow_mode/analysis

# Visualiser graphiques
# Windows: explorer backend\data\rl\shadow_mode\analysis
# Linux/Mac: open backend/data/rl/shadow_mode/analysis/*.png

# Lire rapport JSON
cat backend/data/rl/shadow_mode/analysis/report_*.json | jq '.'
```

---

## 🎯 CRITÈRES VALIDATION SEMAINE 1

```yaml
Volume (MINIMUM):
  ✅ >100 prédictions
  ✅ >100 comparaisons
  ⭐ Idéal: >1000

Taux d'accord (OBJECTIF):
  ✅ >75% global
  ✅ >85% bookings haute priorité
  ✅ >90% prédictions haute confiance

Performance (CRITIQUE):
  ✅ Latence <100ms
  ✅ Zéro erreur critique
  ✅ Aucun impact système

Décision Vendredi:
  ✅ GO Phase 2 si critères atteints
  ⏸️  PAUSE si 60-75%
  ❌ NO-GO si <60%
```

---

## 🆘 SUPPORT

**Documentation :**

- `session/RL/PHASE_1_SHADOW_MODE_GUIDE.md` (Guide complet)
- `session/RL/INTEGRATION_SHADOW_MODE_PRATIQUE.md` (Guide pratique)
- `session/RL/TESTS_MANUELS_SHADOW_MODE.md` (Ce fichier)

**Problèmes :**
Voir section "🆘 DÉPANNAGE" dans `INTEGRATION_SHADOW_MODE_PRATIQUE.md`

**Contact :**
Vérifier logs: `docker-compose logs api --tail=100`

---

## 🏆 SUCCÈS

Si tous les tests passent :

```
╔════════════════════════════════════════════╗
║  ✅ SHADOW MODE OPÉRATIONNEL!              ║
║                                            ║
║  → Modèle DQN chargé                      ║
║  → API monitoring accessible              ║
║  → Logging automatique actif              ║
║  → Prédictions enregistrées               ║
║  → Comparaisons calculées                 ║
║                                            ║
║  🚀 Laisser tourner 1 semaine             ║
╚════════════════════════════════════════════╝
```

**Prochaine étape :**
→ Monitoring quotidien (5 min matin + 5 min soir)  
→ Analyse vendredi (30 min)  
→ Décision GO/NO-GO Phase 2

---

_Guide de tests manuels créé le 21 octobre 2025_  
_Durée totale : 15 minutes_  
_Validation : Shadow Mode opérationnel_ ✅
