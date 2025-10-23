# ⭐ REWARD FUNCTION V2 - ALIGNÉE BUSINESS

**Date :** 21 Octobre 2025  
**Objectif :** Aligner reward avec objectifs business  
**Statut :** ✅ **MODIFIÉ - Prêt pour réentraînement**

---

## 🎯 Objectifs Business

```
1. ✅ Maximiser le nombre d'assignments
2. ✅ Minimiser la distance parcourue
3. ✅ Contrôler les late pickups (<40% acceptable)
4. ✅ Minimiser les cancellations
```

---

## 📊 Changements Appliqués

### Fichier Modifié

**backend/services/rl/dispatch_env.py** (lignes 349-426)

---

### Comparaison V1 vs V2

| Composante               | V1 (Conservateur) | V2 (Aligné Business) | Changement   |
| ------------------------ | ----------------- | -------------------- | ------------ |
| **Assignment réussi**    | +50               | **+100**             | +100% ⭐     |
| **Late pickup**          | -100 max          | **-50 max**          | -50% ⭐      |
| **Cancellation**         | -200 max          | **-60 max**          | -70% ⭐      |
| **Distance penalty**     | Implicite         | **-distance/20**     | Explicite ⭐ |
| **Bonus distance < 5km** | +10 à +20         | +10 à +20            | Inchangé     |
| **Bonus priorité**       | +20 max           | +20 max              | Inchangé     |
| **Bonus rapidité**       | +15               | +15                  | Inchangé     |

---

## 💡 Rationale des Changements

### 1. Assignment Reward : +50 → +100 ⭐

**Problème V1 :**

```
Assignment réussi : +50
Cancellation      : -200
Ratio             : 1:4 (risque trop élevé)

→ Agent évite assignments risqués
→ Trop conservateur
→ Moins d'assignments (6.3 vs 7.5 baseline)
```

**Solution V2 :**

```
Assignment réussi : +100 ⭐
Cancellation      : -60
Ratio             : 1:0.6 (risque acceptable)

→ Encourage assignments
→ Plus agressif
→ Plus d'assignments attendues
```

---

### 2. Late Pickup Penalty : -100 → -50 ⭐

**Problème V1 :**

```
Late pickup : -100 (très pénalisant)

→ Agent refuse assignments risqués
→ Préfère canceller que risquer retard
→ Taux late pickups très bas (36.9%) mais peu d'assignments
```

**Solution V2 :**

```
Late pickup : -50 ⭐ (modéré)

→ Tolérance retards acceptable (<40%)
→ Encourage prendre risques calculés
→ Plus d'assignments avec contrôle late pickups
```

---

### 3. Cancellation Penalty : -200 → -60 ⭐

**Problème V1 :**

```
Cancellation : -200 (énorme pénalité)

→ Agent TERRIFI É d'annuler
→ Préfère ne rien faire que risquer annulation
→ Paralysie décisionnelle
```

**Solution V2 :**

```
Cancellation : -60 ⭐ (raisonnable)

→ Pénalité significative mais pas paralysante
→ Agent peut prendre risques
→ Équilibre assignments vs annulations
```

---

### 4. Distance Penalty : Ajout Explicite ⭐

**Ajout V2 :**

```python
reward -= distance / 20.0  # Pénalité explicite distance
```

**Effet :**

```
5 km  → -0.25 points
10 km → -0.50 points
20 km → -1.00 point
```

**Pourquoi :**

- Encourage proximité
- Mais pas trop pénalisant
- Maintient optimisation distance
- Compatible avec bonus distance < 5km

---

## 📈 Effets Attendus

### Comportement Agent V1 (Conservateur)

```
✅ Distance optimale (-20%)
❌ Trop peu d'assignments (6.3 vs 7.5)
❌ Trop de cancellations
⚠️ Trop prudent
```

### Comportement Agent V2 Attendu (Équilibré)

```
✅ Bon nombre d'assignments (7-8 attendu)
✅ Distance toujours optimisée
✅ Late pickups contrôlés (<40%)
✅ Moins de cancellations
✅ Plus agressif mais intelligent
```

---

## 🎯 Comparaison Équilibre Reward

### V1 : Déséquilibré

```
Best case assignment:
  +50 (base) +20 (distance) +20 (priorité) +15 (rapide)
  = +105 max

Worst case cancellation:
  -200 (priorité max)
  = -200 max

Ratio: +105 vs -200 → Déséquilibré vers prudence
```

### V2 : Équilibré ⭐

```
Best case assignment:
  +100 (base) +20 (distance) +20 (priorité) +15 (rapide)
  = +155 max

Worst case cancellation:
  -60 (priorité max)
  = -60 max

Ratio: +155 vs -60 → Équilibré, encourage action
```

**Agent sera plus confiant pour assigner !**

---

## 🚀 Prochaines Étapes

### Étape 1 : Test Rapide (5 min)

```bash
# Tester que l'environnement fonctionne
docker-compose exec api python scripts/rl/test_env_quick.py

# Devrait montrer rewards plus élevés
```

---

### Étape 2 : Réoptimisation Optuna (2-3h)

```bash
docker-compose exec api python scripts/rl/tune_hyperparameters.py \
  --trials 50 \
  --episodes 200 \
  --study-name dqn_optimization_v2 \
  --output data/rl/optimal_config_v2.json
```

**Attendu :**

- Best reward : -400 à -600 (vs -701.7 V1)
- Amélioration : +15-30% vs V1
- Plus d'assignments
- Distance toujours optimisée

---

### Étape 3 : Réentraînement (2-3h)

```bash
# Utiliser config optimale V2
docker-compose exec api python scripts/rl/train_dqn.py \
  --episodes 1000 \
  --learning-rate <optimal_v2_lr> \
  --gamma <optimal_v2_gamma> \
  --batch-size <optimal_v2_batch> \
  --num-drivers 6 \
  --max-bookings 10
```

---

### Étape 4 : Évaluation Finale

```bash
docker-compose exec api python scripts/rl/evaluate_agent.py \
  --model data/rl/models/dqn_best.pth \
  --episodes 100 \
  --compare-baseline \
  --num-drivers 6 \
  --max-bookings 10
```

**Critères de succès :**

```
✅ Assignments > 7/épisode (vs 6.3 V1, 7.5 baseline)
✅ Distance < 65 km/épisode (vs 59.9 V1, 75.2 baseline)
✅ Late pickups < 40% (vs 36.9% V1, 38.3% baseline)
✅ Complétion > 40% (vs 34.8% V1, 44.8% baseline)
```

---

## 📊 Prédictions V2

### Métriques Attendues

```
Assignments      : 7-8/épisode (+11-27% vs V1)
Distance         : 60-65 km (-13-18% vs baseline)
Late pickups     : 37-39% (-0.5 vs baseline, stable)
Taux complétion  : 40-45% (+5-10 pts vs V1)
Cancellations    : Réduites significativement
```

### Reward Attendu

```
Training reward  : -400 à -600 (vs -664.9 V1)
Best reward      : -300 à -400 (vs -518.2 V1)
Amélioration V2  : +15-30% vs V1
Amélioration totale: +70-80% vs baseline originale
```

---

## ✅ Validation

### Checklist Modifications

- [x] Assignment reward augmenté (+50 → +100)
- [x] Late pickup penalty réduite (-100 → -50)
- [x] Cancellation penalty réduite (-200 → -60)
- [x] Distance penalty ajoutée explicitement (-d/20)
- [x] Documentation reward mise à jour
- [x] 0 erreur linting

### Tests à Lancer

```bash
# Test environnement
python scripts/rl/test_env_quick.py

# Tests unitaires
pytest tests/rl/test_dispatch_env.py -v

# Validation reward plus élevés
```

---

## 🏆 Résumé Changements

```
╔═══════════════════════════════════════════════╗
║  ✅ REWARD FUNCTION V2 CRÉÉE                  ║
║  ⭐ ASSIGNMENT: +50 → +100 (+100%)            ║
║  ⭐ LATE PICKUP: -100 → -50 (-50%)            ║
║  ⭐ CANCELLATION: -200 → -60 (-70%)           ║
║  ⭐ DISTANCE: Pénalité explicite -d/20        ║
║  ✅ ALIGNÉE AVEC OBJECTIFS BUSINESS           ║
╚═══════════════════════════════════════════════╝
```

---

## 🎯 Prochaine Action Immédiate

**Tester que ça fonctionne :**

```bash
docker-compose exec api python scripts/rl/test_env_quick.py
```

**Puis lancer réoptimisation :**

```bash
docker-compose exec api python scripts/rl/tune_hyperparameters.py \
  --trials 50 \
  --episodes 200 \
  --study-name dqn_optimization_v2 \
  --output data/rl/optimal_config_v2.json
```

**Résultat attendu :** Agent plus agressif, plus d'assignments, distance optimisée, meilleur équilibre global ! 🎯

---

_Reward V2 créée le 21 octobre 2025_  
_Alignée avec objectifs business_ ✅  
_Prête pour réoptimisation !_ 🚀
