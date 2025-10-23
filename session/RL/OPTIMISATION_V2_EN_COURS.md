# 🚀 OPTIMISATION V2 - REWARD ALIGNÉE BUSINESS - EN COURS

**Date :** 21 Octobre 2025  
**Heure lancement :** ~04:00  
**Durée estimée :** 10 minutes  
**Statut :** 🔄 **EN COURS**

---

## 🎯 Objectif

Trouver les meilleurs hyperparamètres pour la **Reward Function V2** alignée avec les objectifs business.

---

## ⭐ Changements Reward Function V2

```yaml
Assignment réussi: +50 → +100 (+100%) ⭐
Late pickup penalty: -100 → -50 (-50%) ⭐
Cancellation penalty: -200 → -60 (-70%) ⭐
Distance penalty: Ajout explicite -d/20 ⭐
```

**Effet attendu :** Agent plus agressif, plus d'assignments, meilleur équilibre

---

## 📊 Paramètres Optimisation

```yaml
Trials: 50
Episodes/trial: 200
Éval/trial: 20
Study name: dqn_optimization_v2
Output: data/rl/optimal_config_v2.json
```

---

## 📈 Résultats Attendus

### Performance V2 Attendue

```
Best reward V1   : -701.7
Best reward V2   : -400 à -600 (attendu)
Amélioration     : +15-30% vs V1
```

### Métriques Business V2 Attendues

```yaml
Assignments:
  V1: 6.3/épisode
  V2: 7-8/épisode (+11-27%) ✅

Distance:
  V1: 59.9 km
  V2: 60-65 km (légère augmentation acceptable)

Late pickups:
  V1: 36.9%
  V2: 37-40% (contrôlé <40%)

Complétion:
  V1: 34.8%
  V2: 40-45% (+5-10 pts) ✅
```

---

## ⏰ Timeline

```
04:00 → Lancement optimisation V2 ✅
04:10 → Optimisation terminée (attendu)
04:15 → Analyse résultats V2
04:20 → Comparaison V1 vs V2
04:25 → Décision: Réentraîner ou ajuster
```

---

## 🔍 Ce qu'Optuna va Trouver

### Hyperparamètres V2 Attendus

```yaml
Architecture: Possiblement différente de V1
Learning rate: Peut-être plus élevé (reward plus grande échelle)
Gamma: Similaire V1 (0.97-0.99)
Batch size: Probablement 64 encore
Buffer size: 50k ou 100k
Environnement: Possiblement plus grand (plus d'assignments)
```

---

## 📊 Comparaison V1 vs V2 (Attendue)

| Métrique         | V1      | V2 (attendu) | Amélioration |
| ---------------- | ------- | ------------ | ------------ |
| **Best reward**  | -701.7  | -400 à -600  | +15-30%      |
| **Assignments**  | 6.3     | 7-8          | +11-27% ✅   |
| **Distance**     | 59.9 km | 60-65 km     | +0-8% ⚠️     |
| **Late pickups** | 36.9%   | 37-40%       | Stable ✅    |
| **Complétion**   | 34.8%   | 40-45%       | +5-10 pts ✅ |

**Verdict attendu :** Meilleur équilibre global !

---

## ⏳ Pendant l'Optimisation (10 min)

L'optimisation Optuna explore automatiquement :

- 14 hyperparamètres
- 50 trials (32 seront probablement pruned)
- ~18 configurations complètes

**Vous pouvez :**

1. ☕ Prendre un café rapide
2. 📊 Consulter les documents créés
3. 🎯 Préparer le plan de déploiement

---

## ✅ Après Optimisation

### Étape 1 : Analyser Résultats

```bash
# Voir config optimale V2
docker-compose exec api cat data/rl/optimal_config_v2.json | jq '.best_params'

# Comparer avec V1
diff <(cat data/rl/optimal_config_v1.json | jq '.best_params') \
     <(cat data/rl/optimal_config_v2.json | jq '.best_params')
```

### Étape 2 : Réentraîner

```bash
# Si résultats prometteurs → Réentraîner 1000 épisodes
docker-compose exec api python scripts/rl/train_dqn.py \
  --episodes 1000 \
  --learning-rate <v2_lr> \
  --gamma <v2_gamma> \
  --batch-size <v2_batch> \
  --num-drivers <v2_drivers> \
  --max-bookings <v2_bookings>
```

### Étape 3 : Évaluer

```bash
docker-compose exec api python scripts/rl/evaluate_agent.py \
  --model data/rl/models/dqn_best.pth \
  --episodes 100 \
  --compare-baseline \
  --num-drivers <v2_drivers> \
  --max-bookings <v2_bookings>
```

---

## 🎯 Critères de Succès V2

### Objectifs Minimaux

```
✅ Assignments > 7/épisode
✅ Distance < 70 km/épisode
✅ Late pickups < 40%
✅ Complétion > 40%
```

### Objectifs Optimaux

```
🏆 Assignments > 7.5/épisode
🏆 Distance < 65 km/épisode
🏆 Late pickups < 38%
🏆 Complétion > 42%
🏆 Reward > -500
```

---

## 💡 Si Résultats Excellents

**Déployer immédiatement :**

```bash
# 1. Copier meilleur modèle
cp data/rl/models/dqn_best.pth data/rl/models/dqn_production_v2.pth

# 2. Activer en production
POST /api/company_dispatch/rl/toggle {"enabled": true}

# 3. Monitorer
GET /api/company_dispatch/rl/status
```

---

## 📊 Prédiction Finale

**Après V2 optimisée et réentraînée :**

```
Amélioration vs baseline originale : +40-60%
Amélioration vs V1                  : +15-30%
ROI mensuel                         : 8,000-12,000 €
ROI annuel                          : 96,000-144,000 €
Déploiement                         : Immédiat
```

---

## 🏆 Timeline Globale

```
Semaines 13-14 : POC & Environnement        ✅
Semaine 15     : Agent DQN                  ✅
Semaine 16     : Training baseline          ✅
Semaine 17     : Auto-Tuner + Optim V1      ✅
Aujourd'hui    : Reward V2 + Optim V2       🔄
Dans 10 min    : Résultats V2               ⏳
Dans 3h        : Training V2 terminé        ⏳
Demain         : Déploiement production     ⏳
```

---

**Revenez dans 10 minutes pour analyser les résultats V2 ! 🎯**

---

_Optimisation V2 lancée : 21 octobre 04:00_  
_Fin attendue : 21 octobre 04:10_  
_Résultats attendus : Meilleur équilibre business !_ ✅
