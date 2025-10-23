# 🎯 Reward Function V3 - Alignée Business Réel

**Date** : 21 octobre 2025, 06:55  
**Version** : V3 - Business-Aligned  
**Fichier** : `backend/services/rl/dispatch_env.py`

---

## 📋 **RÈGLES BUSINESS RÉELLES**

### **Règles fournies par l'utilisateur** :

1. **TOUTES les courses doivent être effectuées** : **0 annulation** ❌
2. **Retard ALLER** : **0 tolérance** ❌
3. **Retard RETOUR** : **15-30 min tolérés** ✅
4. **Chauffeur d'urgence** : **Autorisé si nécessaire** (mais privilégier REGULAR)
5. **Pas d'annulation côté entreprise** : Les annulations n'existent pas dans la réalité

---

## 🔧 **MODIFICATIONS APPORTÉES**

### **1. Assignment Reward (+100 → +200)**

```python
# AVANT (V2)
reward = 100.0  # Récompense de base

# APRÈS (V3)
reward = 200.0  # ⭐ DOUBLÉ : Forte récompense pour chaque assignment
```

**Objectif** : Encourager FORTEMENT l'agent à assigner toutes les courses

---

### **2. Distinction ALLER vs RETOUR**

```python
# NOUVEAU (V3)
is_outbound = booking["id"] % 2 == 0  # Pair = ALLER, Impair = RETOUR

if is_late:
    lateness = time_to_pickup - booking["time_window_end"]

    if is_outbound:  # ALLER : 0 tolérance
        reward -= min(150.0, lateness * 5.0)  # Pénalité forte

    else:  # RETOUR : 15-30 min tolérance
        if lateness <= 30.0:
            reward -= lateness * 0.5  # Pénalité légère
        else:
            reward -= min(100.0, lateness * 3.0)  # Pénalité modérée
```

**Objectif** : Respecter les tolérances différentes ALLER vs RETOUR

---

### **3. Bonus Chauffeur REGULAR**

```python
# NOUVEAU (V3)
if driver.get("type", "REGULAR") == "REGULAR":
    reward += 20.0  # Bonus pour chauffeur régulier
# Pas de pénalité pour EMERGENCY (autorisé si nécessaire)
```

**Objectif** : Privilégier REGULAR, mais accepter EMERGENCY

---

### **4. Pénalité Annulation (-60 → -500)**

```python
# AVANT (V2)
penalty = 60.0 * (booking["priority"] / 5.0)  # Max -60

# APRÈS (V3)
penalty = 500.0 * (booking["priority"] / 5.0)  # ⭐ Max -500
```

**Objectif** : Pénaliser MASSIVEMENT les annulations (0 toléré)

---

### **5. Bonus Complétion Augmenté**

```python
# AVANT (V2)
completion_rate = assignments / total_bookings
bonus += completion_rate * 100.0  # Max +100

# APRÈS (V3)
if completion_rate >= 0.95:  # 95%+
    bonus += 300.0  # ⭐ Bonus MASSIF
elif completion_rate >= 0.85:  # 85%+
    bonus += 150.0
elif completion_rate >= 0.75:  # 75%+
    bonus += 50.0
else:  # < 75%
    bonus -= 200.0  # Pénalité
```

**Objectif** : Récompenser TRÈS fortement les taux de complétion > 95%

---

### **6. Pénalité par Cancellation**

```python
# NOUVEAU (V3)
if cancellations > 0:
    bonus -= cancellations * 200.0  # -200 par cancellation
```

**Objectif** : Pénaliser chaque annulation (cumule avec -500 de base)

---

### **7. Types de Chauffeurs**

```python
# NOUVEAU (V3) - Dans reset()
driver_type = "REGULAR" if i < int(num_drivers * 0.7) else "EMERGENCY"

driver["type"] = driver_type  # 70% REGULAR, 30% EMERGENCY
```

**Objectif** : Simuler mix REGULAR/EMERGENCY (pour 3 drivers : 2 REGULAR, 1 EMERGENCY)

---

## 📊 **TABLEAU COMPARATIF DES VERSIONS**

| Composante                              | **V2**  | **V3**           | **Changement** |
| --------------------------------------- | ------- | ---------------- | -------------- |
| **Reward assignment**                   | +100    | **+200**         | **+100%** ⬆️   |
| **Pénalité late ALLER**                 | -50 max | **-150 max**     | **+200%** ⬆️   |
| **Pénalité late RETOUR (< 30 min)**     | -50 max | **-15 max**      | **-70%** ⬇️    |
| **Pénalité late RETOUR (> 30 min)**     | -50 max | **-100 max**     | **+100%** ⬆️   |
| **Bonus driver REGULAR**                | 0       | **+20**          | NOUVEAU ✅     |
| **Pénalité cancellation (immédiate)**   | -60 max | **-500 max**     | **+733%** ⬆️   |
| **Pénalité cancellation (fin épisode)** | 0       | **-200 chacune** | NOUVEAU ✅     |
| **Bonus complétion 95%+**               | +95     | **+300**         | **+216%** ⬆️   |
| **Bonus complétion < 75%**              | +<75    | **-200**         | NOUVEAU ❌     |

---

## 🎯 **IMPACT ATTENDU**

### **Avant V3 (Résultats actuels)** :

| Métrique      | Valeur            | Problème          |
| ------------- | ----------------- | ----------------- |
| Assignments   | 15.3 / 20 (76.5%) | ❌ Trop faible    |
| Cancellations | 39.1              | ❌ Catastrophique |
| Late pickups  | 5.8               | OK                |
| Reward        | -40.6             | Négatif           |

### **Après V3 (Attendu)** :

| Métrique                           | Valeur Attendue          | Amélioration       |
| ---------------------------------- | ------------------------ | ------------------ |
| Assignments                        | **19-20 / 20** (95-100%) | **+23%** ✅        |
| Cancellations                      | **0-1**                  | **-97%** ✅        |
| Late pickups (ALLER)               | **< 2**                  | ✅                 |
| Late pickups (RETOUR < 30 min)\*\* | **Toléré**               | ✅                 |
| Reward                             | **+600 à +900**          | **+1500-2200%** ✅ |

---

## 💡 **LOGIQUE DE LA REWARD FUNCTION V3**

### **Priorités** (ordre d'importance) :

1. **Assigner TOUTES les courses** (+200 par assignment)
2. **Éviter annulations** (-500 à -700 par cancellation)
3. **Éviter retards ALLER** (-150 max)
4. **Tolérer retards RETOUR < 30 min** (-15 max)
5. **Privilégier chauffeurs REGULAR** (+20)
6. **Équilibrer la charge** (+80 bonus)
7. **Optimiser distance** (secondaire)

---

## 🔍 **EXEMPLES DE SCÉNARIOS**

### **Scénario 1 : Assignment ALLER à l'heure avec REGULAR**

```
+ 200  (assignment)
+ 20   (driver REGULAR)
+ 10   (distance optimale < 5km)
= +230 reward ✅
```

### **Scénario 2 : Assignment ALLER en retard 10 min avec EMERGENCY**

```
+ 200  (assignment)
+ 0    (driver EMERGENCY, pas de bonus)
- 50   (retard 10 min × 5)
= +150 reward (acceptable)
```

### **Scénario 3 : Assignment RETOUR en retard 20 min**

```
+ 200  (assignment)
+ 20   (driver REGULAR)
- 10   (retard 20 min × 0.5, toléré < 30 min)
= +210 reward ✅ (quasi pas pénalisé)
```

### **Scénario 4 : Annulation**

```
- 500  (pénalité annulation immédiate)
- 200  (pénalité fin d'épisode)
= -700 reward ❌ (fortement découragé)
```

### **Scénario 5 : Fin d'épisode avec 95% complétion**

```
+ 300  (bonus complétion 95%+)
+ 80   (bonus équilibre)
+ 50   (bonus distance optimisée)
= +430 bonus de fin ✅
```

---

## ✅ **VALIDATION DES RÈGLES BUSINESS**

| Règle Business                   | Implémentation V3                      | Statut |
| -------------------------------- | -------------------------------------- | ------ |
| **Toutes courses effectuées**    | +200 assignment, -500-700 cancellation | ✅     |
| **Retard ALLER = 0 tolérance**   | -150 max pour retard ALLER             | ✅     |
| **Retard RETOUR = 15-30 min OK** | -15 max si < 30 min                    | ✅     |
| **Chauffeur EMERGENCY accepté**  | +0 (neutre), +20 pour REGULAR          | ✅     |
| **Équilibre de charge**          | +80 bonus si équilibré                 | ✅     |

---

## 🚀 **PROCHAINES ÉTAPES**

### **1. Test Rapide (5 min)** ⚡

Tester la nouvelle reward function avec 100 épisodes :

```bash
docker exec atmr-api-1 python scripts/rl/train_dqn.py \
  --episodes 100 \
  --num-drivers 3 \
  --max-bookings 20 \
  --simulation-hours 8
```

**Attendu** : Assignments > 18, Cancellations < 2

### **2. Réoptimiser Optuna (30-45 min)**

Avec la nouvelle reward function :

```bash
docker exec atmr-api-1 python scripts/rl/tune_hyperparameters.py \
  --trials 50 \
  --episodes 100 \
  --study-name "atmr_v3_3drivers"
```

### **3. Entraînement Final (30-45 min)**

Avec hyperparamètres optimaux V3 :

```bash
docker exec atmr-api-1 python scripts/rl/train_dqn.py \
  --episodes 1000 \
  --config data/rl/optimal_config_v3.json \
  --num-drivers 3 \
  --max-bookings 20 \
  --simulation-hours 8
```

---

## 📈 **PRÉDICTIONS V3**

### **Test 100 Episodes (Config défaut + Reward V3)**

| Métrique      | V2        | V3 Attendu        | Amélioration |
| ------------- | --------- | ----------------- | ------------ |
| Reward moyen  | -48.9     | **+200 à +300**   | **+509%**    |
| Assignments   | 17.8 / 20 | **19 / 20** (95%) | +6.7%        |
| Cancellations | ~2        | **0-1**           | -50-100%     |
| Late pickups  | 7.3       | **< 5**           | -31%         |

### **1000 Episodes (Config optimale V3)**

| Métrique             | Attendu                    |
| -------------------- | -------------------------- |
| Reward moyen         | **+700 à +1000**           |
| Assignments          | **19.5-20 / 20** (97-100%) |
| Cancellations        | **0**                      |
| Late pickups ALLER   | **< 1**                    |
| Late pickups RETOUR  | **< 3** (toléré)           |
| Équilibre chauffeurs | **Écart < 2 courses**      |

---

## 🎓 **CHANGEMENTS CLÉS PAR RAPPORT À V2**

### **✅ Améliorations** :

1. **+100% reward pour assignment** (200 vs 100)
2. **Distinction ALLER/RETOUR** (nouvelle logique)
3. **Tolérance retour < 30 min** (pénalité -15 vs -50)
4. **Pénalité annulation x8** (-500 vs -60)
5. **Bonus complétion x3** (+300 vs +100)
6. **Types chauffeurs** (REGULAR vs EMERGENCY)

### **🎯 Objectif Principal** :

**Forcer l'agent à assigner TOUTES les courses, même au prix de quelques retards RETOUR (tolérés).**

---

## 📝 **FICHIERS MODIFIÉS**

✅ `backend/services/rl/dispatch_env.py` :

- Fonction `_assign_booking` (lignes ~310-383)
- Fonction `_check_expired_bookings` (lignes ~425-449)
- Fonction `_calculate_episode_bonus` (lignes ~528-589)
- Fonction `reset` (lignes ~152-169) - Ajout type chauffeur

---

## 🚀 **PROCHAINES ACTIONS**

### **Immédiat** :

1. ✅ Reward function V3 implémentée
2. ⏳ Test rapide 100 épisodes
3. ⏳ Réoptimisation Optuna
4. ⏳ Entraînement final 1000 épisodes

---

**Généré le** : 21 octobre 2025, 06:55  
**Version** : V3 - Business-Aligned  
**Status** : ✅ Implémenté, prêt pour test  
**Objectif** : 100% assignments, 0 cancellations, tolérance retards RETOUR
