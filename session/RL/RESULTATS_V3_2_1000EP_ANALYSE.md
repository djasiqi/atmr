# ⚠️ Résultats Entraînement V3.2 (1000 Episodes) - Analyse Complète

**Date** : 21 octobre 2025, 13:50  
**Durée** : ~40 minutes  
**Configuration** : 4 drivers (3R+1E), 25 bookings, retard RETOUR ≤ 20 min  
**Status** : ❌ **EFFONDREMENT APRÈS EPISODE 200**

---

## 📊 **RÉSULTATS FINAUX**

| Métrique            | Résultat             | Target              | Écart          |
| ------------------- | -------------------- | ------------------- | -------------- |
| **Reward moyen**    | **-8,436.7**         | +2,000 à +3,500     | **-524%** ❌   |
| **Assignments**     | **7.7 / 25** (30.8%) | 23-24 / 25 (92-96%) | **-67.7%** ❌  |
| **Late pickups**    | 2.5                  | < 3                 | ✅ OK          |
| **Epsilon final**   | 0.055                | 0.055               | ✅ OK          |
| **Meilleur modèle** | **Episode 200**      | Episode 1000        | ❌ Dégradation |

---

## 📈 **COURBE D'APPRENTISSAGE - EFFONDREMENT VISIBLE**

### **Évolution du Reward (Evaluations)** :

| Episode  | Reward (Eval)  | Assignments          | Status            |
| -------- | -------------- | -------------------- | ----------------- |
| **50**   | -4,211         | 16.4 / 25 (65.6%)    | Apprentissage     |
| **100**  | **-3,099** ✅  | 18.3 / 25 (73.2%)    | Amélioration      |
| **150**  | -3,549         | 17.4 / 25 (69.6%)    | Stable            |
| **200**  | **-2,201** 🏆  | 18.0 / 25 (72.0%)    | **MEILLEUR !**    |
| **250**  | -3,953 ⚠️      | N/A                  | Début dégradation |
| **300**  | -3,846 ⚠️      | N/A                  | Dégradation       |
| **400**  | -4,880 ⚠️      | N/A                  | Dégradation       |
| **550**  | -4,224 ⚠️      | 17.9 / 25            | Dégradation       |
| **600**  | -4,802 ⚠️      | 14.6 / 25 (58.4%)    | Dégradation forte |
| **650**  | -6,039 ❌      | 11.8 / 25 (47.2%)    | Effondrement      |
| **700**  | -5,506 ❌      | 13.3 / 25 (53.2%)    | Effondrement      |
| **800**  | -5,670 ❌      | 13.9 / 25 (55.6%)    | Effondrement      |
| **850**  | -6,901 ❌      | 9.7 / 25 (38.8%)     | Effondrement      |
| **900**  | **-10,353** ❌ | **4.5 / 25 (18%)**   | **CATASTROPHE**   |
| **1000** | **-9,518** ❌  | **7.7 / 25 (30.8%)** | Catastrophe       |

### **Graphique ASCII** :

```
Reward
    0 ┤
-2000 ┤         ╭─────╮ ← MEILLEUR (Ep 200)
-4000 ┤    ╭────╯     ╰───╮
-6000 ┤────╯                ╰────╮
-8000 ┤                          ╰─────╮
-10000┤                                 ╰──────
      └─────────────────────────────────────────>
      Ep 50  150  250  350  450  550  650  750  900
```

**→ PIC à Episode 200, puis DÉGRADATION PROGRESSIVE → EFFONDREMENT** ❌

---

## 🔍 **DIAGNOSTIC : POURQUOI L'EFFONDREMENT ?**

### **Observation Critique : LOSS EXPLOSE** ⚡

| Episode  | Loss        | Status                     |
| -------- | ----------- | -------------------------- |
| **10**   | 61.6        | ✅ Normal                  |
| **100**  | 194.4       | ✅ Normal                  |
| **200**  | 240.9       | ✅ Acceptable              |
| **400**  | 617.5       | ⚠️ Élevé                   |
| **600**  | 12,921      | ❌ **EXPLOSION !**         |
| **800**  | 271,513     | ❌ **CATASTROPHE !**       |
| **1000** | **850,044** | ❌ **DIVERGENCE TOTALE !** |

**→ Le Q-Network DIVERGE après Episode 200-300 !** ⚡

---

## 🚨 **CAUSES IDENTIFIÉES**

### **1. REWARD FUNCTION ENCORE TROP PUNITIVE** ⚠️

```python
Problème:
├─ Agent assigne 7.7 / 25 courses (30.8%)
├─ 17.3 courses annulées
└─ Pénalité: 17.3 × -150 = -2,595

Agent apprend:
└─ "Ne PAS assigner = éviter pénalités retards"
   → Mais cause ANNULATIONS massives
   → Reward encore plus négatif
   → Cercle vicieux ❌
```

### **2. LEARNING RATE TROP ÉLEVÉ POUR 4 DRIVERS** ⚠️

```
Learning rate: 0.00674
├─ Optimal pour 3 drivers (V3.1) ✅
├─ TROP ÉLEVÉ pour 4 drivers ❌
└─ Cause: State space 118 dim (vs 94)

État plus complexe = LR doit être PLUS BAS
→ LR 0.00674 cause instabilité/divergence
```

### **3. PÉNALITÉ RETOUR 20 MIN TROP STRICTE** ⚠️

```python
Retard RETOUR 25 min:
├─ V3.1 (30 min max): Pénalité -12.5 (toléré)
└─ V3.2 (20 min max): Pénalité -100 (hors tolérance ❌)

Agent apprend:
└─ "Risque retard RETOUR > 20 min = NE PAS ASSIGNER"
   → Préfère ne rien faire
   → Annulations massives
```

---

## 📊 **COMPARAISON V3.1 vs V3.2**

| Config   | Drivers | Bookings | Best Reward (Ep)    | Assignments       | Résultat        |
| -------- | ------- | -------- | ------------------- | ----------------- | --------------- |
| **V3.1** | 3       | 20       | **-233** (Ep 150)   | 12.7 / 20 (63.5%) | ⚠️ Dégradation  |
| **V3.2** | 4       | 25       | **-2,201** (Ep 200) | 7.7 / 25 (30.8%)  | ❌ Effondrement |

**→ V3.2 PIRE que V3.1 malgré plus de drivers !** ❌

---

## 🎯 **CAUSES RACINES**

### **1. Reward Function Inadaptée** 🎯

```
Pénalités actuelles TROP FORTES:
├─ Annulation: -150 (immédiat) + -100 (fin épisode) = -250
├─ Retard RETOUR > 20 min: -120 (vs -100 en V3.1)
├─ Retard ALLER > 30 min: -150

Agent calcule:
├─ Assigner avec risque retard 25 min RETOUR = -100 à -120
├─ Ne pas assigner = -150 (annulation)
└─ Différence: seulement -30 à -50

→ Agent préfère ATTENDRE plutôt qu'assigner ! ❌
```

### **2. Learning Rate Non Adapté** ⚠️

```
Optuna optimisé pour:
├─ 11 drivers, 10 bookings (State dim 90)
└─ LR 0.00674 optimal

Production V3.2:
├─ 4 drivers, 25 bookings (State dim 118)
└─ LR 0.00674 TROP ÉLEVÉ (+31% state complexity)

Impact:
└─ Gradients trop forts → Oubli catastrophique → Divergence
```

### **3. Complexité Augmentée** 🧠

```
V3.1 (3 drivers):
├─ State dim: 94
├─ Actions: 61
└─ Q-Network: 220k params

V3.2 (4 drivers):
├─ State dim: 118 (+25.5%)
├─ Actions: 101 (+65.6%)
└─ Q-Network: 238k params (+7.9%)

→ Plus complexe = Besoin de:
   - LR plus bas
   - Reward function plus tolérante
   - Plus d'episodes (1500-2000)
```

---

## 💡 **SOLUTIONS POSSIBLES**

### **SOLUTION A : Ajuster Reward Function (RECOMMANDÉ)** ⭐

**Changements à faire** :

```python
1. RÉDUIRE pénalité annulation: -150 → -100
2. AUGMENTER tolérance RETOUR: 20 min → 25 min
3. RÉDUIRE pénalité retard RETOUR: -0.75 → -0.4 par minute
4. AUGMENTER reward assignment: +300 → +400

Objectif: Encourager agent à ASSIGNER même avec risque petit retard
```

### **SOLUTION B : Réduire Learning Rate** 🎓

**Nouveau LR optimal** :

```
LR actuel: 0.00674 (trop élevé pour 4 drivers)
LR suggéré: 0.004-0.005 (40% reduction)

Relancer Optuna pour 4 drivers, 25 bookings
→ Trouve LR optimal pour cette config
```

### **SOLUTION C : Retour à 3 Drivers + Reward Ajustée** 🔄

**Configuration** :

```
Drivers: 3 (2 REGULAR + 1 EMERGENCY)
Bookings: 25
Retard RETOUR: 25 min (compromis vs 20 min)
LR: 0.00674 (conservé)
Reward: Ajustée (plus tolérante)

Avantage: Même complexité que V3.1, mais plus de courses
```

---

## 🚀 **MA RECOMMANDATION**

### **SOLUTION A + C COMBINÉES** ⭐

**Configuration Proposée** :

| Paramètre               | V3.2 (échec) | **V3.3 (proposé)** | Justification           |
| ----------------------- | ------------ | ------------------ | ----------------------- |
| **Drivers**             | 4            | **3**              | Réduire complexité      |
| **REGULAR**             | 3            | **2**              | Revenir à config stable |
| **EMERGENCY**           | 1            | **1**              | Conservé                |
| **Bookings**            | 25           | **25**             | Garder volume réel      |
| **Retour tolérance**    | 20 min       | **25 min**         | Plus réaliste           |
| **Pénalité annulation** | -150         | **-100**           | Moins punitive          |
| **Reward assignment**   | +300         | **+400**           | Plus incitatif          |
| **LR**                  | 0.00674      | **0.00674**        | Conservé                |

---

## 📋 **PROCHAINES ÉTAPES RECOMMANDÉES**

### **Option 1 : Ajuster Reward V3.3 (RECOMMANDÉ)** ⭐

```
1. Modifier dispatch_env.py:
   ├─ Retard RETOUR: 25 min (vs 20 min)
   ├─ Pénalité annulation: -100 (vs -150)
   ├─ Reward assignment: +400 (vs +300)
   └─ Pénalité RETOUR: -0.4 (vs -0.75) par minute

2. Retour à 3 drivers (2R+1E):
   └─ Config stable et prouvée

3. Tester 100 episodes:
   └─ Valider que reward function fonctionne

4. Si OK → 1000 episodes
```

### **Option 2 : Relancer Optuna pour 4 Drivers** 🔧

```
Trouver LR optimal pour:
├─ 4 drivers
├─ 25 bookings
└─ Reward V3.2

Durée: 10-15 minutes (50 trials)
```

### **Option 3 : Utiliser Meilleur Modèle (Episode 200)** 📦

```
Charger le checkpoint Episode 200:
└─ dqn_ep0200_r-3977.pth
   Reward: -2,200.6
   Assignments: 18.0 / 25 (72%)

Évaluer si utilisable en production:
└─ 72% assignments = acceptable pour semi-auto mode
```

---

## 🔬 **ANALYSE DÉTAILLÉE**

### **Point de Pic : Episode 200** 🏆

```
Episode 200:
├─ Reward (eval): -2,200.6 ✅ MEILLEUR
├─ Assignments: 18.0 / 25 (72%)
├─ Epsilon: 0.559 (exploration/exploitation)
└─ Loss: 240.9 (acceptable)

→ Agent avait trouvé un bon équilibre !
```

### **Dégradation Progressive (Ep 200-600)** ⚠️

```
Episode 200 → 600:
├─ Reward: -2,201 → -4,802 (-118%)
├─ Assignments: 18.0 → 14.6 (-18.9%)
├─ Loss: 240 → 12,921 (+5,283%)
└─ Epsilon: 0.559 → 0.175 (moins d'exploration)

Cause: Learning rate trop élevé + Epsilon trop bas
→ Agent oublie stratégies apprises (catastrophic forgetting)
```

### **Effondrement Total (Ep 600-1000)** ❌

```
Episode 600 → 1000:
├─ Reward: -4,802 → -9,518 (-98%)
├─ Assignments: 14.6 → 7.7 (-47.3%)
├─ Loss: 12,921 → 850,044 (+6,477%)
└─ Epsilon: 0.175 → 0.055 (pure exploitation)

Agent complètement divergé:
└─ Choisit stratégie "Ne rien faire"
   → Assignments chutent à 30%
   → Reward catastrophique
```

---

## 💡 **LEÇON APPRISE**

### **Reward Function V3.2 EST TROP PUNITIVE** ⚠️

```
Problème fondamental:
├─ Pénalité annulation: -150 (trop forte)
├─ Pénalité retard RETOUR > 20 min: -120 (trop stricte)
└─ Retard RETOUR 20 min: Trop stricte pour 25 courses

Agent apprend:
└─ "Ne pas assigner = moins de pénalités que assigner avec retard"
   → Stratégie d'évitement
   → Annulations massives
   → Reward négatif
```

---

## 🎯 **PROPOSITION : REWARD FUNCTION V3.3**

### **Changements Proposés** :

| Aspect                       | V3.2 (échec)              | **V3.3 (proposé)**              | Justification    |
| ---------------------------- | ------------------------- | ------------------------------- | ---------------- |
| **Reward assignment**        | +300                      | **+500**                        | Incitation FORTE |
| **Pénalité annulation**      | -150 immédiat<br>-100 fin | **-80 immédiat**<br>**-70 fin** | Moins punitive   |
| **Retard RETOUR toléré**     | 20 min                    | **25 min**                      | Plus réaliste    |
| **Pénalité RETOUR ≤ 25 min** | -0.75/min                 | **-0.3/min**                    | Moins stricte    |
| **Pénalité RETOUR > 25 min** | -4/min (max -120)         | **-2/min (max -80)**            | Moins stricte    |

### **Ratios Nouveaux** :

```
V3.2 (échec):
├─ Assignment: +300
├─ Annulation: -250 (total)
└─ Ratio: 1.2:1 (pénalité trop proche de reward)

V3.3 (proposé):
├─ Assignment: +500 ⬆️
├─ Annulation: -150 (total) ⬇️
└─ Ratio: 3.3:1 (reward CLAIREMENT supérieur à pénalité)

→ Agent VEUT assigner ! ✅
```

---

## 📋 **COMMANDE PROPOSÉE V3.3**

### **Test Rapide (100 episodes)** :

```bash
# Après ajustement reward function:
docker exec atmr-api-1 python scripts/rl/train_dqn.py \
  --episodes 100 \
  --num-drivers 3 \
  --max-bookings 25 \
  --simulation-hours 8 \
  --learning-rate 0.00674 \
  --gamma 0.9392 \
  --batch-size 64 \
  --epsilon-decay 0.9971 2>&1 | Tee-Object -FilePath "training_v3_3_test_100ep.txt"
```

**Résultats attendus (100 episodes)** :

- Reward : -1,000 à +500
- Assignments : 20-22 / 25 (80-88%)

---

## ✅ **RÉSUMÉ EXÉCUTIF**

### **❌ ÉCHEC V3.2**

| Aspect              | Résultat             | Cause                          |
| ------------------- | -------------------- | ------------------------------ |
| **Reward final**    | -8,436.7             | Reward function trop punitive  |
| **Assignments**     | 7.7 / 25 (30.8%)     | Agent évite d'assigner         |
| **Loss divergence** | 850,044              | LR trop élevé + complexité     |
| **Point de pic**    | Episode 200 (-2,201) | Après, effondrement progressif |

### **✅ ACTIONS RECOMMANDÉES**

1. **Ajuster reward function** (V3.3) :

   - +500 assignment (vs +300)
   - -150 annulation totale (vs -250)
   - 25 min tolérance RETOUR (vs 20 min)
   - Pénalités plus légères

2. **Retour à 3 drivers** :

   - Réduire complexité
   - LR 0.00674 validé pour 3 drivers

3. **Test 100 episodes** avant final

---

## 🚀 **PROCHAINE ÉTAPE**

**Voulez-vous que j'implémente la Reward Function V3.3 ?** 🎯

**Changements** :

- ✅ 3 drivers (2 REGULAR + 1 EMERGENCY)
- ✅ 25 bookings (volume réel conservé)
- ✅ Retard RETOUR 25 min (plus réaliste)
- ✅ Pénalités réduites (encourager assignments)
- ✅ Reward +500 (incitation forte)

---

**Généré le** : 21 octobre 2025, 13:50  
**Status** : ❌ V3.2 échoué (effondrement Episode 200-1000)  
**Meilleur modèle** : Episode 200 (-2,201 reward, 72% assignments)  
**Cause** : Reward function trop punitive + LR trop élevé  
**Solution** : Reward Function V3.3 (plus tolérante)
