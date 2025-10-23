# 🔴 ANALYSE DES PROBLÈMES DE DISPATCH

## 📋 **RÉSUMÉ DES PROBLÈMES IDENTIFIÉS**

### **Problème 1 : Conflit temporel toujours présent (08:30)** 🔴 CRITIQUE

```
Dris Daoudi a 2 courses À LA MÊME HEURE (08:30) :

1. Francois Bottiglieri : Clinique Anières → Carouge
2. Daniel Richard Bertossa : Clinique Anières → Meyrin

Impact :
- Distance si regroupées : 33 km (détour)
- Temps : 50 minutes
VS
- Distance si séparées : 20 km total
- Temps : 35 minutes total

Conclusion : Regroupement MAUVAIS, devrait être 2 chauffeurs séparés
```

---

### **Problème 2 : Règles de regroupement (pooling) trop permissives** 🟠 IMPORTANT

**Configuration actuelle** :

```python
pooling:
  enabled: true
  time_tolerance_min: 10      # Écart max entre pickups
  pickup_distance_m: 500      # Distance max entre pickups
  max_detour_min: 15          # Détour max acceptable
```

**Problème** :

- Détour de 15 min est TROP permissif
- Distance de 500m peut être trop pour certaines zones
- Regroupe même si destinations très éloignées

**Règles attendues** :

```yaml
Regrouper SEULEMENT si: ✅ Même lieu de départ (< 1 km)
  ✅ Même heure (< 5 min d'écart)
  ✅ Détour minimal (< 5 min OU < 1 km)
  ✅ Destinations dans même direction

Sinon: ❌ Assigner à 2 chauffeurs différents
```

---

### **Problème 3 : Surcharge chauffeur (Giuseppe Bekasy)** 🟡 MOYEN

```
Giuseppe Bekasy : 4 courses consécutives

09:15 → Ketty : Collonge → Anières         ✅ OK
10:00 → Bernard : Clinique → Carouge       ✅ OK (même région)
11:00 → Jeannette : Clinique → Thônex      ⚠️ Surcharge
13:00 → Pierre : Onex → Onex               ⚠️ Surcharge

Total : 4 courses en 4 heures
```

**Problème** :

- Giuseppe a 4 courses alors que d'autres chauffeurs en ont 1-2
- Déséquilibre de charge

**Équilibrage attendu** :

```
Giuseppe : 2 courses (09:15, 10:00)
Yannis : 2 courses (11:00, 13:00)
Dris : 2 courses (08:30 Francois, 16:00 Ketty)
Khalid (urgence) : 0 courses (gardé en réserve)
```

---

### **Problème 4 : Utilisation chauffeur d'urgence** 🟡 MOYEN

```
Khalid Alaoui (URGENCE) assigné à :
  13:15 → Désirée : Thônex → Genève

Question : Pourquoi un chauffeur d'urgence pour une course normale ?
```

**Hypothèses** :

1. Pas assez de chauffeurs réguliers disponibles
2. Paramètre `allow_emergency: true` activé
3. Giuseppe surchargé, système utilise Khalid en dernier recours

**Solution** :

- Mieux équilibrer entre chauffeurs réguliers
- Garder Khalid en réserve (seulement si vraie urgence)

---

## 🔧 **SOLUTIONS PROPOSÉES**

### **Solution 1 : Désactiver regroupement pour ce type de course** ✅ IMMÉDIAT

Le regroupement (pooling) ne fonctionne pas bien pour vos courses médicales.

**Ajustement paramètres** :

```yaml
pooling:
  enabled: false  # ✅ Désactiver complètement

# OU (si vous voulez garder pooling) :
pooling:
  enabled: true
  time_tolerance_min: 5       # ✅ Réduire de 10 → 5 min
  pickup_distance_m: 300      # ✅ Réduire de 500 → 300 m
  max_detour_min: 5           # ✅ Réduire de 15 → 5 min
```

**Impact** :

- ✅ Chaque course = 1 chauffeur
- ✅ Pas de détours inutiles
- ✅ Temps optimisés
- ❌ Plus de chauffeurs nécessaires

---

### **Solution 2 : Augmenter l'équilibrage de charge** ✅ IMMÉDIAT

**Ajustement paramètres** :

```yaml
heuristic:
  proximity: 0.3 # ✅ Réduire (distance moins importante)
  driver_load_balance: 0.85 # ✅ Augmenter (équilibre strict)
  priority: 0.06 # Garder pareil

fairness:
  enable_fairness: true
  fairness_window_days: 7
  fairness_weight: 0.5 # ✅ Augmenter (de 0.3 → 0.5)
```

**Impact** :

- ✅ Giuseppe : 2 courses au lieu de 4
- ✅ Autres chauffeurs plus utilisés
- ✅ Équilibre parfait

---

### **Solution 3 : Limiter courses par chauffeur** ✅ IMMÉDIAT

**Ajustement paramètres** :

```yaml
solver:
  max_bookings_per_driver: 3 # ✅ Réduire de 6 → 3
```

**Impact** :

- ✅ Giuseppe max 3 courses
- ✅ Force distribution sur autres chauffeurs
- ✅ Moins de fatigue

---

### **Solution 4 : Garder chauffeurs d'urgence en réserve** ✅ IMMÉDIAT

**Option A : Désactiver chauffeurs d'urgence**

```yaml
allow_emergency: false # ✅ Ne pas utiliser Khalid sauf vraie urgence
```

**Option B : Augmenter pénalité utilisation urgence**

```python
# Dans heuristic weights
emergency_driver_penalty: 0.5  # Pénalise fortement l'utilisation
```

**Impact** :

- ✅ Khalid gardé en réserve
- ✅ Utilisé seulement si vraie surcharge

---

## 🎯 **CONFIGURATION RECOMMANDÉE POUR VOS BESOINS**

Basé sur votre analyse, voici la configuration optimale :

```yaml
# ========== DISPATCH OVERRIDES (via bouton ⚙️ Avancé) ==========

heuristic:
  proximity: 0.3 # Distance moins prioritaire
  driver_load_balance: 0.85 # Équilibre STRICT
  priority: 0.06 # Pareil

solver:
  time_limit_sec: 60
  max_bookings_per_driver: 3 # ✅ MAX 3 courses/chauffeur
  unassigned_penalty_base: 10000

service_times:
  pickup_service_min: 5
  dropoff_service_min: 10
  min_transition_margin_min: 20 # ✅ Marge 20 min (éviter rush)

pooling:
  enabled: false # ✅ DÉSACTIVER le regroupement

fairness:
  enable_fairness: true
  fairness_window_days: 7
  fairness_weight: 0.5 # ✅ Équité importante

# Chauffeurs d'urgence : NE PAS utiliser
allow_emergency: false # ✅ Khalid gardé en réserve
```

---

## 📊 **RÉSULTAT ATTENDU AVEC CETTE CONFIG**

### **Répartition optimale** :

```
Giuseppe Bekasy (3 courses max) :
  09:15 → Ketty : Collonge → Anières
  10:00 → Bernard : Clinique → Carouge
  11:00 → Jeannette : Clinique → Thônex

Dris Daoudi (3 courses max) :
  08:30 → Francois : Clinique → Carouge
  16:00 → Ketty : Anières → Collonge

Yannis Labrot (2-3 courses) :
  13:00 → Gisèle : Vesenaz → Genève
  (+ 1 autre si nécessaire)

Autre chauffeur régulier (2-3 courses) :
  08:30 → Daniel : Clinique → Meyrin      ← Séparé de Francois !
  13:00 → Pierre : Onex → Onex
  13:15 → Désirée : Thônex → Genève      ← Chauffeur régulier, pas Khalid !

Khalid Alaoui (urgence) :
  0 courses (gardé en réserve)            ← Disponible pour vraies urgences
```

---

## 🛠️ **COMMENT APPLIQUER ?**

### **Méthode 1 : Via interface (Paramètres Avancés)** ⭐ RECOMMANDÉ

1. Page Dispatch → Cliquer **"⚙️ Avancé"**
2. Ajuster :
   - Équilibre charge : **0.85**
   - Courses max/chauffeur : **3**
   - Activer regroupement : **DÉCOCHER** ❌
   - Poids équité : **0.5**
3. Cliquer **"✅ Appliquer ces paramètres"**
4. Relancer dispatch

---

### **Méthode 2 : Modification backend (Permanent)**

Si vous voulez que ces paramètres soient **par défaut** :

```python
# backend/services/unified_dispatch/settings.py

DEFAULT_HEURISTIC_WEIGHTS = {
    "proximity": 0.3,           # ← Modifier de 0.2 → 0.3
    "driver_load_balance": 0.85, # ← Modifier de 0.7 → 0.85
    "priority": 0.06
}

DEFAULT_SOLVER_SETTINGS = {
    "max_bookings_per_driver": 3  # ← Modifier de 6 → 3
}

DEFAULT_POOLING_SETTINGS = {
    "enabled": False             # ← Modifier de True → False
}

DEFAULT_FAIRNESS_SETTINGS = {
    "fairness_weight": 0.5       # ← Modifier de 0.3 → 0.5
}
```

---

## 🧪 **TEST IMMÉDIAT**

**Sans modifier le code** : Utilisez les paramètres avancés !

1. Page Dispatch
2. Cliquer **"⚙️ Avancé"**
3. Configurer :

```
🎯 Poids Heuristique
├─ Proximité : 0.3
├─ Équilibre charge : 0.85
└─ Priorité : 0.06

🔧 Optimiseur
├─ Temps limite : 60s
├─ Courses max/chauffeur : 3
└─ Pénalité non-assigné : 10000

⏱️ Temps Service
├─ Pickup : 5 min
├─ Dropoff : 10 min
└─ Marge transition : 20 min

👥 Regroupement
└─ Activer : DÉCOCHER ❌

⚖️ Équité
├─ Activer : COCHER ✅
├─ Fenêtre : 7 jours
└─ Poids : 0.5
```

4. Cliquer **"✅ Appliquer"**
5. **Relancer dispatch**
6. **Vérifier résultat** :
   - Giuseppe max 3 courses
   - Dris 1 seule course à 08:30 (pas 2 !)
   - Khalid 0 courses (gardé en réserve)

---

## 📊 **POURQUOI CES PROBLÈMES ?**

### **Cause 1 : Pooling trop agressif**

```python
# Actuellement
max_detour_min: 15  # Accepte 15 min de détour
→ Système regroupe Francois + Daniel
→ Détour de 50 min (> 15, mais mal calculé ?)
```

### **Cause 2 : Équilibrage insuffisant**

```python
# Actuellement
driver_load_balance: 0.7  # 70% seulement
→ Giuseppe peut avoir 4 courses pendant que Yannis en a 2
```

### **Cause 3 : Limite trop haute**

```python
# Actuellement
max_bookings_per_driver: 6
→ Giuseppe peut prendre jusqu'à 6 courses
→ Système ne force pas la distribution
```

### **Cause 4 : Chauffeurs d'urgence utilisés par défaut**

```python
# Actuellement
allow_emergency: true  # Activé par défaut
→ Khalid utilisé comme chauffeur normal
```

---

## 🎯 **ACTIONS IMMÉDIATES**

### **Action 1 : Appliquer la config recommandée** ⏱️ 2 minutes

Via l'interface **"⚙️ Avancé"**, appliquez les paramètres ci-dessus.

### **Action 2 : Relancer le dispatch** ⏱️ 10 secondes

Cliquez **"🚀 Lancer Dispatch"** à nouveau.

### **Action 3 : Vérifier amélioration** ⏱️ 1 minute

Compter :

- Courses de Giuseppe : Devrait être ≤ 3
- Courses à 08:30 pour Dris : Devrait être 1 (pas 2)
- Courses de Khalid : Devrait être 0

---

## 📈 **RÉSULTAT ATTENDU**

### **Avant (actuel)** ❌

```
Giuseppe : 4 courses (surchargé)
Dris : 2 courses à 08:30 (impossible)
Khalid : 1 course (urgence utilisée)
Yannis : 2 courses (sous-utilisé)
```

### **Après (avec nouvelle config)** ✅

```
Giuseppe : 3 courses max (équilibré)
Dris : 1 course à 08:30 (possible)
Autre chauffeur : 1 course à 08:30 (Daniel séparé)
Khalid : 0 courses (réserve)
Yannis : 2-3 courses (mieux utilisé)
```

---

## 💡 **EXPLICATION TECHNIQUE**

### **Pourquoi le pooling créait le problème 08:30 ?**

```
OR-Tools voit :
  - Francois : Clinique Anières, 08:30
  - Daniel : Clinique Anières, 08:30

OR-Tools pense :
  "Même lieu, même heure → Je peux regrouper !"

Calcul du détour :
  - Clinique → Meyrin direct : 15 km, 20 min
  - Clinique → Carouge → Meyrin : 25 km, 35 min
  - Détour : 15 min (≤ max_detour_min = 15) ✅ Accepté

Problème :
  - Calcul ne prend pas en compte les 2 pickups/dropoffs
  - Temps réel : 50 min (pas 35)
  - Distance réelle : 33 km (pas 25)

Solution :
  - Désactiver pooling
  - OU réduire max_detour_min à 5 min
```

---

## 🎓 **BONNES PRATIQUES POUR VOS COURSES**

### **Type de courses : Transport médical**

**Caractéristiques** :

- Départs souvent depuis même clinique
- Horaires précis (rendez-vous médicaux)
- Passagers fragiles (pas de stress)
- Ponctualité critique

**Configuration optimale** :

```yaml
pooling: false # Pas de regroupement
max_bookings_per_driver: 3 # Limite stricte
min_transition_margin_min: 20 # Marge large (imprévus)
driver_load_balance: 0.85 # Équilibre strict
allow_emergency: false # Urgences en réserve
```

---

## 📝 **CHECKLIST CORRECTION**

- [ ] Ouvrir page Dispatch
- [ ] Cliquer **"⚙️ Avancé"**
- [ ] Décocher **"Activer le regroupement"**
- [ ] Mettre **"Courses max par chauffeur"** à **3**
- [ ] Mettre **"Équilibre charge"** à **0.85**
- [ ] Mettre **"Poids équité"** à **0.5**
- [ ] Cliquer **"✅ Appliquer ces paramètres"**
- [ ] Relancer dispatch
- [ ] Vérifier résultat

**Temps total : 3 minutes**

---

## 🚀 **PROCHAINES ÉTAPES**

1. **Testez la nouvelle config** (ci-dessus)
2. **Partagez le nouveau résultat** (liste des assignations)
3. **J'analyserai** si c'est mieux
4. **Si besoin**, on ajustera finement

---

**Voulez-vous que j'applique ces paramètres directement dans le code pour qu'ils soient permanents ?**
