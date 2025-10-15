# 📐 Documentation des Algorithmes - `heuristics.py`

## 📋 Vue d'ensemble

Le fichier `heuristics.py` contient les algorithmes d'assignation de courses aux chauffeurs. C'est le cœur du système de dispatch automatique.

**Fichier :** `backend/services/unified_dispatch/heuristics.py`  
**Lignes :** ~1037 lignes  
**Rôle :** Assignation heuristique rapide (< 1s pour 50+ courses)

---

## 🎯 Objectifs des Algorithmes

### 1. Assignation Intelligente

- ✅ Assigner chaque course au meilleur chauffeur disponible
- ✅ Minimiser le temps d'attente client
- ✅ Minimiser les distances à vide (chauffeur → pickup)
- ✅ Respecter les contraintes métier (capacité, horaires, urgences)

### 2. Équité entre Chauffeurs

- ✅ Répartir équitablement les courses
- ✅ Éviter qu'un chauffeur soit surchargé
- ✅ Maximiser l'utilisation de tous les chauffeurs

### 3. Performance

- ✅ Traitement en < 1 seconde pour 50 courses
- ✅ Scalabilité jusqu'à 200+ courses/jour
- ✅ Résultats déterministes (même input → même output)

---

## 📊 Architecture Globale

```
┌─────────────────────────────────────────┐
│ assign(problem, settings)               │  ← Fonction principale
├─────────────────────────────────────────┤
│                                         │
│  1. Trier les courses                   │
│     ├─ Urgentes (retours < 30min)      │
│     └─ Régulières (FIFO temporel)      │
│                                         │
│  2. Pour chaque course                  │
│     ├─ Scorer tous les chauffeurs      │
│     ├─ Sélectionner le meilleur        │
│     └─ Assigner + mettre à jour état   │
│                                         │
│  3. Retourner résultat                  │
│     ├─ Assignations                    │
│     └─ Non-assignées                   │
│                                         │
└─────────────────────────────────────────┘
```

---

## 🔧 Fonction Principale : `assign()`

### Signature

```python
def assign(problem: Dict[str, Any], settings: Settings = DEFAULT_SETTINGS) -> HeuristicResult
```

### Paramètres

| Paramètre  | Type             | Description                                            |
| ---------- | ---------------- | ------------------------------------------------------ |
| `problem`  | `Dict[str, Any]` | Données du problème (courses, chauffeurs, contraintes) |
| `settings` | `Settings`       | Configuration de l'algorithme (poids, seuils, etc.)    |

**Structure de `problem` :**

```python
problem = {
    "bookings": [Booking],           # Liste des courses à assigner
    "drivers": [Driver],             # Liste des chauffeurs disponibles
    "driver_windows": [(int, int)],  # Fenêtres horaires (minutes)
    "fairness_counts": {id: int},    # Courses déjà assignées aujourd'hui
    "busy_until": {id: int},         # Timestamp jusqu'à quand occupé
    "driver_scheduled_times": {id: [int]},  # Horaires déjà assignés
    "proposed_load": {id: int},      # Courses proposées dans ce run
}
```

### Retour

```python
@dataclass
class HeuristicResult:
    assignments: List[HeuristicAssignment]  # Assignations proposées
    unassigned_booking_ids: List[int]       # Courses non assignées
    debug: Dict[str, Any]                   # Infos de débogage
```

### Algorithme Détaillé

```python
# ÉTAPE 1 : CLASSIFICATION
urgent = [b for b in bookings if _is_return_urgent(b, settings)]
regular = [b for b in bookings if not urgent]

# Trier
urgent.sort(key=lambda b: scheduled_time)    # Plus proches en premier
regular.sort(key=lambda b: scheduled_time)   # FIFO temporel

# ÉTAPE 2 : ASSIGNATION URGENTE
for booking in urgent:
    best_driver = None
    best_score = -1

    for driver in drivers:
        # Vérifier contraintes dures
        if not can_assign(driver, booking):
            continue

        # Calculer score
        score = _score_driver_for_booking(booking, driver, ...)

        if score > best_score:
            best_score = score
            best_driver = driver

    if best_driver:
        assign(booking, best_driver)
        update_state(best_driver, booking)

# ÉTAPE 3 : ASSIGNATION RÉGULIÈRE (même logique)
# ...

# ÉTAPE 4 : RETOUR
return HeuristicResult(assignments, unassigned, debug)
```

---

## 🎯 Fonction de Scoring : `_score_driver_for_booking()`

### Objectif

Calculer un **score de pertinence** pour assigner une course à un chauffeur.

**Score :** `0.0` (mauvais) à `1.0+` (excellent)

### Facteurs de Scoring

| Facteur             | Poids | Description                       | Range  |
| ------------------- | ----- | --------------------------------- | ------ |
| **Proximité**       | 0.40  | Distance chauffeur ↔ pickup       | 0-1    |
| **Équité**          | 0.30  | Équilibrage charge de travail     | 0-1    |
| **Priorité**        | 0.20  | Type de course (urgence, médical) | 0-1    |
| **Régularité**      | 0.10  | Client habituel du chauffeur      | 0-0.15 |
| **Pénalité retard** | -0.6  | Arriverait trop tard au pickup    | -0.6   |

### Formule

```
SCORE TOTAL = (proximité × 0.4)
            + ((1 - fairness_penalty) × 0.3)
            + (priorité × 0.2)
            + (régularité × 0.1)
            - pénalité_retard
```

### Exemple de Calcul

**Contexte :**

- Course urgente (retour médical)
- Chauffeur à 2 km du pickup (5 min)
- Chauffeur a déjà 2 courses aujourd'hui
- Pas de relation client-chauffeur

**Calcul :**

```python
# 1. Proximité (2 km = 5 min → bon score)
proximité = 1.0  # < 5 min = 1.0

# 2. Équité (2 courses → pénalité de 0.10)
fairness_penalty = 0.10
équité = 1.0 - 0.10 = 0.90

# 3. Priorité (retour médical)
priorité = 0.6  # Poids "medical"

# 4. Régularité
régularité = 0.0  # Pas de bonus

# 5. Pénalité retard
pénalité_retard = 0.0  # Peut arriver à temps

# TOTAL
score = (1.0 × 0.4) + (0.90 × 0.3) + (0.6 × 0.2) + (0.0 × 0.1) - 0.0
score = 0.40 + 0.27 + 0.12 + 0.00 - 0.00
score = 0.79  ← Bon score !
```

### Code

```python
def _score_driver_for_booking(
    b: Booking,
    d: Driver,
    driver_window: Tuple[int, int],
    settings: Settings,
    fairness_counts: Dict[int, int],
) -> Tuple[float, Dict[str, float], Tuple[int, int]]:
    # 1. Calculer proximité
    to_pickup_min = haversine_minutes(driver_pos, pickup_pos, avg_kmh=25)

    if to_pickup_min <= 5:
        prox_score = 1.0
    elif to_pickup_min >= 30:
        prox_score = 0.0
    else:
        prox_score = 1.0 - (to_pickup_min - 5) / 25.0

    # 2. Pénalité équité
    driver_load = fairness_counts.get(driver_id, 0)
    fairness_pen = min(0.4, 0.05 * driver_load)

    # 3. Priorité (urgence, médical, etc.)
    priority = _priority_weight(b, weights)

    # 4. Bonus régularité
    regular_bonus = _regular_driver_bonus(b, d)

    # 5. Pénalité retard
    if to_pickup_min > mins_to_booking + buffer:
        lateness_penalty = 0.6
    else:
        lateness_penalty = 0.0

    # Agrégation
    w = settings.heuristic
    total = (
        prox_score * w.proximity
        + (1.0 - fairness_pen) * w.driver_load_balance
        + priority * w.priority
        + regular_bonus * w.regular_driver_bonus
    ) - lateness_penalty

    return (total, breakdown, (est_start, est_finish))
```

---

## 🚨 Gestion des Urgences : `_is_return_urgent()`

### Définition

Une course est **urgente** si :

- C'est un **retour** (trajet de retour d'un aller-retour)
- ET scheduled_time est **dans moins de 30 min**

### Logique

```python
def _is_return_urgent(b: Booking, settings: Settings) -> bool:
    if not b.is_return:
        return False

    mins = minutes_from_now(b.scheduled_time)
    threshold = settings.emergency.return_urgent_threshold_min  # 30 min par défaut

    return mins <= threshold
```

### Impact

Les courses urgentes sont :

1. **Traitées en priorité** (avant toutes les autres)
2. **Triées par proximité temporelle** (plus proche = premier)
3. **Peuvent "casser" l'équité** (chauffeur le plus proche, même si chargé)

**Exemple :**

```
10:00 - Course régulière (pickup 11:00)
10:15 - Course URGENTE (retour dans 20 min)
10:20 - Course régulière (pickup 12:00)

→ Ordre de traitement : 10:15 (urgente), 10:00, 10:20
```

---

## ⚖️ Équité : `_driver_fairness_penalty()`

### Objectif

Éviter qu'un chauffeur soit surchargé pendant que d'autres attendent.

### Formule

```
pénalité = min(0.4, 0.05 × nb_courses_déjà_assignées)
```

### Exemples

| Courses déjà assignées | Pénalité | Impact sur score |
| ---------------------- | -------- | ---------------- |
| 0                      | 0.00     | Aucun            |
| 1                      | 0.05     | -1.5%            |
| 2                      | 0.10     | -3%              |
| 5                      | 0.25     | -7.5%            |
| 8+                     | 0.40     | -12% (max)       |

### Résultat

Un chauffeur avec **8 courses** aura un **malus de 12%**, favorisant les chauffeurs moins chargés.

---

## 🔒 Contraintes Dures (Must-Have)

### 1. Fenêtre Horaire Chauffeur

```python
def _check_driver_window_feasible(
    driver_window: Tuple[int, int],
    est_start_min: int,
    est_finish_min: int
) -> bool:
    start_w, end_w = driver_window  # Ex: (480, 1080) = 8h-18h

    # Si course commence après fin de journée → course pour demain, OK
    if est_start_min > end_w:
        return True

    # Sinon, vérifier que le début est dans la fenêtre
    return est_start_min >= start_w
```

### 2. Capacité Maximum

```python
max_cap = settings.solver.max_bookings_per_driver  # 10 par défaut

if proposed_load[did] + fairness_counts.get(did, 0) >= max_cap:
    continue  # Chauffeur plein → skip
```

### 3. Conflit Horaire

```python
min_gap_minutes = 30  # 30 min minimum entre 2 courses

for existing_time in driver_scheduled_times[did]:
    if abs(est_start - existing_time) < min_gap_minutes:
        conflict = True  # Conflit → skip
```

### 4. Disponibilité (busy_until)

```python
if est_start < busy_until[did]:
    continue  # Chauffeur occupé à ce moment → skip
```

---

## 🔄 Mise à Jour de l'État

Après chaque assignation, on met à jour 3 états :

### 1. `busy_until`

```python
busy_until[driver_id] = max(busy_until[driver_id], est_finish_min)
```

**Exemple :**

- Avant : `busy_until[1] = 480` (8h00)
- Course : 8h30 → 9h15 (570 min)
- Après : `busy_until[1] = 570` (9h15)

### 2. `driver_scheduled_times`

```python
driver_scheduled_times[driver_id].append(est_start_min)
```

**Exemple :**

- Avant : `[480, 540]` (8h00, 9h00)
- Course : 10h00 (600 min)
- Après : `[480, 540, 600]`

### 3. `proposed_load`

```python
proposed_load[driver_id] += 1
```

**Exemple :**

- Avant : `2` courses
- Après : `3` courses

---

## 📈 Optimisations & Astuces

### 1. Tri Intelligent

```python
# Urgentes : par proximité temporelle (plus proche = premier)
urgent.sort(key=lambda b: scheduled_time)

# Régulières : FIFO strict (ordre chronologique)
regular.sort(key=lambda b: scheduled_time)
```

**Pourquoi ?** Les urgences doivent être traitées immédiatement, les régulières dans l'ordre naturel.

### 2. Pénalité Progressive

```python
# Pénalité augmente progressivement avec la charge
if current_load <= 2:
    load_penalty = current_load * 0.10      # 0, 0.10, 0.20
elif current_load <= 4:
    load_penalty = 0.20 + (current_load - 2) * 0.20  # 0.40, 0.60
else:
    load_penalty = 0.60 + (current_load - 4) * 0.35  # 0.95, 1.30, ...
```

**Effet :** Favorise fortement les chauffeurs peu chargés.

### 3. Haversine Borné

```python
to_pickup_min = haversine_minutes(
    driver_pos, pickup_pos,
    avg_kmh=25,
    min_minutes=1,    # Plancher : 1 min minimum
    max_minutes=180   # Plafond : 3h maximum
)
```

**Pourquoi ?** Évite les valeurs extrêmes qui casseraient le scoring.

---

## 🧪 Exemple Complet : Scénario Réel

### Contexte

**Courses :**

1. Course A : Retour médical urgent (dans 15 min)
2. Course B : Course régulière (dans 2h)
3. Course C : Course régulière (dans 3h)

**Chauffeurs :**

1. Driver 1 : 5 km du pickup A, 2 courses déjà assignées
2. Driver 2 : 15 km du pickup A, 0 courses
3. Driver 3 : 2 km du pickup A, 5 courses déjà assignées

### Étape 1 : Tri

```python
urgent = [Course A]    # Retour < 30 min
regular = [Course B, Course C]  # Ordre chronologique
```

### Étape 2 : Scorer pour Course A (urgente)

| Driver | Proximité | Équité | Priorité | Total    | Sélectionné |
| ------ | --------- | ------ | -------- | -------- | ----------- |
| 1      | 0.60      | 0.90   | 0.6      | **0.75** | ❌          |
| 2      | 0.20      | 1.00   | 0.6      | 0.62     | ❌          |
| 3      | 1.00      | 0.75   | 0.6      | **0.87** | ✅          |

**Résultat :** Driver 3 sélectionné (meilleur score malgré 5 courses, car le plus proche)

### Étape 3 : Mettre à jour l'état

```python
busy_until[3] = 75  # Occupé jusqu'à 15 min (pickup) + 20 min (trajet) + 5 min (drop)
driver_scheduled_times[3].append(15)
proposed_load[3] = 6
```

### Étape 4 : Scorer pour Course B

Driver 3 maintenant **busy** → skip  
→ Driver 1 ou 2 sera choisi

---

## 🎓 Conseils d'Utilisation

### Quand Utiliser l'Heuristique ?

✅ **OUI :**

- < 50 courses à assigner
- Besoin de résultat rapide (< 1s)
- Contraintes simples

❌ **NON (utiliser solver OR-Tools) :**

- > 100 courses
- Contraintes complexes (pause, multi-dépôt)
- Besoin d'optimum garanti

### Tuning des Paramètres

```python
settings = Settings()

# Plus de poids sur proximité → moins de km à vide
settings.heuristic.proximity = 0.50  # au lieu de 0.40

# Plus de poids sur équité → meilleure répartition
settings.heuristic.driver_load_balance = 0.40  # au lieu de 0.30

# Augmenter le seuil d'urgence
settings.emergency.return_urgent_threshold_min = 45  # au lieu de 30
```

---

## 📊 Complexité Algorithmique

| Opération | Complexité   | Explication              |
| --------- | ------------ | ------------------------ |
| Tri       | O(n log n)   | n = nombre de courses    |
| Scoring   | O(n × m)     | n courses × m chauffeurs |
| **Total** | **O(n × m)** | Linéaire en pratique     |

**Exemple :**

- 50 courses × 10 chauffeurs = **500 comparaisons**
- Temps : **< 0.5 seconde**

---

## 🐛 Debugging

### Activer les Logs Détaillés

```python
import logging
logging.getLogger("heuristics").setLevel(logging.DEBUG)
```

### Analyser le Debug Dict

```python
result = assign(problem, settings)
print(result.debug)

# {
#   "urgent_count": 3,
#   "regular_count": 12,
#   "assignments": 14,
#   "unassigned": 1,
#   "breakdown": {
#     "proximity": 0.40,
#     "fairness": 0.27,
#     "priority": 0.12,
#     ...
#   }
# }
```

---

## 🔗 Fichiers Liés

| Fichier            | Rôle                               |
| ------------------ | ---------------------------------- |
| `engine.py`        | Orchestrateur (appelle `assign()`) |
| `data.py`          | Construit le `problem` dict        |
| `settings.py`      | Configuration des poids            |
| `problem_state.py` | Gestion de l'état des chauffeurs   |
| `solver.py`        | Alternative optimale (OR-Tools)    |

---

**Documentation complète et à jour au 15 octobre 2025** 🚀
