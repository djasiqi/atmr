# ⚠️ PROBLÈME IDENTIFIÉ : CONFLITS TEMPORELS NON DÉTECTÉS

## 🔴 **SYMPTÔME**

Le système a assigné **4 courses simultanées** (toutes à 07:00) au **même chauffeur (Yannis Labrot)** :

```
Client: Djelor Jasiqi | Date: 22.10.2025 • 07:00
├── Course 1: Avenue Ernest-Pictet 9 → Rte de Chevrens 145, Anières
├── Course 2: Avenue Ernest-Pictet 9 → Rue du Soleil-Levant, Genève
├── Course 3: Avenue Ernest-Pictet 9 → Av. J.-D.-Maillard 3, Meyrin
└── Course 4: Avenue Ernest-Pictet 9 → Rue Alcide-Jentzer 17, Genève

Chauffeur assigné: Yannis Labrot (pour les 4)
```

**C'est physiquement impossible** : un chauffeur ne peut pas être à 4 endroits différents simultanément !

---

## 🔍 **ANALYSE TECHNIQUE**

### **Cause probable #1 : Données identiques**

Ces 4 courses ont été créées avec :

- **Même client** : "Djelor Jasiqi"
- **Même pickup** : "Avenue Ernest-Pictet 9, 1203, Genève"
- **Même heure** : 07:00 exactement
- **Destinations différentes** : Anières, Genève (×2), Meyrin

Cela suggère un **client régulier** qui a besoin de transports vers différents lieux **chaque jour de la semaine**, mais qui a été mal encodé avec la même date/heure.

### **Cause probable #2 : Algorithme de dispatch**

L'algorithme OR-Tools utilise :

1. **Time Windows** (fenêtres temporelles) pour chaque course
2. **Service Times** (temps de service : pickup + dropoff)
3. **Transition Matrix** (temps de trajet entre points)

**Problème identifié** :

- Quand plusieurs courses ont exactement la **même heure** (07:00), OR-Tools les considère comme des "fenêtres compatibles"
- Le solveur essaie de les séquencer, mais si le **temps de service n'est pas assez contraignant**, il peut les assigner au même chauffeur
- **Manque de validation post-solver** : Aucun vérificateur ne détecte qu'un chauffeur ne peut pas faire 2 pickups simultanés

---

## 🛠️ **SOLUTIONS PROPOSÉES**

### **Solution Immédiate (Données)**

#### Correction manuelle des 4 courses :

Ces courses devraient probablement être sur **4 jours différents de la semaine** :

```yaml
Course 1: Lundi 20.10.2025 07:00 → Anières
Course 2: Mardi 21.10.2025 07:00 → Rue du Soleil-Levant
Course 3: Mercredi 22.10.2025 07:00 → Meyrin
Course 4: Jeudi 23.10.2025 07:00 → Rue Alcide-Jentzer
```

**OU** sur la même journée mais à **heures différentes** :

```yaml
Course 1: 22.10.2025 07:00 → Anières
Course 2: 22.10.2025 08:30 → Rue du Soleil-Levant
Course 3: 22.10.2025 10:00 → Meyrin
Course 4: 22.10.2025 11:30 → Rue Alcide-Jentzer
```

---

### **Solution Technique #1 : Ajouter validation post-dispatch** ✅ **RECOMMANDÉ**

Créer une fonction de validation qui détecte les conflits temporels **après** le dispatch :

```python
def validate_assignments_no_temporal_conflicts(
    assignments: List[Assignment],
    tolerance_minutes: int = 30  # Temps minimum entre deux courses
) -> Tuple[bool, List[str]]:
    """
    Vérifie qu'aucun chauffeur n'a deux courses qui se chevauchent temporellement.

    Returns:
        (is_valid, errors)
    """
    errors = []

    # Grouper assignments par driver_id
    by_driver = {}
    for assignment in assignments:
        driver_id = assignment.driver_id
        if driver_id not in by_driver:
            by_driver[driver_id] = []
        by_driver[driver_id].append(assignment)

    # Vérifier chaque chauffeur
    for driver_id, driver_assignments in by_driver.items():
        # Trier par scheduled_time
        driver_assignments.sort(
            key=lambda a: a.booking.scheduled_time
        )

        # Vérifier overlaps
        for i in range(len(driver_assignments) - 1):
            current = driver_assignments[i]
            next_assign = driver_assignments[i + 1]

            # Calculer fin estimée de la course actuelle
            current_end = (
                current.booking.scheduled_time +
                timedelta(minutes=estimated_trip_duration(current)) +
                timedelta(minutes=tolerance_minutes)  # Marge
            )

            next_start = next_assign.booking.scheduled_time

            # Conflit si next_start < current_end
            if next_start < current_end:
                time_gap = (next_start - current_end).total_seconds() / 60
                errors.append(
                    f"⚠️ Chauffeur {driver_id}: Conflit temporel "
                    f"entre courses {current.booking_id} (fin {current_end:%H:%M}) "
                    f"et {next_assign.booking_id} (début {next_start:%H:%M}) "
                    f"→ Écart: {time_gap:.0f} min"
                )

    return (len(errors) == 0, errors)
```

**Utilisation** :

```python
# Dans dispatch_routes.py après le dispatch
result = engine.run(...)
assignments = result.get("assignments", [])

# Validation
is_valid, errors = validate_assignments_no_temporal_conflicts(assignments)

if not is_valid:
    logger.error("[Dispatch] Conflits temporels détectés:")
    for error in errors:
        logger.error(f"  {error}")

    # Option 1: Rejeter le dispatch
    return {
        "status": "error",
        "message": "Conflits temporels détectés",
        "errors": errors
    }, 400

    # Option 2: Avertissement seulement
    result["warnings"] = errors
```

---

### **Solution Technique #2 : Améliorer contraintes OR-Tools** ⚙️

Renforcer les contraintes dans le solveur pour **interdire physiquement** les chevauchements :

```python
# Dans solver.py

# Ajouter contrainte : Pickup time + Service time + Travel time < Next pickup time
for vehicle in range(num_vehicles):
    # Pour chaque paire de nœuds consécutifs dans la route
    routing.AddDimension(
        time_callback_index,
        slack_max=0,  # ✅ Aucun slack = pas de chevauchement
        capacity=horizon,
        fix_start_cumul_to_zero=True,
        name='Time'
    )
```

**Avantage** : Le solveur lui-même **ne produira jamais** de solution invalide.  
**Inconvénient** : Plus complexe, nécessite tests approfondis.

---

### **Solution Technique #3 : Détection préventive en frontend** 🖥️

Ajouter validation côté frontend **avant** de soumettre plusieurs courses :

```javascript
// CompanyBooking.jsx

const validateBookings = (bookings) => {
  // Détecter courses avec même heure
  const byTime = {};

  bookings.forEach((booking) => {
    const timeKey = `${booking.customer_name}_${booking.scheduled_time}`;
    if (!byTime[timeKey]) {
      byTime[timeKey] = [];
    }
    byTime[timeKey].push(booking);
  });

  const conflicts = [];
  Object.entries(byTime).forEach(([key, duplicates]) => {
    if (duplicates.length > 1) {
      conflicts.push({
        customer: duplicates[0].customer_name,
        time: duplicates[0].scheduled_time,
        count: duplicates.length,
        bookings: duplicates,
      });
    }
  });

  if (conflicts.length > 0) {
    showWarning(
      `⚠️ Attention : ${conflicts.length} client(s) ont plusieurs courses à la même heure. ` +
        `Cela créera des conflits lors du dispatch !`
    );
  }
};
```

---

## 📋 **PLAN D'ACTION RECOMMANDÉ**

### **Phase 1 : Correction immédiate (aujourd'hui)**

1. ✅ **Corriger manuellement les 4 courses** de "Djelor Jasiqi"

   - Les répartir sur des jours différents OU des heures différentes
   - Script SQL fourni ci-dessous

2. ✅ **Ajouter validation post-dispatch** (Solution Technique #1)
   - Implémenter `validate_assignments_no_temporal_conflicts()`
   - L'appeler dans `dispatch_routes.py` après chaque dispatch
   - Retourner erreur si conflits détectés

### **Phase 2 : Prévention (cette semaine)**

3. ✅ **Ajouter alerte frontend** (Solution Technique #3)
   - Détecter doublons lors de création/import de courses
   - Afficher avertissement à l'utilisateur

### **Phase 3 : Amélioration structurelle (futur)**

4. ⚙️ **Renforcer contraintes OR-Tools** (Solution Technique #2)
   - Audit complet du solveur
   - Tests de régression avec cas pathologiques

---

## 📝 **SCRIPT SQL DE CORRECTION**

### **Option A : Répartir sur 4 jours différents** ⭐ **RECOMMANDÉ**

```sql
-- Identifier les 4 courses
SELECT id, customer_name, scheduled_time, dropoff_address
FROM bookings
WHERE customer_name = 'Djelor Jasiqi'
  AND scheduled_time::date = '2025-10-22'
ORDER BY id;

-- Supposons que les IDs sont: 101, 102, 103, 104

-- Course 1: Lundi 20.10.2025
UPDATE bookings
SET scheduled_time = '2025-10-20 07:00:00'
WHERE id = 101;  -- Anières

-- Course 2: Mardi 21.10.2025
UPDATE bookings
SET scheduled_time = '2025-10-21 07:00:00'
WHERE id = 102;  -- Rue du Soleil-Levant

-- Course 3: Mercredi 22.10.2025 (reste inchangée)
-- WHERE id = 103;  -- Meyrin

-- Course 4: Jeudi 23.10.2025
UPDATE bookings
SET scheduled_time = '2025-10-23 07:00:00'
WHERE id = 104;  -- Rue Alcide-Jentzer

-- Supprimer les assignations existantes
DELETE FROM assignments
WHERE booking_id IN (101, 102, 103, 104);
```

### **Option B : Espacer sur la même journée**

```sql
-- Course 1: 07:00 (inchangée)
-- Course 2: 08:30
UPDATE bookings
SET scheduled_time = '2025-10-22 08:30:00'
WHERE id = 102;

-- Course 3: 10:00
UPDATE bookings
SET scheduled_time = '2025-10-22 10:00:00'
WHERE id = 103;

-- Course 4: 11:30
UPDATE bookings
SET scheduled_time = '2025-10-22 11:30:00'
WHERE id = 104;

-- Supprimer les assignations existantes
DELETE FROM assignments
WHERE booking_id IN (101, 102, 103, 104);
```

---

## 🧪 **TESTS À EFFECTUER**

Après implémentation de la validation :

```python
def test_temporal_conflict_detection():
    """Test que la validation détecte les conflits."""

    # Créer 2 courses à 07:00 et 07:15
    booking1 = create_booking(scheduled_time="2025-10-22 07:00")
    booking2 = create_booking(scheduled_time="2025-10-22 07:15")

    # Assigner les deux au même chauffeur
    assignments = [
        Assignment(booking=booking1, driver_id=1),
        Assignment(booking=booking2, driver_id=1),
    ]

    # Validation doit détecter le conflit (15 min entre deux courses = impossible)
    is_valid, errors = validate_assignments_no_temporal_conflicts(
        assignments,
        tolerance_minutes=30
    )

    assert is_valid == False
    assert len(errors) == 1
    assert "Conflit temporel" in errors[0]
```

---

## 💡 **RECOMMANDATIONS FINALES**

1. **Court terme** : Implémenter la **Solution Technique #1** (validation post-dispatch)

   - Facile à implémenter (≈ 2 heures)
   - Impact immédiat
   - Empêche le problème de se reproduire

2. **Moyen terme** : Ajouter **Solution Technique #3** (alerte frontend)

   - Prévient les erreurs de saisie
   - UX améliorée

3. **Long terme** : Audit complet des contraintes OR-Tools

   - Garantir robustesse mathématique
   - Cas pathologiques (10+ courses simultanées)

4. **Formation** : Sensibiliser les utilisateurs
   - Bonnes pratiques de saisie
   - Différence entre "client régulier" et "courses simultanées"

---

## 🔗 **FICHIERS CONCERNÉS**

- `backend/services/unified_dispatch/data.py` (build_problem_data)
- `backend/services/unified_dispatch/solver.py` (contraintes OR-Tools)
- `backend/routes/dispatch_routes.py` (appel du dispatch)
- `backend/services/unified_dispatch/validation.py` (nouveau fichier à créer)
- `frontend/src/pages/company/Booking/CompanyBooking.jsx` (validation frontend)

---

**📌 Prochaine étape** : Voulez-vous que je :

1. Crée le script de correction SQL pour les 4 courses ?
2. Implémente la fonction de validation des conflits temporels ?
3. Les deux ?
