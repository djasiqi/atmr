# 🚨 Correction heure retour d'urgence sur mobile

**Date**: 2026-01-13  
**Version**: 1.0  
**Fichier concerné**: `backend/routes/company_mobile_dispatch.py`

---

## 🎯 Problème

Quand l'entreprise déclenche un retour d'urgence avec **+15 minutes** sur mobile, l'heure affichée était **00h15** au lieu de l'**heure actuelle + 15 minutes**.

### Exemple

**Scénario** :
- Heure actuelle : **15h31**
- Action : Retour d'urgence +15 min
- Heure attendue : **15h46** ✅
- Heure affichée : **00h15** ❌

---

## 🔍 Cause

**Fichier** : `backend/routes/company_mobile_dispatch.py` (ligne 2065-2068)

**Code problématique** :
```python
booking.is_urgent = True
if booking.scheduled_time:
    booking.scheduled_time = booking.scheduled_time + timedelta(
        minutes=extra_delay_minutes  # +15 minutes
    )
```

**Problème** :
- Si `booking.scheduled_time` = **00:00:00** (minuit par défaut)
- Le backend ajoute 15 minutes → **00:15:00** ❌
- L'application mobile affiche → **00h15** ❌

Pour un retour d'urgence, `booking.scheduled_time` peut être :
- `None` (pas encore planifié)
- **00:00:00** (valeur par défaut à minuit)
- Une heure dans le passé (si course déjà terminée)

Dans tous ces cas, le backend doit utiliser **l'heure actuelle** au lieu de `booking.scheduled_time`.

---

## ✅ Solution implémentée

### Nouvelle logique

```python
booking.is_urgent = True

# ✅ Calculer la nouvelle heure planifiée
from datetime import UTC, datetime
now = datetime.now(UTC)

# Si scheduled_time est None, à minuit (00:00), ou dans le passé,
# utiliser l'heure actuelle + délai
if not booking.scheduled_time:
    booking.scheduled_time = now + timedelta(minutes=extra_delay_minutes)
else:
    # Vérifier si l'heure est à minuit (00:00)
    is_midnight = (
        booking.scheduled_time.hour == 0 
        and booking.scheduled_time.minute == 0
    )
    # Vérifier si l'heure est dans le passé
    is_past = booking.scheduled_time < now
    
    if is_midnight or is_past:
        # Utiliser l'heure actuelle + délai
        booking.scheduled_time = now + timedelta(minutes=extra_delay_minutes)
    else:
        # Ajouter le délai à l'heure existante
        booking.scheduled_time = booking.scheduled_time + timedelta(
            minutes=extra_delay_minutes
        )
```

### Conditions gérées

| Situation | Ancienne logique | Nouvelle logique |
|-----------|------------------|------------------|
| `scheduled_time = None` | Pas de modification | `now + 15 min` ✅ |
| `scheduled_time = 00:00:00` | `00:15:00` ❌ | `now + 15 min` ✅ |
| `scheduled_time = 10h00` (passé) | `10h15` ❌ | `now + 15 min` ✅ |
| `scheduled_time = 16h00` (futur) | `16h15` ✅ | `16h15` ✅ |

---

## 📊 Exemples concrets

### Exemple 1 : Retour d'urgence sans heure planifiée

**État initial** :
- `booking.scheduled_time` = `None`
- Heure actuelle = 15h31

**Action** : Retour d'urgence +15 min

**Avant la correction** :
- `booking.scheduled_time` reste `None` ❌
- Affichage mobile : "⏱️ À définir" ❌

**Après la correction** :
- `booking.scheduled_time` = 15h46 ✅
- Affichage mobile : "15:46" ✅

---

### Exemple 2 : Retour d'urgence avec heure à minuit

**État initial** :
- `booking.scheduled_time` = `2026-01-13 00:00:00`
- Heure actuelle = 15h31

**Action** : Retour d'urgence +15 min

**Avant la correction** :
- `booking.scheduled_time` = `2026-01-13 00:15:00` ❌
- Affichage mobile : "00:15" ❌

**Après la correction** :
- `booking.scheduled_time` = `2026-01-13 15:46:00` ✅
- Affichage mobile : "15:46" ✅

---

### Exemple 3 : Retour d'urgence avec heure future valide

**État initial** :
- `booking.scheduled_time` = `2026-01-13 16:00:00`
- Heure actuelle = 15h31

**Action** : Retour d'urgence +15 min

**Avant la correction** :
- `booking.scheduled_time` = `2026-01-13 16:15:00` ✅

**Après la correction** :
- `booking.scheduled_time` = `2026-01-13 16:15:00` ✅
- Comportement identique (pas de régression)

---

### Exemple 4 : Retour d'urgence avec heure passée

**État initial** :
- `booking.scheduled_time` = `2026-01-13 10:00:00` (passé)
- Heure actuelle = 15h31

**Action** : Retour d'urgence +15 min

**Avant la correction** :
- `booking.scheduled_time` = `2026-01-13 10:15:00` ❌
- Affichage mobile : "10:15" (dans le passé) ❌

**Après la correction** :
- `booking.scheduled_time` = `2026-01-13 15:46:00` ✅
- Affichage mobile : "15:46" ✅

---

## 🔄 Flux complet

### Avant la correction

```
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│  Mobile Entreprise│─────▶│  Backend Flask   │─────▶│  Database        │
│  (15h31)         │      │                  │      │                  │
└──────────────────┘      └──────────────────┘      └──────────────────┘
        │                          │                          │
        │ POST /urgent             │                          │
        │ {extra_delay: 15}        │                          │
        ├─────────────────────────▶│                          │
        │                          │ scheduled_time = 00:00   │
        │                          │ + 15 min                 │
        │                          │ = 00:15 ❌               │
        │                          ├─────────────────────────▶│
        │                          │                          │
        │◀─────────────────────────┤ {scheduled_time: "00:15"}│
        │                          │                          │
        ▼                          │                          │
   Affiche "00h15" ❌             │                          │
```

### Après la correction

```
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│  Mobile Entreprise│─────▶│  Backend Flask   │─────▶│  Database        │
│  (15h31)         │      │                  │      │                  │
└──────────────────┘      └──────────────────┘      └──────────────────┘
        │                          │                          │
        │ POST /urgent             │                          │
        │ {extra_delay: 15}        │                          │
        ├─────────────────────────▶│                          │
        │                          │ scheduled_time = 00:00   │
        │                          │ → detect midnight        │
        │                          │ → use NOW + 15 min       │
        │                          │ = 15:46 ✅               │
        │                          ├─────────────────────────▶│
        │                          │                          │
        │◀─────────────────────────┤ {scheduled_time: "15:46"}│
        │                          │                          │
        ▼                          │                          │
   Affiche "15h46" ✅             │                          │
```

---

## 🧪 Tests de validation

### Test 1 : Retour d'urgence sans heure planifiée
```python
# État initial
booking.scheduled_time = None
now = datetime(2026, 1, 13, 15, 31, tzinfo=UTC)

# Appeler l'endpoint
POST /dispatch/v1/rides/123/urgent
{
  "extra_delay_minutes": 15
}

# Vérifier
assert booking.scheduled_time == datetime(2026, 1, 13, 15, 46, tzinfo=UTC)
```

### Test 2 : Retour d'urgence avec heure à minuit
```python
# État initial
booking.scheduled_time = datetime(2026, 1, 13, 0, 0, tzinfo=UTC)
now = datetime(2026, 1, 13, 15, 31, tzinfo=UTC)

# Appeler l'endpoint
POST /dispatch/v1/rides/123/urgent
{
  "extra_delay_minutes": 15
}

# Vérifier
assert booking.scheduled_time == datetime(2026, 1, 13, 15, 46, tzinfo=UTC)
```

### Test 3 : Retour d'urgence avec heure future valide
```python
# État initial
booking.scheduled_time = datetime(2026, 1, 13, 16, 0, tzinfo=UTC)
now = datetime(2026, 1, 13, 15, 31, tzinfo=UTC)

# Appeler l'endpoint
POST /dispatch/v1/rides/123/urgent
{
  "extra_delay_minutes": 15
}

# Vérifier (pas de changement de comportement)
assert booking.scheduled_time == datetime(2026, 1, 13, 16, 15, tzinfo=UTC)
```

### Test 4 : Retour d'urgence avec heure passée
```python
# État initial
booking.scheduled_time = datetime(2026, 1, 13, 10, 0, tzinfo=UTC)
now = datetime(2026, 1, 13, 15, 31, tzinfo=UTC)

# Appeler l'endpoint
POST /dispatch/v1/rides/123/urgent
{
  "extra_delay_minutes": 15
}

# Vérifier
assert booking.scheduled_time == datetime(2026, 1, 13, 15, 46, tzinfo=UTC)
```

---

## 🔧 Déploiement

### Redémarrer le backend

```bash
# En production
docker compose -f docker-compose.production.yml restart backend

# Vérifier les logs
docker compose -f docker-compose.production.yml logs -f backend
```

### Aucune modification mobile nécessaire

Le frontend mobile utilise déjà `dayjs(summary.time.pickup_at).format("DD MMM HH:mm")` qui affichera correctement l'heure retournée par le backend.

---

## 📌 Points importants

### ✅ Avantages de la solution

1. **Robuste** : Gère tous les cas (None, minuit, passé, futur)
2. **Rétrocompatible** : Pas de régression pour les heures futures valides
3. **Intuitif** : Pour un retour d'urgence, utilise toujours l'heure actuelle si nécessaire
4. **Timezone-aware** : Utilise UTC pour éviter les problèmes de fuseaux horaires

### ⚠️ Cas particuliers

1. **Heure déjà planifiée dans le futur** : Le comportement reste identique (ajout du délai)
2. **Heure à minuit** : Maintenant traitée comme "non planifiée"
3. **Heure dans le passé** : Maintenant réinitialisée à l'heure actuelle + délai

---

## 🔮 Évolutions possibles

### 1. Délai configurable par entreprise
```python
# Récupérer le délai par défaut depuis les settings de l'entreprise
company_settings = get_company_settings(company_id)
default_delay = company_settings.get("urgent_return_delay_minutes", 15)
```

### 2. Notification au chauffeur
```python
# Notifier le chauffeur de la mise à jour de l'heure
if booking.driver_id:
    notify_driver(
        booking.driver_id,
        f"Retour d'urgence planifié à {booking.scheduled_time.strftime('%H:%M')}"
    )
```

### 3. Historique des modifications
```python
# Logger l'ancienne et la nouvelle heure
logger.info(
    f"Urgent return: booking {booking_id} scheduled_time changed from "
    f"{old_time} to {booking.scheduled_time}"
)
```

---

**Version**: 1.0  
**Dernière mise à jour**: 2026-01-13  
**Auteur**: Assistant IA  
**Status**: ✅ Implémenté et prêt pour test
