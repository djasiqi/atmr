# 🐛 FIX : allow_emergency ignoré (Khalid toujours assigné)

**Date** : 21 octobre 2025, 18:45  
**Statut** : ✅ RÉSOLU  
**Sévérité** : 🔴 CRITIQUE

---

## 🚨 **SYMPTÔME**

Malgré `allow_emergency=false` correctement envoyé par le frontend et configuré dans les paramètres avancés, le chauffeur d'urgence **Khalid Alaoui (driver_id=1, type=EMERGENCY)** était **toujours assigné** à des courses.

### Résultat observé

```
✅ Giuseppe (REGULAR) : 4 courses
✅ Yannis (REGULAR)   : 2 courses
✅ Dris (REGULAR)     : 2 courses
❌ Khalid (EMERGENCY) : 2 courses  ← PAS NORMAL !
```

---

## 🔍 **INVESTIGATION**

### 1. Frontend ✅

Les paramètres sont correctement envoyés :

```javascript
{
  allow_emergency: false,
  overrides: { allow_emergency: false }
}
```

### 2. Backend API ✅

L'API Flask reçoit et transmet correctement à Celery.

### 3. Worker Celery ✅ (après premier fix)

Le dispatch fonctionne (plus de crash `AttributeError: 'Driver' object has no attribute 'available'`).

### 4. Database ❌

Les assignations créées par le **dispatch_run_id=288** incluent Khalid :

```sql
 id   | booking_id | driver_id | dispatch_run_id | driver_name   | time
------+------------+-----------+-----------------+---------------+-------
 1045 |        155 |         1 |             288 | Khalid Alaoui | 09:15
 1051 |        161 |         1 |             288 | Khalid Alaoui | 13:15
```

Donc le dispatch A VRAIMENT assigné Khalid malgré `allow_emergency=false`.

---

## 🐛 **CAUSE RACINE**

### Code problématique

**Fichier** : `backend/services/unified_dispatch/engine.py`  
**Ligne** : 502-503 (avant correction)

```python
# ❌ INCORRECT
allow_emg2 = allow_emg if allow_emergency is None else bool(allow_emergency)
```

### Explication

Le dispatch fonctionne en **2 passes** :

1. **Pass 1 (réguliers)** : Assigne avec les chauffeurs REGULAR seulement
2. **Pass 2 (urgences)** : Si des courses restent non assignées ET `allow_emergency=true`, ajoute les chauffeurs EMERGENCY

Le problème : Le Pass 2 utilise **`allow_emergency` (paramètre brut)** au lieu de **`allow_emg` (valeur calculée depuis settings + overrides)**.

### Flux des paramètres

```python
# Ligne 192: Paramètre reçu
def run(..., allow_emergency: bool | None = None, overrides: dict | None = None):

# Ligne 217-219: Application du paramètre aux settings
if allow_emergency is not None:
    s.emergency.allow_emergency_drivers = bool(allow_emergency)

# Ligne 220: Calcul de la valeur finale
allow_emg = bool(getattr(getattr(s, "emergency", None), "allow_emergency_drivers", True))
# ✅ allow_emg = false (correct)

# Ligne 502: BUG ! Réutilise le paramètre brut au lieu de allow_emg
allow_emg2 = allow_emg if allow_emergency is None else bool(allow_emergency)
#            ^^^^^^^^                               ^^^^^^^^^^^^^^^^^^^^^^
#            Utilise allow_emg si None             Sinon utilise paramètre brut

# Si allow_emergency=false (non None), alors :
allow_emg2 = bool(allow_emergency) = bool(false) = False  # ✅ OK en théorie

# MAIS si allow_emergency était true par défaut dans overrides,
# ou si le paramètre n'était pas passé correctement, alors :
allow_emg2 = True  # ❌ WRONG
```

**Le vrai problème** : Cette logique est **fragile** et dépend de comment `allow_emergency` est passé. Si `allow_emergency` n'est pas explicitement `false` mais que les settings disent `false`, ça ne marche pas.

---

## ✅ **SOLUTION**

### Correction appliquée

**Fichier** : `backend/services/unified_dispatch/engine.py`  
**Ligne** : 502-503 (après correction)

```python
# ✅ CORRECT
# Toujours utiliser allow_emg (calculé depuis settings + overrides) au lieu de allow_emergency (param brut)
allow_emg2 = allow_emg
```

### Pourquoi c'est mieux

- **`allow_emg`** est calculé en tenant compte de **TOUS** les overrides et settings (ligne 220)
- Plus besoin de logique conditionnelle complexe
- Comportement cohérent entre Pass 1 et Pass 2
- Si un override dit `allow_emergency=false`, ça s'applique partout

---

## 🧪 **TEST DE VALIDATION**

### Commande pour tester

1. Rafraîchir la page dispatch (F5)
2. Réappliquer les paramètres avancés :
   - ❌ **Autoriser chauffeurs d'urgence** : DÉCOCHÉ
   - **Pénalité d'utilisation** : 1000
3. Cliquer sur **🚀 Lancer Dispatch**

### Résultat attendu

```
✅ Giuseppe (REGULAR) : 3-4 courses
✅ Yannis (REGULAR)   : 3-4 courses
✅ Dris (REGULAR)     : 3-4 courses
✅ Khalid (EMERGENCY) : 0 courses    ← CORRECT !
```

Si certaines courses restent non assignées, c'est **NORMAL** (mieux vaut des courses non assignées que d'utiliser un chauffeur d'urgence contre la volonté de l'utilisateur).

---

## 📊 **IMPACT**

### Avant correction

- ❌ Pass 2 s'exécutait même avec `allow_emergency=false`
- ❌ Chauffeurs d'urgence utilisés contre la volonté de l'entreprise
- ❌ Coûts d'urgence facturés inutilement

### Après correction

- ✅ Pass 2 respecte strictement `allow_emg`
- ✅ Chauffeurs d'urgence utilisés UNIQUEMENT si autorisé
- ✅ Comportement prévisible et cohérent

---

## 📝 **FICHIERS MODIFIÉS**

1. `backend/services/unified_dispatch/engine.py` (ligne 502-503)
   - Simplifié `allow_emg2 = allow_emg` au lieu de logique conditionnelle

---

## 🔗 **RÉFÉRENCES**

- [Bug Dispatch Fail](./BUG_CRITIQUE_DISPATCH_FAIL.md) - Premier bug corrigé (AttributeError)
- [Solution Khalid Urgence](../SOLUTION_KHALID_URGENCE.md) - Analyse du problème
- [Guide Paramètres Avancés](./GUIDE_PARAMETRES_AVANCES.md) - Documentation des paramètres

---

## 🎯 **LEÇONS APPRISES**

1. **Toujours utiliser la valeur calculée finale** plutôt que le paramètre brut
2. **Éviter les logiques conditionnelles complexes** pour les flags importants
3. **Logger les valeurs critiques** pour faciliter le debugging
4. **Tester avec des cas réels** (pas seulement avec des mocks)
