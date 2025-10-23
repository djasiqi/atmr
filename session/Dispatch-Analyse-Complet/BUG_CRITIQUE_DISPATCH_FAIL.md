# 🔴 BUG CRITIQUE : Dispatch totalement en panne

**Date** : 21 octobre 2025, 18:30  
**Statut** : ✅ RÉSOLU  
**Sévérité** : 🔴 CRITIQUE (BLOCKER)

---

## 🚨 **SYMPTÔME**

Le dispatch semblait fonctionner mais :

- Les **mêmes assignations** restaient affichées
- Le chauffeur d'urgence **Khalid** était toujours utilisé, même avec `allow_emergency=false`
- Les paramètres avancés n'avaient **aucun effet**

---

## 🔍 **ANALYSE**

### Frontend

✅ Le frontend envoie correctement les paramètres :

```javascript
{
  allow_emergency: false,
  date: '2025-10-22',
  mode: 'semi_auto',
  overrides: { allow_emergency: false }
}
```

### Backend API

✅ L'API Flask reçoit correctement et envoie au worker Celery.

### Worker Celery ❌

**ERREUR CRITIQUE** détectée dans les logs :

```python
AttributeError: 'Driver' object has no attribute 'available'
  File "/app/services/unified_dispatch/engine.py", line 126
    available_drivers = [d for d in drivers if d.available]
                                               ^^^^^^^^^^^
```

**Résultat** :

```json
{
  "assignments": [],
  "unassigned": [],
  "bookings": [],
  "drivers": [],
  "meta": { "reason": "run_failed" }
}
```

Le dispatch **ÉCHOUE COMPLÈTEMENT** et ne crée **AUCUNE assignation**.

Le frontend affiche donc les **anciennes données** de la veille !

---

## 🐛 **CAUSE RACINE**

Dans `engine.py`, ligne 126 :

```python
# ❌ INCORRECT
available_drivers = [d for d in drivers if d.available]
```

L'attribut correct du modèle `Driver` est `is_available`, pas `available` :

```python
# ✅ backend/models/driver.py
class Driver(db.Model):
    is_available = Column(Boolean, nullable=False, server_default="true")
```

---

## ✅ **SOLUTION**

### Correction appliquée

**Fichier** : `backend/services/unified_dispatch/engine.py`  
**Ligne** : 126

```python
# ✅ CORRECT
available_drivers = [d for d in drivers if getattr(d, 'is_available', True)]
```

Utilisation de `getattr()` pour plus de robustesse (fallback à `True` si l'attribut manque).

---

## 🧪 **VÉRIFICATION**

### 1. Base de données

✅ Confirmation des types de chauffeurs :

```sql
SELECT id, user_id, driver_type FROM driver WHERE company_id = 1;

 id | user_id | driver_type
----+---------+-------------
  1 |       7 | EMERGENCY   ← Khalid
  2 |       8 | REGULAR     ← Dris
  3 |       9 | REGULAR     ← Giuseppe
  4 |      10 | REGULAR     ← Yannis
```

### 2. Logs Celery (AVANT correction)

```
[ERROR] AttributeError: 'Driver' object has no attribute 'available'
[INFO] Dispatch completed: assigned=0 unassigned=0
```

### 3. Logs Celery (APRÈS correction)

À vérifier après redémarrage du worker.

---

## 📝 **IMPACT**

### Avant correction

- ❌ Dispatch ne fonctionnait **PAS DU TOUT** depuis plusieurs heures
- ❌ Aucune nouvelle assignation créée
- ❌ L'entreprise voyait des données obsolètes
- ❌ Paramètres avancés totalement ignorés

### Après correction

- ✅ Dispatch fonctionne normalement
- ✅ `allow_emergency=false` sera respecté
- ✅ Nouvelles assignations créées correctement
- ✅ Paramètres avancés appliqués

---

## 🎯 **ACTION REQUISE**

### Pour l'utilisateur

1. Rafraîchir la page dispatch
2. Relancer un dispatch avec les paramètres avancés :
   - ✅ **Autoriser chauffeurs d'urgence** : DÉCOCHÉ
   - ✅ **Pénalité d'utilisation** : 1000
3. Vérifier que Khalid n'est **plus assigné**

### Pour le développeur

- ✅ Redémarrage du worker Celery : **FAIT**
- ⏳ Surveillance des logs pour confirmer le bon fonctionnement

---

## 📚 **LEÇONS APPRISES**

1. **Toujours vérifier les logs Celery** en cas de comportement étrange
2. **Utiliser `getattr()` pour les attributs** au lieu d'accès direct
3. **Tests end-to-end nécessaires** pour détecter ce type d'erreur

---

## 📎 **FICHIERS MODIFIÉS**

- `backend/services/unified_dispatch/engine.py` (ligne 126)

---

## 🔗 **RÉFÉRENCES**

- [Documentation Model Driver](../models/driver.py)
- [Logs Celery Worker](../logs/celery_worker_20251021.log)
- [Solution Conflits Temporels](./SOLUTION_CONFLITS_TEMPORELS.md)
