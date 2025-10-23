# 🧪 TESTS À EFFECTUER - SYSTÈME MDI/DQN

**Date** : 21 Octobre 2025  
**Status** : ✅ **Prêt pour test**

---

## ⚠️ PROBLÈME RÉSOLU

**Avant** : `Error 111 connecting to localhost:6379. ECONNREFUSED`  
**Cause** : `backend/.env` pointait vers `localhost` au lieu du service Docker  
**Solution** : Modifié vers `redis://redis:6379/0` ✅  
**Résultat** : Celery connecté, tous services healthy ✅

---

## 🚀 TESTS À EFFECTUER

### **Test 1 : Lancer le Dispatch** (Priorité 1)

1. **Rafraîchir** la page frontend (F5)
2. **Aller en mode** : Semi-Auto
3. **Cliquer** : "🚀 Lancer Dispatch"
4. **Attendre** : 10-30 secondes

**Résultat attendu** :

```
✅ Dispatch lancé avec succès !
```

### **Test 2 : Vérifier les Assignments** (Priorité 2)

Après le dispatch, vérifier dans la base :

```bash
docker exec atmr-postgres-1 psql -U atmr -d atmr -c \
  "SELECT COUNT(*) as assignments FROM assignment WHERE created_at::date = CURRENT_DATE;"
```

**Résultat attendu** : Au moins 1 assignment créé

### **Test 3 : Voir les Suggestions MDI** (Priorité 3)

1. **Attendre** 30 secondes (auto-refresh)
2. **Observer** : Section "🤖 Suggestions IA (MDI)"
3. **Vérifier** : Les suggestions s'affichent

**Résultat attendu** :

```
┌─────────────────────────────────┐
│ 🤖 Suggestion MDI      [70% 🟡] │
│ Réassigner: Bob → Alice          │
│ Gain: +5 min                     │
│ [✅ Appliquer cette suggestion]  │
└─────────────────────────────────┘
```

### **Test 4 : Vérifier le Chargement du Modèle DQN** (Priorité 4)

Après que les suggestions apparaissent :

```bash
docker logs atmr-api-1 | grep -i "dqn\|modèle"
```

**Résultats possibles** :

**A. Modèle chargé** ✨ :

```
[RL] ✅ Modèle DQN chargé: data/ml/dqn_agent_best_v2.pth
```

→ Suggestions avec Q-values réelles !

**B. Modèle non trouvé** (fallback) :

```
[RL] Modèle DQN non trouvé: data/ml/dqn_agent_best_v2.pth. Les suggestions seront basiques.
```

→ Suggestions basiques à 70% de confiance

---

## 📊 COMPARAISON DES MODES

### **Mode Fallback** (Sans modèle DQN)

```json
{
  "confidence": 0.7,
  "q_value": null,
  "expected_gain_minutes": 5,
  "source": "basic_heuristic"
}
```

### **Mode DQN** (Avec modèle)

```json
{
  "confidence": 0.85,
  "q_value": 12.5,
  "expected_gain_minutes": 25,
  "source": "dqn_model"
}
```

---

## 🔍 DÉBOGAGE SI PROBLÈME

### **Aucune suggestion n'apparaît** ❌

**Vérifier assignments** :

```bash
docker exec atmr-postgres-1 psql -U atmr -d atmr -c \
  "SELECT id, driver_id, status, created_at FROM assignment WHERE created_at::date = CURRENT_DATE LIMIT 5;"
```

**Vérifier logs** :

```bash
docker logs atmr-api-1 --tail 50 | grep -i "suggestion\|rl"
```

### **Erreur dispatch** ❌

**Vérifier Celery** :

```bash
docker logs atmr-celery-worker-1 --tail 30
```

**Vérifier Redis** :

```bash
docker exec atmr-redis-1 redis-cli ping
# Devrait retourner: PONG
```

### **Modèle ne charge pas** ❌

**Vérifier le fichier** :

```bash
docker exec atmr-api-1 ls -lh /app/data/ml/dqn_agent_best_v2.pth
```

**Vérifier les imports** :

```bash
docker exec atmr-api-1 python -c "import torch; print(f'PyTorch {torch.__version__}')"
```

---

## ✅ CHECKLIST

Après les tests, vérifier :

- [ ] Dispatch lancé avec succès
- [ ] Assignments créés dans la DB
- [ ] Suggestions MDI visibles
- [ ] Modèle DQN chargé (ou fallback actif)
- [ ] Aucune erreur dans les logs
- [ ] Celery connecté à Redis

---

## 📞 COMMANDES RAPIDES

### **Status global**

```bash
docker ps --format "table {{.Names}}\t{{.Status}}"
```

### **Logs en temps réel**

```bash
# API
docker logs atmr-api-1 -f

# Celery Worker
docker logs atmr-celery-worker-1 -f

# Tous ensemble
docker compose logs -f
```

### **Test direct API**

```bash
curl "http://localhost:5000/api/company_dispatch/rl/suggestions?for_date=2025-10-21"
```

---

## 🎯 PROCHAINE ÉTAPE

**Après les tests** :

1. ✅ Si tout fonctionne → Documenter et passer à Shadow Mode
2. ❌ Si problème → Partager les logs pour debug

---

**Tous les services sont prêts ! Lancez les tests maintenant ! 🚀**
