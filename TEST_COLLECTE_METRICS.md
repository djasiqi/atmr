# 🧪 Test de Collecte des Métriques

**Objectif** : Vérifier que les métriques sont collectées après un dispatch

---

## ✅ Statut Actuel

- ✅ API Analytics fonctionne (200 OK)
- ✅ Frontend charge correctement
- ✅ Tables créées dans PostgreSQL
- ⚠️ **Aucune métrique collectée encore**

---

## 🔍 Pourquoi Pas de Données ?

### Raison Probable

Le dispatch que vous avez lancé était **avant** que j'ajoute le code de collecte dans `engine.py`.

**Solution** : Relancer un nouveau dispatch pour tester la collecte.

---

## 🧪 Test Complet

### Étape 1 : Lancer un Nouveau Dispatch

1. Allez dans **Dispatch & Planification**
2. Sélectionnez **aujourd'hui** (14 octobre 2025)
3. Cliquez **Lancer Dispatch**
4. Attendez la fin (barre de progression 100%)

### Étape 2 : Vérifier les Logs

Dans votre terminal PowerShell :

```powershell
docker compose logs api --tail=100 | Select-String -Pattern "MetricsCollector|Collected metrics"
```

**Résultat attendu** :

```
[MetricsCollector] Collected metrics for dispatch run 123: Quality=XX.X, On-time=XX/XX...
```

### Étape 3 : Vérifier la Base de Données

```powershell
docker compose exec -T api python -c "from models import DispatchMetrics; from ext import db; print('Métriques:', DispatchMetrics.query.count())"
```

**Résultat attendu** : `Métriques: 1` (ou plus)

### Étape 4 : Rafraîchir Analytics

1. Retournez sur la page **Analytics**
2. Rafraîchissez (F5)
3. Vous devriez maintenant voir les KPIs !

---

## 📊 Ce Que Vous Devriez Voir

### Après le dispatch

**KPIs** :

- Total Courses : 15
- Taux à l'heure : ~100% (si aucun retard)
- Retard moyen : 0-5 min
- Score Qualité : 70-90/100

**Graphiques** :

- 1 point sur chaque graphique (1 jour de données)

**Insights** :

- Peuvent apparaître selon les données

---

## 🐛 Si Toujours Pas de Données

### Vérification 1 : Code de Collecte Présent ?

```powershell
docker compose exec api grep -n "collect_dispatch_metrics" /app/services/unified_dispatch/engine.py
```

**Attendu** : Devrait retourner le numéro de ligne (~560)

### Vérification 2 : Import OK ?

```powershell
docker compose exec -T api python -c "from services.analytics.metrics_collector import collect_dispatch_metrics; print('Import OK')"
```

**Attendu** : `Import OK`

### Vérification 3 : Tables Existent ?

```powershell
docker compose exec -T api python -c "from models import DispatchMetrics, DailyStats; print('Models OK')"
```

**Attendu** : `Models OK`

---

## 💡 Actions à Effectuer

### Action Immédiate

**Relancez un dispatch MAINTENANT** pour tester :

1. Dispatch & Planification
2. Date : Aujourd'hui
3. Lancer Dispatch
4. Attendez 2 minutes
5. Retournez sur Analytics
6. Rafraîchissez (F5)

### Si Ça Ne Fonctionne Toujours Pas

Envoyez-moi le résultat de :

```powershell
docker compose logs api --tail=200 | Select-String -Pattern "Engine.*Dispatch|mark_completed"
```

Je pourrai voir si le dispatch se termine correctement.

---

**Prochaine étape** : Lancer 1 dispatch test et vérifier les résultats ! 🚀
