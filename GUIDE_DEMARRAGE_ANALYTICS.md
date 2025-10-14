# 🚀 Guide de Démarrage Rapide - Analytics

**Pour** : Démarrer avec le système Analytics  
**Temps** : 5 minutes

---

## ⚡ Démarrage Rapide

### 1. Migration déjà appliquée ✅

La migration est déjà effectuée ! Vos tables `dispatch_metrics` et `daily_stats` sont créées.

### 2. Tester que Tout Fonctionne

#### a) Lancer un Dispatch

Via votre interface actuelle :
1. Allez dans "Dispatch & Planification"
2. Cliquez "Lancer Dispatch"
3. Attendez la fin

**Résultat attendu** : Les métriques sont collectées automatiquement ! 🎉

#### b) Vérifier les Métriques (Optionnel)

```bash
docker compose exec db psql -U user -d atmr_db -c "SELECT COUNT(*) FROM dispatch_metrics;"
```

Si ça retourne un nombre > 0, c'est que ça fonctionne ! ✅

#### c) Tester l'API Analytics

**Option simple** : Dans votre navigateur, ouvrez DevTools (F12) et dans la Console :

```javascript
// Remplacez <company_public_id> par votre ID
fetch('/api/analytics/dashboard/<company_public_id>?period=7d', {
  headers: {
    'Authorization': 'Bearer ' + localStorage.getItem('token')
  }
})
.then(r => r.json())
.then(data => console.log(data));
```

Si vous voyez des données JSON, l'API fonctionne ! ✅

---

## 📊 Que Faire Maintenant ?

### Option 1 : Attendre les Données (Recommandé pour Débuter)

**Laissez le système collecter des données pendant 1 semaine.**

- Chaque dispatch collecte des métriques
- Après 7 jours, vous aurez assez de données pour voir des tendances
- Les insights deviendront plus pertinents

**Avantage** : Données riches pour le dashboard

---

### Option 2 : Créer le Dashboard Frontend Maintenant

**Si vous voulez voir visuellement les données dès maintenant.**

**Fichiers à créer** :
- `frontend/src/pages/company/Analytics/AnalyticsDashboard.jsx`
- `frontend/src/pages/company/Analytics/AnalyticsDashboard.module.css`

**Bibliothèque à installer** :
```bash
cd frontend
npm install recharts
```

**Code de base fourni** dans `PHASE_1_COMPLETION_SUMMARY.md`

---

### Option 3 : Activer les Rapports Automatiques

**Pour recevoir des emails quotidiens/hebdomadaires.**

1. **Configurer Celery Beat** (si pas déjà fait)
2. **Ajouter les tâches planifiées** dans `celery_app.py`
3. **Configurer l'envoi d'email** dans `notification_service.py`

**Code à ajouter** fourni dans `PHASE_1_COMPLETION_SUMMARY.md`

---

## 🎯 Commandes Utiles

### Vérifier les Métriques en Base

```sql
-- Dernières métriques collectées
SELECT 
  date,
  total_bookings,
  on_time_bookings,
  quality_score,
  average_delay_minutes
FROM dispatch_metrics
ORDER BY created_at DESC
LIMIT 5;
```

### Voir les Stats Agrégées

```sql
-- Stats des 7 derniers jours
SELECT 
  date,
  total_bookings,
  on_time_rate,
  avg_delay,
  quality_score
FROM daily_stats
ORDER BY date DESC
LIMIT 7;
```

### Analyser la Performance

```sql
-- Score moyen du mois
SELECT 
  AVG(quality_score) as avg_quality,
  AVG(average_delay_minutes) as avg_delay,
  SUM(total_bookings) as total_courses
FROM dispatch_metrics
WHERE date >= CURRENT_DATE - INTERVAL '30 days';
```

---

## 🐛 Dépannage

### Problème : Aucune Métrique Collectée

**Vérification** :
```bash
# Voir les logs du backend
docker compose logs api --tail=50 | grep "MetricsCollector"
```

**Solutions** :
1. Vérifier que le dispatch se termine correctement
2. Vérifier les logs d'erreur
3. Relancer un dispatch test

---

### Problème : API Retourne "No data"

**Normal si** :
- Vous n'avez pas encore lancé de dispatch depuis l'installation
- Les données sont sur une période différente

**Solution** :
- Lancer un dispatch
- Attendre 1 minute
- Re-tester l'API

---

## 📞 Support

### Logs à Consulter

```bash
# Logs backend
docker compose logs api --tail=100

# Logs Celery (si activé)
docker compose logs celery --tail=100

# Logs PostgreSQL
docker compose logs db --tail=50
```

### Commandes de Debug

```bash
# Vérifier l'état des services
docker compose ps

# Redémarrer le backend
docker compose restart api

# Vérifier les tables créées
docker compose exec db psql -U user -d atmr_db -c "\dt"
```

---

## ✅ Checklist de Validation

Cochez au fur et à mesure :

- [ ] J'ai lancé au moins 1 dispatch depuis l'installation
- [ ] J'ai vérifié que les métriques sont en DB
- [ ] J'ai testé l'API `/analytics/dashboard`
- [ ] J'ai consulté un fichier de documentation
- [ ] Je comprends comment fonctionne le système

**Si tous cochés** : Vous êtes prêt pour utiliser Analytics ! 🎉

---

## 🎊 Prêt à Utiliser !

Le système Analytics est maintenant **opérationnel** sur votre environnement Docker PostgreSQL.

**Chaque dispatch collecte automatiquement des métriques.**  
**Chaque jour, des stats sont agrégées.**  
**Vous avez accès aux données via API.**

**Félicitations ! 🚀**

---

**Date** : 14 octobre 2025  
**Statut** : ✅ Opérationnel  
**Prochaine étape** : Frontend ou laisser collecter des données

