# 🔍 Guide de Diagnostic : Monitoring Automatique des Retards

## 📋 Problème Identifié

Le monitoring automatique **ne transmettait pas les bonnes informations** et **n'était pas actif** en raison d'une erreur de **contexte Flask manquant** dans les threads d'arrière-plan.

### ❌ Erreur Originale

```
[RealtimeOptimizer] Failed to check assignments for company 1: Working outside of application context.
```

## ✅ Correction Appliquée

### 1. **Ajout du Contexte Flask au Thread de Monitoring**

**Fichier modifié** : `backend/services/unified_dispatch/realtime_optimizer.py`

#### Changements clés :

1. **Import de `current_app`** :

   ```python
   from flask import current_app
   ```

2. **Stockage de l'instance Flask dans `__init__`** :

   ```python
   def __init__(self, company_id: int, check_interval_seconds: int = 120, app=None):
       # ...
       self._app = app or current_app._get_current_object()  # Sauvegarder l'app Flask
   ```

3. **Utilisation du contexte dans `_monitoring_loop`** :

   ```python
   def _monitoring_loop(self) -> None:
       """Boucle principale de monitoring"""
       while self._running:
           try:
               # ⭐ IMPORTANT : Utiliser le contexte Flask dans le thread
               with self._app.app_context():
                   # Vérifier les assignations du jour
                   opportunities = self.check_current_assignments()
                   # ...
   ```

4. **Mise à jour des fonctions helper** :
   - `start_optimizer_for_company(company_id, check_interval, app=None)`
   - `check_opportunities_manual(company_id, for_date, app=None)`

## 🧪 Comment Tester

### Option 1 : Via le Frontend (Recommandé)

1. **Ouvrir la page Dispatch & Planification** :

   - URL : `http://localhost:3000/dashboard/company/{votre_company_id}/dispatch`

2. **Démarrer le monitoring** :

   - Cliquer sur le bouton **"Démarrer le Monitoring Automatique"** (ou équivalent)
   - Le bouton devrait afficher **"Monitoring Actif ✅"** après le démarrage

3. **Vérifier les statistiques** :
   - Les statistiques de retards doivent s'afficher
   - Les alertes doivent apparaître si des retards sont détectés
   - Le compteur de retards dans le header doit se mettre à jour

### Option 2 : Via l'API Directement

#### 1. **Démarrer le Monitoring**

```bash
curl -X POST http://localhost:5000/api/company_dispatch/optimizer/start \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"check_interval_seconds": 120}'
```

**Réponse attendue** :

```json
{
  "message": "Monitoring temps réel démarré",
  "status": {
    "running": true,
    "company_id": 1,
    "check_interval": 120,
    "last_check": null,
    "opportunities_count": 0
  }
}
```

#### 2. **Vérifier le Statut**

```bash
curl -X GET http://localhost:5000/api/company_dispatch/optimizer/status \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Réponse attendue** :

```json
{
  "running": true,
  "company_id": 1,
  "check_interval": 120,
  "last_check": "2025-10-10T17:30:00+02:00",
  "opportunities_count": 2
}
```

#### 3. **Récupérer les Retards en Temps Réel**

```bash
curl -X GET "http://localhost:5000/api/company_dispatch/delays/live?date=2025-10-10" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Réponse attendue** :

```json
{
  "delays": [
    {
      "booking_id": 24,
      "driver_id": 2,
      "current_delay": 15,
      "severity": "medium",
      "suggestions": [...]
    }
  ],
  "summary": {
    "total_delays": 1,
    "critical": 0,
    "high": 0,
    "medium": 1,
    "low": 0
  },
  "timestamp": "2025-10-10T17:30:00+02:00"
}
```

#### 4. **Récupérer les Opportunités d'Optimisation**

```bash
curl -X GET http://localhost:5000/api/company_dispatch/optimizer/opportunities \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### Option 3 : Script de Test Python

Un script de test complet a été créé : `backend/test_monitoring.py`

**Utilisation** :

```bash
cd backend
python test_monitoring.py
```

**Note** : Mettez à jour les credentials dans le script avant de l'exécuter.

## 🔧 Vérifications Docker

### 1. **Vérifier que le conteneur API est à jour**

```bash
docker-compose restart api
```

### 2. **Suivre les logs en temps réel**

```bash
docker logs -f atmr-api-1
```

**Logs à surveiller** :

- `[RealtimeOptimizer] Started PERSISTENT monitoring for company X`
- `[RealtimeOptimizer] Checking X assignments...`
- `[RealtimeOptimizer] Found X opportunities`

### 3. **Vérifier les erreurs**

```bash
docker logs atmr-api-1 2>&1 | grep -i "error\|exception\|traceback"
```

**❌ Ne devrait plus apparaître** :

- `Working outside of application context`

## 📊 Fonctionnement du Monitoring

### Cycle de Vérification

1. **Démarrage** : L'entreprise démarre le monitoring via le frontend ou l'API
2. **Thread Persistant** : Un thread non-daemon est créé avec le contexte Flask
3. **Vérification Périodique** (par défaut toutes les 2 minutes) :
   - Récupère toutes les assignations du jour
   - Calcule l'ETA en temps réel (position GPS → destination)
   - Détecte les retards (ETA > heure prévue + buffer)
   - Génère des suggestions intelligentes
   - Notifie le dispatcher via WebSocket si retards critiques
4. **Persistance** : Le thread continue même après la fin de la requête HTTP
5. **Arrêt** : Manuel via le frontend ou automatiquement lors du redémarrage du serveur

### Détection des Retards

Le système utilise **deux méthodes** pour détecter les retards :

#### 1. **Si GPS disponible** (Méthode Précise)

```python
current_time = maintenant
driver_pos = position GPS du chauffeur
pickup_pos = position du point de prise en charge
ETA = calculate_eta(driver_pos, pickup_pos)  # Via OSRM ou Haversine
arrival_time = current_time + ETA
delay = arrival_time - scheduled_time
```

#### 2. **Si GPS indisponible** (Méthode de Fallback)

```python
current_time = maintenant
scheduled_time = heure prévue de pickup
delay = current_time - scheduled_time + buffer(15min)
```

### Seuils de Sévérité

- **Critical** : ≥ 30 minutes de retard
- **High** : 15-29 minutes
- **Medium** : 5-14 minutes
- **Low** : < 5 minutes

## 🐛 Problèmes Courants

### 1. **Le monitoring ne démarre pas**

**Symptôme** : `"running": false` même après avoir cliqué sur "Démarrer"

**Solutions** :

- Vérifier les logs Docker : `docker logs atmr-api-1`
- Redémarrer l'API : `docker-compose restart api`
- Vérifier le token JWT dans le frontend

### 2. **Aucun retard détecté alors qu'il devrait y en avoir**

**Symptôme** : `"delays": []` mais vous savez qu'un chauffeur est en retard

**Solutions** :

- **Vérifier la position GPS** : Le chauffeur a-t-il partagé sa position récemment ?
- **Vérifier l'heure prévue** : L'assignation a-t-elle une `eta_pickup_at` ?
- **Vérifier le statut** : L'assignation doit être `assigned` ou `en_route`
- **Logs** : Regarder `docker logs atmr-api-1` pour voir les calculs de retard

### 3. **Erreur "Working outside of application context"**

**Symptôme** : Cette erreur apparaît dans les logs

**Solution** : **Ce problème est maintenant corrigé** par les modifications ci-dessus. Si l'erreur persiste :

- Redémarrer l'API : `docker-compose restart api`
- Vérifier que le code a bien été mis à jour dans le conteneur

### 4. **Le monitoring s'arrête après un moment**

**Symptôme** : `"running": true` devient `false` après quelques minutes

**Solutions** :

- **Vérifier les logs** pour voir si une exception a été levée
- **Thread daemon** : S'assurer que `daemon=False` dans `realtime_optimizer.py`
- **Redémarrer** : `docker-compose restart api`

## 📈 Optimisation & Configuration

### Paramètres Ajustables

Dans `backend/services/unified_dispatch/settings.py` :

```python
# Seuil de détection de retard
DELAY_THRESHOLD_MINUTES = 5  # Retard minimum pour alerte

# Intervalle de vérification
CHECK_INTERVAL_SECONDS = 120  # Toutes les 2 minutes

# Buffer de temps (marge d'erreur)
BUFFER_MINUTES = 15  # Ajouter 15 min au calcul

# Sévérités
CRITICAL_THRESHOLD = 30  # ≥ 30 min = critique
HIGH_THRESHOLD = 15      # 15-29 min = élevé
MEDIUM_THRESHOLD = 5     # 5-14 min = moyen
```

## 🎯 Prochaines Étapes

1. **✅ Correction du contexte Flask** → **FAIT**
2. **🔄 Redémarrage de l'API** → **FAIT**
3. **🧪 Test du monitoring** → **À FAIRE par l'utilisateur**
4. **📊 Vérification des retards détectés** → **À FAIRE**
5. **🔔 Test des notifications WebSocket** → **À FAIRE**

## 📞 Support

Si le problème persiste après avoir suivi ce guide :

1. **Récupérer les logs complets** :

   ```bash
   docker logs atmr-api-1 > logs_api.txt
   ```

2. **Vérifier la base de données** :

   ```bash
   docker exec -it atmr-postgres-1 psql -U atmr_user -d atmr_db -c "SELECT * FROM assignment WHERE date(created_at) = CURRENT_DATE;"
   ```

3. **Tester manuellement l'endpoint** :
   - Utiliser Postman ou curl pour tester `/api/company_dispatch/optimizer/start`
   - Vérifier la réponse et les headers

---

**Date de dernière mise à jour** : 10 octobre 2025  
**Version** : 2.0 (Correction du contexte Flask)
