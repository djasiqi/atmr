# 🔧 INTÉGRATION SHADOW MODE - GUIDE PRATIQUE

**Date :** 21 Octobre 2025  
**Type :** Guide d'intégration pratique  
**Durée estimée :** 2-3 heures

---

## 🎯 OBJECTIF

Intégrer le Shadow Mode DQN dans votre système de dispatch actuel en **3 étapes simples**.

---

## 📋 PRÉ-REQUIS

```yaml
Modèle DQN: ✅ dqn_best.pth (épisode 600, +810.5 reward)
  ✅ Testé et validé

Code: ✅ backend/services/rl/shadow_mode_manager.py
  ✅ backend/routes/shadow_mode_routes.py
  ✅ backend/scripts/rl/shadow_mode_analysis.py

Environnement: ✅ Docker/PostgreSQL opérationnels
  ✅ API backend fonctionnelle
  ✅ Accès admin configuré
```

---

## 🚀 ÉTAPE 1 : ENREGISTRER LES ROUTES (5 min)

### Fichier : `backend/routes_api.py`

**Ajouter :**

```python
# Imports existants...
from routes.shadow_mode_routes import shadow_mode_bp

def register_routes(app):
    """Enregistre toutes les routes de l'API."""
    # Routes existantes...
    app.register_blueprint(admin_bp)
    app.register_blueprint(analytics_bp)

    # ✅ NOUVEAU: Shadow Mode
    app.register_blueprint(shadow_mode_bp)

    print("✅ Routes Shadow Mode enregistrées")
```

**Vérifier :**

```bash
# Redémarrer l'API
docker-compose restart api

# Tester que les routes sont accessibles
curl http://localhost:5000/api/shadow-mode/status \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN"

# Réponse attendue: {"status": "active", "model_loaded": true, ...}
```

---

## 🔌 ÉTAPE 2 : INTÉGRER DANS DISPATCH (15 min)

### Fichier : `backend/routes/dispatch_routes.py`

**1. Importer le Shadow Manager (en haut du fichier) :**

```python
from services.rl.shadow_mode_manager import ShadowModeManager
import logging

logger = logging.getLogger(__name__)

# Instance globale du shadow manager
_shadow_manager = None

def get_shadow_manager():
    """Singleton pour le shadow manager."""
    global _shadow_manager
    if _shadow_manager is None:
        try:
            _shadow_manager = ShadowModeManager(
                model_path="data/rl/models/dqn_best.pth",
                log_dir="data/rl/shadow_mode",
                enable_logging=True
            )
            logger.info("✅ Shadow Mode Manager initialisé")
        except Exception as e:
            logger.error(f"❌ Erreur initialisation Shadow Mode: {e}")
            _shadow_manager = None
    return _shadow_manager
```

**2. Modifier la fonction d'assignation principale :**

Chercher votre fonction d'assignation (ex: `assign_booking`, `auto_assign`, etc.)

**AVANT (code existant) :**

```python
@dispatch_bp.route('/assign-booking/<int:booking_id>', methods=['POST'])
@jwt_required()
def assign_booking(booking_id):
    booking = Booking.query.get_or_404(booking_id)
    available_drivers = get_available_drivers(booking.company_id)

    # Logique d'assignation actuelle
    assigned_driver = your_current_assignment_logic(booking, available_drivers)

    # Sauvegarder
    booking.driver_id = assigned_driver.id
    db.session.commit()

    return jsonify({"success": True, "driver_id": assigned_driver.id})
```

**APRÈS (avec shadow mode) :**

```python
@dispatch_bp.route('/assign-booking/<int:booking_id>', methods=['POST'])
@jwt_required()
def assign_booking(booking_id):
    booking = Booking.query.get_or_404(booking_id)
    available_drivers = get_available_drivers(booking.company_id)

    # ✅ SHADOW MODE: Prédiction DQN (NON-BLOQUANTE)
    shadow_prediction = None
    try:
        shadow_mgr = get_shadow_manager()
        if shadow_mgr:
            shadow_prediction = shadow_mgr.predict_driver_assignment(
                booking=booking,
                available_drivers=available_drivers,
                current_assignments=get_current_assignments()  # À implémenter
            )
            logger.debug(f"Shadow prediction: {shadow_prediction}")
    except Exception as e:
        logger.warning(f"Shadow mode error (non-critique): {e}")

    # ✅ SYSTÈME ACTUEL: Logique INCHANGÉE
    assigned_driver = your_current_assignment_logic(booking, available_drivers)

    # Sauvegarder (COMME AVANT)
    booking.driver_id = assigned_driver.id
    db.session.commit()

    # ✅ SHADOW MODE: Comparaison (NON-BLOQUANTE)
    if shadow_prediction:
        try:
            shadow_mgr.compare_with_actual_decision(
                prediction=shadow_prediction,
                actual_driver_id=assigned_driver.id,
                outcome_metrics={
                    'distance_km': calculate_distance(booking, assigned_driver),
                    'estimated_pickup_minutes': estimate_time(booking, assigned_driver)
                }
            )
        except Exception as e:
            logger.warning(f"Shadow comparison error (non-critique): {e}")

    return jsonify({"success": True, "driver_id": assigned_driver.id})
```

**3. Implémenter les fonctions auxiliaires :**

```python
def get_current_assignments():
    """
    Retourne les assignations actuelles par driver.

    Returns:
        dict: {driver_id: [booking_id1, booking_id2, ...]}
    """
    # Exemple simple:
    from collections import defaultdict
    assignments = defaultdict(list)

    active_bookings = Booking.query.filter(
        Booking.status.in_(['pending', 'assigned', 'in_progress'])
    ).all()

    for booking in active_bookings:
        if booking.driver_id:
            assignments[booking.driver_id].append(booking.id)

    return dict(assignments)


def calculate_distance(booking, driver):
    """Calcule la distance entre booking et driver (en km)."""
    from shared.geo_utils import haversine_distance

    if not booking.pickup_lat or not driver.current_lat:
        return None

    return haversine_distance(
        booking.pickup_lat, booking.pickup_lon,
        driver.current_lat, driver.current_lon
    )


def estimate_time(booking, driver):
    """Estime le temps de pickup (en minutes)."""
    distance = calculate_distance(booking, driver)
    if distance is None:
        return None

    # Vitesse moyenne: 30 km/h en ville
    return (distance / 30.0) * 60.0
```

---

## 📊 ÉTAPE 3 : TESTER & MONITORER (10 min)

### Test 1 : Vérifier l'initialisation

```bash
# Redémarrer l'API
docker-compose restart api

# Vérifier les logs au démarrage
docker-compose logs api | grep "Shadow"

# Logs attendus:
# ✅ Shadow Mode Manager initialisé (model: data/rl/models/dqn_best.pth)
# ✅ Modèle DQN chargé depuis data/rl/models/dqn_best.pth
```

### Test 2 : Faire une assignation test

```bash
# Via l'API (remplacer <booking_id> par un vrai ID)
curl -X POST http://localhost:5000/api/dispatch/assign-booking/123 \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json"

# Vérifier qu'il n'y a PAS d'erreur
# L'assignation doit fonctionner normalement
```

### Test 3 : Vérifier les logs shadow

```bash
# Vérifier que les fichiers de log sont créés
ls -lh backend/data/rl/shadow_mode/

# Vous devriez voir:
# predictions_20251021.jsonl
# comparisons_20251021.jsonl

# Regarder les premières prédictions
cat backend/data/rl/shadow_mode/predictions_20251021.jsonl | head -1 | jq '.'

# Exemple de sortie:
# {
#   "booking_id": 123,
#   "predicted_driver_id": 45,
#   "action_type": "assign",
#   "confidence": 0.82,
#   "q_value": 674.3,
#   "timestamp": "2025-10-21T10:30:15.123456",
#   "available_drivers_count": 5
# }
```

### Test 4 : API de monitoring

```bash
# 1. Statut global
curl http://localhost:5000/api/shadow-mode/status \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN" \
  | jq '.'

# 2. Statistiques détaillées
curl http://localhost:5000/api/shadow-mode/stats \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN" \
  | jq '.session_stats'

# 3. Dernières prédictions
curl "http://localhost:5000/api/shadow-mode/predictions?limit=5" \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN" \
  | jq '.predictions[] | {booking_id, action_type, confidence}'
```

---

## 📈 MONITORING CONTINU

### Dashboard Admin (À Créer)

**Fichier : `frontend/src/pages/admin/ShadowModeDashboard.jsx`**

```jsx
import React, { useState, useEffect } from "react";
import { Card, CardHeader, CardContent } from "@/components/ui/card";

export default function ShadowModeDashboard() {
  const [stats, setStats] = useState(null);

  useEffect(() => {
    // Charger les stats toutes les 30 secondes
    const fetchStats = async () => {
      const response = await fetch("/api/shadow-mode/stats", {
        headers: { Authorization: `Bearer ${getToken()}` },
      });
      const data = await response.json();
      setStats(data.session_stats);
    };

    fetchStats();
    const interval = setInterval(fetchStats, 30000);
    return () => clearInterval(interval);
  }, []);

  if (!stats) return <div>Chargement...</div>;

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-4">🔍 Shadow Mode DQN</h1>

      <div className="grid grid-cols-3 gap-4">
        <Card>
          <CardHeader>Prédictions</CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{stats.predictions_count}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>Comparaisons</CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{stats.comparisons_count}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>Taux d'accord</CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">
              {(stats.agreement_rate * 100).toFixed(1)}%
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="mt-6">
        <h2 className="text-xl font-semibold mb-2">Détails</h2>
        <pre className="bg-gray-100 p-4 rounded">
          {JSON.stringify(stats, null, 2)}
        </pre>
      </div>
    </div>
  );
}
```

### Alertes (Optionnel)

**Créer : `backend/services/rl/shadow_alerts.py`**

```python
def check_alerts(shadow_manager):
    """Vérifie les métriques et envoie des alertes si nécessaire."""
    stats = shadow_manager.get_stats()

    # Alerte 1: Taux d'accord faible
    if stats['comparisons_count'] > 50 and stats['agreement_rate'] < 0.60:
        send_alert(
            level="warning",
            message=f"⚠️ Taux d'accord Shadow Mode faible: {stats['agreement_rate']:.1%}",
            details=stats
        )

    # Alerte 2: Modèle non chargé
    if not shadow_manager.agent:
        send_alert(
            level="error",
            message="❌ Modèle DQN non chargé dans Shadow Mode",
            details={"model_path": shadow_manager.model_path}
        )

def send_alert(level, message, details):
    """Envoie une alerte (email, Slack, etc.)."""
    logger.warning(f"ALERT [{level}]: {message}")
    # TODO: Intégrer avec votre système d'alertes
```

---

## ✅ CHECKLIST FINALE

### Immédiatement après intégration

- [ ] Routes shadow mode enregistrées
- [ ] Shadow manager initialisé au démarrage
- [ ] Code intégré dans fonction d'assignation
- [ ] Tests manuels passent (3-5 assignations)
- [ ] Logs créés dans `data/rl/shadow_mode/`
- [ ] API monitoring accessible

### Après 1 heure

- [ ] Au moins 10 prédictions enregistrées
- [ ] Aucune erreur critique dans les logs
- [ ] Taux d'accord calculé (>0%)
- [ ] Performance système normale

### Après 1 jour

- [ ] > 100 prédictions enregistrées
- [ ] Rapport quotidien généré
- [ ] Taux d'accord analysé
- [ ] Graphiques créés

### Après 1 semaine

- [ ] > 1000 prédictions au total
- [ ] Taux d'accord stable
- [ ] Analyse complète effectuée
- [ ] Décision GO/NO-GO Phase 2

---

## 🆘 DÉPANNAGE

### Problème : Modèle non chargé

**Symptôme :**

```
❌ Erreur lors du chargement du modèle DQN: ...
```

**Solution :**

```bash
# Vérifier que le modèle existe
ls -lh backend/data/rl/models/dqn_best.pth

# Si manquant, copier depuis le training
cp backend/data/rl/models/dqn_ep0600_r672.pth \
   backend/data/rl/models/dqn_best.pth

# Recharger via API
curl -X POST http://localhost:5000/api/shadow-mode/reload-model \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN"
```

### Problème : Aucune prédiction enregistrée

**Symptôme :**
Fichiers `predictions_*.jsonl` vides ou absents

**Solution :**

```bash
# 1. Vérifier que la fonction est appelée
docker-compose logs api | grep "Shadow prediction"

# 2. Vérifier les permissions
chmod 755 backend/data/rl/shadow_mode

# 3. Tester manuellement
docker-compose exec api python -c "
from services.rl.shadow_mode_manager import ShadowModeManager
mgr = ShadowModeManager()
print('Manager créé:', mgr)
print('Agent chargé:', mgr.agent is not None)
"
```

### Problème : Performance dégradée

**Symptôme :**
Assignations devenues lentes après intégration

**Solution :**

```python
# Ajouter du profiling
import time

start = time.time()
shadow_prediction = shadow_mgr.predict_driver_assignment(...)
duration = time.time() - start

logger.info(f"Shadow prediction took {duration*1000:.1f}ms")

# Si >100ms:
# 1. Désactiver logging verbeux
shadow_mgr.enable_logging = False

# 2. Optimiser construction état
# 3. Réduire complexité modèle
```

---

## 🎉 SUCCÈS !

Si vous avez suivi toutes les étapes :

```
╔═══════════════════════════════════════════════╗
║  ✅ SHADOW MODE INTÉGRÉ AVEC SUCCÈS!          ║
║                                               ║
║  → DQN prédit en parallèle                    ║
║  → Logging automatique actif                  ║
║  → Monitoring disponible                      ║
║  → Aucun impact utilisateurs                  ║
║                                               ║
║  🚀 Laisser tourner 1 semaine                 ║
╚═══════════════════════════════════════════════╝
```

**Prochaines étapes :**

1. ✅ Laisser tourner le shadow mode pendant 1 semaine
2. 📊 Monitoring quotidien (5 min/jour)
3. 📈 Analyse hebdomadaire (30 min vendredi)
4. 🚦 Décision GO/NO-GO Phase 2

---

_Guide d'intégration pratique créé le 21 octobre 2025_  
_Temps total estimé : 2-3 heures_  
_Support : shadow_mode_manager.py + documentation complète_ ✅
