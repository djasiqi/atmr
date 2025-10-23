# 🚀 PLAN DE DÉPLOIEMENT PRODUCTION - AGENT DQN

**Date :** 20 Octobre 2025  
**Objectif :** Intégrer l'agent DQN au système de dispatch réel  
**Durée estimée :** 2-3 heures

---

## 🎯 Objectifs

1. ✅ Créer module d'intégration RL
2. ✅ Intégrer dans autonomous_manager
3. ✅ Créer endpoints API
4. ✅ Implémenter A/B Testing
5. ✅ Monitoring production

---

## 📋 Plan d'Action Détaillé

### Étape 1 : Module d'Intégration RL

**Créer :** `backend/services/rl/rl_dispatch_manager.py`

**Fonctionnalités :**
- Charger agent DQN
- Convertir état réel → état RL
- Obtenir suggestion de dispatch
- Logger les décisions
- Gestion erreurs

**Interface :**
```python
class RLDispatchManager:
    def __init__(self, model_path: str)
    def get_dispatch_suggestion(booking, drivers) -> driver_id
    def convert_to_rl_state(booking, drivers) -> np.ndarray
    def convert_action_to_driver(action, drivers) -> Driver
```

### Étape 2 : Intégration Autonomous Manager

**Modifier :** `backend/services/unified_dispatch/autonomous_manager.py`

**Ajouts :**
- Initialiser RLDispatchManager
- Utiliser suggestions RL si mode activé
- Fallback sur heuristique si erreur
- Logger comparaisons

### Étape 3 : Endpoints API

**Créer/Modifier :** `backend/routes/dispatch_routes.py`

**Nouveaux endpoints :**
```
POST /dispatch/rl/suggest    - Obtenir suggestion RL
GET  /dispatch/rl/status     - Statut agent RL
POST /dispatch/rl/toggle     - Activer/désactiver RL
GET  /dispatch/rl/metrics    - Métriques performance
```

### Étape 4 : A/B Testing

**Créer :** `backend/services/ab_testing/dispatch_ab_test.py`

**Fonctionnalités :**
- Split traffic 50/50 (RL vs Heuristique)
- Tracking par groupe
- Comparaison métriques
- Décision automatique

### Étape 5 : Monitoring Production

**Créer :** `backend/services/rl/rl_monitoring.py`

**Métriques à tracker :**
- Reward réel vs prédit
- Latence inférence
- Taux d'utilisation RL
- Comparaison RL vs Heuristique
- Erreurs et fallbacks

---

## 🔧 Implémentation

### 1. RLDispatchManager

```python
class RLDispatchManager:
    """Gestionnaire de dispatch avec agent RL."""
    
    def __init__(self, model_path: str = "data/rl/models/dqn_best.pth"):
        self.agent = DQNAgent(state_dim=122, action_dim=201)
        self.agent.load(model_path)
        self.agent.q_network.eval()  # Mode évaluation
        
    def get_suggestion(self, booking, available_drivers):
        """Obtient suggestion de l'agent RL."""
        # Convertir état
        state = self._build_state(booking, available_drivers)
        
        # Obtenir action
        action = self.agent.select_action(state, training=False)
        
        # Convertir en driver
        if action < len(available_drivers):
            return available_drivers[action]
        return None  # Wait action
```

### 2. Configuration Company

**Ajouter dans `autonomous_config` :**
```json
{
  "rl_dispatch": {
    "enabled": false,
    "model_path": "data/rl/models/dqn_best.pth",
    "fallback_to_heuristic": true,
    "ab_test_ratio": 0.5
  }
}
```

### 3. Endpoints API

```python
@dispatch_ns.route('/rl/suggest')
class RLDispatchSuggestion(Resource):
    @jwt_required()
    def post(self):
        """Obtenir suggestion de dispatch RL."""
        data = request.get_json()
        booking_id = data.get('booking_id')
        
        # Charger booking et drivers
        booking = Booking.query.get(booking_id)
        drivers = Driver.query.filter_by(available=True).all()
        
        # Obtenir suggestion RL
        rl_manager = RLDispatchManager()
        suggested_driver = rl_manager.get_suggestion(booking, drivers)
        
        return {
            'suggested_driver_id': suggested_driver.id if suggested_driver else None,
            'confidence': 'high',
            'source': 'rl_agent'
        }
```

---

## 📊 A/B Testing

### Configuration

```python
# 50% des bookings utilisent RL
# 50% des bookings utilisent Heuristique

class ABTestManager:
    def should_use_rl(self, booking_id: int) -> bool:
        return booking_id % 2 == 0  # Simple split
```

### Métriques à Comparer

| Métrique | Groupe RL | Groupe Heuristique |
|----------|-----------|-------------------|
| Reward moyen | ? | ? |
| Distance moyenne | ? | ? |
| Late pickups | ? | ? |
| Temps de réponse | ? | ? |

---

## 🎯 Tests à Effectuer

1. **Test unitaire RLDispatchManager**
2. **Test intégration avec autonomous_manager**
3. **Test endpoints API**
4. **Test A/B split**
5. **Test monitoring**

---

## ⏱️ Timeline

| Étape | Durée | Statut |
|-------|-------|--------|
| Module RL Manager | 30 min | 🔄 En cours |
| Intégration autonomous | 30 min | ⏳ |
| Endpoints API | 30 min | ⏳ |
| A/B Testing | 30 min | ⏳ |
| Monitoring | 20 min | ⏳ |
| Tests | 20 min | ⏳ |
| **TOTAL** | **~3h** | |

---

**Démarrons l'implémentation ! 🚀**

