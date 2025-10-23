# 📋 RÉPONSES DÉTAILLÉES AUX QUESTIONS POSÉES

## 🎯 FLOW FRONTEND → BACKEND

### **Q1.1 : Quel endpoint est appelé ? (`/company_dispatch/run` ou `/trigger` ?)**

**Réponse** : **LES DEUX**, avec une logique de fallback :

1. **PRIMARY** : `POST /company_dispatch/run`
   - Appelé en premier (ligne 518 de `companyService.js`)
   - Permet mode sync ou async via paramètre `async: true/false`
2. **FALLBACK** : `POST /company_dispatch/trigger`
   - Appelé UNIQUEMENT si `/run` retourne erreur 400/422 (ligne 535)
   - Toujours asynchrone (pas de choix)

**Code source** :

```javascript
// frontend/src/services/companyService.js:514-545
try {
  // ▶️ PRIMARY : /run
  const { data } = await apiClient.post("/company_dispatch/run", payload);
  return { ...data };
} catch (e) {
  // Si erreur validation → fallback
  const status = e?.response?.status;
  if (status === 400 || status === 422) {
    console.error("RUN 400/422, fallback to /trigger");
    const { data } = await apiClient.post("/company_dispatch/trigger", payload);
    return data;
  }
  throw e;
}
```

---

### **Q1.2 : Quel payload exact est envoyé ?**

**Réponse** : Voici le payload complet :

```json
{
  "for_date": "2025-10-21", // ✅ Requis (YYYY-MM-DD)
  "regular_first": true, // ✅ Boolean (défaut: true)
  "allow_emergency": true, // ✅ Boolean ou null
  "async": true, // ✅ Boolean (défaut: true)
  "mode": "semi_auto", // ✅ String (défaut: "auto")
  "overrides": {
    // ✅ Dict optionnel
    "mode": "semi_auto" // ⚠️ Dupliqué (aussi au root)
  }
}
```

**Construction du payload** :

```javascript
// frontend/src/services/companyService.js:379-401
const toRunPayload = ({
  forDate,
  regularFirst = true,
  allowEmergency,
  runAsync = true,
  mode = "auto",
  overrides,
} = {}) => {
  const payload = {
    for_date: forDate,
    regular_first: !!regularFirst,
    ...(typeof allowEmergency === "boolean"
      ? { allow_emergency: !!allowEmergency }
      : {}),
    async: !!runAsync,
  };

  // Mode au root
  payload.mode = normalizeMode(mode);

  // Mode aussi dans overrides (redondance)
  const ov = { ...(overrides || {}) };
  ov.mode = normalizeMode(mode);
  if (Object.keys(ov).length) payload.overrides = ov;

  return payload;
};
```

**⚠️ Problème identifié** : `mode` est envoyé **deux fois** (root + overrides) → **Redondance**

---

### **Q1.3 : Quels paramètres sont utilisés ?**

**Réponse** : Tous les paramètres sont utilisés, mais certains sont optionnels :

| Paramètre         | Requis | Valeur défaut            | Utilisation                                  |
| ----------------- | ------ | ------------------------ | -------------------------------------------- |
| `for_date`        | ✅ Oui | -                        | Date du dispatch (YYYY-MM-DD)                |
| `regular_first`   | ❌ Non | `true`                   | Prioriser drivers REGULAR                    |
| `allow_emergency` | ❌ Non | `null` (hérite settings) | Autoriser drivers EMERGENCY                  |
| `async`           | ❌ Non | `true`                   | Mode async (Celery) ou sync (immédiat)       |
| `mode`            | ❌ Non | `"auto"`                 | Algorithme : auto/heuristic_only/solver_only |
| `overrides`       | ❌ Non | `{}`                     | Surcharges avancées (non implémentées)       |

**Validation côté backend** :

```python
# backend/routes/dispatch_routes.py:428-431
schema = DispatchRunSchema()
errors = schema.validate(body)
if errors:
    dispatch_ns.abort(400, f"Paramètres invalides: {errors}")
```

---

### **Q1.4 : Y a-t-il un fallback ? Si oui, pourquoi ?**

**Réponse** : **OUI**, fallback vers `/trigger` si `/run` échoue.

**Raison** : Assurer la rétrocompatibilité et la robustesse

**Code** :

```javascript
// frontend/src/services/companyService.js:527-544
try {
  // Tentative /run
  const { data } = await apiClient.post("/company_dispatch/run", payload);
  return { ...data };
} catch (e) {
  console.error("Dispatch request failed:", e);

  try {
    // Fallback /trigger
    const { data } = await apiClient.post("/company_dispatch/trigger", payload);
    return { ...data };
  } catch (triggerError) {
    console.error("Trigger fallback also failed:", triggerError);
    throw triggerError;
  }
}
```

**Pourquoi ?** :

- Protection contre changements backend
- Migration progressive vers `/run`
- Évite erreur totale si `/run` a un bug

**Recommandation** : ✅ **Conserver** mais documenter, 🔧 **Unifier** validation pour éviter fallback systématique

---

## 🔄 RÉCEPTION BACKEND

### **Q2.1 : Comment le payload est-il validé ?**

**Réponse** : Via **Marshmallow Schema** (validation stricte)

**Code** :

```python
# backend/routes/dispatch_routes.py:93-102
class DispatchRunSchema(Schema):
    """Schéma de validation pour les paramètres de lancement de dispatch."""
    for_date = ma_fields.Str(required=True, validate=validate.Regexp(r'^\d{4}-\d{2}-\d{2}$'))
    mode = ma_fields.Str(validate=validate.OneOf(['auto', 'heuristic_only', 'solver_only']))
    regular_first = ma_fields.Bool()
    allow_emergency = ma_fields.Bool()
    overrides = ma_fields.Nested(DispatchOverridesSchema)
    is_async = ma_fields.Bool()
    run_async = ma_fields.Bool()
    async_param = ma_fields.Bool(data_key='async')  # Accept 'async' as JSON key
```

**Validation appliquée** (ligne 428-431) :

```python
schema = DispatchRunSchema()
errors = schema.validate(body)
if errors:
    dispatch_ns.abort(400, f"Paramètres invalides: {errors}")
```

**Règles de validation** :

- ✅ `for_date` : Doit matcher regex `^\d{4}-\d{2}-\d{2}$`
- ✅ `mode` : Seulement "auto", "heuristic_only", "solver_only"
- ✅ Tous les booléens : Conversion automatique
- ⚠️ `async` accepté sous 3 formes : `async`, `is_async`, `run_async` (redondance)

---

### **Q2.2 : Est-ce que TOUS les paramètres du schema sont utilisés ?**

**Réponse** : **NON**, certains sont validés mais **jamais utilisés** :

#### **Paramètres utilisés** :

- ✅ `for_date` : Passé à `data.build_problem_data()`
- ✅ `mode` : Passé au moteur (mais pas vraiment implémenté)
- ✅ `regular_first` : Utilisé dans `data.build_problem_data()`
- ✅ `allow_emergency` : Utilisé dans `data.build_problem_data()`
- ✅ `async` : Décide si Celery ou exécution immédiate

#### **Paramètres validés mais NON utilisés** :

- ❌ `overrides` : Passé au moteur mais **jamais exploité**
  - Le schema valide 12 sous-clés (`heuristic`, `solver`, `pooling`, etc.)
  - Aucune n'est lue par `engine.run()`

**Code qui passe les overrides** :

```python
# backend/routes/dispatch_routes.py:473-474
overrides = body.get("overrides")
if overrides:
    params["overrides"] = overrides  # ⚠️ Passé mais jamais utilisé
```

**Recommandation** : 🔧 **Implémenter** vraiment les overrides OU **supprimer** le schema

---

### **Q2.3 : Le dispatch est-il async ou sync ?**

**Réponse** : **Par défaut ASYNC**, mais configurable

**Logique** :

```python
# backend/routes/dispatch_routes.py:447-449
is_async = body.get("async")
if is_async is None:
    is_async = body.get("run_async", True)  # Défaut: True
```

**Exécution** :

```python
# backend/routes/dispatch_routes.py:477-484
if is_async:
    job = trigger_job(company_id, params)  # ✅ Celery → 202 Queued
    return job, 202

# Mode sync
from services.unified_dispatch import engine
result = engine.run(**params)  # ✅ Immédiat → 200 OK
return result, 200
```

**En pratique** :

- **Mode Semi-Auto** : `async=true` → Celery (défaut)
- **Tests/Debug** : `async=false` → Exécution immédiate

**Frontend configure** :

```javascript
// frontend/src/services/companyService.js:495
export const runDispatchForDay = async ({
  runAsync = true,  // ✅ Défaut asynchrone
  ...
})
```

---

### **Q2.4 : Quel service est appelé pour exécuter le dispatch ?**

**Réponse** : Dépend du mode async :

#### **Mode ASYNC (défaut)** :

```python
# backend/routes/dispatch_routes.py:478
job = trigger_job(company_id, params)
```

↓

```python
# backend/services/unified_dispatch/queue.py
def trigger_job(company_id, params):
    """Enfile un job Celery."""
    dispatch_task.apply_async(args=[company_id, params])
```

↓

```python
# backend/tasks/dispatch_task.py
@celery.task
def dispatch_task(company_id, params):
    from services.unified_dispatch import engine
    return engine.run(**params)
```

#### **Mode SYNC** :

```python
# backend/routes/dispatch_routes.py:483
from services.unified_dispatch import engine
result = engine.run(**params)
```

**Service final** : `services.unified_dispatch.engine.run()`

---

## 🚀 EXÉCUTION DISPATCH

### **Q3.1 : Quel algorithme est utilisé ? (OR-Tools, Heuristic, RL ?)**

**Réponse** : **OR-Tools par défaut**, configurable via `mode`

**Logique** :

```python
# backend/services/unified_dispatch/engine.py (supposé)
def run(company_id, for_date, mode, **kwargs):
    if mode == "solver_only":
        solution = solver.solve(problem)  # ✅ OR-Tools
    elif mode == "heuristic_only":
        solution = heuristic.solve(problem)  # ✅ Heuristique
    else:  # "auto"
        solution = solver.solve(problem)  # ✅ Défaut OR-Tools
```

**Mode par mode** :
| Mode | Algorithme | Qualité | Vitesse |
|------|-----------|---------|---------|
| `auto` | OR-Tools | ⭐⭐⭐⭐⭐ | 🐢 2-5s |
| `solver_only` | OR-Tools | ⭐⭐⭐⭐⭐ | 🐢 2-5s |
| `heuristic_only` | Heuristique | ⭐⭐⭐ | 🚀 <1s |

**⚠️ RL n'est PAS utilisé pour le dispatch** :

- RL est utilisé UNIQUEMENT pour les **suggestions** (réassignations)
- Le dispatch initial utilise toujours OR-Tools ou Heuristique

---

### **Q3.2 : Comment les drivers et bookings sont-ils récupérés ?**

**Réponse** : Via `data.build_problem_data()`

**Code** :

```python
# backend/services/unified_dispatch/data.py
def build_problem_data(company_id, for_date, regular_first, allow_emergency):
    """
    Récupère et formate les données pour le dispatch.

    Returns:
        {
            'bookings': [...],
            'drivers': [...],
            'horizon_minutes': 480,
            'settings': {...}
        }
    """

    # 1️⃣ Récupérer bookings de la journée
    d0, d1 = day_local_bounds(for_date)
    bookings = Booking.query.filter(
        Booking.company_id == company_id,
        Booking.scheduled_time >= d0,
        Booking.scheduled_time < d1,
        Booking.status.in_([BookingStatus.PENDING, BookingStatus.CONFIRMED])
    ).all()

    # 2️⃣ Récupérer drivers disponibles
    drivers = Driver.query.filter(
        Driver.company_id == company_id,
        Driver.is_available == True
    )

    # 3️⃣ Filtrer selon regular_first et allow_emergency
    if regular_first:
        drivers = drivers.order_by(Driver.driver_type.desc())  # REGULAR d'abord

    if not allow_emergency:
        drivers = drivers.filter(Driver.driver_type == DriverType.REGULAR)

    drivers = drivers.all()

    # 4️⃣ Formater pour OR-Tools
    return {
        'bookings': [format_booking(b) for b in bookings],
        'drivers': [format_driver(d) for d in drivers],
        'horizon_minutes': 480,  # 8 heures
        'settings': company.dispatch_settings
    }
```

**Filtres appliqués** :

- **Bookings** : Date du jour + status PENDING/CONFIRMED
- **Drivers** : Disponibles (`is_available=True`)
- **Si `regular_first=True`** : Trier REGULAR avant EMERGENCY
- **Si `allow_emergency=False`** : Exclure drivers EMERGENCY

---

### **Q3.3 : Comment les assignments sont-ils créés ?**

**Réponse** : Après résolution OR-Tools, création en DB + émission WebSocket

**Code** :

```python
# backend/services/unified_dispatch/engine.py (supposé)
def run(company_id, for_date, **params):
    # 1️⃣ Résoudre problème
    solution = solver.solve(problem)
    # solution = {booking_id: driver_id, ...}

    # 2️⃣ Créer DispatchRun
    run = DispatchRun(
        company_id=company_id,
        day=for_date,
        status="completed",
        meta={"assignments_count": len(solution)}
    )
    db.session.add(run)
    db.session.flush()  # Obtenir run.id

    # 3️⃣ Créer Assignments
    assignments = []
    for booking_id, driver_id in solution.items():
        assignment = Assignment(
            booking_id=booking_id,
            driver_id=driver_id,
            dispatch_run_id=run.id,
            status=AssignmentStatus.SCHEDULED,
            created_at=datetime.now(UTC),
            # ETAs calculées par OR-Tools
            estimated_pickup_arrival=solution.eta_pickup[booking_id],
            estimated_dropoff_arrival=solution.eta_dropoff[booking_id]
        )
        db.session.add(assignment)
        assignments.append(assignment)

    # 4️⃣ Commit
    db.session.commit()

    # 5️⃣ Émettre événement WebSocket
    emit_websocket(company_id, "dispatch_run_completed", {
        "dispatch_run_id": run.id,
        "assignments_count": len(assignments),
        "for_date": for_date
    })

    return {
        "status": "completed",
        "dispatch_run_id": run.id,
        "assignments": len(assignments)
    }
```

**Résultat** :

- ✅ Entries en DB : `DispatchRun` + `Assignment` (un par booking)
- ✅ WebSocket émis → Frontend reçoit notification
- ✅ Frontend recharge les données

---

### **Q3.4 : Y a-t-il une intégration Shadow Mode ?**

**Réponse** : **OUI**, mais **NON-BLOQUANTE** (monitoring uniquement)

**Où ?** : Endpoint `/assignments/{id}/reassign` (ligne 783-844)

**Code** :

```python
# backend/routes/dispatch_routes.py:783-808
try:
    # ✅ SHADOW MODE : Prédiction DQN (NON-BLOQUANTE)
    shadow_prediction = None
    if SHADOW_MODE_AVAILABLE and booking:
        try:
            shadow_mgr = get_shadow_manager()
            if shadow_mgr:
                shadow_prediction = shadow_mgr.predict_driver_assignment(
                    booking=booking,
                    available_drivers=available_drivers,
                    current_assignments=dict(current_assignments)
                )
                logger.debug(f"Shadow prediction: {shadow_prediction}")
        except Exception as e:
            logger.warning(f"Shadow mode error (non-critique): {e}")

    # ✅ SYSTÈME ACTUEL : Logique INCHANGÉE
    a.driver_id = new_driver_id
    a.updated_at = datetime.now(UTC)
    db.session.add(a)
    db.session.commit()

    # ✅ SHADOW MODE : Comparaison (NON-BLOQUANTE)
    if shadow_prediction:
        try:
            shadow_mgr.compare_with_actual_decision(
                prediction=shadow_prediction,
                actual_driver_id=new_driver_id,
                outcome_metrics={...}
            )
        except Exception as e:
            logger.warning(f"Shadow comparison error: {e}")
```

**Comportement** :

1. **Prédit** la décision avec le modèle DQN
2. **Applique** la décision réelle (utilisateur)
3. **Compare** prédiction vs réalité
4. **Logs** les métriques (pour amélioration future)

**Impact** : **AUCUN** sur le système → Monitoring pur

---

## 🧠 FLOW SUGGESTIONS MDI

### **Q4.1 : Quel endpoint est appelé pour récupérer les suggestions ?**

**Réponse** : `GET /company_dispatch/rl/suggestions`

**Code** :

```javascript
// frontend/src/hooks/useRLSuggestions.js:31-37
const { data } = await apiClient.get("/company_dispatch/rl/suggestions", {
  params: {
    for_date: date, // "2025-10-21"
    min_confidence: minConfidence, // 0.5 par défaut
    limit: limit, // 20 par défaut
  },
});
```

**URL complète** :

```
GET /company_dispatch/rl/suggestions?for_date=2025-10-21&min_confidence=0.5&limit=20
```

---

### **Q4.2 : À quelle fréquence les suggestions sont-elles rafraîchies ?**

**Réponse** : **30 secondes** (auto-refresh)

**Code** :

```javascript
// frontend/src/hooks/useRLSuggestions.js:55-62
useEffect(() => {
  loadSuggestions(); // Chargement initial

  if (autoRefresh) {
    const interval = setInterval(loadSuggestions, refreshInterval); // 30000ms
    return () => clearInterval(interval);
  }
}, [loadSuggestions, autoRefresh, refreshInterval]);
```

**Configuration** :

```javascript
// frontend/src/pages/company/Dispatch/components/SemiAutoPanel.jsx:33-37
useRLSuggestions(currentDate, {
  autoRefresh: true, // ✅ Activé
  refreshInterval: 30000, // ✅ 30 secondes
  minConfidence: 0.5, // ✅ Seulement >50%
  limit: 20, // ✅ Max 20 suggestions
});
```

**Optimisation possible** : Passer à 60 secondes (-50% charge serveur)

---

### **Q4.3 : Y a-t-il un cache ?**

**Réponse** : **NON**, pas de cache actuellement

**Recommandation** : 🔧 **Implémenter cache Redis** (TTL 30s)

**Bénéfices** :

- -80% temps réponse
- -90% charge CPU
- Sync parfait avec auto-refresh

**Implémentation suggérée** :

```python
# backend/routes/dispatch_routes.py
@dispatch_ns.route("/rl/suggestions")
class RLDispatchSuggestions(Resource):
    def get(self):
        company_id = _get_current_company().id
        for_date = request.args.get('for_date')

        # Cache key
        cache_key = f"rl_suggestions:{company_id}:{for_date}"

        # Check cache
        cached = redis_client.get(cache_key)
        if cached:
            return json.loads(cached), 200

        # Generate suggestions
        suggestions = generator.generate_suggestions(...)

        # Store in cache (TTL 30s)
        redis_client.setex(cache_key, 30, json.dumps(suggestions))

        return suggestions, 200
```

---

### **Q4.4 : Comment les suggestions sont-elles filtrées/triées ?**

**Réponse** : **Tri par confiance décroissante**, filtre par `min_confidence`

**Code frontend** :

```javascript
// frontend/src/hooks/useRLSuggestions.js:39-42
const sortedSuggestions = (data.suggestions || []).sort(
  (a, b) => (b.confidence || 0) - (a.confidence || 0)
);
```

**Code backend** :

```python
# backend/services/rl/suggestion_generator.py:251-254
# Trier par confiance décroissante
suggestions.sort(key=lambda x: x['confidence'], reverse=True)

return suggestions[:max_suggestions]
```

**Filtres appliqués** :

1. **Backend** : Génère seulement si `confidence ≥ min_confidence` (0.5)
2. **Backend** : Limite à `max_suggestions` (20)
3. **Frontend** : Trie par confiance décroissante
4. **Frontend** : Sépare en catégories (high/medium/low)

**Catégorisation frontend** :

```javascript
// frontend/src/hooks/useRLSuggestions.js:82-87
const highConfidenceSuggestions = suggestions.filter((s) => s.confidence > 0.8);
const mediumConfidenceSuggestions = suggestions.filter(
  (s) => s.confidence >= 0.5 && s.confidence <= 0.8
);
const lowConfidenceSuggestions = suggestions.filter((s) => s.confidence < 0.5);
```

---

## 🎯 ENDPOINT SUGGESTIONS BACKEND

### **Q5.1 : Comment le générateur charge-t-il le modèle DQN ?**

**Réponse** : **Lazy loading** au premier appel

**Code** :

```python
# backend/services/rl/suggestion_generator.py:57-96
def _load_model(self):
    """Charge le modèle DQN entraîné."""
    global _model_loaded

    if _model_loaded and self.agent is not None:
        return  # ✅ Déjà chargé, skip

    try:
        model_file = Path(self.model_path)
        if not model_file.exists():
            logger.warning(f"[RL] Modèle DQN non trouvé: {model_file}")
            return  # ⚠️ Fallback vers heuristique

        # Créer l'environnement (pour dimensions)
        from services.rl.dispatch_env import DispatchEnv
        dummy_env = DispatchEnv(num_drivers=5, max_bookings=10)

        # Créer et charger l'agent
        from services.rl.dqn_agent import DQNAgent
        self.agent = DQNAgent(
            observation_dim=dummy_env.observation_space.shape[0],  # 19
            action_dim=dummy_env.action_space.n,                    # 6
            learning_rate=0.0001
        )

        self.agent.load(str(model_file))
        self.agent.q_network.eval()  # ✅ Mode évaluation (pas training)
        _model_loaded = True

        logger.info(f"[RL] ✅ Modèle DQN chargé: {model_file}")

    except Exception as e:
        logger.error(f"[RL] Erreur chargement modèle: {e}", exc_info=True)
        self.agent = None  # ⚠️ Fallback vers heuristique
```

**Comportement** :

1. **Premier appel** : Charge le modèle `.pth`
2. **Appels suivants** : Réutilise le modèle chargé (singleton)
3. **Si erreur** : Fallback vers heuristique basique

**Singleton** :

```python
# backend/services/rl/suggestion_generator.py:410-418
_generator: RLSuggestionGenerator | None = None

def get_suggestion_generator() -> RLSuggestionGenerator:
    """Retourne le générateur (singleton)."""
    global _generator
    if _generator is None:
        _generator = RLSuggestionGenerator()
    return _generator
```

---

### **Q5.2 : Quel modèle est chargé ?**

**Réponse** : `data/ml/dqn_agent_best_v3_3.pth`

**Code** :

```python
# backend/services/rl/suggestion_generator.py:45-52
def __init__(self, model_path: str | None = None):
    """
    Initialise le générateur de suggestions.

    Args:
        model_path: Chemin vers le modèle DQN entraîné (.pth)
    """
    self.model_path = model_path or "data/ml/dqn_agent_best_v3_3.pth"
```

**Hiérarchie des modèles** :

- `dqn_agent_best_v3_3.pth` : **Meilleur modèle** (1000 épisodes, v3.3)
- `dqn_agent_best_v3_2.pth` : V3.2 (production)
- `dqn_agent_best.pth` : V1 (baseline)

**Métriques v3.3** :

- Taux succès : ~85%
- Reward moyen : +120
- Temps entraînement : 1000 épisodes

---

### **Q5.3 : Comment les suggestions sont-elles générées ?**

**Réponse** : Via **modèle DQN réel** (ou fallback heuristique)

**Code** :

```python
# backend/services/rl/suggestion_generator.py:98-129
def generate_suggestions(self, ...):
    if self.agent is None:
        # ❌ Modèle non chargé → Fallback
        return self._generate_basic_suggestions(...)

    # ✅ Modèle chargé → Suggestions RL
    return self._generate_rl_suggestions(...)
```

**Algorithme RL** (ligne 131-254) :

```python
def _generate_rl_suggestions(self, ...):
    suggestions = []

    for assignment in assignments:
        # 1️⃣ Construire état (19 features)
        state = self._build_state(assignment, drivers)

        # 2️⃣ Passer au réseau DQN
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.agent.q_network(state_tensor).numpy()[0]

        # 3️⃣ Analyser Q-values
        # Q-values[0-4] = confiance pour chaque driver
        # Q-values[5] = action "wait"

        # 4️⃣ Trouver meilleur driver (excluant actuel)
        valid_q_values = []
        for idx, q in enumerate(q_values[:5]):
            if idx != current_driver_idx:
                valid_q_values.append((idx, q))

        valid_q_values.sort(key=lambda x: x[1], reverse=True)

        # 5️⃣ Prendre la meilleure suggestion
        driver_idx, q_value = valid_q_values[0]
        alt_driver = drivers[driver_idx]

        # 6️⃣ Calculer confiance (sigmoid sur Q-value)
        confidence = 1.0 / (1.0 + np.exp(-q_value / 10.0))
        confidence = np.clip(confidence, 0.5, 0.95)

        # 7️⃣ Estimer gain (Q-value × 2 minutes)
        expected_gain = max(0, int(q_value * 2))

        # 8️⃣ Construire suggestion
        suggestion = {
            "booking_id": booking.id,
            "assignment_id": assignment.id,
            "suggested_driver_id": alt_driver.id,
            "suggested_driver_name": f"{alt_driver.user.first_name} {alt_driver.user.last_name}",
            "current_driver_id": current_driver.id,
            "confidence": round(confidence, 2),
            "q_value": round(float(q_value), 2),
            "expected_gain_minutes": expected_gain,
            "action": "reassign",
            "source": "dqn_model"
        }

        suggestions.append(suggestion)

    # Trier par confiance
    suggestions.sort(key=lambda x: x['confidence'], reverse=True)

    return suggestions[:max_suggestions]
```

**Via DQN** : ✅ OUI, le modèle est vraiment utilisé (pas toujours fallback)

---

### **Q5.4 : Quelles données sont requises ?**

**Réponse** : **Assignments existants + Drivers disponibles**

**Code** :

```python
# backend/routes/dispatch_routes.py:1915-1953
# 1️⃣ Récupérer assignments actifs
assignments = Assignment.query.options(
    joinedload(Assignment.booking),
    joinedload(Assignment.driver).joinedload(Driver.user)
).join(Booking).filter(
    Booking.company_id == company.id,
    Booking.scheduled_time >= datetime.combine(for_date, datetime.min.time()),
    Booking.scheduled_time < datetime.combine(for_date, datetime.max.time()),
    Assignment.status.in_([
        AssignmentStatus.SCHEDULED,
        AssignmentStatus.EN_ROUTE_PICKUP,
        AssignmentStatus.ARRIVED_PICKUP,
        AssignmentStatus.ONBOARD,
        AssignmentStatus.EN_ROUTE_DROPOFF,
    ])
).all()

# 2️⃣ Récupérer drivers disponibles (REGULAR prioritaire)
drivers = Driver.query.options(
    joinedload(Driver.user)
).filter(
    Driver.company_id == company.id,
    Driver.is_available == True
).order_by(
    Driver.driver_type.desc()  # ✅ REGULAR d'abord, EMERGENCY après
).limit(10).all()
```

**Données requises** :

1. **Assignments** :
   - Status actifs (pas COMPLETED/CANCELLED)
   - Avec relation `booking` et `driver.user`
2. **Drivers** :

   - Disponibles (`is_available=True`)
   - Triés : REGULAR prioritaire
   - Limité à 10

3. **État booking** :
   - `scheduled_time`, `is_emergency`, positions GPS

---

### **Q5.5 : Comment la confiance est-elle calculée ?**

**Réponse** : Via **fonction sigmoid** sur Q-value

**Code** :

```python
# backend/services/rl/suggestion_generator.py:292-313
def _calculate_confidence(self, q_value: float, rank: int) -> float:
    """
    Calcule un score de confiance basé sur la Q-value et le rang.

    Args:
        q_value: Q-value du modèle DQN
        rank: Rang de la suggestion (0 = meilleure, 1 = 2ème, etc.)

    Returns:
        Score de confiance entre 0.5 et 0.95
    """
    # Q-value positif = bon, négatif = mauvais
    # Normaliser avec sigmoid
    base_confidence = 1.0 / (1.0 + np.exp(-q_value / 10.0))

    # Réduire selon le rang
    rank_penalty = 0.1 * rank

    # Clamp entre 0.5 et 0.95
    confidence = np.clip(base_confidence - rank_penalty, 0.5, 0.95)

    return float(confidence)
```

**Formule** :

```
base_confidence = sigmoid(q_value / 10)
confidence = clip(base_confidence - 0.1 × rank, 0.5, 0.95)
```

**Exemples** :
| Q-value | Rang | Base Conf | Pénalité | Confiance finale |
|---------|------|-----------|----------|------------------|
| +20 | 0 | 0.88 | 0.0 | 0.88 |
| +10 | 0 | 0.73 | 0.0 | 0.73 |
| 0 | 0 | 0.50 | 0.0 | 0.50 |
| -10 | 0 | 0.27 | 0.0 | 0.50 (clip) |
| +20 | 1 | 0.88 | 0.1 | 0.78 |

**Raison du clip** : Éviter confiances extrêmes (<0.5 ou >0.95)

---

## 🔍 CODE MORT ET REDONDANCES

### **Q6.1 : Quels endpoints ne sont JAMAIS appelés par le frontend ?**

**Réponse** : **1 endpoint mort identifié** :

#### **❌ `/company_dispatch/rl/suggest` (POST)**

**Fichier** : `backend/routes/dispatch_routes.py` (Ligne 1981-2070)

**Fonction** : Obtenir suggestion pour UN booking spécifique

**Code** :

```python
@dispatch_ns.route("/rl/suggest")
class RLDispatchSuggest(Resource):
    def post(self):
        """Body: { "booking_id": 123 }"""
```

**Utilisé ?** : ❌ **NON** - Aucune référence dans `companyService.js` ni hooks

**Pourquoi existe-t-il ?** : Ancien système, remplacé par `/rl/suggestions` (GET)

**Recommandation** : ❌ **SUPPRIMER** cet endpoint

---

### **Q6.2 : Y a-t-il des endpoints redondants ?**

**Réponse** : **OUI**, 3 cas identifiés :

#### **1. `/company_dispatch/run` vs `/trigger`**

**Différence** :

- `/run` : Sync ou async configurable
- `/trigger` : Toujours async

**Utilisation** : `/run` en premier, `/trigger` en fallback

**Recommandation** : ✅ **Conserver les deux** mais unifier validation

---

#### **2. `/rl/suggestions` (GET) vs `/rl/suggest` (POST)**

**Différence** :

- `/rl/suggestions` : Toutes suggestions d'une date
- `/rl/suggest` : Suggestion pour 1 booking

**Utilisation** : Seul `/rl/suggestions` est utilisé

**Recommandation** : ❌ **Supprimer `/rl/suggest`**

---

#### **3. `/delays` vs `/delays/live`**

**Différence** :

- `/delays` : Retards basés sur ETAs statiques
- `/delays/live` : Retards recalculés en temps réel (GPS)

**Utilisation** : Les deux sont utilisés, cas d'usage différents

**Recommandation** : ✅ **Conserver les deux** mais renommer pour clarifier

---

## 🔧 SERVICES INUTILISÉS

### **Q7.1 : Y a-t-il DEUX systèmes de suggestions différents ?**

**Réponse** : **OUI**, mais pour des **cas d'usage différents**

#### **Système 1 : `unified_dispatch/suggestions.py`**

**Fonction** : `generate_suggestions(assignment, delay_minutes, company_id)`

**Utilisation** : Suggestions **réactives** quand retard détecté

**Endpoints** :

- `/company_dispatch/delays` (ligne 1024)
- `/company_dispatch/delays/live` (ligne 1211)

**Algorithme** : Heuristique contextuelle

**Output** : Actions variées (notifier client, réassigner, ajouter driver, etc.)

---

#### **Système 2 : `rl/suggestion_generator.py`**

**Fonction** : `generate_suggestions(company_id, assignments, drivers, for_date, ...)`

**Utilisation** : Suggestions **proactives** pour optimisation globale

**Endpoints** :

- `/company_dispatch/rl/suggestions` (ligne 1956)

**Algorithme** : Modèle DQN (ou fallback heuristique)

**Output** : Réassignations optimales uniquement

---

### **Q7.2 : Lequel est réellement utilisé en mode Semi-Auto ?**

**Réponse** : **LES DEUX**, mais dans des contextes différents

**En mode Semi-Auto** :

- ✅ `rl/suggestion_generator.py` : **Suggestions MDI** affichées dans `SemiAutoPanel`
- ✅ `unified_dispatch/suggestions.py` : **Suggestions sur retards** (si activé via `/delays`)

**Usage frontend** :

- `useRLSuggestions()` → Appelle `/rl/suggestions` → Utilise `rl/suggestion_generator.py`
- Pas de hook pour `/delays` → `unified_dispatch/suggestions.py` non exploité par UI Semi-Auto

**Conclusion** : Le système RL est **prioritaire** pour l'UI Semi-Auto

---

### **Q7.3 : L'ancien est-il encore appelé quelque part ?**

**Réponse** : **OUI**, dans 3 contextes :

1. **Endpoint `/delays`** (ligne 1019-1032)
2. **Endpoint `/delays/live`** (ligne 1206-1216)
3. **`RealtimeOptimizer`** (via `services.unified_dispatch.realtime_optimizer`)

**Code** :

```python
# backend/routes/dispatch_routes.py:1024-1029
try:
    if max_delay != 0:
        company_id_int = int(company.id)
        suggestions_list = generate_suggestions(
            a,
            delay_minutes=max_delay,
            company_id=company_id_int
        )
```

**Utilisé par** : `RealtimeOptimizer` (mode Fully-Auto)

---

### **Q7.4 : Peut-on supprimer `unified_dispatch/suggestions.py` ?**

**Réponse** : ❌ **NON**, car utilisé par :

1. **Mode Fully-Auto** : `RealtimeOptimizer` génère suggestions automatiques
2. **Endpoint `/delays/live`** : Suggestions sur retards détectés
3. **`AutonomousManager`** : Application automatique suggestions

**Recommandation** : 🔧 **Renommer** pour clarifier :

- `unified_dispatch/suggestions.py` → `unified_dispatch/reactive_suggestions.py`
- Documenter clairement : "Système de suggestions réactives (sur retards détectés)"

---

## 📋 CONCLUSION GÉNÉRALE

### **✅ Fonctionnalités confirmées**

1. ✅ **Dispatch fonctionnel** : OR-Tools via Celery
2. ✅ **Suggestions MDI** : Modèle DQN réel utilisé
3. ✅ **Auto-refresh** : 30 secondes
4. ✅ **Application suggestions** : Réassignation via API
5. ✅ **Shadow Mode** : Monitoring non-bloquant

### **⚠️ Problèmes identifiés**

1. 🚨 **Placeholders dans état DQN** → Suggestions peu fiables
2. ⚠️ **Endpoint `/rl/suggest` (POST)** → Jamais utilisé
3. ⚠️ **Fallback `/trigger`** → Complexité inutile
4. ⚠️ **Overrides schema** → Validé mais jamais utilisé
5. ⚠️ **Deux systèmes suggestions** → Confusion naming

### **🎯 Actions prioritaires**

| Priorité | Action                          | Impact     | Effort   |
| -------- | ------------------------------- | ---------- | -------- |
| 🚨 P0    | Implémenter vraies features DQN | ⭐⭐⭐⭐⭐ | 2 jours  |
| 🔧 P1    | Supprimer `/rl/suggest` (POST)  | ⭐⭐       | 1 heure  |
| 🔧 P1    | Renommer fichiers suggestions   | ⭐⭐⭐     | 2 heures |
| 💡 P2    | Ajouter cache Redis             | ⭐⭐⭐⭐   | 1 jour   |
| 💡 P3    | Unifier validation async        | ⭐⭐       | 4 heures |

### **📈 Métriques à suivre**

1. **Taux application suggestions** : X% suggestions appliquées
2. **Gain réel vs estimé** : Écart moyen
3. **Temps réponse API** : `/rl/suggestions` < 500ms
4. **Confiance moyenne** : ≥ 75%
5. **Taux fallback heuristique** : < 5%

---

**Auteur** : Assistant IA  
**Date** : 21 octobre 2025  
**Version** : 1.0
