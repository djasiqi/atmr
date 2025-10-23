# 🔬 ANALYSE DISPATCH - PARTIE 2 : Qualité Code & ML

---

## 4. QUALITÉ DU CODE ET ARCHITECTURE

### 4.1 Structure Backend Flask

#### Points Forts ✅

1. **Séparation des responsabilités** :

   ```
   backend/
   ├── models/          # Data models (ORM)
   ├── routes/          # API endpoints
   ├── services/        # Business logic
   ├── tasks/           # Celery async jobs
   ├── sockets/         # WebSocket handlers
   └── shared/          # Utilities
   ```

2. **Services unified_dispatch bien organisés** :

   ```
   services/unified_dispatch/
   ├── engine.py           # Orchestration principale
   ├── data.py             # Problem construction
   ├── heuristics.py       # Greedy algorithms
   ├── solver.py           # OR-Tools wrapper
   ├── queue.py            # Celery queue manager
   ├── settings.py         # Configuration centralisée
   ├── autonomous_manager.py  # Mode fully-auto
   ├── realtime_optimizer.py  # Monitoring temps réel
   ├── suggestions.py      # Génération suggestions
   ├── ml_predictor.py     # ML (non utilisé)
   └── delay_predictor.py  # Prédiction retards basique
   ```

3. **Type hints** :

   ```python
   # ✅ Bon typage partout
   def build_problem_data(
       company_id: int,
       settings: Settings = DEFAULT_SETTINGS,
       for_date: str | None = None,
   ) -> Dict[str, Any]:
       ...
   ```

4. **Logging structuré** :
   ```python
   logger.info(
       "[Engine] Dispatch start company=%s mode=%s for_date=%s",
       company_id, mode, for_date,
       extra={"company_id": company_id, "mode": mode}
   )
   ```

#### Points Faibles ❌

1. **engine.py trop long** (951 lignes) :

   - Responsabilités multiples : orchestration, DB writes, events
   - Devrait être split en :
     - `DispatchOrchestrator` (workflow)
     - `ProblemBuilder` (data preparation)
     - `AssignmentApplier` (DB writes)
     - `EventEmitter` (notifications)

2. **Couplage fort Models ↔ Services** :

   ```python
   # ❌ engine.py importe directement les models
   from models import Assignment, Booking, Driver, Company
   # Devrait passer par une abstraction (Repository pattern)
   ```

3. **Transactions SQL non optimisées** :

   ```python
   # ❌ Commits multiples dans une boucle
   for assignment in assignments:
       a = Assignment(...)
       db.session.add(a)
       db.session.commit()  # N commits !
   ```

   **Solution** :

   ```python
   # ✅ Bulk insert
   db.session.bulk_insert_mappings(Assignment, [
       {"booking_id": a.booking_id, "driver_id": a.driver_id}
       for a in assignments
   ])
   db.session.commit()  # 1 seul commit
   ```

4. **Validations dispersées** :

   ```python
   # Validations dans models/dispatch.py
   @validates('driver_id')
   def _v_driver_id(self, _k, v):
       if v is None: return None
       if not isinstance(v, int) or v <= 0:
           raise ValueError("driver_id invalide")
       return v

   # Mais aussi dans routes/dispatch_routes.py
   if not for_date:
       dispatch_ns.abort(400, "for_date manquant")

   # Et encore dans services/unified_dispatch/data.py
   if not company_id or company_id <= 0:
       raise ValueError(f"Company {company_id} introuvable")
   ```

   **Solution** : Centraliser dans `validators/` avec schemas Marshmallow

5. **Pas de tests unitaires visibles** :
   - Dossier `backend/tests/` existe mais contenu inconnu
   - Pas de CI/CD apparent (GitHub Actions, pytest)

### 4.2 Structure Frontend React

#### Points Forts ✅

1. **Hooks personnalisés** :

   ```javascript
   // ✅ Bonne séparation logique métier / UI
   const { dispatches, loading, error, loadDispatches } = useDispatchData(date);
   const { delays, summary, loadDelays } = useLiveDelays(date);
   const { dispatchMode, loadDispatchMode } = useDispatchMode();
   ```

2. **Composants mode-specific** :

   ```
   Dispatch/
   ├── UnifiedDispatchRefactored.jsx  # Container principal
   ├── components/
   │   ├── DispatchHeader.jsx
   │   ├── ManualModePanel.jsx        # Mode manuel
   │   ├── SemiAutoPanel.jsx          # Mode semi-auto
   │   └── FullyAutoPanel.jsx         # Mode fully-auto
   └── modes/
       ├── Common.module.css
       ├── Manual.module.css
       ├── SemiAuto.module.css
       └── FullyAuto.module.css
   ```

3. **WebSocket temps réel** :
   ```javascript
   useEffect(() => {
     socket.on("dispatch_run_completed", handleDispatchComplete);
     socket.on("booking_updated", handleBookingUpdated);
     return () => {
       socket.off("dispatch_run_completed", handleDispatchComplete);
       socket.off("booking_updated", handleBookingUpdated);
     };
   }, [socket]);
   ```

#### Points Faibles ❌

1. **Pas de state management centralisé** :

   - Devrait utiliser Redux/Zustand pour partager state entre composants
   - Actuellement : chaque composant refetch les données

2. **Duplication de logique** :

   ```javascript
   // ❌ Même code dans plusieurs composants
   const onAssignDriver = async (reservationId, driverId) => {
     const success = await handleAssignDriver(reservationId, driverId);
     if (success) {
       setSelectedReservationForAssignment(null);
       loadDispatches();
     }
   };
   ```

   Devrait être dans un hook `useAssignmentActions`.

3. **Props drilling** :

   ```javascript
   // ❌ Props passées à travers 3-4 niveaux
   <UnifiedDispatch
       styles={styles}
       dispatches={dispatches}
       onAssign={onAssign}
   />
     ↓
   <SemiAutoPanel
       styles={styles}
       dispatches={dispatches}
       onAssign={onAssign}
   />
     ↓
   <DispatchTable
       styles={styles}
       dispatches={dispatches}
       onAssign={onAssign}
   />
   ```

4. **Pas de gestion d'erreurs réseau** :

   ```javascript
   // ❌ Si API down, pas de retry
   const loadDispatches = async () => {
     const data = await api.getDispatches();
     setDispatches(data);
   };
   ```

   **Solution** : Ajouter React Query pour retry + cache :

   ```javascript
   const { data, error, isLoading } = useQuery(
     ["dispatches", date],
     () => api.getDispatches(date),
     { retry: 3, staleTime: 30000 }
   );
   ```

---

## 5. INTÉGRATION ML/IA

### 5.1 État Actuel

#### Composants ML Existants

1. **`ml_predictor.py`** : ML Predictor (RandomForest)

   - ✅ Entraînement sur historique
   - ✅ Features extraction (9 features)
   - ✅ Prédiction retard
   - ❌ **PAS UTILISÉ** dans le pipeline de dispatch

2. **`delay_predictor.py`** : Delay Predictor (règles simples)
   - ✅ Calcul ETA basé sur GPS
   - ✅ Comparaison scheduled vs estimated
   - ❌ Pas de ML, juste arithmétique

#### Features Actuelles (ml_predictor.py)

```python
{
    "time_of_day": 18.0,           # Heure (0-23)
    "day_of_week": 4,              # Jour (0-6, 0=lundi)
    "distance_km": 12.5,           # Distance Haversine
    "is_medical": 1.0,             # Booking médical
    "is_urgent": 0.0,              # Booking urgent
    "driver_punctuality_score": 0.85,  # Score ponctualité chauffeur
    "booking_priority": 0.8,       # Priorité (calculée)
    "traffic_density": 0.8,        # Densité trafic (heuristique)
    "weather_factor": 0.5,         # Météo (placeholder)
}
```

### 5.2 Plan d'Intégration ML

#### Phase 1 : Proof of Concept (POC) - 2 semaines

**Objectif** : Prouver que le ML améliore les prédictions de retard.

**Étapes** :

1. **Collecte données historiques** :

   ```python
   # Créer script `scripts/collect_training_data.py`
   def collect_historical_data(start_date, end_date):
       """
       Extrait données des 90 derniers jours :
       - Assignments complétés
       - Retard réel (actual_pickup_time - scheduled_time)
       - Features au moment de l'assignation
       """
       assignments = Assignment.query.join(Booking).filter(
           Booking.status == BookingStatus.COMPLETED,
           Booking.completed_at >= start_date,
           Booking.completed_at <= end_date
       ).all()

       training_data = []
       for a in assignments:
           booking = a.booking
           driver = a.driver

           # Calculer retard réel
           actual_delay = (booking.actual_pickup_time - booking.scheduled_time).total_seconds() / 60

           # Extraire features
           features = ml_predictor.extract_features(booking, driver, booking.scheduled_time)

           training_data.append({
               "features": features,
               "actual_delay_minutes": actual_delay
           })

       return training_data
   ```

2. **Entraînement modèle** :

   ```python
   from services.unified_dispatch.ml_predictor import DelayMLPredictor

   predictor = DelayMLPredictor()
   training_data = collect_historical_data(start_date, end_date)

   metrics = predictor.train_on_historical_data(training_data, save_model=True)
   print(f"R² score: {metrics['r2_score']:.3f}")
   print(f"Feature importance: {metrics['feature_importance']}")
   ```

3. **Validation croisée** :

   ```python
   from sklearn.model_selection import cross_val_score

   scores = cross_val_score(predictor.model, X, y, cv=5, scoring='r2')
   print(f"Cross-validation R² scores: {scores}")
   print(f"Mean: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
   ```

**Critère de succès** :

- R² score > 0.70 (explique 70% de la variance)
- MAE (Mean Absolute Error) < 5 min
- Meilleures prédictions que le baseline (delay_predictor.py)

#### Phase 2 : Prototype Intégration - 4 semaines

**Objectif** : Intégrer le ML dans le pipeline de dispatch.

**Architecture** :

```python
# services/unified_dispatch/engine.py

def run(...):
    # ... (existing code)

    # NOUVEAU : Prédiction ML AVANT assignation finale
    if settings.features.enable_ml_predictions:
        ml_predictor = get_ml_predictor()

        for assignment in final_assignments:
            booking = bookings_map[assignment.booking_id]
            driver = drivers_map[assignment.driver_id]

            # Prédire le retard
            prediction = ml_predictor.predict_delay(booking, driver)

            # Si retard prédit >10 min, marquer l'assignation comme "at_risk"
            if prediction.predicted_delay_minutes > 10:
                assignment.risk_level = prediction.risk_level
                assignment.confidence = prediction.confidence

                logger.warning(
                    "[ML] Assignment at risk: booking=%s driver=%s predicted_delay=%d min",
                    booking.id, driver.id, prediction.predicted_delay_minutes
                )

                # Optionnel : essayer de réassigner à un meilleur chauffeur
                if settings.features.enable_ml_reoptimization:
                    better_driver = find_better_driver(booking, prediction)
                    if better_driver:
                        assignment.driver_id = better_driver.id
                        logger.info("[ML] Réassigné à chauffeur %s pour réduire retard", better_driver.id)

    # ... (rest of existing code)
```

**Nouvelles tables DB** :

```sql
-- Table pour stocker les prédictions ML
CREATE TABLE ml_prediction (
    id SERIAL PRIMARY KEY,
    assignment_id INTEGER REFERENCES assignment(id),
    predicted_delay_minutes FLOAT NOT NULL,
    confidence FLOAT NOT NULL,
    risk_level VARCHAR(20) NOT NULL,  -- 'low', 'medium', 'high'
    actual_delay_minutes FLOAT,       -- Rempli après coup pour feedback
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Index pour analyse performance du modèle
CREATE INDEX idx_ml_prediction_risk ON ml_prediction(risk_level, created_at);
CREATE INDEX idx_ml_prediction_assignment ON ml_prediction(assignment_id);
```

**API Endpoints pour monitoring** :

```python
# routes/ml_routes.py

@ml_ns.route("/predictions/accuracy")
class MLAccuracyResource(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        """
        Retourne l'accuracy du modèle ML sur les 7 derniers jours.
        """
        company_id = get_current_company_id()

        cutoff = datetime.now() - timedelta(days=7)
        predictions = MLPrediction.query.filter(
            MLPrediction.assignment.has(company_id=company_id),
            MLPrediction.created_at >= cutoff,
            MLPrediction.actual_delay_minutes.isnot(None)
        ).all()

        # Calculer métriques
        mae = mean_absolute_error(
            [p.actual_delay_minutes for p in predictions],
            [p.predicted_delay_minutes for p in predictions]
        )

        r2 = r2_score(
            [p.actual_delay_minutes for p in predictions],
            [p.predicted_delay_minutes for p in predictions]
        )

        return {
            "predictions_count": len(predictions),
            "mae": round(mae, 2),
            "r2_score": round(r2, 3),
            "period": "last_7_days"
        }
```

#### Phase 3 : Production ML-Driven - 8 semaines

**Objectif** : Système de dispatch entièrement piloté par le ML.

**Composants** :

1. **Reinforcement Learning (RL)** :
   - Agent RL qui apprend la meilleure politique de dispatch
   - Reward = -retard - distance - coût_emergency
   - State = (bookings, drivers, time, traffic)
   - Action = assign(booking_i, driver_j)

```python
# services/unified_dispatch/rl_agent.py
import torch
import torch.nn as nn

class DispatchAgent(nn.Module):
    """
    Deep Q-Network (DQN) pour dispatch optimal.
    """
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, state):
        return self.network(state)

    def select_action(self, state, epsilon=0.1):
        """
        Epsilon-greedy policy.
        """
        if random.random() < epsilon:
            # Explore: random assignment
            return random.randint(0, self.action_dim - 1)
        else:
            # Exploit: best predicted Q-value
            with torch.no_grad():
                q_values = self.forward(torch.FloatTensor(state))
                return q_values.argmax().item()

def train_rl_agent(historical_episodes):
    """
    Entraîne l'agent RL sur des épisodes historiques.
    Un épisode = un dispatch complet (toutes les assignations d'une journée).
    """
    agent = DispatchAgent(state_dim=50, action_dim=100)
    optimizer = torch.optim.Adam(agent.parameters(), lr=0.001)

    for epoch in range(100):
        for episode in historical_episodes:
            state = episode['initial_state']
            actions = episode['actions']
            rewards = episode['rewards']

            # Q-learning update
            q_values = agent(torch.FloatTensor(state))
            target_q = rewards + 0.99 * agent(torch.FloatTensor(episode['next_state'])).max()

            loss = nn.MSELoss()(q_values, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return agent
```

2. **Multi-Objective Optimization** :
   - Objectifs concurrents :
     - Minimiser retard
     - Minimiser distance totale
     - Maximiser équité entre chauffeurs
     - Minimiser coût (chauffeurs d'urgence)
   - Algorithme : NSGA-II (Non-dominated Sorting Genetic Algorithm)

```python
# services/unified_dispatch/multi_objective_solver.py
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem

class DispatchProblem(Problem):
    def __init__(self, bookings, drivers):
        super().__init__(
            n_var=len(bookings),          # Variables: driver_id pour chaque booking
            n_obj=4,                       # 4 objectifs
            n_constr=2,                    # Contraintes: time windows, capacité
            xl=0,                          # Min driver_id = 0
            xu=len(drivers)-1              # Max driver_id = len(drivers)-1
        )
        self.bookings = bookings
        self.drivers = drivers

    def _evaluate(self, X, out, *args, **kwargs):
        """
        X: matrice (n_solutions, n_bookings) avec driver_id pour chaque booking
        """
        # Calculer les 4 objectifs pour chaque solution
        objectives = []
        for solution in X:
            # Objective 1: Total delay
            total_delay = calculate_total_delay(solution, self.bookings, self.drivers)

            # Objective 2: Total distance
            total_distance = calculate_total_distance(solution, self.bookings, self.drivers)

            # Objective 3: Fairness (std dev of bookings per driver)
            fairness = calculate_fairness(solution, self.drivers)

            # Objective 4: Cost (emergency drivers penalty)
            total_cost = calculate_cost(solution, self.drivers)

            objectives.append([total_delay, total_distance, -fairness, total_cost])

        out["F"] = np.array(objectives)

def solve_multi_objective(bookings, drivers):
    problem = DispatchProblem(bookings, drivers)
    algorithm = NSGA2(pop_size=100)

    res = minimize(
        problem,
        algorithm,
        termination=('n_gen', 200),
        seed=42,
        verbose=True
    )

    # Retourner le Pareto front
    return res.F, res.X
```

3. **AutoML Pipeline** :
   - Entraînement automatique tous les jours
   - A/B testing : comparer ML vs heuristiques

```python
# tasks/ml_tasks.py

@shared_task(name="tasks.ml_tasks.retrain_model_daily")
def retrain_model_daily():
    """
    Réentraîne le modèle ML tous les jours sur les 30 derniers jours.
    Compare avec l'ancien modèle et déploie si meilleur.
    """
    logger.info("[ML] Starting daily retraining...")

    # Collecter données
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    training_data = collect_historical_data(start_date, end_date)

    # Entraîner nouveau modèle
    new_predictor = DelayMLPredictor(model_path="backend/data/ml_models/delay_predictor_new.pkl")
    new_metrics = new_predictor.train_on_historical_data(training_data, save_model=True)

    # Charger ancien modèle
    old_predictor = DelayMLPredictor(model_path="backend/data/ml_models/delay_predictor.pkl")

    # Comparer sur validation set
    val_data = collect_historical_data(end_date - timedelta(days=7), end_date)

    new_mae = evaluate_model(new_predictor, val_data)
    old_mae = evaluate_model(old_predictor, val_data)

    logger.info(f"[ML] New model MAE: {new_mae:.2f}, Old model MAE: {old_mae:.2f}")

    # Déployer si meilleur
    if new_mae < old_mae:
        os.rename(
            "backend/data/ml_models/delay_predictor_new.pkl",
            "backend/data/ml_models/delay_predictor.pkl"
        )
        logger.info("[ML] ✅ New model deployed!")
    else:
        logger.info("[ML] ⏸️ Old model kept (better performance)")

    return {
        "new_mae": new_mae,
        "old_mae": old_mae,
        "deployed": new_mae < old_mae
    }
```

---

## 6. SYSTÈME AUTO-AMÉLIORANT (SELF-LEARNING)

### 6.1 Architecture Proposée

```
┌─────────────────────────────────────────────────────────┐
│                  DISPATCH EXECUTION                      │
│  ├─ Heuristics                                           │
│  ├─ OR-Tools Solver                                      │
│  └─ ML Predictor                                         │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              METRICS COLLECTION                          │
│  ├─ Quality Score (0-100)                                │
│  ├─ On-Time Rate (%)                                     │
│  ├─ Assignment Rate (%)                                  │
│  ├─ Pooling Rate (%)                                     │
│  ├─ Fairness Score                                       │
│  └─ Average Delay (min)                                  │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│             PERFORMANCE ANALYSIS                         │
│  ├─ Compare vs baseline                                  │
│  ├─ Identify patterns (day/time/driver)                  │
│  └─ Detect anomalies                                     │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│             PARAMETER TUNING (Auto-Tuning)               │
│  ├─ Adjust heuristic weights                             │
│  ├─ Adjust solver time limits                            │
│  ├─ Adjust buffer times                                  │
│  └─ Retrain ML models                                    │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼ (feedback loop)
         ┌────────────────────────┐
         │   NEXT DISPATCH RUN    │
         └────────────────────────┘
```

### 6.2 KPIs Métier

#### 6.2.1 Quality Score (0-100)

```python
# services/unified_dispatch/dispatch_metrics.py

def calculate_quality_score(metrics: DispatchMetrics) -> float:
    """
    Calcule un score de qualité composite (0-100).

    Pondération :
    - 40% : Taux d'assignation (bookings assignés / total)
    - 30% : Taux à l'heure (on_time / total)
    - 20% : Équité (std dev bookings par chauffeur, normalisé)
    - 10% : Pooling rate (optimisation ressources)
    """
    # Assignment rate (0-40 points)
    assignment_rate = (metrics.total_bookings - metrics.cancelled_bookings) / max(1, metrics.total_bookings)
    assignment_score = assignment_rate * 40

    # On-time rate (0-30 points)
    on_time_rate = metrics.on_time_bookings / max(1, metrics.total_bookings)
    on_time_score = on_time_rate * 30

    # Fairness (0-20 points)
    # Calculer écart-type des charges par chauffeur
    bookings_per_driver = [...]  # Distribution
    fairness = 1.0 - (np.std(bookings_per_driver) / np.mean(bookings_per_driver))
    fairness_score = fairness * 20

    # Pooling rate (0-10 points)
    pooling_score = metrics.pooling_rate * 10

    return assignment_score + on_time_score + fairness_score + pooling_score
```

#### 6.2.2 Tableau de Bord KPIs

```python
# routes/analytics.py

@analytics_ns.route("/kpi/dashboard")
class KPIDashboardResource(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        """
        Dashboard KPI pour une entreprise.
        """
        company_id = get_current_company_id()

        # Derniers 7 jours
        end_date = date.today()
        start_date = end_date - timedelta(days=7)

        metrics = DispatchMetrics.query.filter(
            DispatchMetrics.company_id == company_id,
            DispatchMetrics.date >= start_date,
            DispatchMetrics.date <= end_date
        ).all()

        # Agréger
        total_bookings = sum(m.total_bookings for m in metrics)
        on_time_bookings = sum(m.on_time_bookings for m in metrics)
        delayed_bookings = sum(m.delayed_bookings for m in metrics)
        avg_delay = np.mean([m.average_delay_minutes for m in metrics])
        quality_scores = [m.quality_score for m in metrics]

        return {
            "period": {"start": start_date.isoformat(), "end": end_date.isoformat()},
            "kpis": {
                "total_bookings": total_bookings,
                "on_time_rate": round(on_time_bookings / max(1, total_bookings) * 100, 1),
                "delay_rate": round(delayed_bookings / max(1, total_bookings) * 100, 1),
                "average_delay_minutes": round(avg_delay, 1),
                "quality_score_avg": round(np.mean(quality_scores), 1),
                "quality_score_trend": round(quality_scores[-1] - quality_scores[0], 1),
            },
            "daily_breakdown": [
                {
                    "date": m.date.isoformat(),
                    "quality_score": m.quality_score,
                    "on_time_rate": round(m.on_time_bookings / max(1, m.total_bookings) * 100, 1),
                    "avg_delay": m.average_delay_minutes,
                }
                for m in metrics
            ]
        }
```

### 6.3 Auto-Tuning des Paramètres

```python
# services/unified_dispatch/auto_tuner.py

class DispatchAutoTuner:
    """
    Ajuste automatiquement les paramètres du dispatch selon les performances.
    """

    def __init__(self, company_id: int):
        self.company_id = company_id
        self.company = Company.query.get(company_id)
        self.settings = Settings.for_company(self.company)

    def analyze_performance(self, days=7) -> Dict[str, Any]:
        """
        Analyse les performances des N derniers jours.
        """
        end_date = date.today()
        start_date = end_date - timedelta(days=days)

        metrics = DispatchMetrics.query.filter(
            DispatchMetrics.company_id == self.company_id,
            DispatchMetrics.date >= start_date
        ).all()

        return {
            "avg_quality_score": np.mean([m.quality_score for m in metrics]),
            "avg_delay": np.mean([m.average_delay_minutes for m in metrics]),
            "on_time_rate": np.mean([m.on_time_bookings / max(1, m.total_bookings) for m in metrics]),
            "fairness": self._calculate_fairness(metrics),
            "pooling_rate": np.mean([m.pooling_rate for m in metrics]),
        }

    def suggest_tuning(self, performance: Dict[str, Any]) -> Dict[str, Any]:
        """
        Suggère des ajustements de paramètres.
        """
        suggestions = {}

        # Si quality_score < 80, ajuster
        if performance["avg_quality_score"] < 80:
            # Identifier le goulot d'étranglement
            if performance["on_time_rate"] < 0.85:
                # Problème de retards → augmenter buffers
                suggestions["time.pickup_buffer_min"] = self.settings.time.pickup_buffer_min + 2
                suggestions["time.post_trip_buffer_min"] = self.settings.time.post_trip_buffer_min + 5

            if performance["fairness"] < 0.7:
                # Problème d'équité → augmenter poids fairness
                suggestions["heuristic.driver_load_balance"] = min(0.8, self.settings.heuristic.driver_load_balance + 0.1)

            if performance["pooling_rate"] < 0.2:
                # Pas assez de pooling → assouplir contraintes
                suggestions["pooling.time_tolerance_min"] = self.settings.pooling.time_tolerance_min + 5
                suggestions["pooling.pickup_distance_m"] = self.settings.pooling.pickup_distance_m + 200

        return suggestions

    def apply_tuning(self, suggestions: Dict[str, Any], dry_run=False):
        """
        Applique les ajustements suggérés.
        """
        if dry_run:
            logger.info(f"[AutoTuner] Would apply: {suggestions}")
            return

        # Charger settings actuels
        current_settings = json.loads(self.company.dispatch_settings or "{}")

        # Appliquer suggestions
        updated_settings = merge_overrides(current_settings, suggestions)

        # Sauvegarder
        self.company.dispatch_settings = json.dumps(updated_settings)
        db.session.add(self.company)
        db.session.commit()

        logger.info(f"[AutoTuner] Applied tuning for company {self.company_id}: {suggestions}")
```

**Celery Task Périodique** :

```python
# tasks/dispatch_tasks.py

@shared_task(name="tasks.dispatch_tasks.auto_tune_parameters")
def auto_tune_parameters():
    """
    Tâche hebdomadaire : ajuste automatiquement les paramètres de dispatch
    pour toutes les entreprises avec quality_score < 80.
    """
    companies = Company.query.filter_by(dispatch_enabled=True).all()

    for company in companies:
        try:
            tuner = DispatchAutoTuner(company.id)

            # Analyser performances
            performance = tuner.analyze_performance(days=7)

            # Suggérer ajustements
            suggestions = tuner.suggest_tuning(performance)

            if not suggestions:
                logger.info(f"[AutoTuner] Company {company.id}: No tuning needed (quality: {performance['avg_quality_score']:.1f}/100)")
                continue

            # Appliquer (dry_run=False en production)
            tuner.apply_tuning(suggestions, dry_run=False)

            # Notifier l'admin
            notify_admin_tuning_applied(company.id, suggestions, performance)

        except Exception as e:
            logger.exception(f"[AutoTuner] Failed for company {company.id}: {e}")
```

---
