# 🎯 PLAN D'ACTION CONCRET - Amélioration Système Dispatch

**Date de début** : 21 octobre 2025  
**Durée totale** : 12 semaines (3 mois)  
**Équipe** : 1 Dev Senior + 1 Data Scientist (temps partiel)

---

## 📅 SPRINT PLANNING

### SEMAINE 1 : Quick Wins & Cleanup

**Objectif** : Nettoyer le code et améliorer la base

#### Tâches

**Lundi - Mardi : Code Mort (2 jours)** 🗑️

- [ ] **Task 1.1** : Supprimer fichiers inutiles

  ```bash
  rm backend/Classeur1.xlsx
  rm backend/transport.xlsx
  rm backend/check_bookings.py
  ```

  **Effort** : 30 min  
  **Impact** : -150 KB code mort

- [ ] **Task 1.2** : Refactoriser redondances Haversine

  - Créer `backend/shared/geo_utils.py`
  - Migrer 3 implémentations → 1 seule
  - Update imports dans `heuristics.py`, `data.py`

  **Effort** : 3h  
  **Impact** : -100 lignes code dupliqué

- [ ] **Task 1.3** : Centraliser sérialisation assignations

  - Créer `backend/schemas/dispatch_schemas.py`
  - Schema Marshmallow pour `Assignment`
  - Remplacer `.serialize` / `.to_dict()` partout

  **Effort** : 4h  
  **Impact** : Code plus maintenable

**Mercredi - Jeudi : Optimisations SQL (2 jours)** 🚀

- [ ] **Task 1.4** : Bulk inserts dans `apply.py`

  ```python
  # Remplacer boucle de commits par bulk_insert_mappings
  db.session.bulk_insert_mappings(Assignment, assignment_dicts)
  db.session.commit()  # 1 seul commit
  ```

  **Effort** : 4h  
  **Impact** : -90% temps d'écriture DB

- [ ] **Task 1.5** : Ajouter index DB manquants
  ```sql
  CREATE INDEX idx_assignment_company_created ON assignment(booking_id, created_at DESC);
  CREATE INDEX idx_booking_status_scheduled_company ON booking(status, scheduled_time, company_id);
  CREATE INDEX idx_driver_available_company ON driver(company_id) WHERE is_available=true;
  ```
  **Effort** : 2h (migration Alembic)  
  **Impact** : -50% temps queries critiques

**Vendredi : Tests Unitaires Setup (1 jour)** 🧪

- [ ] **Task 1.6** : Setup pytest infrastructure

  ```bash
  pip install pytest pytest-cov pytest-flask factory-boy faker
  ```

  - Créer `tests/conftest.py` (fixtures DB, app, etc.)
  - Créer `tests/factories.py` (Factory Boy pour models)

  **Effort** : 6h  
  **Impact** : Infrastructure tests prête

**Livrable Semaine 1** :

- ✅ -10% code inutile
- ✅ +50% performance SQL
- ✅ Infrastructure tests opérationnelle

---

### SEMAINE 2 : Tests Critiques

**Objectif** : Couvrir les modules critiques avec tests

#### Tâches

**Lundi : Tests engine.py (1 jour)** 🧪

- [ ] **Task 2.1** : Tests `engine.run()`

  ```python
  # tests/test_engine.py
  def test_engine_run_creates_dispatch_run():
      ...

  def test_engine_run_assigns_bookings():
      ...

  def test_engine_run_handles_no_drivers():
      ...

  def test_engine_run_with_emergency_drivers():
      ...

  def test_engine_run_respects_time_windows():
      ...
  ```

  **Effort** : 8h (10 tests)  
  **Coverage cible** : 70% de `engine.py`

**Mardi : Tests heuristics.py (1 jour)** 🧪

- [ ] **Task 2.2** : Tests algorithmes heuristiques

  ```python
  # tests/test_heuristics.py
  def test_assign_urgent_returns():
      ...

  def test_greedy_assignment():
      ...

  def test_fairness_scoring():
      ...

  def test_pooling_detection():
      ...
  ```

  **Effort** : 8h (8 tests)  
  **Coverage cible** : 60% de `heuristics.py`

**Mercredi : Tests solver.py (1 jour)** 🧪

- [ ] **Task 2.3** : Tests OR-Tools wrapper

  ```python
  # tests/test_solver.py
  def test_solver_simple_case():
      # 2 bookings, 2 drivers → 2 assignments
      ...

  def test_solver_respects_time_windows():
      ...

  def test_solver_handles_capacity_constraints():
      ...

  def test_solver_too_large_fallback():
      ...
  ```

  **Effort** : 8h (12 tests)  
  **Coverage cible** : 75% de `solver.py`

**Jeudi : Tests autonomous_manager.py (1 jour)** 🧪

- [ ] **Task 2.4** : Tests gestionnaire autonome

  ```python
  # tests/test_autonomous_manager.py
  def test_should_run_autorun_manual_mode():
      assert manager.should_run_autorun() == False

  def test_should_run_autorun_fully_auto():
      assert manager.should_run_autorun() == True

  def test_can_auto_apply_notification():
      assert manager.can_auto_apply_suggestion(notify_suggestion) == True

  def test_can_auto_apply_reassignment():
      assert manager.can_auto_apply_suggestion(reassign_suggestion) == False
  ```

  **Effort** : 6h (15 tests)  
  **Coverage cible** : 90% de `autonomous_manager.py`

**Vendredi : CI/CD GitHub Actions (1 jour)** 🔧

- [ ] **Task 2.5** : Configuration pipeline CI

  - Créer `.github/workflows/ci.yml`
  - Tests automatiques sur chaque PR
  - Coverage report (Codecov)
  - Bloquer merge si tests fail ou coverage < 70%

  **Effort** : 4h  
  **Impact** : Prévention régressions

**Livrable Semaine 2** :

- ✅ 45 tests unitaires créés
- ✅ 70% coverage modules critiques
- ✅ CI/CD opérationnel

---

### SEMAINE 3-4 : ML POC (Proof of Concept)

**Objectif** : Prouver que le ML améliore les prédictions

#### SEMAINE 3 : Collecte & Préparation Données

**Lundi : Script collecte (1 jour)** 📊

- [ ] **Task 3.1** : Implémenter `collect_training_data.py`

  - Extraction 90 derniers jours
  - Features engineering
  - Sauvegarde JSON + CSV

  **Effort** : 6h  
  **Output** : `training_data.csv` (estimé 5,000-10,000 échantillons)

**Mardi : Analyse exploratoire (1 jour)** 📈

- [ ] **Task 3.2** : EDA (Exploratory Data Analysis)

  ```python
  import pandas as pd
  import seaborn as sns

  df = pd.read_csv("training_data.csv")

  # Distribution retards
  sns.histplot(df['actual_delay_minutes'], bins=50)

  # Corrélations features
  sns.heatmap(df.corr(), annot=True)

  # Outliers
  df[df['actual_delay_minutes'] > 60]  # Retards extrêmes
  ```

  **Effort** : 4h  
  **Output** : `data_analysis_report.html` (Pandas Profiling)

**Mercredi : Feature engineering (1 jour)** 🔧

- [ ] **Task 3.3** : Enrichir features (v1 → v2)

  - Ajouter 15 features supplémentaires (historique driver, météo, etc.)
  - Normalisation / standardisation
  - Encoding variables catégorielles

  **Effort** : 6h  
  **Output** : `extract_features_v2()` implémenté

**Jeudi - Vendredi : Data cleaning (2 jours)** 🧹

- [ ] **Task 3.4** : Nettoyage données

  - Supprimer outliers (retards > 120 min = erreurs)
  - Imputer valeurs manquantes (médiane/mode)
  - Équilibrage dataset (over/under-sampling si déséquilibré)

  **Effort** : 8h  
  **Output** : Dataset propre prêt pour entraînement

#### SEMAINE 4 : Training & Évaluation

**Lundi - Mardi : Training modèle (2 jours)** 🤖

- [ ] **Task 4.1** : Entraîner RandomForest

  ```python
  from sklearn.model_selection import train_test_split, cross_val_score

  # Split
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.15, random_state=42
  )

  # Train
  predictor = DelayMLPredictor()
  metrics = predictor.train_on_historical_data(training_data, save_model=True)

  # Cross-validation
  scores = cross_val_score(predictor.model, X, y, cv=5, scoring='r2')
  print(f"CV R² scores: {scores}")
  print(f"Mean: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
  ```

  **Effort** : 10h (iterations hyperparamètres)  
  **Critère succès** : R² > 0.70, MAE < 5 min

- [ ] **Task 4.2** : Comparer RandomForest vs XGBoost

  ```python
  from sklearn.metrics import mean_absolute_error, r2_score

  # RandomForest
  rf_pred = rf_model.predict(X_test)
  rf_mae = mean_absolute_error(y_test, rf_pred)
  rf_r2 = r2_score(y_test, rf_pred)

  # XGBoost
  xgb_pred = xgb_model.predict(X_test)
  xgb_mae = mean_absolute_error(y_test, xgb_pred)
  xgb_r2 = r2_score(y_test, xgb_pred)

  print(f"RandomForest: MAE={rf_mae:.2f}, R²={rf_r2:.3f}")
  print(f"XGBoost:      MAE={xgb_mae:.2f}, R²={xgb_r2:.3f}")
  ```

  **Effort** : 6h  
  **Décision** : Choisir meilleur modèle

**Mercredi : Validation croisée (1 jour)** ✅

- [ ] **Task 4.3** : Validation rigoureuse

  - K-fold cross-validation (k=5)
  - Test sur données complètement holdout (mois N-1)
  - Analyse erreurs (où le modèle se trompe le plus ?)

  **Effort** : 6h  
  **Output** : Rapport validation

**Jeudi : Feature importance (1 jour)** 📊

- [ ] **Task 4.4** : Analyser importance features

  ```python
  # Top features
  importances = model.feature_importances_
  feature_ranking = sorted(
      zip(feature_names, importances),
      key=lambda x: x[1],
      reverse=True
  )

  print("Top 5 features:")
  for name, importance in feature_ranking[:5]:
      print(f"  {name}: {importance:.3f}")
  ```

  - Si feature inutile (importance < 0.02) → supprimer
  - Si feature très importante → investiguer pourquoi

  **Effort** : 4h  
  **Output** : Liste features optimales

**Vendredi : Go/No-Go Decision** 🎯

- [ ] **Task 4.5** : Rapport POC + décision

  - Présentation résultats (MAE, R², feature importance)
  - Comparaison vs baseline (`delay_predictor.py`)
  - **GO** si ML > baseline OU **NO-GO** si ML ≤ baseline

  **Effort** : 4h  
  **Décision** : Continuer vers intégration production

**Livrable Semaines 3-4** :

- ✅ Modèle ML entraîné (MAE < 5 min, R² > 0.70)
- ✅ Rapport validation complet
- ✅ Go/No-Go décision documentée

---

### SEMAINE 5 : Safety & Audit Trail

**Objectif** : Rendre fully-auto mode production-ready

#### Tâches

**Lundi : Migration DB (1 jour)** 💾

- [ ] **Task 5.1** : Créer tables ML + Audit

  ```bash
  # backend/migrations/versions/xxx_add_ml_tables.py

  # Table 1 : ml_prediction
  op.create_table(
      'ml_prediction',
      sa.Column('id', sa.Integer(), primary_key=True),
      sa.Column('assignment_id', sa.Integer(), sa.ForeignKey('assignment.id')),
      sa.Column('predicted_delay_minutes', sa.Float(), nullable=False),
      sa.Column('confidence', sa.Float(), nullable=False),
      sa.Column('risk_level', sa.String(20), nullable=False),
      sa.Column('feature_vector', JSONB(), nullable=False),
      sa.Column('actual_delay_minutes', sa.Float(), nullable=True),
      sa.Column('prediction_error', sa.Float(), nullable=True),
      sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
  )

  # Table 2 : autonomous_action
  op.create_table(
      'autonomous_action',
      sa.Column('id', sa.Integer(), primary_key=True),
      sa.Column('company_id', sa.Integer(), sa.ForeignKey('company.id')),
      sa.Column('action_type', sa.String(50), nullable=False),
      sa.Column('entity_type', sa.String(50), nullable=False),
      sa.Column('entity_id', sa.Integer(), nullable=False),
      sa.Column('decision_context', JSONB(), nullable=False),
      sa.Column('applied_at', sa.DateTime(timezone=True), nullable=False),
      sa.Column('success', sa.Boolean(), nullable=False),
      ...
  )
  ```

  **Effort** : 6h (migration + tests)  
  **Impact** : Audit trail complet

**Mardi : Safety Limits (1 jour)** 🛡️

- [ ] **Task 5.2** : Implémenter `check_safety_limits()`
  ```python
  # autonomous_manager.py
  def check_safety_limits(self, action_type):
      # Rate limiting
      one_hour_ago = datetime.now(UTC) - timedelta(hours=1)
      recent_count = AutonomousAction.query.filter(
          AutonomousAction.company_id == self.company_id,
          AutonomousAction.action_type == action_type,
          AutonomousAction.applied_at >= one_hour_ago
      ).count()

      limits = self.config["safety_limits"]
      max_per_hour = limits["max_auto_actions_per_hour"]

      if recent_count >= max_per_hour:
          return False, f"Rate limit: {recent_count}/{max_per_hour}/h"

      # Daily limits (réassignations)
      if action_type == "reassign":
          today_start = datetime.now(UTC).replace(hour=0, minute=0, second=0)
          today_count = AutonomousAction.query.filter(
              AutonomousAction.company_id == self.company_id,
              AutonomousAction.action_type == "reassign",
              AutonomousAction.applied_at >= today_start
          ).count()

          max_per_day = limits["max_auto_reassignments_per_day"]
          if today_count >= max_per_day:
              return False, f"Daily limit: {today_count}/{max_per_day}/day"

      return True, "OK"
  ```
  **Effort** : 6h  
  **Impact** : Sécurité fully-auto garantie

**Mercredi : Logging Actions (1 jour)** 📝

- [ ] **Task 5.3** : Logger toutes actions automatiques
  ```python
  # autonomous_manager.py
  def process_opportunities(...):
      for suggestion in suggestions:
          if self.can_auto_apply_suggestion(suggestion):
              try:
                  # Appliquer
                  result = apply_suggestion(suggestion, ...)

                  # Logger dans AutonomousAction
                  action = AutonomousAction(
                      company_id=self.company_id,
                      action_type=suggestion.action,
                      entity_type="assignment",
                      entity_id=suggestion.assignment_id,
                      decision_context={
                          "suggestion": suggestion.to_dict(),
                          "mode": self.mode.value,
                          "config": self.config,
                      },
                      applied_at=datetime.now(UTC),
                      success=result.get("success", False),
                      error_message=result.get("error"),
                  )
                  db.session.add(action)
                  db.session.commit()

              except Exception as e:
                  logger.exception("Failed to apply suggestion")
  ```
  **Effort** : 4h  
  **Impact** : Traçabilité complète

**Jeudi : Dashboard Admin (1 jour)** 📊

- [ ] **Task 5.4** : UI pour review actions automatiques

  - Route `/admin/autonomous-actions`
  - Liste actions 7 derniers jours
  - Filtres par type, success/failure
  - Export CSV pour audit

  **Effort** : 6h (backend + frontend)  
  **Impact** : Visibilité pour admins

**Vendredi : Tests Safety (1 jour)** 🧪

- [ ] **Task 5.5** : Tests safety limits
  ```python
  def test_rate_limiting_blocks_excess_actions():
      # Créer 50 actions dans la dernière heure
      for i in range(50):
          create_autonomous_action(company_id=1, action_type="notify")

      # Tenter 51ème action
      can_apply, reason = manager.check_safety_limits("notify")

      assert can_apply == False
      assert "Rate limit" in reason
  ```
  **Effort** : 4h  
  **Coverage cible** : 100% de `check_safety_limits()`

**Livrable Semaine 5** :

- ✅ Tables ML + Audit créées
- ✅ Safety limits implémentés + testés
- ✅ Dashboard admin opérationnel

---

### SEMAINE 6 : Intégration ML dans Pipeline

**Objectif** : ML actif dans engine.py (mode expérimental)

#### Tâches

**Lundi : Feature flag (1 jour)** 🚩

- [ ] **Task 6.1** : Ajouter configuration ML

  ```python
  # settings.py
  @dataclass
  class MLSettings:
      enabled: bool = False  # Désactivé par défaut
      model_path: str = "backend/data/ml_models/delay_predictor.pkl"
      reoptimize_threshold_minutes: int = 10  # Si prédit >10 min → réassigner
      min_confidence: float = 0.70  # Confiance min pour réassigner
      min_gain_minutes: int = 5  # Gain min pour valoir la peine

  @dataclass
  class Settings:
      # ... (existing)
      ml: MLSettings = field(default_factory=MLSettings)
  ```

  **Effort** : 2h

- [ ] **Task 6.2** : Endpoint admin pour activer/désactiver ML
  ```python
  @admin_ns.route("/ml/toggle")
  def post(self):
      enabled = request.json.get("enabled", False)
      company.dispatch_settings["ml"] = {"enabled": enabled}
      db.session.commit()
      return {"ml_enabled": enabled}
  ```
  **Effort** : 2h

**Mardi - Mercredi : Intégration engine.py (2 jours)** 🔌

- [ ] **Task 6.3** : Code ML dans engine.py (voir section 5.1)

  - Prédiction pour chaque assignment
  - Réassignation si risque élevé
  - Sauvegarde prédictions en DB

  **Effort** : 12h  
  **Impact** : ML actif !

**Jeudi : Tests intégration (1 jour)** 🧪

- [ ] **Task 6.4** : Tests end-to-end avec ML
  ```python
  def test_engine_run_with_ml_enabled():
      # Activer ML
      settings = Settings()
      settings.ml.enabled = True

      # Run dispatch
      result = engine.run(company_id=1, for_date="2025-10-20", custom_settings=settings)

      # Vérifier que des prédictions ont été créées
      predictions = MLPrediction.query.filter_by(
          assignment_id__in=[a.id for a in result["assignments"]]
      ).all()

      assert len(predictions) > 0
      assert all(p.confidence >= 0 for p in predictions)
  ```
  **Effort** : 6h

**Vendredi : Monitoring ML (1 jour)** 📈

- [ ] **Task 6.5** : Dashboard métriques ML

  - Endpoint `/api/ml/stats`
  - Grafana dashboard :
    - MAE over time
    - R² score over time
    - Predictions count
    - Feature importance

  **Effort** : 6h  
  **Output** : Monitoring opérationnel

**Livrable Semaine 6** :

- ✅ ML intégré dans pipeline (feature flag)
- ✅ Tests intégration passent
- ✅ Monitoring actif

---

### SEMAINE 7-8 : A/B Testing & Production

**Objectif** : Valider ML en conditions réelles

#### SEMAINE 7 : A/B Testing

**Lundi : Setup A/B test (1 jour)** 🧪

- [ ] **Task 7.1** : Split entreprises en 2 groupes

  ```python
  # Groupe A : ML enabled (50% entreprises)
  # Groupe B : ML disabled (50% entreprises, contrôle)

  companies = Company.query.filter_by(dispatch_enabled=True).all()
  random.shuffle(companies)

  group_a = companies[:len(companies)//2]
  group_b = companies[len(companies)//2:]

  for c in group_a:
      settings = json.loads(c.dispatch_settings or "{}")
      settings["ml"] = {"enabled": True}
      c.dispatch_settings = json.dumps(settings)

  db.session.commit()
  ```

  **Effort** : 2h

**Mardi - Vendredi : Collecte données (4 jours)** 📊

- [ ] **Task 7.2** : Laisser tourner 1 semaine

  - Collecter métriques quotidiennes (groupe A vs B)
  - Surveiller erreurs / anomalies
  - Pas de modifications pendant le test

  **Effort** : Monitoring quotidien (1h/jour)

#### SEMAINE 8 : Analyse Résultats

**Lundi - Mardi : Analyse statistique (2 jours)** 📈

- [ ] **Task 8.1** : Comparer métriques A vs B

  ```python
  import scipy.stats as stats

  # Métriques groupe A (ML enabled)
  group_a_metrics = DispatchMetrics.query.filter(
      company_id.in_([c.id for c in group_a]),
      date >= test_start_date
  ).all()

  # Métriques groupe B (ML disabled)
  group_b_metrics = DispatchMetrics.query.filter(
      company_id.in_([c.id for c in group_b]),
      date >= test_start_date
  ).all()

  # T-test : On-Time Rate
  a_on_time = [m.on_time_rate for m in group_a_metrics]
  b_on_time = [m.on_time_rate for m in group_b_metrics]

  t_stat, p_value = stats.ttest_ind(a_on_time, b_on_time)

  print(f"On-Time Rate:")
  print(f"  Groupe A (ML): {np.mean(a_on_time):.1f}%")
  print(f"  Groupe B (Control): {np.mean(b_on_time):.1f}%")
  print(f"  Différence: {np.mean(a_on_time) - np.mean(b_on_time):.1f}%")
  print(f"  p-value: {p_value:.4f}")

  if p_value < 0.05:
      print("✅ Différence statistiquement significative (p<0.05)")
  else:
      print("⚠️ Pas de différence significative (p≥0.05)")
  ```

  **Effort** : 10h  
  **Output** : Rapport A/B test

**Mercredi : Décision Déploiement (1 jour)** 🎯

- [ ] **Task 8.2** : Critères de déploiement

  **Si résultats positifs** (p<0.05 ET amélioration >5%) :

  - ✅ Activer ML pour toutes les entreprises
  - ✅ Passer en mode production

  **Si résultats mitigés** (amélioration <5% OU p≥0.05) :

  - ⚠️ Itérer sur modèle (plus de données ? autres features ?)
  - ⚠️ Retester dans 2 semaines

  **Si résultats négatifs** (ML pire que baseline) :

  - ❌ Désactiver ML
  - ❌ Analyser pourquoi (overfitting ? mauvaises features ?)

**Jeudi - Vendredi : Déploiement Production (2 jours)** 🚀

- [ ] **Task 8.3** : Rollout ML à 100%

  - Activer pour toutes entreprises
  - Monitoring intensif (1h)
  - Rollback plan prêt (désactiver ML en 1 click)

  **Effort** : 4h + surveillance

- [ ] **Task 8.4** : Documentation

  - Guide utilisateur (comment interpréter prédictions ML)
  - Runbook opérationnel (incidents ML, rollback)
  - Update README.md

  **Effort** : 6h

**Livrable Semaines 7-8** :

- ✅ A/B test complété
- ✅ ML en production (si résultats positifs)
- ✅ Documentation complète

---

### SEMAINE 9-10 : Reinforcement Learning (Optionnel)

**Objectif** : Agent RL qui apprend la politique optimale

**⚠️ PRÉREQUIS** : ML production réussi + données suffisantes (>50,000 épisodes)

#### Tâches

**Semaine 9 : Implémentation RL** 🤖

- [ ] **Task 9.1** : Environnement Gym

  ```python
  import gym

  class DispatchEnv(gym.Env):
      """OpenAI Gym environment pour dispatch."""

      def __init__(self, historical_data):
          self.historical_data = historical_data
          self.action_space = gym.spaces.Discrete(200)  # 200 actions possibles
          self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(50,))

      def reset(self):
          # Charger un épisode aléatoire
          episode = random.choice(self.historical_data)
          self.state = episode["initial_state"]
          return self.state

      def step(self, action):
          # Simuler l'action
          next_state, reward = simulate_dispatch_action(self.state, action)
          done = len(self.state["unassigned_bookings"]) == 0
          return next_state, reward, done, {}
  ```

  **Effort** : 3 jours

- [ ] **Task 9.2** : Entraîner agent DQN

  - 1000 épisodes (replay historique)
  - Early stopping si plateau
  - Évaluer vs RandomForest

  **Effort** : 2 jours (GPU cloud recommandé)

**Semaine 10 : Évaluation RL** 📊

- [ ] **Task 10.1** : Validation agent RL

  - Test sur épisodes holdout
  - Comparer : RL vs ML vs Heuristics
  - Décision Go/No-Go pour RL production

  **Effort** : 2 jours

- [ ] **Task 10.2** : Documentation RL

  - Architecture agent
  - Reward function
  - Hyperparamètres

  **Effort** : 1 jour

**Livrable Semaines 9-10** :

- ✅ Agent RL entraîné (si résultats prometteurs)
- ✅ OU rapport d'échec (si RL < ML)

---

### SEMAINE 11-12 : Auto-Tuning & Polish

**Objectif** : Système qui s'améliore automatiquement

#### Tâches

**Semaine 11 : Auto-Tuning** 🎛️

- [ ] **Task 11.1** : Implémenter `DispatchAutoTuner`

  ```python
  class DispatchAutoTuner:
      def analyze_performance(self, days=7):
          metrics = get_metrics_last_n_days(self.company_id, days)
          return {
              "avg_quality_score": np.mean([m.quality_score for m in metrics]),
              "on_time_rate": ...,
              "fairness": ...,
          }

      def suggest_tuning(self, performance):
          suggestions = {}

          if performance["on_time_rate"] < 0.85:
              # Augmenter buffers
              suggestions["time.pickup_buffer_min"] = current + 2

          if performance["fairness"] < 0.7:
              # Augmenter poids équité
              suggestions["heuristic.driver_load_balance"] = current + 0.1

          return suggestions

      def apply_tuning(self, suggestions, dry_run=False):
          if not dry_run:
              # Update company.dispatch_settings
              ...
  ```

  **Effort** : 3 jours

- [ ] **Task 11.2** : Celery task hebdomadaire
  ```python
  @shared_task(name="tasks.dispatch_tasks.auto_tune_parameters")
  def auto_tune_parameters():
      for company in companies_with_low_quality():
          tuner = DispatchAutoTuner(company.id)
          performance = tuner.analyze_performance()
          suggestions = tuner.suggest_tuning(performance)
          tuner.apply_tuning(suggestions, dry_run=False)
  ```
  **Effort** : 1 jour

**Semaine 12 : Polish & Documentation** ✨

- [ ] **Task 12.1** : Documentation complète

  - API Reference (Swagger/OpenAPI)
  - Architecture diagrams (mise à jour)
  - Troubleshooting guide
  - Performance tuning guide

  **Effort** : 3 jours

- [ ] **Task 12.2** : Final polish

  - Corriger tous warnings linter
  - Update dependencies (security patches)
  - Final tests (regression suite)

  **Effort** : 2 jours

**Livrable Semaines 11-12** :

- ✅ Auto-tuning opérationnel
- ✅ Documentation complète
- ✅ Système prêt pour production à grande échelle

---

## 📊 MÉTRIQUES DE SUCCÈS

### Baseline (Actuel)

| KPI             | Valeur |
| --------------- | ------ |
| Quality Score   | 75/100 |
| On-Time Rate    | 82%    |
| Avg Delay       | 8 min  |
| Assignment Rate | 95%    |
| Solver Time     | 45s    |

### Objectif Post-ML (Semaine 8)

| KPI             | Valeur | Amélioration |
| --------------- | ------ | ------------ |
| Quality Score   | 85/100 | +10 pts      |
| On-Time Rate    | 90%    | +8%          |
| Avg Delay       | 5 min  | -3 min       |
| Assignment Rate | 98%    | +3%          |
| Solver Time     | 20s    | -25s         |

### Objectif Post-RL (Semaine 12+)

| KPI             | Valeur | Amélioration |
| --------------- | ------ | ------------ |
| Quality Score   | 92/100 | +17 pts      |
| On-Time Rate    | 95%    | +13%         |
| Avg Delay       | 3 min  | -5 min       |
| Assignment Rate | 99%    | +4%          |
| Solver Time     | 10s    | -35s         |

---

## 💰 BUDGET & RESSOURCES

### Équipe

| Rôle               | Durée                       | Coût                             |
| ------------------ | --------------------------- | -------------------------------- |
| **Dev Senior**     | 12 semaines                 | 15,000€/mois × 3 = **45,000€**   |
| **Data Scientist** | 6 semaines (temps partiel)  | 17,000€/mois × 1.5 = **25,500€** |
| **DevOps**         | 2 semaines (setup infra ML) | 12,000€/mois × 0.5 = **6,000€**  |
| **Total**          |                             | **76,500€**                      |

### Infrastructure

| Ressource                       | Coût Mensuel | Durée  | Total      |
| ------------------------------- | ------------ | ------ | ---------- |
| **GPU Cloud** (entraînement ML) | 500€/mois    | 3 mois | **1,500€** |
| **Redis Cluster** (upgrade)     | 200€/mois    | 3 mois | **600€**   |
| **Monitoring** (Datadog)        | 300€/mois    | 3 mois | **900€**   |
| **Total**                       |              |        | **3,000€** |

### Budget Total : **79,500 €**

### ROI Estimé

**Gains Année 1** : 4,450,000 €  
**ROI** : (4,450,000 - 79,500) / 79,500 = **5,495%** 🚀

---

## ✅ CHECKLIST FINALE

### Avant Production ML

- [ ] Modèle entraîné (MAE <5 min, R² >0.70)
- [ ] A/B test réussi (amélioration >5%, p<0.05)
- [ ] Safety limits implémentés + testés
- [ ] Audit trail opérationnel (table AutonomousAction)
- [ ] Monitoring actif (Grafana dashboards)
- [ ] Documentation complète (API + architecture)
- [ ] Tests coverage >80% modules critiques
- [ ] Rollback plan documenté et testé
- [ ] Équipe formée (runbook reviewed)
- [ ] Clients pilotes informés

---

**FIN DU PLAN D'ACTION**
