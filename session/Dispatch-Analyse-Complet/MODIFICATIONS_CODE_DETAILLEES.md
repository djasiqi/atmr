# ⚙️ MODIFICATIONS CODE DÉTAILLÉES

**Guide pratique** : Modifications exactes à apporter au code, ligne par ligne

---

## 📦 MODIFICATIONS PAR FICHIER

### 1. BACKEND - Models

#### 1.1 `backend/models/dispatch.py`

**Ajouter** : Tables MLPrediction et AutonomousAction

**Ligne 610** (après classe `DailyStats`) :

```python
class MLPrediction(db.Model):
    """Stocke les prédictions ML pour feedback loop et monitoring."""
    __tablename__ = "ml_prediction"
    __table_args__ = (
        Index('idx_ml_prediction_assignment', 'assignment_id'),
        Index('idx_ml_prediction_risk_date', 'risk_level', 'created_at'),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    assignment_id: Mapped[int] = mapped_column(
        ForeignKey('assignment.id', ondelete="CASCADE"),
        nullable=False
    )

    # Prédiction
    predicted_delay_minutes: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)  # 0.0-1.0
    risk_level: Mapped[str] = mapped_column(String(20), nullable=False)  # low, medium, high, critical

    # Features utilisées (pour reproductibilité et debugging)
    feature_vector: Mapped[Dict[str, Any]] = mapped_column(MutableDict.as_mutable(JSONB()), nullable=False)

    # Résultat réel (rempli après coup par Celery task)
    actual_delay_minutes: Mapped[float | None] = mapped_column(Float, nullable=True)
    prediction_error: Mapped[float | None] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        server_default=func.now(),
        nullable=False
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        onupdate=lambda: datetime.now(UTC)
    )

    # Relations
    assignment: Mapped[Assignment] = relationship("Assignment", backref="ml_predictions")

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'assignment_id': self.assignment_id,
            'predicted_delay_minutes': round(self.predicted_delay_minutes, 2),
            'confidence': round(self.confidence, 3),
            'risk_level': self.risk_level,
            'actual_delay_minutes': round(self.actual_delay_minutes, 2) if self.actual_delay_minutes else None,
            'prediction_error': round(self.prediction_error, 2) if self.prediction_error else None,
            'created_at': _iso(self.created_at),
            'updated_at': _iso(self.updated_at),
        }


class AutonomousAction(db.Model):
    """Trace toutes les actions automatiques effectuées par le système."""
    __tablename__ = "autonomous_action"
    __table_args__ = (
        Index('idx_autonomous_action_company_time', 'company_id', 'applied_at'),
        Index('idx_autonomous_action_type', 'action_type'),
        Index('idx_autonomous_action_success', 'success', 'applied_at'),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int] = mapped_column(ForeignKey('company.id', ondelete="CASCADE"), nullable=False)

    # Type d'action
    action_type: Mapped[str] = mapped_column(String(50), nullable=False)  # reassign, notify_customer, adjust_time
    entity_type: Mapped[str] = mapped_column(String(50), nullable=False)  # assignment, booking, driver
    entity_id: Mapped[int] = mapped_column(Integer, nullable=False)

    # Contexte de la décision
    trigger_reason: Mapped[str] = mapped_column(String(200), nullable=False)
    decision_context: Mapped[Dict[str, Any]] = mapped_column(JSONB(), nullable=False)

    # Résultat
    applied_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False
    )
    success: Mapped[bool] = mapped_column(Boolean, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Traçabilité ML
    ml_prediction_id: Mapped[int | None] = mapped_column(
        ForeignKey('ml_prediction.id', ondelete="SET NULL"),
        nullable=True
    )
    confidence_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Impact mesuré (rempli après coup pour feedback)
    actual_impact_minutes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    quality_improvement: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Relations
    company: Mapped[Company] = relationship("Company", backref="autonomous_actions")
    ml_prediction: Mapped[MLPrediction | None] = relationship("MLPrediction", backref="autonomous_actions")

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'company_id': self.company_id,
            'action_type': self.action_type,
            'entity_type': self.entity_type,
            'entity_id': self.entity_id,
            'trigger_reason': self.trigger_reason,
            'applied_at': _iso(self.applied_at),
            'success': self.success,
            'error_message': self.error_message,
            'confidence_score': round(self.confidence_score, 3) if self.confidence_score else None,
        }
```

#### 1.2 Migration Alembic

**Créer** : `backend/migrations/versions/xxx_add_ml_tables.py`

```bash
cd backend
alembic revision -m "add_ml_prediction_and_autonomous_action_tables"
```

**Fichier généré** : Editer avec :

```python
def upgrade():
    # Table ml_prediction
    op.create_table(
        'ml_prediction',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('assignment_id', sa.Integer(), nullable=False),
        sa.Column('predicted_delay_minutes', sa.Float(), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('risk_level', sa.String(length=20), nullable=False),
        sa.Column('feature_vector', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('actual_delay_minutes', sa.Float(), nullable=True),
        sa.Column('prediction_error', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['assignment_id'], ['assignment.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_ml_prediction_assignment', 'ml_prediction', ['assignment_id'])
    op.create_index('idx_ml_prediction_risk_date', 'ml_prediction', ['risk_level', 'created_at'])

    # Table autonomous_action
    op.create_table(
        'autonomous_action',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('company_id', sa.Integer(), nullable=False),
        sa.Column('action_type', sa.String(length=50), nullable=False),
        sa.Column('entity_type', sa.String(length=50), nullable=False),
        sa.Column('entity_id', sa.Integer(), nullable=False),
        sa.Column('trigger_reason', sa.String(length=200), nullable=False),
        sa.Column('decision_context', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('applied_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('success', sa.Boolean(), nullable=False),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('ml_prediction_id', sa.Integer(), nullable=True),
        sa.Column('confidence_score', sa.Float(), nullable=True),
        sa.Column('actual_impact_minutes', sa.Integer(), nullable=True),
        sa.Column('quality_improvement', sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(['company_id'], ['company.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['ml_prediction_id'], ['ml_prediction.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_autonomous_action_company_time', 'autonomous_action', ['company_id', 'applied_at'])
    op.create_index('idx_autonomous_action_type', 'autonomous_action', ['action_type'])
    op.create_index('idx_autonomous_action_success', 'autonomous_action', ['success', 'applied_at'])

def downgrade():
    op.drop_table('autonomous_action')
    op.drop_table('ml_prediction')
```

**Exécuter** :

```bash
alembic upgrade head
```

---

### 2. BACKEND - Services

#### 2.1 `backend/services/unified_dispatch/settings.py`

**Ligne 150** (après `class LoggingSettings`) : Ajouter

```python
@dataclass
class MLSettings:
    """Configuration du système ML."""
    enabled: bool = False  # Désactivé par défaut (feature flag)
    model_path: str = "backend/data/ml_models/delay_predictor.pkl"

    # Seuils de réoptimisation
    reoptimize_threshold_minutes: int = 10  # Si prédit >10 min retard → chercher meilleur driver
    min_confidence: float = 0.70            # Confiance minimale pour déclencher réoptimization
    min_gain_minutes: int = 5               # Gain minimal pour valoir la peine de réassigner

    # Feedback loop
    update_actuals_daily: bool = True       # Update actual_delay_minutes chaque nuit
    retrain_weekly: bool = False            # Réentraîner modèle chaque semaine
    min_samples_retrain: int = 1000         # Minimum échantillons pour retrain valide
```

**Ligne 153** (dans `class Settings`) : Ajouter champ

```python
@dataclass
class Settings:
    # ... (existing fields)
    ml: MLSettings = field(default_factory=MLSettings)  # ← AJOUTER
```

#### 2.2 `backend/services/unified_dispatch/engine.py`

**Ligne 583** (avant `_apply_and_emit`) : Insérer

```python
    # ============================================================
    # 6.5) ML PREDICTION & RE-OPTIMIZATION (NOUVEAU)
    # ============================================================

    ml_predictions: List[Tuple[Any, Any]] = []  # (assignment, MLPrediction)

    if settings.features.enable_ml_predictions and getattr(settings, "ml", None) and settings.ml.enabled:
        logger.info("[Engine] 🤖 ML Prediction enabled, analyzing %d assignments...", len(final_assignments))

        from services.unified_dispatch.ml_predictor import get_ml_predictor
        from models.dispatch import MLPrediction

        try:
            ml_predictor = get_ml_predictor()

            # Vérifier que le modèle est entraîné
            if not ml_predictor.is_trained:
                logger.warning("[ML] Model not trained, skipping ML predictions")
            else:
                bookings_map = {b.id: b for b in problem.get("bookings", [])}
                drivers_map = {d.id: d for d in problem.get("drivers", [])}

                risky_count = 0
                reassigned_count = 0

                for assignment in final_assignments:
                    booking = bookings_map.get(assignment.booking_id)
                    driver = drivers_map.get(assignment.driver_id)

                    if not booking or not driver:
                        continue

                    # Prédire le retard
                    prediction = ml_predictor.predict_delay(booking, driver)

                    # Créer objet MLPrediction (sera sauvegardé après apply)
                    ml_pred = MLPrediction(
                        assignment_id=None,  # Rempli après création Assignment
                        predicted_delay_minutes=prediction.predicted_delay_minutes,
                        confidence=prediction.confidence,
                        risk_level=prediction.risk_level,
                        feature_vector=ml_predictor.extract_features(booking, driver),
                        actual_delay_minutes=None,  # Rempli le lendemain par Celery task
                        prediction_error=None,
                    )
                    ml_predictions.append((assignment, ml_pred))

                    # Si retard prédit > seuil ET confiance élevée → chercher meilleur driver
                    if (prediction.predicted_delay_minutes > settings.ml.reoptimize_threshold_minutes and
                        prediction.confidence >= settings.ml.min_confidence):

                        risky_count += 1

                        logger.warning(
                            "[ML] ⚠️ High risk: booking=%s driver=%s predicted_delay=%d min (conf=%.2f)",
                            booking.id, driver.id,
                            prediction.predicted_delay_minutes,
                            prediction.confidence
                        )

                        # Chercher meilleur chauffeur
                        better_driver = _find_better_driver_ml(
                            booking,
                            driver,
                            list(drivers_map.values()),
                            ml_predictor,
                            settings
                        )

                        if better_driver:
                            # Calculer gain
                            new_prediction = ml_predictor.predict_delay(booking, better_driver)
                            gain = prediction.predicted_delay_minutes - new_prediction.predicted_delay_minutes

                            if gain >= settings.ml.min_gain_minutes:
                                # Réassigner
                                assignment.driver_id = better_driver.id
                                reassigned_count += 1

                                logger.info(
                                    "[ML] ✅ Reassigned booking=%s: driver %s→%s (gain=%d min)",
                                    booking.id, driver.id, better_driver.id, gain
                                )

                if risky_count > 0:
                    logger.info(
                        "[ML] Summary: %d risky assignments detected, %d reassigned",
                        risky_count, reassigned_count
                    )

        except Exception as e:
            logger.exception("[ML] ML prediction failed, continuing without ML: %s", e)

    # ============================================================
    # 7) Application en DB (EXISTANT - INCHANGÉ)
    # ============================================================
```

**Ligne 815** (après `apply_assignments`) : Ajouter

```python
    # ============================================================
    # 7.5) SAVE ML PREDICTIONS (NOUVEAU)
    # ============================================================

    if ml_predictions:
        try:
            # Récupérer les IDs des assignments créés
            booking_to_assignment_id = {}
            for a in final_assignments:
                db_assignment = Assignment.query.filter_by(
                    booking_id=a.booking_id,
                    dispatch_run_id=drid
                ).first()
                if db_assignment:
                    booking_to_assignment_id[a.booking_id] = db_assignment.id

            # Sauvegarder les prédictions
            for assignment_obj, ml_pred in ml_predictions:
                assignment_id = booking_to_assignment_id.get(assignment_obj.booking_id)
                if assignment_id:
                    ml_pred.assignment_id = assignment_id
                    db.session.add(ml_pred)

            db.session.commit()
            logger.info("[ML] Saved %d ML predictions to DB", len(ml_predictions))

        except Exception as e:
            logger.exception("[ML] Failed to save ML predictions: %s", e)
            db.session.rollback()
```

**À la fin du fichier** (ligne 951+) : Ajouter fonction helper

```python
def _find_better_driver_ml(
    booking: Any,
    current_driver: Any,
    all_drivers: List[Any],
    ml_predictor: Any,
    settings: Settings
) -> Any | None:
    """
    Cherche un chauffeur avec meilleure prédiction ML.

    Args:
        booking: Booking à assigner
        current_driver: Driver actuel
        all_drivers: Liste de tous les drivers disponibles
        ml_predictor: Instance de DelayMLPredictor
        settings: Settings du dispatch

    Returns:
        Meilleur Driver ou None
    """
    current_prediction = ml_predictor.predict_delay(booking, current_driver)

    candidates = []

    for driver in all_drivers:
        # Skip current driver
        d_id = int(cast(Any, getattr(driver, "id", 0)))
        c_id = int(cast(Any, getattr(current_driver, "id", 0)))
        if d_id == c_id:
            continue

        # Vérifier disponibilité basique
        is_active = bool(getattr(driver, "is_active", False))
        is_available = bool(getattr(driver, "is_available", False))
        if not is_active or not is_available:
            continue

        # Prédire avec ce driver
        try:
            prediction = ml_predictor.predict_delay(booking, driver)

            # Calculer gain potentiel
            gain = current_prediction.predicted_delay_minutes - prediction.predicted_delay_minutes

            if gain > 0:
                candidates.append((driver, gain, prediction))

        except Exception as e:
            logger.warning("[ML] Failed to predict for driver %s: %s", d_id, e)
            continue

    if not candidates:
        return None

    # Trier par gain décroissant
    candidates.sort(key=lambda x: x[1], reverse=True)

    # Retourner le meilleur
    best_driver, gain, prediction = candidates[0]

    logger.info(
        "[ML] Best alternative: driver=%s gain=%d min (predicted=%d, conf=%.2f)",
        getattr(best_driver, "id", None),
        gain,
        prediction.predicted_delay_minutes,
        prediction.confidence
    )

    return best_driver
```

---

#### 2.3 `backend/services/unified_dispatch/autonomous_manager.py`

**Ligne 160** (remplacer la fonction `check_safety_limits`) :

```python
    def check_safety_limits(self, action_type: str) -> tuple[bool, str]:
        """
        Vérifie que les limites de sécurité ne sont pas dépassées.

        Args:
            action_type: Type d'action ('notify', 'reassign', 'adjust_time')

        Returns:
            Tuple (can_proceed, reason)
        """
        from datetime import timedelta
        from models.dispatch import AutonomousAction

        try:
            limits = self.config["safety_limits"]

            # 1. Rate limiting (actions/heure)
            one_hour_ago = now_local() - timedelta(hours=1)
            recent_count = AutonomousAction.query.filter(
                AutonomousAction.company_id == self.company_id,
                AutonomousAction.action_type == action_type,
                AutonomousAction.applied_at >= one_hour_ago
            ).count()

            max_per_hour = limits.get("max_auto_actions_per_hour", 50)
            if recent_count >= max_per_hour:
                return False, f"Rate limit exceeded: {recent_count}/{max_per_hour} actions in last hour"

            # 2. Daily limits (réassignations uniquement)
            if action_type == "reassign":
                today_start = now_local().replace(hour=0, minute=0, second=0, microsecond=0)
                today_count = AutonomousAction.query.filter(
                    AutonomousAction.company_id == self.company_id,
                    AutonomousAction.action_type == "reassign",
                    AutonomousAction.applied_at >= today_start
                ).count()

                max_per_day = limits.get("max_auto_reassignments_per_day", 10)
                if today_count >= max_per_day:
                    return False, f"Daily limit exceeded: {today_count}/{max_per_day} reassignments today"

            # 3. Consecutive failures check (si >3 échecs consécutifs → pause)
            last_5_actions = AutonomousAction.query.filter(
                AutonomousAction.company_id == self.company_id,
                AutonomousAction.action_type == action_type
            ).order_by(AutonomousAction.applied_at.desc()).limit(5).all()

            if len(last_5_actions) >= 5:
                consecutive_failures = all(not a.success for a in last_5_actions)
                if consecutive_failures:
                    return False, "Too many consecutive failures (5), system paused for safety"

            return True, "OK"

        except Exception as e:
            logger.exception("[AutonomousManager] Error checking safety limits: %s", e)
            # En cas d'erreur, refuser l'action par sécurité
            return False, f"Safety check failed: {e}"
```

**Ligne 230** (dans `apply_suggestion`) : Ajouter logging

```python
                # Appliquer la suggestion
                try:
                    if not dry_run:
                        result = apply_suggestion(suggestion, self.company_id, dry_run=False)

                        # ✨ NOUVEAU : Logger dans AutonomousAction
                        from models.dispatch import AutonomousAction

                        action_log = AutonomousAction(
                            company_id=self.company_id,
                            action_type=suggestion.action,
                            entity_type="assignment" if suggestion.booking_id else "unknown",
                            entity_id=suggestion.booking_id or 0,
                            trigger_reason=f"opportunity_detected_{opportunity.severity}",
                            decision_context={
                                "suggestion": suggestion.to_dict(),
                                "opportunity": opportunity.to_dict(),
                                "mode": self.mode.value,
                                "config": self.config,
                            },
                            applied_at=now_local(),
                            success=result.get("success", False),
                            error_message=result.get("error"),
                        )
                        db.session.add(action_log)
                        db.session.commit()
                        # ✨ FIN NOUVEAU

                        if result.get("success"):
                            stats["auto_applied"] += 1
                            # ... (reste du code existant)
```

---

### 3. BACKEND - Tasks Celery

#### 3.1 Créer `backend/tasks/ml_tasks.py` (NOUVEAU FICHIER)

```python
"""
Tâches Celery pour le système Machine Learning.
Gère l'entraînement, le feedback loop et la maintenance des modèles.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any, Dict

import numpy as np
from celery import shared_task
from sklearn.metrics import mean_absolute_error, r2_score

from ext import db
from models import Assignment, Booking, BookingStatus, MLPrediction
from services.unified_dispatch.ml_predictor import get_ml_predictor

logger = logging.getLogger(__name__)


@shared_task(
    name="tasks.ml_tasks.update_ml_predictions_actuals",
    acks_late=True,
    task_time_limit=600  # 10 min max
)
def update_ml_predictions_actuals() -> Dict[str, Any]:
    """
    Tâche nocturne : calcule les retards réels et met à jour les prédictions ML.
    Permet de mesurer la performance du modèle en conditions réelles.

    S'exécute tous les jours à 2h du matin (configuré dans Celery Beat).
    """
    yesterday = datetime.now(UTC).date() - timedelta(days=1)
    yesterday_start = datetime.combine(yesterday, datetime.min.time())
    yesterday_end = yesterday_start + timedelta(days=1)

    logger.info("[ML] Starting nightly update of actual delays for %s", yesterday)

    # Récupérer toutes les prédictions d'hier sans actual
    predictions = (
        MLPrediction.query
        .join(Assignment, Assignment.id == MLPrediction.assignment_id)
        .join(Booking, Booking.id == Assignment.booking_id)
        .filter(
            MLPrediction.actual_delay_minutes.is_(None),
            Booking.completed_at >= yesterday_start,
            Booking.completed_at < yesterday_end,
            Booking.status == BookingStatus.COMPLETED,
            Assignment.actual_pickup_at.isnot(None)
        )
        .all()
    )

    logger.info("[ML] Found %d predictions to update", len(predictions))

    updated_count = 0

    for pred in predictions:
        try:
            assignment = pred.assignment
            booking = assignment.booking

            # Calculer retard réel
            actual_delay_seconds = (
                assignment.actual_pickup_at - booking.scheduled_time
            ).total_seconds()
            actual_delay_minutes = actual_delay_seconds / 60.0

            # Mettre à jour
            pred.actual_delay_minutes = actual_delay_minutes
            pred.prediction_error = abs(actual_delay_minutes - pred.predicted_delay_minutes)
            pred.updated_at = datetime.now(UTC)

            db.session.add(pred)
            updated_count += 1

        except Exception as e:
            logger.warning("[ML] Failed to update prediction %s: %s", pred.id, e)
            continue

    db.session.commit()
    logger.info("[ML] Updated %d predictions with actual delays", updated_count)

    # Calculer métriques globales (derniers 7 jours)
    seven_days_ago = datetime.now(UTC) - timedelta(days=7)
    recent_predictions = (
        MLPrediction.query
        .filter(
            MLPrediction.actual_delay_minutes.isnot(None),
            MLPrediction.created_at >= seven_days_ago
        )
        .all()
    )

    mae = None
    r2 = None

    if recent_predictions:
        y_true = [p.actual_delay_minutes for p in recent_predictions]
        y_pred = [p.predicted_delay_minutes for p in recent_predictions]

        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        logger.info(
            "[ML] Model performance (last 7 days): "
            "MAE=%.2f min, R²=%.3f, samples=%d",
            mae, r2, len(recent_predictions)
        )

        # Alerter si dégradation
        if mae > 8.0:
            logger.error(
                "[ML] ⚠️ Model performance DEGRADED! MAE=%.2f (target: <5 min)",
                mae
            )
            _notify_admin_ml_degradation(mae, r2)

        elif mae > 6.0:
            logger.warning(
                "[ML] ⚠️ Model performance declining. MAE=%.2f (target: <5 min)",
                mae
            )

    return {
        "date": yesterday.isoformat(),
        "updated_count": updated_count,
        "mae": float(mae) if mae else None,
        "r2_score": float(r2) if r2 else None,
        "samples": len(recent_predictions),
    }


@shared_task(
    name="tasks.ml_tasks.retrain_model_weekly",
    acks_late=True,
    task_time_limit=3600  # 1h max
)
def retrain_model_weekly() -> Dict[str, Any]:
    """
    Tâche hebdomadaire : réentraîne le modèle ML sur les 30 derniers jours.
    Compare avec l'ancien modèle et déploie si meilleur.

    S'exécute tous les lundis à 3h du matin.
    """
    logger.info("[ML] Starting weekly model retraining...")

    # Collecter données des 30 derniers jours
    end_date = datetime.now(UTC)
    start_date = end_date - timedelta(days=30)

    from scripts.ml.collect_training_data import collect_historical_data

    training_data = collect_historical_data(
        start_date,
        end_date,
        output_file="backend/data/ml_datasets/training_data_weekly.json"
    )

    if len(training_data) < 1000:
        logger.warning(
            "[ML] Not enough data for retraining: %d samples (min: 1000)",
            len(training_data)
        )
        return {
            "retrained": False,
            "reason": "insufficient_data",
            "samples": len(training_data)
        }

    # Entraîner nouveau modèle
    from services.unified_dispatch.ml_predictor import DelayMLPredictor

    new_predictor = DelayMLPredictor(
        model_path="backend/data/ml_models/delay_predictor_new.pkl"
    )

    new_metrics = new_predictor.train_on_historical_data(
        training_data,
        save_model=True
    )

    # Charger ancien modèle
    old_predictor = get_ml_predictor()

    # Comparer sur validation set (7 derniers jours)
    val_start = end_date - timedelta(days=7)
    val_data = collect_historical_data(val_start, end_date)

    def evaluate_model(predictor, data):
        """Évalue MAE sur validation set."""
        y_true = [d["actual_delay_minutes"] for d in data]
        y_pred = [
            predictor.predict_delay(
                d["booking"], d["driver"]
            ).predicted_delay_minutes
            for d in data
        ]
        return mean_absolute_error(y_true, y_pred)

    new_mae = evaluate_model(new_predictor, val_data)
    old_mae = evaluate_model(old_predictor, val_data)

    logger.info(
        "[ML] Validation results: New MAE=%.2f, Old MAE=%.2f",
        new_mae, old_mae
    )

    # Déployer si meilleur
    deployed = False
    if new_mae < old_mae:
        import os
        import shutil

        # Backup ancien modèle
        shutil.copy(
            "backend/data/ml_models/delay_predictor.pkl",
            "backend/data/ml_models/delay_predictor_backup.pkl"
        )

        # Déployer nouveau
        shutil.move(
            "backend/data/ml_models/delay_predictor_new.pkl",
            "backend/data/ml_models/delay_predictor.pkl"
        )

        deployed = True
        logger.info("[ML] ✅ New model deployed! Improvement: %.2f min", old_mae - new_mae)

    else:
        logger.info("[ML] ⏸️ Old model kept (better performance)")

    return {
        "retrained": True,
        "deployed": deployed,
        "new_mae": float(new_mae),
        "old_mae": float(old_mae),
        "improvement": float(old_mae - new_mae),
        "training_samples": len(training_data),
        "validation_samples": len(val_data),
    }


def _notify_admin_ml_degradation(mae: float, r2: float):
    """Notifie les admins si le modèle ML dégrade."""
    from services.notification_service import notify_admin

    message = (
        f"⚠️ ALERT: ML model performance degraded!\n"
        f"MAE: {mae:.2f} min (target: <5 min)\n"
        f"R² score: {r2:.3f} (target: >0.70)\n"
        f"Action: Manual review recommended."
    )

    notify_admin(
        title="ML Model Degradation",
        message=message,
        severity="high"
    )
```

#### 3.2 `backend/celery_app.py`

**Ajouter** dans `beat_schedule` :

```python
from celery.schedules import crontab

app.conf.beat_schedule = {
    # ... (existing tasks: autorun_tick, realtime_monitoring_tick)

    # ✨ NOUVEAU : Update ML predictions actuals (nightly)
    'update-ml-predictions-actuals': {
        'task': 'tasks.ml_tasks.update_ml_predictions_actuals',
        'schedule': crontab(hour=2, minute=0),  # Tous les jours à 2h du matin
    },

    # ✨ NOUVEAU : Retrain ML model (weekly)
    'retrain-ml-model-weekly': {
        'task': 'tasks.ml_tasks.retrain_model_weekly',
        'schedule': crontab(day_of_week=1, hour=3, minute=0),  # Lundi 3h
        'options': {'queue': 'ml_queue'}  # Queue dédiée pour jobs longs
    },
}
```

---

### 4. SCRIPTS

#### 4.1 Créer `backend/scripts/ml/collect_training_data.py` (NOUVEAU)

**Contenu** : Voir section complète dans `IMPLEMENTATION_ML_RL_GUIDE.md` section 1.1

**Utilisation** :

```bash
cd backend
python scripts/ml/collect_training_data.py
```

**Output** :

- `backend/data/ml_datasets/training_data.json`
- `backend/data/ml_datasets/training_data.csv`
- `backend/data/ml_datasets/data_report.html`

---

### 5. FRONTEND - Configuration

#### 5.1 Créer `frontend/src/services/mlService.js` (NOUVEAU)

```javascript
/**
 * Service pour les fonctionnalités ML.
 */
import apiClient from "./apiClient";

export const mlService = {
  /**
   * Récupère les statistiques du modèle ML.
   */
  async getMLStats() {
    const response = await apiClient.get("/api/ml/stats");
    return response.data;
  },

  /**
   * Récupère l'accuracy du modèle (7 derniers jours).
   */
  async getAccuracy() {
    const response = await apiClient.get("/api/ml/predictions/accuracy");
    return response.data;
  },

  /**
   * Active/désactive le ML pour l'entreprise.
   */
  async toggleML(enabled) {
    const response = await apiClient.post("/api/admin/ml/toggle", { enabled });
    return response.data;
  },
};
```

---

## 🎯 ORDRE D'EXÉCUTION

### Sprint 1 (Semaine 1-2) : Foundation

```bash
# Jour 1-2 : Cleanup
git checkout -b feature/cleanup-code-debt
rm backend/Classeur1.xlsx backend/transport.xlsx backend/check_bookings.py
# ... (refactoring Haversine, schemas, etc.)
git commit -m "chore: cleanup code debt and refactor duplications"

# Jour 3-4 : SQL optimizations
git checkout -b feature/sql-optimizations
# ... (bulk inserts, indexes)
alembic revision -m "add_performance_indexes"
alembic upgrade head
git commit -m "perf: optimize SQL queries with bulk ops and indexes"

# Jour 5-10 : Tests
git checkout -b feature/unit-tests
# ... (pytest, factories, tests)
git commit -m "test: add unit tests for critical modules (70% coverage)"

# Review + Merge
git checkout main
git merge feature/cleanup-code-debt
git merge feature/sql-optimizations
git merge feature/unit-tests
git push origin main
```

### Sprint 2 (Semaine 3-4) : ML POC

```bash
# Jour 11-15 : Data collection & training
git checkout -b feature/ml-poc
# ... (collect_training_data.py, training, evaluation)
git commit -m "feat: ML POC - RandomForest delay predictor (MAE=4.2, R²=0.76)"

# Jour 16-20 : Validation & documentation
# ... (cross-validation, feature importance)
git commit -m "docs: ML POC validation report and feature analysis"

# Decision: GO/NO-GO
# Si GO → continuer Sprint 3
# Si NO-GO → analyser causes, retry avec plus de données
```

### Sprint 3 (Semaine 5-6) : ML Production

```bash
# Jour 21-25 : Safety + Integration
git checkout -b feature/ml-production
alembic revision -m "add_ml_prediction_and_autonomous_action_tables"
alembic upgrade head
# ... (safety limits, engine.py modifications)
git commit -m "feat: integrate ML in dispatch pipeline with safety limits"

# Jour 26-30 : Monitoring & testing
# ... (Celery tasks, dashboard, tests)
git commit -m "feat: ML monitoring and feedback loop"

# Merge to main
git checkout main
git merge feature/ml-production
git push origin main
```

### Sprint 4 (Semaine 7-8) : A/B Testing & Rollout

```bash
# Semaine 7 : A/B test
# ... (split companies, monitor metrics)

# Semaine 8 : Analysis & deployment
# Si résultats positifs → activer ML pour tous
UPDATE company SET dispatch_settings =
    jsonb_set(dispatch_settings::jsonb, '{ml,enabled}', 'true')
WHERE dispatch_enabled = true;
```

---

## 🔧 COMMANDES UTILES

### Tests

```bash
# Tests unitaires
pytest tests/ -v

# Tests avec coverage
pytest tests/ --cov=backend --cov-report=html

# Tests d'un seul module
pytest tests/test_engine.py -v

# Tests avec logs
pytest tests/ -v -s --log-cli-level=INFO
```

### Database

```bash
# Créer migration
alembic revision -m "description"

# Appliquer migrations
alembic upgrade head

# Rollback dernière migration
alembic downgrade -1

# Voir historique
alembic history
```

### Celery

```bash
# Lancer worker (dev)
celery -A celery_app worker --loglevel=info

# Lancer beat (scheduler)
celery -A celery_app beat --loglevel=info

# Lancer worker + beat ensemble
celery -A celery_app worker --beat --loglevel=info

# Purge queue
celery -A celery_app purge

# Inspect tasks
celery -A celery_app inspect active
```

### ML

```bash
# Collecter données d'entraînement
python backend/scripts/ml/collect_training_data.py

# Entraîner modèle
python backend/scripts/ml/train_model.py

# Évaluer modèle
python backend/scripts/ml/evaluate_model.py

# Analyser feature importance
python backend/scripts/ml/analyze_features.py
```

---

## 📝 COMMIT MESSAGES RECOMMANDÉS

**Format** : `<type>(<scope>): <subject>`

**Types** :

- `feat` : Nouvelle fonctionnalité
- `fix` : Correction bug
- `perf` : Amélioration performance
- `refactor` : Refactoring (sans changement fonctionnel)
- `test` : Ajout/modification tests
- `docs` : Documentation
- `chore` : Tâches maintenance (deps, config)

**Exemples** :

```bash
git commit -m "feat(ml): integrate RandomForest delay predictor in dispatch pipeline"
git commit -m "perf(sql): optimize queries with bulk inserts and indexes"
git commit -m "fix(safety): implement rate limiting in autonomous manager"
git commit -m "test(engine): add unit tests for dispatch phases (70% coverage)"
git commit -m "docs(ml): add ML integration guide and API reference"
```

---

## ✅ VALIDATION CHECKLIST

### Avant chaque commit

- [ ] Code lint (ruff) : `ruff check backend/`
- [ ] Type check (mypy) : `mypy backend/`
- [ ] Tests passent : `pytest tests/ -v`
- [ ] Pas de secrets hardcodés : `git secrets --scan`
- [ ] Commit message suit convention

### Avant chaque merge to main

- [ ] Review code (1+ reviewer)
- [ ] Tests CI passent (GitHub Actions)
- [ ] Coverage ≥ 70%
- [ ] Documentation mise à jour
- [ ] Changelog.md mis à jour

### Avant déploiement production

- [ ] Tests E2E passent
- [ ] Load test réussi (100 req/s pendant 10 min)
- [ ] Rollback plan documenté et testé
- [ ] Monitoring dashboards opérationnels
- [ ] Équipe formée sur nouvelles features
- [ ] Clients pilotes informés (si breaking changes)

---

**FIN DES MODIFICATIONS DÉTAILLÉES**

Avec ce guide, vous avez **toutes les modifications ligne par ligne** à apporter.  
Prêt à implémenter ? 🚀
