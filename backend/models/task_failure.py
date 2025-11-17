"""Model TaskFailure pour stocker les tâches Celery échouées (DLQ)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Dict

from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB

from ext import db


class TaskFailure(db.Model):
    """Stocke les métadonnées des tâches Celery échouées pour observabilité DLQ.
    
    A3: Amélioration pour empêcher le blocage des files et tracer les échecs.
    """
    
    __tablename__ = "task_failure"
    
    id = Column(Integer, primary_key=True)
    task_id = Column(String(255), unique=True, nullable=False, index=True)
    task_name = Column(String(255), nullable=False, index=True)
    
    # Métadonnées de l'erreur
    exception = Column(Text, nullable=False)
    traceback = Column(Text, nullable=True)
    args = Column(String(2000), nullable=True)  # Limité pour éviter trop long
    kwargs = Column(JSONB, nullable=True)
    
    # Timestamps
    first_seen = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC))
    last_seen = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC))
    
    # Compteur d'échecs
    failure_count = Column(Integer, nullable=False, default=1)
    
    # Métadonnées additionnelles
    worker_name = Column(String(255), nullable=True)
    hostname = Column(String(255), nullable=True)
    
    # Relation optionnelle avec dispatch_run si applicable
    dispatch_run_id = Column(Integer, nullable=True, index=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire pour export."""
        return {
            "id": self.id,
            "task_id": self.task_id,
            "task_name": self.task_name,
            "exception": self.exception[:500],  # Tronquer pour affichage
            "traceback": self.traceback[:500] if self.traceback else None,
            "args": self.args,
            "kwargs": self.kwargs,
            "first_seen": self.first_seen.isoformat() if self.first_seen else None,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "failure_count": self.failure_count,
            "worker_name": self.worker_name,
            "hostname": self.hostname,
            "dispatch_run_id": self.dispatch_run_id,
        }
    
    def __repr__(self):
        return f"<TaskFailure task_id={self.task_id} task_name={self.task_name} failures={self.failure_count}>"


