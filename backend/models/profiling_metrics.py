"""✅ 3.4: Modèle pour stocker les métriques de profiling automatique.

Permet de tracker l'évolution des performances et identifier les fonctions chaudes
sur plusieurs semaines/mois.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import BigInteger, DateTime, Float, Integer, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from ext import db


class ProfilingMetrics(db.Model):
    """✅ 3.4: Métriques de profiling hebdomadaire.

    Stocke les résultats du profiling automatique pour analyse historique
    et identification des fonctions chaudes récurrentes.
    """

    __tablename__ = "profiling_metrics"

    # Clé primaire
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)

    # Date/heure du profiling
    profiling_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), index=True
    )

    # Durée du profiling (secondes)
    duration_seconds: Mapped[float] = mapped_column(Float, nullable=False)

    # Nombre de requêtes effectuées pendant le profiling
    request_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Top N fonctions chaudes (JSON)
    top_functions = db.Column(JSONB, nullable=True)
    """
    Format:
    [
        {
            "function": "function_name",
            "file": "filename.py",
            "ncalls": "100",
            "tottime": 1.234,
            "cumtime": 2.345,
            "raw": "..."
        },
        ...
    ]
    """

    # Métriques système avant profiling (JSON)
    system_metrics_before = db.Column(JSONB, nullable=True)
    """
    Format:
    {
        "cpu_percent": 25.5,
        "memory_percent": 12.3,
        "memory_available_mb": 512.0,
        "memory_total_mb": 4096.0,
        "psutil_available": true
    }
    """

    # Métriques système après profiling (JSON)
    system_metrics_after = db.Column(JSONB, nullable=True)

    # Statistiques totales (JSON)
    total_stats = db.Column(JSONB, nullable=True)
    """
    Format:
    {
        "total_calls": 12345,
        "primitive_calls": 9876
    }
    """

    # Rapport textuel (optionnel, pour consultation rapide)
    report_text = db.Column(Text, nullable=True)

    # Métadonnées
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()
    )

    def to_dict(self) -> dict[str, Any]:
        """Convertit en dictionnaire pour export."""
        return {
            "id": self.id,
            "profiling_date": self.profiling_date.isoformat() if self.profiling_date else None,
            "duration_seconds": self.duration_seconds,
            "request_count": self.request_count,
            "top_functions": self.top_functions or [],
            "system_metrics_before": self.system_metrics_before or {},
            "system_metrics_after": self.system_metrics_after or {},
            "total_stats": self.total_stats or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
