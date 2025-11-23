"""✅ 2.7: Profiler DB pour détecter requêtes N+1 et problèmes de performance.

Utilise les event listeners SQLAlchemy natifs pour compter et analyser les requêtes.
Activable via variable d'environnement ENABLE_DB_PROFILING=true.
"""

import logging
import os
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, DefaultDict

from sqlalchemy import event as sqlalchemy_event
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

# ✅ 2.7: Constantes pour DB Profiler
SLOW_QUERY_THRESHOLD_MS = 1000  # Seuil pour détecter requêtes lentes (1 seconde)
N_PLUS_1_REPORT_THRESHOLD = 20  # Seuil pour avertir trop de requêtes dans rapport
N_PLUS_1_CONTEXT_THRESHOLD = 10  # Seuil pour avertir trop de requêtes dans contexte

# Stockage global des métriques de profiling
_profile_stats: DefaultDict[str, list[float]] = defaultdict(list)
_profile_query_counts: DefaultDict[str, int] = defaultdict(int)
_profile_context: dict[str, Any] = {}


def is_profiling_enabled() -> bool:
    """Vérifie si le profiling DB est activé via variable d'environnement."""
    return os.getenv("ENABLE_DB_PROFILING", "false").lower() in ("true", "1", "yes")


class DBProfiler:
    """Profiler pour détecter requêtes N+1 et problèmes de performance."""

    def __init__(self, enabled: bool = False):
        """Initialise le profiler.

        Args:
            enabled: Active le profiling si True
        """
        super().__init__()
        self.enabled = enabled
        self.query_count = 0
        self.query_times: list[float] = []
        self.query_statements: list[str] = []

        if enabled:
            logger.info("[DB Profiler] ✅ Profiling DB activé")
            self._setup_event_listeners()
        else:
            logger.debug("[DB Profiler] Profiling DB désactivé")

    def _setup_event_listeners(self):
        """Configure les event listeners SQLAlchemy pour profiler les requêtes."""

        @sqlalchemy_event.listens_for(Engine, "before_cursor_execute")
        def receive_before_cursor_execute(  # pyright: ignore[reportUnusedFunction]
            _conn, _cursor, statement, _parameters, context, _executemany
        ):
            """Capture le début d'exécution d'une requête."""
            if not self.enabled:
                return

            # Stocker le statement pour analyse
            self.query_statements.append(statement[:200])  # Limiter taille

            # Mesurer le temps d'exécution
            import time

            context._query_start_time = time.time()

        @sqlalchemy_event.listens_for(Engine, "after_cursor_execute")
        def receive_after_cursor_execute(  # pyright: ignore[reportUnusedFunction]
            _conn, _cursor, statement, _parameters, context, _executemany
        ):
            """Capture la fin d'exécution d'une requête."""
            if not self.enabled:
                return

            # Calculer durée
            if hasattr(context, "_query_start_time"):
                import time

                duration = time.time() - context._query_start_time
                self.query_count += 1
                self.query_times.append(duration)

                # Détecter requêtes lentes (> 1 seconde)
                if duration > (SLOW_QUERY_THRESHOLD_MS / 1000):
                    logger.warning(
                        "[DB Profiler] ⚠️ Requête lente détectée (%.2fs): %s...",
                        duration,
                        statement[:100],
                    )

    def reset(self):
        """Réinitialise les statistiques de profiling."""
        self.query_count = 0
        self.query_times.clear()
        self.query_statements.clear()

    def get_stats(self) -> dict[str, Any]:
        """Retourne les statistiques de profiling.

        Returns:
            Dict avec query_count, avg_time, max_time, total_time
        """
        if not self.query_times:
            return {
                "query_count": 0,
                "avg_time_ms": 0.0,
                "min_time_ms": 0.0,
                "max_time_ms": 0.0,
                "total_time_ms": 0.0,
            }

        total_time = sum(self.query_times)
        avg_time = total_time / len(self.query_times) if self.query_times else 0.0

        return {
            "query_count": self.query_count,
            "avg_time_ms": round(avg_time * 1000, 2),
            "min_time_ms": round(min(self.query_times) * 1000, 2),
            "max_time_ms": round(max(self.query_times) * 1000, 2),
            "total_time_ms": round(total_time * 1000, 2),
            "queries": self.query_statements[-10:],  # Dernières 10 requêtes
        }

    def detect_n_plus_1(self, threshold: int = 10) -> bool:
        """Détecte si un pattern N+1 est suspect (trop de requêtes similaires).

        Args:
            threshold: Nombre minimum de requêtes similaires pour suspecter N+1

        Returns:
            True si pattern N+1 suspecté
        """
        if not self.enabled or len(self.query_statements) < threshold:
            return False

        # Compter occurrences de chaque type de requête
        query_patterns: DefaultDict[str, int] = defaultdict(int)
        for stmt in self.query_statements:
            # Normaliser la requête (enlever IDs, valeurs)
            normalized = self._normalize_query(stmt)
            query_patterns[normalized] += 1

        # Si une requête apparaît > threshold fois, suspecter N+1
        for pattern, count in query_patterns.items():
            if count >= threshold:
                logger.warning(
                    "[DB Profiler] 🚨 Pattern N+1 suspecté: '%s' exécutée %d fois",
                    pattern[:100],
                    count,
                )
                return True

        return False

    def _normalize_query(self, query: str) -> str:
        """Normalise une requête SQL pour détecter les patterns similaires.

        Remplace les valeurs numériques et strings par des placeholders.

        Args:
            query: Requête SQL brute

        Returns:
            Requête normalisée
        """
        import re

        # Remplacer nombres par ?
        normalized = re.sub(r"\b\d+\b", "?", query)

        # Remplacer strings entre quotes par ?
        normalized = re.sub(r"'[^']*'", "'?'", normalized)
        normalized = re.sub(r'"[^"]*"', '"?"', normalized)

        # Normaliser espaces
        normalized = " ".join(normalized.split())

        return normalized[:150]  # Limiter longueur

    def generate_report(self) -> str:
        """Génère un rapport textuel de profiling.

        Returns:
            Rapport formaté
        """
        stats = self.get_stats()
        lines = []

        lines.append("=" * 80)
        lines.append("DB PROFILING REPORT (2.7)")
        lines.append("=" * 80)
        lines.append("")

        if not self.enabled:
            lines.append(
                "⚠️ Profiling désactivé (set ENABLE_DB_PROFILING=true to enable)"
            )
            return "\n".join(lines)

        lines.append(f"Nombre total de requêtes: {stats['query_count']}")
        lines.append(f"Temps total: {stats['total_time_ms']}ms")
        lines.append(f"Temps moyen: {stats['avg_time_ms']}ms")
        lines.append(f"Temps min: {stats['min_time_ms']}ms")
        lines.append(f"Temps max: {stats['max_time_ms']}ms")
        lines.append("")

        # Avertissement si trop de requêtes
        if stats["query_count"] > N_PLUS_1_REPORT_THRESHOLD:
            lines.append(
                (
                    f"⚠️ ATTENTION: {stats['query_count']} requêtes détectées "
                    "(suspect N+1?)"
                )
            )

        # Avertissement si requêtes lentes
        if stats["max_time_ms"] > SLOW_QUERY_THRESHOLD_MS:
            lines.append(
                f"⚠️ ATTENTION: Requête lente détectée ({stats['max_time_ms']}ms)"
            )

        # Dernières requêtes
        if stats["queries"]:
            lines.append("Dernières requêtes:")
            for i, q in enumerate(stats["queries"][-5:], 1):
                lines.append(f"  {i}. {q[:100]}...")

        n_plus_1_detected = self.detect_n_plus_1()
        if n_plus_1_detected:
            lines.append("")
            lines.append(
                "🚨 PATTERN N+1 SUSPECTÉ - Action recommandée: vérifier eager loading"
            )

        lines.append("=" * 80)

        return "\n".join(lines)


# Singleton global
_db_profiler: DBProfiler | None = None


def get_db_profiler() -> DBProfiler:
    """Récupère l'instance singleton du profiler DB."""
    global _db_profiler  # noqa: PLW0603

    if _db_profiler is None:
        _db_profiler = DBProfiler(enabled=is_profiling_enabled())

    return _db_profiler


def reset_db_profiler() -> None:
    """Reset le profiler (pour tests)."""
    global _db_profiler  # noqa: PLW0603
    _db_profiler = None


@contextmanager
def profile_db_context(context_name: str = "request"):
    """Context manager pour profiler une section de code.

    Usage:
        with profile_db_context("endpoint_/api/bookings"):
            # Code à profiler
            ...
        stats = get_db_profiler().get_stats()

    Args:
        context_name: Nom du contexte (pour logs)
    """
    profiler = get_db_profiler()

    if not profiler.enabled:
        yield
        return

    # Réinitialiser avant le contexte
    profiler.reset()

    try:
        yield profiler
    finally:
        # Générer rapport si trop de requêtes
        stats = profiler.get_stats()
        if stats["query_count"] > N_PLUS_1_CONTEXT_THRESHOLD:
            logger.warning(
                "[DB Profiler] Contexte '%s': %d requêtes, %.2fms total",
                context_name,
                stats["query_count"],
                stats["total_time_ms"],
            )

            # Détecter N+1
            if profiler.detect_n_plus_1():
                logger.error(
                    "[DB Profiler] 🚨 N+1 détecté dans contexte '%s'!", context_name
                )
