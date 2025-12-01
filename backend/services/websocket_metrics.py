# backend/services/websocket_metrics.py

from collections import defaultdict
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional


class WebSocketMetrics:
    """Collecte et expose les métriques WebSocket pour monitoring."""

    def __init__(self) -> None:  # pyright: ignore[reportMissingSuperCall]
        # Pas besoin d'appeler super().__init__() car la classe n'hérite d'aucune classe
        self.connections_active: Dict[int, int] = defaultdict(
            int
        )  # company_id -> count
        self.connections_total = 0
        self.disconnections_total = 0
        self.reconnections_count: Dict[int, int] = defaultdict(int)  # user_id -> count
        self.heartbeat_latencies: List[float] = []  # Latences en ms
        self.errors_count: Dict[str, int] = defaultdict(int)  # error_type -> count
        self.start_time = datetime.now(UTC)
        # Garder seulement les 1000 dernières latences pour éviter consommation mémoire excessive
        self._max_latency_samples = 1000
        # Tracking des rooms : room_name -> nombre de clients connectés
        self.rooms_active: Dict[str, int] = defaultdict(int)

    def on_connect(
        self,
        company_id: Optional[int] = None,
        user_id: Optional[int] = None,  # noqa: ARG002
    ) -> None:
        """Enregistre une nouvelle connexion.

        Args:
            company_id: ID de l'entreprise (optionnel)
            user_id: ID de l'utilisateur (optionnel, réservé pour usage futur)
        """
        self.connections_total += 1
        if company_id:
            self.connections_active[company_id] += 1

    def on_disconnect(self, company_id: Optional[int] = None):
        """Enregistre une déconnexion."""
        self.disconnections_total += 1
        if company_id:
            self.connections_active[company_id] = max(
                0, self.connections_active[company_id] - 1
            )

    def on_reconnect(self, user_id: Optional[int] = None):
        """Enregistre une reconnexion."""
        if user_id:
            self.reconnections_count[user_id] += 1

    def on_heartbeat_pong(self, latency_ms: float):
        """Enregistre la latence d'un heartbeat pong."""
        self.heartbeat_latencies.append(latency_ms)
        # Garder seulement les N dernières pour éviter consommation mémoire
        if len(self.heartbeat_latencies) > self._max_latency_samples:
            self.heartbeat_latencies = self.heartbeat_latencies[
                -self._max_latency_samples :
            ]

    def on_error(self, error_type: str):
        """Enregistre une erreur."""
        self.errors_count[error_type] += 1

    def on_room_join(self, room_name: str):
        """Enregistre qu'un client a rejoint une room.

        Args:
            room_name: Nom de la room (ex: "company_1", "driver_10")
        """
        self.rooms_active[room_name] += 1

    def on_room_leave(self, room_name: str):
        """Enregistre qu'un client a quitté une room.

        Args:
            room_name: Nom de la room (ex: "company_1", "driver_10")
        """
        if room_name in self.rooms_active:
            self.rooms_active[room_name] = max(0, self.rooms_active[room_name] - 1)
            # Nettoyer les rooms vides (optionnel, pour garder le dict propre)
            if self.rooms_active[room_name] == 0:
                del self.rooms_active[room_name]

    def get_rooms_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques par room.

        Returns:
            Dict avec active_by_room, rooms_total, clients_total
        """
        rooms_dict = dict(self.rooms_active)
        clients_total = sum(rooms_dict.values())
        rooms_total = len(rooms_dict)

        return {
            "active_by_room": rooms_dict,
            "rooms_total": rooms_total,
            "clients_total": clients_total,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques agrégées."""
        latencies = sorted(self.heartbeat_latencies)
        n = len(latencies)

        # Calcul des percentiles
        p50 = latencies[n // 2] if n > 0 else 0.0
        p95 = (
            latencies[int(n * 0.95)]
            if n > 0 and n > 1
            else (latencies[0] if n > 0 else 0.0)
        )
        p99 = (
            latencies[int(n * 0.99)]
            if n > 0 and n > 1
            else (latencies[0] if n > 0 else 0.0)
        )

        return {
            "uptime_seconds": (datetime.now(UTC) - self.start_time).total_seconds(),
            "connections": {
                "active_total": sum(self.connections_active.values()),
                "active_by_company": dict(self.connections_active),
                "total": self.connections_total,
                "disconnections_total": self.disconnections_total,
            },
            "heartbeat": {
                "latency_ms": {
                    "p50": p50,
                    "p95": p95,
                    "p99": p99,
                    "avg": sum(latencies) / n if n > 0 else 0.0,
                    "min": latencies[0] if n > 0 else 0.0,
                    "max": latencies[-1] if n > 0 else 0.0,
                    "samples": n,
                }
            },
            "reconnections": {
                "total": sum(self.reconnections_count.values()),
                "by_user": dict(self.reconnections_count),
            },
            "errors": dict(self.errors_count),
            "rooms": self.get_rooms_stats(),
        }

    def reset(self):
        """Réinitialise les métriques (pour tests)."""
        self.__init__()


# Instance globale
ws_metrics = WebSocketMetrics()
