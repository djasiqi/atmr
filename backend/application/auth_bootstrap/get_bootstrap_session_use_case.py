"""Use case : session bootstrap (GET /auth/me)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Protocol

from application.auth_bootstrap.session_access_rules import (
    evaluate_access_denial,
    observe_driver_without_profile,
)
from application.auth_bootstrap.session_response import build_auth_me_payload
from application.auth_bootstrap.sql_session_bootstrap_reader import (
    SqlSessionBootstrapReader,
)


class JwtIdentityPort(Protocol):
    def get_jwt_identity(self) -> str | None: ...


@dataclass(frozen=True, slots=True)
class BootstrapSessionHttpResult:
    status_code: int
    body: dict[str, Any]


class GetBootstrapSessionUseCase:
    """Charge le snapshot, applique les règles métier, construit le JSON contractuel."""

    def __init__(
        self,
        *,
        jwt_port: JwtIdentityPort | None = None,
        reader: SqlSessionBootstrapReader | None = None,
    ) -> None:
        if jwt_port is None:
            from shared.infrastructure.adapters.jwt_adapter import JwtIdentityAdapter

            jwt_port = JwtIdentityAdapter()
        self._jwt = jwt_port
        self._reader = reader or SqlSessionBootstrapReader()

    def execute(self) -> BootstrapSessionHttpResult:
        t0 = time.perf_counter()
        public_id = self._jwt.get_jwt_identity()
        if not public_id:
            body = {"error": "Token JWT invalide ou manquant"}
            self._finish_metrics(401, t0, body)
            return BootstrapSessionHttpResult(401, body)

        loaded = self._reader.load_user_for_bootstrap(str(public_id))
        if loaded is None:
            body = {"error": "Utilisateur non trouvé"}
            self._finish_metrics(404, t0, body)
            return BootstrapSessionHttpResult(404, body)

        snapshot, user_orm = loaded
        denial = evaluate_access_denial(snapshot, user_orm)
        if denial is not None:
            payload = build_auth_me_payload(snapshot, denial)
            self._forbidden_metrics(denial[0])
            self._finish_metrics(403, t0, payload)
            return BootstrapSessionHttpResult(403, payload)

        observe_driver_without_profile(snapshot)
        payload = build_auth_me_payload(snapshot, None)
        self._finish_metrics(200, t0, payload)
        return BootstrapSessionHttpResult(200, payload)

    def _forbidden_metrics(self, code: str) -> None:
        try:
            from services.monitoring import auth_bootstrap_metrics as abm

            abm.observe_auth_me_forbidden(code)
        except Exception:  # noqa: BLE001
            pass

    def _finish_metrics(self, status: int, t0: float, body: dict[str, Any]) -> None:
        try:
            from services.monitoring import auth_bootstrap_metrics as abm

            raw = json.dumps(body, ensure_ascii=False, default=str)
            abm.observe_auth_me(status, time.perf_counter() - t0, len(raw.encode("utf-8")))
        except Exception:  # noqa: BLE001
            pass
