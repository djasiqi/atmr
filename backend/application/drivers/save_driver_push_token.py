from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

MIN_TOKEN_LENGTH = 10


class _UserRepo(Protocol):
    def find_by_public_id(self, public_id: str) -> Any | None: ...


class _DriverRepo(Protocol):
    def find_model_by_user_id(self, user_id: int) -> Any | None: ...
    def find_model_by_id(self, driver_id: int) -> Any | None: ...


@dataclass(frozen=True, slots=True)
class SaveDriverPushTokenResult:
    response: dict[str, Any]
    status_code: int
    driver: Any | None = None
    should_commit: bool = False


class SaveDriverPushTokenUseCase:
    """Use-case Application: enregistrer un push token pour un chauffeur."""

    def __init__(self, *, user_repo: _UserRepo, driver_repo: _DriverRepo) -> None:
        super().__init__()
        self._user_repo = user_repo
        self._driver_repo = driver_repo

    def execute(
        self,
        *,
        payload: dict[str, Any],
        jwt_identity: str | None,
    ) -> SaveDriverPushTokenResult:
        response: dict[str, Any] | None = None
        status_code: int = 200
        driver: Any | None = None
        should_commit = False

        token_any: Any = (
            payload.get("token")
            or payload.get("expo_token")
            or payload.get("push_token")
        )
        token: str | None = None
        if not isinstance(token_any, str) or len(token_any) < MIN_TOKEN_LENGTH:
            response = {"error": "Token FCM/Expo invalide ou manquant."}
            status_code = 400
        else:
            token = token_any

        # 1) driverId explicite
        driver_id: int | None = None
        if response is None:
            raw_id: Any = payload.get("driverId") or payload.get("driver_id")
            if raw_id is not None:
                try:
                    driver_id = int(float(raw_id))
                except (ValueError, TypeError):
                    response = {"error": f"Format de driverId invalide: {raw_id}"}
                    status_code = 400

        # 2) fallback JWT (user -> driver)
        if response is None and driver_id is None:
            if not jwt_identity:
                response = {"error": "Token JWT invalide ou expiré."}
                status_code = 401
            else:
                user = self._user_repo.find_by_public_id(public_id=jwt_identity)
                if not user:
                    response = {"error": "Utilisateur non trouvé pour le JWT."}
                    status_code = 404
                else:
                    drv = self._driver_repo.find_model_by_user_id(user_id=int(user.id))
                    if not drv:
                        response = {
                            "error": "Chauffeur introuvable pour cet utilisateur."
                        }
                        status_code = 404
                    else:
                        driver_id = int(drv.id)

        # 3) charger driver + setter token
        if response is None:
            if driver_id is None:
                response = {"error": "driver_id manquant"}
                status_code = 400
            else:
                driver = self._driver_repo.find_model_by_id(driver_id=driver_id)
            if not driver:
                response = {"error": f"Chauffeur introuvable pour l'ID {driver_id}."}
                status_code = 404
            else:
                assert token is not None
                driver.push_token = token
                response = {
                    "message": "✅ Push token enregistré avec succès.",
                    "driver_id": driver_id,
                }
                status_code = 200
                should_commit = True

        # basedpyright peut ne pas reconnaître les kwargs sur dataclass(slots=True)
        # selon la config/version; on utilise un appel positionnel.
        return SaveDriverPushTokenResult(response, status_code, driver, should_commit)
