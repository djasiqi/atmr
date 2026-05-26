from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from ext import app_logger

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
        jwt_driver_id: int | None = None
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
                        jwt_driver_id = int(drv.id)
                        driver_id = jwt_driver_id
        elif response is None and driver_id is not None:
            # ✅ CORRECTIF: Vérifier que le JWT correspond au driverId fourni
            if not jwt_identity:
                response = {"error": "Token JWT requis pour enregistrer un token push."}
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
                        jwt_driver_id = int(drv.id)
                        # ✅ Vérifier que le driverId du payload correspond au JWT
                        if jwt_driver_id != driver_id:
                            response = {
                                "error": "Vous ne pouvez enregistrer un token que pour votre propre compte."
                            }
                            status_code = 403
                            driver_id = None  # Empêcher l'enregistrement

        # 3) charger driver + enregistrer token dans DeviceToken (multi-device)
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
                # Métadonnée client (Phase 4A mobile) — ignorée pour la persistance, loguée pour corrélation / 4B.
                _surf = payload.get("client_auth_surface") or payload.get(
                    "clientAuthSurface"
                )
                if _surf is not None:
                    app_logger.info(
                        "[push-token] client_auth_surface=%r driver_id=%s",
                        _surf,
                        driver_id,
                    )
                # ✅ CORRECTIF #3: Enregistrer dans DeviceToken (support multi-device)
                from ext import db
                from models import DeviceToken

                device_id = payload.get("device_id") or payload.get("deviceId")
                platform = payload.get("platform")  # "ios" | "android"
                provider = payload.get("provider", "expo")  # "expo" | "fcm"
                if provider not in ("expo", "fcm"):
                    provider = (
                        "fcm" if not token.startswith("ExponentPushToken") else "expo"
                    )
                if (
                    provider == "fcm"
                    and platform == "ios"
                    and token.startswith(("APA91", "APA91b"))
                ):
                    app_logger.warning(
                        "[push-token] platform ios->android inferred for FCM Android token driver_id=%s",
                        driver_id,
                    )
                    platform = "android"

                existing_token = DeviceToken.query.filter_by(
                    driver_id=driver_id,
                    token=token,
                ).first()

                if existing_token:
                    existing_token.is_active = True
                    existing_token.updated_at = datetime.utcnow()
                    if device_id:
                        existing_token.device_id = device_id
                    if platform:
                        existing_token.platform = platform
                    existing_token.provider = provider
                    app_logger.info(
                        "[push-token] Token existant réactivé/mis à jour pour driver %s (provider=%s)",
                        driver_id,
                        provider,
                    )
                else:
                    device_token = DeviceToken()
                    device_token.driver_id = driver_id
                    device_token.token = token
                    device_token.device_id = device_id
                    device_token.platform = platform
                    device_token.provider = provider
                    device_token.is_active = True
                    db.session.add(device_token)
                    app_logger.info(
                        "[push-token] Nouveau token enregistré pour driver %s (device: %s, platform: %s, provider: %s)",
                        driver_id,
                        device_id,
                        platform,
                        provider,
                    )

                # ✅ CONSERVER driver.push_token pour rétrocompatibilité (legacy)
                # Les anciens codes qui utilisent driver.push_token continueront de fonctionner
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
