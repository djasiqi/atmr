from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from application.notifications.upsert_device_token import upsert_device_token
from ext import app_logger

MIN_TOKEN_LENGTH = 10


class _UserRepo(Protocol):
    def find_by_public_id(self, public_id: str) -> Any | None: ...


class _CompanyRepo(Protocol):
    def find_model_by_id(self, company_id: int) -> Any | None: ...


@dataclass(frozen=True, slots=True)
class SaveCompanyPushTokenResult:
    response: dict[str, Any]
    status_code: int
    should_commit: bool = False


class SaveCompanyPushTokenUseCase:
    """Enregistre un push token pour une entreprise (multi-appareils)."""

    def __init__(
        self,
        *,
        user_repo: _UserRepo,
        company_repo: _CompanyRepo,
    ) -> None:
        super().__init__()
        self._user_repo = user_repo
        self._company_repo = company_repo

    def execute(
        self,
        *,
        payload: dict[str, Any],
        _jwt_identity: str | None,
        role_claim: str,
        company_from_user: Any | None,
    ) -> SaveCompanyPushTokenResult:
        token_any: Any = (
            payload.get("token")
            or payload.get("push_token")
            or payload.get("expo_token")
        )
        if not isinstance(token_any, str) or len(token_any.strip()) < MIN_TOKEN_LENGTH:
            return SaveCompanyPushTokenResult(
                {"error": "Token push invalide ou manquant."},
                400,
            )

        token = token_any.strip()
        device_id_raw = payload.get("device_id") or payload.get("deviceId")
        if not device_id_raw or not str(device_id_raw).strip():
            return SaveCompanyPushTokenResult(
                {
                    "error": "device_id obligatoire pour l'enregistrement push entreprise."
                },
                400,
            )
        device_id = str(device_id_raw).strip()

        platform = payload.get("platform")
        provider = payload.get("provider", "expo")
        company_id_payload = payload.get("companyId") or payload.get("company_id")

        company: Any | None = None

        if role_claim == "COMPANY":
            company = company_from_user
            if not company:
                return SaveCompanyPushTokenResult(
                    {"error": "Entreprise introuvable pour ce compte."},
                    403,
                )
            if company_id_payload is not None:
                try:
                    requested_id = int(company_id_payload)
                except (TypeError, ValueError):
                    return SaveCompanyPushTokenResult(
                        {"error": "Format companyId invalide."},
                        400,
                    )
                if int(company.id) != requested_id:
                    return SaveCompanyPushTokenResult(
                        {"error": "Accès refusé (companyId ne correspond pas)."},
                        403,
                    )
        else:
            if company_id_payload is None:
                return SaveCompanyPushTokenResult(
                    {"error": "companyId requis pour un admin."},
                    400,
                )
            try:
                requested_id = int(company_id_payload)
            except (TypeError, ValueError):
                return SaveCompanyPushTokenResult(
                    {"error": "Format companyId invalide."},
                    400,
                )
            company = self._company_repo.find_model_by_id(company_id=requested_id)
            if not company:
                return SaveCompanyPushTokenResult(
                    {"error": f"Entreprise introuvable pour l'ID {requested_id}."},
                    404,
                )

        company_id = int(company.id)

        _surf = payload.get("client_auth_surface") or payload.get("clientAuthSurface")
        if _surf is not None:
            app_logger.info(
                "[push-token] client_auth_surface=%r company_id=%s",
                _surf,
                company_id,
            )

        try:
            upsert_device_token(
                company_id=company_id,
                device_id=device_id,
                token=token,
                platform=platform if isinstance(platform, str) else None,
                provider=provider if isinstance(provider, str) else None,
            )
        except ValueError as e:
            return SaveCompanyPushTokenResult({"error": str(e)}, 400)

        from models import User

        company_user = User.query.get(company.user_id)
        if company_user:
            company_user.push_token = token

        return SaveCompanyPushTokenResult(
            {
                "message": "✅ Push token entreprise enregistré.",
                "company_id": company_id,
            },
            200,
            should_commit=True,
        )
