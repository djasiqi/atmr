from __future__ import annotations

import contextlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Protocol


class _DriverLike(Protocol):
    id: int
    driver_type: Any
    company_id: int | None


class _CompanyLike(Protocol):
    id: int
    name: str


class _UserLike(Protocol):
    id: int
    public_id: Any
    email: str | None
    first_name: str | None
    last_name: str | None


class _TokenCreator(Protocol):
    def __call__(
        self,
        *,
        identity: str,
        additional_claims: dict[str, Any],
        expires_delta: timedelta,
    ) -> str: ...


class _RefreshTokenStorer(Protocol):
    def __call__(
        self,
        *,
        token: str,
        user_id: int,
        expires_at: datetime,
        device_id: str | None,
        device_name: str | None,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class SwitchToEnterpriseCommand:
    driver: _DriverLike
    access_expires_delta: timedelta
    refresh_expires_delta: timedelta
    device_id: str | None
    device_name: str | None


@dataclass(frozen=True, slots=True)
class SwitchToEnterpriseResult:
    response: dict[str, Any]
    status_code: int


class SwitchToEnterpriseUseCase:
    """Use-case Application: basculer chauffeur → entreprise (tokens + payload)."""

    def __init__(
        self,
        *,
        find_company_fn: Callable[[int], _CompanyLike | None],
        find_company_user_fn: Callable[[_DriverLike, _CompanyLike], _UserLike | None],
        create_access_token_fn: _TokenCreator,
        create_refresh_token_fn: _TokenCreator,
        store_refresh_token_fn: _RefreshTokenStorer | None,
        now_utc_fn: Callable[[], datetime],
        driver_type_emergency: Any,
    ) -> None:
        super().__init__()
        self._find_company = find_company_fn
        self._find_company_user = find_company_user_fn
        self._create_access_token = create_access_token_fn
        self._create_refresh_token = create_refresh_token_fn
        self._store_refresh_token = store_refresh_token_fn
        self._now_utc = now_utc_fn
        self._driver_type_emergency = driver_type_emergency

    def execute(self, cmd: SwitchToEnterpriseCommand) -> SwitchToEnterpriseResult:
        driver = cmd.driver
        # #region agent log
        import json
        from datetime import UTC, datetime
        from pathlib import Path

        log_path = Path(r"c:\Users\jasiq\atmr\.cursor\debug.log")
        try:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "location": "switch_to_enterprise.py:execute",
                            "message": "driver type check",
                            "data": {
                                "driver_id": driver.id,
                                "driver_type": str(driver.driver_type),
                                "driver_type_value": driver.driver_type.value
                                if hasattr(driver.driver_type, "value")
                                else str(driver.driver_type),
                                "expected_emergency": str(self._driver_type_emergency),
                                "expected_emergency_value": self._driver_type_emergency.value
                                if hasattr(self._driver_type_emergency, "value")
                                else str(self._driver_type_emergency),
                                "is_equal": driver.driver_type
                                == self._driver_type_emergency,
                                "type_comparison": type(driver.driver_type)
                                is type(self._driver_type_emergency),
                            },
                            "timestamp": datetime.now(UTC).isoformat(),
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "C",
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass
        # #endregion
        # Comparaison robuste : comparer les valeurs plutôt que les objets enum
        # pour gérer les cas où driver_type pourrait être une chaîne ou un enum différent
        driver_type_value = (
            driver.driver_type.value
            if hasattr(driver.driver_type, "value")
            else str(driver.driver_type)
        )
        expected_value = (
            self._driver_type_emergency.value
            if hasattr(self._driver_type_emergency, "value")
            else str(self._driver_type_emergency)
        )

        if driver_type_value != expected_value:
            # #region agent log
            try:
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "location": "switch_to_enterprise.py:execute",
                                "message": "driver type mismatch - returning 403",
                                "data": {
                                    "driver_type": str(driver.driver_type),
                                    "driver_type_value": driver_type_value,
                                    "expected": str(self._driver_type_emergency),
                                    "expected_value": expected_value,
                                },
                                "timestamp": datetime.now(UTC).isoformat(),
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "C",
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            # #endregion
            return SwitchToEnterpriseResult(
                response={
                    "error": "Seuls les chauffeurs d'urgence peuvent basculer vers le compte entreprise."
                },
                status_code=403,
            )

        if not driver.company_id:
            return SwitchToEnterpriseResult(
                response={"error": "Entreprise associée au chauffeur introuvable."},
                status_code=404,
            )

        company = self._find_company(int(driver.company_id))
        if not company:
            return SwitchToEnterpriseResult(
                response={"error": "Entreprise introuvable."},
                status_code=404,
            )

        company_user = self._find_company_user(driver, company)
        if not company_user:
            return SwitchToEnterpriseResult(
                response={"error": "Aucun compte entreprise associé à ce chauffeur."},
                status_code=404,
            )

        claims = {
            "role": "company",
            "company_id": company.id,
            "aud": "atmr-mobile-enterprise",
        }
        access_token = self._create_access_token(
            identity=str(company_user.public_id),
            additional_claims=claims,
            expires_delta=cmd.access_expires_delta,
        )

        refresh_token = self._create_refresh_token(
            identity=str(company_user.public_id),
            additional_claims={"aud": "atmr-mobile-enterprise", "role": "company"},
            expires_delta=cmd.refresh_expires_delta,
        )

        storer = self._store_refresh_token
        if storer is not None:
            with contextlib.suppress(Exception):
                storer(
                    token=refresh_token,
                    user_id=company_user.id,
                    expires_at=self._now_utc() + cmd.refresh_expires_delta,
                    device_id=cmd.device_id,
                    device_name=cmd.device_name,
                )

        return SwitchToEnterpriseResult(
            response={
                "token": access_token,
                "refresh_token": refresh_token,
                "user": {
                    "public_id": company_user.public_id,
                    "email": company_user.email,
                    "first_name": company_user.first_name,
                    "last_name": company_user.last_name,
                    "role": "company",
                },
                "company": {"id": company.id, "name": company.name},
            },
            status_code=200,
        )
