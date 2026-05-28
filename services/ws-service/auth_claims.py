"""Normalisation JWT pour ws-service (sub UUID + claims métier)."""

from __future__ import annotations

from typing import Any, TypedDict


class NormalizedClaims(TypedDict, total=False):
    user_id: str
    role: str
    company_id: int
    driver_id: int
    sub: str


def normalize_auth_payload(payload: dict[str, Any]) -> NormalizedClaims | None:
    role_obj = payload.get("role")
    if not isinstance(role_obj, str) or not role_obj.strip():
        return None

    role = role_obj.strip()
    company_id = payload.get("company_id") if isinstance(payload.get("company_id"), int) else None
    driver_id = payload.get("driver_id") if isinstance(payload.get("driver_id"), int) else None

    user_id: str | None = None
    uid = payload.get("user_id")
    if isinstance(uid, int):
        user_id = str(uid)
    elif isinstance(uid, str) and uid.strip():
        user_id = uid.strip()
    else:
        sub = payload.get("sub")
        if isinstance(sub, str) and sub.strip():
            user_id = sub.strip()
        elif isinstance(sub, int):
            user_id = str(sub)

    if not user_id:
        return None

    sub_val = payload.get("sub")
    sub_str = sub_val if isinstance(sub_val, str) else user_id

    return NormalizedClaims(
        user_id=user_id,
        role=role,
        company_id=company_id,
        driver_id=driver_id,
        sub=sub_str,
    )
