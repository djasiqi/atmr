"""Feature flags et helpers — flux contractuels PORTAL (7B / 7B.5)."""

from __future__ import annotations

from typing import Any

from flask import current_app, has_app_context

FLOW_LEGACY = "legacy"
FLOW_DOUBLE_VALIDATION_V2 = "double_validation_v2"
FLOW_CONDITIONAL_ORDER_V1 = "conditional_order_v1"

ERROR_PORTAL_OFFER_ABOVE_CLIENT_LIMIT = "portal_offer_above_client_limit"
ERROR_PORTAL_OFFER_CONFLICT = "portal_offer_conflict"
ERROR_PORTAL_OFFER_STALE = "portal_offer_stale"
ERROR_PORTAL_MAXIMUM_REQUIRED = "portal_maximum_accepted_required"
ERROR_PORTAL_NO_CANCELLATION_POLICY = "portal_cancellation_policy_required"
ERROR_PORTAL_PRICING_CEILING_UNAVAILABLE = "portal_pricing_ceiling_unavailable"
ERROR_TRANSPORT_ALREADY_ASSIGNED = "transport_already_assigned"
ERROR_CARRIER_NOT_IN_ORDER_POOL = "carrier_not_in_order_pool"
ERROR_PORTAL_CANCELLATION_EXCEEDS_CHANNEL_CAP = (
    "portal_cancellation_exceeds_channel_cap"
)
ERROR_PORTAL_CANCELLATION_DIMENSION_REJECTED = "portal_cancellation_dimension_rejected"
ERROR_PORTAL_CHANNEL_POLICY_REQUIRED = "portal_channel_cancellation_policy_required"


def _config_bool(key: str) -> bool:
    if not has_app_context():
        return False
    raw = current_app.config.get(key, False)
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def is_portal_double_validation_enabled() -> bool:
    """Activation DV — OFF par défaut."""
    return _config_bool("PORTAL_DOUBLE_VALIDATION_ENABLED")


def is_portal_conditional_order_enabled() -> bool:
    """Activation commande conditionnelle 7B.5 — OFF par défaut."""
    return _config_bool("PORTAL_CONDITIONAL_ORDER_ENABLED")


def resolve_portal_contract_flow(*, client: Any) -> str:
    """Version de flux pour une nouvelle création PORTAL.

    Prérequis : coordination d'activation déjà validée au boot
    (flags mutuellement exclusifs).
    """
    from services.auth.portal_phone_verification import is_portal_client

    if client is None or not is_portal_client(client):
        return FLOW_LEGACY
    if is_portal_conditional_order_enabled():
        return FLOW_CONDITIONAL_ORDER_V1
    if is_portal_double_validation_enabled():
        return FLOW_DOUBLE_VALIDATION_V2
    return FLOW_LEGACY


def booking_uses_double_validation(booking: Any) -> bool:
    flow = getattr(booking, "portal_contract_flow", None) or FLOW_LEGACY
    return flow == FLOW_DOUBLE_VALIDATION_V2


def booking_uses_conditional_order(booking: Any) -> bool:
    flow = getattr(booking, "portal_contract_flow", None) or FLOW_LEGACY
    return flow == FLOW_CONDITIONAL_ORDER_V1


def booking_uses_portal_contract_flow(booking: Any) -> bool:
    """DV v2 ou conditional_order_v1 (plafond / pas Saferpay)."""
    return booking_uses_double_validation(booking) or booking_uses_conditional_order(
        booking
    )
