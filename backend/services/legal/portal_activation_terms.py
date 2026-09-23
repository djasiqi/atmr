"""Acceptation initiale des CGU et CGV pendant l'activation PORTAL.

Le navigateur n'envoie qu'une intention. La version, l'empreinte et les
instantanés viennent du catalogue et du compte. Les sessions d'activation
créées avant ce parcours ne sont pas concernées.
"""

from __future__ import annotations

from typing import Any

from models.client_terms_acceptance import ClientTermsAcceptance
from services.auth.portal_phone_verification import (
    is_portal_client_user,
    latest_activation_session,
)
from services.legal.portal_terms_catalog import current_portal_terms


def serialize_current_portal_terms() -> list[dict[str, str]]:
    """Représentation servie à l'écran d'activation. Même source que l'enregistrement."""
    return [
        {
            "document_type": spec.document_type,
            "terms_version": spec.terms_version,
            "terms_hash": spec.terms_hash,
            "canonical_body": spec.canonical_body,
            "locale": spec.locale,
        }
        for spec in current_portal_terms()
    ]


def has_current_portal_terms(user_id: int) -> bool:
    specs = current_portal_terms()
    rows = ClientTermsAcceptance.query.filter_by(user_id=user_id).all()
    found = {(row.document_type, row.terms_version, row.terms_hash) for row in rows}
    return all(
        (spec.document_type, spec.terms_version, spec.terms_hash) in found
        for spec in specs
    )


def portal_terms_block_lazy_promotion(user: Any) -> bool:
    """Empêche le login de finaliser un nouveau compte qui n'a pas encore accepté.

    Les comptes dont la session n'exige pas les conditions (inscriptions
    antérieures) restent promouvables dès l'e-mail, comme AUTH-SMS-02.
    """
    if user is None or not is_portal_client_user(user):
        return False
    session = latest_activation_session(user)
    if session is None or not getattr(session, "portal_terms_required", False):
        return False
    return not has_current_portal_terms(user.id)
