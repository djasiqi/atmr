"""Enregistrement append-only des acceptations CGU / CGV du client privé.

Le navigateur ne fournit ni version, ni empreinte, ni horodatage, ni
instantanés. Ces valeurs viennent du catalogue serveur et du compte.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from ext import db
from models.client_terms_acceptance import (
    VERIFICATION_NOT_VERIFIED,
    VERIFICATION_OTP_SMS,
    ClientTermsAcceptance,
    LegalDocumentVersion,
)
from services.auth.portal_phone_verification import (
    is_portal_client,
    latest_activation_session,
)
from services.legal.portal_terms_catalog import (
    CatalogIntegrityError,
    PublishedTerms,
    current_portal_terms,
)

CLIENT_SUPPLIED_TERMS_FIELDS = frozenset(
    {
        "accepted_at",
        "document_type",
        "email_snapshot",
        "email_verified_at_snapshot",
        "phone_snapshot",
        "phone_verified_at_snapshot",
        "terms_hash",
        "terms_version",
        "user_id",
        "verification_method",
    }
)


class PortalTermsContextError(Exception):
    """L'acceptation de ce registre est réservée au compte client privé."""

    code = "portal_terms_context_required"


class ClientSuppliedTermsError(Exception):
    """Le client a tenté de choisir une preuve que le serveur doit fixer."""

    code = "client_supplied_terms_forbidden"


def reject_client_supplied_terms(payload: dict[str, Any] | None) -> None:
    if not payload:
        return
    supplied = CLIENT_SUPPLIED_TERMS_FIELDS.intersection(payload)
    if supplied:
        fields = ", ".join(sorted(supplied))
        raise ClientSuppliedTermsError(
            f"Ces champs sont fixés par le serveur: {fields}."
        )


def ensure_document_version(spec: PublishedTerms) -> LegalDocumentVersion:
    """Insère la version si elle n'existe pas. Ne réécrit jamais une ligne existante."""
    existing = LegalDocumentVersion.query.filter_by(
        document_type=spec.document_type,
        terms_version=spec.terms_version,
    ).one_or_none()
    if existing is not None:
        if (
            existing.terms_hash != spec.terms_hash
            or existing.canonical_body != spec.canonical_body
        ):
            raise CatalogIntegrityError(
                "La version déjà publiée ne correspond pas au catalogue."
            )
        return existing
    row = LegalDocumentVersion(
        document_type=spec.document_type,
        terms_version=spec.terms_version,
        terms_hash=spec.terms_hash,
        locale=spec.locale,
        canonical_body=spec.canonical_body,
    )
    db.session.add(row)
    db.session.flush()
    return row


def record_portal_terms_acceptance(
    user: Any,
    client: Any,
    documents: tuple[PublishedTerms, ...] | list[PublishedTerms] | None = None,
) -> list[ClientTermsAcceptance]:
    """Crée une ligne par document. N'altère aucune acceptation antérieure."""
    if not is_portal_client(client):
        raise PortalTermsContextError(
            "Seuls les comptes client privé peuvent accepter ces conditions."
        )
    specs = list(documents) if documents is not None else list(current_portal_terms())
    activation = latest_activation_session(user)
    email_verified_at = getattr(activation, "email_verified_at", None)
    phone_verified_at = getattr(user, "phone_verified_at", None)
    accepted_at = datetime.now(UTC)
    created: list[ClientTermsAcceptance] = []
    for spec in specs:
        version = ensure_document_version(spec)
        row = ClientTermsAcceptance(
            user_id=user.id,
            document_version_id=version.id,
            document_type=spec.document_type,
            terms_version=spec.terms_version,
            terms_hash=spec.terms_hash,
            accepted_at=accepted_at,
            email_snapshot=getattr(user, "email", None),
            email_verified_at_snapshot=email_verified_at,
            phone_snapshot=getattr(user, "phone", None),
            phone_verified_at_snapshot=phone_verified_at,
            verification_method=(
                VERIFICATION_OTP_SMS
                if phone_verified_at is not None
                else VERIFICATION_NOT_VERIFIED
            ),
        )
        db.session.add(row)
        created.append(row)
    db.session.flush()
    return created


def list_portal_terms_acceptances(user_id: int) -> list[ClientTermsAcceptance]:
    return (
        ClientTermsAcceptance.query.filter_by(user_id=user_id)
        .order_by(
            ClientTermsAcceptance.accepted_at.asc(), ClientTermsAcceptance.id.asc()
        )
        .all()
    )
