"""Statut contractuel des CGU et CGV du client privé.

Une seule source pour le dashboard, la route de réservation et le use case.
La version courante vient du catalogue. ``requires_reacceptance`` est figé à
la publication : ce service ne compare pas les textes pour deviner si un
changement est substantiel.

Si la version courante est publiée avec ``requires_reacceptance=false`` et
qu'une acceptation antérieure existe, cette acceptation reste la base
contractuelle des nouvelles réservations. Aucune ligne n'est inventée.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ext import db
from models.client_terms_acceptance import ClientTermsAcceptance
from models.user import User
from services.auth.portal_phone_verification import is_portal_client
from services.legal import portal_terms_catalog
from services.legal.portal_terms_catalog import PublishedTerms
from services.legal.record_terms_acceptance import (
    PortalTermsContextError,
    record_portal_terms_acceptance,
)

STATUS_CURRENT = "current"
STATUS_REACCEPTANCE_REQUIRED = "reacceptance_required"
TERMS_REACCEPTANCE_REQUIRED = "terms_reacceptance_required"

BASIS_CURRENT_ACCEPTANCE = "current_acceptance"
BASIS_PRIOR_ACCEPTANCE = "prior_acceptance"
BASIS_MISSING = "missing"


class PortalTermsReacceptanceRequired(Exception):
    """Nouvelle réservation PORTAL refusée tant que les conditions exigées manquent."""

    code = TERMS_REACCEPTANCE_REQUIRED

    def __init__(self, documents: list[str]) -> None:
        self.documents = list(documents)
        self.message = (
            "Acceptez les conditions en vigueur avant une nouvelle réservation."
        )
        super().__init__(self.message)


@dataclass(frozen=True)
class DocumentTermsStatus:
    document_type: str
    current_version: str
    current_hash: str
    canonical_body: str
    requires_reacceptance: bool
    accepted_version: str | None
    accepted_at: datetime | None
    acceptance_id: int | None
    acceptance_required: bool
    contractual_basis: str


@dataclass(frozen=True)
class PortalTermsStatus:
    status: str
    documents: tuple[DocumentTermsStatus, ...]


def _latest_acceptance(
    user_id: int, document_type: str
) -> ClientTermsAcceptance | None:
    return (
        ClientTermsAcceptance.query.filter_by(
            user_id=user_id, document_type=document_type
        )
        .order_by(
            ClientTermsAcceptance.accepted_at.desc(),
            ClientTermsAcceptance.id.desc(),
        )
        .first()
    )


def _document_status(user_id: int, spec: PublishedTerms) -> DocumentTermsStatus:
    latest = _latest_acceptance(user_id, spec.document_type)
    covers_current = (
        latest is not None
        and latest.terms_version == spec.terms_version
        and latest.terms_hash == spec.terms_hash
    )
    if latest is None:
        required = True
        basis = BASIS_MISSING
    elif covers_current:
        required = False
        basis = BASIS_CURRENT_ACCEPTANCE
    elif spec.requires_reacceptance:
        required = True
        basis = BASIS_MISSING
    else:
        required = False
        basis = BASIS_PRIOR_ACCEPTANCE
    return DocumentTermsStatus(
        document_type=spec.document_type,
        current_version=spec.terms_version,
        current_hash=spec.terms_hash,
        canonical_body=spec.canonical_body,
        requires_reacceptance=bool(spec.requires_reacceptance),
        accepted_version=latest.terms_version if latest is not None else None,
        accepted_at=latest.accepted_at if latest is not None else None,
        acceptance_id=latest.id if latest is not None else None,
        acceptance_required=required,
        contractual_basis=basis,
    )


def resolve_portal_terms_status(user: Any) -> PortalTermsStatus:
    """Compare les acceptations du compte aux versions courantes du catalogue."""
    specs = portal_terms_catalog.current_portal_terms()
    types = [spec.document_type for spec in specs]
    if len(types) != len(set(types)):
        raise portal_terms_catalog.CatalogIntegrityError(
            "Plusieurs versions courantes pour un même document."
        )
    documents = tuple(_document_status(int(user.id), spec) for spec in specs)
    required = any(document.acceptance_required for document in documents)
    return PortalTermsStatus(
        status=STATUS_REACCEPTANCE_REQUIRED if required else STATUS_CURRENT,
        documents=documents,
    )


def assert_portal_terms_current(*, user_id: int, client: object) -> None:
    """Refuse une nouvelle réservation PORTAL si une acceptation exigée manque."""
    if not is_portal_client(client):
        return
    user = db.session.get(User, user_id)
    if user is None:
        specs = portal_terms_catalog.current_portal_terms()
        raise PortalTermsReacceptanceRequired([spec.document_type for spec in specs])
    resolved = resolve_portal_terms_status(user)
    required = [
        document.document_type
        for document in resolved.documents
        if document.acceptance_required
    ]
    if required:
        raise PortalTermsReacceptanceRequired(required)


def accept_current_required_portal_terms(
    user: Any, client: Any
) -> tuple[list[ClientTermsAcceptance], bool]:
    """Insère uniquement les documents exigés. La même action ne duplique pas.

    Le verrou sur le compte sérialise deux requêtes concurrentes. Une version
    future publiée ensuite crée de nouvelles lignes : ce n'est pas un UPSERT.
    """
    if not is_portal_client(client):
        raise PortalTermsContextError(
            "Seuls les comptes client privé peuvent accepter ces conditions."
        )
    orm_user = User.query.filter_by(id=int(user.id)).with_for_update().one()
    resolved = resolve_portal_terms_status(orm_user)
    required_types = [
        document.document_type
        for document in resolved.documents
        if document.acceptance_required
    ]
    if not required_types:
        ids = [
            document.acceptance_id
            for document in resolved.documents
            if document.acceptance_id is not None
        ]
        if not ids:
            return [], False
        rows = (
            ClientTermsAcceptance.query.filter(ClientTermsAcceptance.id.in_(ids))
            .order_by(ClientTermsAcceptance.id.asc())
            .all()
        )
        return rows, False
    specs = [
        spec
        for spec in portal_terms_catalog.current_portal_terms()
        if spec.document_type in required_types
    ]
    created = record_portal_terms_acceptance(orm_user, client, documents=specs)
    return created, True
