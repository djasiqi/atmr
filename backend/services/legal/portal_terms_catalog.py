"""Versions canoniques des CGU et des CGV du client privé.

Le texte vivant du frontend (``TermsOfService.jsx``) n'est pas une source
contractuelle : c'est une page mutable, et elle ne contient pas les CGV de
transport du client privé. Chaque version publiée ici a un corps figé et une
empreinte SHA-256 attendue. Un écart entre le fichier et l'empreinte bloque
la publication au lieu de recalculer silencieusement le hash.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from models.client_terms_acceptance import (
    DOCUMENT_TERMS_OF_SERVICE,
    DOCUMENT_TRANSPORT_TERMS,
    TERMS_LOCALE_FR_CH,
)

_CANONICAL_DIR = Path(__file__).resolve().parents[2] / "legal" / "canonical" / "fr"

TERMS_OF_SERVICE_V1_SHA256 = (
    "68c1002662ead280f4a7729bc2ad17051408c9d6861439a4155205f2fc4472fb"
)
TRANSPORT_TERMS_V1_SHA256 = (
    "45ae4ed0a6c2fdbf62ff0c6ca5c07f3e13bdffa71f4819bf6df8df0dee25fa7f"
)


class CatalogIntegrityError(Exception):
    """Le corps canonique ne correspond plus à l'empreinte publiée."""


@dataclass(frozen=True)
class PublishedTerms:
    document_type: str
    terms_version: str
    terms_hash: str
    canonical_body: str
    locale: str = TERMS_LOCALE_FR_CH


def canonical_sha256(body: str) -> str:
    """Empreinte SHA-256 hex du texte canonique UTF-8, retours ligne LF."""
    normalized = body.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _load_frozen(filename: str, expected_hash: str) -> str:
    raw = (_CANONICAL_DIR / filename).read_text(encoding="utf-8")
    normalized = raw.replace("\r\n", "\n").replace("\r", "\n")
    digest = canonical_sha256(normalized)
    if digest != expected_hash:
        raise CatalogIntegrityError(
            f"Empreinte inattendue pour {filename}: le fichier n'est pas la version publiée."
        )
    return normalized


def current_portal_terms() -> tuple[PublishedTerms, PublishedTerms]:
    """CGU et CGV de transport actuellement opposables au client privé."""
    terms_of_service = _load_frozen(
        "terms_of_service_v1.0.txt", TERMS_OF_SERVICE_V1_SHA256
    )
    transport_terms = _load_frozen(
        "transport_terms_v1.0.txt", TRANSPORT_TERMS_V1_SHA256
    )
    return (
        PublishedTerms(
            document_type=DOCUMENT_TERMS_OF_SERVICE,
            terms_version="1.0",
            terms_hash=TERMS_OF_SERVICE_V1_SHA256,
            canonical_body=terms_of_service,
        ),
        PublishedTerms(
            document_type=DOCUMENT_TRANSPORT_TERMS,
            terms_version="1.0",
            terms_hash=TRANSPORT_TERMS_V1_SHA256,
            canonical_body=transport_terms,
        ),
    )
