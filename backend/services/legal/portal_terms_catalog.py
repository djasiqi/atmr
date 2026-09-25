"""Versions canoniques des CGU et des CGV du client privé.

Le texte vivant du frontend (``TermsOfService.jsx``) n'est pas une source
contractuelle : c'est une page mutable, et elle ne contient pas les CGV de
transport du client privé. Chaque version publiée ici a un corps figé et une
empreinte SHA-256 attendue. Un écart entre le fichier et l'empreinte bloque
la publication au lieu de recalculer silencieusement le hash.

Séparation PREPARED vs CURRENT
------------------------------
``current_portal_terms()`` est le pointeur explicite des versions opposables.
Ajouter un fichier 2.0 (ou ``prepared_portal_terms_v2()``) **ne** rend **pas**
2.0 courant. Le basculement vers 2.0 se fait uniquement via
``PORTAL_TERMS_EFFECTIVE_VERSION=2.0``, coordonné avec
``PORTAL_DOUBLE_VALIDATION_ENABLED=true`` (voir ``assert_activation_coordination``).
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, replace
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
TERMS_OF_SERVICE_V2_SHA256 = (
    "97942048422ddd18078f02f533f427b9db048bed5ac6a33fcd8b64277f4139e2"
)
TRANSPORT_TERMS_V2_SHA256 = (
    "5e502ef659fe35c3a65c2ff288e99092ddc213ca159559bdf195ceafdac9bc12"
)
TERMS_OF_SERVICE_V21_SHA256 = (
    "d08fa9431073c1b189cbe8de9f5c4df2d5a63f2e549377eb02f499ceacba694e"
)
TRANSPORT_TERMS_V21_SHA256 = (
    "7c28686c5c107c26d29a65e9e5217d92fa3b8445aad50009dadfebde5b5d4f81"
)

_EFFECTIVE_PORTAL_TERMS_VERSION_ENV = "PORTAL_TERMS_EFFECTIVE_VERSION"


class CatalogIntegrityError(Exception):
    """Le corps canonique ne correspond plus à l'empreinte publiée."""


class PortalTermsActivationError(Exception):
    """Basculement 2.0 / double validation incohérent ou interdit."""


@dataclass(frozen=True)
class PublishedTerms:
    document_type: str
    terms_version: str
    terms_hash: str
    canonical_body: str
    locale: str = TERMS_LOCALE_FR_CH
    requires_reacceptance: bool = True
    status: str = "current"  # current | prepared


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


def portal_terms_v1() -> tuple[PublishedTerms, PublishedTerms]:
    """Documents 1.0 — historiques, immuables."""
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
            requires_reacceptance=True,
            status="current",
        ),
        PublishedTerms(
            document_type=DOCUMENT_TRANSPORT_TERMS,
            terms_version="1.0",
            terms_hash=TRANSPORT_TERMS_V1_SHA256,
            canonical_body=transport_terms,
            requires_reacceptance=True,
            status="current",
        ),
    )


def prepared_portal_terms_v2() -> tuple[PublishedTerms, PublishedTerms]:
    """Documents 2.0 préparés — catalogue uniquement, non opposables par défaut."""
    terms_of_service = _load_frozen(
        "terms_of_service_v2.0.txt", TERMS_OF_SERVICE_V2_SHA256
    )
    transport_terms = _load_frozen(
        "transport_terms_v2.0.txt", TRANSPORT_TERMS_V2_SHA256
    )
    return (
        PublishedTerms(
            document_type=DOCUMENT_TERMS_OF_SERVICE,
            terms_version="2.0",
            terms_hash=TERMS_OF_SERVICE_V2_SHA256,
            canonical_body=terms_of_service,
            requires_reacceptance=True,
            status="prepared",
        ),
        PublishedTerms(
            document_type=DOCUMENT_TRANSPORT_TERMS,
            terms_version="2.0",
            terms_hash=TRANSPORT_TERMS_V2_SHA256,
            canonical_body=transport_terms,
            requires_reacceptance=True,
            status="prepared",
        ),
    )


def prepared_portal_terms_v21() -> tuple[PublishedTerms, PublishedTerms]:
    """Documents 2.1 préparés — conditional_order_v1 ; non opposables par défaut."""
    terms_of_service = _load_frozen(
        "terms_of_service_v2.1.txt", TERMS_OF_SERVICE_V21_SHA256
    )
    transport_terms = _load_frozen(
        "transport_terms_v2.1.txt", TRANSPORT_TERMS_V21_SHA256
    )
    return (
        PublishedTerms(
            document_type=DOCUMENT_TERMS_OF_SERVICE,
            terms_version="2.1",
            terms_hash=TERMS_OF_SERVICE_V21_SHA256,
            canonical_body=terms_of_service,
            requires_reacceptance=True,
            status="prepared",
        ),
        PublishedTerms(
            document_type=DOCUMENT_TRANSPORT_TERMS,
            terms_version="2.1",
            terms_hash=TRANSPORT_TERMS_V21_SHA256,
            canonical_body=transport_terms,
            requires_reacceptance=True,
            status="prepared",
        ),
    )


def effective_portal_terms_version() -> str:
    """Version courante opposable (défaut 1.0). Jamais « max version »."""
    raw = None
    try:
        from flask import current_app, has_app_context

        if has_app_context():
            raw = current_app.config.get("PORTAL_TERMS_EFFECTIVE_VERSION")
    except Exception:
        raw = None
    if raw is None or str(raw).strip() == "":
        raw = os.getenv(_EFFECTIVE_PORTAL_TERMS_VERSION_ENV) or "1.0"
    raw = str(raw).strip()
    if raw not in ("1.0", "2.0", "2.1"):
        raise CatalogIntegrityError(f"PORTAL_TERMS_EFFECTIVE_VERSION invalide: {raw!r}")
    return raw


def _flag_bool(app_key: str, env_key: str) -> bool:
    try:
        from flask import current_app, has_app_context

        if has_app_context():
            return bool(current_app.config.get(app_key, False))
    except Exception:
        pass
    return (os.getenv(env_key) or "false").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def is_portal_double_validation_enabled_for_activation() -> bool:
    """Lit le flag DV (app config si disponible, sinon env)."""
    return _flag_bool(
        "PORTAL_DOUBLE_VALIDATION_ENABLED", "PORTAL_DOUBLE_VALIDATION_ENABLED"
    )


def is_portal_conditional_order_enabled_for_activation() -> bool:
    """Lit le flag commande conditionnelle 7B.5."""
    return _flag_bool(
        "PORTAL_CONDITIONAL_ORDER_ENABLED", "PORTAL_CONDITIONAL_ORDER_ENABLED"
    )


def assert_activation_coordination(
    *,
    effective_version: str | None = None,
    double_validation_enabled: bool | None = None,
    conditional_order_enabled: bool | None = None,
) -> None:
    """Gate 7D / 7B.5 : flags mutuellement exclusifs + couple terms cohérent.

    Refus au boot si DV et conditional sont tous deux ON (pas de priorité silencieuse).
    """
    version = effective_version or effective_portal_terms_version()
    if double_validation_enabled is None:
        dv_on = is_portal_double_validation_enabled_for_activation()
    else:
        dv_on = bool(double_validation_enabled)
    if conditional_order_enabled is None:
        co_on = is_portal_conditional_order_enabled_for_activation()
    else:
        co_on = bool(conditional_order_enabled)

    if dv_on and co_on:
        raise PortalTermsActivationError(
            "PORTAL_DOUBLE_VALIDATION_ENABLED et "
            "PORTAL_CONDITIONAL_ORDER_ENABLED ne peuvent pas être true ensemble"
        )

    if version == "2.1":
        if not co_on:
            raise PortalTermsActivationError(
                "terms 2.1 effective interdit tant que "
                "PORTAL_CONDITIONAL_ORDER_ENABLED=false"
            )
        if dv_on:
            raise PortalTermsActivationError(
                "terms 2.1 incompatible avec PORTAL_DOUBLE_VALIDATION_ENABLED"
            )
    elif version == "2.0":
        if not dv_on:
            raise PortalTermsActivationError(
                "terms 2.0 effective interdit tant que "
                "PORTAL_DOUBLE_VALIDATION_ENABLED=false"
            )
        if co_on:
            raise PortalTermsActivationError(
                "terms 2.0 incompatible avec PORTAL_CONDITIONAL_ORDER_ENABLED"
            )
    else:
        # 1.0
        if dv_on:
            raise PortalTermsActivationError(
                "double validation ON interdit tant que les conditions "
                "courantes ne sont pas 2.0"
            )
        if co_on:
            raise PortalTermsActivationError(
                "conditional order ON interdit tant que les conditions "
                "courantes ne sont pas 2.1"
            )


def enforce_portal_activation_coordination_at_startup(app) -> None:
    """Fail-fast au boot si le couple terms / flags flux est incohérent."""
    version = str(app.config.get("PORTAL_TERMS_EFFECTIVE_VERSION") or "1.0").strip()
    dv_on = bool(app.config.get("PORTAL_DOUBLE_VALIDATION_ENABLED", False))
    co_on = bool(app.config.get("PORTAL_CONDITIONAL_ORDER_ENABLED", False))
    assert_activation_coordination(
        effective_version=version,
        double_validation_enabled=dv_on,
        conditional_order_enabled=co_on,
    )


def current_portal_terms() -> tuple[PublishedTerms, PublishedTerms]:
    """CGU et CGV actuellement opposables.

    Défaut : 1.0. La présence de fichiers 2.x PREPARED n'a aucun effet.
    """
    version = effective_portal_terms_version()
    if version == "2.1":
        tos, transport = prepared_portal_terms_v21()
        return (
            replace(tos, status="current"),
            replace(transport, status="current"),
        )
    if version == "2.0":
        tos, transport = prepared_portal_terms_v2()
        return (
            replace(tos, status="current"),
            replace(transport, status="current"),
        )
    return portal_terms_v1()
