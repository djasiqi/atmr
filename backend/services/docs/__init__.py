"""Services de documentation (seed tenant Institution, hors produit)."""

from services.docs.institution_docs_seed import (
    DOCS_EMAIL_DOMAIN,
    DOCS_USER_EMAIL,
    reset_and_seed_institution_docs,
)

__all__ = [
    "DOCS_EMAIL_DOMAIN",
    "DOCS_USER_EMAIL",
    "reset_and_seed_institution_docs",
]
