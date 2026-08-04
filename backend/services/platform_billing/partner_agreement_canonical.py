"""Documents canoniques CG / DPA : lecture, vérification SHA, pas de régénération."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from services.platform_billing.partner_agreement_versions import (
    DPA_VERSION,
    GENERAL_TERMS_VERSION,
)

PDF_MIME = "application/pdf"


class CanonicalDocumentError(Exception):
    """Artefact canonique manquant, altéré ou incohérent."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


@dataclass(frozen=True)
class CanonicalDocument:
    version: str
    kind: str  # "general_terms" | "dpa"
    pdf_path: Path
    sha256: str
    size_bytes: int
    content_type: str = PDF_MIME

    @property
    def pdf_bytes(self) -> bytes:
        return self.pdf_path.read_bytes()


def canonical_root() -> Path:
    return Path(__file__).resolve().parents[2] / "assets" / "contracts" / "canonical"


def canonical_pdf_path(version: str) -> Path:
    return canonical_root() / "pdf" / f"{version}.pdf"


def canonical_source_path(version: str) -> Path:
    return canonical_root() / "sources" / f"{version}.md"


def canonical_hashes_path() -> Path:
    return canonical_root() / "canonical_hashes.json"


def _load_expected_hashes() -> dict[str, Any]:
    path = canonical_hashes_path()
    if not path.is_file():
        raise CanonicalDocumentError(
            f"Manifeste des empreintes canoniques introuvable : {path}"
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CanonicalDocumentError(
            f"Manifeste canonique illisible : {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise CanonicalDocumentError("Manifeste canonique invalide")
    return data


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_one(version: str, kind: str, expected: dict[str, Any]) -> CanonicalDocument:
    pdf_path = canonical_pdf_path(version)
    if not pdf_path.is_file():
        raise CanonicalDocumentError(
            f"PDF canonique manquant pour {version} ({pdf_path})"
        )
    digest = _sha256_file(pdf_path)
    expected_sha = str(expected.get("sha256") or "")
    if not expected_sha:
        raise CanonicalDocumentError(
            f"SHA attendu manquant dans le manifeste pour {version}"
        )
    if digest != expected_sha:
        raise CanonicalDocumentError(
            f"PDF canonique altéré pour {version} : "
            f"SHA calculé {digest}, attendu {expected_sha}. "
            "Ne pas écraser une version existante ; publier une nouvelle version."
        )
    size = pdf_path.stat().st_size
    expected_size = expected.get("size_bytes")
    if expected_size is not None and int(expected_size) != size:
        raise CanonicalDocumentError(
            f"Taille PDF canonique incohérente pour {version}"
        )
    return CanonicalDocument(
        version=version,
        kind=kind,
        pdf_path=pdf_path,
        sha256=digest,
        size_bytes=size,
    )


def ensure_canonical_documents(
    *,
    general_terms_version: str = GENERAL_TERMS_VERSION,
    dpa_version: str = DPA_VERSION,
) -> dict[str, CanonicalDocument]:
    """Lit et vérifie les PDF canoniques. Ne régénère jamais un artefact."""
    manifest = _load_expected_hashes()
    terms_meta = manifest.get(general_terms_version)
    dpa_meta = manifest.get(dpa_version)
    if not isinstance(terms_meta, dict):
        raise CanonicalDocumentError(
            f"Version CG absente du manifeste : {general_terms_version}"
        )
    if not isinstance(dpa_meta, dict):
        raise CanonicalDocumentError(
            f"Version DPA absente du manifeste : {dpa_version}"
        )
    return {
        "general_terms": _verify_one(
            general_terms_version, "general_terms", terms_meta
        ),
        "dpa": _verify_one(dpa_version, "dpa", dpa_meta),
    }
