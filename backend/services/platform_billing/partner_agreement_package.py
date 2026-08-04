"""Assemblage ZIP déterministe du dossier de remise partenaire."""

from __future__ import annotations

import hashlib
import zipfile
from io import BytesIO


# Horodatage ZIP fixe (epoch DOS) pour un SHA stable.
_FIXED_DATE_TIME = (2020, 1, 1, 0, 0, 0)
_FIXED_EXTERNAL_ATTR = 0o644 << 16


def _safe_ref_token(reference: str) -> str:
    return (
        (reference or "LIRIE_PART")
        .replace("/", "_")
        .replace(" ", "_")
        .replace("\\", "_")
    )


def build_delivery_zip_bytes(
    *,
    reference: str,
    manifest_pdf: bytes,
    particular_pdf: bytes,
    general_terms_pdf: bytes,
    dpa_pdf: bytes,
    general_terms_version: str,
    dpa_version: str,
) -> bytes:
    """Construit un ZIP déterministe (ordre, dates, compression fixes)."""
    ref = _safe_ref_token(reference)
    terms_short = general_terms_version.replace("lirie-partner-terms-", "LIRIE_")
    dpa_short = dpa_version.replace("lirie-dpa-", "LIRIE_")
    entries = [
        (f"00_Bordereau-remise_{ref}.pdf", manifest_pdf),
        (f"01_Contrat-particulier_{ref}.pdf", particular_pdf),
        (f"02_Conditions-generales_{terms_short}.pdf", general_terms_pdf),
        (f"03_Accord-traitement-donnees_{dpa_short}.pdf", dpa_pdf),
    ]
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries:
            info = zipfile.ZipInfo(filename=name, date_time=_FIXED_DATE_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = _FIXED_EXTERNAL_ATTR
            zf.writestr(info, data)
    return buffer.getvalue()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def delivery_zip_filename(reference: str) -> str:
    return f"{_safe_ref_token(reference)}_Dossier-remise.zip"
