"""Nom de fichier PDF facture personnalisé (téléchargement / Content-Disposition)."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

_MONTHS_FR = (
    "Janvier",
    "Fevrier",
    "Mars",
    "Avril",
    "Mai",
    "Juin",
    "Juillet",
    "Aout",
    "Septembre",
    "Octobre",
    "Novembre",
    "Decembre",
)


def slugify_invoice_filename_part(value: Any, *, max_len: int = 48) -> str:
    """Normalise un libellé pour un nom de fichier sûr (ASCII, underscores)."""
    if value is None:
        return ""
    normalized = unicodedata.normalize("NFKD", str(value))
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^A-Za-z0-9]+", "_", ascii_text).strip("_")
    slug = re.sub(r"_+", "_", slug)
    return slug[:max_len] if slug else ""


def format_invoice_amount_for_filename(amount: Any) -> str:
    """Ex. 155CHF ou 155_50CHF."""
    try:
        value = float(amount)
    except (TypeError, ValueError):
        return "0CHF"
    rounded = round(value * 100) / 100
    if abs(rounded - round(rounded)) < 1e-9:
        return f"{round(rounded)}CHF"
    return f"{rounded:.2f}".replace(".", "_") + "CHF"


def _client_label_from_mapping(data: dict[str, Any] | None) -> str:
    if not data:
        return ""
    if data.get("is_institution") and data.get("institution_name"):
        return str(data["institution_name"]).strip()
    patient = (data.get("patient_display_name") or "").strip()
    if patient:
        return patient
    last = (data.get("last_name") or "").strip()
    first = (data.get("first_name") or "").strip()
    if last or first:
        return " ".join(p for p in (last, first) if p)
    for key in ("display_name", "username", "institution_name"):
        raw = (data.get(key) or "").strip()
        if raw:
            return raw
    return ""


def resolve_invoice_filename_client_label(invoice: Any) -> str:
    """Libellé client / payeur pour le nom de fichier."""
    billing_party = getattr(invoice, "billing_party", None)
    if billing_party is not None:
        name = (getattr(billing_party, "display_name", None) or "").strip()
        if name:
            return name

    billed_to = getattr(invoice, "billed_to_company", None)
    if billed_to is not None:
        name = (getattr(billed_to, "name", None) or "").strip()
        if name:
            return name

    bill_to = getattr(invoice, "bill_to_client", None)
    if bill_to is not None:
        is_inst = bool(getattr(bill_to, "is_institution", False))
        inst_name = (getattr(bill_to, "institution_name", None) or "").strip()
        if is_inst and inst_name:
            return inst_name
        user = getattr(bill_to, "user", None)
        last = (
            (getattr(user, "last_name", None) or "").strip() if user is not None else ""
        )
        first = (
            (getattr(user, "first_name", None) or "").strip()
            if user is not None
            else ""
        )
        if last or first:
            return " ".join(p for p in (last, first) if p)
        username = (
            (getattr(user, "username", None) or "").strip() if user is not None else ""
        )
        if username:
            return username

    # Repli via sérialisation légère si relations absentes
    try:
        serialized = invoice._serialize_client()
        label = _client_label_from_mapping(serialized)
        if label:
            return label
    except Exception:
        pass

    return "Client"


def build_invoice_pdf_download_filename(invoice: Any) -> str:
    """Ex. Facture_Juillet_2026_EM-2026-07-0005_VUILLE_Michel_155CHF.pdf"""
    try:
        month_idx = int(getattr(invoice, "period_month", 0) or 0)
    except (TypeError, ValueError):
        month_idx = 0
    try:
        year = int(getattr(invoice, "period_year", 0) or 0)
    except (TypeError, ValueError):
        year = 0

    month_label = _MONTHS_FR[month_idx - 1] if 1 <= month_idx <= 12 else "Periode"
    year_label = str(year) if year > 0 else ""

    number_raw = (getattr(invoice, "invoice_number", None) or "").strip()
    # Conserver les tirets du n° (ex. EM-2026-07-0005)
    if number_raw:
        normalized = unicodedata.normalize("NFKD", number_raw)
        ascii_num = normalized.encode("ascii", "ignore").decode("ascii")
        number_part = re.sub(r'[/\\?%*:|"<>]', "-", ascii_num)
        number_part = re.sub(r"\s+", "_", number_part).strip("_")[:64]
    else:
        number_part = ""
    if not number_part:
        invoice_id = getattr(invoice, "id", None)
        number_part = f"ID_{invoice_id}" if invoice_id is not None else "SansNumero"

    client_part = (
        slugify_invoice_filename_part(
            resolve_invoice_filename_client_label(invoice), max_len=56
        )
        or "Client"
    )
    amount_part = format_invoice_amount_for_filename(
        getattr(invoice, "total_amount", 0)
    )

    parts = ["Facture", month_label]
    if year_label:
        parts.append(year_label)
    parts.extend([number_part, client_part, amount_part])
    return f"{'_'.join(parts)}.pdf"
