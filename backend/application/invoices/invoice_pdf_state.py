"""État PDF facture sous ``invoice.meta["pdf"]`` : lecture pure, merge sûr, envoi."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from models import Invoice
from models.enums import InvoiceStatus

logger = logging.getLogger(__name__)

PdfStatus = Literal["ready", "stale", "failed", "pending"]

_META_PDF_KEY = "pdf"
_MAX_META_ERROR_LEN = 240


@dataclass(frozen=True, slots=True)
class PdfState:
    """État PDF effectif (persisté ou inféré — ``get_pdf_state`` ne persiste pas)."""

    status: PdfStatus
    generated_at: str | None
    content_updated_at: str | None
    error: str | None


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _sanitize_error_for_meta(message: str, *, max_len: int = _MAX_META_ERROR_LEN) -> str:
    msg = (message or "").strip().replace("\n", " ")
    if len(msg) > max_len:
        return msg[: max_len - 1] + "…"
    return msg or "PDF_ERROR"


def normalize_invoice_meta_dict(meta: Any) -> dict[str, Any]:
    """Convertit ``invoice.meta`` en dict mutable (JSON string → parse, sinon {})."""
    if meta is None:
        return {}
    if isinstance(meta, dict):
        return dict(meta)
    if isinstance(meta, str):
        try:
            parsed = json.loads(meta)
            return dict(parsed) if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, TypeError):
            logger.warning("invoice.meta string JSON invalide, traité comme vide")
            return {}
    return {}


def _set_pdf_meta(invoice: Invoice, pdf_meta: dict[str, Any]) -> None:
    meta = normalize_invoice_meta_dict(invoice.meta)
    meta[_META_PDF_KEY] = pdf_meta
    invoice.meta = meta


def _persisted_pdf_blob(invoice: Invoice) -> dict[str, Any] | None:
    meta = normalize_invoice_meta_dict(invoice.meta)
    raw = meta.get(_META_PDF_KEY)
    return raw if isinstance(raw, dict) else None


def _invoice_status_value(invoice: Invoice) -> str:
    st = invoice.status
    return st.value if hasattr(st, "value") else str(st)


def _locked_non_draft(invoice: Invoice) -> bool:
    return _invoice_status_value(invoice) in (
        InvoiceStatus.SENT.value,
        InvoiceStatus.PARTIALLY_PAID.value,
        InvoiceStatus.PAID.value,
    )


def get_pdf_state(invoice: Invoice) -> PdfState:
    """Retourne l'état PDF effectif sans modifier ``invoice.meta`` (lecture pure)."""
    blob = _persisted_pdf_blob(invoice)
    pdf_url = (invoice.pdf_url or "").strip()

    if blob:
        raw_status = blob.get("status")
        status: PdfStatus
        if raw_status in ("ready", "stale", "failed", "pending"):
            status = raw_status
        else:
            status = "ready"
        ga = blob.get("generated_at")
        cu = blob.get("content_updated_at")
        err = blob.get("error")
        return PdfState(
            status=status,
            generated_at=ga if isinstance(ga, str) else None,
            content_updated_at=cu if isinstance(cu, str) else None,
            error=err if isinstance(err, str) else None,
        )

    # Inférence conservative sans persistance
    if pdf_url and _locked_non_draft(invoice):
        return PdfState(status="ready", generated_at=None, content_updated_at=None, error=None)

    if pdf_url and _invoice_status_value(invoice) == InvoiceStatus.DRAFT.value:
        # Legacy sans meta.pdf : prêt par défaut ; mark_pdf_stale après mutations reprend la main.
        return PdfState(status="ready", generated_at=None, content_updated_at=None, error=None)

    if not pdf_url and _invoice_status_value(invoice) == InvoiceStatus.DRAFT.value:
        return PdfState(status="stale", generated_at=None, content_updated_at=None, error=None)

    if not pdf_url and _locked_non_draft(invoice):
        return PdfState(
            status="failed",
            generated_at=None,
            content_updated_at=None,
            error="MISSING_PDF",
        )

    return PdfState(status="ready", generated_at=None, content_updated_at=None, error=None)


def is_pdf_stale(invoice: Invoice) -> bool:
    return get_pdf_state(invoice).status == "stale"


def is_pdf_sendable(invoice: Invoice) -> bool:
    return get_pdf_state(invoice).status == "ready"


def mark_pdf_stale(invoice: Invoice, *, content_changed: bool = True) -> None:
    now = _utc_now_iso()
    prev = _persisted_pdf_blob(invoice) or {}
    pdf_meta: dict[str, Any] = {
        "status": "stale",
        "generated_at": prev.get("generated_at") if isinstance(prev.get("generated_at"), str) else None,
        "content_updated_at": now if content_changed else prev.get("content_updated_at"),
        "error": None,
    }
    if not content_changed and isinstance(prev.get("content_updated_at"), str):
        pdf_meta["content_updated_at"] = prev.get("content_updated_at")
    _set_pdf_meta(invoice, pdf_meta)


def mark_pdf_ready(invoice: Invoice, pdf_url: str) -> None:
    """Met à jour ``pdf_url`` ; ``content_updated_at`` reste la dernière modification facturable."""
    now = _utc_now_iso()
    prev = _persisted_pdf_blob(invoice) or {}
    prev_cu = prev.get("content_updated_at")
    content_updated_at = prev_cu if isinstance(prev_cu, str) else now
    invoice.pdf_url = pdf_url
    _set_pdf_meta(
        invoice,
        {
            "status": "ready",
            "generated_at": now,
            "content_updated_at": content_updated_at,
            "error": None,
        },
    )


def mark_pdf_failed(invoice: Invoice, error: str) -> None:
    prev = _persisted_pdf_blob(invoice) or {}
    short = _sanitize_error_for_meta(error)
    _set_pdf_meta(
        invoice,
        {
            "status": "failed",
            "generated_at": prev.get("generated_at") if isinstance(prev.get("generated_at"), str) else None,
            "content_updated_at": prev.get("content_updated_at")
            if isinstance(prev.get("content_updated_at"), str)
            else None,
            "error": short,
        },
    )


def ensure_draft_pdf_ready_for_send(invoice: Invoice) -> tuple[bool, str | None]:  # noqa: PLR0911
    """Brouillon : refuse si ``failed`` ; régénère si nécessaire. Ne fait pas ``commit``."""
    if _invoice_status_value(invoice) != InvoiceStatus.DRAFT.value:
        return True, None

    st = get_pdf_state(invoice)
    if st.status == "failed":
        return False, st.error or "PDF indisponible. Régénérez le document."

    if st.status == "pending":
        return False, "Génération PDF en attente."

    if is_pdf_sendable(invoice):
        return True, None

    from services.documents.pdf import PDFService

    try:
        pdf = PDFService()
        url = pdf.generate_invoice_pdf(invoice, force_regenerate=True)
        if not url:
            logger.error(
                "ensure_draft_pdf_ready_for_send: PDF vide invoice_id=%s",
                invoice.id,
            )
            mark_pdf_failed(invoice, "PDF_EMPTY")
            return False, "Échec de la génération du PDF."
        mark_pdf_ready(invoice, url)
        return True, None
    except Exception as e:
        logger.exception(
            "ensure_draft_pdf_ready_for_send invoice_id=%s",
            getattr(invoice, "id", None),
        )
        mark_pdf_failed(invoice, _sanitize_error_for_meta(str(e)))
        return False, "Échec de la génération du PDF."
