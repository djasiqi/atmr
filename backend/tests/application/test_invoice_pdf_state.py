"""Tests unitaires pour invoice_pdf_state (lecture pure get_pdf_state, merge meta)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from application.invoices.invoice_pdf_state import (
    PdfState,
    get_pdf_state,
    is_pdf_sendable,
    is_pdf_stale,
    mark_pdf_failed,
    mark_pdf_ready,
    mark_pdf_stale,
    normalize_invoice_meta_dict,
)
from models.enums import InvoiceStatus


def _inv(
    *,
    status=InvoiceStatus.DRAFT,
    pdf_url: str | None = None,
    meta: dict | str | None = None,
):
    m = MagicMock()
    m.status = status
    m.pdf_url = pdf_url
    m.meta = meta
    m.id = 1
    return m


def test_get_pdf_state_does_not_persist_defaults():
    inv = _inv(status=InvoiceStatus.DRAFT, pdf_url="/uploads/x.pdf", meta=None)
    before = inv.meta
    st = get_pdf_state(inv)
    assert isinstance(st, PdfState)
    assert st.status == "ready"
    assert inv.meta is before


def test_get_pdf_state_draft_no_pdf_url_infer_stale():
    inv = _inv(status=InvoiceStatus.DRAFT, pdf_url=None, meta=None)
    st = get_pdf_state(inv)
    assert st.status == "stale"


def test_get_pdf_state_sent_no_pdf_infer_failed():
    inv = _inv(status=InvoiceStatus.SENT, pdf_url=None, meta=None)
    st = get_pdf_state(inv)
    assert st.status == "failed"


def test_mark_pdf_ready_sets_pdf_url_and_meta():
    inv = _inv(meta={"vat": {"applicable": True}})
    mark_pdf_ready(inv, "/uploads/new.pdf")
    assert inv.pdf_url == "/uploads/new.pdf"
    meta = normalize_invoice_meta_dict(inv.meta)
    assert meta["vat"]["applicable"] is True
    assert meta["pdf"]["status"] == "ready"
    assert meta["pdf"]["error"] is None
    assert isinstance(meta["pdf"].get("content_updated_at"), str)


def test_mark_pdf_ready_preserves_content_updated_at():
    inv = _inv(
        meta={
            "pdf": {
                "status": "stale",
                "content_updated_at": "2026-01-15T10:00:00+00:00",
                "generated_at": None,
                "error": None,
            }
        }
    )
    mark_pdf_ready(inv, "/uploads/new.pdf")
    meta = normalize_invoice_meta_dict(inv.meta)
    assert meta["pdf"]["content_updated_at"] == "2026-01-15T10:00:00+00:00"
    assert meta["pdf"]["status"] == "ready"
    assert meta["pdf"]["generated_at"]


def test_mark_pdf_failed_preserves_pdf_url():
    inv = _inv(pdf_url="/uploads/old.pdf", meta=None)
    mark_pdf_failed(inv, "x" * 500)
    assert inv.pdf_url == "/uploads/old.pdf"
    assert get_pdf_state(inv).status == "failed"


def test_is_pdf_sendable_only_ready():
    inv = _inv(
        meta={
            "pdf": {
                "status": "stale",
                "generated_at": None,
                "content_updated_at": None,
                "error": None,
            }
        }
    )
    assert is_pdf_sendable(inv) is False
    inv2 = _inv(
        meta={
            "pdf": {
                "status": "ready",
                "generated_at": "2020-01-01T00:00:00+00:00",
                "content_updated_at": None,
                "error": None,
            }
        }
    )
    assert is_pdf_sendable(inv2) is True


def test_normalize_meta_json_string():
    inv = _inv(meta=json.dumps({"a": 1}))
    assert normalize_invoice_meta_dict(inv.meta) == {"a": 1}


def test_mark_pdf_stale_merge_keeps_other_meta_keys():
    inv = _inv(meta={"vat": {"x": 1}, "global_discount": {"percent": 10}})
    mark_pdf_stale(inv)
    meta = normalize_invoice_meta_dict(inv.meta)
    assert meta["vat"]["x"] == 1
    assert meta["global_discount"]["percent"] == 10
    assert meta["pdf"]["status"] == "stale"
