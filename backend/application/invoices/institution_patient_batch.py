"""Batch idempotent de factures patients — consomme institution_invoice_plan.

Une seule source de vérité : les buckets patients du plan certifié.
Ne recalcule ni le payeur, ni l'éligibilité Market, ni le regroupement A/R.

Idempotence : verrou advisory session (backend/DB) + clé de scope persistée
sur ``invoice.meta``. Un second POST identique réutilise les mêmes drafts.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Literal
from unittest.mock import MagicMock

from sqlalchemy import text

from application.invoices.active_invoice_claim import (
    BLOCKING_INVOICE_STATUSES_FOR_CLAIM,
    find_blocking_invoice_claims,
)
from application.invoices.generate_invoice import (
    GenerateInvoiceInput,
    GenerateInvoiceUseCase,
)
from application.invoices.institution_invoice_eligibility import PayerBucket
from application.invoices.institution_invoice_plan import build_institution_invoice_plan
from ext import db
from models import Invoice, InvoiceLine
from models.enums import InvoiceBillingStrategy
from models.invoice import _ordered_unique_booking_ids_from_lines

logger = logging.getLogger(__name__)

BATCH_SCOPE_META_KEY = "institution_patient_batch_scope"
_ADVISORY_NAMESPACE = "institution-patient-batch-v1"
PatientResultKind = Literal["created", "existing", "failed", "skipped"]


def patient_batch_scope_key(
    *,
    company_id: int,
    clinic_company_id: int,
    period_year: int,
    period_month: int,
    institution_patient_id: int | None,
    billing_party_id: int | None,
    client_id: int | None,
    bucket_key: str,
) -> str:
    """Clé d'idempotence d'un débiteur patient sur une période institutionnelle."""
    subject = (
        f"ipid:{int(institution_patient_id)}"
        if institution_patient_id is not None
        else f"key:{bucket_key}:bpid:{billing_party_id or 0}:cid:{client_id or 0}"
    )
    return (
        f"ipb:{int(company_id)}:{int(clinic_company_id)}:"
        f"{int(period_year):04d}-{int(period_month):02d}:{subject}"
    )


def _advisory_pair(lock_name: str) -> tuple[int, int]:
    digest = hashlib.sha256(f"{_ADVISORY_NAMESPACE}:{lock_name}".encode()).digest()
    return (
        int.from_bytes(digest[:4], "big") & 0x7FFFFFFF,
        int.from_bytes(digest[4:8], "big") & 0x7FFFFFFF,
    )


class _SessionAdvisoryLock:
    """Verrou PostgreSQL de session sur une connexion dédiée.

    ``generate_invoice`` commit en interne et peut rendre la connexion
    SQLAlchemy au pool : le verrou ne doit pas vivre sur ``db.session``.
    """

    def __init__(self, lock_name: str) -> None:
        self._pair = _advisory_pair(lock_name)
        self._conn: Any = None

    def __enter__(self) -> _SessionAdvisoryLock:
        self._conn = db.engine.connect()
        self._conn.execute(
            text("SELECT pg_advisory_lock(:a, :b)"),
            {"a": self._pair[0], "b": self._pair[1]},
        )
        return self

    def __exit__(self, *_exc: object) -> None:
        conn = self._conn
        self._conn = None
        if conn is None:
            return
        try:
            conn.execute(
                text("SELECT pg_advisory_unlock(:a, :b)"),
                {"a": self._pair[0], "b": self._pair[1]},
            )
        except Exception:
            logger.exception("Échec pg_advisory_unlock batch patients")
        finally:
            conn.close()


def _as_int_list(raw: Any) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    if raw is None:
        return out
    for item in raw:
        try:
            value = int(item)
        except (TypeError, ValueError):
            continue
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _invoice_booking_ids(invoice: Invoice) -> list[int]:
    lines = list(getattr(invoice, "lines", None) or [])
    if not lines:
        lines = InvoiceLine.query.filter_by(invoice_id=int(invoice.id)).all()
    return _ordered_unique_booking_ids_from_lines(lines)


def _invoice_total_ht(invoice: Invoice) -> float:
    raw = getattr(invoice, "subtotal_amount", None)
    if raw is None:
        raw = getattr(invoice, "total_amount", None)
    try:
        return float(Decimal(str(raw)).quantize(Decimal("0.01")))
    except Exception:
        return 0.0


def _status_value(invoice: Invoice) -> str:
    status = getattr(invoice, "status", None)
    return str(getattr(status, "value", status) or "")


@dataclass
class PatientBatchItem:
    patient_id: int | None
    bucket_key: str
    invoice_id: int | None
    status: str | None
    booking_ids: list[int]
    total_ht: float
    result: PatientResultKind
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "patient_id": self.patient_id,
            "invoice_id": self.invoice_id,
            "status": self.status,
            "booking_ids": list(self.booking_ids),
            "total_ht": self.total_ht,
            "result": self.result,
        }
        if self.error:
            payload["error"] = self.error
        return payload


@dataclass
class InstitutionPatientBatchResult:
    period_year: int
    period_month: int
    institution_id: int | None
    clinic_company_id: int
    requested_patient_count: int
    created_count: int
    reused_count: int
    skipped_count: int
    failed_count: int
    invoices: list[PatientBatchItem] = field(default_factory=list)
    failures: list[PatientBatchItem] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "period_year": self.period_year,
            "period_month": self.period_month,
            "period": f"{self.period_year:04d}-{self.period_month:02d}",
            "institution_id": self.institution_id,
            "clinic_company_id": self.clinic_company_id,
            "requested_patient_count": self.requested_patient_count,
            "patient_count": self.requested_patient_count,
            "created_count": self.created_count,
            "reused_count": self.reused_count,
            "skipped_count": self.skipped_count,
            "failed_count": self.failed_count,
            "invoices": [item.to_dict() for item in self.invoices],
            "failures": [item.to_dict() for item in self.failures],
        }


@dataclass(frozen=True, slots=True)
class InstitutionPatientBatchInput:
    company_id: int
    clinic_company_id: int
    period_year: int
    period_month: int
    clinic_client_id: int | None = None
    institution_patient_ids: list[int] | None = None
    patient_bucket_keys: list[str] | None = None


def _default_pdf_service() -> MagicMock:
    pdf = MagicMock()
    pdf.generate_invoice_pdf.return_value = None
    return pdf


def _match_bucket(
    bucket: PayerBucket,
    *,
    selected_ipids: set[int] | None,
    selected_keys: set[str] | None,
) -> bool:
    if selected_ipids is None and selected_keys is None:
        return True
    if (
        selected_ipids is not None
        and bucket.institution_patient_id is not None
        and int(bucket.institution_patient_id) in selected_ipids
    ):
        return True
    return bool(selected_keys is not None and bucket.key in selected_keys)


def _find_existing_invoice(
    *,
    company_id: int,
    period_year: int,
    period_month: int,
    scope: str,
    institution_patient_id: int | None,
    billing_party_id: int | None,
) -> Invoice | None:
    query = Invoice.query.filter(
        Invoice.company_id == int(company_id),
        Invoice.period_year == int(period_year),
        Invoice.period_month == int(period_month),
        Invoice.billing_strategy == InvoiceBillingStrategy.S1_PATIENT,
        Invoice.status.in_(tuple(BLOCKING_INVOICE_STATUSES_FOR_CLAIM)),
    )
    if institution_patient_id is not None:
        query = query.filter(
            Invoice.institution_patient_id == int(institution_patient_id)
        )
    elif billing_party_id is not None:
        query = query.filter(
            Invoice.billing_party_id == int(billing_party_id),
            Invoice.institution_patient_id.is_(None),
        )
    candidates = query.order_by(Invoice.id.asc()).all()
    scoped = [
        inv
        for inv in candidates
        if isinstance(inv.meta, dict) and inv.meta.get(BATCH_SCOPE_META_KEY) == scope
    ]
    if scoped:
        return scoped[0]
    if institution_patient_id is not None and candidates:
        return candidates[0]
    return None


def _list_scoped_period_invoices(
    *,
    company_id: int,
    clinic_company_id: int,
    period_year: int,
    period_month: int,
) -> list[Invoice]:
    prefix = (
        f"ipb:{int(company_id)}:{int(clinic_company_id)}:"
        f"{int(period_year):04d}-{int(period_month):02d}:"
    )
    rows = (
        Invoice.query.filter(
            Invoice.company_id == int(company_id),
            Invoice.period_year == int(period_year),
            Invoice.period_month == int(period_month),
            Invoice.billing_strategy == InvoiceBillingStrategy.S1_PATIENT,
            Invoice.status.in_(tuple(BLOCKING_INVOICE_STATUSES_FOR_CLAIM)),
        )
        .order_by(Invoice.id.asc())
        .all()
    )
    found: list[Invoice] = []
    for invoice in rows:
        meta = invoice.meta if isinstance(invoice.meta, dict) else {}
        scope = str(meta.get(BATCH_SCOPE_META_KEY) or "")
        if scope.startswith(prefix):
            found.append(invoice)
    return found


def _item_from_invoice(
    invoice: Invoice,
    bucket: PayerBucket,
    result: PatientResultKind,
) -> PatientBatchItem:
    return PatientBatchItem(
        patient_id=bucket.institution_patient_id,
        bucket_key=bucket.key,
        invoice_id=int(invoice.id),
        status=_status_value(invoice),
        booking_ids=_invoice_booking_ids(invoice),
        total_ht=_invoice_total_ht(invoice),
        result=result,
    )


class InstitutionPatientBatchUseCase:
    """Prépare 1 draft patient par débiteur du plan, sans envoi."""

    def __init__(
        self,
        *,
        generate_uc: GenerateInvoiceUseCase | None = None,
        pdf_service: Any | None = None,
    ) -> None:
        if generate_uc is not None:
            self._generate_uc = generate_uc
        else:
            self._generate_uc = GenerateInvoiceUseCase(
                pdf_service=pdf_service or _default_pdf_service()
            )

    def execute(
        self,
        input_data: InstitutionPatientBatchInput,
        *,
        now: datetime | None = None,
    ) -> InstitutionPatientBatchResult:
        plan = build_institution_invoice_plan(
            company_id=input_data.company_id,
            period_year=input_data.period_year,
            period_month=input_data.period_month,
            clinic_company_id=input_data.clinic_company_id,
            clinic_client_id=input_data.clinic_client_id,
            now=now,
        )
        selected_ipids: set[int] | None = None
        if input_data.institution_patient_ids is not None:
            selected_ipids = set(_as_int_list(input_data.institution_patient_ids))
        selected_keys: set[str] | None = None
        if input_data.patient_bucket_keys is not None:
            selected_keys = {
                str(k) for k in input_data.patient_bucket_keys if str(k).strip()
            }

        selected_explicit = selected_ipids is not None or selected_keys is not None
        selected_buckets = [
            bucket
            for bucket in plan.patients
            if _match_bucket(
                bucket,
                selected_ipids=selected_ipids,
                selected_keys=selected_keys,
            )
        ]
        skipped_in_plan = [
            bucket
            for bucket in plan.patients
            if selected_explicit
            and not _match_bucket(
                bucket,
                selected_ipids=selected_ipids,
                selected_keys=selected_keys,
            )
        ]

        requested_ids = selected_ipids or set()
        if selected_explicit:
            requested_count = (
                len(requested_ids)
                if selected_ipids is not None
                else len(selected_keys or ())
            )
            if selected_ipids is not None and selected_keys:
                requested_count = len(
                    {
                        *(f"id:{i}" for i in selected_ipids),
                        *(f"key:{k}" for k in selected_keys),
                    }
                )
        else:
            requested_count = len(plan.patients)

        institution_id = None
        if input_data.clinic_client_id is not None:
            from models import Client

            clinic_client = db.session.get(Client, int(input_data.clinic_client_id))
            linked = getattr(clinic_client, "linked_institution_id", None)
            if linked:
                institution_id = int(linked)

        created: list[PatientBatchItem] = []
        reused: list[PatientBatchItem] = []
        failed: list[PatientBatchItem] = []
        skipped_items: list[PatientBatchItem] = []

        for bucket in skipped_in_plan:
            skipped_items.append(
                PatientBatchItem(
                    patient_id=bucket.institution_patient_id,
                    bucket_key=bucket.key,
                    invoice_id=None,
                    status=None,
                    booking_ids=list(bucket.booking_ids),
                    total_ht=float(bucket.estimated_total),
                    result="skipped",
                    error="patient_non_selectionne",
                )
            )

        lock_name = (
            f"period:{input_data.company_id}:{input_data.clinic_company_id}:"
            f"{input_data.period_year}:{input_data.period_month}"
        )
        with _SessionAdvisoryLock(lock_name):
            known_ipids = {
                int(b.institution_patient_id)
                for b in plan.patients
                if b.institution_patient_id is not None
            }
            if selected_ipids:
                for ipid in selected_ipids:
                    if ipid in known_ipids:
                        continue
                    existing = _find_existing_invoice(
                        company_id=input_data.company_id,
                        period_year=input_data.period_year,
                        period_month=input_data.period_month,
                        scope=patient_batch_scope_key(
                            company_id=input_data.company_id,
                            clinic_company_id=input_data.clinic_company_id,
                            period_year=input_data.period_year,
                            period_month=input_data.period_month,
                            institution_patient_id=ipid,
                            billing_party_id=None,
                            client_id=None,
                            bucket_key=f"patient:{ipid}:0",
                        ),
                        institution_patient_id=ipid,
                        billing_party_id=None,
                    )
                    phantom = PayerBucket(
                        payer_type="patient",
                        key=f"patient:{ipid}:0",
                        display_name=f"Patient #{ipid}",
                        transports_count=0,
                        estimated_total=0.0,
                        institution_patient_id=ipid,
                    )
                    if existing is not None:
                        reused.append(_item_from_invoice(existing, phantom, "existing"))
                    else:
                        skipped_items.append(
                            PatientBatchItem(
                                patient_id=ipid,
                                bucket_key=phantom.key,
                                invoice_id=None,
                                status=None,
                                booking_ids=[],
                                total_ht=0.0,
                                result="skipped",
                                error="absent_du_plan",
                            )
                        )

            for bucket in selected_buckets:
                item = self._process_bucket(
                    input_data=input_data,
                    bucket=bucket,
                    now=now,
                )
                if item.result == "created":
                    created.append(item)
                elif item.result == "existing":
                    reused.append(item)
                elif item.result == "skipped":
                    skipped_items.append(item)
                else:
                    failed.append(item)

            if not selected_explicit:
                already = {item.invoice_id for item in (*created, *reused)}
                for invoice in _list_scoped_period_invoices(
                    company_id=input_data.company_id,
                    clinic_company_id=input_data.clinic_company_id,
                    period_year=input_data.period_year,
                    period_month=input_data.period_month,
                ):
                    if int(invoice.id) in already:
                        continue
                    phantom = PayerBucket(
                        payer_type="patient",
                        key=f"patient:{invoice.institution_patient_id or invoice.id}:0",
                        display_name=f"Patient #{invoice.institution_patient_id}",
                        transports_count=0,
                        estimated_total=_invoice_total_ht(invoice),
                        institution_patient_id=invoice.institution_patient_id,
                        client_id=invoice.client_id,
                        billing_party_id=invoice.billing_party_id,
                    )
                    reused.append(_item_from_invoice(invoice, phantom, "existing"))
                requested_count = max(requested_count, len(created) + len(reused))

        invoices = [*created, *reused]
        return InstitutionPatientBatchResult(
            period_year=input_data.period_year,
            period_month=input_data.period_month,
            institution_id=institution_id,
            clinic_company_id=input_data.clinic_company_id,
            requested_patient_count=requested_count,
            created_count=len(created),
            reused_count=len(reused),
            skipped_count=len(skipped_items),
            failed_count=len(failed),
            invoices=invoices,
            failures=failed,
        )

    def _process_bucket(
        self,
        *,
        input_data: InstitutionPatientBatchInput,
        bucket: PayerBucket,
        now: datetime | None,
    ) -> PatientBatchItem:
        scope = patient_batch_scope_key(
            company_id=input_data.company_id,
            clinic_company_id=input_data.clinic_company_id,
            period_year=input_data.period_year,
            period_month=input_data.period_month,
            institution_patient_id=bucket.institution_patient_id,
            billing_party_id=bucket.billing_party_id,
            client_id=bucket.client_id,
            bucket_key=bucket.key,
        )
        existing = _find_existing_invoice(
            company_id=input_data.company_id,
            period_year=input_data.period_year,
            period_month=input_data.period_month,
            scope=scope,
            institution_patient_id=bucket.institution_patient_id,
            billing_party_id=bucket.billing_party_id,
        )
        if existing is not None:
            return _item_from_invoice(existing, bucket, "existing")

        booking_ids = [int(i) for i in (bucket.booking_ids or [])]
        if not booking_ids:
            return PatientBatchItem(
                patient_id=bucket.institution_patient_id,
                bucket_key=bucket.key,
                invoice_id=None,
                status=None,
                booking_ids=[],
                total_ht=0.0,
                result="skipped",
                error="aucune_prestation",
            )
        if bucket.client_id is None:
            return PatientBatchItem(
                patient_id=bucket.institution_patient_id,
                bucket_key=bucket.key,
                invoice_id=None,
                status=None,
                booking_ids=booking_ids,
                total_ht=float(bucket.estimated_total),
                result="failed",
                error="client_porteur_manquant",
            )

        claims = find_blocking_invoice_claims(booking_ids)
        if claims:
            owner_ids = {int(c.invoice_id) for c in claims.values()}
            compatible: Invoice | None = None
            if len(owner_ids) == 1:
                owner = db.session.get(Invoice, next(iter(owner_ids)))
                if owner is not None and self._invoice_matches_bucket(
                    owner, bucket, scope
                ):
                    compatible = owner
            if compatible is not None:
                return _item_from_invoice(compatible, bucket, "existing")
            return PatientBatchItem(
                patient_id=bucket.institution_patient_id,
                bucket_key=bucket.key,
                invoice_id=None,
                status=None,
                booking_ids=booking_ids,
                total_ht=float(bucket.estimated_total),
                result="failed",
                error="prestation_deja_facturee",
            )

        try:
            output = self._generate_uc.execute(
                GenerateInvoiceInput(
                    company_id=input_data.company_id,
                    client_id=int(bucket.client_id),
                    period_year=input_data.period_year,
                    period_month=input_data.period_month,
                    billing_party_id=bucket.billing_party_id,
                    reservation_ids=booking_ids,
                    institution_patient_id=bucket.institution_patient_id,
                    strict_reservation_ids=True,
                    invoice_meta_extra={BATCH_SCOPE_META_KEY: scope},
                ),
                now=now,
            )
        except ValueError as exc:
            return PatientBatchItem(
                patient_id=bucket.institution_patient_id,
                bucket_key=bucket.key,
                invoice_id=None,
                status=None,
                booking_ids=booking_ids,
                total_ht=float(bucket.estimated_total),
                result="failed",
                error=str(exc),
            )

        if not output.success:
            existing_id = (output.error or {}).get("existing_invoice_id")
            if output.status_code == 409 and existing_id:
                reused = db.session.get(Invoice, int(existing_id))
                if reused is not None:
                    self._ensure_scope(reused, scope)
                    return _item_from_invoice(reused, bucket, "existing")
            return PatientBatchItem(
                patient_id=bucket.institution_patient_id,
                bucket_key=bucket.key,
                invoice_id=int(existing_id) if existing_id else None,
                status=None,
                booking_ids=booking_ids,
                total_ht=float(bucket.estimated_total),
                result="failed",
                error=str((output.error or {}).get("error") or "generation_echouee"),
            )

        invoice = output.invoice or db.session.get(Invoice, int(output.invoice_id or 0))
        if invoice is None:
            return PatientBatchItem(
                patient_id=bucket.institution_patient_id,
                bucket_key=bucket.key,
                invoice_id=output.invoice_id,
                status=None,
                booking_ids=booking_ids,
                total_ht=float(bucket.estimated_total),
                result="failed",
                error="facture_introuvable_apres_creation",
            )
        self._ensure_scope(invoice, scope)
        created = _item_from_invoice(invoice, bucket, "created")
        created_ids = set(created.booking_ids)
        expected = set(booking_ids)
        if created_ids != expected:
            return PatientBatchItem(
                patient_id=bucket.institution_patient_id,
                bucket_key=bucket.key,
                invoice_id=created.invoice_id,
                status=created.status,
                booking_ids=created.booking_ids,
                total_ht=created.total_ht,
                result="failed",
                error="conservation_booking_ids",
            )
        return created

    @staticmethod
    def _invoice_matches_bucket(
        invoice: Invoice, bucket: PayerBucket, scope: str
    ) -> bool:
        meta = invoice.meta if isinstance(invoice.meta, dict) else {}
        if meta.get(BATCH_SCOPE_META_KEY) == scope:
            return True
        return bool(
            bucket.institution_patient_id is not None
            and invoice.institution_patient_id == bucket.institution_patient_id
        )

    @staticmethod
    def _ensure_scope(invoice: Invoice, scope: str) -> None:
        meta = dict(invoice.meta) if isinstance(invoice.meta, dict) else {}
        if meta.get(BATCH_SCOPE_META_KEY) == scope:
            return
        meta[BATCH_SCOPE_META_KEY] = scope
        invoice.meta = meta
        db.session.add(invoice)
        db.session.commit()
