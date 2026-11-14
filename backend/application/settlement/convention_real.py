"""Convention Client réelle : création, signature, archivage, applicabilité.

Réutilise le stockage documentaire local (UPLOAD_FOLDER) déjà employé pour
les spécimens de démonstration. Ne lève jamais ``legally_effective`` tant
que ``CONVENTION_LEGAL_ACTIVATION`` n'est pas explicitement ouvert.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import UTC, date, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

from domain.end_of_trip_settlement.convention_real_lifecycle import (
    ConventionLifecycleRefusal,
    assert_transition,
    deferred_billing_gate,
    frozen_document_must_not_change,
)
from ext import db
from models.client import Client
from models.company import Company
from models.patient_company_convention import (
    PatientCompanyConvention,
    PatientCompanyConventionEvent,
)

DOCUMENT_TITLE = "Convention de transport et facturation différée"
DEFAULT_BILLING = (
    "Facturation différée selon les conditions convenues avec l'entreprise "
    "de transport. Paiement immédiat toujours possible sans signature."
)


class ConventionRealRefusal(Exception):
    def __init__(self, code: str, message: str, http_status: int = 400) -> None:
        self.code = code
        self.message = message
        self.http_status = http_status
        super().__init__(message)


def legal_activation_unlocked() -> bool:
    """Activation juridique fermée par défaut. Aucune certification inventée."""
    raw = (os.getenv("CONVENTION_LEGAL_ACTIVATION") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _archive_root() -> Path:
    from flask import current_app, has_app_context

    configured = None
    if has_app_context():
        configured = current_app.config.get("UPLOAD_FOLDER")
    return Path(configured or "/app/uploads")


def archive_reference(row: PatientCompanyConvention) -> str:
    return (
        f"conventions/{int(row.company_id)}/"
        f"{int(row.id)}-v{int(row.version_number)}.pdf"
    )


def _client_for_company(company_id: int, client_id: int) -> Client:
    client = db.session.get(Client, int(client_id))
    if client is None or int(client.company_id) != int(company_id):
        raise ConventionRealRefusal(
            "client_autre_entreprise",
            "Ce client n'appartient pas à cette entreprise.",
            404,
        )
    return client


def _company_name(company_id: int) -> str:
    company = db.session.get(Company, int(company_id))
    return str(getattr(company, "name", None) or "Entreprise")


def _active(company_id: int, client_id: int) -> PatientCompanyConvention | None:
    return (
        db.session.query(PatientCompanyConvention)
        .filter_by(
            company_id=int(company_id),
            client_id=int(client_id),
            active=True,
        )
        .one_or_none()
    )


def _lock(convention_id: int, company_id: int) -> PatientCompanyConvention:
    row = (
        db.session.query(PatientCompanyConvention)
        .filter_by(id=int(convention_id), company_id=int(company_id))
        .with_for_update()
        .one_or_none()
    )
    if row is None:
        raise ConventionRealRefusal(
            "convention_absente", "Convention introuvable.", 404
        )
    return row


def _append_event(
    row: PatientCompanyConvention,
    *,
    action: str,
    detail: str,
    actor_role: str,
    actor_id: int | None,
    idempotency_key: str,
) -> None:
    existing = (
        db.session.query(PatientCompanyConventionEvent)
        .filter_by(convention_id=int(row.id), idempotency_key=idempotency_key[:80])
        .one_or_none()
    )
    if existing is not None:
        return
    db.session.add(
        PatientCompanyConventionEvent(
            convention_id=int(row.id),
            idempotency_key=idempotency_key[:80],
            actor_role=actor_role,
            actor_id=actor_id,
            action=action,
            detail=detail,
        )
    )


def _canonical_body(row: PatientCompanyConvention, company_name: str) -> str:
    """Texte exact présenté / haché. Toute modification change l'empreinte."""
    recipient = row.invoice_recipient_label or "(même que le débiteur)"
    return "\n".join(
        [
            DOCUMENT_TITLE,
            f"Entreprise : {company_name}",
            f"Client id : {int(row.client_id)}",
            f"Débiteur : {row.debtor_label}",
            f"Destinataire facture : {recipient}",
            f"Conditions : {row.billing_conditions}",
            f"Tarif convenu : {row.tariff_version}",
            f"Périmètre : {row.service_scope}",
            f"Version document : {row.document_version}",
            f"Version n° : {int(row.version_number)}",
            f"Validité du : {row.valid_from or 'signature'}",
            f"Validité jusqu'au : {row.valid_until or 'résiliation / remplacement'}",
            "Une convention applicable peut couvrir plusieurs transports.",
            "Un destinataire de facture n'est pas automatiquement débiteur ni signataire.",
            "Signature manuscrite à l'écran : modalité de capture, pas une certification.",
        ]
    )


def document_sha256_for(row: PatientCompanyConvention, company_name: str) -> str:
    return hashlib.sha256(
        _canonical_body(row, company_name).encode("utf-8")
    ).hexdigest()


def _pdf_bytes(row: PatientCompanyConvention, company_name: str) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    buffer = BytesIO()
    sheet = canvas.Canvas(buffer, pagesize=A4)
    sheet.setTitle(DOCUMENT_TITLE)
    y = 800
    sheet.setFont("Helvetica-Bold", 12)
    sheet.drawString(40, y, DOCUMENT_TITLE[:90])
    y -= 24
    sheet.setFont("Helvetica", 10)
    for line in _canonical_body(row, company_name).split("\n"):
        safe = line.encode("latin-1", "replace").decode("latin-1")[:110]
        sheet.drawString(40, y, safe)
        y -= 14
        if y < 60:
            sheet.showPage()
            y = 800
            sheet.setFont("Helvetica", 10)
    sheet.setFont("Helvetica", 9)
    y -= 10
    sheet.drawString(40, y, f"Empreinte SHA-256 : {row.document_sha256 or 'en cours'}")
    y -= 14
    sheet.drawString(
        40,
        y,
        (
            f"Signé le {row.signed_at.isoformat() if row.signed_at else '—'} "
            f"par {row.signer_kind or '—'} / {row.signer_name or '—'}"
        )[:110],
    )
    y -= 14
    sheet.drawString(
        40,
        y,
        (
            f"Statut : {row.status} · legally_effective={bool(row.legally_effective)} "
            f"· activation={'ouverte' if legal_activation_unlocked() else 'fermée'}"
        )[:110],
    )
    sheet.showPage()
    sheet.save()
    return buffer.getvalue()


def _store_pdf(relative: str, payload: bytes) -> None:
    if not payload.startswith(b"%PDF"):
        raise ConventionRealRefusal("pdf_illisible", "Document PDF illisible.")
    target = _archive_root() / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_bytes(payload)
    temporary.replace(target)


def _read_pdf(relative: str | None) -> bytes | None:
    if not relative or not relative.startswith("conventions/"):
        return None
    path = _archive_root() / relative
    if not path.is_file():
        return None
    return path.read_bytes()


def _public(
    row: PatientCompanyConvention, *, company_name: str | None = None
) -> dict[str, Any]:
    name = company_name or _company_name(int(row.company_id))
    gate = deferred_billing_gate(
        status=str(row.status),
        legally_effective=bool(row.legally_effective),
        legal_activation_unlocked=legal_activation_unlocked(),
    )
    return {
        "id": int(row.id),
        "public_id": row.public_id,
        "company_id": int(row.company_id),
        "company_name": name,
        "client_id": int(row.client_id),
        "debtor_id": row.debtor_id,
        "debtor_label": row.debtor_label,
        "invoice_recipient_id": row.invoice_recipient_id,
        "invoice_recipient_label": row.invoice_recipient_label,
        "billing_conditions": row.billing_conditions,
        "tariff_version": row.tariff_version,
        "service_scope": row.service_scope,
        "document_version": row.document_version,
        "version_number": int(row.version_number),
        "active": bool(row.active),
        "status": row.status,
        "legally_effective": bool(row.legally_effective),
        "legal_activation_required": bool(row.legal_activation_required),
        "legal_activation_unlocked": legal_activation_unlocked(),
        "consent_acknowledged": bool(row.consent_acknowledged),
        "signer_kind": row.signer_kind,
        "signer_name": row.signer_name,
        "signature_modality": row.signature_modality,
        "signed_at": None if row.signed_at is None else row.signed_at.isoformat(),
        "document_sha256": row.document_sha256,
        "archive_reference": row.archive_reference,
        "paper_scan_reference": row.paper_scan_reference,
        "valid_from": None if row.valid_from is None else row.valid_from.isoformat(),
        "valid_until": None if row.valid_until is None else row.valid_until.isoformat(),
        "presented_text": _canonical_body(row, name),
        "mandate_kind": gate.mandate_kind,
        "authorizes_deferred_billing": gate.authorizes_deferred_billing,
        "deferred_reason": gate.reason,
        "disclaimer_legal": (
            "Capture de signature et archivage techniques. "
            "Valeur probatoire / signature électronique suisse : validation juridique requise. "
            "Facturation différée réelle fermée tant que l'activation n'est pas approuvée."
        ),
    }


def list_for_client(company_id: int, client_id: int) -> dict[str, Any]:
    _client_for_company(company_id, client_id)
    rows = (
        db.session.query(PatientCompanyConvention)
        .filter_by(company_id=int(company_id), client_id=int(client_id))
        .order_by(PatientCompanyConvention.version_number.desc())
        .all()
    )
    name = _company_name(company_id)
    return {
        "company_id": int(company_id),
        "client_id": int(client_id),
        "conventions": [_public(row, company_name=name) for row in rows],
        "active": None
        if not any(r.active for r in rows)
        else _public(next(r for r in rows if r.active), company_name=name),
    }


def get_convention(company_id: int, convention_id: int) -> dict[str, Any]:
    row = (
        db.session.query(PatientCompanyConvention)
        .filter_by(id=int(convention_id), company_id=int(company_id))
        .one_or_none()
    )
    if row is None:
        raise ConventionRealRefusal(
            "convention_absente", "Convention introuvable.", 404
        )
    events = (
        db.session.query(PatientCompanyConventionEvent)
        .filter_by(convention_id=int(row.id))
        .order_by(PatientCompanyConventionEvent.id)
        .all()
    )
    body = _public(row)
    body["history"] = [
        {
            "action": e.action,
            "detail": e.detail,
            "actor_role": e.actor_role,
            "at": None if e.created_at is None else e.created_at.isoformat(),
        }
        for e in events
    ]
    return body


def create_draft(
    company_id: int,
    client_id: int,
    payload: dict[str, Any],
    *,
    actor_id: int | None,
) -> dict[str, Any]:
    _client_for_company(company_id, client_id)
    key = str(payload.get("idempotency_key") or "").strip()[:80]
    if len(key) < 8:
        raise ConventionRealRefusal("cle_absente", "Clé d'idempotence requise.")
    existing = (
        db.session.query(PatientCompanyConvention)
        .filter_by(company_id=int(company_id), idempotency_key=key)
        .one_or_none()
    )
    if existing is not None:
        view = _public(existing)
        view["replay"] = True
        return view

    debtor_label = str(payload.get("debtor_label") or "").strip()[:160]
    if len(debtor_label) < 2:
        raise ConventionRealRefusal("debiteur_absent", "Libellé débiteur requis.")
    recipient_label = (
        str(payload.get("invoice_recipient_label") or "").strip()[:160] or None
    )
    # Destinataire ≠ débiteur automatique
    if recipient_label and recipient_label == debtor_label:
        # autorisé s'ils sont la même personne, mais pas de promotion silencieuse
        pass

    active = _active(company_id, client_id)
    version = 1
    if active is not None:
        if active.status == "draft":
            # Mise à jour du brouillon actif uniquement
            active.debtor_label = debtor_label
            active.debtor_id = payload.get("debtor_id")
            active.invoice_recipient_id = payload.get("invoice_recipient_id")
            active.invoice_recipient_label = recipient_label
            active.billing_conditions = str(
                payload.get("billing_conditions") or DEFAULT_BILLING
            ).strip()
            active.tariff_version = str(
                payload.get("tariff_version") or "standard"
            ).strip()[:64]
            active.service_scope = str(
                payload.get("service_scope") or "Transports de cette entreprise."
            ).strip()
            active.document_version = str(
                payload.get("document_version") or "v1"
            ).strip()[:32]
            active.valid_from = _parse_date(payload.get("valid_from"))
            active.valid_until = _parse_date(payload.get("valid_until"))
            _append_event(
                active,
                action="draft_updated",
                detail="Brouillon mis à jour.",
                actor_role="company",
                actor_id=actor_id,
                idempotency_key=f"upd-{key}",
            )
            db.session.commit()
            view = _public(active)
            view["replay"] = False
            return view
        # Nouvelle version : désactive l'ancienne sans l'écraser
        assert_transition(active.status, "replaced") if active.status in {
            "applicable",
            "expired",
            "revoked",
        } or active.status == "applicable" else None
        # Remplacement depuis n'importe quel statut actif non draft
        previous_status = active.status
        active.active = False
        if previous_status not in {"revoked", "replaced", "expired"}:
            try:
                assert_transition(previous_status, "replaced")
                active.status = "replaced"
            except ConventionLifecycleRefusal:
                pass
        version = int(active.version_number) + 1

    row = PatientCompanyConvention(
        public_id=str(uuid.uuid4()),
        company_id=int(company_id),
        client_id=int(client_id),
        debtor_id=payload.get("debtor_id"),
        debtor_label=debtor_label,
        invoice_recipient_id=payload.get("invoice_recipient_id"),
        invoice_recipient_label=recipient_label,
        billing_conditions=str(
            payload.get("billing_conditions") or DEFAULT_BILLING
        ).strip(),
        tariff_version=str(payload.get("tariff_version") or "standard").strip()[:64],
        service_scope=str(
            payload.get("service_scope") or "Transports de cette entreprise."
        ).strip(),
        document_version=str(payload.get("document_version") or "v1").strip()[:32],
        version_number=version,
        active=True,
        status="draft",
        idempotency_key=key,
        legally_effective=False,
        legal_activation_required=True,
        consent_acknowledged=False,
        signature_modality="none",
        valid_from=_parse_date(payload.get("valid_from")),
        valid_until=_parse_date(payload.get("valid_until")),
    )
    db.session.add(row)
    db.session.flush()
    _append_event(
        row,
        action="created",
        detail=f"Convention v{version} créée en brouillon.",
        actor_role="company",
        actor_id=actor_id,
        idempotency_key=key,
    )
    db.session.commit()
    view = _public(row)
    view["replay"] = False
    return view


def submit_for_signature(
    company_id: int,
    convention_id: int,
    *,
    actor_id: int | None,
    idempotency_key: str,
) -> dict[str, Any]:
    row = _lock(convention_id, company_id)
    key = str(idempotency_key or "").strip()[:80]
    if len(key) < 8:
        raise ConventionRealRefusal("cle_absente", "Clé d'idempotence requise.")
    if row.status == "awaiting_signature":
        view = _public(row)
        view["replay"] = True
        return view
    try:
        assert_transition(row.status, "awaiting_signature")
    except ConventionLifecycleRefusal as exc:
        raise ConventionRealRefusal(exc.code, exc.message) from exc
    name = _company_name(company_id)
    row.document_sha256 = document_sha256_for(row, name)
    row.status = "awaiting_signature"
    _append_event(
        row,
        action="submitted_for_signature",
        detail="Document figé pour consultation avant signature.",
        actor_role="company",
        actor_id=actor_id,
        idempotency_key=key,
    )
    db.session.commit()
    view = _public(row)
    view["replay"] = False
    return view


def _parse_date(value: object) -> date | None:
    if not value or not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _trace(raw: object) -> list[dict[str, float]]:
    if not isinstance(raw, list) or not raw:
        raise ConventionRealRefusal("trace_vide", "Tracé de signature vide.")
    if len(raw) > 400:
        raise ConventionRealRefusal("trace_trop_long", "Tracé trop long.")
    points: list[dict[str, float]] = []
    for item in raw:
        if not isinstance(item, dict) or "x" not in item or "y" not in item:
            raise ConventionRealRefusal("trace_illisible", "Tracé illisible.")
        points.append({"x": float(item["x"]), "y": float(item["y"])})
    return points


def sign_convention(
    company_id: int,
    convention_id: int,
    payload: dict[str, Any],
    *,
    actor_role: str,
    actor_id: int | None,
) -> dict[str, Any]:
    """Consentement + identité + tracé + horodatage + PDF figé.

    Passe en ``signed_pending_validation``. Ne devient pas ``applicable``
    automatiquement. ``legally_effective`` reste false.
    """
    row = _lock(convention_id, company_id)
    key = str(payload.get("idempotency_key") or "").strip()[:80]
    if len(key) < 8:
        raise ConventionRealRefusal("cle_absente", "Clé d'idempotence requise.")
    if row.status == "signed_pending_validation" and row.archive_reference:
        view = _public(row)
        view["replay"] = True
        return view
    try:
        assert_transition(row.status, "signed_pending_validation")
    except ConventionLifecycleRefusal as exc:
        raise ConventionRealRefusal(exc.code, exc.message) from exc
    if payload.get("acknowledged") is not True:
        raise ConventionRealRefusal(
            "consentement_absent",
            "Le consentement explicite à la version affichée est requis.",
        )
    signer_kind = str(payload.get("signer_kind") or "").strip()
    if signer_kind not in {"client", "representative"}:
        raise ConventionRealRefusal("signataire_inconnu", "Signataire non reconnu.")
    signer_name = str(payload.get("signer_name") or "").strip()[:160]
    if len(signer_name) < 2:
        raise ConventionRealRefusal(
            "signataire_nom_absent", "Nom du signataire requis."
        )
    presented = str(payload.get("document_sha256_presented") or "").strip()
    name = _company_name(company_id)
    current_hash = document_sha256_for(row, name)
    if row.document_sha256 and row.document_sha256 != current_hash:
        raise ConventionRealRefusal(
            "document_altere",
            "Le document a changé depuis la présentation.",
        )
    if presented and presented != current_hash:
        raise ConventionRealRefusal(
            "version_presentee_differente",
            "La version consultée ne correspond pas au document à signer.",
        )
    points = _trace(payload.get("trace"))
    row.consent_acknowledged = True
    row.signer_kind = signer_kind
    row.signer_name = signer_name
    row.signature_modality = "on_screen_trace"
    row.signed_at = datetime.now(UTC)
    row.document_sha256 = current_hash
    row.signature_trace = {
        "points": points,
        "tactile_reference": hashlib.sha256(
            json.dumps(points, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16],
        "kind": "on_screen_trace",
        "not_a_qualified_electronic_signature": True,
    }
    reference = archive_reference(row)
    try:
        payload_pdf = _pdf_bytes(row, name)
        _store_pdf(reference, payload_pdf)
    except OSError as exc:
        db.session.rollback()
        raise ConventionRealRefusal(
            "archive_indisponible",
            "Archivage impossible. La signature n'est pas enregistrée.",
        ) from exc
    stored = _read_pdf(reference)
    if stored != payload_pdf:
        db.session.rollback()
        raise ConventionRealRefusal(
            "archive_indisponible",
            "Vérification d'archive échouée. Signature non enregistrée.",
        )
    row.archive_reference = reference
    row.status = "signed_pending_validation"
    row.legally_effective = False
    _append_event(
        row,
        action="signed_on_screen",
        detail=(
            "Signature à l'écran capturée et PDF figé. "
            "Pas une signature électronique qualifiée. "
            "En attente de validation administrative / juridique."
        ),
        actor_role=actor_role,
        actor_id=actor_id,
        idempotency_key=key,
    )
    db.session.commit()
    view = _public(row)
    view["replay"] = False
    return view


def refuse_signature(
    company_id: int,
    convention_id: int,
    payload: dict[str, Any],
    *,
    actor_role: str,
    actor_id: int | None,
) -> dict[str, Any]:
    row = _lock(convention_id, company_id)
    key = str(payload.get("idempotency_key") or "").strip()[:80]
    reason = str(payload.get("reason") or "").strip()
    if len(key) < 8 or not (3 <= len(reason) <= 500):
        raise ConventionRealRefusal("motif_absent", "Motif de refus requis.")
    if row.status != "awaiting_signature":
        raise ConventionRealRefusal(
            "refus_impossible",
            "Seul un document en attente de signature peut être refusé.",
        )
    try:
        assert_transition(row.status, "draft")
    except ConventionLifecycleRefusal as exc:
        raise ConventionRealRefusal(exc.code, exc.message) from exc
    row.status = "draft"
    _append_event(
        row,
        action="signature_refused",
        detail=f"Signature refusée ou interrompue : {reason}",
        actor_role=actor_role,
        actor_id=actor_id,
        idempotency_key=key,
    )
    db.session.commit()
    view = _public(row)
    view["replay"] = False
    return view


def register_paper_scan(
    company_id: int,
    convention_id: int,
    payload: dict[str, Any],
    *,
    actor_id: int | None,
) -> dict[str, Any]:
    """Dépôt admin d'une convention papier scannée — contrôle administratif requis."""
    row = _lock(convention_id, company_id)
    key = str(payload.get("idempotency_key") or "").strip()[:80]
    note = str(payload.get("receipt_note") or "").strip()
    scan_ref = str(payload.get("paper_scan_reference") or "").strip()[:255]
    if len(key) < 8 or (not scan_ref and len(note) < 3):
        raise ConventionRealRefusal(
            "preuve_papier_absente",
            "Référence de scan ou note administrative requise.",
        )
    if row.status not in {"draft", "awaiting_signature", "signed_pending_validation"}:
        raise ConventionRealRefusal(
            "papier_refuse",
            "Dépôt papier impossible pour ce statut.",
        )
    if row.status != "signed_pending_validation":
        try:
            assert_transition(row.status, "signed_pending_validation")
        except ConventionLifecycleRefusal as exc:
            raise ConventionRealRefusal(exc.code, exc.message) from exc
    name = _company_name(company_id)
    row.document_sha256 = document_sha256_for(row, name)
    row.signature_modality = "paper_scan"
    row.paper_scan_reference = scan_ref or f"paper-note:{note[:80]}"
    row.consent_acknowledged = True
    row.signer_kind = str(payload.get("signer_kind") or "client")[:32]
    row.signer_name = str(payload.get("signer_name") or row.debtor_label)[:160]
    row.signed_at = datetime.now(UTC)
    row.status = "signed_pending_validation"
    row.legally_effective = False
    reference = archive_reference(row)
    payload_pdf = _pdf_bytes(row, name)
    _store_pdf(reference, payload_pdf)
    row.archive_reference = reference
    _append_event(
        row,
        action="paper_scan_registered",
        detail=(
            "Scan papier enregistré. Validation administrative requise. "
            "Aucune valeur juridique automatique."
        ),
        actor_role="company",
        actor_id=actor_id,
        idempotency_key=key,
    )
    db.session.commit()
    view = _public(row)
    view["replay"] = False
    return view


def validate_applicable(
    company_id: int,
    convention_id: int,
    *,
    actor_id: int | None,
    idempotency_key: str,
) -> dict[str, Any]:
    """Passe en applicable. N'active PAS legally_effective (gate fermée)."""
    row = _lock(convention_id, company_id)
    key = str(idempotency_key or "").strip()[:80]
    if len(key) < 8:
        raise ConventionRealRefusal("cle_absente", "Clé d'idempotence requise.")
    if row.status == "applicable":
        view = _public(row)
        view["replay"] = True
        return view
    try:
        assert_transition(row.status, "applicable")
    except ConventionLifecycleRefusal as exc:
        raise ConventionRealRefusal(exc.code, exc.message) from exc
    if not row.archive_reference or not row.document_sha256:
        raise ConventionRealRefusal(
            "archive_requise",
            "Une convention applicable exige un document figé archivé.",
        )
    if row.valid_until is not None and row.valid_until < date.today():
        raise ConventionRealRefusal(
            "deja_expiree",
            "La période de validité est déjà dépassée.",
        )
    row.status = "applicable"
    # Gate juridique : ne jamais lever ici sans activation explicite.
    if legal_activation_unlocked() and not row.legal_activation_required:
        row.legally_effective = True
    else:
        row.legally_effective = False
    if row.valid_from is None:
        row.valid_from = date.today()
    _append_event(
        row,
        action="marked_applicable",
        detail=(
            "Statut applicable (technique). "
            f"legally_effective={row.legally_effective}. "
            "Différé réel fermé tant que l'activation juridique n'est pas ouverte."
        ),
        actor_role="company",
        actor_id=actor_id,
        idempotency_key=key,
    )
    db.session.commit()
    view = _public(row)
    view["replay"] = False
    return view


def revoke(
    company_id: int,
    convention_id: int,
    *,
    actor_id: int | None,
    idempotency_key: str,
    reason: str,
) -> dict[str, Any]:
    row = _lock(convention_id, company_id)
    key = str(idempotency_key or "").strip()[:80]
    if len(key) < 8 or len(reason.strip()) < 3:
        raise ConventionRealRefusal("motif_absent", "Motif de révocation requis.")
    try:
        assert_transition(row.status, "revoked")
    except ConventionLifecycleRefusal as exc:
        raise ConventionRealRefusal(exc.code, exc.message) from exc
    previous_hash = row.document_sha256 or ""
    row.status = "revoked"
    row.active = False
    row.legally_effective = False
    if previous_hash:
        frozen_document_must_not_change(
            previous_sha256=previous_hash,
            candidate_sha256=previous_hash,
            status="revoked",
        )
    _append_event(
        row,
        action="revoked",
        detail=reason.strip()[:500],
        actor_role="company",
        actor_id=actor_id,
        idempotency_key=key,
    )
    db.session.commit()
    return _public(row)


def expire_if_needed(row: PatientCompanyConvention) -> PatientCompanyConvention:
    if (
        row.status == "applicable"
        and row.valid_until is not None
        and row.valid_until < date.today()
    ):
        try:
            assert_transition(row.status, "expired")
        except ConventionLifecycleRefusal:
            return row
        row.status = "expired"
        row.legally_effective = False
        _append_event(
            row,
            action="expired",
            detail="Convention expirée par date de fin.",
            actor_role="system",
            actor_id=None,
            idempotency_key=f"expire-{row.id}-{row.valid_until}",
        )
        db.session.commit()
    return row


def read_pdf(company_id: int, convention_id: int) -> bytes:
    row = (
        db.session.query(PatientCompanyConvention)
        .filter_by(id=int(convention_id), company_id=int(company_id))
        .one_or_none()
    )
    if row is None:
        raise ConventionRealRefusal(
            "convention_absente", "Convention introuvable.", 404
        )
    stored = _read_pdf(row.archive_reference)
    if stored is None:
        raise ConventionRealRefusal(
            "archive_absente", "Document archivé indisponible.", 404
        )
    return stored


def gate_for_client(company_id: int, client_id: int) -> dict[str, Any]:
    """Lecture gate différé pour clôture chauffeur / facturation."""
    row = _active(company_id, client_id)
    if row is None:
        gate = deferred_billing_gate(
            status="draft",
            legally_effective=False,
            legal_activation_unlocked=legal_activation_unlocked(),
        )
        # sans ligne : absent
        return {
            "mandate_kind": "absent",
            "legally_effective": False,
            "authorizes_deferred_billing": False,
            "reason": "convention_inexistante",
            "convention_id": None,
        }
    expire_if_needed(row)
    db.session.refresh(row)
    gate = deferred_billing_gate(
        status=str(row.status),
        legally_effective=bool(row.legally_effective),
        legal_activation_unlocked=legal_activation_unlocked(),
    )
    return {
        "mandate_kind": gate.mandate_kind,
        "legally_effective": gate.legally_effective,
        "authorizes_deferred_billing": gate.authorizes_deferred_billing,
        "reason": gate.reason,
        "convention_id": int(row.id),
        "status": row.status,
    }
