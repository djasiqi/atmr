"""Registre fictif patient–entreprise. Aucune facture, aucun mandat valide."""

from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path
from typing import Any

from domain.end_of_trip_settlement.recipe_simulation import (
    recipe_simulation_read_allowed,
)
from ext import db
from models.client import Client
from models.company import Company
from models.convention_demo import PatientCompanyConventionDemo

DISCLAIMER = "DOCUMENT DE DÉMONSTRATION — SANS VALEUR CONTRACTUELLE"
TARIFF_VERSION = "demo-2026.1"
SERVICE_SCOPE = (
    "Transports simulés de cette entreprise uniquement. "
    "La convention d'une autre entreprise ne couvre jamais ces courses."
)
DURATION_LABEL = "Spécimen sans durée juridique."
DOCUMENT_VERSION = "specimen-1"


class ConventionClosed(Exception):
    """La recette locale n'est pas ouverte."""


class ConventionRefusal(Exception):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


def _require_open() -> None:
    if not recipe_simulation_read_allowed():
        raise ConventionClosed
    # Création sur la connexion de la session : une seconde connexion pendant
    # une transaction ouverte s'interbloquerait sur les clés étrangères.
    PatientCompanyConventionDemo.__table__.create(
        bind=db.session.connection(), checkfirst=True
    )


def _client_for_company(company_id: int, client_id: int) -> Client:
    client = db.session.get(Client, client_id)
    if client is None or client.company_id != company_id:
        raise ConventionRefusal(
            "patient_autre_entreprise",
            "Ce patient n'appartient pas à cette entreprise.",
        )
    return client


def _row(company_id: int, client_id: int) -> PatientCompanyConventionDemo | None:
    """Version active seulement. Les versions précédentes restent en base."""
    return (
        db.session.query(PatientCompanyConventionDemo)
        .filter_by(company_id=company_id, client_id=client_id, active=True)
        .one_or_none()
    )


def present(
    company_id: int, client_id: int, *, company_name: str, debtor_label: str
) -> dict[str, object]:
    """Lecture. L'absence de ligne n'est pas un mandat."""
    _require_open()
    _client_for_company(company_id, client_id)
    row = _row(company_id, client_id)
    if row is None:
        return _absent(company_id, client_id, company_name, debtor_label)
    return _view(row, company_name)


def prepare(
    company_id: int, client_id: int, *, company_name: str, debtor_label: str
) -> dict[str, object]:
    """Ouvre un brouillon de démonstration. Ne valide rien."""
    _require_open()
    client = _client_for_company(company_id, client_id)
    label = (debtor_label or "").strip() or f"Patient {client.id}"
    row = _row(company_id, client_id)
    created = row is None
    if row is None:
        row = PatientCompanyConventionDemo(
            client_id=client_id,
            company_id=company_id,
            debtor_label=label,
            tariff_version=TARIFF_VERSION,
            service_scope=SERVICE_SCOPE,
            duration_label=DURATION_LABEL,
            status="demo_draft",
            document_version=DOCUMENT_VERSION,
            version_number=1,
            active=True,
            idempotency_key=f"ouverture-{client_id}-{company_id}",
            legally_effective=False,
            specimen_trace=None,
        )
        db.session.add(row)
    elif row.status != "demo_draft":
        # Une version signée ou simulée n'est pas réécrite par une nouvelle préparation.
        db.session.commit()
        return _view(row, company_name)
    elif row.tariff_version != TARIFF_VERSION:
        row.status = "demo_draft"
        row.specimen_trace = None
        row.tariff_version = TARIFF_VERSION
        row.legally_effective = False
    row.debtor_label = label
    row.legally_effective = False
    if created:
        from application.settlement.mandate_registry import remember

        remember(row, "prepared", "Mandat préparé pour cette relation patient–entreprise.")
    db.session.commit()
    return _view(row, company_name)


def record_specimen(
    company_id: int,
    client_id: int,
    trace: object,
    *,
    company_name: str,
    representative_name: str | None = None,
) -> dict[str, object]:
    """Enregistre un tracé de démonstration. Jamais une signature réelle."""
    _require_open()
    _client_for_company(company_id, client_id)
    row = _row(company_id, client_id)
    if row is None or row.status not in {"demo_draft", "demo_specimen_signed"}:
        # Un mandat valide de simulation ne se re-signe pas depuis le téléphone.
        raise ConventionRefusal(
            "specimen_sans_brouillon",
            "Préparez d'abord le document de démonstration.",
        )
    points = _trace(trace)
    kept = row.specimen_trace if isinstance(row.specimen_trace, dict) else {}
    row.specimen_trace = {
        "points": points,
        "kind": "demonstration",
        "mandate": dict(kept.get("mandate") or {}),
    }
    row.representative_name = (representative_name or "").strip() or None
    row.status = "demo_specimen_signed"
    row.legally_effective = False
    db.session.commit()
    return _view(row, company_name)


def specimen_pdf(company_id: int, client_id: int, *, company_name: str) -> bytes:
    _require_open()
    _client_for_company(company_id, client_id)
    row = _row(company_id, client_id)
    if row is None:
        raise ConventionRefusal("specimen_absent", "Aucun document de démonstration.")
    stored = read_archived_pdf(row)
    if stored is not None:
        return stored
    return _pdf(row, company_name)


def archive_reference(row: PatientCompanyConventionDemo) -> str:
    """Chemin stable d'une version. Une version suivante n'écrase pas ce fichier."""
    return f"mandates/{int(row.company_id)}/{int(row.id)}-v{int(row.version_number)}.pdf"


def tactile_reference(points: list[dict[str, float]]) -> str:
    raw = json.dumps(points, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def store_mandate_pdf(relative: str, payload: bytes) -> None:
    """Écrit le PDF de façon durable. Un échec ne laisse pas de fichier final."""
    if not payload.startswith(b"%PDF"):
        raise OSError("document illisible")
    target = _archive_root() / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_bytes(payload)
    temporary.replace(target)


def read_archived_pdf(row: PatientCompanyConventionDemo) -> bytes | None:
    trace = row.specimen_trace if isinstance(row.specimen_trace, dict) else {}
    mandate = trace.get("mandate") if isinstance(trace.get("mandate"), dict) else {}
    signature = mandate.get("signature") if isinstance(mandate.get("signature"), dict) else {}
    relative = signature.get("archive_reference")
    if not isinstance(relative, str) or not relative.startswith(f"mandates/{int(row.company_id)}/"):
        return None
    path = _archive_root() / relative
    if not path.is_file():
        return None
    return path.read_bytes()


def consult_archived_pdf(company_id: int, client_id: int, convention_id: int) -> bytes:
    """Relit le PDF archivé d'une version. Ne le régénère pas."""
    _require_open()
    _client_for_company(company_id, client_id)
    row = db.session.get(PatientCompanyConventionDemo, int(convention_id))
    if (
        row is None
        or int(row.company_id) != int(company_id)
        or int(row.client_id) != int(client_id)
    ):
        raise ConventionRefusal(
            "specimen_absent",
            "Ce document n'appartient pas à cette entreprise.",
        )
    stored = read_archived_pdf(row)
    if stored is None:
        raise ConventionRefusal("archive_absente", "Le document archivé est indisponible.")
    return stored


def archived_pdf_for(company_id: int, convention_id: int) -> bytes:
    """Relit l'archive d'une version précise, y compris une version inactive."""
    _require_open()
    row = db.session.get(PatientCompanyConventionDemo, int(convention_id))
    if row is None or int(row.company_id) != int(company_id):
        raise ConventionRefusal("specimen_absent", "Ce document n'appartient pas à cette entreprise.")
    stored = read_archived_pdf(row)
    if stored is None:
        raise ConventionRefusal("archive_absente", "Aucun document archivé pour cette version.")
    return stored


def _archive_root() -> Path:
    from flask import current_app, has_app_context

    configured = None
    if has_app_context():
        configured = current_app.config.get("UPLOAD_FOLDER")
    return Path(configured or "/app/uploads")


def company_name_for(company_id: int) -> str:
    company = db.session.get(Company, company_id)
    name = getattr(company, "name", None) if company is not None else None
    return str(name or "Entreprise fictive")


def _absent(
    company_id: int, client_id: int, company_name: str, debtor_label: str
) -> dict[str, object]:
    return {
        "client_id": client_id,
        "company_id": company_id,
        "company_name": company_name,
        "debtor_label": debtor_label,
        "status": "demo_absent",
        "status_label": "Absent",
        "tariff_version": None,
        "service_scope": SERVICE_SCOPE,
        "duration_label": DURATION_LABEL,
        "document_version": None,
        "legally_effective": False,
        "real_signature_collected": False,
        "payment_modes": ["cash", "invoice"],
        "tap_to_pay": "bientot_disponible",
        "disclaimer": DISCLAIMER,
        "covers_other_companies": False,
    }


_STATUS_LABELS = {
    "demo_draft": "Mandat à signer",
    "demo_specimen_signed": "Mandat signé",
    "demo_valid_simulated": "Mandat signé",
}


def _view(row: PatientCompanyConventionDemo, company_name: str) -> dict[str, object]:
    signed = row.status in {"demo_specimen_signed", "demo_valid_simulated"}
    return {
        "client_id": int(row.client_id),
        "company_id": int(row.company_id),
        "company_name": company_name,
        "debtor_label": row.debtor_label,
        "representative_name": row.representative_name,
        "status": row.status,
        "status_label": _STATUS_LABELS.get(row.status, "Mandat à signer"),
        "tariff_version": row.tariff_version,
        "service_scope": row.service_scope,
        "duration_label": row.duration_label,
        "document_version": row.document_version,
        "version_number": int(row.version_number),
        "active": bool(row.active),
        "legally_effective": False,
        "real_signature_collected": False,
        "payment_modes": ["cash", "invoice"],
        "tap_to_pay": "bientot_disponible",
        "disclaimer": DISCLAIMER,
        "covers_other_companies": False,
        "specimen_recorded": signed,
    }


def _trace(raw: object) -> list[dict[str, float]]:
    if not isinstance(raw, list) or not raw:
        raise ConventionRefusal("trace_vide", "Le tracé de démonstration est vide.")
    if len(raw) > 400:
        raise ConventionRefusal(
            "trace_trop_long", "Le tracé de démonstration est trop long."
        )
    points: list[dict[str, float]] = []
    for item in raw:
        if not isinstance(item, dict) or "x" not in item or "y" not in item:
            raise ConventionRefusal(
                "trace_illisible", "Le tracé de démonstration est illisible."
            )
        points.append({"x": float(item["x"]), "y": float(item["y"])})
    return points


def _pdf_line(value: str) -> str:
    """Helvetica n'encode pas le tiret long. Le texte écran garde le libellé exact."""
    return value.replace("\u2014", "-").encode("latin-1", "replace").decode("latin-1")


def _pdf(row: PatientCompanyConventionDemo, company_name: str) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    buffer = BytesIO()
    sheet = canvas.Canvas(buffer, pagesize=A4)
    sheet.setTitle(DISCLAIMER)
    y = 800
    sheet.setFont("Helvetica-Bold", 12)
    sheet.drawString(40, y, DISCLAIMER)
    y -= 28
    sheet.setFont("Helvetica", 11)
    signature: dict[str, Any] = {}
    mandate: dict[str, Any] = {}
    if isinstance(row.specimen_trace, dict):
        raw_mandate = row.specimen_trace.get("mandate")
        if isinstance(raw_mandate, dict):
            mandate = raw_mandate
            raw_signature = raw_mandate.get("signature")
            if isinstance(raw_signature, dict):
                signature = raw_signature
    covered_services = _covered(signature, "covered_services") or _covered(mandate, "covered_services")
    covered_tariffs = _covered(signature, "covered_tariff_versions") or _covered(
        mandate, "covered_tariff_versions"
    )
    lines = [
        "Convention de transport et autorisation de facturation différée",
        "Document provisoire. Il n'a aucune valeur contractuelle.",
        f"Créancier fictif : {company_name}",
        f"Débiteur fictif : {row.debtor_label}",
        f"Représentant : {row.representative_name or 'aucun'}",
        f"Version tarifaire de démonstration : {row.tariff_version}",
        row.service_scope,
        f"Durée : {row.duration_label}",
        f"Version du mandat : {int(row.version_number)}",
        f"Version du document : {row.document_version}",
        f"Prestations couvertes : {', '.join(covered_services) or row.service_scope}",
        f"Tarifs couverts : {', '.join(covered_tariffs) or row.tariff_version}",
        f"Statut de démonstration : {row.status}",
        "Statut V1 locale : non juridiquement opposable.",
        "Modalités affichées : espèces simulées, facture patient simulée.",
        "Tap to Pay : bientôt disponible, non activé.",
        "Zone de signature : spécimen uniquement."
        if isinstance(row.specimen_trace, dict) and row.specimen_trace.get("points")
        else "Zone de signature : vide.",
        "Signature : démonstration enregistrée, sans valeur juridique.",
        "real_signature_collected : false",
        "legally_effective : false",
        "Facture : non émise.",
        "Suppléments configurés : aucun.",
        f"Signataire de démonstration : {signature.get('signer') or 'non renseigné'}",
        f"Qualité déclarée : {signature.get('representative_quality') or 'patient'}",
        f"Date de signature : {signature.get('signed_at') or 'non enregistrée'}",
        f"Référence du tracé : {signature.get('tactile_reference') or 'absente'}",
        f"Référence du document : {signature.get('archive_reference') or 'non archivé'}",
        (
            "Vérification administrative : à effectuer."
            if signature.get("admin_review_required")
            else "Vérification administrative : non requise."
        ),
        "Ce fichier n'autorise aucune facture réelle.",
    ]
    for line in lines:
        sheet.drawString(40, y, _pdf_line(line)[:110])
        y -= 18
    _draw_trace(sheet, row, y)
    sheet.showPage()
    sheet.save()
    return buffer.getvalue()


def _covered(source: dict[str, Any], key: str) -> list[str]:
    raw = source.get(key)
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw if str(item).strip()]


def _draw_trace(sheet: Any, row: PatientCompanyConventionDemo, y: int) -> None:
    trace = row.specimen_trace if isinstance(row.specimen_trace, dict) else {}
    points = trace.get("points")
    if not isinstance(points, list) or len(points) < 1:
        return
    usable = [
        (float(item["x"]), float(item["y"]))
        for item in points
        if isinstance(item, dict) and "x" in item and "y" in item
    ]
    if not usable:
        return
    xs = [item[0] for item in usable]
    ys = [item[1] for item in usable]
    span_x = max(xs) - min(xs) or 1.0
    span_y = max(ys) - min(ys) or 1.0
    sheet.setStrokeColorRGB(0.1, 0.1, 0.1)
    previous = None
    for raw_x, raw_y in usable[:80]:
        x = 40 + ((raw_x - min(xs)) / span_x) * 180
        point_y = y - 70 + ((raw_y - min(ys)) / span_y) * 50
        if previous is not None:
            sheet.line(previous[0], previous[1], x, point_y)
        previous = (x, point_y)
