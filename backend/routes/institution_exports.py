# routes/institution_exports.py
"""Endpoints d'export des transports institution (PDF patient, PDF/CSV journalier, PDF mission).

Accès réservé aux rôles autorisés (`EXPORT_ALLOWED_ROLES`) :
- `institution_admin`
- `institution_billing`
- `institution_reception`

Exports disponibles :
- GET /institutions/exports/patients/<patient_id>/pdf  — historique patient (période)
- GET /institutions/exports/daily/pdf?date=YYYY-MM-DD  — journalier (synthèse + détail)
- GET /institutions/exports/daily/rapports.zip?date=YYYY-MM-DD  — journalier (1 rapport PDF / transport)
- GET /institutions/exports/daily/csv?date=YYYY-MM-DD  — journalier (tableur)
- GET /institutions/exports/requests/<request_id>/pdf?variant=  — bon / rapport mission
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime
from io import BytesIO

import sentry_sdk
from flask import request, send_file
from flask_jwt_extended import verify_jwt_in_request
from flask_restx import Namespace, Resource

from models import Institution, InstitutionPatient
from models.enums import InstitutionRole
from routes.api_error_models import (
    create_not_found_error_model,
    create_permission_error_model,
)
from routes.institution_requests import get_institution_read_context
from models import TransportRequest
from services.institutions.export_transports import (
    build_daily_csv,
    build_daily_mission_reports_zip,
    build_daily_pdf,
    build_patient_pdf,
    collect_daily_transports,
    collect_patient_transports,
    day_bounds,
)
from services.institutions.mission_report_context import (
    build_mission_pdf_filename,
    collect_mission_report_context,
)
from services.institutions.mission_report_pdf import (
    build_mission_audit_report_pdf,
    build_operational_voucher_pdf,
)

logger = logging.getLogger(__name__)

institution_exports_ns = Namespace(
    "institution_exports",
    description="Export des transports institution (PDF / CSV)",
)

not_found_error_model = create_not_found_error_model(institution_exports_ns)
permission_error_model = create_permission_error_model(institution_exports_ns)

# Rôles autorisés à exporter (admin + facturation + réception)
EXPORT_ALLOWED_ROLES = frozenset(
    {
        InstitutionRole.ADMIN.value,
        InstitutionRole.BILLING.value,
        InstitutionRole.RECEPTION.value,
    }
)


class _ExportForbidden(Exception):
    """Levée quand le rôle courant n'a pas le droit d'export."""


def _require_export_context() -> int:
    """Vérifie l'auth JWT + le rôle d'export et retourne l'institution_id.

    Raises:
        _ExportForbidden: si le rôle n'autorise pas l'export.
    """
    institution_id, _user_id, role = get_institution_read_context()
    if role not in EXPORT_ALLOWED_ROLES:
        raise _ExportForbidden()
    return institution_id


def _parse_date_param(name: str = "date") -> date | None:
    raw = request.args.get(name)
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw).date()
    except ValueError:
        return None


def _parse_datetime_param(name: str) -> datetime | None:
    raw = request.args.get(name)
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _pdf_response(pdf_bytes: bytes, filename: str):
    return send_file(
        BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=filename,
    )


def _csv_response(csv_bytes: bytes, filename: str):
    return send_file(
        BytesIO(csv_bytes),
        mimetype="text/csv; charset=utf-8",
        as_attachment=True,
        download_name=filename,
    )


@institution_exports_ns.route("/patients/<int:patient_id>/pdf")
class PatientTransportExportPdf(Resource):
    """Export PDF de l'historique des transports d'un patient."""

    @institution_exports_ns.doc(
        description="Exporte l'historique des transports d'un patient en PDF.",
        security="Bearer",
        params={
            "from": "Date de début (ISO8601, optionnel)",
            "to": "Date de fin (ISO8601, optionnel)",
        },
    )
    @institution_exports_ns.response(200, "PDF généré")
    @institution_exports_ns.response(403, "Accès refusé", permission_error_model)
    @institution_exports_ns.response(404, "Patient non trouvé", not_found_error_model)
    def get(self, patient_id: int):
        try:
            verify_jwt_in_request()
            institution_id = _require_export_context()

            patient = InstitutionPatient.query.filter_by(
                id=patient_id, institution_id=institution_id
            ).first()
            if not patient:
                return {"error": "Patient non trouvé"}, 404

            institution = Institution.query.get(institution_id)
            date_from = _parse_datetime_param("from")
            date_to = _parse_datetime_param("to")

            requests = collect_patient_transports(
                institution_id,
                patient_id,
                date_from=date_from,
                date_to=date_to,
            )

            if date_from or date_to:
                start_label = (
                    date_from.strftime("%d.%m.%Y") if date_from else "origine"
                )
                end_label = date_to.strftime("%d.%m.%Y") if date_to else "à ce jour"
                period_label = f"{start_label} → {end_label}"
            else:
                period_label = "Tout l'historique"

            pdf_bytes = build_patient_pdf(
                institution, patient, requests, period_label
            )
            filename = f"transports_patient_{patient_id}.pdf"
            response = _pdf_response(pdf_bytes, filename)
            response.headers["X-Rows-Count"] = str(len(requests))
            return response
        except _ExportForbidden:
            return {"error": "Accès refusé : export non autorisé pour ce rôle"}, 403
        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.error("[Export] PDF patient %s: %s", patient_id, e)
            return {"error": "Erreur serveur"}, 500


@institution_exports_ns.route("/daily/pdf")
class DailyTransportExportPdf(Resource):
    """Export PDF journalier des transports."""

    @institution_exports_ns.doc(
        description="Exporte les transports d'une journée en PDF (synthèse + détail).",
        security="Bearer",
        params={"date": "Date du jour à exporter (YYYY-MM-DD, défaut: aujourd'hui)"},
    )
    @institution_exports_ns.response(200, "PDF généré")
    @institution_exports_ns.response(403, "Accès refusé", permission_error_model)
    def get(self):
        try:
            verify_jwt_in_request()
            institution_id = _require_export_context()

            target_day = _parse_date_param("date") or datetime.now(UTC).date()
            day_start, day_end = day_bounds(target_day)

            institution = Institution.query.get(institution_id)
            requests = collect_daily_transports(institution_id, day_start, day_end)

            day_label = target_day.strftime("%d.%m.%Y")
            pdf_bytes = build_daily_pdf(institution, day_label, requests)
            filename = f"transports_{target_day.isoformat()}.pdf"
            response = _pdf_response(pdf_bytes, filename)
            response.headers["X-Rows-Count"] = str(len(requests))
            return response
        except _ExportForbidden:
            return {"error": "Accès refusé : export non autorisé pour ce rôle"}, 403
        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.error("[Export] PDF journalier: %s", e)
            return {"error": "Erreur serveur"}, 500


def _zip_response(zip_bytes: bytes, filename: str):
    return send_file(
        BytesIO(zip_bytes),
        mimetype="application/zip",
        as_attachment=True,
        download_name=filename,
    )


@institution_exports_ns.route("/daily/rapports.zip")
class DailyMissionReportsExportZip(Resource):
    """Export ZIP journalier : un rapport de mission par demande du jour."""

    @institution_exports_ns.doc(
        description=(
            "Exporte les rapports de mission (PDF audit) de la journée "
            "dans une archive ZIP (1 fichier par transport)."
        ),
        security="Bearer",
        params={"date": "Date du jour à exporter (YYYY-MM-DD, défaut: aujourd'hui)"},
    )
    @institution_exports_ns.response(200, "ZIP généré")
    @institution_exports_ns.response(403, "Accès refusé", permission_error_model)
    @institution_exports_ns.response(404, "Aucun transport", not_found_error_model)
    def get(self):
        try:
            verify_jwt_in_request()
            institution_id = _require_export_context()

            target_day = _parse_date_param("date") or datetime.now(UTC).date()
            day_start, day_end = day_bounds(target_day)

            institution = Institution.query.get(institution_id)
            requests = collect_daily_transports(institution_id, day_start, day_end)
            if not requests:
                return {"error": "Aucun transport pour cette date"}, 404

            zip_bytes = build_daily_mission_reports_zip(institution, requests)
            filename = f"rapports_mission_{target_day.isoformat()}.zip"
            response = _zip_response(zip_bytes, filename)
            response.headers["X-Rows-Count"] = str(len(requests))
            return response
        except _ExportForbidden:
            return {"error": "Accès refusé : export non autorisé pour ce rôle"}, 403
        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.error("[Export] ZIP rapports journaliers: %s", e)
            return {"error": "Erreur serveur"}, 500


@institution_exports_ns.route("/daily/csv")
class DailyTransportExportCsv(Resource):
    """Export CSV journalier des transports."""

    @institution_exports_ns.doc(
        description="Exporte les transports d'une journée en CSV.",
        security="Bearer",
        params={"date": "Date du jour à exporter (YYYY-MM-DD, défaut: aujourd'hui)"},
    )
    @institution_exports_ns.response(200, "CSV généré")
    @institution_exports_ns.response(403, "Accès refusé", permission_error_model)
    def get(self):
        try:
            verify_jwt_in_request()
            institution_id = _require_export_context()

            target_day = _parse_date_param("date") or datetime.now(UTC).date()
            day_start, day_end = day_bounds(target_day)

            institution = Institution.query.get(institution_id)
            requests = collect_daily_transports(institution_id, day_start, day_end)

            day_label = target_day.strftime("%d.%m.%Y")
            csv_bytes = build_daily_csv(institution, day_label, requests)
            filename = f"transports_{target_day.isoformat()}.csv"
            response = _csv_response(csv_bytes, filename)
            response.headers["X-Rows-Count"] = str(len(requests))
            return response
        except _ExportForbidden:
            return {"error": "Accès refusé : export non autorisé pour ce rôle"}, 403
        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.error("[Export] CSV journalier: %s", e)
            return {"error": "Erreur serveur"}, 500


@institution_exports_ns.route("/requests/<int:request_id>/pdf")
class RequestMissionExportPdf(Resource):
    """Export PDF unitaire : bon de transport ou rapport de mission."""

    @institution_exports_ns.doc(
        description=(
            "Exporte une demande en PDF (bon opérationnel ou rapport de mission audit)."
        ),
        security="Bearer",
        params={
            "variant": "operational (bon de transport) ou audit (rapport de mission, défaut)",
        },
    )
    @institution_exports_ns.response(200, "PDF généré")
    @institution_exports_ns.response(403, "Accès refusé", permission_error_model)
    @institution_exports_ns.response(404, "Demande non trouvée", not_found_error_model)
    def get(self, request_id: int):
        try:
            verify_jwt_in_request()
            institution_id = _require_export_context()

            tr = TransportRequest.query.filter_by(
                id=request_id, institution_id=institution_id
            ).first()
            if not tr:
                return {"error": "Demande non trouvée"}, 404

            institution = Institution.query.get(institution_id)
            if not institution:
                return {"error": "Institution non trouvée"}, 404

            variant = (request.args.get("variant") or "audit").strip().lower()
            if variant not in {"operational", "audit"}:
                return {"error": "variant invalide (operational ou audit)"}, 400

            ctx = collect_mission_report_context(
                tr, institution, variant=variant, show_amount=True
            )
            if variant == "operational":
                pdf_bytes = build_operational_voucher_pdf(ctx, layout="operational")
            else:
                pdf_bytes = build_mission_audit_report_pdf(ctx)

            filename = build_mission_pdf_filename(tr, variant=variant)
            return _pdf_response(pdf_bytes, filename)
        except _ExportForbidden:
            return {"error": "Accès refusé : export non autorisé pour ce rôle"}, 403
        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.error("[Export] PDF demande %s: %s", request_id, e)
            return {"error": "Erreur serveur"}, 500
