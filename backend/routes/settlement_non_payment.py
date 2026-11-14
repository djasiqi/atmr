"""API signalements de non-règlement + carte terminal externe.

Sépare transport terminé et règlement non effectué. Aucun Tap to Pay.
Aucune facturation différée réelle via convention de démonstration.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation

from flask import request
from flask_jwt_extended import jwt_required
from flask_restx import Resource

from application.settlement import settlement_incident_store as store
from domain.end_of_trip_settlement.closure_options import (
    ClosureContext,
    build_closure_menu,
    demo_convention_blocks_real_deferred,
)
from domain.end_of_trip_settlement.mandate_applicability import assess_mandate
from ext import db, role_required
from models.booking import Booking
from models.enums import UserRole
from routes.companies import companies_ns
from routes.driver import driver_ns, get_driver_from_token


def _money(raw: object) -> Decimal:
    try:
        return Decimal(str(raw)).quantize(Decimal("0.01"))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise store.SettlementStoreRefusal("montant_invalide", 400) from exc


def _driver_booking(booking_id: int):
    driver, error, status = get_driver_from_token()
    if error:
        return None, None, (error, status)
    booking = db.session.get(Booking, booking_id)
    if booking is None or int(booking.company_id) != int(driver.company_id):
        return None, None, ({"error": "Mission indisponible"}, 404)
    return driver, booking, None


def _company():
    from routes.companies import _get_current_company_via_use_case

    company, error, status = _get_current_company_via_use_case()
    if error or company is None:
        return None, None, (error or {"error": "Entreprise non trouvée"}, status or 404)
    return company, int(company.user_id), None


def _failure(exc: store.SettlementStoreRefusal):
    db.session.rollback()
    return {"error": exc.code, "error_code": exc.code}, exc.status


def _mandate_kind_for(booking: Booking) -> tuple[str, bool]:
    """Retourne (kind, legally_effective).

    Priorité à la convention réelle (gate différé fermée sans activation).
    La démo reste ``legally_effective=False`` (contrainte DB).
    """
    from datetime import date

    from application.settlement import convention_real as convention_real_service
    from models.convention_demo import PatientCompanyConventionDemo

    client_id = getattr(booking, "client_id", None)
    if client_id is None:
        return "absent", False

    real_gate = convention_real_service.gate_for_client(
        int(booking.company_id), int(client_id)
    )
    if real_gate.get("convention_id") is not None:
        return str(real_gate["mandate_kind"]), bool(real_gate["legally_effective"])

    row = (
        db.session.query(PatientCompanyConventionDemo)
        .filter_by(
            client_id=int(client_id),
            company_id=int(booking.company_id),
            active=True,
        )
        .one_or_none()
    )
    if row is None:
        kind = assess_mandate(
            mandate_company_id=None,
            booking_company_id=int(booking.company_id),
            status="demo_absent",
            covered_tariff_versions=(),
            booking_tariff_version=None,
            valid_from=None,
            valid_until=None,
            on_date=date.today(),
            billed_to_type=str(getattr(booking, "billed_to_type", None) or "patient"),
            already_paid=False,
        ).kind
        return kind, False
    kind = assess_mandate(
        mandate_company_id=int(row.company_id),
        booking_company_id=int(booking.company_id),
        status=str(row.status),
        covered_tariff_versions=(str(row.tariff_version),),
        booking_tariff_version=None,
        valid_from=None,
        valid_until=None,
        on_date=date.today(),
        billed_to_type=str(getattr(booking, "billed_to_type", None) or "patient"),
        already_paid=False,
    ).kind
    # Contrainte DB : legally_effective toujours false sur convention_demo.
    return kind, False


@driver_ns.route("/me/bookings/<int:booking_id>/closure-options")
class DriverClosureOptions(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    def get(self, booking_id: int):
        driver, booking, error = _driver_booking(booking_id)
        if error:
            return error
        kind, legal = _mandate_kind_for(booking)
        amount = (
            getattr(booking, "amount", None) or getattr(booking, "price", None) or 0
        )
        menu = build_closure_menu(
            ClosureContext(
                booking_id=int(booking.id),
                company_id=int(booking.company_id),
                driver_company_id=int(driver.company_id),
                billed_to_type=str(
                    getattr(booking, "billed_to_type", None) or "patient"
                ),
                mandate_kind=kind,
                amount_due_chf=_money(amount),
                payment_received=False,
                convention_legally_effective=legal,
                tap_to_pay_operational=False,
            )
        )
        return {
            "applicable": menu.applicable,
            "reasons": list(menu.reasons),
            "mission_close_without_payment_allowed_via_incident": (
                menu.mission_close_without_payment_allowed_via_incident
            ),
            "transport_complete_without_settlement_allowed": (
                menu.transport_complete_without_settlement_allowed
            ),
            "demo_blocks_real_deferred": demo_convention_blocks_real_deferred(
                mandate_kind=kind,
                convention_legally_effective=legal,
            ),
            "options": [
                {
                    "code": option.code,
                    "label": option.label,
                    "available": option.available,
                    "records_payment": option.records_payment,
                    "authorizes_deferred_billing": option.authorizes_deferred_billing,
                    "opens_incident": option.opens_incident,
                    "reason": option.reason,
                    "channel": option.channel,
                }
                for option in menu.options
            ],
        }, 200


@driver_ns.route("/me/bookings/<int:booking_id>/settlement-incidents")
class DriverSettlementIncident(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    def post(self, booking_id: int):
        driver, booking, error = _driver_booking(booking_id)
        if error:
            return error
        body = request.get_json(silent=True) or {}
        amount = body.get("amount_due_chf")
        if amount is None:
            amount = (
                getattr(booking, "amount", None) or getattr(booking, "price", None) or 0
            )
        try:
            view = store.open_incident(
                booking_id=int(booking.id),
                company_id=int(booking.company_id),
                driver_id=int(driver.id),
                actor_company_id=int(driver.company_id),
                client_id=getattr(booking, "client_id", None),
                debtor_id=getattr(booking, "client_id", None),
                debtor_label=str(
                    getattr(booking, "customer_name", None)
                    or getattr(booking, "passenger_name", None)
                    or "Client"
                ),
                amount_due_chf=_money(amount),
                transport_date=str(
                    getattr(booking, "scheduled_time", None)
                    or getattr(booking, "mission_date", None)
                    or ""
                )[:32],
                motif=str(body.get("motif") or ""),
                observation=str(body.get("observation") or ""),
                idempotency_key=str(body.get("idempotency_key") or ""),
            )
        except store.SettlementStoreRefusal as exc:
            return _failure(exc)
        return view, 200


@driver_ns.route("/me/bookings/<int:booking_id>/external-card-settlement")
class DriverExternalCardSettlement(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    def post(self, booking_id: int):
        driver, booking, error = _driver_booking(booking_id)
        if error:
            return error
        body = request.get_json(silent=True) or {}
        # Refus explicite du Tap to Pay simulé sur cet endpoint.
        if body.get("tap_to_pay") is True or body.get("channel") == "tap_to_pay":
            return {
                "error": "tap_to_pay_interdit_ici",
                "error_code": "tap_to_pay_interdit_ici",
            }, 400
        amount = body.get("amount_chf")
        if amount is None:
            amount = (
                getattr(booking, "amount", None) or getattr(booking, "price", None) or 0
            )
        try:
            view = store.record_external_card(
                booking_id=int(booking.id),
                company_id=int(booking.company_id),
                driver_id=int(driver.id),
                actor_company_id=int(driver.company_id),
                amount_chf=_money(amount),
                reference=str(body.get("reference") or ""),
                receipt_note=str(body.get("receipt_note") or ""),
                idempotency_key=str(body.get("idempotency_key") or ""),
            )
        except store.SettlementStoreRefusal as exc:
            return _failure(exc)
        return view, 200


@companies_ns.route("/me/settlement-incidents")
class CompanySettlementIncidents(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        company, _actor, error = _company()
        if error:
            return error
        return store.list_company(int(company.id)), 200


@companies_ns.route("/me/settlement-incidents/<int:incident_id>")
class CompanySettlementIncidentDetail(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self, incident_id: int):
        company, _actor, error = _company()
        if error:
            return error
        view = store.get_company(int(company.id), int(incident_id))
        if view is None:
            return {"error": "introuvable", "error_code": "introuvable"}, 404
        return view, 200


@companies_ns.route("/me/settlement-incidents/<int:incident_id>/status")
class CompanySettlementIncidentStatus(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def post(self, incident_id: int):
        company, actor, error = _company()
        if error:
            return error
        body = request.get_json(silent=True) or {}
        try:
            view = store.admin_set_status(
                company_id=int(company.id),
                incident_id=int(incident_id),
                new_status=str(body.get("status") or ""),
                actor_id=int(actor),
                justification=str(body.get("justification") or ""),
                idempotency_key=str(body.get("idempotency_key") or ""),
            )
        except store.SettlementStoreRefusal as exc:
            return _failure(exc)
        return view, 200


@companies_ns.route("/me/settlement-incidents/<int:incident_id>/later-payment")
class CompanySettlementIncidentLaterPayment(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def post(self, incident_id: int):
        company, actor, error = _company()
        if error:
            return error
        body = request.get_json(silent=True) or {}
        try:
            view = store.admin_note_later_payment(
                company_id=int(company.id),
                incident_id=int(incident_id),
                actor_id=int(actor),
                note=str(body.get("note") or ""),
                idempotency_key=str(body.get("idempotency_key") or ""),
            )
        except store.SettlementStoreRefusal as exc:
            return _failure(exc)
        return view, 200
