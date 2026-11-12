"""Ancien suivi fictif côté entreprise. Aucun livre de caisse, aucune créance.

L'écran portail n'est plus monté. Le contrat GET et POST reste pour les
tests des étapes 25 et 27. Aucune route ne supprime un dossier.
"""

from __future__ import annotations

from flask import request
from flask_jwt_extended import get_jwt_identity, jwt_required
from flask_restx import Resource

from application.settlement.simulation_cases import (
    LedgerClosed,
    list_company_cases,
    read_company_case,
    transition_case,
)
from domain.end_of_trip_settlement.simulation_barrier import ledger_open
from domain.end_of_trip_settlement.simulation_machine import SimulationRefusal
from ext import db, role_required
from models import User, UserRole
from models.booking import Booking
from routes.companies import companies_ns


def _closed():
    return {"error": "Rubrique indisponible", "error_code": "LEDGER_CLOSED"}, 404


def _company():
    from routes.companies import _get_current_company_via_use_case

    company, error, status = _get_current_company_via_use_case()
    if error or company is None:
        return None, (error or {"error": "Entreprise non trouvée"}, status or 404)
    return company, None


@companies_ns.route("/me/settlement-simulations")
class CompanySettlementSimulations(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        if not ledger_open():
            return _closed()
        company, error = _company()
        if error:
            return error
        try:
            rows = list_company_cases(int(company.id))
        except LedgerClosed:
            return _closed()
        return {
            "banner": "SIMULATION — AUCUN PAIEMENT RÉEL",
            "cash_book": False,
            "cases": rows,
        }, 200


@companies_ns.route("/me/settlement-simulations/<int:case_id>")
class CompanySettlementSimulationDetail(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self, case_id: int):
        if not ledger_open():
            return _closed()
        company, error = _company()
        if error:
            return error
        try:
            view = read_company_case(int(company.id), case_id)
        except LedgerClosed:
            return _closed()
        if view is None:
            return {"error": "Dossier absent"}, 404
        return view, 200


@companies_ns.route("/me/settlement-simulations/<int:case_id>/transitions")
class CompanySettlementSimulationTransition(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def post(self, case_id: int):
        if not ledger_open():
            return _closed()
        company, error = _company()
        if error:
            return error
        from models.settlement_simulation import SettlementSimulationCase

        case = db.session.get(SettlementSimulationCase, case_id)
        if case is None or int(case.company_id) != int(company.id):
            return {"error": "Dossier absent"}, 404
        booking = db.session.get(Booking, int(case.booking_id))
        if booking is None:
            return {"error": "Dossier absent"}, 404
        user = User.query.filter_by(public_id=get_jwt_identity()).first()
        body = request.get_json(silent=True) or {}
        try:
            view = transition_case(
                booking,
                driver_id=int(booking.driver_id or 0),
                company_id=int(company.id),
                actor_role="company",
                actor_id=int(user.id) if user is not None else None,
                action="company_cash",
                idempotency_key=str(body.get("idempotency_key") or ""),
                payload=body,
            )
        except LedgerClosed:
            db.session.rollback()
            return _closed()
        except SimulationRefusal as exc:
            db.session.rollback()
            return {"error": exc.code, "error_code": exc.code}, exc.status
        return view, 200
