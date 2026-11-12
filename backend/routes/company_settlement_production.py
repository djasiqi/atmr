"""Registre de caisse et dossiers patient. Fermés hors préparation serveur."""

from __future__ import annotations

from flask import request
from flask_jwt_extended import jwt_required
from flask_restx import Resource

from application.settlement import cash_ledger, patient_billing_prep
from domain.end_of_trip_settlement.production_prep_barrier import (
    PrepClosed,
    PrepRefusal,
    prep_open,
)
from ext import role_required
from models.enums import UserRole
from routes.companies import companies_ns


def _closed():
    return {"error": "Rubrique indisponible", "error_code": "PREP_CLOSED"}, 404


def _company():
    from routes.companies import _get_current_company_via_use_case

    company, error, status = _get_current_company_via_use_case()
    if error or company is None:
        return None, None, (error or {"error": "Entreprise non trouvée"}, status or 404)
    return company, int(company.user_id), None


def _failure(exc: PrepRefusal):
    from ext import db

    db.session.rollback()
    return {"error": exc.code, "error_code": exc.code}, exc.status


@companies_ns.route("/me/cash-register")
class CompanyCashRegister(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        if not prep_open():
            return _closed()
        company, _actor, error = _company()
        if error:
            return error
        try:
            return cash_ledger.list_company(int(company.id)), 200
        except PrepClosed:
            return _closed()


@companies_ns.route("/me/cash-register/<int:collection_id>/confirm-handover")
class CompanyCashHandover(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def post(self, collection_id: int):
        if not prep_open():
            return _closed()
        company, actor, error = _company()
        if error:
            return error
        try:
            payload = request.get_json(silent=True) or {}
            return cash_ledger.confirm_handover(
                collection_id,
                company_id=int(company.id),
                actor_id=int(actor),
                idempotency_key=str(payload.get("idempotency_key") or ""),
            ), 200
        except PrepRefusal as exc:
            return _failure(exc)


@companies_ns.route("/me/cash-register/<int:collection_id>/reconcile")
class CompanyCashReconcile(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def post(self, collection_id: int):
        if not prep_open():
            return _closed()
        company, actor, error = _company()
        if error:
            return error
        try:
            return cash_ledger.reconcile(
                collection_id,
                request.get_json(silent=True) or {},
                company_id=int(company.id),
                actor_id=int(actor),
            ), 200
        except PrepRefusal as exc:
            return _failure(exc)


@companies_ns.route("/me/clients/<int:client_id>/cash-receipts")
class CompanyClientCashReceipts(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self, client_id: int):
        if not prep_open():
            return _closed()
        company, _actor, error = _company()
        if error:
            return error
        try:
            return cash_ledger.list_client_receipts(
                company_id=int(company.id), client_id=int(client_id)
            ), 200
        except PrepRefusal as exc:
            return _failure(exc)


@companies_ns.route("/me/cash-register/<int:collection_id>/receipt")
class CompanyCashReceipt(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self, collection_id: int):
        if not prep_open():
            return _closed()
        company, _actor, error = _company()
        if error:
            return error
        try:
            return cash_ledger.reprint_receipt(
                collection_id, company_id=int(company.id)
            ), 200
        except PrepRefusal as exc:
            return _failure(exc)


@companies_ns.route("/me/patient-billing-dossiers/<int:dossier_id>/review")
class CompanyPatientBillingReview(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def post(self, dossier_id: int):
        if not prep_open():
            return _closed()
        company, actor, error = _company()
        if error:
            return error
        try:
            return patient_billing_prep.review_representative(
                dossier_id,
                request.get_json(silent=True) or {},
                company_id=int(company.id),
                actor_id=int(actor),
            ), 200
        except PrepRefusal as exc:
            return _failure(exc)


@companies_ns.route("/me/patient-billing-followups")
class CompanyPatientFollowup(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def post(self):
        if not prep_open():
            return _closed()
        company, actor, error = _company()
        if error:
            return error
        body = request.get_json(silent=True) or {}
        try:
            booking_id = int(body.get("booking_id"))
        except (TypeError, ValueError):
            return {"error": "reservation_absente", "error_code": "reservation_absente"}, 404
        try:
            return patient_billing_prep.ensure_patient_followup(
                booking_id,
                company_id=int(company.id),
                actor_id=actor,
            ), 200
        except PrepClosed:
            return _closed()
        except PrepRefusal as exc:
            return _failure(exc)


@companies_ns.route("/me/patient-billing-dossiers")
class CompanyPatientBilling(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        if not prep_open():
            return _closed()
        company, _actor, error = _company()
        if error:
            return error
        try:
            return patient_billing_prep.list_company(int(company.id)), 200
        except PrepClosed:
            return _closed()
