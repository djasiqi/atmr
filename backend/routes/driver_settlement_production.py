"""Espèces et facture patient. 404 tant que la préparation serveur est fermée."""

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
from ext import db, role_required
from models.booking import Booking
from models.enums import UserRole
from routes.driver import driver_ns, get_driver_from_token


def _closed():
    return {"error": "Rubrique indisponible", "error_code": "PREP_CLOSED"}, 404


def _driver_booking(booking_id: int):
    if not prep_open():
        return None, None, _closed()
    driver, error, status = get_driver_from_token()
    if error:
        return None, None, (error, status)
    booking = db.session.get(Booking, booking_id)
    if booking is None or int(booking.company_id) != int(driver.company_id):
        return None, None, ({"error": "Mission indisponible"}, 404)
    return driver, booking, None


def _failure(exc: PrepRefusal):
    db.session.rollback()
    return {"error": exc.code, "error_code": exc.code}, exc.status


@driver_ns.route("/me/bookings/<int:booking_id>/cash-collection")
class DriverCashCollection(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    def get(self, booking_id: int):
        driver, booking, error = _driver_booking(booking_id)
        if error:
            return error
        try:
            view = cash_ledger.read_collection(
                booking,
                driver_id=int(driver.id),
                company_id=int(driver.company_id),
            )
        except PrepClosed:
            return _closed()
        except PrepRefusal as exc:
            return _failure(exc)
        if view is None:
            return {"error": "Dossier absent", "error_code": "dossier_absent"}, 404
        return view, 200

    @jwt_required()
    @role_required(UserRole.driver)
    def post(self, booking_id: int):
        driver, booking, error = _driver_booking(booking_id)
        if error:
            return error
        body = request.get_json(silent=True) or {}
        try:
            return cash_ledger.open_collection(
                booking,
                driver_id=int(driver.id),
                company_id=int(driver.company_id),
                actor_id=int(driver.user_id),
                exceptional=bool(body.get("exceptional")),
            ), 200
        except PrepClosed:
            db.session.rollback()
            return _closed()
        except PrepRefusal as exc:
            return _failure(exc)


@driver_ns.route("/me/bookings/<int:booking_id>/cash-receipt")
class DriverCashReceipt(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    def get(self, booking_id: int):
        driver, booking, error = _driver_booking(booking_id)
        if error:
            return error
        try:
            return cash_ledger.read_receipt(
                booking,
                driver_id=int(driver.id),
                company_id=int(driver.company_id),
            ), 200
        except PrepClosed:
            return _closed()
        except PrepRefusal as exc:
            return _failure(exc)


@driver_ns.route("/me/bookings/<int:booking_id>/cash-collection/transitions")
class DriverCashTransition(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    def post(self, booking_id: int):
        driver, booking, error = _driver_booking(booking_id)
        if error:
            return error
        body = request.get_json(silent=True) or {}
        action = str(body.get("action") or "declare_cash")
        try:
            if action == "mark_uncertain":
                view = cash_ledger.mark_uncertain(
                    booking,
                    body,
                    driver_id=int(driver.id),
                    company_id=int(driver.company_id),
                    actor_id=int(driver.user_id),
                )
            elif action == "refuse_payment":
                view = cash_ledger.refuse_payment(
                    booking,
                    body,
                    driver_id=int(driver.id),
                    company_id=int(driver.company_id),
                    actor_id=int(driver.user_id),
                )
            elif action == "prepare_receipt":
                view = cash_ledger.prepare_receipt(
                    booking,
                    body,
                    driver_id=int(driver.id),
                    company_id=int(driver.company_id),
                    actor_id=int(driver.user_id),
                )
            elif action == "dispatch_receipt":
                view = cash_ledger.dispatch_receipt(
                    booking,
                    body,
                    driver_id=int(driver.id),
                    company_id=int(driver.company_id),
                    actor_id=int(driver.user_id),
                )
            elif action == "compensate_cash":
                view = cash_ledger.compensate_cash(
                    booking,
                    body,
                    driver_id=int(driver.id),
                    company_id=int(driver.company_id),
                    actor_id=int(driver.user_id),
                )
            elif action == "declare_handover":
                view = cash_ledger.declare_handover(
                    booking,
                    driver_id=int(driver.id),
                    company_id=int(driver.company_id),
                    actor_id=int(driver.user_id),
                    idempotency_key=str(body.get("idempotency_key") or ""),
                    amount_chf=body.get("amount_chf"),
                )
            elif action == "reconcile":
                return {
                    "error": "rapprochement_reserve",
                    "error_code": "rapprochement_reserve",
                }, 403
            else:
                view = cash_ledger.declare_cash(
                    booking,
                    body,
                    driver_id=int(driver.id),
                    company_id=int(driver.company_id),
                    actor_id=int(driver.user_id),
                )
        except PrepClosed:
            db.session.rollback()
            return _closed()
        except PrepRefusal as exc:
            return _failure(exc)
        return view, 200


@driver_ns.route("/me/bookings/<int:booking_id>/patient-billing-dossier")
class DriverPatientBilling(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    def get(self, booking_id: int):
        driver, booking, error = _driver_booking(booking_id)
        if error:
            return error
        try:
            view = patient_billing_prep.read_dossier(
                booking,
                driver_id=int(driver.id),
                company_id=int(driver.company_id),
            )
        except PrepClosed:
            return _closed()
        except PrepRefusal as exc:
            return _failure(exc)
        if view is None:
            return {"error": "Dossier absent", "error_code": "dossier_absent"}, 404
        return view, 200

    @jwt_required()
    @role_required(UserRole.driver)
    def post(self, booking_id: int):
        driver, booking, error = _driver_booking(booking_id)
        if error:
            return error
        try:
            return patient_billing_prep.open_dossier(
                booking,
                driver_id=int(driver.id),
                company_id=int(driver.company_id),
                actor_id=int(driver.user_id),
            ), 200
        except PrepClosed:
            db.session.rollback()
            return _closed()
        except PrepRefusal as exc:
            return _failure(exc)


@driver_ns.route("/me/bookings/<int:booking_id>/patient-billing-dossier/transitions")
class DriverPatientBillingTransition(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    def post(self, booking_id: int):
        driver, booking, error = _driver_booking(booking_id)
        if error:
            return error
        body = request.get_json(silent=True) or {}
        try:
            if body.get("action") == "issue_invoice":
                patient_billing_prep.refuse_issue(
                    booking, company_id=int(driver.company_id)
                )
            view = patient_billing_prep.record_event(
                booking,
                body,
                driver_id=int(driver.id),
                company_id=int(driver.company_id),
                actor_id=int(driver.user_id),
            )
        except PrepClosed:
            db.session.rollback()
            return _closed()
        except PrepRefusal as exc:
            return _failure(exc)
        return view, 200
