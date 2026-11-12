"""Dossier fictif chauffeur. 404 tant que la barrière serveur est fermée."""

from __future__ import annotations

from flask import request
from flask_jwt_extended import jwt_required
from flask_restx import Resource

from application.settlement.simulation_cases import (
    LedgerClosed,
    open_case,
    read_case,
    transition_case,
)
from domain.end_of_trip_settlement.simulation_barrier import ledger_open
from domain.end_of_trip_settlement.simulation_machine import SimulationRefusal
from ext import db, role_required
from models.booking import Booking
from models.enums import UserRole
from routes.driver import driver_ns, get_driver_from_token


def _closed():
    return {"error": "Rubrique indisponible", "error_code": "LEDGER_CLOSED"}, 404


def _driver_booking(booking_id: int):
    if not ledger_open():
        return None, None, _closed()
    driver, error, status = get_driver_from_token()
    if error:
        return None, None, (error, status)
    booking = db.session.get(Booking, booking_id)
    if booking is None or int(booking.company_id) != int(driver.company_id):
        return None, None, ({"error": "Mission indisponible"}, 404)
    return driver, booking, None


def _failure(exc: SimulationRefusal):
    db.session.rollback()
    status = 404 if exc.status == 404 else exc.status
    return {"error": exc.code, "error_code": exc.code}, status


@driver_ns.route("/me/bookings/<int:booking_id>/settlement-simulation-case")
class DriverSettlementSimulationCase(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    def get(self, booking_id: int):
        driver, booking, error = _driver_booking(booking_id)
        if error:
            return error
        try:
            view = read_case(
                booking, driver_id=int(driver.id), company_id=int(driver.company_id)
            )
        except LedgerClosed:
            return _closed()
        except SimulationRefusal as exc:
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
            view = open_case(
                booking, driver_id=int(driver.id), company_id=int(driver.company_id)
            )
        except LedgerClosed:
            db.session.rollback()
            return _closed()
        except SimulationRefusal as exc:
            return _failure(exc)
        return view, 200


@driver_ns.route("/me/bookings/<int:booking_id>/settlement-simulation-case/transitions")
class DriverSettlementSimulationTransition(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    def post(self, booking_id: int):
        driver, booking, error = _driver_booking(booking_id)
        if error:
            return error
        body = request.get_json(silent=True) or {}
        try:
            view = transition_case(
                booking,
                driver_id=int(driver.id),
                company_id=int(driver.company_id),
                actor_role="driver",
                actor_id=int(driver.id),
                action=str(body.get("action") or ""),
                idempotency_key=str(body.get("idempotency_key") or ""),
                payload=body,
            )
        except LedgerClosed:
            db.session.rollback()
            return _closed()
        except SimulationRefusal as exc:
            return _failure(exc)
        return view, 200
