"""API convention Client réelle — entreprise, client, chauffeur.

Aucune facturation différée réelle activée ici.
"""

from __future__ import annotations

from flask import Response, request
from flask_jwt_extended import get_jwt_identity, jwt_required
from flask_restx import Resource

from application.settlement import convention_real as service
from ext import db, role_required
from models.booking import Booking
from models.client import Client
from models.enums import UserRole
from routes.companies import companies_ns
from routes.driver import driver_ns, get_driver_from_token


def _company():
    from routes.companies import _get_current_company_via_use_case

    company, error, status = _get_current_company_via_use_case()
    if error or company is None:
        return None, (error or {"error": "Entreprise introuvable"}, status or 404)
    return company, None


def _actor_id() -> int | None:
    raw = get_jwt_identity()
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _refuse(exc: service.ConventionRealRefusal):
    return {"error": exc.message, "error_code": exc.code}, exc.http_status


@companies_ns.route("/me/clients/<int:client_id>/conventions")
class CompanyClientConventions(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self, client_id: int):
        company, err = _company()
        if err:
            return err
        try:
            return service.list_for_client(int(company.id), int(client_id)), 200
        except service.ConventionRealRefusal as exc:
            return _refuse(exc)

    @jwt_required()
    @role_required(UserRole.company)
    def post(self, client_id: int):
        company, err = _company()
        if err:
            return err
        try:
            body = service.create_draft(
                int(company.id),
                int(client_id),
                request.get_json(silent=True) or {},
                actor_id=_actor_id(),
            )
            return body, 200
        except service.ConventionRealRefusal as exc:
            return _refuse(exc)


@companies_ns.route("/me/clients/<int:client_id>/conventions/<int:convention_id>")
class CompanyClientConventionDetail(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self, client_id: int, convention_id: int):
        company, err = _company()
        if err:
            return err
        try:
            body = service.get_convention(int(company.id), int(convention_id))
            if int(body["client_id"]) != int(client_id):
                return {"error": "Convention introuvable.", "error_code": "convention_absente"}, 404
            return body, 200
        except service.ConventionRealRefusal as exc:
            return _refuse(exc)


@companies_ns.route(
    "/me/clients/<int:client_id>/conventions/<int:convention_id>/submit"
)
class CompanyConventionSubmit(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def post(self, client_id: int, convention_id: int):
        company, err = _company()
        if err:
            return err
        payload = request.get_json(silent=True) or {}
        try:
            body = service.get_convention(int(company.id), int(convention_id))
            if int(body["client_id"]) != int(client_id):
                return {"error": "Convention introuvable."}, 404
            return (
                service.submit_for_signature(
                    int(company.id),
                    int(convention_id),
                    actor_id=_actor_id(),
                    idempotency_key=str(payload.get("idempotency_key") or ""),
                ),
                200,
            )
        except service.ConventionRealRefusal as exc:
            return _refuse(exc)


@companies_ns.route(
    "/me/clients/<int:client_id>/conventions/<int:convention_id>/paper"
)
class CompanyConventionPaper(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def post(self, client_id: int, convention_id: int):
        company, err = _company()
        if err:
            return err
        try:
            body = service.get_convention(int(company.id), int(convention_id))
            if int(body["client_id"]) != int(client_id):
                return {"error": "Convention introuvable."}, 404
            return (
                service.register_paper_scan(
                    int(company.id),
                    int(convention_id),
                    request.get_json(silent=True) or {},
                    actor_id=_actor_id(),
                ),
                200,
            )
        except service.ConventionRealRefusal as exc:
            return _refuse(exc)


@companies_ns.route(
    "/me/clients/<int:client_id>/conventions/<int:convention_id>/validate"
)
class CompanyConventionValidate(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def post(self, client_id: int, convention_id: int):
        company, err = _company()
        if err:
            return err
        payload = request.get_json(silent=True) or {}
        try:
            body = service.get_convention(int(company.id), int(convention_id))
            if int(body["client_id"]) != int(client_id):
                return {"error": "Convention introuvable."}, 404
            return (
                service.validate_applicable(
                    int(company.id),
                    int(convention_id),
                    actor_id=_actor_id(),
                    idempotency_key=str(payload.get("idempotency_key") or ""),
                ),
                200,
            )
        except service.ConventionRealRefusal as exc:
            return _refuse(exc)


@companies_ns.route(
    "/me/clients/<int:client_id>/conventions/<int:convention_id>/revoke"
)
class CompanyConventionRevoke(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def post(self, client_id: int, convention_id: int):
        company, err = _company()
        if err:
            return err
        payload = request.get_json(silent=True) or {}
        try:
            body = service.get_convention(int(company.id), int(convention_id))
            if int(body["client_id"]) != int(client_id):
                return {"error": "Convention introuvable."}, 404
            return (
                service.revoke(
                    int(company.id),
                    int(convention_id),
                    actor_id=_actor_id(),
                    idempotency_key=str(payload.get("idempotency_key") or ""),
                    reason=str(payload.get("reason") or ""),
                ),
                200,
            )
        except service.ConventionRealRefusal as exc:
            return _refuse(exc)


@companies_ns.route(
    "/me/clients/<int:client_id>/conventions/<int:convention_id>/document.pdf"
)
class CompanyConventionPdf(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self, client_id: int, convention_id: int):
        company, err = _company()
        if err:
            return err
        try:
            body = service.get_convention(int(company.id), int(convention_id))
            if int(body["client_id"]) != int(client_id):
                return {"error": "Convention introuvable."}, 404
            pdf = service.read_pdf(int(company.id), int(convention_id))
            return Response(pdf, mimetype="application/pdf")
        except service.ConventionRealRefusal as exc:
            return _refuse(exc)


@driver_ns.route("/me/bookings/<int:booking_id>/convention")
class DriverBookingConvention(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    def get(self, booking_id: int):
        driver, error, status = get_driver_from_token()
        if error:
            return error, status
        booking = db.session.get(Booking, int(booking_id))
        if booking is None or int(booking.company_id) != int(driver.company_id):
            return {"error": "Mission indisponible"}, 404
        if booking.client_id is None:
            return {
                "status": "unlinked",
                "authorizes_deferred_billing": False,
                "mandate_kind": "absent",
            }, 200
        listing = service.list_for_client(int(driver.company_id), int(booking.client_id))
        gate = service.gate_for_client(int(driver.company_id), int(booking.client_id))
        return {"active": listing.get("active"), "gate": gate, "versions": listing["conventions"]}, 200


@driver_ns.route("/me/bookings/<int:booking_id>/convention/sign")
class DriverBookingConventionSign(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    def post(self, booking_id: int):
        driver, error, status = get_driver_from_token()
        if error:
            return error, status
        booking = db.session.get(Booking, int(booking_id))
        if booking is None or int(booking.company_id) != int(driver.company_id):
            return {"error": "Mission indisponible"}, 404
        if booking.client_id is None:
            return {"error": "Client non rattaché", "error_code": "patient_non_rattache"}, 400
        payload = request.get_json(silent=True) or {}
        convention_id = payload.get("convention_id")
        if convention_id is None:
            active = service.list_for_client(
                int(driver.company_id), int(booking.client_id)
            ).get("active")
            if not active:
                return {"error": "Aucune convention active", "error_code": "convention_absente"}, 404
            convention_id = active["id"]
        try:
            return (
                service.sign_convention(
                    int(driver.company_id),
                    int(convention_id),
                    payload,
                    actor_role="driver",
                    actor_id=int(driver.id),
                ),
                200,
            )
        except service.ConventionRealRefusal as exc:
            return _refuse(exc)


def _client_from_token() -> tuple[Client | None, tuple | None]:
    raw = get_jwt_identity()
    try:
        user_id = int(raw)
    except (TypeError, ValueError):
        return None, ({"error": "Non authentifié"}, 401)
    client = (
        db.session.query(Client).filter_by(user_id=user_id).order_by(Client.id).first()
    )
    if client is None:
        return None, ({"error": "Fiche client introuvable"}, 404)
    return client, None


# Portail client — namespace companies pour rester branché si clients_ns absent.
@companies_ns.route("/portal/me/conventions")
class PortalClientConventions(Resource):
    @jwt_required()
    @role_required(UserRole.client)
    def get(self):
        client, err = _client_from_token()
        if err:
            return err
        try:
            return service.list_for_client(int(client.company_id), int(client.id)), 200
        except service.ConventionRealRefusal as exc:
            return _refuse(exc)


@companies_ns.route("/portal/me/conventions/<int:convention_id>/sign")
class PortalClientConventionSign(Resource):
    @jwt_required()
    @role_required(UserRole.client)
    def post(self, convention_id: int):
        client, err = _client_from_token()
        if err:
            return err
        try:
            body = service.get_convention(int(client.company_id), int(convention_id))
            if int(body["client_id"]) != int(client.id):
                return {"error": "Convention introuvable."}, 404
            return (
                service.sign_convention(
                    int(client.company_id),
                    int(convention_id),
                    request.get_json(silent=True) or {},
                    actor_role="client",
                    actor_id=int(client.id),
                ),
                200,
            )
        except service.ConventionRealRefusal as exc:
            return _refuse(exc)


@companies_ns.route("/portal/me/conventions/<int:convention_id>/document.pdf")
class PortalClientConventionPdf(Resource):
    @jwt_required()
    @role_required(UserRole.client)
    def get(self, convention_id: int):
        client, err = _client_from_token()
        if err:
            return err
        try:
            body = service.get_convention(int(client.company_id), int(convention_id))
            if int(body["client_id"]) != int(client.id):
                return {"error": "Convention introuvable."}, 404
            pdf = service.read_pdf(int(client.company_id), int(convention_id))
            return Response(pdf, mimetype="application/pdf")
        except service.ConventionRealRefusal as exc:
            return _refuse(exc)
