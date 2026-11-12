"""Lecture chauffeur du règlement.

Sans panneau ni recette, la réponse confirme que la situation financière
n'est pas décidée. Elle n'autorise aucun moyen et n'écrit rien.
"""

from flask_jwt_extended import jwt_required
from flask_restx import Resource

from application.settlement.driver_read_contract import (
    _safe_contract,
    driver_settlement_contract,
)
from domain.end_of_trip_settlement import activation
from domain.end_of_trip_settlement.mission_facts import (
    simulation_catalog,
    simulation_scenario,
)
from ext import role_required
from models.enums import UserRole
from routes.driver import driver_ns, get_driver_from_token


def _unavailable():
    return {"error": "Rubrique indisponible"}, 404


@driver_ns.route("/me/bookings/<int:booking_id>/settlement-read")
class DriverSettlementRead(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    def get(self, booking_id: int):
        from domain.end_of_trip_settlement.recipe_simulation import (
            recipe_simulation_read_allowed,
        )

        recipe = recipe_simulation_read_allowed()
        driver, error_response, status_code = get_driver_from_token()
        if error_response:
            return error_response, status_code
        if not activation.driver_panel_enabled() and not recipe:
            from application.settlement.unconfirmed_financial_read import (
                read_unconfirmed_financial_contract,
            )

            contract = read_unconfirmed_financial_contract(int(booking_id), driver)
            if contract is None:
                return {"error": "Mission indisponible"}, 404
            return contract, 200
        from ext import db

        try:
            contract = driver_settlement_contract(int(booking_id), driver)
        except Exception:
            db.session.rollback()
            return {"error": "Lecture indisponible"}, 503
        if contract is None:
            return {"error": "Mission indisponible"}, 404
        return contract, 200


@driver_ns.route("/me/settlement-simulation")
class DriverSettlementSimulationCatalog(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    def get(self):
        if not activation.simulation_preview_enabled():
            return _unavailable()
        return {"scenarios": simulation_catalog()}, 200


@driver_ns.route("/me/settlement-simulation/<string:scenario_id>")
class DriverSettlementSimulation(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    def get(self, scenario_id: str):
        if not activation.simulation_preview_enabled():
            return _unavailable()
        try:
            contract = simulation_scenario(scenario_id)
        except KeyError:
            return {"error": "Scénario inconnu"}, 404
        return _safe_contract(contract), 200
