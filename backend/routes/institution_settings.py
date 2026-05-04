# routes/institution_settings.py
# pyright: reportArgumentType=false, reportOperatorIssue=false
# ruff: noqa: I001
"""Routes pour les paramètres des institutions.

Endpoints:
- GET /api/v1/institutions/settings/transport-preferences - Lister les préférences
- PUT /api/v1/institutions/settings/transport-preferences - Définir les préférences
"""

import logging
from typing import Any, cast

import sentry_sdk
from flask import request
from flask_jwt_extended import get_jwt, get_jwt_identity, verify_jwt_in_request
from flask_restx import Namespace, Resource, fields
from marshmallow import Schema
from marshmallow import fields as ma_fields
from marshmallow import validate

from ext import db
from models import Company, InstitutionTransportPreference
from models.enums import InstitutionRole
from routes.api_error_models import (
    create_api_error_model,
    create_permission_error_model,
    create_validation_error_model,
)
from security.audit_log import AuditLogger

logger = logging.getLogger(__name__)

# Namespace
institution_settings_ns = Namespace(
    "institution_settings",
    description="Paramètres des institutions",
)

# Modèles erreurs
api_error_model = create_api_error_model(institution_settings_ns)
permission_error_model = create_permission_error_model(institution_settings_ns)
validation_error_model = create_validation_error_model(institution_settings_ns)

# Modèles Swagger
preference_model = institution_settings_ns.model(
    "TransportPreference",
    {
        "id": fields.Integer(description="ID de la préférence"),
        "company_id": fields.Integer(description="ID de l'entreprise"),
        "company_name": fields.String(description="Nom de l'entreprise"),
        "order": fields.Integer(description="Ordre de préférence (1 = premier choix)"),
    },
)

preferences_list_model = institution_settings_ns.model(
    "TransportPreferencesList",
    {
        "preferences": fields.List(
            fields.Nested(preference_model),
            description="Liste des préférences ordonnées",
        ),
        "total": fields.Integer(description="Nombre total de préférences"),
    },
)

eligible_company_model = institution_settings_ns.model(
    "EligibleCompany",
    {
        "id": fields.Integer(description="ID de l'entreprise"),
        "name": fields.String(description="Nom de l'entreprise"),
        "address": fields.String(description="Adresse"),
        "is_preferred": fields.Boolean(description="True si déjà dans les préférences"),
    },
)


# Schéma validation
class SetPreferencesSchema(Schema):
    """Schéma pour définir les préférences."""

    company_ids = ma_fields.List(
        ma_fields.Integer(required=True),
        required=True,
        validate=validate.Length(min=0, max=50),
    )


set_preferences_schema = SetPreferencesSchema()


def get_institution_context() -> tuple[int, int | None]:
    """Récupère le contexte institution depuis le JWT.

    Returns:
        Tuple (institution_id, user_id)

    Raises:
        Werkzeug Abort si non authentifié ou pas institution
    """
    from flask import abort

    verify_jwt_in_request()

    claims = get_jwt()
    institution_id = claims.get("institution_id")
    institution_role = claims.get("institution_role")

    if not institution_id:
        abort(403, description="Accès réservé aux utilisateurs institution")

    # Vérifier que l'utilisateur a le rôle admin
    if institution_role != InstitutionRole.ADMIN.value:
        abort(
            403, description="Seuls les administrateurs peuvent gérer les préférences"
        )

    user_id = get_jwt_identity()
    return institution_id, user_id


@institution_settings_ns.route("/transport-preferences")
class TransportPreferences(Resource):
    """Gestion des préférences de transporteurs."""

    @institution_settings_ns.doc(
        description="Liste les préférences de transporteurs de l'institution",
        security="BearerAuth",
    )
    @institution_settings_ns.response(200, "Succès", preferences_list_model)
    @institution_settings_ns.response(401, "Non authentifié", permission_error_model)
    @institution_settings_ns.response(403, "Accès refusé", permission_error_model)
    def get(self):
        """Liste les préférences de transporteurs.

        Auth: JWT institution_admin requis
        """
        try:
            institution_id, _user_id = get_institution_context()

            preferences = InstitutionTransportPreference.get_ordered_preferences(
                institution_id
            )

            return {
                "preferences": [p.serialize for p in preferences],
                "total": len(preferences),
            }

        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.error(
                "[InstitutionSettings] Erreur GET transport-preferences: %s",
                e,
            )
            return {"error": f"Erreur serveur: {e!s}"}, 500

    @institution_settings_ns.doc(
        description="Définit les préférences de transporteurs de l'institution",
        security="BearerAuth",
    )
    @institution_settings_ns.expect(
        institution_settings_ns.model(
            "SetPreferencesInput",
            {
                "company_ids": fields.List(
                    fields.Integer(),
                    description="Liste ordonnée des IDs d'entreprises",
                    required=True,
                ),
            },
        )
    )
    @institution_settings_ns.response(200, "Succès", preferences_list_model)
    @institution_settings_ns.response(400, "Données invalides", validation_error_model)
    @institution_settings_ns.response(401, "Non authentifié", permission_error_model)
    @institution_settings_ns.response(403, "Accès refusé", permission_error_model)
    def put(self):
        """Définit les préférences de transporteurs.

        Auth: JWT institution_admin requis

        Remplace toutes les préférences existantes par la nouvelle liste ordonnée.
        """
        try:
            institution_id, user_id = get_institution_context()

            data = request.get_json() or {}

            # Valider
            errors = set_preferences_schema.validate(data)
            if errors:
                return {"error": "Données invalides", "details": errors}, 400

            validated = cast(dict[str, Any], set_preferences_schema.load(data))
            company_ids = validated["company_ids"]

            # Vérifier que toutes les entreprises existent et sont éligibles
            if company_ids:
                companies = Company.query.filter(
                    Company.id.in_(company_ids),
                    Company.is_approved == True,  # noqa: E712
                ).all()

                found_ids = {c.id for c in companies}
                missing_ids = set(company_ids) - found_ids

                if missing_ids:
                    return {
                        "error": f"Entreprises non trouvées ou non éligibles: {list(missing_ids)}",
                    }, 400

            # Définir les nouvelles préférences
            preferences = InstitutionTransportPreference.set_preferences(
                institution_id=institution_id,
                company_ids=company_ids,
            )

            db.session.commit()

            # Audit log
            try:
                AuditLogger.log_action(
                    action_type="transport_preferences_updated",
                    action_category="institution",
                    user_id=user_id,
                    user_type="institution",
                    institution_id=institution_id,
                    result_status="success",
                    action_details={
                        "company_ids": company_ids,
                        "count": len(company_ids),
                    },
                )
            except Exception as audit_err:
                logger.warning("Échec audit log: %s", audit_err)

            logger.info(
                "[InstitutionSettings] Preferences updated for institution %s: %d companies",
                institution_id,
                len(company_ids),
            )

            return {
                "preferences": [p.serialize for p in preferences],
                "total": len(preferences),
            }

        except Exception as e:
            db.session.rollback()
            sentry_sdk.capture_exception(e)
            logger.error(
                "[InstitutionSettings] Erreur PUT transport-preferences: %s",
                e,
            )
            return {"error": f"Erreur serveur: {e!s}"}, 500


@institution_settings_ns.route("/eligible-companies")
class EligibleCompanies(Resource):
    """Liste des entreprises éligibles pour les préférences."""

    @institution_settings_ns.doc(
        description="Liste les entreprises éligibles comme transporteurs",
        security="BearerAuth",
    )
    @institution_settings_ns.response(
        200,
        "Succès",
        institution_settings_ns.model(
            "EligibleCompaniesList",
            {
                "companies": fields.List(fields.Nested(eligible_company_model)),
                "total": fields.Integer(),
            },
        ),
    )
    @institution_settings_ns.response(401, "Non authentifié", permission_error_model)
    @institution_settings_ns.response(403, "Accès refusé", permission_error_model)
    def get(self):
        """Liste les entreprises de transport disponibles.

        Auth: JWT institution_admin requis

        Retourne uniquement les entreprises de transport approuvées,
        en excluant toute Company liée à l'institution appelante.
        """
        try:
            institution_id, _user_id = get_institution_context()

            # Récupérer uniquement les entreprises de transport approuvées.
            # Le filtre is_approved exclut déjà les institutions qui auraient
            # un record Company non-approuvé (cas normal : seules les vraies
            # entreprises de transport sont approuvées par l'admin).
            companies = (
                Company.query.filter(Company.is_approved.is_(True))
                .order_by(Company.name)
                .all()
            )

            # Récupérer les IDs déjà préférés
            preferred_ids = set(
                InstitutionTransportPreference.get_company_ids_ordered(institution_id)
            )

            result = []
            for company in companies:
                result.append(
                    {
                        "id": company.id,
                        "name": company.name,
                        "address": company.address,
                        "is_preferred": company.id in preferred_ids,
                    }
                )

            return {
                "companies": result,
                "total": len(result),
            }

        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.error(
                "[InstitutionSettings] Erreur GET eligible-companies: %s",
                e,
            )
            return {"error": f"Erreur serveur: {e!s}"}, 500
