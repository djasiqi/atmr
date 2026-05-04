# routes/institution_patients.py
# pyright: reportArgumentType=false, reportOperatorIssue=false, reportCallIssue=false
"""Routes pour la gestion des patients institutionnels.

Endpoints:
- POST /api/v1/institutions/patients - Créer un patient
- GET /api/v1/institutions/patients - Lister les patients
- GET /api/v1/institutions/patients/{id} - Détail patient
- PUT /api/v1/institutions/patients/{id} - Modifier patient
"""

import logging
import re
from datetime import date, datetime
from typing import Any, cast

import sentry_sdk
from flask import g, request
from flask_restx import Namespace, Resource, fields
from marshmallow import ValidationError
from sqlalchemy import and_, or_

from ext import db
from models import InstitutionPatient
from models.enums import InstitutionRole
from routes.api_error_models import (
    create_api_error_model,
    create_not_found_error_model,
    create_permission_error_model,
    create_validation_error_model,
)
from schemas.institution_schemas import (
    InstitutionPatientCreateSchema,
    InstitutionPatientQuerySchema,
    InstitutionPatientUpdateSchema,
)
from security.api_key_auth import api_key_or_jwt_required
from security.audit_log import AuditLogger
from security.authorization import AuthorizationService, get_user_team_ids
from shared.error_handlers import APIErrorHandler

logger = logging.getLogger(__name__)

# ── Constantes de validation ──
_DIGITS_ONLY_DDMMYYYY_LEN = 8
_DIGITS_ONLY_DDMMYY_LEN = 6
_SHORT_YEAR_LEN = 2
_SHORT_YEAR_THRESHOLD = 50
_MIN_NAME_SEARCH_LEN = 2
_MAX_SEARCH_RESULTS = 50

# Namespace
institution_patients_ns = Namespace(
    "institution_patients",
    description="Gestion des patients institutionnels",
)

# Modèles Swagger
api_error_model = create_api_error_model(institution_patients_ns)
not_found_error_model = create_not_found_error_model(institution_patients_ns)
permission_error_model = create_permission_error_model(institution_patients_ns)
validation_error_model = create_validation_error_model(institution_patients_ns)

# Schemas
patient_create_schema = InstitutionPatientCreateSchema()
patient_update_schema = InstitutionPatientUpdateSchema()
patient_query_schema = InstitutionPatientQuerySchema()

# Modèle de réponse patient
patient_model = institution_patients_ns.model(
    "InstitutionPatient",
    {
        "id": fields.Integer(description="ID interne"),
        "public_id": fields.String(description="ID public UUID"),
        "external_reference": fields.String(description="Référence externe DPI"),
        "first_name": fields.String(description="Prénom"),
        "last_name": fields.String(description="Nom"),
        "full_name": fields.String(description="Nom complet"),
        "dob": fields.String(description="Date de naissance (YYYY-MM-DD)"),
        "gender": fields.String(description="Genre"),
        "address": fields.String(description="Adresse"),
        "city": fields.String(description="Ville"),
        "postal_code": fields.String(description="Code postal"),
        "phone": fields.String(description="Téléphone"),
        "door_code": fields.String(description="Code porte / digicode"),
        "floor": fields.String(description="Étage"),
        "access_notes": fields.String(description="Notes d'accès"),
        "residence_name": fields.String(description="Établissement de résidence"),
        "avs_number": fields.String(description="Numéro AVS"),
        "insurance_name": fields.String(description="Caisse maladie"),
        "insurance_number": fields.String(description="Numéro d'assuré"),
        "has_guardianship": fields.Boolean(description="Sous curatelle"),
        "guardianship_type": fields.String(
            description="Type de curatelle (curatorship, opad, lawyer, family, other)"
        ),
        "guardian_name": fields.String(description="Nom du curateur"),
        "guardian_organization": fields.String(description="Organisation du curateur"),
        "guardian_phone": fields.String(description="Téléphone du curateur"),
        "guardian_email": fields.String(description="Email du curateur"),
        "guardian_address": fields.String(
            description="Adresse du curateur (facturation)"
        ),
        "notes": fields.String(description="Notes"),
        "created_at": fields.String(description="Date création"),
        "updated_at": fields.String(description="Date modification"),
    },
)

patient_list_model = institution_patients_ns.model(
    "InstitutionPatientList",
    {
        "patients": fields.List(fields.Nested(patient_model)),
        "total": fields.Integer(description="Nombre total de résultats"),
        "page": fields.Integer(description="Page courante"),
        "per_page": fields.Integer(description="Résultats par page"),
        "pages": fields.Integer(description="Nombre total de pages"),
    },
)


def get_institution_context():
    """Récupère le contexte institution pour actions d'écriture (JWT ou API Key).

    Rôles autorisés: admin, requester, curator.

    Returns:
        Tuple (institution_id, user_id_or_none)
    """
    # Si authentifié par API Key
    if hasattr(g, "institution_id") and g.get("auth_method") == "api_key":
        return g.institution_id, None

    # Sinon JWT
    institution, user = AuthorizationService.require_institution_role(
        InstitutionRole.ADMIN.value,
        InstitutionRole.REQUESTER.value,
        InstitutionRole.CURATOR.value,
    )
    return institution.id, user.id


def get_institution_read_context():
    """Récupère le contexte institution pour lecture seule (JWT ou API Key).

    Rôles autorisés: admin, requester, billing, reader, curator.

    Returns:
        Tuple (institution_id, user_id_or_none)
    """
    # Si authentifié par API Key
    if hasattr(g, "institution_id") and g.get("auth_method") == "api_key":
        return g.institution_id, None

    # Sinon JWT — tous les rôles institution peuvent lire
    institution, user = AuthorizationService.require_institution_role(
        InstitutionRole.ADMIN.value,
        InstitutionRole.REQUESTER.value,
        InstitutionRole.BILLING.value,
        InstitutionRole.READER.value,
        InstitutionRole.CURATOR.value,
    )
    return institution.id, user.id


def get_institution_write_context():
    """Récupère le contexte institution pour écriture patient (JWT ou API Key).

    Rôles autorisés: admin, requester, billing, curator.
    Le billing et le curator ont un accès restreint (filtrage des champs dans le handler).

    Returns:
        Tuple (institution_id, user_id_or_none)
    """
    # Si authentifié par API Key
    if hasattr(g, "institution_id") and g.get("auth_method") == "api_key":
        return g.institution_id, None

    # Sinon JWT — admin, requester + billing/curator (restreints)
    institution, user = AuthorizationService.require_institution_role(
        InstitutionRole.ADMIN.value,
        InstitutionRole.REQUESTER.value,
        InstitutionRole.BILLING.value,
        InstitutionRole.CURATOR.value,
    )
    return institution.id, user.id


# Champs modifiables par le rôle billing sur un patient
BILLING_PATIENT_EDITABLE_FIELDS = {
    # Coordonnées / adresse
    "phone",
    "address",
    "postal_code",
    "city",
    "residence_name",
    # Accès & logistique
    "door_code",
    "floor",
    "access_notes",
    # Assurance
    "avs_number",
    "insurance_name",
    "insurance_number",
    # Curatelle
    "has_guardianship",
    "guardianship_type",
    "guardian_name",
    "guardian_organization",
    "guardian_phone",
    "guardian_email",
    "guardian_address",
    # Notes
    "notes",
}


def _parse_date_token(token: str) -> date | None:
    """Try to parse a date from various Swiss/EU formats.

    Supported: 24.01.1993, 24/01/1993, 24-01-1993, 24.01.93, 24/01/93,
               24011993, 240193, 1993-01-24 (ISO).
    """
    token = token.strip()
    # Digits only: 24011993 or 240193
    if re.match(r"^\d{6,8}$", token):
        if len(token) == _DIGITS_ONLY_DDMMYYYY_LEN:
            # ddmmyyyy
            try:
                return datetime.strptime(token, "%d%m%Y").date()
            except ValueError:
                pass
        if len(token) == _DIGITS_ONLY_DDMMYY_LEN:
            # ddmmyy
            try:
                return datetime.strptime(token, "%d%m%y").date()
            except ValueError:
                pass
        return None

    # Try common separators: . / -
    for fmt in (
        "%d.%m.%Y",
        "%d.%m.%y",
        "%d/%m/%Y",
        "%d/%m/%y",
        "%d-%m-%Y",
        "%d-%m-%y",
        "%Y-%m-%d",  # ISO
    ):
        try:
            return datetime.strptime(token, fmt).date()
        except ValueError:
            continue

    # French month names: "24 janvier 1993"
    french_months = {
        "janvier": "01",
        "février": "02",
        "mars": "03",
        "avril": "04",
        "mai": "05",
        "juin": "06",
        "juillet": "07",
        "août": "08",
        "septembre": "09",
        "octobre": "10",
        "novembre": "11",
        "décembre": "12",
    }
    m = re.match(r"^(\d{1,2})\s+(\w+)\s+(\d{2,4})$", token, re.IGNORECASE)
    if m:
        day, month_name, year = m.group(1), m.group(2).lower(), m.group(3)
        if month_name in french_months:
            month = french_months[month_name]
            if len(year) == _SHORT_YEAR_LEN:
                year = f"20{year}" if int(year) < _SHORT_YEAR_THRESHOLD else f"19{year}"
            try:
                return date(int(year), int(month), int(day))
            except ValueError:
                pass

    return None


def _apply_smart_search(query, raw_query: str):
    """Apply intelligent search: split tokens, match name/dob."""
    # Split by spaces (supports "Dupont 24.01.1993" or "Jean Dupont")
    tokens = raw_query.split()
    filters = []

    for token in tokens:
        parsed_date = _parse_date_token(token)
        if parsed_date:
            filters.append(InstitutionPatient.dob == parsed_date)
        else:
            like = f"%{token}%"
            filters.append(
                or_(
                    InstitutionPatient.first_name.ilike(like),
                    InstitutionPatient.last_name.ilike(like),
                    InstitutionPatient.external_reference.ilike(like),
                )
            )

    if filters:
        query = query.filter(and_(*filters))

    return query


@institution_patients_ns.route("")
class InstitutionPatientList(Resource):
    """Endpoints pour lister et créer des patients."""

    @institution_patients_ns.doc(
        description="Liste les patients de l'institution.",
        params={
            "query": "Recherche par nom/prénom",
            "external_reference": "Filtre par référence externe",
            "page": "Numéro de page (défaut: 1)",
            "per_page": "Résultats par page (défaut: 20, max: 100)",
        },
    )
    @institution_patients_ns.response(200, "Succès", patient_list_model)
    @institution_patients_ns.response(401, "Non authentifié", permission_error_model)
    @institution_patients_ns.response(403, "Accès refusé", permission_error_model)
    @api_key_or_jwt_required(scopes=["patients:read"])
    def get(self):
        """Liste les patients de l'institution.

        Auth: JWT (tous rôles institution) ou API Key (scope patients:read)
        """
        try:
            institution_id, _ = get_institution_read_context()

            # Valider query params
            try:
                params = cast(dict[str, Any], patient_query_schema.load(request.args))
            except ValidationError as err:
                return {"error": "Paramètres invalides", "details": err.messages}, 400

            # Base query
            query = InstitutionPatient.query.filter_by(institution_id=institution_id)

            # Filtrage curator : si des équipes existent et que le curateur y est assigné,
            # ne montrer que les patients de ses équipes. Sinon (pas d'équipe ou non assigné),
            # montrer tous les patients de l'institution (mode bootstrapping).
            institution_role = AuthorizationService.get_institution_role_from_jwt()
            if institution_role == InstitutionRole.CURATOR.value:
                user = AuthorizationService.require_user()
                team_ids = get_user_team_ids(user.id)
                if team_ids:
                    query = query.filter(
                        db.or_(
                            InstitutionPatient.curator_team_id.in_(team_ids),
                            InstitutionPatient.curator_team_id.is_(None),
                        )
                    )

            # Filtres — recherche intelligente (nom, prénom, date de naissance)
            if params.get("query"):
                raw_query = params["query"].strip()
                query = _apply_smart_search(query, raw_query)

            if params.get("external_reference"):
                query = query.filter_by(external_reference=params["external_reference"])

            # Pagination
            page = params.get("page", 1)
            per_page = params.get("per_page", 20)
            total = query.count()
            pages = (total + per_page - 1) // per_page

            patients = (
                query.order_by(
                    InstitutionPatient.last_name, InstitutionPatient.first_name
                )
                .offset((page - 1) * per_page)
                .limit(per_page)
                .all()
            )

            return {
                "patients": [p.serialize for p in patients],
                "total": total,
                "page": page,
                "per_page": per_page,
                "pages": pages,
            }, 200

        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.error("[InstitutionPatients] GET error: %s", e)
            return APIErrorHandler.handle_exception(e, logger)

    @institution_patients_ns.doc(
        description="Crée un nouveau patient.",
    )
    @institution_patients_ns.response(201, "Patient créé", patient_model)
    @institution_patients_ns.response(400, "Données invalides", validation_error_model)
    @institution_patients_ns.response(401, "Non authentifié", permission_error_model)
    @institution_patients_ns.response(403, "Accès refusé", permission_error_model)
    @institution_patients_ns.response(
        409, "Référence externe déjà utilisée", api_error_model
    )
    @api_key_or_jwt_required(scopes=["patients:write"])
    def post(self):
        """Crée un nouveau patient.

        Auth: JWT (institution_admin/requester) ou API Key (scope patients:write)

        Idempotence: Si external_reference existe déjà, retourne 409.
        """
        try:
            institution_id, user_id = get_institution_context()

            data = request.get_json() or {}

            # Valider
            try:
                validated = cast(dict[str, Any], patient_create_schema.load(data))
            except ValidationError as err:
                return {"error": "Données invalides", "details": err.messages}, 400

            # Vérifier unicité external_reference
            ext_ref = validated.get("external_reference")
            if ext_ref:
                existing = InstitutionPatient.find_by_external_reference(
                    institution_id, ext_ref
                )
                if existing:
                    return {
                        "error": f"Patient avec external_reference '{ext_ref}' existe déjà",
                        "existing_patient_id": existing.id,
                        "existing_patient_public_id": existing.public_id,
                    }, 409

            # Vérifier doublon nom + prénom + date de naissance
            first_name = validated["first_name"].strip()
            last_name = validated["last_name"].strip()
            dob_str = validated.get("dob")
            force_create = data.get("force_create", False)

            if not force_create:
                dup_query = InstitutionPatient.query.filter(
                    InstitutionPatient.institution_id == institution_id,
                    db.func.lower(db.func.trim(InstitutionPatient.first_name))
                    == first_name.lower(),
                    db.func.lower(db.func.trim(InstitutionPatient.last_name))
                    == last_name.lower(),
                )
                if dob_str:
                    dob_parsed = datetime.strptime(dob_str, "%Y-%m-%d").date()
                    dup_query = dup_query.filter(InstitutionPatient.dob == dob_parsed)

                duplicates = dup_query.limit(3).all()
                if duplicates:
                    return {
                        "error": "Un patient avec le même nom et prénom existe déjà",
                        "code": "DUPLICATE_PATIENT",
                        "duplicates": [
                            {
                                "id": d.id,
                                "first_name": d.first_name,
                                "last_name": d.last_name,
                                "dob": d.dob.isoformat() if d.dob else None,
                                "phone": d.phone,
                                "address": d.address,
                            }
                            for d in duplicates
                        ],
                    }, 409

            # Créer patient
            patient = InstitutionPatient()
            patient.institution_id = institution_id
            patient.external_reference = ext_ref
            patient.first_name = validated["first_name"]
            patient.last_name = validated["last_name"]
            patient.dob = (
                datetime.strptime(validated["dob"], "%Y-%m-%d").date()
                if validated.get("dob")
                else None
            )
            patient.gender = validated.get("gender")
            patient.address = validated.get("address")
            patient.city = validated.get("city")
            patient.postal_code = validated.get("postal_code")
            patient.phone = validated.get("phone")
            patient.door_code = validated.get("door_code")
            patient.floor = validated.get("floor")
            patient.access_notes = validated.get("access_notes")
            patient.residence_name = validated.get("residence_name")
            patient.avs_number = validated.get("avs_number")
            patient.insurance_name = validated.get("insurance_name")
            patient.insurance_number = validated.get("insurance_number")
            patient.has_guardianship = validated.get("has_guardianship", False)
            patient.guardianship_type = validated.get("guardianship_type")
            patient.guardian_name = validated.get("guardian_name")
            patient.guardian_organization = validated.get("guardian_organization")
            patient.guardian_phone = validated.get("guardian_phone")
            patient.guardian_email = validated.get("guardian_email")
            patient.guardian_address = validated.get("guardian_address")
            patient.notes = validated.get("notes")

            # Auto-assignation équipe : si le créateur est curateur,
            # assigner le patient à sa première équipe
            institution_role = AuthorizationService.get_institution_role_from_jwt()
            assigned_team_id = None
            if institution_role == InstitutionRole.CURATOR.value and user_id:
                team_ids = get_user_team_ids(user_id)
                if team_ids:
                    assigned_team_id = team_ids[0]
                    patient.curator_team_id = assigned_team_id

            db.session.add(patient)
            db.session.flush()

            sync_result = {
                "status": "none",
                "suggestions_count": 0,
                "identity_id": None,
            }
            try:
                from services.patient_sync.patient_identity_service import (
                    trigger_sync_on_create,
                )

                result = trigger_sync_on_create(patient, user_id)
                if result:
                    sync_result = result
            except Exception as sync_err:
                logger.warning(
                    "[InstitutionPatients] Sync on create error: %s", sync_err
                )

            db.session.commit()

            try:
                AuditLogger.log_action(
                    action_type="patient_created",
                    action_category="institution",
                    user_id=user_id,
                    user_type="institution" if user_id else "api_key",
                    institution_id=institution_id,
                    result_status="success",
                    action_details={
                        "patient_id": patient.id,
                        "external_reference": ext_ref,
                        "curator_team_id": assigned_team_id,
                        "sync_status": sync_result["status"],
                    },
                    ip_address=request.remote_addr,
                    user_agent=request.headers.get("User-Agent"),
                )
            except Exception as audit_err:
                logger.warning("[InstitutionPatients] Audit log error: %s", audit_err)

            logger.info(
                "[InstitutionPatients] Patient créé: id=%s, institution=%s, sync=%s",
                patient.id,
                institution_id,
                sync_result["status"],
            )

            return {
                "patient": patient.serialize,
                "sync": sync_result,
            }, 201

        except Exception as e:
            db.session.rollback()
            sentry_sdk.capture_exception(e)
            logger.error("[InstitutionPatients] POST error: %s", e)
            return APIErrorHandler.handle_exception(e, logger)


@institution_patients_ns.route("/<int:patient_id>")
class InstitutionPatientDetail(Resource):
    """Endpoints pour détail et modification d'un patient."""

    @institution_patients_ns.doc(
        description="Récupère les détails d'un patient.",
    )
    @institution_patients_ns.response(200, "Succès", patient_model)
    @institution_patients_ns.response(401, "Non authentifié", permission_error_model)
    @institution_patients_ns.response(403, "Accès refusé", permission_error_model)
    @institution_patients_ns.response(404, "Patient non trouvé", not_found_error_model)
    @api_key_or_jwt_required(scopes=["patients:read"])
    def get(self, patient_id: int):
        """Récupère les détails d'un patient.

        Auth: JWT (tous rôles institution) ou API Key (scope patients:read)
        """
        try:
            institution_id, _ = get_institution_read_context()

            patient = InstitutionPatient.query.filter_by(
                id=patient_id,
                institution_id=institution_id,
            ).first()

            if not patient:
                return {"error": "Patient non trouvé"}, 404

            # Vérifier accès curator : si le curateur a des équipes,
            # autoriser accès si le patient est dans son équipe ou sans équipe
            institution_role = AuthorizationService.get_institution_role_from_jwt()
            if institution_role == InstitutionRole.CURATOR.value:
                user = AuthorizationService.require_user()
                team_ids = get_user_team_ids(user.id)
                if (
                    team_ids
                    and patient.curator_team_id
                    and patient.curator_team_id not in team_ids
                ):
                    return {"error": "Patient non trouvé"}, 404

            return patient.serialize, 200

        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.error("[InstitutionPatients] GET /%s error: %s", patient_id, e)
            return APIErrorHandler.handle_exception(e, logger)

    @institution_patients_ns.doc(
        description="Modifie un patient.",
    )
    @institution_patients_ns.response(200, "Patient modifié", patient_model)
    @institution_patients_ns.response(400, "Données invalides", validation_error_model)
    @institution_patients_ns.response(401, "Non authentifié", permission_error_model)
    @institution_patients_ns.response(403, "Accès refusé", permission_error_model)
    @institution_patients_ns.response(404, "Patient non trouvé", not_found_error_model)
    @institution_patients_ns.response(
        409, "Référence externe déjà utilisée", api_error_model
    )
    @api_key_or_jwt_required(scopes=["patients:write"])
    def put(self, patient_id: int):
        """Modifie un patient.

        Auth: JWT (institution_admin/requester/billing) ou API Key (scope patients:write)

        Le rôle billing ne peut modifier que les champs suivants :
        adresse/coordonnées, assurance, curatelle.
        """
        try:
            institution_id, user_id = get_institution_write_context()

            patient = InstitutionPatient.query.filter_by(
                id=patient_id,
                institution_id=institution_id,
            ).first()

            if not patient:
                return {"error": "Patient non trouvé"}, 404

            # Capturer les anciennes valeurs pour le delta sync
            from services.patient_sync.patient_identity_service import SYNCABLE_FIELDS

            old_values = {f: getattr(patient, f, None) for f in SYNCABLE_FIELDS}

            data = request.get_json() or {}

            # Valider
            try:
                validated = cast(dict[str, Any], patient_update_schema.load(data))
            except ValidationError as err:
                return {"error": "Données invalides", "details": err.messages}, 400

            # Si rôle billing, restreindre aux champs autorisés
            institution_role = AuthorizationService.get_institution_role_from_jwt()
            if institution_role == InstitutionRole.BILLING.value:
                validated = {
                    k: v
                    for k, v in validated.items()
                    if k in BILLING_PATIENT_EDITABLE_FIELDS
                }
                if not validated:
                    return {
                        "error": "Aucun champ modifiable avec le rôle facturation"
                    }, 403

            # Vérifier unicité external_reference si changé
            new_ext_ref = validated.get("external_reference")
            if new_ext_ref and new_ext_ref != patient.external_reference:
                existing = InstitutionPatient.find_by_external_reference(
                    institution_id, new_ext_ref
                )
                if existing and existing.id != patient.id:
                    return {
                        "error": f"Patient avec external_reference '{new_ext_ref}' existe déjà",
                    }, 409

            # Appliquer modifications
            if "external_reference" in validated:
                patient.external_reference = validated["external_reference"]
            if "first_name" in validated:
                patient.first_name = validated["first_name"]
            if "last_name" in validated:
                patient.last_name = validated["last_name"]
            if "dob" in validated:
                patient.dob = (
                    datetime.strptime(validated["dob"], "%Y-%m-%d").date()
                    if validated["dob"]
                    else None
                )
            if "gender" in validated:
                patient.gender = validated["gender"]
            if "address" in validated:
                patient.address = validated["address"]
            if "city" in validated:
                patient.city = validated["city"]
            if "postal_code" in validated:
                patient.postal_code = validated["postal_code"]
            if "phone" in validated:
                patient.phone = validated["phone"]
            if "door_code" in validated:
                patient.door_code = validated["door_code"]
            if "floor" in validated:
                patient.floor = validated["floor"]
            if "access_notes" in validated:
                patient.access_notes = validated["access_notes"]
            if "residence_name" in validated:
                patient.residence_name = validated["residence_name"]
            if "avs_number" in validated:
                patient.avs_number = validated["avs_number"]
            if "insurance_name" in validated:
                patient.insurance_name = validated["insurance_name"]
            if "insurance_number" in validated:
                patient.insurance_number = validated["insurance_number"]
            if "has_guardianship" in validated:
                patient.has_guardianship = validated["has_guardianship"]
            if "guardianship_type" in validated:
                patient.guardianship_type = validated["guardianship_type"]
            if "guardian_name" in validated:
                patient.guardian_name = validated["guardian_name"]
            if "guardian_organization" in validated:
                patient.guardian_organization = validated["guardian_organization"]
            if "guardian_phone" in validated:
                patient.guardian_phone = validated["guardian_phone"]
            if "guardian_email" in validated:
                patient.guardian_email = validated["guardian_email"]
            if "guardian_address" in validated:
                patient.guardian_address = validated["guardian_address"]
            if "notes" in validated:
                patient.notes = validated["notes"]

            db.session.commit()

            # Déclencher la synchronisation cross-plateforme si curatelle + AVS
            try:
                from services.patient_sync.patient_identity_service import (
                    trigger_sync_if_needed,
                )

                sync_event = trigger_sync_if_needed(patient, old_values, user_id)
                if sync_event:
                    db.session.commit()
                    logger.info(
                        "[InstitutionPatients] Sync event créé pour patient %s",
                        patient.id,
                    )
            except Exception as sync_err:
                logger.warning("[InstitutionPatients] Sync trigger error: %s", sync_err)

            # Audit log — séparer champs sensibles des champs normaux
            SENSITIVE_FIELDS = {
                "avs_number",
                "insurance_name",
                "insurance_number",
                "has_guardianship",
                "guardianship_type",
                "guardian_name",
                "guardian_organization",
                "guardian_phone",
                "guardian_email",
                "guardian_address",
            }
            updated_fields = list(validated.keys())
            sensitive_updated = [f for f in updated_fields if f in SENSITIVE_FIELDS]
            normal_updated = [f for f in updated_fields if f not in SENSITIVE_FIELDS]

            try:
                # Audit standard (champs non-sensibles)
                if normal_updated:
                    AuditLogger.log_action(
                        action_type="patient_updated",
                        action_category="institution",
                        user_id=user_id,
                        user_type="institution" if user_id else "api_key",
                        institution_id=institution_id,
                        result_status="success",
                        action_details={
                            "patient_id": patient.id,
                            "updated_fields": normal_updated,
                        },
                        ip_address=request.remote_addr,
                        user_agent=request.headers.get("User-Agent"),
                    )
                # Audit dédié données sensibles (noms de champs SANS valeurs)
                if sensitive_updated:
                    AuditLogger.log_action(
                        action_type="patient_admin_data_updated",
                        action_category="institution",
                        user_id=user_id,
                        user_type="institution" if user_id else "api_key",
                        institution_id=institution_id,
                        result_status="success",
                        action_details={
                            "patient_id": patient.id,
                            "sensitive_fields_changed": sensitive_updated,
                        },
                        ip_address=request.remote_addr,
                        user_agent=request.headers.get("User-Agent"),
                    )
            except Exception as audit_err:
                logger.warning("[InstitutionPatients] Audit log error: %s", audit_err)

            logger.info(
                "[InstitutionPatients] Patient modifié: id=%s",
                patient.id,
            )

            return patient.serialize, 200

        except Exception as e:
            db.session.rollback()
            sentry_sdk.capture_exception(e)
            logger.error("[InstitutionPatients] PUT /%s error: %s", patient_id, e)
            return APIErrorHandler.handle_exception(e, logger)


@institution_patients_ns.route("/by-reference/<string:external_reference>")
class InstitutionPatientByReference(Resource):
    """Endpoint pour récupérer un patient par référence externe."""

    @institution_patients_ns.doc(
        description="Récupère un patient par sa référence externe DPI.",
    )
    @institution_patients_ns.response(200, "Succès", patient_model)
    @institution_patients_ns.response(401, "Non authentifié", permission_error_model)
    @institution_patients_ns.response(403, "Accès refusé", permission_error_model)
    @institution_patients_ns.response(404, "Patient non trouvé", not_found_error_model)
    @api_key_or_jwt_required(scopes=["patients:read"])
    def get(self, external_reference: str):
        """Récupère un patient par référence externe.

        Auth: JWT (tous rôles institution) ou API Key (scope patients:read)
        """
        try:
            institution_id, _ = get_institution_read_context()

            patient = InstitutionPatient.find_by_external_reference(
                institution_id, external_reference
            )

            if not patient:
                return {"error": "Patient non trouvé"}, 404

            return patient.serialize, 200

        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.error(
                "[InstitutionPatients] GET /by-reference/%s error: %s",
                external_reference,
                e,
            )
            return APIErrorHandler.handle_exception(e, logger)


# ── Endpoints Identity / Matching / Sync ──────────────────────────────────


@institution_patients_ns.route("/<int:patient_id>/identity")
class PatientIdentityInfo(Resource):
    """Informations de liaison du patient avec le Master Index."""

    @api_key_or_jwt_required(scopes=["patients:read"])
    def get(self, patient_id: int):
        """Retourne les infos d'identité et liens du patient (admin only)."""
        try:
            from models.patient_identity import (
                PatientAuditLog,
                PatientIdentity,
                PatientIdentityLink,
            )

            institution_id, user_id = get_institution_read_context()

            # Réservé admin
            institution_role = AuthorizationService.get_institution_role_from_jwt()
            if institution_role not in (InstitutionRole.ADMIN.value,):
                return {"error": "Accès réservé aux administrateurs"}, 403

            patient = InstitutionPatient.query.filter_by(
                id=patient_id,
                institution_id=institution_id,
            ).first()
            if not patient:
                return {"error": "Patient non trouvé"}, 404

            link = PatientIdentityLink.query.filter_by(
                entity_type="institution_patient",
                entity_id=patient.id,
                is_active=True,
            ).first()

            if not link:
                return {"identity": None, "linked": False}, 200

            identity = PatientIdentity.query.get(link.patient_identity_id)

            # Audit log lecture
            db.session.add(
                PatientAuditLog(
                    actor_user_id=user_id,
                    action="READ_IDENTITY_LINKS",
                    entity_type="institution_patient",
                    entity_id=patient.id,
                    metadata_json={"identity_id": identity.id if identity else None},
                )
            )
            db.session.commit()

            return {
                "linked": True,
                "identity": identity.serialize if identity else None,
                "link": link.serialize,
            }, 200

        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.error(
                "[InstitutionPatients] GET /%s/identity error: %s", patient_id, e
            )
            return APIErrorHandler.handle_exception(e, logger)


@institution_patients_ns.route("/<int:patient_id>/matches")
class PatientMatches(Resource):
    """Suggestions de correspondance pour un patient sans AVS."""

    @api_key_or_jwt_required(scopes=["patients:read"])
    def get(self, patient_id: int):
        """Retourne les correspondances potentielles (score + signaux)."""
        try:
            from services.patient_sync.patient_matching_service import (
                find_potential_matches,
            )

            institution_id, _ = get_institution_read_context()

            patient = InstitutionPatient.query.filter_by(
                id=patient_id,
                institution_id=institution_id,
            ).first()
            if not patient:
                return {"error": "Patient non trouvé"}, 404

            # Vérifier accès curator
            institution_role = AuthorizationService.get_institution_role_from_jwt()
            if institution_role == InstitutionRole.CURATOR.value:
                user = AuthorizationService.require_user()
                team_ids = get_user_team_ids(user.id)
                if (
                    team_ids
                    and patient.curator_team_id
                    and patient.curator_team_id not in team_ids
                ):
                    return {"error": "Patient non trouvé"}, 404

            matches = find_potential_matches(
                patient_id=patient.id,
                first_name=patient.first_name,
                last_name=patient.last_name,
                dob=patient.dob,
                city=patient.city,
                phone=patient.phone,
            )

            return {"matches": matches, "total": len(matches)}, 200

        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.error(
                "[InstitutionPatients] GET /%s/matches error: %s", patient_id, e
            )
            return APIErrorHandler.handle_exception(e, logger)


@institution_patients_ns.route("/<int:patient_id>/matches/<int:identity_id>/confirm")
class PatientMatchConfirm(Resource):
    """Confirmer un match et créer le lien."""

    @api_key_or_jwt_required(scopes=["patients:write"])
    def post(self, patient_id: int, identity_id: int):
        """Confirme la correspondance et crée le lien identity."""
        try:
            from models.patient_identity import (
                PatientAuditLog,
                PatientIdentity,
                PatientIdentityLink,
            )

            institution_id, user_id = get_institution_write_context()

            patient = InstitutionPatient.query.filter_by(
                id=patient_id,
                institution_id=institution_id,
            ).first()
            if not patient:
                return {"error": "Patient non trouvé"}, 404

            identity = PatientIdentity.query.get(identity_id)
            if not identity:
                return {"error": "Identité non trouvée"}, 404

            # Vérifier pas déjà lié
            existing = PatientIdentityLink.query.filter_by(
                patient_identity_id=identity.id,
                entity_type="institution_patient",
                entity_id=patient.id,
            ).first()
            if existing and existing.is_active:
                return {"error": "Lien déjà existant"}, 409

            link = PatientIdentityLink(
                patient_identity_id=identity.id,
                entity_type="institution_patient",
                entity_id=patient.id,
                link_method="name_dob_confirmed",
                is_active=True,
                linked_by_user_id=user_id,
            )
            db.session.add(link)

            db.session.add(
                PatientAuditLog(
                    actor_user_id=user_id,
                    action="LINK_CONFIRMED",
                    entity_type="institution_patient",
                    entity_id=patient.id,
                    metadata_json={
                        "identity_id": identity.id,
                        "link_method": "name_dob_confirmed",
                    },
                )
            )
            db.session.commit()

            return {"message": "Correspondance confirmée", "link": link.serialize}, 201

        except Exception as e:
            db.session.rollback()
            sentry_sdk.capture_exception(e)
            logger.error(
                "[InstitutionPatients] POST /%s/matches/%s/confirm error: %s",
                patient_id,
                identity_id,
                e,
            )
            return APIErrorHandler.handle_exception(e, logger)


@institution_patients_ns.route("/<int:patient_id>/matches/<int:identity_id>/reject")
class PatientMatchReject(Resource):
    """Rejeter un match pour ne pas le reproposer."""

    @api_key_or_jwt_required(scopes=["patients:write"])
    def post(self, patient_id: int, identity_id: int):
        """Rejette la suggestion de correspondance."""
        try:
            from models.patient_identity import (
                PatientAuditLog,
                PatientMatchRejection,
            )

            institution_id, user_id = get_institution_write_context()

            patient = InstitutionPatient.query.filter_by(
                id=patient_id,
                institution_id=institution_id,
            ).first()
            if not patient:
                return {"error": "Patient non trouvé"}, 404

            # Vérifier pas déjà rejeté
            existing = PatientMatchRejection.query.filter_by(
                patient_id=patient.id,
                identity_id=identity_id,
            ).first()
            if existing:
                return {"message": "Déjà rejeté"}, 200

            rejection = PatientMatchRejection(
                patient_id=patient.id,
                identity_id=identity_id,
                rejected_by_user_id=user_id,
            )
            db.session.add(rejection)

            db.session.add(
                PatientAuditLog(
                    actor_user_id=user_id,
                    action="MATCH_REJECTED",
                    entity_type="institution_patient",
                    entity_id=patient.id,
                    metadata_json={"identity_id": identity_id},
                )
            )
            db.session.commit()

            return {"message": "Suggestion rejetée"}, 200

        except Exception as e:
            db.session.rollback()
            sentry_sdk.capture_exception(e)
            logger.error(
                "[InstitutionPatients] POST /%s/matches/%s/reject error: %s",
                patient_id,
                identity_id,
                e,
            )
            return APIErrorHandler.handle_exception(e, logger)


@institution_patients_ns.route("/<int:patient_id>/identity/detach")
class PatientIdentityDetach(Resource):
    """Détacher un patient de son identité (soft detach, admin only)."""

    @api_key_or_jwt_required(scopes=["patients:write"])
    def put(self, patient_id: int):
        """Soft detach : désactive le lien sans le supprimer."""
        try:
            from datetime import UTC, datetime

            from models.patient_identity import PatientAuditLog, PatientIdentityLink

            institution_id, user_id = get_institution_write_context()

            # Admin uniquement
            institution_role = AuthorizationService.get_institution_role_from_jwt()
            if institution_role not in (InstitutionRole.ADMIN.value,):
                return {"error": "Accès réservé aux administrateurs"}, 403

            patient = InstitutionPatient.query.filter_by(
                id=patient_id,
                institution_id=institution_id,
            ).first()
            if not patient:
                return {"error": "Patient non trouvé"}, 404

            link = PatientIdentityLink.query.filter_by(
                entity_type="institution_patient",
                entity_id=patient.id,
                is_active=True,
            ).first()
            if not link:
                return {"error": "Aucun lien actif à détacher"}, 404

            data = request.get_json(silent=True) or {}
            reason = data.get("reason", "")

            link.is_active = False
            link.detached_at = datetime.now(UTC)
            link.detached_by_user_id = user_id
            link.detach_reason = reason[:200] if reason else None

            db.session.add(
                PatientAuditLog(
                    actor_user_id=user_id,
                    action="DETACH",
                    entity_type="institution_patient",
                    entity_id=patient.id,
                    metadata_json={
                        "identity_id": link.patient_identity_id,
                        "reason": reason[:200] if reason else None,
                    },
                )
            )
            db.session.commit()

            return {"message": "Lien détaché", "link": link.serialize}, 200

        except Exception as e:
            db.session.rollback()
            sentry_sdk.capture_exception(e)
            logger.error(
                "[InstitutionPatients] PUT /%s/identity/detach error: %s",
                patient_id,
                e,
            )
            return APIErrorHandler.handle_exception(e, logger)


@institution_patients_ns.route("/<int:patient_id>/sync-status")
class PatientSyncStatus(Resource):
    """Statut de synchronisation du patient."""

    @api_key_or_jwt_required(scopes=["patients:read"])
    def get(self, patient_id: int):
        """Retourne le statut de sync et les derniers événements."""
        try:
            from models.patient_identity import (
                PatientIdentity,
                PatientIdentityLink,
                PatientSyncEvent,
            )

            institution_id, _ = get_institution_read_context()

            patient = InstitutionPatient.query.filter_by(
                id=patient_id,
                institution_id=institution_id,
            ).first()
            if not patient:
                return {"error": "Patient non trouvé"}, 404

            link = PatientIdentityLink.query.filter_by(
                entity_type="institution_patient",
                entity_id=patient.id,
                is_active=True,
            ).first()

            if not link:
                return {"synced": False, "events": []}, 200

            identity = PatientIdentity.query.get(link.patient_identity_id)

            # Derniers events
            recent_events = (
                PatientSyncEvent.query.filter_by(
                    patient_identity_id=link.patient_identity_id
                )
                .order_by(PatientSyncEvent.created_at.desc())
                .limit(5)
                .all()
            )

            source_info = None
            if identity and identity.source_institution_id:
                from models.institution import Institution

                source_inst = Institution.query.get(identity.source_institution_id)
                if source_inst:
                    source_info = {
                        "institution_name": source_inst.name,
                        "institution_type": source_inst.institution_type,
                    }

            return {
                "synced": True,
                "identity": identity.serialize if identity else None,
                "source": source_info,
                "data_source_flags": patient.data_source_flags,
                "events": [e.serialize for e in recent_events],
            }, 200

        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.error(
                "[InstitutionPatients] GET /%s/sync-status error: %s",
                patient_id,
                e,
            )
            return APIErrorHandler.handle_exception(e, logger)


# ── Endpoints Link Suggestions ─────────────────────────────────────────────


@institution_patients_ns.route("/<int:patient_id>/suggestions")
class PatientLinkSuggestions(Resource):
    """Suggestions de lien en attente de confirmation humaine."""

    @api_key_or_jwt_required(scopes=["patients:read"])
    def get(self, patient_id: int):
        """Retourne les suggestions pending et non expirées."""
        try:
            from models.patient_identity import PatientLinkSuggestion

            institution_id, _ = get_institution_read_context()

            patient = InstitutionPatient.query.filter_by(
                id=patient_id,
                institution_id=institution_id,
            ).first()
            if not patient:
                return {"error": "Patient non trouvé"}, 404

            suggestions = (
                PatientLinkSuggestion.query.filter_by(
                    source_patient_id=patient.id, status="pending"
                )
                .filter(PatientLinkSuggestion.expires_at > db.func.now())
                .order_by(PatientLinkSuggestion.match_score.desc())
                .all()
            )

            results = []
            for s in suggestions:
                data = s.serialize
                if s.target_entity_type == "institution_patient":
                    target = InstitutionPatient.query.get(s.target_entity_id)
                    if target:
                        data["target_patient"] = {
                            "id": target.id,
                            "first_name": target.first_name,
                            "last_name": target.last_name,
                            "dob": target.dob.isoformat() if target.dob else None,
                            "city": target.city,
                            "phone": target.phone,
                            "institution_name": (
                                target.institution.name if target.institution else None
                            ),
                        }
                results.append(data)

            return {"suggestions": results, "total": len(results)}, 200

        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.error(
                "[InstitutionPatients] GET /%s/suggestions error: %s",
                patient_id,
                e,
            )
            return APIErrorHandler.handle_exception(e, logger)


@institution_patients_ns.route(
    "/<int:patient_id>/suggestions/<int:suggestion_id>/confirm"
)
class PatientLinkSuggestionConfirm(Resource):
    """Confirmer une suggestion de lien."""

    @api_key_or_jwt_required(scopes=["patients:write"])
    def post(self, patient_id: int, suggestion_id: int):
        """Confirme la suggestion, crée le lien et déclenche la sync."""
        try:
            from datetime import UTC, datetime

            from models.patient_identity import (
                PatientAuditLog,
                PatientIdentity,
                PatientIdentityLink,
                PatientLinkSuggestion,
            )
            from services.patient_sync.patient_identity_service import (
                compute_creation_delta,
                create_sync_event,
                ensure_identity_and_link,
            )

            institution_id, user_id = get_institution_write_context()

            patient = InstitutionPatient.query.filter_by(
                id=patient_id,
                institution_id=institution_id,
            ).first()
            if not patient:
                return {"error": "Patient non trouvé"}, 404

            suggestion = (
                db.session.query(PatientLinkSuggestion)
                .filter_by(id=suggestion_id, source_patient_id=patient.id)
                .with_for_update()
                .first()
            )
            if not suggestion:
                return {"error": "Suggestion non trouvée"}, 404

            if suggestion.status != "pending":
                return {
                    "error": "Suggestion déjà traitée",
                    "status": suggestion.status,
                }, 409

            already_linked = PatientIdentityLink.query.filter_by(
                entity_type="institution_patient",
                entity_id=patient.id,
                is_active=True,
            ).first()
            if already_linked:
                suggestion.status = "expired"
                suggestion.resolved_at = datetime.now(UTC)
                db.session.commit()
                return {"error": "Patient déjà lié à une identité"}, 409

            identity = None
            if suggestion.target_identity_id:
                identity = PatientIdentity.query.get(suggestion.target_identity_id)

            if not identity and suggestion.target_entity_type == "institution_patient":
                target_patient = InstitutionPatient.query.get(
                    suggestion.target_entity_id
                )
                if target_patient and target_patient.avs_number:
                    identity = ensure_identity_and_link(target_patient, user_id)

            if not identity:
                identity = PatientIdentity(
                    avs_hash=f"noavs_{patient.id}_{suggestion.target_entity_id}",
                    avs_status="unknown",
                    canonical_first_name=patient.first_name,
                    canonical_last_name=patient.last_name,
                    canonical_dob=patient.dob,
                    version=1,
                    confidence_level="medium",
                    source_institution_id=patient.institution_id,
                    source_patient_id=patient.id,
                )
                db.session.add(identity)
                db.session.flush()

            link = PatientIdentityLink(
                patient_identity_id=identity.id,
                entity_type="institution_patient",
                entity_id=patient.id,
                link_method="name_dob_confirmed",
                is_active=True,
                linked_by_user_id=user_id,
            )
            db.session.add(link)

            if suggestion.target_entity_type == "institution_patient":
                target_link_exists = PatientIdentityLink.query.filter_by(
                    patient_identity_id=identity.id,
                    entity_type="institution_patient",
                    entity_id=suggestion.target_entity_id,
                ).first()
                if not target_link_exists:
                    target_link = PatientIdentityLink(
                        patient_identity_id=identity.id,
                        entity_type="institution_patient",
                        entity_id=suggestion.target_entity_id,
                        link_method="name_dob_confirmed",
                        is_active=True,
                        linked_by_user_id=user_id,
                    )
                    db.session.add(target_link)

            suggestion.status = "confirmed"
            suggestion.resolved_by_user_id = user_id
            suggestion.resolved_at = datetime.now(UTC)

            db.session.add(
                PatientAuditLog(
                    actor_user_id=user_id,
                    action="LINK_CONFIRMED",
                    entity_type="institution_patient",
                    entity_id=patient.id,
                    metadata_json={
                        "identity_id": identity.id,
                        "suggestion_id": suggestion.id,
                        "link_method": "name_dob_confirmed",
                        "target_entity_type": suggestion.target_entity_type,
                        "target_entity_id": suggestion.target_entity_id,
                    },
                )
            )

            changed = compute_creation_delta(patient)
            sync_event = None
            if changed:
                sync_event = create_sync_event(identity, patient, changed, user_id)

            db.session.commit()

            return {
                "message": "Lien confirmé, synchronisation déclenchée",
                "link": link.serialize,
                "sync_event_created": sync_event is not None,
            }, 201

        except Exception as e:
            db.session.rollback()
            sentry_sdk.capture_exception(e)
            logger.error(
                "[InstitutionPatients] POST /%s/suggestions/%s/confirm error: %s",
                patient_id,
                suggestion_id,
                e,
            )
            return APIErrorHandler.handle_exception(e, logger)


@institution_patients_ns.route(
    "/<int:patient_id>/suggestions/<int:suggestion_id>/reject"
)
class PatientLinkSuggestionReject(Resource):
    """Rejeter une suggestion de lien."""

    @api_key_or_jwt_required(scopes=["patients:write"])
    def post(self, patient_id: int, suggestion_id: int):
        """Rejette la suggestion pour ne plus la reproposer."""
        try:
            from datetime import UTC, datetime

            from models.patient_identity import (
                PatientAuditLog,
                PatientLinkSuggestion,
                PatientMatchRejection,
            )

            institution_id, user_id = get_institution_write_context()

            patient = InstitutionPatient.query.filter_by(
                id=patient_id,
                institution_id=institution_id,
            ).first()
            if not patient:
                return {"error": "Patient non trouvé"}, 404

            suggestion = PatientLinkSuggestion.query.filter_by(
                id=suggestion_id,
                source_patient_id=patient.id,
            ).first()
            if not suggestion:
                return {"error": "Suggestion non trouvée"}, 404

            if suggestion.status != "pending":
                return {
                    "message": "Suggestion déjà traitée",
                    "status": suggestion.status,
                }, 200

            suggestion.status = "rejected"
            suggestion.resolved_by_user_id = user_id
            suggestion.resolved_at = datetime.now(UTC)

            if suggestion.target_identity_id:
                existing_rejection = PatientMatchRejection.query.filter_by(
                    patient_id=patient.id,
                    identity_id=suggestion.target_identity_id,
                ).first()
                if not existing_rejection:
                    db.session.add(
                        PatientMatchRejection(
                            patient_id=patient.id,
                            identity_id=suggestion.target_identity_id,
                            rejected_by_user_id=user_id,
                        )
                    )

            db.session.add(
                PatientAuditLog(
                    actor_user_id=user_id,
                    action="MATCH_REJECTED",
                    entity_type="institution_patient",
                    entity_id=patient.id,
                    metadata_json={
                        "suggestion_id": suggestion.id,
                        "target_entity_type": suggestion.target_entity_type,
                        "target_entity_id": suggestion.target_entity_id,
                    },
                )
            )
            db.session.commit()

            return {"message": "Suggestion rejetée"}, 200

        except Exception as e:
            db.session.rollback()
            sentry_sdk.capture_exception(e)
            logger.error(
                "[InstitutionPatients] POST /%s/suggestions/%s/reject error: %s",
                patient_id,
                suggestion_id,
                e,
            )
            return APIErrorHandler.handle_exception(e, logger)
