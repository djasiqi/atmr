from __future__ import annotations

from typing import Any, cast

from flask import current_app, request
from flask_restx import Namespace, Resource, reqparse
from sqlalchemy import func, or_

from models import MedicalEstablishment, MedicalService, db

# --- Namespace et Parsers ---

medical_ns = Namespace("medical", description="Recherche d'établissements et services médicaux")

# Parser pour l'endpoint /establishments
establishments_parser = reqparse.RequestParser()
establishments_parser.add_argument("q", type=str, required=False, help="Texte à rechercher", location="args")
establishments_parser.add_argument("limit", type=int, default=8, help="Nombre max de résultats", location="args")

# Parser pour l'endpoint /services
services_parser = reqparse.RequestParser()
services_parser.add_argument(
    "establishment_id", type=int, required=True, help="ID de l'établissement est requis", location="args"
)
services_parser.add_argument("q", type=str, required=False, help="Texte pour filtrer les services", location="args")


def _like_ci(col: Any, like_query: str):
    """LIKE insensible à la casse, portable (équivalent à ILIKE)."""
    return func.lower(col).like(like_query)


# --- Routes ---


@medical_ns.route("/establishments")
class Establishments(Resource):
    @medical_ns.expect(establishments_parser)
    def get(self):
        """Autocomplete d'établissements (alias + nom + display_name)."""
        # ✅ 2.4: Validation Marshmallow pour query params (optionnel mais recommandé)

        from schemas.medical_schemas import MedicalEstablishmentQuerySchema
        from schemas.validation_utils import validate_request

        args_dict = dict(request.args)
        try:
            validated_args = validate_request(MedicalEstablishmentQuerySchema(), args_dict, strict=False)
            q = (validated_args.get("q") or "").strip()
            limit = validated_args.get("limit", 8)
        except Exception:
            # Fallback sur reqparse si validation échoue
            args = establishments_parser.parse_args(strict=True)
            q = (args.get("q") or "").strip()
            limit = max(1, min(int(args.get("limit") or 8), 25))

        qr = db.session.query(MedicalEstablishment)

        # active = True si la colonne existe (certaines bases peuvent ne pas
        # l'avoir)
        est_active_col = getattr(MedicalEstablishment, "active", None)
        if est_active_col is not None:
            qr = qr.filter(est_active_col.is_(True))

        if q:
            like_query = f"%{q.lower()}%"
            qr = qr.filter(
                or_(
                    _like_ci(MedicalEstablishment.name, like_query),
                    _like_ci(MedicalEstablishment.display_name, like_query),
                    _like_ci(MedicalEstablishment.aliases, like_query),
                    _like_ci(MedicalEstablishment.address, like_query),
                )
            )

        rows = qr.order_by(cast("Any", MedicalEstablishment.display_name).asc()).limit(limit).all()

        items: list[dict[str, Any]] = []
        for r in rows:
            items.append(
                {
                    "id": r.id,
                    "source": "establishment",
                    "type": getattr(r, "type", None),
                    "name": getattr(r, "name", None),
                    "label": getattr(r, "display_name", None),
                    "address": getattr(r, "address", None),
                    "lat": getattr(r, "lat", None),
                    "lon": getattr(r, "lon", None),
                    "aliases": r.alias_list() if hasattr(r, "alias_list") else [],
                }
            )
        return items, 200


@medical_ns.route("/services")
class Services(Resource):
    @medical_ns.expect(services_parser)
    def get(self):
        """Liste les services actifs d'un établissement, avec filtre optionnel."""
        # ✅ 2.4: Validation Marshmallow pour query params (optionnel mais recommandé)
        from marshmallow import ValidationError

        from schemas.medical_schemas import MedicalServiceQuerySchema
        from schemas.validation_utils import handle_validation_error, validate_request

        args_dict = dict(request.args)
        try:
            validated_args = validate_request(MedicalServiceQuerySchema(), args_dict)
            estab_id = validated_args["establishment_id"]
            q = (validated_args.get("q") or "").strip()
        except ValidationError as e:
            return handle_validation_error(e)
        except Exception as fallback_err:
            # Fallback sur reqparse si validation échoue
            try:
                args = services_parser.parse_args(strict=True)
                estab_id = int(args["establishment_id"])
                q = (args.get("q") or "").strip()
            except Exception:
                # Si reqparse échoue aussi, retourner 400
                return {"error": "establishment_id is required", "message": str(fallback_err)}, 400

        query = db.session.query(MedicalService).filter(cast("Any", MedicalService.establishment_id) == estab_id)

        # active = True si la colonne existe
        svc_active_col = getattr(MedicalService, "active", None)
        if svc_active_col is not None:
            query = query.filter(svc_active_col.is_(True))

        if q:
            like_query = f"%{q.lower()}%"
            query = query.filter(
                or_(
                    _like_ci(MedicalService.name, like_query),
                    _like_ci(MedicalService.category, like_query),
                )
            )

        rows = query.order_by(cast("Any", MedicalService.category).asc(), cast("Any", MedicalService.name).asc()).all()

        current_app.logger.info("✅ Found %s services for establishment %s with query '%s'", len(rows), estab_id, q)

        result = [
            {
                "id": r.id,
                "establishment_id": r.establishment_id,
                "category": getattr(r, "category", None),
                "name": getattr(r, "name", None),
                "active": True if svc_active_col is None else bool(getattr(r, "active", True)),
            }
            for r in rows
        ]

        return result, 200
