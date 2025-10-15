from __future__ import annotations

from typing import Any, cast

from flask import current_app
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
services_parser.add_argument("establishment_id", type=int, required=True, help="ID de l'établissement est requis", location="args")
services_parser.add_argument("q", type=str, required=False, help="Texte pour filtrer les services", location="args")


def _like_ci(col: Any, like_query: str):
    """LIKE insensible à la casse, portable (équivalent à ILIKE)."""
    return func.lower(cast(Any, col)).like(like_query)


# --- Routes ---

@medical_ns.route("/establishments")
class Establishments(Resource):
    @medical_ns.expect(establishments_parser)
    def get(self):
        """Autocomplete d'établissements (alias + nom + display_name)."""
        args = establishments_parser.parse_args(strict=True)
        q = (args.get("q") or "").strip()
        # borne 1..25 et assure un int
        limit = max(1, min(int(args.get("limit") or 8), 25))

        qr = db.session.query(MedicalEstablishment)

        # active = True si la colonne existe (certaines bases peuvent ne pas l'avoir)
        est_active_col = getattr(MedicalEstablishment, "active", None)
        if est_active_col is not None:
            qr = qr.filter(cast(Any, est_active_col).is_(True))

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

        rows = (
            qr.order_by(cast(Any, MedicalEstablishment.display_name).asc())
              .limit(limit)
              .all()
        )

        items: list[dict[str, Any]] = []
        for r in rows:
            items.append({
                "id": r.id,
                "source": "establishment",
                "type": getattr(r, "type", None),
                "name": getattr(r, "name", None),
                "label": getattr(r, "display_name", None),
                "address": getattr(r, "address", None),
                "lat": getattr(r, "lat", None),
                "lon": getattr(r, "lon", None),
                "aliases": r.alias_list() if hasattr(r, "alias_list") else [],
            })
        return items, 200


@medical_ns.route("/services")
class Services(Resource):
    @medical_ns.expect(services_parser)
    def get(self):
        """Liste les services actifs d'un établissement, avec filtre optionnel."""
        args = services_parser.parse_args(strict=True)
        estab_id = int(args["establishment_id"])
        q = (args.get("q") or "").strip()

        query = db.session.query(MedicalService).filter(
            cast(Any, MedicalService.establishment_id) == estab_id
        )

        # active = True si la colonne existe
        svc_active_col = getattr(MedicalService, "active", None)
        if svc_active_col is not None:
            query = query.filter(cast(Any, svc_active_col).is_(True))

        if q:
            like_query = f"%{q.lower()}%"
            query = query.filter(
                or_(
                    _like_ci(MedicalService.name, like_query),
                    _like_ci(MedicalService.category, like_query),
                )
            )

        rows = (
            query.order_by(
                cast(Any, MedicalService.category).asc(),
                cast(Any, MedicalService.name).asc()
            )
            .all()
        )

        current_app.logger.info(
            f"✅ Found {len(rows)} services for establishment {estab_id} with query '{q}'"
        )

        result = [{
            "id": r.id,
            "establishment_id": r.establishment_id,
            "category": getattr(r, "category", None),
            "name": getattr(r, "name", None),
            "active": True if svc_active_col is None else bool(getattr(r, "active", True)),
        } for r in rows]

        return result, 200
