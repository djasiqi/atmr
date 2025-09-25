from flask import request, current_app
from flask_restx import Namespace, Resource, reqparse
from sqlalchemy import or_, func
from models import db, MedicalEstablishment, MedicalService

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

# --- Routes ---

@medical_ns.route("/establishments")
class Establishments(Resource):
    @medical_ns.expect(establishments_parser)
    def get(self):
        """Autocomplete d'établissements (alias + nom + display_name)."""
        args = establishments_parser.parse_args(strict=True)
        q = (args["q"] or "").strip()
        limit = max(1, min(args["limit"], 25)) # Sécurise la limite

        qr = db.session.query(MedicalEstablishment).filter(MedicalEstablishment.active.is_(True))
        
        if q:
            like_query = f"%{q.lower()}%"
            qr = qr.filter(
                or_(
                    func.lower(MedicalEstablishment.name).like(like_query),
                    func.lower(MedicalEstablishment.display_name).like(like_query),
                    func.lower(MedicalEstablishment.aliases).like(like_query),
                    func.lower(MedicalEstablishment.address).like(like_query),
                )
            )
            
        rows = qr.order_by(MedicalEstablishment.display_name.asc()).limit(limit).all()

        items = []
        for r in rows:
            items.append({
                "id": r.id,
                "source": "establishment",
                "type": r.type,
                "name": r.name,
                "label": r.display_name,
                "address": r.address,
                "lat": r.lat,
                "lon": r.lon,
                "aliases": r.alias_list() if hasattr(r, "alias_list") else [],
            })
        return items

@medical_ns.route("/services")
class Services(Resource):
    @medical_ns.expect(services_parser)
    def get(self):
        """Liste les services actifs d'un établissement, avec filtre optionnel."""
        args = services_parser.parse_args(strict=True)
        estab_id = args["establishment_id"]
        q = (args["q"] or "").strip()

        # Requête de base
        query = (
            db.session.query(MedicalService)
            .filter(MedicalService.establishment_id == estab_id, MedicalService.active.is_(True))
        )

        # Ajout du filtre de recherche si 'q' est fourni
        if q:
            like_query = f"%{q.lower()}%"
            query = query.filter(
                or_(
                    func.lower(MedicalService.name).like(like_query),
                    func.lower(MedicalService.category).like(like_query)
                )
            )

        rows = query.order_by(MedicalService.category.asc(), MedicalService.name.asc()).all()

        current_app.logger.info(f"✅ Found {len(rows)} services for establishment {estab_id} with query '{q}'")
        
        result = [{
            "id": r.id,
            "establishment_id": r.establishment_id,
            "category": r.category,
            "name": r.name,
            "active": True,
        } for r in rows]
        
        return result