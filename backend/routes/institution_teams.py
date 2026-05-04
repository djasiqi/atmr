# routes/institution_teams.py
# pyright: reportArgumentType=false
"""Routes CRUD pour la gestion des équipes de curateurs (curatelle).

Endpoints (admin only):
- GET    /institutions/teams          — lister les équipes avec membres + nb patients
- POST   /institutions/teams          — créer une équipe
- PUT    /institutions/teams/<id>     — renommer une équipe
- DELETE /institutions/teams/<id>     — supprimer (désassigne patients)
- POST   /institutions/teams/<id>/members            — ajouter un membre
- DELETE /institutions/teams/<id>/members/<user_id>  — retirer un membre
- PUT    /institutions/patients/<id>/team             — assigner patient à une équipe
"""

from __future__ import annotations

import logging

from flask import request
from flask_jwt_extended import jwt_required
from flask_restx import Namespace, Resource

from ext import db
from models import InstitutionPatient, User
from models.curator_team import CuratorTeam, CuratorTeamMember
from models.enums import InstitutionRole
from security.authorization import AuthorizationService

logger = logging.getLogger(__name__)

institution_teams_ns = Namespace(
    "institution_teams",
    description="Gestion des équipes de curateurs (curatelle)",
)


def _require_admin_curatelle():
    """Vérifie admin + institution type curatelle. Retourne (institution, user)."""
    institution, user = AuthorizationService.require_institution_role(
        InstitutionRole.ADMIN.value,
    )
    if (institution.institution_type or "").lower() != "curatelle":
        from flask import abort

        abort(
            400,
            description="Cette fonctionnalité est réservée aux institutions de type curatelle",
        )
    return institution, user


# ── CRUD Équipes ──────────────────────────────────────────────────────────


@institution_teams_ns.route("")
class TeamList(Resource):
    @jwt_required()
    def get(self):
        """Lister les équipes avec membres et nombre de patients."""
        institution, _user = _require_admin_curatelle()
        teams = CuratorTeam.query.filter_by(institution_id=institution.id).all()
        result = []
        for team in teams:
            t = team.serialize
            t["members"] = [m.serialize for m in team.members]
            result.append(t)
        return result, 200

    @jwt_required()
    def post(self):
        """Créer une équipe."""
        institution, _user = _require_admin_curatelle()
        data = request.get_json(silent=True) or {}
        name = (data.get("name") or "").strip()
        if not name:
            return {"error": "Le nom de l'équipe est requis"}, 400

        team = CuratorTeam(institution_id=institution.id, name=name)
        db.session.add(team)
        db.session.commit()
        logger.info(
            "[Teams] Équipe créée: %s (institution=%s)", team.name, institution.id
        )
        return team.serialize, 201


@institution_teams_ns.route("/<int:team_id>")
class TeamDetail(Resource):
    @jwt_required()
    def put(self, team_id: int):
        """Renommer une équipe."""
        institution, _user = _require_admin_curatelle()
        team = CuratorTeam.query.filter_by(
            id=team_id, institution_id=institution.id
        ).first()
        if not team:
            return {"error": "Équipe non trouvée"}, 404

        data = request.get_json(silent=True) or {}
        name = (data.get("name") or "").strip()
        if not name:
            return {"error": "Le nom de l'équipe est requis"}, 400

        team.name = name
        db.session.commit()
        return team.serialize, 200

    @jwt_required()
    def delete(self, team_id: int):
        """Supprimer une équipe (désassigne les patients, ne les supprime pas)."""
        institution, _user = _require_admin_curatelle()
        team = CuratorTeam.query.filter_by(
            id=team_id, institution_id=institution.id
        ).first()
        if not team:
            return {"error": "Équipe non trouvée"}, 404

        # Désassigner les patients de cette équipe
        InstitutionPatient.query.filter_by(curator_team_id=team_id).update(
            {"curator_team_id": None}
        )
        db.session.delete(team)
        db.session.commit()
        logger.info(
            "[Teams] Équipe supprimée: %s (institution=%s)", team.name, institution.id
        )
        return {"message": "Équipe supprimée"}, 200


# ── Gestion des membres ──────────────────────────────────────────────────


@institution_teams_ns.route("/<int:team_id>/members")
class TeamMembers(Resource):
    @jwt_required()
    def post(self, team_id: int):
        """Ajouter un membre à l'équipe."""
        institution, _user = _require_admin_curatelle()
        team = CuratorTeam.query.filter_by(
            id=team_id, institution_id=institution.id
        ).first()
        if not team:
            return {"error": "Équipe non trouvée"}, 404

        data = request.get_json(silent=True) or {}
        user_id = data.get("user_id")
        if not user_id:
            return {"error": "user_id requis"}, 400

        # Vérifier que l'utilisateur appartient à la même institution
        target_user = User.query.get(user_id)
        if not target_user or target_user.institution_id != institution.id:
            return {"error": "Utilisateur non trouvé dans cette institution"}, 404

        # Vérifier doublon
        existing = CuratorTeamMember.query.filter_by(
            team_id=team_id, user_id=user_id
        ).first()
        if existing:
            return {"error": "Utilisateur déjà membre de cette équipe"}, 409

        member = CuratorTeamMember(team_id=team_id, user_id=user_id)
        db.session.add(member)
        db.session.commit()
        return member.serialize, 201


@institution_teams_ns.route("/<int:team_id>/members/<int:user_id>")
class TeamMemberDetail(Resource):
    @jwt_required()
    def delete(self, team_id: int, user_id: int):
        """Retirer un membre de l'équipe."""
        institution, _user = _require_admin_curatelle()
        team = CuratorTeam.query.filter_by(
            id=team_id, institution_id=institution.id
        ).first()
        if not team:
            return {"error": "Équipe non trouvée"}, 404

        member = CuratorTeamMember.query.filter_by(
            team_id=team_id, user_id=user_id
        ).first()
        if not member:
            return {"error": "Membre non trouvé"}, 404

        db.session.delete(member)
        db.session.commit()
        return {"message": "Membre retiré"}, 200


# ── Assignation patient → équipe ─────────────────────────────────────────


@institution_teams_ns.route("/assign-patient/<int:patient_id>")
class AssignPatientTeam(Resource):
    @jwt_required()
    def put(self, patient_id: int):
        """Assigner un patient à une équipe (ou désassigner si team_id=null)."""
        institution, _user = _require_admin_curatelle()
        patient = InstitutionPatient.query.filter_by(
            id=patient_id, institution_id=institution.id
        ).first()
        if not patient:
            return {"error": "Patient non trouvé"}, 404

        data = request.get_json(silent=True) or {}
        team_id = data.get("team_id")

        if team_id is not None:
            team = CuratorTeam.query.filter_by(
                id=team_id, institution_id=institution.id
            ).first()
            if not team:
                return {"error": "Équipe non trouvée"}, 404

        patient.curator_team_id = team_id
        db.session.commit()
        return patient.serialize, 200
