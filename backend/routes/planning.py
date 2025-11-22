from __future__ import annotations

import contextlib
from datetime import date, datetime

from flask import request
from flask_jwt_extended import jwt_required
from flask_restx import Namespace, Resource

from ext import db
from models import (
    CompanyPlanningSettings,
    DriverBreak,
    DriverShift,
    DriverUnavailability,
    DriverWeeklyTemplate,
)
from routes.companies import get_company_from_token
from services.planning_service import (
    materialize_template,
    serialize_shift,
    validate_shift_overlap,
)

# from shared.time_utils import parse_local_naive
from sockets.planning import (
    emit_shift_created,
    emit_shift_deleted,
    emit_shift_updated,
)

planning_ns = Namespace("planning", description="Planning Chauffeurs")


@planning_ns.route("/companies/me/planning/shifts")
class ShiftsMe(Resource):
    @jwt_required()
    @planning_ns.param(
        "driver_id", "ID du chauffeur (optionnel, > 0)", type="integer", minimum=1
    )
    def get(self):
        """Récupère les shifts (planning) de l'entreprise.

        Query params:
            - driver_id: ID du chauffeur pour filtrer (optionnel)
        """
        company, _, _ = get_company_from_token()
        if not company:
            return {"error": "unauthorized"}, 401
        # ✅ 2.4: Validation Marshmallow pour query params
        from marshmallow import ValidationError

        from schemas.planning_schemas import PlanningShiftsQuerySchema
        from schemas.validation_utils import handle_validation_error, validate_request

        args_dict = dict(request.args)
        try:
            validated_args = validate_request(
                PlanningShiftsQuerySchema(), args_dict, strict=False
            )
            driver_id = validated_args.get("driver_id")
        except ValidationError as e:
            return handle_validation_error(e)

        q = db.session.query(DriverShift).filter(DriverShift.company_id == company.id)
        if driver_id:
            q = q.filter(DriverShift.driver_id == driver_id)
        # TODO: parse from/to as ISO
        items = [
            serialize_shift(s)
            for s in q.order_by(DriverShift.start_local).limit(500).all()
        ]
        return {"items": items, "total": len(items)}

    @jwt_required()
    def post(self):
        company, _, _ = get_company_from_token()
        if not company:
            return {"error": "unauthorized"}, 401
        data = request.json or {}
        try:
            driver_id = int(data.get("driver_id", 0))
            start_local = datetime.fromisoformat(data.get("start_local", ""))
            end_local = datetime.fromisoformat(data.get("end_local", ""))
        except Exception:
            return {"error": "payload invalide"}, 400
        try:
            validate_shift_overlap(company.id, driver_id, start_local, end_local)
        except ValueError as e:
            return {"error": str(e)}, 409
        s = DriverShift()
        s.company_id = company.id
        s.driver_id = driver_id
        s.start_local = start_local
        s.end_local = end_local
        s.type = data.get("type", "regular")
        s.status = data.get("status", "planned")
        db.session.add(s)
        db.session.commit()
        with contextlib.suppress(Exception):
            emit_shift_created(company.id, serialize_shift(s))
        return serialize_shift(s), 201


@planning_ns.route("/companies/me/planning/shifts/<int:shift_id>")
class ShiftDetailMe(Resource):
    @jwt_required()
    def put(self, shift_id: int):
        # Variables pour stocker le résultat
        result = None
        status_code = 200

        company, _, _ = get_company_from_token()
        if not company:
            result = {"error": "unauthorized"}
            status_code = 401
        else:
            s = db.session.get(DriverShift, shift_id)
            if s is None or getattr(s, "company_id", None) != getattr(
                company, "id", None
            ):
                result = {"error": "not found"}
                status_code = 404
            else:
                data = request.json or {}
                start_local = getattr(s, "start_local", None)
                end_local = getattr(s, "end_local", None)
                if "start_local" in data:
                    try:
                        start_local = datetime.fromisoformat(data["start_local"])
                    except Exception:
                        result = {"error": "start_local invalide"}
                        status_code = 400
                if result is None and "end_local" in data:
                    try:
                        end_local = datetime.fromisoformat(data["end_local"])
                    except Exception:
                        result = {"error": "end_local invalide"}
                        status_code = 400
                if result is None:
                    try:
                        if start_local is None or end_local is None:
                            result = {"error": "Invalid shift times"}
                            status_code = 400
                        else:
                            validate_shift_overlap(
                                int(getattr(company, "id", 0)),
                                int(getattr(s, "driver_id", 0)),
                                start_local,
                                end_local,
                                exclude_id=int(getattr(s, "id", 0)),
                            )
                            s.start_local = start_local
                            s.end_local = end_local
                            if "type" in data:
                                s.type = data["type"]
                            if "status" in data:
                                s.status = data["status"]
                            db.session.commit()
                            with contextlib.suppress(Exception):
                                emit_shift_updated(company.id, serialize_shift(s))
                            result = serialize_shift(s)
                    except ValueError as e:
                        result = {"error": str(e)}
                        status_code = 409

        return result, status_code

    @jwt_required()
    def delete(self, shift_id: int):
        company, _, _ = get_company_from_token()
        if not company:
            return {"error": "unauthorized"}, 401
        s = db.session.get(DriverShift, shift_id)
        if s is None or getattr(s, "company_id", None) != getattr(company, "id", None):
            return {"error": "not found"}, 404
        db.session.delete(s)
        db.session.commit()
        with contextlib.suppress(Exception):
            emit_shift_deleted(company.id, {"id": shift_id})
        return {"ok": True}, 204


# --- Breaks ---
@planning_ns.route("/companies/me/planning/shifts/<int:shift_id>/breaks")
class ShiftBreaks(Resource):
    @jwt_required()
    def post(self, shift_id: int):
        company, _, _ = get_company_from_token()
        if not company:
            return {"error": "unauthorized"}, 401
        s = db.session.get(DriverShift, shift_id)
        if s is None or getattr(s, "company_id", None) != getattr(company, "id", None):
            return {"error": "not found"}, 404
        data = request.json or {}
        try:
            start_local = datetime.fromisoformat(data.get("start_local", ""))
            end_local = datetime.fromisoformat(data.get("end_local", ""))
        except Exception:
            return {"error": "payload invalide"}, 400
        b = DriverBreak()
        b.shift_id = getattr(s, "id", None)
        b.start_local = start_local
        b.end_local = end_local
        b.type = data.get("type", "mandatory")
        db.session.add(b)
        db.session.commit()
        return {"id": b.id}, 201


@planning_ns.route("/companies/me/planning/shifts/<int:shift_id>/breaks/<int:break_id>")
class ShiftBreakDetail(Resource):
    @jwt_required()
    def delete(self, shift_id: int, break_id: int):
        company, _, _ = get_company_from_token()
        if not company:
            return {"error": "unauthorized"}, 401
        b = db.session.get(DriverBreak, break_id)
        if not b:
            return {"error": "not found"}, 404
        s = db.session.get(DriverShift, shift_id)
        if (
            s is None
            or getattr(s, "company_id", None) != getattr(company, "id", None)
            or getattr(b, "shift_id", None) != getattr(s, "id", None)
        ):
            return {"error": "not found"}, 404
        db.session.delete(b)
        db.session.commit()
        return {"ok": True}, 204


# --- Unavailability ---
@planning_ns.route("/companies/me/planning/unavailability")
class Unavailability(Resource):
    @jwt_required()
    @planning_ns.param(
        "driver_id", "ID du chauffeur (optionnel, > 0)", type="integer", minimum=1
    )
    def get(self):
        """Récupère les indisponibilités de l'entreprise.

        Query params:
            - driver_id: ID du chauffeur pour filtrer (optionnel)
        """
        company, _, _ = get_company_from_token()
        if not company:
            return {"error": "unauthorized"}, 401
        # ✅ 2.4: Validation Marshmallow pour query params
        from marshmallow import ValidationError

        from schemas.planning_schemas import PlanningUnavailabilityQuerySchema
        from schemas.validation_utils import handle_validation_error, validate_request

        args_dict = dict(request.args)
        try:
            validated_args = validate_request(
                PlanningUnavailabilityQuerySchema(), args_dict, strict=False
            )
            driver_id = validated_args.get("driver_id")
        except ValidationError as e:
            return handle_validation_error(e)

        q = db.session.query(DriverUnavailability).filter(
            DriverUnavailability.company_id == company.id
        )
        if driver_id:
            q = q.filter(DriverUnavailability.driver_id == driver_id)
        items = [
            {
                "id": u.id,
                "driver_id": u.driver_id,
                "start_local": u.start_local.isoformat(),
                "end_local": u.end_local.isoformat(),
                "reason": getattr(u.reason, "value", str(u.reason)).lower(),
                "note": u.note,
            }
            for u in q.order_by(DriverUnavailability.start_local).limit(500).all()
        ]
        return {"items": items, "total": len(items)}

    @jwt_required()
    def post(self):
        company, _, _ = get_company_from_token()
        if not company:
            return {"error": "unauthorized"}, 401
        data = request.json or {}
        try:
            driver_id = int(data.get("driver_id", 0))
            start_local = datetime.fromisoformat(data.get("start_local", ""))
            end_local = datetime.fromisoformat(data.get("end_local", ""))
        except Exception:
            return {"error": "payload invalide"}, 400
        u = DriverUnavailability()
        u.company_id = getattr(company, "id", None)
        u.driver_id = driver_id
        u.start_local = start_local
        u.end_local = end_local
        u.reason = data.get("reason", "other")
        u.note = data.get("note")
        db.session.add(u)
        db.session.commit()
        return {"id": u.id}, 201


@planning_ns.route("/companies/me/planning/unavailability/<int:uid>")
class UnavailabilityDetail(Resource):
    @jwt_required()
    def delete(self, uid: int):
        company, _, _ = get_company_from_token()
        if not company:
            return {"error": "unauthorized"}, 401
        u = db.session.get(DriverUnavailability, uid)
        if u is None or getattr(u, "company_id", None) != getattr(company, "id", None):
            return {"error": "not found"}, 404
        db.session.delete(u)
        db.session.commit()
        return {"ok": True}, 204


# --- Weekly templates ---
@planning_ns.route("/companies/me/planning/weekly-template")
class WeeklyTemplate(Resource):
    @jwt_required()
    @planning_ns.param(
        "driver_id", "ID du chauffeur (optionnel, > 0)", type="integer", minimum=1
    )
    def get(self):
        """Récupère les templates hebdomadaires de l'entreprise.

        Query params:
            - driver_id: ID du chauffeur pour filtrer (optionnel)
        """
        company, _, _ = get_company_from_token()
        if not company:
            return {"error": "unauthorized"}, 401
        # ✅ 2.4: Validation Marshmallow pour query params
        from marshmallow import ValidationError

        from schemas.planning_schemas import PlanningWeeklyTemplateQuerySchema
        from schemas.validation_utils import handle_validation_error, validate_request

        args_dict = dict(request.args)
        try:
            validated_args = validate_request(
                PlanningWeeklyTemplateQuerySchema(), args_dict, strict=False
            )
            driver_id = validated_args.get("driver_id")
        except ValidationError as e:
            return handle_validation_error(e)

        q = db.session.query(DriverWeeklyTemplate).filter(
            DriverWeeklyTemplate.company_id == company.id
        )
        if driver_id:
            q = q.filter(DriverWeeklyTemplate.driver_id == driver_id)

        def safe_isoformat(value):
            return value.isoformat() if value is not None else None

        items = [
            {
                "id": t.id,
                "driver_id": t.driver_id,
                "weekday": t.weekday,
                "start_time": safe_isoformat(getattr(t, "start_time", None)),
                "end_time": safe_isoformat(getattr(t, "end_time", None)),
                "effective_from": safe_isoformat(getattr(t, "effective_from", None)),
                "effective_to": safe_isoformat(getattr(t, "effective_to", None)),
            }
            for t in q.order_by(
                DriverWeeklyTemplate.driver_id, DriverWeeklyTemplate.weekday
            ).all()
        ]
        return {"items": items, "total": len(items)}

    @jwt_required()
    def post(self):
        company, _, _ = get_company_from_token()
        if not company:
            return {"error": "unauthorized"}, 401
        data = request.json or {}
        try:
            t = DriverWeeklyTemplate()
            t.company_id = getattr(company, "id", None)
            t.driver_id = int(data["driver_id"])
            t.weekday = int(data["weekday"])
            t.start_time = datetime.fromisoformat(data["start_time"]).time()
            t.end_time = datetime.fromisoformat(data["end_time"]).time()
            t.effective_from = date.fromisoformat(data["effective_from"])
            t.effective_to = (
                date.fromisoformat(data["effective_to"])
                if data.get("effective_to")
                else None
            )
        except Exception:
            return {"error": "payload invalide"}, 400
        db.session.add(t)
        db.session.commit()
        return {"id": t.id}, 201


@planning_ns.route("/companies/me/planning/weekly-template/<int:tid>")
class WeeklyTemplateDetail(Resource):
    @jwt_required()
    def put(self, tid: int):
        company, _, _ = get_company_from_token()
        if not company:
            return {"error": "unauthorized"}, 401
        t = db.session.get(DriverWeeklyTemplate, tid)
        if t is None or getattr(t, "company_id", None) != getattr(company, "id", None):
            return {"error": "not found"}, 404
        data = request.json or {}
        if "weekday" in data:
            t.weekday = int(data["weekday"])
        if "start_time" in data:
            t.start_time = datetime.fromisoformat(data["start_time"]).time()
        if "end_time" in data:
            t.end_time = datetime.fromisoformat(data["end_time"]).time()
        if "effective_from" in data:
            t.effective_from = date.fromisoformat(data["effective_from"])
        if "effective_to" in data:
            t.effective_to = (
                date.fromisoformat(data["effective_to"])
                if data["effective_to"]
                else None
            )
        db.session.commit()
        return {"ok": True}

    @jwt_required()
    def delete(self, tid: int):
        company, _, _ = get_company_from_token()
        if not company:
            return {"error": "unauthorized"}, 401
        t = db.session.get(DriverWeeklyTemplate, tid)
        if t is None or getattr(t, "company_id", None) != getattr(company, "id", None):
            return {"error": "not found"}, 404
        db.session.delete(t)
        db.session.commit()
        return {"ok": True}, 204


@planning_ns.route("/companies/me/planning/weekly-template/materialize")
class WeeklyTemplateMaterialize(Resource):
    @jwt_required()
    def post(self):
        company, _, _ = get_company_from_token()
        if not company:
            return {"error": "unauthorized"}, 401
        data = request.json or {}
        try:
            driver_id = int(data["driver_id"]) if data.get("driver_id") else None
            from_date = (
                date.fromisoformat(data["from_date"]) if data.get("from_date") else None
            )
            to_date = (
                date.fromisoformat(data["to_date"]) if data.get("to_date") else None
            )
        except Exception:
            return {"error": "payload invalide"}, 400
        if not (driver_id and from_date and to_date):
            return {"error": "missing fields"}, 400
        created = materialize_template(company.id, driver_id, from_date, to_date)
        return {"created": int(created)}


# --- Settings ---
@planning_ns.route("/companies/me/planning/settings")
class PlanningSettings(Resource):
    @jwt_required()
    def get(self):
        company, _, _ = get_company_from_token()
        if not company:
            return {"error": "unauthorized"}, 401
        s = db.session.get(CompanyPlanningSettings, company.id)
        return {"settings": s.settings if s else {}}

    @jwt_required()
    def put(self):
        company, _, _ = get_company_from_token()
        if not company:
            return {"error": "unauthorized"}, 401
        payload = (request.json or {}).get("settings") or {}
        s = db.session.get(CompanyPlanningSettings, company.id)
        if not s:
            s = CompanyPlanningSettings()
            s.company_id = getattr(company, "id", None)
            s.settings = payload
            db.session.add(s)
        else:
            s.settings = payload
        db.session.commit()
        return {"ok": True}


# --- Assignments overlay (lecture seule placeholder) ---
@planning_ns.route("/companies/me/planning/assignments")
class PlanningAssignments(Resource):
    @jwt_required()
    def get(self):
        company, _, _ = get_company_from_token()
        if not company:
            return {"error": "unauthorized"}, 401
        # Placeholder: renvoie liste vide pour l'instant
        return {"items": [], "total": 0}


# --- ICS export (placeholder) ---
@planning_ns.route("/companies/me/planning/ics")
class PlanningICS(Resource):
    @jwt_required()
    def get(self):
        company, _, _ = get_company_from_token()
        if not company:
            return {"error": "unauthorized"}, 401
        # Placeholder minimal: renvoie ICS vide
        ics = "BEGIN:VCALENDAR\nVERSION:2.0\nPRODID:-//ATMR//Planning//EN\nEND:VCALENDAR\n"
        return {"ics": ics}
