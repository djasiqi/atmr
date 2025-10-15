from __future__ import annotations

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
    def get(self):
        company, _, _ = get_company_from_token()
        if not company:
            return {"error": "unauthorized"}, 401
        args = request.args
        driver_id = args.get("driver_id", type=int)
        q = db.session.query(DriverShift).filter(DriverShift.company_id == company.id)
        if driver_id:
            q = q.filter(DriverShift.driver_id == driver_id)
        # TODO: parse from/to as ISO
        items = [serialize_shift(s) for s in q.order_by(DriverShift.start_local).limit(500).all()]
        return {"items": items, "total": len(items)}

    @jwt_required()
    def post(self):
        company, _, _ = get_company_from_token()
        if not company:
            return {"error": "unauthorized"}, 401
        data = request.json or {}
        try:
            driver_id = int(data.get("driver_id"))
            start_local = datetime.fromisoformat(data.get("start_local"))
            end_local = datetime.fromisoformat(data.get("end_local"))
        except Exception:
            return {"error": "payload invalide"}, 400
        try:
            validate_shift_overlap(company.id, driver_id, start_local, end_local)
        except ValueError as e:
            return {"error": str(e)}, 409
        s = DriverShift(
            company_id=company.id,
            driver_id=driver_id,
            start_local=start_local,
            end_local=end_local,
            type=data.get("type", "regular"),
            status=data.get("status", "planned"),
        )
        db.session.add(s)
        db.session.commit()
        try:
            emit_shift_created(company.id, serialize_shift(s))
        except Exception:
            pass
        return serialize_shift(s), 201


@planning_ns.route("/companies/me/planning/shifts/<int:shift_id>")
class ShiftDetailMe(Resource):
    @jwt_required()
    def put(self, shift_id: int):
        company, _, _ = get_company_from_token()
        if not company:
            return {"error": "unauthorized"}, 401
        s = db.session.get(DriverShift, shift_id)
        if not s or s.company_id != company.id:
            return {"error": "not found"}, 404
        data = request.json or {}
        start_local = s.start_local
        end_local = s.end_local
        if "start_local" in data:
            try:
                start_local = datetime.fromisoformat(data["start_local"])
            except Exception:
                return {"error": "start_local invalide"}, 400
        if "end_local" in data:
            try:
                end_local = datetime.fromisoformat(data["end_local"])
            except Exception:
                return {"error": "end_local invalide"}, 400
        try:
            validate_shift_overlap(company.id, s.driver_id, start_local, end_local, exclude_id=s.id)
        except ValueError as e:
            return {"error": str(e)}, 409
        s.start_local = start_local
        s.end_local = end_local
        if "type" in data:
            s.type = data["type"]
        if "status" in data:
            s.status = data["status"]
        db.session.commit()
        try:
            emit_shift_updated(company.id, serialize_shift(s))
        except Exception:
            pass
        return serialize_shift(s)

    @jwt_required()
    def delete(self, shift_id: int):
        company, _, _ = get_company_from_token()
        if not company:
            return {"error": "unauthorized"}, 401
        s = db.session.get(DriverShift, shift_id)
        if not s or s.company_id != company.id:
            return {"error": "not found"}, 404
        db.session.delete(s)
        db.session.commit()
        try:
            emit_shift_deleted(company.id, {"id": shift_id})
        except Exception:
            pass
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
        if not s or s.company_id != company.id:
            return {"error": "not found"}, 404
        data = request.json or {}
        try:
            start_local = datetime.fromisoformat(data.get("start_local"))
            end_local = datetime.fromisoformat(data.get("end_local"))
        except Exception:
            return {"error": "payload invalide"}, 400
        b = DriverBreak(shift_id=s.id, start_local=start_local, end_local=end_local, type=data.get("type", "mandatory"))
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
        if not s or s.company_id != company.id or b.shift_id != s.id:
            return {"error": "not found"}, 404
        db.session.delete(b)
        db.session.commit()
        return {"ok": True}, 204


# --- Unavailability ---
@planning_ns.route("/companies/me/planning/unavailability")
class Unavailability(Resource):
    @jwt_required()
    def get(self):
        company, _, _ = get_company_from_token()
        if not company:
            return {"error": "unauthorized"}, 401
        args = request.args
        driver_id = args.get("driver_id", type=int)
        q = db.session.query(DriverUnavailability).filter(DriverUnavailability.company_id == company.id)
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
            driver_id = int(data.get("driver_id"))
            start_local = datetime.fromisoformat(data.get("start_local"))
            end_local = datetime.fromisoformat(data.get("end_local"))
        except Exception:
            return {"error": "payload invalide"}, 400
        u = DriverUnavailability(
            company_id=company.id,
            driver_id=driver_id,
            start_local=start_local,
            end_local=end_local,
            reason=data.get("reason", "other"),
            note=data.get("note"),
        )
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
        if not u or u.company_id != company.id:
            return {"error": "not found"}, 404
        db.session.delete(u)
        db.session.commit()
        return {"ok": True}, 204


# --- Weekly templates ---
@planning_ns.route("/companies/me/planning/weekly-template")
class WeeklyTemplate(Resource):
    @jwt_required()
    def get(self):
        company, _, _ = get_company_from_token()
        if not company:
            return {"error": "unauthorized"}, 401
        driver_id = request.args.get("driver_id", type=int)
        q = db.session.query(DriverWeeklyTemplate).filter(DriverWeeklyTemplate.company_id == company.id)
        if driver_id:
            q = q.filter(DriverWeeklyTemplate.driver_id == driver_id)
        items = [
            {
                "id": t.id,
                "driver_id": t.driver_id,
                "weekday": t.weekday,
                "start_time": t.start_time.isoformat(),
                "end_time": t.end_time.isoformat(),
                "effective_from": t.effective_from.isoformat(),
                "effective_to": t.effective_to.isoformat() if t.effective_to else None,
            }
            for t in q.order_by(DriverWeeklyTemplate.driver_id, DriverWeeklyTemplate.weekday).all()
        ]
        return {"items": items, "total": len(items)}

    @jwt_required()
    def post(self):
        company, _, _ = get_company_from_token()
        if not company:
            return {"error": "unauthorized"}, 401
        data = request.json or {}
        try:
            t = DriverWeeklyTemplate(
                company_id=company.id,
                driver_id=int(data["driver_id"]),
                weekday=int(data["weekday"]),
                start_time=datetime.fromisoformat(data["start_time"]).time(),
                end_time=datetime.fromisoformat(data["end_time"]).time(),
                effective_from=date.fromisoformat(data["effective_from"]),
                effective_to=date.fromisoformat(data["effective_to"]) if data.get("effective_to") else None,
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
        if not t or t.company_id != company.id:
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
            t.effective_to = date.fromisoformat(data["effective_to"]) if data["effective_to"] else None
        db.session.commit()
        return {"ok": True}

    @jwt_required()
    def delete(self, tid: int):
        company, _, _ = get_company_from_token()
        if not company:
            return {"error": "unauthorized"}, 401
        t = db.session.get(DriverWeeklyTemplate, tid)
        if not t or t.company_id != company.id:
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
            from_date = date.fromisoformat(data["from_date"]) if data.get("from_date") else None
            to_date = date.fromisoformat(data["to_date"]) if data.get("to_date") else None
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
            s = CompanyPlanningSettings(company_id=company.id, settings=payload)
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


