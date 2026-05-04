from __future__ import annotations

from app import create_app
from ext import db
from models.contact_request import ContactRequest
from models.demo_request import DemoRequest
from services.demo.dispatcher import get_demo_destination_email
from services.demo.scoring import compute_demo_score

ORG_TYPE_MAP = {
    "transport": "transport_company",
    "transport_company": "transport_company",
    "institution": "institution",
    "curatorship": "curatorship",
}

USE_CASE_MAP = {
    "transport_company": "planning_dispatch",
    "institution": "reporting",
    "curatorship": "multi_company_coordination",
}

SLOT_MAP = {
    "to_schedule": "to_define",
}


def _norm_org_type(raw: str | None) -> str:
    value = str(raw or "").strip().lower()
    return ORG_TYPE_MAP.get(value, "other")


def _norm_use_case(org_type: str) -> str:
    return USE_CASE_MAP.get(org_type, "other")


def _norm_slot(raw: str | None) -> str:
    value = str(raw or "").strip().lower()
    if not value:
        return "to_define"
    return SLOT_MAP.get(value, value)


def run_backfill() -> int:
    app = create_app()
    with app.app_context():
        rows = (
            ContactRequest.query.filter(ContactRequest.category == "demo")
            .order_by(ContactRequest.created_at.asc())
            .all()
        )

        created = 0
        skipped = 0
        destination = get_demo_destination_email()

        for row in rows:
            source = f"recovered_contact_{row.id}"
            existing = DemoRequest.query.filter(DemoRequest.source == source).first()
            if existing:
                skipped += 1
                continue

            payload = row.payload_json or {}
            org_type = _norm_org_type(payload.get("organization_type"))
            preferred_slot = _norm_slot(payload.get("preferred_slot"))
            timing = (
                str(payload.get("timing") or "exploration").strip().lower()
                or "exploration"
            )

            score_payload = {
                "name": row.name or "",
                "email": row.email or "",
                "phone": row.phone,
                "organization": row.organization or "",
                "organization_type": org_type,
                "use_case": _norm_use_case(org_type),
                "volume_range": payload.get("volume_range"),
                "integration_required": "evaluate",
                "integration_system": None,
                "timing": timing,
                "preferred_slot": preferred_slot,
                "preferred_period": "flexible",
                "comment": row.message,
            }
            score = compute_demo_score(score_payload)

            demo = DemoRequest(
                name=row.name or "",
                email=row.email or "",
                phone=row.phone,
                organization=row.organization or "",
                organization_type=org_type,
                use_case=_norm_use_case(org_type),
                volume_range=payload.get("volume_range"),
                integration_required="evaluate",
                integration_system=None,
                timing=timing,
                preferred_slot=preferred_slot,
                preferred_period="flexible",
                comment=row.message,
                score=score,
                status="new",
                trace_id=f"{row.trace_id}_demo",
                source=source,
                ip_address=None,
                user_agent=row.user_agent,
                assigned_channel=destination,
                email_delivery_status="recovered",
            )
            db.session.add(demo)
            created += 1

        db.session.commit()
        print(
            f"[backfill-demo] created={created} skipped={skipped} total_contacts={len(rows)}"
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(run_backfill())
