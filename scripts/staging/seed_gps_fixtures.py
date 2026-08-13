"""Fixtures GPS staging (données synthétiques @staging.invalid). Aucun patient réel."""

from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from flask_jwt_extended import create_access_token
from sqlalchemy import text

OUTPUT = Path(os.getenv("STAGING_FIXTURES_PATH", "/output/gps-fixtures.json"))


def _token(app, user, *, company_id: int) -> str:
    with app.app_context():
        return create_access_token(
            identity=str(user.public_id),
            additional_claims={
                "role": "driver",
                "company_id": company_id,
                "aud": "atmr-api",
            },
            expires_delta=timedelta(days=7),
        )


def _user(db, *, prefix: str, role):
    from models import User

    suffix = uuid.uuid4().hex[:8]
    user = User()
    user.username = f"{prefix}_{suffix}"
    user.email = f"{prefix}_{suffix}@staging.invalid"
    user.role = role
    user.public_id = str(uuid.uuid4())
    user.first_name = "Staging"
    user.last_name = prefix
    user.set_password(f"AtmrStg-{suffix}-Aa1!", force_change=False)
    db.session.add(user)
    db.session.flush()
    return user


def _booking(db, *, client_user, company, client, driver, status, scheduled, name):
    from models import Booking

    booking = Booking()
    booking.user_id = client_user.id
    booking.company_id = company.id
    booking.client_id = client.id
    booking.driver_id = driver.id
    booking.customer_name = name
    booking.pickup_location = "Staging Pickup"
    booking.dropoff_location = "Staging Dropoff"
    booking.scheduled_time = scheduled
    booking.status = status
    booking.amount = Decimal("10.00")
    booking.billed_to_type = "patient"
    booking.time_confirmed = True
    db.session.add(booking)
    db.session.flush()
    return booking


def main() -> None:
    os.environ.setdefault("FLASK_CONFIG", "production")
    from app import create_app
    from models import Client, Company, Driver, db
    from models.enums import BookingStatus, ClientType, ManagementMode, UserRole

    app = create_app("production")
    now = datetime.now(UTC)
    with app.app_context():
        db.session.execute(text("CREATE EXTENSION IF NOT EXISTS pg_stat_statements"))
        db.session.commit()

        company_user = _user(db, prefix="stgco", role=UserRole.company)
        company = Company()
        company.name = "Staging GPS Co"
        company.address = "Rue Staging 1"
        company.contact_email = company_user.email
        company.user_id = company_user.id
        company.is_approved = True
        company.dispatch_enabled = False
        db.session.add(company)
        db.session.flush()

        client_user = _user(db, prefix="stgcl", role=UserRole.client)
        client = Client()
        client.user_id = client_user.id
        client.company_id = company.id
        client.contact_email = client_user.email
        client.is_active = True
        client.client_type = ClientType.TRANSPORT
        client.management_mode = ManagementMode.MANAGED
        db.session.add(client)
        db.session.flush()

        def make_driver(prefix: str):
            user = _user(db, prefix=prefix, role=UserRole.driver)
            driver = Driver()
            driver.user_id = user.id
            driver.company_id = company.id
            driver.is_active = True
            db.session.add(driver)
            db.session.flush()
            return user, driver

        d_single_u, d_single = make_driver("stgsingle")
        d_none_u, d_none = make_driver("stgnone")
        d_amb_u, d_amb = make_driver("stgamb")
        d_stale_u, d_stale = make_driver("stgstale")
        d_term_u, d_term = make_driver("stgterm")
        d_mis_u, d_mis = make_driver("stgmis")

        b_none = _booking(
            db,
            client_user=client_user,
            company=company,
            client=client,
            driver=d_none,
            status=BookingStatus.ASSIGNED,
            scheduled=now + timedelta(days=10),
            name="NONE-OUT-WINDOW",
        )
        b_single = _booking(
            db,
            client_user=client_user,
            company=company,
            client=client,
            driver=d_single,
            status=BookingStatus.IN_PROGRESS,
            scheduled=now,
            name="SINGLE",
        )
        b_amb_a = _booking(
            db,
            client_user=client_user,
            company=company,
            client=client,
            driver=d_amb,
            status=BookingStatus.IN_PROGRESS,
            scheduled=now,
            name="AMBIGUOUS-A",
        )
        b_amb_b = _booking(
            db,
            client_user=client_user,
            company=company,
            client=client,
            driver=d_amb,
            status=BookingStatus.IN_PROGRESS,
            scheduled=now + timedelta(minutes=5),
            name="AMBIGUOUS-B",
        )
        b_stale_auth = _booking(
            db,
            client_user=client_user,
            company=company,
            client=client,
            driver=d_stale,
            status=BookingStatus.IN_PROGRESS,
            scheduled=now,
            name="STALE-AUTH",
        )
        b_stale_old = _booking(
            db,
            client_user=client_user,
            company=company,
            client=client,
            driver=d_stale,
            status=BookingStatus.ASSIGNED,
            scheduled=now + timedelta(days=10),
            name="STALE-OLD",
        )
        b_term = _booking(
            db,
            client_user=client_user,
            company=company,
            client=client,
            driver=d_term,
            status=BookingStatus.COMPLETED,
            scheduled=now + timedelta(hours=1),
            name="TERMINAL",
        )
        b_mis = _booking(
            db,
            client_user=client_user,
            company=company,
            client=client,
            driver=d_mis,
            status=BookingStatus.IN_PROGRESS,
            scheduled=now,
            name="MISMATCH-CANON",
        )
        db.session.commit()

        fixtures = {
            "app_env": "staging",
            "company_id": company.id,
            "scenarios": {
                "single": {
                    "expected_reason": "mission_ok",
                    "driver_id": d_single.id,
                    "mission_id": b_single.id,
                    "token": _token(app, d_single_u, company_id=company.id),
                },
                "none": {
                    "expected_reason": "assigned_outside_tracking_window",
                    "driver_id": d_none.id,
                    "mission_id": b_none.id,
                    "token": _token(app, d_none_u, company_id=company.id),
                },
                "ambiguous": {
                    "expected_reason": "ambiguous_mission",
                    "driver_id": d_amb.id,
                    "mission_id": b_amb_a.id,
                    "other_mission_id": b_amb_b.id,
                    "token": _token(app, d_amb_u, company_id=company.id),
                },
                "stale": {
                    "expected_reason": "stale_mission",
                    "driver_id": d_stale.id,
                    "mission_id": b_stale_old.id,
                    "authoritative_mission_id": b_stale_auth.id,
                    "token": _token(app, d_stale_u, company_id=company.id),
                },
                "terminal": {
                    "expected_reason": "completed_mission",
                    "driver_id": d_term.id,
                    "mission_id": b_term.id,
                    "token": _token(app, d_term_u, company_id=company.id),
                },
                "correct": {
                    "expected_reason": "mission_ok",
                    "driver_id": d_single.id,
                    "mission_id": b_single.id,
                    "token": _token(app, d_single_u, company_id=company.id),
                },
                "mismatch_canonical": {
                    "expected_reason": "mission_ok",
                    "driver_id": d_mis.id,
                    "mission_id": b_mis.id,
                    "token": _token(app, d_mis_u, company_id=company.id),
                    "canonical_poison_mission_id": 999999,
                },
            },
        }
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(json.dumps(fixtures, indent=2), encoding="utf-8")
        print(f"fixtures écrites: {OUTPUT}")
        for name, sc in fixtures["scenarios"].items():
            print(f"  {name}: driver={sc['driver_id']} mission={sc.get('mission_id')}")


if __name__ == "__main__":
    main()
