"""Créer chauffeur canary B : atmr1@atmr / Atmr1234 (staging only)."""
from __future__ import annotations

import uuid

from app import create_app
from models import Driver, User, db
from models.enums import UserRole
from sqlalchemy import text


def main() -> None:
    app = create_app()
    # atmr1@atmr rejeté par validation email → même domaine que driver 19
    email = "atmr1@atmr.ch"
    password = "Atmr1234"
    with app.app_context():
        existing = (
            db.session.execute(
                text('SELECT id, email, role FROM "user" WHERE email = :e'),
                {"e": email},
            )
            .mappings()
            .first()
        )
        if existing:
            print("USER_EXISTS", dict(existing))
            drv = (
                db.session.execute(
                    text(
                        "SELECT id, company_id, is_active FROM driver WHERE user_id = :uid"
                    ),
                    {"uid": existing["id"]},
                )
                .mappings()
                .first()
            )
            print("DRIVER", dict(drv) if drv else None)
            # reset password
            user = db.session.get(User, existing["id"])
            if user is None:
                raise SystemExit("user missing")
            user.set_password(password, force_change=False)
            user.force_password_change = False
            if user.account_status in (None, "pending_activation", "invited", "disabled"):
                user.account_status = "active"
            db.session.commit()
            print("PASSWORD_RESET_OK")
            return

        d19 = (
            db.session.execute(
                text(
                    """
                    SELECT d.id, d.company_id, u.email
                    FROM driver d
                    JOIN "user" u ON u.id = d.user_id
                    WHERE d.id = 19
                    """
                )
            )
            .mappings()
            .first()
        )
        if not d19:
            raise SystemExit("driver 19 not found — cannot attach company")
        company_id = int(d19["company_id"])
        print("ATTACH_COMPANY", company_id, "from_driver_19", d19["email"])

        user = User()
        user.username = "atmr1"
        user.email = email
        user.role = UserRole.driver
        user.public_id = str(uuid.uuid4())
        user.first_name = "Canary"
        user.last_name = "Atmr1"
        user.account_status = "active"
        user.force_password_change = False
        user.set_password(password, force_change=False)
        db.session.add(user)
        db.session.flush()

        driver = Driver()
        driver.user_id = user.id
        driver.company_id = company_id
        driver.is_active = True
        # mirror availability if column exists
        if hasattr(driver, "is_available"):
            driver.is_available = True
        db.session.add(driver)
        db.session.commit()

        print(
            "CREATED",
            {
                "user_id": user.id,
                "public_id": user.public_id,
                "email": user.email,
                "driver_id": driver.id,
                "company_id": company_id,
                "password": password,
            },
        )


if __name__ == "__main__":
    main()
