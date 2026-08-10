#!/usr/bin/env python3
"""Reporter ops lecture seule — sessions mobile d'un compte.

Usage (Docker) :
  docker compose exec atmr_api python scripts/report_mobile_device_sessions.py --email user@example.com
  docker compose exec atmr_api python scripts/report_mobile_device_sessions.py --user-id 5

Ne révoque jamais. Champ classification purement informatif.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime, timedelta

from models.mobile_device_session import (
    MobileDeviceSession,
    MobileDeviceSessionStatus,
    _device_code_from_installation,
)
from security.mobile_device_session_service import get_device_session_limit


def _iso(dt: datetime | None) -> str:
    if dt is None:
        return "—"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


def _classify(session: MobileDeviceSession, *, now: datetime) -> str:
    if session.status != MobileDeviceSessionStatus.active:
        return f"revoked:{session.status.value}"
    if session.confirmed_at is None:
        if session.provisional_expires_at and session.provisional_expires_at <= now:
            return "provisional_expired"
        return "provisional"
    has_meta = bool(
        session.last_platform
        or session.last_app_version
        or session.device_model
        or session.device_manufacturer
    )
    if not has_meta:
        return "legacy_metadata"
    last = session.last_seen_at or session.last_refresh_at or session.created_at
    if last and (now - last) < timedelta(hours=24):
        return "current_recent" if session.last_refresh_at else "active_recent"
    if last and session.created_at and last <= session.created_at + timedelta(minutes=1):
        return "never_reused"
    if last and (now - last) < timedelta(days=7):
        return "active_recent"
    return "active_recent"


def report(*, email: str | None, user_id: int | None) -> int:
    from app import create_app
    from models import User

    app = create_app()
    with app.app_context():
        user = None
        if user_id is not None:
            user = User.query.filter_by(id=user_id).first()
        elif email:
            user = User.query.filter(User.email.ilike(email.strip())).first()
        if user is None:
            print("Compte introuvable.", file=sys.stderr)
            return 1

        role = user.role.value if user.role else None
        limit = get_device_session_limit(role)
        now = datetime.now(UTC)
        sessions = (
            MobileDeviceSession.query.filter_by(user_id=user.id)
            .order_by(MobileDeviceSession.created_at.desc())
            .all()
        )
        active = [
            s for s in sessions if s.status == MobileDeviceSessionStatus.active
        ]
        confirmed = [s for s in active if s.confirmed_at is not None]
        provisional = [
            s
            for s in active
            if s.confirmed_at is None
            and (
                s.provisional_expires_at is None or s.provisional_expires_at > now
            )
        ]
        expired_prov = [
            s
            for s in active
            if s.confirmed_at is None
            and s.provisional_expires_at is not None
            and s.provisional_expires_at <= now
        ]

        print("Compte")
        print(f"  user_id={user.id}")
        print(f"  email={user.email}")
        print(f"  role={role}")
        print()
        print("Quota")
        print(f"  active={len(active)}")
        print(f"  limit={limit}")
        print(f"  confirmed={len(confirmed)}")
        print(f"  provisional={len(provisional)}")
        print(f"  expired_provisional={len(expired_prov)}")
        print()
        print("Sessions")
        for idx, s in enumerate(sessions, start=1):
            code = _device_code_from_installation(s.device_installation_id)
            print(f"\n#{idx}")
            print(f"  session_id       {s.session_id}")
            print(f"  device_code      {code}")
            print(f"  status           {s.status.value if s.status else None}")
            print(f"  device_name      {s.device_name or '—'}")
            print(f"  device_model     {s.device_model or '—'}")
            print(f"  manufacturer     {s.device_manufacturer or '—'}")
            print(f"  device_type      {s.device_type or '—'}")
            print(f"  platform         {s.last_platform or '—'}")
            print(f"  OS               {s.last_os_version or '—'}")
            ver = s.last_app_version or "—"
            build = s.last_app_build or "—"
            print(f"  Lirie            {ver} ({build})")
            print(
                f"  confirmed        "
                f"{'yes' if s.confirmed_at else 'no'}"
            )
            print(f"  created          {_iso(s.created_at)}")
            print(f"  last_seen        {_iso(s.last_seen_at)}")
            print(f"  last_refresh     {_iso(s.last_refresh_at)}")
            print(f"  metadata_updated {_iso(s.metadata_updated_at)}")
            print(f"  provisional_exp  {_iso(s.provisional_expires_at)}")
            print(f"  classification   {_classify(s, now=now)}")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reporter lecture seule des sessions mobile d'un compte."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--email", type=str, help="Email du compte")
    group.add_argument("--user-id", type=int, help="ID numérique du compte")
    args = parser.parse_args()
    return report(email=args.email, user_id=args.user_id)


if __name__ == "__main__":
    raise SystemExit(main())
