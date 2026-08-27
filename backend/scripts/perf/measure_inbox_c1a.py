"""C1a - mesure avant/apres du de-N+1 inbox messages hub.

Usage (dans le container atmr_api, jamais sur l'hote) :
    python scripts/perf/measure_inbox_c1a.py --seed
    python scripts/perf/measure_inbox_c1a.py --measure --label before
    python scripts/perf/measure_inbox_c1a.py --measure --label after

Mesure : nb requetes SQL, duree SQL cumulee, duree handler, nb conversations/
threads, unread_total - pour build_company_inbox, hub_threads_for_company et
build_driver_inbox. Dump JSON complet trie pour verifier l'equivalence
fonctionnelle avant/apres.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import perf_counter

os.environ.setdefault("FLASK_CONFIG", "development")

SEED_EMAIL_COMPANY = "c1a.perf.company@lirie.test"
SEED_EMAIL_DRIVER = "c1a.perf.driver@lirie.test"
SEED_COMPANY_NAME = "_C1A_PERF_CO"
BASE_TS = datetime(2026, 8, 20, 8, 0, 0, tzinfo=UTC)

N_COMPANY_CONVS = 25
N_MISSION_ACTIVE = 15
N_MISSION_TERMINAL = 10
N_GROUP = 5
MSGS_PER_CONV = 50
READ_FIRST = 30  # messages lus (MessageRead) sur une conv sur deux

OUT_DIR = Path(__file__).parent / "out"


def _get_app():
    from app import create_app

    return create_app(os.environ.get("FLASK_CONFIG", "development"))


def seed() -> None:
    from ext import db
    from models import (
        Booking,
        Company,
        Conversation,
        ConversationParticipant,
        Driver,
        Message,
        MessageRead,
        User,
    )
    from models.enums import SenderRole, UserRole
    from models.messaging_enums import ConversationContext, ConversationType

    existing = User.query.filter_by(email=SEED_EMAIL_COMPANY).first()
    if existing:
        company = Company.query.filter_by(name=SEED_COMPANY_NAME).first()
        n_conv = (
            Conversation.query.filter_by(company_id=company.id).count()
            if company
            else 0
        )
        print(f"seed: deja present (company_id={getattr(company, 'id', None)}, convs={n_conv})")
        return

    owner = User(
        email=SEED_EMAIL_COMPANY,
        username="c1a_perf_company",
        password="x" * 60,
        role=UserRole.COMPANY,
    )
    db.session.add(owner)
    db.session.flush()
    company = Company(name=SEED_COMPANY_NAME, user_id=owner.id)
    db.session.add(company)
    db.session.flush()

    drv_user = User(
        email=SEED_EMAIL_DRIVER,
        username="c1a_perf_driver",
        password="x" * 60,
        role=UserRole.DRIVER,
    )
    db.session.add(drv_user)
    db.session.flush()
    driver = Driver(user_id=drv_user.id, company_id=company.id)
    db.session.add(driver)
    db.session.flush()

    convs: list[Conversation] = []

    def add_conv(ctype: str, title: str, *, context_id: int | None = None,
                 context_type: str | None = None, legacy: str | None = None) -> Conversation:
        conv = Conversation(
            company_id=company.id,
            conversation_type=ctype,
            context_type=context_type or ConversationContext.COMPANY.value,
            context_id=context_id,
            title=title,
            legacy_thread_id=legacy,
            conversation_metadata=Conversation.default_metadata(),
        )
        db.session.add(conv)
        db.session.flush()
        convs.append(conv)
        return conv

    # Conversations COMPANY (fils chauffeurs 1-1)
    for i in range(N_COMPANY_CONVS):
        add_conv(
            ConversationType.COMPANY.value,
            f"C1A driver-chan {i}",
            legacy=f"company_driver:{90000 + i}",
        )

    # Missions actives + terminales avec bookings reels
    statuses = [("CONFIRMED", N_MISSION_ACTIVE), ("COMPLETED", N_MISSION_TERMINAL)]
    b_idx = 0
    for status, count in statuses:
        for _ in range(count):
            b_idx += 1
            booking = Booking(
                company_id=company.id,
                customer_name=f"C1A Client {b_idx}",
                pickup_location="Rue A 1, Geneve",
                dropoff_location="Rue B 2, Geneve",
                amount=10.0,
                scheduled_time=BASE_TS + timedelta(hours=b_idx),
            )
            db.session.add(booking)
            db.session.flush()
            if status != "CONFIRMED":
                # statut terminal via SQL brut (colonne texte / enum selon schema)
                db.session.execute(
                    db.text("UPDATE booking SET status = :s WHERE id = :i"),
                    {"s": status, "i": booking.id},
                )
            add_conv(
                ConversationType.MISSION.value,
                f"Mission #{booking.id}",
                context_id=booking.id,
                context_type=ConversationContext.MISSION.value,
                legacy=f"mission:{booking.id}",
            )

    for i in range(N_GROUP):
        add_conv(ConversationType.GROUP.value, f"C1A groupe {i}")

    # Participants driver (inbox driver) sur missions + groupes + 5 company
    participant_convs = [
        c for c in convs
        if c.conversation_type in (ConversationType.MISSION.value, ConversationType.GROUP.value)
    ] + convs[:5]
    for conv in participant_convs:
        db.session.add(
            ConversationParticipant(
                conversation_id=conv.id,
                user_id=drv_user.id,
                participant_role="DRIVER",
            )
        )

    # Messages : 50/conv, alternance company/driver, timestamps fixes
    msg_ids_by_conv: dict[int, list[int]] = {}
    for ci, conv in enumerate(convs):
        ids: list[int] = []
        for mi in range(MSGS_PER_CONV):
            from_company = mi % 2 == 0
            msg = Message(
                company_id=company.id,
                conversation_id=conv.id,
                sender_id=owner.id if from_company else drv_user.id,
                sender_role=SenderRole.COMPANY if from_company else SenderRole.DRIVER,
                content=f"c1a {ci}:{mi}",
                timestamp=BASE_TS + timedelta(minutes=ci * 100 + mi),
                thread_id=conv.legacy_thread_id,
            )
            db.session.add(msg)
            db.session.flush()
            ids.append(msg.id)
        msg_ids_by_conv[conv.id] = ids

    # Lectures : le user company a lu les 30 premiers messages d'une conv sur deux
    for ci, conv in enumerate(convs):
        if ci % 2 == 0:
            for mid in msg_ids_by_conv[conv.id][:READ_FIRST]:
                db.session.add(MessageRead(user_id=owner.id, message_id=mid))

    db.session.commit()
    total_msgs = sum(len(v) for v in msg_ids_by_conv.values())
    print(
        f"seed: OK company_id={company.id} convs={len(convs)} msgs={total_msgs} "
        f"reads={sum(1 for i, _ in enumerate(convs) if i % 2 == 0) * READ_FIRST}"
    )


class SqlCounter:
    """Compte les requetes SQL + duree cumulee sur l'engine (usage script local)."""

    def __init__(self) -> None:
        self.count = 0
        self.sql_ms = 0.0
        self._starts: dict[int, float] = {}

    def before(self, *args: object) -> None:
        # Signature imposee par l'event API SQLAlchemy ; args[1] = cursor.
        self.count += 1
        self._starts[id(args[1])] = perf_counter()

    def after(self, *args: object) -> None:
        t0 = self._starts.pop(id(args[1]), None)
        if t0 is not None:
            self.sql_ms += (perf_counter() - t0) * 1000.0


def measure(label: str) -> None:
    from sqlalchemy import event

    from ext import db
    from models import Company, Driver, User
    from services.messaging.conversation_service import ConversationService

    owner = User.query.filter_by(email=SEED_EMAIL_COMPANY).first()
    company = Company.query.filter_by(name=SEED_COMPANY_NAME).first()
    driver = (
        Driver.query.filter_by(company_id=company.id).first() if company else None
    )
    if not owner or not company or not driver:
        raise SystemExit("seed manquant - lancer --seed d'abord")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    engine = db.engine
    results = []

    scopes = [
        ("company_inbox", lambda: ConversationService.build_company_inbox(owner)),
        ("company_hub_threads", lambda: ConversationService.hub_threads_for_company(owner)),
        ("driver_inbox", lambda: ConversationService.build_driver_inbox(driver)),
    ]
    for scope, fn in scopes:
        # Warmup identity map neuve : session propre par scope
        db.session.expire_all()
        counter = SqlCounter()
        event.listen(engine, "before_cursor_execute", counter.before)
        event.listen(engine, "after_cursor_execute", counter.after)
        t0 = perf_counter()
        try:
            payload = fn()
        finally:
            handler_ms = (perf_counter() - t0) * 1000.0
            event.remove(engine, "before_cursor_execute", counter.before)
            event.remove(engine, "after_cursor_execute", counter.after)

        if isinstance(payload, dict):
            threads_n = len(payload.get("threads") or [])
            unread_total = payload.get("unread_total")
        else:
            threads_n = len(payload)
            unread_total = None
        metrics = {
            "label": label,
            "scope": scope,
            "query_count": counter.count,
            "sql_ms": round(counter.sql_ms, 1),
            "handler_ms": round(handler_ms, 1),
            "threads": threads_n,
            "unread_total": unread_total,
        }
        results.append(metrics)
        out = OUT_DIR / f"inbox_{label}_{scope}.json"
        out.write_text(
            json.dumps(payload, default=str, sort_keys=True, indent=1),
            encoding="utf-8",
        )
        print(json.dumps(metrics, ensure_ascii=False))

    summary = OUT_DIR / f"metrics_{label}.json"
    summary.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"-> {summary}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", action="store_true")
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--label", default="run")
    args = parser.parse_args()

    app = _get_app()
    with app.app_context():
        if args.seed:
            seed()
        if args.measure:
            measure(args.label)


if __name__ == "__main__":
    main()
