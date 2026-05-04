from __future__ import annotations

from sqlalchemy import or_

import manage
from ext import db
from models import Booking, Client, Company, User
from models.enums import ClientType, GenderEnum, ManagementMode

NAMES: list[tuple[str, str]] = [
    ("Aline", "Morel"),
    ("Karim", "Haddad"),
    ("Sophie", "Vuille"),
    ("Nicolas", "Bernasconi"),
    ("Lea", "Rochat"),
    ("Omar", "Bensaid"),
    ("Camille", "Fournier"),
    ("Mathis", "Perrin"),
]

DROPOFFS: list[str] = [
    "Hopital cantonal, Rue Gabrielle-Perret-Gentil 4, 1205 Geneve",
    "Clinique de Carouge, Avenue Cardinal-Mermillod 1, 1227 Carouge",
    "Centre de soins de Chene, Route de Chene 100, 1224 Chene-Bougeries",
    "Policlinique des Acacias, Rue des Epinettes 19, 1227 Les Acacias",
]


def _is_demo_placeholder_name(first_name: str | None, last_name: str | None) -> bool:
    first = str(first_name or "").strip().lower()
    last = str(last_name or "").strip().lower()
    return first == "patient" or "demo" in first or "demo" in last


def _is_placeholder_text(value: str | None, prefixes: tuple[str, ...]) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return True
    return any(text.startswith(prefix) for prefix in prefixes)


def run() -> dict[str, int]:
    app = manage.app
    with app.app_context():
        updated_users = 0
        updated_bookings = 0
        updated_clients = 0

        users = User.query.filter(
            or_(
                User.first_name.ilike("Patient%"),
                User.last_name.ilike("Demo %"),
                User.last_name.ilike("Démo %"),
            )
        ).all()
        for idx, user in enumerate(users):
            if not _is_demo_placeholder_name(user.first_name, user.last_name):
                continue
            first_name, last_name = NAMES[idx % len(NAMES)]
            user.first_name = first_name
            user.last_name = last_name
            # Eviter "Civilite: Autre" sur les fiches demo quand non renseigné.
            if getattr(user, "gender", None) in (None, GenderEnum.AUTRE):
                user.gender = GenderEnum.FEMME if idx % 2 == 0 else GenderEnum.HOMME
            updated_users += 1

        bookings = Booking.query.filter(
            or_(
                Booking.customer_name.ilike("Patient Demo %"),
                Booking.pickup_location.ilike("Adresse pickup demo %"),
                Booking.dropoff_location.ilike("HUG Service %"),
                Booking.dropoff_location.ilike("HUG - Service %"),
            )
        ).all()
        for idx, booking in enumerate(bookings):
            client_user = getattr(getattr(booking, "client", None), "user", None)
            if (
                booking.customer_name
                and booking.customer_name.lower().startswith("patient demo")
                and client_user
            ):
                full_name = (
                    f"{client_user.first_name or ''} {client_user.last_name or ''}"
                ).strip()
                if full_name:
                    booking.customer_name = full_name
                    updated_bookings += 1

            if booking.pickup_location and booking.pickup_location.lower().startswith(
                "adresse pickup demo"
            ):
                client = getattr(booking, "client", None)
                candidate_address = (
                    getattr(client, "domicile_address", None)
                    or getattr(client, "billing_address", None)
                    or "Rue de Carouge 58, Geneve"
                )
                booking.pickup_location = candidate_address
                updated_bookings += 1

            if booking.dropoff_location and (
                booking.dropoff_location.lower().startswith("hug service")
                or booking.dropoff_location.lower().startswith("hug - service")
            ):
                booking.dropoff_location = DROPOFFS[idx % len(DROPOFFS)]
                updated_bookings += 1

        demo_company_ids = [
            cid
            for (cid,) in db.session.query(Company.id)
            .join(User, Company.user_id == User.id)
            .filter(
                or_(
                    User.email.ilike("%@demo.lirie.ch"),
                    User.email.ilike("%@demo.local"),
                    User.email.ilike("demo-%@%"),
                )
            )
            .all()
        ]
        if demo_company_ids:
            clients = Client.query.filter(Client.company_id.in_(demo_company_ids)).all()
            for idx, client in enumerate(clients):
                if client.client_type is None:
                    client.client_type = ClientType.TRANSPORT
                    client.management_mode = ManagementMode.MANAGED
                    updated_clients += 1
                if not bool(getattr(client, "is_active", True)):
                    client.is_active = True
                    updated_clients += 1
                user = getattr(client, "user", None)
                user_full_name = ""
                if user:
                    user_full_name = (
                        f"{getattr(user, 'first_name', '') or ''} {getattr(user, 'last_name', '') or ''}"
                    ).strip()

                if (
                    _is_placeholder_text(
                        client.default_billed_to_contact,
                        ("patient demo", "demo ", "patient "),
                    )
                    and user_full_name
                ):
                    client.default_billed_to_contact = user_full_name
                    updated_clients += 1

                if _is_placeholder_text(
                    client.residence_facility, ("residence demo", "etablissement demo")
                ):
                    client.residence_facility = (
                        f"Residence Les Tilleuls {((idx % 4) + 1)}"
                    )
                    updated_clients += 1

                if _is_placeholder_text(client.gp_name, ("dr demo", "dr seed")):
                    last_name = str(getattr(user, "last_name", "") or "").strip()
                    client.gp_name = f"Dr {last_name}" if last_name else "Dr Martin"
                    updated_clients += 1

        db.session.commit()
        return {
            "updated_users": updated_users,
            "updated_bookings": updated_bookings,
            "updated_clients": updated_clients,
            "matched_bookings": len(bookings),
        }


if __name__ == "__main__":
    result = run()
    print(result)
