"""Corrige les fiches inversées Amandine HAUSER / Hospice général.

Cas cible (référence métier LIRIE) :

  BillingParty.display_name = "Hospice général"
  BillingParty.type         = "other"
  ClientBillingParty.contact_name  = "Amandine HAUSER"
  ClientBillingParty.role          = "Coordinatrice"

Usage (via Docker) ::

  docker compose exec backend python scripts/fix_hospice_general_amandine_billing.py --dry-run
  docker compose exec backend python scripts/fix_hospice_general_amandine_billing.py --apply

Ne jamais déduire une curatelle : type reste ``other``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.billing_party import BillingParty, ClientBillingParty
from models.enums import BillingPartyType

# Patterns qui identifient le cas Amandine inversé (personne = payeur, fonction = contact).
_AMANDINE_NAME = "amandine hauser"
_COORD_FUNCTION = "coordinatrice"
_HOSPICE = "hospice"


def _norm(value: str | None) -> str:
    return (value or "").strip().casefold()


def _is_inverted_amandine_party(bp: BillingParty) -> bool:
    name = _norm(bp.display_name)
    return _AMANDINE_NAME in name and bp.type == BillingPartyType.OTHER


def _is_inverted_contact(link: ClientBillingParty) -> bool:
    contact = _norm(link.contact_name)
    return _COORD_FUNCTION in contact and _HOSPICE in contact


def find_candidates() -> list[tuple[BillingParty, list[ClientBillingParty]]]:
    parties = BillingParty.query.filter(
        BillingParty.type == BillingPartyType.OTHER
    ).all()
    out: list[tuple[BillingParty, list[ClientBillingParty]]] = []
    for bp in parties:
        if not _is_inverted_amandine_party(bp):
            continue
        links = ClientBillingParty.query.filter_by(billing_party_id=bp.id).all()
        # Au moins un lien avec la fonction dans contact_name, ou aucun lien.
        if links and not any(_is_inverted_contact(link) for link in links):
            # Payeur = Amandine mais contact déjà autre chose : on corrige quand même
            # le display_name + contact si l'email hospice est présent.
            email = _norm(bp.contact_email)
            if "hospicegeneral" not in email and not any(
                "hospicegeneral" in _norm(link.contact_email) for link in links
            ):
                continue
        out.append((bp, links))
    return out


def apply_fix(
    bp: BillingParty,
    links: list[ClientBillingParty],
    *,
    dry_run: bool,
) -> None:
    old_name = bp.display_name
    print(f"  BillingParty id={bp.id} company_id={bp.company_id}")
    print(f"    display_name : {old_name!r} → 'Hospice général'")
    print(f"    type         : {bp.type.value} (inchangé)")

    if not dry_run:
        bp.display_name = "Hospice général"

    if not links:
        print("    (aucun ClientBillingParty — créer le lien manuellement si besoin)")
        return

    for link in links:
        old_contact = link.contact_name
        old_role = link.role
        new_contact = "Amandine HAUSER"
        new_role = "Coordinatrice"
        # Préserver email/téléphone déjà sur le lien ; sinon remonter depuis le BP.
        new_email = (link.contact_email or bp.contact_email or "").strip() or None
        new_phone = (link.contact_phone or bp.contact_phone or "").strip() or None
        print(f"    Link id={link.id} client_id={link.client_id}")
        print(f"      contact_name : {old_contact!r} → {new_contact!r}")
        print(f"      role         : {old_role!r} → {new_role!r}")
        if new_email:
            print(f"      contact_email: {new_email}")
        if new_phone:
            print(f"      contact_phone: {new_phone}")
        if not dry_run:
            link.contact_name = new_contact
            link.role = new_role
            if new_email and not link.contact_email:
                link.contact_email = new_email
            if new_phone and not link.contact_phone:
                link.contact_phone = new_phone


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true", help="Afficher sans écrire")
    group.add_argument("--apply", action="store_true", help="Appliquer les corrections")
    args = parser.parse_args(argv)

    from app import create_app
    from ext import db

    app = create_app()
    with app.app_context():
        candidates = find_candidates()
        if not candidates:
            print("Aucun candidat Amandine HAUSER inversé trouvé.")
            return 0
        print(f"{len(candidates)} candidat(s) trouvé(s) ({'dry-run' if args.dry_run else 'APPLY'}) :")
        for bp, links in candidates:
            apply_fix(bp, links, dry_run=args.dry_run)
        if args.apply:
            db.session.commit()
            print("Commit OK.")
        else:
            db.session.rollback()
            print("Dry-run terminé (aucune écriture).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
