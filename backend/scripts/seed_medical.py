# backend/scripts/seed_medical.py
import os
import sys
from typing import Any, Dict, List, cast

from flask import Flask

from models import MedicalEstablishment, MedicalService, db

# Ajuste le path pour "import app, models" quand lancé directement
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

def get_app() -> Flask:
    """Récupère l'app Flask (factory ou app globale)."""
    # Essayez d'abord la factory create_app()
    try:
        from app import create_app  # type: ignore[attr-defined]
        return cast(Flask, create_app())
    except Exception:
        # Fallback: un objet `app` global dans app.py
        try:
            from app import app as flask_app  # type: ignore[attr-defined]
            return cast(Flask, flask_app)
        except Exception as e:
            raise RuntimeError("Impossible de charger l'application Flask") from e


def _assign_if_present(obj: Any, **fields: Any) -> None:
    """
    Assigne dynamiquement des champs seulement s'ils existent sur le modèle.
    Évite les erreurs si la migration n'a pas encore ajouté une colonne.
    """
    for k, v in fields.items():
        if hasattr(obj, k):
            setattr(obj, k, v)


def upsert_estab(
    name: str,
    display_name: str,
    address: str,
    lat: float,
    lon: float,
    est_type: str = "hospital",
    aliases: str = "",
) -> MedicalEstablishment:
    """
    Crée ou met à jour un établissement.
    (On évite d'appeler le constructeur SQLAlchemy avec des kwargs non typés
    pour ne pas déclencher Pylance `No parameter named "..."`.)
    """
    e: MedicalEstablishment | None = MedicalEstablishment.query.filter_by(name=name).first()
    if not e:
        e = MedicalEstablishment()  # type: ignore[call-arg]
        _assign_if_present(
            e,
            name=name,
            display_name=display_name,
            address=address,
            lat=lat,
            lon=lon,
            type=est_type,
            aliases=aliases,
        )
        db.session.add(e)
    else:
        _assign_if_present(
            e,
            display_name=display_name,
            address=address,
            lat=lat,
            lon=lon,
            type=est_type,
            aliases=aliases,
        )
    db.session.commit()
    return e


def add_services(estab: MedicalEstablishment, items: List[Any]) -> None:
    """
    Ajoute (ou met à jour) des services pour un établissement.

    items peut contenir :
      - des 2-tuples/lists:    (category, name)
      - ou des dicts riches:   {
            "category": "...", "name": "...",
            # champs optionnels si ton modèle les a :
            "slug": "...",
            "address_line": "...", "postcode": "...", "city": "...",
            "building": "...", "floor": "...", "site_note": "...",
            "phone": "...", "email": "...",
            "lat": 46.19, "lon": 6.14, "active": True/False
        }
    """
    created, updated = 0, 0
    for it in items:
        if isinstance(it, (list, tuple)) and len(it) >= 2:
            category, name = it[0], it[1]
            extra: Dict[str, Any] = {}
        elif isinstance(it, dict):
            category = it.get("category") or "Service"
            name = it.get("name")
            if not name:
                continue
            extra = {k: v for k, v in it.items() if k not in ("category", "name")}
        else:
            continue

        svc: MedicalService | None = MedicalService.query.filter_by(
            establishment_id=estab.id, name=name
        ).first()

        if not svc:
            # Création sans kwargs pour éviter les warnings Pylance
            svc = MedicalService()  # type: ignore[call-arg]
            _assign_if_present(
                svc,
                establishment_id=int(estab.id),
                category=category,
                name=name,
                **extra,
            )
            db.session.add(svc)
            created += 1
        else:
            # Mise à jour
            _assign_if_present(
                svc,
                category=category or getattr(svc, "category", None),
                **extra,
            )
            updated += 1

    db.session.commit()
    print(f"   → services: +{created} créés, ~{updated} mis à jour pour {estab.display_name}.")


def main() -> None:
    app = get_app()
    with app.app_context():
        # === HUG ===
        hug = upsert_estab(
            name="HUG",
            display_name="HUG - Hôpitaux Universitaires de Genève",
            address="Rue Gabrielle-Perret-Gentil 4, 1205 Genève",
            lat=46.19226,
            lon=6.14262,
            est_type="hospital",
            aliases="hug;hôpital cantonal;hopital geneve;hopital cantonal geneve",
        )

        # --- Liste “riche” : tu peux mettre ici tous les services détaillés que tu m’as fournis. ---
        # Exemple demandé : Radiologie → Consultation ambulatoire de radiologie, avec Bât/étage & adresse auto.
        hug_services = [
            # Addictologie — PEPS
            {
                "category": "Service",
                "name": "Programme expérimental de prescription de stupéfiants (PEPS)",
                "address_line": "Route des Acacias 3",
                "postcode": "1227",
                "city": "Genève",
            },
            # Addictologie — CAAP (Grand-Pré)
            {
                "category": "Service",
                "name": "Consultation du Centre ambulatoire d'addictologie psychiatrique (CAAP)",
                "address_line": "Rue du Grand-Pré 70C",
                "postcode": "1202",
                "city": "Genève",
            },

            # Allergologie — Consultation ambulatoire
            {
                "category": "Service",
                "name": "Consultation ambulatoire de l'unité d'allergologie",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
                "building": "Bâtiment Jean-Louis Prévost",
            },
            # Immunologie clinique — Consultation ambulatoire
            {
                "category": "Service",
                "name": "Consultation ambulatoire de l'unité d'immunologie clinique",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
                "building": "Bâtiment Jean-Louis Prévost",
            },

            # Anesthésiologie
            {
                "category": "Service",
                "name": "Anesthésiologie",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1211",
                "city": "Genève 4",
            },

            # Angiologie — Consultation ambulatoire
            {
                "category": "Service",
                "name": "Consultation ambulatoire de l'unité d'angiologie",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
                "building": "Bâtiment Prévost",
                "floor": "5e étage",
            },
            # Hémostase — Consultation ambulatoire
            {
                "category": "Service",
                "name": "Consultation ambulatoire de l'unité d'hémostase",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
                "building": "Bâtiment Prévost",
                "floor": "5e étage",
            },

            # Biologie médicale — BATLab
            {
                "category": "Laboratoire",
                "name": "BATLab",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
            },

            # Cardiologie — Consultation ambulatoire
            {
                "category": "Service",
                "name": "Consultation ambulatoire du service de cardiologie",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
                "building": "Bâtiment Prévost",
                "floor": "6e étage",
            },

            # Chirurgie cardiovasculaire — Consultation ambulatoire chirurgie cardiaque pédiatrique
            {
                "category": "Service",
                "name": "Consultation ambulatoire de l'unité de chirurgie cardiaque pédiatrique",
                "address_line": "Rue Willy Donzé 6",
                "postcode": "1205",
                "city": "Genève",
                "building": "Hôpital des enfants",
            },
            # Chirurgie vasculaire périphérique — Consultation ambulatoire
            {
                "category": "Service",
                "name": "Consultation ambulatoire de l'unité de chirurgie vasculaire périphérique",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
            },
            # Chirurgie vasculaire périphérique — Consultation varices
            {
                "category": "Service",
                "name": "Consultation varices de l'unité de chirurgie vasculaire périphérique",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
            },

                # ORL — Oto-neurologie (troubles de l’équilibre, vertiges)
            {
                "category": "Service",
                "name": "Consultation ambulatoire de l'unité d'oto-neurologie (troubles de l’équilibre, vertiges)",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
                "building": "Bât Prévost",
                "floor": "3e étage",
            },
            # ORL — Consultation du service (général)
            {
                "category": "Service",
                "name": "Consultation ambulatoire du service ORL et chirurgie cervico-faciale",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1211",
                "city": "Genève 14",
                "building": "Bât. Prévost",
                "floor": "3e étage",
            },
            # ORL — Otologie
            {
                "category": "Service",
                "name": "Consultation ambulatoire de l'unité d'otologie",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
                "building": "Bât Prévost",
                "floor": "3e étage",
            },
            # ORL — Pédiatrique
            {
                "category": "Service",
                "name": "Consultation ambulatoire ORL pédiatrique",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
                "building": "Bât Prévost",
                "floor": "3e étage",
            },
            # ORL — Phoniatrie
            {
                "category": "Service",
                "name": "Consultation ambulatoire de l'unité de phoniatrie",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
                "building": "Bât Prévost",
                "floor": "3e étage",
            },
            # ORL — Chirurgie cervico-faciale
            {
                "category": "Service",
                "name": "Consultation ambulatoire de l'unité de chirurgie cervico-faciale",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
                "building": "Bât Prévost",
                "floor": "3e étage",
            },
            # ORL — Rhinologie / Olfactologie
            {
                "category": "Service",
                "name": "Consultation ambulatoire de l'unité de rhinologie - olfactologie",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
                "building": "Bât Prévost",
                "floor": "3e étage",
            },

            # Chirurgie orthopédique & traumatologie de l’appareil moteur — Consultation ambulatoire
            {
                "category": "Service",
                "name": "Consultation ambulatoire du service de chirurgie orthopédique et traumatologie de l’appareil moteur",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
                "building": "Bât Prévost",
                "floor": "étage 0",
            },

            # Chirurgie plastique, reconstructive et esthétique — Consultation ambulatoire
            {
                "category": "Service",
                "name": "Consultation ambulatoire de chirurgie plastique, reconstructive et esthétique",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
                "building": "Bât Morier",
                "floor": "3e étage",
            },

            # Chirurgie thoracique et endocrinienne — Consultation ambulatoire (service de chirurgie thoracique)
            {
                "category": "Service",
                "name": "Consultation ambulatoire du service de chirurgie thoracique",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
            },

            # Chirurgie viscérale — Proctologique
            {
                "category": "Service",
                "name": "Consultation ambulatoire du service de chirurgie viscérale (équipe de chirurgie proctologique)",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
            },
            # Chirurgie viscérale — Oesophage-estomac (Oeso-gastrique)
            {
                "category": "Service",
                "name": "Consultation ambulatoire de chirurgie oesophage-estomac du service de chirurgie viscérale",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
            },
            # Chirurgie viscérale — Colorectale
            {
                "category": "Service",
                "name": "Consultation ambulatoire du service de chirurgie viscérale (équipe de chirurgie colorectale)",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
            },
            # Chirurgie viscérale — Hépato-biliaire
            {
                "category": "Service",
                "name": "Consultation ambulatoire du service de chirurgie viscérale (équipe de chirurgie hépato-biliaire)",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
            },
            # Chirurgie viscérale — Bariatrique et obésité
            {
                "category": "Service",
                "name": "Consultation ambulatoire de chirurgie bariatrique et de l’obésité du service de chirurgie viscérale",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
            },
            # Chirurgie viscérale — Générale (sans sous-équipe)
            {
                "category": "Service",
                "name": "Consultation ambulatoire du service de chirurgie viscérale",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
            },

                # Clinique de Crans-Montana – Médecine interne de réhabilitation
            {
                "category": "Service",
                "name": "Clinique de Crans-Montana - Médecine interne de réhabilitation",
                "address_line": "Impasse Clairmont 2",
                "postcode": "3963",
                "city": "Crans-Montana",

            },

            # Cybersanté et télémédecine
            {
                "category": "Service",
                "name": "Service de cybersanté et télémédecine",
                "address_line": "Bd de la Tour 8",
                "postcode": "1205",
                "city": "Genève",
                "floor": "2e étage",
            },

            # Département de réadaptation et gériatrie — UGC (Thônex)
            {
                "category": "Service",
                "name": "Consultation ambulatoire de l'unité de gériatrie communautaire",
                "address_line": "Chemin du Petit Bel-Air 2",
                "postcode": "1226",
                "city": "Thônex",
            },

            # Dermatologie et vénéréologie — Consultation ambulatoire
            {
                "category": "Service",
                "name": "Consultation ambulatoire du service de dermatologie et vénéréologie",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
                "building": "Bât Prévost",
                "floor": "4e étage",
            },

            # Endocrinologie — Consultation ambulatoire (de l'unité d'endocrinologie)
            {
                "category": "Service",
                "name": "Consultation ambulatoire de l'unité d'endocrinologie",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
                "building": "Bât Morier / Bât Klein",
                "floor": "4e étage / 1er étage",
            },

            # Diabétologie — Consultation ambulatoire (de l'unité de diabétologie)
            {
                "category": "Service",
                "name": "Consultation ambulatoire de l'unité de diabétologie",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
                "building": "Bât Morier / Bât Klein",
                "floor": "4e étage / 1er étage",
            },

            # Diabétologie (lipidologie) — Consultation commune diabétologie et cardiologie
            {
                "category": "Service",
                "name": "Consultation ambulatoire de l'unité de diabétologie (lipidologie)",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
            },

                # Département de réadaptation et gériatrie — UGC (Thônex)
            {
                "category": "Service",
                "name": "Consultation ambulatoire de l'unité de gériatrie communautaire",
                "address_line": "Chemin du Petit Bel-Air 2",
                "postcode": "1226",
                "city": "Thônex",
            },

            # Dermatologie et vénéréologie — Consultation ambulatoire
            {
                "category": "Service",
                "name": "Consultation ambulatoire du service de dermatologie et vénéréologie",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
                "building": "Bât Prévost",
                "floor": "4e étage",
            },

            # Endocrinologie — Consultation ambulatoire (de l'unité d'endocrinologie)
            {
                "category": "Service",
                "name": "Consultation ambulatoire de l'unité d'endocrinologie",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
                "building": "Bât Morier / Bât Klein",
                "floor": "4e étage / 1er étage",
            },

            # Diabétologie — Consultation ambulatoire (de l'unité de diabétologie)
            {
                "category": "Service",
                "name": "Consultation ambulatoire de l'unité de diabétologie",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
                "building": "Bât Morier / Bât Klein",
                "floor": "4e étage / 1er étage",
            },

            # Diabétologie (lipidologie) — Consultation commune diabétologie et cardiologie
            {
                "category": "Service",
                "name": "Consultation ambulatoire de l'unité de diabétologie (lipidologie)",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
            },

            {
                "category": "Service",
                "name": "Consultation ambulatoire de l'unité d'endocrinologie",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
                "building": "Bât Morier / Bât Klein",
                "floor": "4e étage / 1er étage",
            },
            # Diabétologie — consultation ambulatoire
            {
                "category": "Service",
                "name": "Consultation ambulatoire de l'unité de diabétologie",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
                "building": "Bât Morier / Bât Klein",
                "floor": "4e étage / 1er étage",
            },
            # Diabétologie (lipidologie) — consultation commune
            {
                "category": "Service",
                "name": "Consultation ambulatoire de l'unité de diabétologie (lipidologie)",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
            },

            # Nutrition — consultation ambulatoire
            {
                "category": "Service",
                "name": "Consultation ambulatoire de l'unité de nutrition",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
            },
            # Gastro-entérologie & hépatologie — consultation ambulatoire
            {
                "category": "Service",
                "name": "Consultation ambulatoire de gastro-entérologie et hépatologie",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
                "building": "Bât Stern",
                "floor": "étage P",
            },

            # Gynécologie — Sénologie chirurgicale
            {
                "category": "Service",
                "name": "Consultation ambulatoire de l'unité de sénologie chirurgicale",
                "address_line": "Boulevard de la Cluse 30",
                "postcode": "1205",
                "city": "Genève",
                "building": "Maternité",
            },
            # Gynécologie — Médecine de la reproduction & endocrinologie gynécologique
            {
                "category": "Service",
                "name": "Consultation ambulatoire de l'unité de médecine de la reproduction et d'endocrinologie gynécologique",
                "address_line": "Boulevard de la Cluse 30",
                "postcode": "1205",
                "city": "Genève",
                "building": "Maternité",
            },
            # Gynécologie — Périnéologie
            {
                "category": "Service",
                "name": "Consultation ambulatoire de l'unité de périnéologie",
                "address_line": "Boulevard de la Cluse 30",
                "postcode": "1205",
                "city": "Genève",
                "building": "Maternité",
            },
            # Gynécologie — Consultation du service (général)
            {
                "category": "Service",
                "name": "Consultation ambulatoire du service de gynécologie",
                "address_line": "Boulevard de la Cluse 30",
                "postcode": "1205",
                "city": "Genève",
                "building": "Maternité",
            },
            # Gynécologie — Onco-gynécologie chirurgicale
            {
                "category": "Service",
                "name": "Consultation ambulatoire de l'unité d'onco-gynécologie chirurgicale",
                "address_line": "Boulevard de la Cluse 30",
                "postcode": "1205",
                "city": "Genève",
                "building": "Maternité",
            },

             # Hématologie — Unité d'hématologie clinique
            {
                "category": "Service",
                "name": "Consultation ambulatoire de l'unité d'hématologie clinique",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
                "building": "Bâtiment Klein",
                "floor": "4e étage",
            },
            # Hématologie — Unité d'hématologie-oncologie
            {
                "category": "Service",
                "name": "Consultation ambulatoire de l'unité d'hématologie-oncologie",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
                "building": "Bâtiment Klein",
                "floor": "4e étage",
            },

            # Immunologie & allergologie — Unité d'allergologie
            {
                "category": "Service",
                "name": "Consultation ambulatoire de l'unité d'allergologie",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
                "building": "Bâtiment Jean-Louis Prévost",
            },
            # Immunologie & allergologie — Unité d'immunologie clinique
            {
                "category": "Service",
                "name": "Consultation ambulatoire de l'unité d'immunologie clinique",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
                "building": "Bâtiment Jean-Louis Prévost",
            },

                # Laboratoire d'hémostase
            {
                "category": "Laboratoire",
                "name": "Laboratoire d'hémostase",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
                "building": "Bâtiment des laboratoires (BATLab)",
            },

            # Laboratoire de bactériologie
            {
                "category": "Laboratoire",
                "name": "Laboratoire de bactériologie",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
            },

            # Laboratoire de virologie
            {
                "category": "Laboratoire",
                "name": "Laboratoire de virologie",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1211",
                "city": "Genève 14",
                "building": "Bâtiment des laboratoires (BATLab)",
            },

            # Lipodystrophie et troubles métaboliques (Groupe)
            {
                "category": "Groupe",
                "name": "Lipodystrophie et troubles métaboliques",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
                "building": "Bâtiment de liaison",
                "floor": "2e étage",
            },

            # Maladies infectieuses, VIH / Sida — Consultation ambulatoire
            {
                "category": "Service",
                "name": "Consultation ambulatoire du service des maladies infectieuses",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
                "floor": "2e étage",
            },

            # Maladies osseuses — Adresse générale (Boulevard de la Cluse)
            {
                "category": "Service",
                "name": "Maladies osseuses",
                "address_line": "Boulevard de la Cluse 30",
                "postcode": "1205",
                "city": "Genève",
            },

            # Maladies osseuses — Consultation ambulatoire (Prévost 8e)
            {
                "category": "Service",
                "name": "Consultation ambulatoire du service des maladies osseuses",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
                "building": "Bât Prévost",
                "floor": "8e étage",
            },

            # Santé asile et réfugiés BIS (USAR BIS)
            {
                "category": "Service",
                "name": "Consultation ambulatoire - Santé asile et réfugiés BIS",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1211",
                "city": "Genève 14",
            },

            # Consultation Post-COVID
            {
                "category": "Service",
                "name": "Consultation Post-COVID",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
                "building": "Bâtiment Morier",
                "floor": "2e étage",
            },

            # Unité interdisciplinaire de médecine et de prévention de la violence (UIMPV)
            {
                "category": "Service",
                "name": "Consultation ambulatoire - Violence",
                "address_line": "Boulevard de la Cluse 75",
                "postcode": "1205",
                "city": "Genève",
            },

            # Médecine de premiers recours — Site Cluse Roseraie
            {
                "category": "Service",
                "name": "Consultation ambulatoire de médecine de premiers recours (Site Cluse Roseraie)",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1211",
                "city": "Genève 14",
            },
            # Médecine de premiers recours — Site Nations (Centre Archimed)
            {
                "category": "Service",
                "name": "Consultation ambulatoire de médecine de premiers recours (Site Nations)",
                "address_line": "La Voie-Creuse 16",
                "postcode": "1202",
                "city": "Genève",
                "building": "Centre Archimed",
            },

            # Consultation ambulatoire de l'hypertension
            {
                "category": "Service",
                "name": "Consultation ambulatoire de l'hypertension",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1211",
                "city": "Genève",
            },

            # Santé asile et réfugiés (USAR)
            {
                "category": "Service",
                "name": "Consultation ambulatoire - Santé asile et réfugiés (USAR)",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1211",
                "city": "Genève",
            },

            # Unité d'urgences ambulatoire (UUA)
            {
                "category": "Service",
                "name": "Consultation de l'unité d'urgences ambulatoire",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1211",
                "city": "Genève 14",
            },

            # Santé sexuelle et planning familial
            {
                "category": "Service",
                "name": "Consultation ambulatoire de santé sexuelle",
                "address_line": "Boulevard de la Cluse 47",
                "postcode": "1205",
                "city": "Genève",
                "floor": "4e étage",
            },

            # UMSCOM — Unité de médecine et soins dans la communauté
            {
                "category": "Service",
                "name": "Unité de médecine et soins dans la communauté (UMSCOM)",
                "address_line": "Rue Hugo-de-Senger 2-4",
                "postcode": "1205",
                "city": "Genève",
            },

            # UDMPR — Tabac, alcool et autres substances
            {
                "category": "Service",
                "name": "Consultation ambulatoire tabac, alcool et autres substances",
                "address_line": "Boulevard de la Cluse 75",
                "postcode": "1205",
                "city": "Genève",
            },

            # Médecine génétique — Consultation (Morier 3e, local 8D-3-856)
            {
                "category": "Service",
                "name": "Consultation ambulatoire du service de médecine génétique",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1211",
                "city": "Genève 14",
                "building": "Bâtiment Morier",
                "floor": "3e étage",
                "site_note": "Local 8D-3-856",
            },
            # Médecine génétique — Consultation (Morier 3e, bureau 8D-3-857)
            {
                "category": "Service",
                "name": "Consultation ambulatoire du service de médecine génétique",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1211",  # adresse générique indiquée dans ta source
                "city": "Genève 14",
                "building": "Bâtiment Morier",
                "floor": "3e étage",
                "site_note": "Bureau 8D-3-857",
            },

            # Médecine interne générale — adresse générale
            {
                "category": "Service",
                "name": "Médecine interne générale",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
            },

            # Médecine nucléaire et imagerie moléculaire — J.-L. Prévost 1er
            {
                "category": "Service",
                "name": "Médecine nucléaire et imagerie moléculaire",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
                "building": "Bâtiment Jean-Louis Prévost",
                "floor": "1er étage",
            },

            # Médecine palliative — Hôpital de Bellerive
            {
                "category": "Service",
                "name": "Consultation ambulatoire du service de médecine palliative",
                "address_line": "Chemin de la Savonnière 11",
                "postcode": "1245",
                "city": "Collonge-Bellerive",
                "building": "Hôpital de Bellerive",
            },

            # Médecine pénitentiaire — Petit-Bel-Air (Thônex)
            {
                "category": "Service",
                "name": "Médecine pénitentiaire",
                "address_line": "Chemin du Petit-Bel-Air 2",
                "postcode": "1226",
                "city": "Thônex",
            },

            # Médecine tropicale et humanitaire — RGP-G 6, 3e étage
            {
                "category": "Service",
                "name": "Consultation ambulatoire du service de médecine tropicale et humanitaire",
                "address_line": "Rue Gabrielle-Perret-Gentil 6",
                "postcode": "1205",
                "city": "Genève",
                "floor": "3e étage",
            },


            # Néphrologie et hypertension — Unité d'hypertension artérielle
            {
                "category": "Service",
                "name": "Consultation ambulatoire de l'unité d'hypertension artérielle",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
            },
            # Néphrologie — Consultation (Klein 3e)
            {
                "category": "Service",
                "name": "Consultation ambulatoire du service de néphrologie",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
                "building": "Bâtiment Klein",
                "floor": "3e étage",
            },

            # Neurochirurgie — réception neurologie/neurochirurgie
            {
                "category": "Service",
                "name": "Service de neurochirurgie — Réception de neurologie/neurochirurgie",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
                "building": "Bâtiment Prévost",
                "floor": "2e étage",
            },

            # Neurologie — SLA
            {
                "category": "Service",
                "name": "Consultation de la sclérose latérale amyotrophique (SLA)",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1211",
                "city": "Genève 14",
            },
            # Neurologie — consultation générale
            {
                "category": "Service",
                "name": "Consultation ambulatoire du service de neurologie",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
                "building": "Bât Prévost",
                "floor": "2e étage",
            },
            # Neurologie — Parkinson / troubles du mouvement
            {
                "category": "Service",
                "name": "Consultation ambulatoire de l'unité des troubles du mouvement - Parkinson",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
                "building": "Bât Prévost",
                "floor": "2e étage",
            },
            # Neurologie — Neuropsychologie
            {
                "category": "Service",
                "name": "Consultation ambulatoire de l'unité de neuropsychologie",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
                "building": "Bât Prévost",
                "floor": "2e étage",
            },
            # Neurologie — Neuroimmunologie & SEP
            {
                "category": "Service",
                "name": "Consultations ambulatoires de l'unité de neuroimmunologie et sclérose en plaques",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
                "building": "Bât Prévost",
                "floor": "2e étage",
            },
            # Neurologie — EEG / Épilepsie
            {
                "category": "Service",
                "name": "Consultation ambulatoire de l'unité d'électro-encéphalographie (EEG) et d'exploration de l'épilepsie",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
                "building": "Bât Prévost",
                "floor": "2e étage",
            },
            # Neurologie — Vasculaire
            {
                "category": "Service",
                "name": "Consultation ambulatoire de l'unité de neurologie vasculaire",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
                "building": "Bât Prévost",
                "floor": "2e étage",
            },
            # Neurologie — ENMG / affections neuromusculaires
            {
                "category": "Service",
                "name": "Consultation ambulatoire de l'unité d'électroneuromyographie et des affections neuromusculaires (ENMG)",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
            },

            # Neuroradiologie diagnostique et interventionnelle
            {
                "category": "Service",
                "name": "Service de neuroradiologie diagnostique et interventionnelle",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
            },

            # Neurorééducation — consultation ambulatoire
            {
                "category": "Service",
                "name": "Consultation ambulatoire du service de neuro-rééducation",
                "address_line": "Avenue Beau-Séjour 26",
                "postcode": "1205",
                "city": "Genève",
            },

            # Obstétrique — consultation du service
            {
                "category": "Service",
                "name": "Consultation ambulatoire du service d'obstétrique",
                "address_line": "Boulevard de la Cluse 30",
                "postcode": "1205",
                "city": "Genève",
                "building": "Maternité",
            },
            # Obstétrique — haut risque
            {
                "category": "Service",
                "name": "Consultation ambulatoire de l'unité d'obstétrique à haut risque",
                "address_line": "Boulevard de la Cluse 30",
                "postcode": "1205",
                "city": "Genève",
                "building": "Maternité",
            },
            # Obstétrique — médecine fœtale et échographie
            {
                "category": "Service",
                "name": "Consultation ambulatoire de l'unité de médecine fœtale et d'échographie",
                "address_line": "Boulevard de la Cluse 30",
                "postcode": "1205",
                "city": "Genève",
                "building": "Maternité",
            },

            # Oncologie — consultation du service (Prévost 5e)
            {
                "category": "Service",
                "name": "Consultation ambulatoire du Service d'oncologie",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
                "building": "Bâtiment Prévost",
                "floor": "5e étage",
            },
            # Oncologie — onco-gynécologie médicale (Centre du sein, Maternité 1er)
            {
                "category": "Service",
                "name": "Consultation ambulatoire de l'unité d'onco-gynécologie médicale",
                "address_line": "Boulevard de la Cluse 30",
                "postcode": "1205",
                "city": "Genève",
                "building": "Maternité",
                "floor": "1er étage",
            },
            # Oncologie — oncogénétique (au sein de la Maternité)
            {
                "category": "Service",
                "name": "Consultation ambulatoire de l'unité d'oncogénétique",
                "address_line": "Boulevard de la Cluse 30",
                "postcode": "1205",
                "city": "Genève",
                "building": "Maternité",
            },

            # Oncologie de précision
            {
                "category": "Service",
                "name": "Service d'oncologie de précision",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
            },

            # Ophtalmologie
            {
                "category": "Service",
                "name": "Consultation ambulatoire du service d'ophtalmologie",
                "address_line": "Rue Alcide-Jentzer 22",
                "postcode": "1205",
                "city": "Genève",
            },

            # ORL — Oto-neurologie
            {
                "category": "Service",
                "name": "Consultation ambulatoire de l'unité d'oto-neurologie (troubles de l’équilibre, vertiges)",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
                "building": "Bât Prévost",
                "floor": "3e étage",
            },
            # ORL — Consultation du service
            {
                "category": "Service",
                "name": "Consultation ambulatoire du service ORL et chirurgie cervico-faciale",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1211",
                "city": "Genève 14",
                "building": "Bât Prévost",
                "floor": "3e étage",
            },
            # ORL — Otologie
            {
                "category": "Service",
                "name": "Consultation ambulatoire de l'unité d'otologie",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
                "building": "Bât Prévost",
                "floor": "3e étage",
            },
            # ORL — Pédiatrique
            {
                "category": "Service",
                "name": "Consultation ambulatoire ORL pédiatrique",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
                "building": "Bât Prévost",
                "floor": "3e étage",
            },
            # ORL — Phoniatrie
            {
                "category": "Service",
                "name": "Consultation ambulatoire de l'unité de phoniatrie",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
                "building": "Bât Prévost",
                "floor": "3e étage",
            },
            # ORL — Chirurgie cervico-faciale
            {
                "category": "Service",
                "name": "Consultation ambulatoire de l'unité de chirurgie cervico-faciale",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
                "building": "Bât Prévost",
                "floor": "3e étage",
            },
            # ORL — Rhinologie / Olfactologie
            {
                "category": "Service",
                "name": "Consultation ambulatoire de l'unité de rhinologie - olfactologie",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
                "building": "Bât Prévost",
                "floor": "3e étage",
            },

            # Pathologie clinique — C.M.U.
            {
                "category": "Service",
                "name": "Pathologie clinique — C.M.U.",
                "address_line": "Rue Michel-Servet 1",
                "postcode": "1206",
                "city": "Genève",
                "building": "Bât. E-F",
                "floor": "5e étage",
            },

            # Pharmacologie & toxicologie cliniques — CIT & Pharmacovigilance
            {
                "category": "Service",
                "name": "Centre d'informations thérapeutiques et de pharmacovigilance (consultation ambulatoire)",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
            },
            # Pharmacologie & toxicologie cliniques — Centre multidisciplinaire de la douleur
            {
                "category": "Service",
                "name": "Centre multidisciplinaire de la douleur : consultation ambulatoire",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1211",
                "city": "Genève 14",
                "building": "Bât Morier",
                "floor": "niveau 0",
            },

            # Pneumologie — consultation du service
            {
                "category": "Service",
                "name": "Consultation ambulatoire du service de pneumologie",
                "address_line": "Rue Gabrielle-Perret-Gentil 6",
                "postcode": "1205",
                "city": "Genève",
                "building": "Bât Morier",
                "floor": "1er étage",
            },
            # Pneumologie — transplantation pulmonaire
            {
                "category": "Service",
                "name": "Consultation ambulatoire de transplantation pulmonaire",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
                "building": "Bât Morier",
                "floor": "1er étage",
            },
            # Pneumologie — mycobactéries non tuberculeuses
            {
                "category": "Service",
                "name": "Consultation ambulatoire de mycobactéries non tuberculeuses",
                "address_line": "Rue Gabrielle-Perret-Gentil 6",
                "postcode": "1205",
                "city": "Genève",
                "building": "Bât Morier",
                "floor": "1er étage",
            },
            # Pneumologie — interventionnelle / nodules pulmonaires
            {
                "category": "Service",
                "name": "Pneumologie interventionnelle / nodules pulmonaires",
                "address_line": "Rue Gabrielle-Perret-Gentil 6",
                "postcode": "1205",
                "city": "Genève",
                "building": "Bât Morier",
                "floor": "1er étage",
            },
            # Pneumologie — Centre de médecine du sommeil (Belle-Idée)
            {
                "category": "Service",
                "name": "Consultation ambulatoire du Centre de médecine du sommeil",
                "address_line": "Chemin du Petit-Bel-Air 2",
                "postcode": "1224",
                "city": "Thônex",
            },
            # Pneumologie — mucoviscidose adulte
            {
                "category": "Service",
                "name": "Consultation de mucoviscidose adulte",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
                "building": "Bât Morier",
                "floor": "1er étage",
            },

             # Psychiatrie (spécialités psychiatriques) — troubles anxieux
            {
                "category": "Service",
                "name": "Consultation ambulatoire du programme troubles anxieux",
                "address_line": "Rue de Lausanne 20bis",
                "postcode": "1201",
                "city": "Genève",
            },
            # Psychiatrie (spécialités) — Sexologie
            {
                "category": "Service",
                "name": "Consultation de Sexologie",
                "address_line": "Rue de Lausanne 20bis",
                "postcode": "1201",
                "city": "Genève",
            },
            # Psychiatrie (spécialités) — troubles de l’humeur
            {
                "category": "Service",
                "name": "Programme des troubles de l'humeur",
                "address_line": "Rue de Lausanne 20bis",
                "postcode": "1201",
                "city": "Genève",
            },
            # Psychiatrie (spécialités) — régulation émotionnelle
            {
                "category": "Service",
                "name": "Consultation trouble de la régulation émotionnelle",
                "address_line": "Rue de Lausanne 20bis",
                "postcode": "1201",
                "city": "Genève",
            },
            # Psychiatrie (spécialités) — familles & couples
            {
                "category": "Service",
                "name": "Consultation psychothérapeutique pour familles et couples",
                "address_line": "Chemin du Petit-Bel-Air 2",
                "postcode": "1226",
                "city": "Thônex",
            },
            # Psychiatrie (spécialités) — développement mental
            {
                "category": "Service",
                "name": "Consultation ambulatoire de l'unité de psychiatrie du développement mental",
                "address_line": "Chemin du Petit-Bel-Air 2",
                "postcode": "1225",
                "city": "Chêne-Bourg",
            },

            # Psychiatrie adulte — CAPPI Eaux-Vives
            {
                "category": "Service",
                "name": "CAPPI Eaux-Vives (Centre ambulatoire de psychiatrie et psychothérapie intégrées)",
                "address_line": "Rue du 31-Décembre 8-9",
                "postcode": "1207",
                "city": "Genève",
            },
            # Psychiatrie adulte — CAPPI Jonction
            {
                "category": "Service",
                "name": "CAPPI Jonction (Centre ambulatoire de psychiatrie et psychothérapie intégrées)",
                "address_line": "Rue des Bains 35",
                "postcode": "1205",
                "city": "Genève",
            },
            # Psychiatrie adulte — CAPPI Servette
            {
                "category": "Service",
                "name": "CAPPI Servette (Centre ambulatoire de psychiatrie et psychothérapie intégrées)",
                "address_line": "Rue de Lyon 89-91",
                "postcode": "1203",
                "city": "Genève",
            },
            # Psychiatrie adulte — JADE jeunes adultes
            {
                "category": "Service",
                "name": "JADE (Programme ambulatoire Jeunes adultes avec troubles débutants)",
                "address_line": "Rue du Grand-Pré 70A",
                "postcode": "1202",
                "city": "Genève",
            },

            # Psychiatrie de liaison & crise — psychotraumatologie
            {
                "category": "Service",
                "name": "Consultation de psychotraumatologie",
                "address_line": "Boulevard de la Cluse 51",
                "postcode": "1205",
                "city": "Genève",
            },
            # Psychiatrie de liaison & crise — consultation ambulatoire
            {
                "category": "Service",
                "name": "Consultation ambulatoire du service de psychiatrie de liaison et d'intervention de crise",
                "address_line": "Boulevard de la Cluse 51",
                "postcode": "1205",
                "city": "Genève",
            },
            # Psychiatrie de liaison & crise — troubles du comportement alimentaire
            {
                "category": "Service",
                "name": "Programme ambulatoire pour les troubles du comportement alimentaire",
                "address_line": "Rue des Pitons 15",
                "postcode": "1205",
                "city": "Genève",
            },

            # Psychiatrie gériatrique
            {
                "category": "Service",
                "name": "Psychiatrie gériatrique — Secrétariat du médecin-chef de service",
                "address_line": "Chemin du Petit-Bel-Air 2",
                "postcode": "1226",
                "city": "Thônex",
            },

            # Qualité des soins
            {
                "category": "Service",
                "name": "Service qualité des soins",
                "address_line": "Avenue de la Tour 8",
                "postcode": "1205",
                "city": "Genève",
            },

            # Radio-oncologie
            {
                "category": "Service",
                "name": "Consultation ambulatoire du service de radio-oncologie",
                "address_line": "Avenue de la Roseraie 53",
                "postcode": "1205",
                "city": "Genève",
            },

            # Radiologie — consultation ambulatoire
            {
                "category": "Service",
                "name": "Consultation ambulatoire de radiologie",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
                "building": "Bât Prévost",
                "floor": "étage P",
            },

            # Rhumatologie — consultation multidisciplinaire du dos
            {
                "category": "Service",
                "name": "Consultation multidisciplinaire du dos",
                "address_line": "Avenue de Beau-Séjour 26",
                "postcode": "1205",
                "city": "Genève",
            },
            # Rhumatologie — consultation du service
            {
                "category": "Service",
                "name": "Consultation ambulatoire du service de rhumatologie",
                "address_line": "Avenue Beau-Séjour 26",
                "postcode": "1206",
                "city": "Genève",
            },

            # Sciences de l’information médicale
            {
                "category": "Service",
                "name": "Sciences de l’information médicale",
                "address_line": "Bd de la Tour 8",
                "postcode": "1205",
                "city": "Genève",
                "floor": "2e étage",
            },

            # Service Biomédical et Equipements
            {
                "category": "Service",
                "name": "Service Biomédical et Equipements",
                "address_line": "Boulevard de la Cluse 77",
                "postcode": "1205",
                "city": "Genève",
            },

            # Service de médecine de laboratoire
            {
                "category": "Service",
                "name": "Service de médecine de laboratoire",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
            },

            # Service des soins intensifs
            {
                "category": "Service",
                "name": "Service des soins intensifs",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
            },

            # Service prévention et contrôle de l'infection — Secrétariat
            {
                "category": "Service",
                "name": "Service prévention et contrôle de l'infection — Secrétariat",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
                "building": "Bâtiment d’appui (Jean-Louis Prévost)",
                "floor": "9e étage",
            },

             # Spécialités psychiatriques — Rue de Lausanne 20bis
            {
                "category": "Service",
                "name": "Consultation ambulatoire du programme troubles anxieux",
                "address_line": "Rue de Lausanne 20bis",
                "postcode": "1201",
                "city": "Genève",
            },
            {
                "category": "Service",
                "name": "Consultation de Sexologie",
                "address_line": "Rue de Lausanne 20bis",
                "postcode": "1201",
                "city": "Genève",
            },
            {
                "category": "Service",
                "name": "Programme des troubles de l'humeur",
                "address_line": "Rue de Lausanne 20bis",
                "postcode": "1201",
                "city": "Genève",
            },
            {
                "category": "Service",
                "name": "Consultation trouble de la régulation émotionnelle",
                "address_line": "Rue de Lausanne 20bis",
                "postcode": "1201",
                "city": "Genève",
            },
            {
                "category": "Service",
                "name": "Consultation psychothérapeutique pour familles et couples",
                "address_line": "Chemin du Petit-Bel-Air 2",
                "postcode": "1226",
                "city": "Thônex",
            },
            {
                "category": "Service",
                "name": "Consultation ambulatoire de l'unité de psychiatrie du développement mental",
                "address_line": "Chemin du Petit-Bel-Air 2",
                "postcode": "1225",
                "city": "Chêne-Bourg",
            },

            # Transplantation — adresse générale
            {
                "category": "Service",
                "name": "Transplantation",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
            },

            # Unité d'éducation thérapeutique du patient — Obésité & diabète (Villa Soleillane)
            {
                "category": "Service",
                "name": "Consultation ambulatoire obésité et diabète",
                "address_line": "Chemin Venel 7",
                "postcode": "1206",
                "city": "Genève",
            },

            # Urgences adultes
            {
                "category": "Service",
                "name": "Urgences adultes",
                "address_line": "Rue Gabrielle-Perret-Gentil 2",
                "postcode": "1205",
                "city": "Genève",
            },

            # Urgences de gynécologie et d’obstétrique (entrée Maternité)
            {
                "category": "Service",
                "name": "Urgences de gynécologie et d’obstétrique",
                "address_line": "Boulevard de la Cluse 30",
                "postcode": "1205",
                "city": "Genève",
                "building": "Maternité",
            },

            # Urgences gériatriques non vitales (pas d’adresse précisée)
            ("Service", "Urgences gériatriques non vitales"),

            # Urgences psychiatriques (pas d’adresse précisée)
            ("Service", "Urgences psychiatriques"),

            # Urologie — consultation ambulatoire du service
            {
                "category": "Service",
                "name": "Consultation ambulatoire du service d'urologie",
                "address_line": "Rue Gabrielle-Perret-Gentil 4",
                "postcode": "1205",
                "city": "Genève",
            },

        ]

        add_services(hug, hug_services)

        # === Clinique La Colline ===
        colline = upsert_estab(
            name="Clinique La Colline",
            display_name="Clinique La Colline",
            address="Av. de Beau-Séjour 6, 1206 Genève",
            lat=46.19878, lon=6.15838,
            est_type="clinic",
            aliases="la colline;clinique colline"
        )
        add_services(colline, [
            ("Service", "Urgences"),
            ("Service", "Orthopédie"),
            # tu peux enrichir avec des dicts riches si tu as bâtiment/étage
        ])

        # === Clinique des Grangettes ===
        grangettes = upsert_estab(
            name="Clinique des Grangettes",
            display_name="Clinique des Grangettes",
            address="Ch. des Grangettes 7, 1224 Chêne-Bougeries",
            lat=46.20986, lon=6.18118,
            est_type="clinic",
            aliases="grangettes;clinique grangettes"
        )
        add_services(grangettes, [
            ("Service", "Maternité"),
            ("Service", "Pédiatrie"),
        ])

        # === Hôpital de La Tour ===
        latour = upsert_estab(
            name="Hôpital de La Tour",
            display_name="Hôpital de La Tour",
            address="Av. J.-D. Maillard 3, 1217 Meyrin",
            lat=46.22996, lon=6.07363,
            est_type="hospital",
            aliases="la tour;hopital la tour;hôpital la tour"
        )
        add_services(latour, [
            ("Service", "Urgences"),
            ("Service", "Radiologie"),
        ])

        print("✅ Seed établissements + services OK.")

if __name__ == "__main__":
    main()
