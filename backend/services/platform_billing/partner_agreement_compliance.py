"""Registre minimal des prestataires et politique de conservation (contrat v1.20)."""

from __future__ import annotations

from typing import Any

from services.platform_billing.partner_agreement_versions import (
    PENALTY_CALCULATION_VERSION,
    RETENTION_POLICY_VERSION,
    SUBPROCESSORS_VERSION,
)

# Prestataires : seuls active=True sont imprimés dans l'annexe C.
# Avant première signature : revue manuelle obligatoire (gate process).
# Google Maps : audit code = Geocoding / Distance Matrix / Directions / Places ;
# paramètres API = adresses ou coordonnées (pas de nom/DOB/notes médicales/tél
# /réf. dossier dans les appels observés). Rôle = responsable distinct (conditions
# Maps EEE), pas « sous-traitant Google Cloud » par défaut.
TECHNICAL_PROVIDERS: list[dict[str, Any]] = [
    {
        "name": "Hetzner",
        "service": "Hébergement infrastructure et stockage",
        "active": True,
        "source": "code",
        "data_categories": [
            "base de données applicative",
            "fichiers et pièces jointes",
            "journaux techniques",
        ],
        "processing_countries": ["Union européenne (localisation sélectionnée)"],
        "transfer_guarantees": (
            "Accord de traitement des données Hetzner ; mesures techniques "
            "et organisationnelles ; localisation européenne sélectionnée par LIRIE"
        ),
        "legal_role": "sous-traitant d'hébergement",
    },
    {
        "name": "Brevo",
        "service": "E-mails transactionnels",
        "active": True,
        "source": "code",
        "data_categories": [
            "adresses e-mail",
            "contenu limité des messages transactionnels",
            "pièces jointes éventuelles (ex. factures)",
        ],
        "processing_countries": ["Union européenne (hébergement principal)"],
        "transfer_guarantees": (
            "DPA Brevo ; hébergement principal dans l'Union européenne ; "
            "sous-traitants ultérieurs selon la liste contractuelle en vigueur"
        ),
        "legal_role": "sous-traitant pour l'envoi d'e-mails",
    },
    {
        "name": "Google Maps Platform",
        "service": "Géocodage, itinéraires, Distance Matrix, Places et cartographie",
        "active": True,
        "source": "code",
        "data_categories": [
            "requêtes cartographiques",
            "adresses ou coordonnées strictement nécessaires",
            "sans nom du patient ni identifiant métier dans les appels API",
        ],
        "processing_countries": [
            "Selon l'infrastructure et les conditions applicables de Google"
        ],
        "transfer_guarantees": (
            "Conditions responsable-à-responsable Google Maps Platform (EEE) "
            "et mécanismes de transfert applicables"
        ),
        "legal_role": (
            "responsable distinct selon les conditions applicables "
            "à Google Maps Platform"
        ),
    },
    {
        "name": "Twilio",
        "service": "SMS transactionnels",
        "active": False,
        "source": "configuration",
        "data_categories": ["numéros de téléphone", "contenu SMS"],
        "processing_countries": ["selon configuration"],
        "transfer_guarantees": "DPA Twilio le cas échéant",
        "legal_role": "sous-traitant SMS (si activé)",
    },
]

# Politique versionnée — formulations honnêtes (pas de faux « 90 j » GPS).
RETENTION_CATEGORIES: list[dict[str, Any]] = [
    {
        "category": "contrats_et_annexes",
        "description": (
            "Contrats signés et annexes : dix (10) ans après la fin du contrat, "
            "sous réserve d'un litige en cours."
        ),
    },
    {
        "category": "factures_comptables",
        "description": (
            "Factures et pièces comptables : dix (10) ans à compter de la fin "
            "de l'exercice concerné."
        ),
    },
    {
        "category": "donnees_de_mission",
        "description": (
            "Données de mission et messages opérationnels : durée nécessaire à "
            "la prestation, à la facturation et aux prétentions légales, puis "
            "suppression ou anonymisation."
        ),
    },
    {
        "category": "informations_medicales_operationnelles",
        "description": (
            "Informations médicales ou de mobilité opérationnelles : conservation "
            "limitée à la nécessité de la mission et des obligations applicables, "
            "puis restriction ou suppression."
        ),
    },
    {
        "category": "gps_brut",
        "description": (
            "Données de géolocalisation brutes : conservées pendant la durée "
            "nécessaire aux finalités opérationnelles, de sécurité et de preuve, "
            "puis supprimées ou anonymisées conformément à la politique de "
            "conservation en vigueur. Une conservation distincte peut intervenir "
            "lorsqu'un élément est nécessaire à une facture, un incident ou un litige."
        ),
    },
    {
        "category": "journaux_techniques",
        "description": (
            "Journaux techniques et de sécurité : durée définie par le risque, "
            "les exigences de sécurité et les obligations applicables."
        ),
    },
]


def active_technical_providers() -> list[dict[str, Any]]:
    return [p for p in TECHNICAL_PROVIDERS if p.get("active")]


def compliance_snapshot() -> dict[str, Any]:
    return {
        "subprocessors_version": SUBPROCESSORS_VERSION,
        "retention_policy_version": RETENTION_POLICY_VERSION,
        "penalty_calculation_version": PENALTY_CALCULATION_VERSION,
        "providers": active_technical_providers(),
        "retention_categories": RETENTION_CATEGORIES,
    }
