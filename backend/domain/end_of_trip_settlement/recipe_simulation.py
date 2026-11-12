"""Lecture simulée d'une recette locale. Fermée par défaut.

Ouverte seulement par l'environnement de recette locale explicite
(voir local_recipe.py). Elle ne rend aucun paiement exécutable.

Une simulation n'apparaît que si la réservation porte une politique
de règlement explicitement marquée simulation. Les refus réels
(institution, déjà payé, facture émise, aller-retour, payeur ambigu,
prix divergent) ne deviennent jamais une simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from domain.end_of_trip_settlement.driver_read import simulated_driver_contract
from domain.end_of_trip_settlement.local_recipe import local_recipe_environment

SIMULATION_POLICY_PREFIX = "SIMULATION-"
_ACTIONABLE_MODES = ("cash", "invoice")
# Seules ces revues administratives sont remplacées par la simulation :
# l'encaissement réel est éteint, ou aucun moyen réel n'est autorisé.
# Tout autre motif (payeur, prix, paiement en ligne, aller-retour, entreprise)
# est un refus réel conservé tel quel.
_OVERLAY_REASONS = frozenset(
    {
        "collection_disabled",
        "settlement_policy_missing",
        "no_authorized_mode",
        "invoice_signature_mandatory",
        "signature_text_unapproved",
        "missing_billing_party",
    }
)


def recipe_simulation_read_allowed(
    *,
    database: str | None = None,
    switch: str | None = None,
    production: bool | None = None,
    host: str | None = None,
) -> bool:
    """Alias de l'environnement de recette locale. Aucune condition en moins."""
    return local_recipe_environment(
        database=database, switch=switch, production=production, host=host
    )


def simulation_policy(policy_version: str | None) -> bool:
    """Seule une politique marquée SIMULATION- alimente la recette."""
    return str(policy_version or "").startswith(SIMULATION_POLICY_PREFIX)


@dataclass(frozen=True)
class RecipeBookingFacts:
    billed_to_type: str
    is_round_trip: bool
    amount_chf: str | None
    customer_name: str | None
    company_id: int
    # Modes de la politique de simulation explicite ; None si aucune.
    simulation_modes: tuple[str, ...] | None = None
    # Modalité de cette politique. « patient_invoice_after » ne propose que la facture.
    simulation_modality: str | None = None
    # Aller-retour dont CE trajet a un montant propre et un accord séparé.
    separate_leg_approved: bool = False


def overlay_recipe_contract(
    contract: dict[str, object], facts: RecipeBookingFacts
) -> dict[str, object]:
    """Propose les modes de la politique de simulation. Les refus réels restent."""
    if facts.simulation_modes is None:
        return contract
    if facts.simulation_modality not in (None, "on_site", "patient_invoice_after"):
        return contract
    if facts.is_round_trip and not facts.separate_leg_approved:
        return contract
    if (
        facts.separate_leg_approved
        and contract.get("reason") == "separate_leg_collection_unapproved"
    ):
        contract = dict(contract)
        contract["kind"] = "administrative_review"
        contract["reason"] = "collection_disabled"
    if facts.billed_to_type.strip().lower() != "patient":
        return contract
    if contract.get("kind") != "administrative_review":
        return contract
    if contract.get("reason") not in _OVERLAY_REASONS:
        return contract
    modes = tuple(mode for mode in _ACTIONABLE_MODES if mode in facts.simulation_modes)
    if facts.simulation_modality == "patient_invoice_after":
        modes = tuple(mode for mode in modes if mode == "invoice")
    if not modes:
        return contract
    if not facts.amount_chf:
        return contract
    amount = Decimal(facts.amount_chf)
    if amount <= 0:
        return contract
    kind = "collection_authorized" if "cash" in modes else "invoice_authorized"
    return simulated_driver_contract(
        kind=kind,
        modes=modes,
        amount_chf=format(amount.quantize(Decimal("0.01")), "f"),
        reason="recipe_simulation",
        debtor_label=facts.customer_name or "Patient fictif",
        company_id=facts.company_id,
        preview_screen="summary",
        document_approved=False,
        signature_required=False,
    )
