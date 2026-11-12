"""Droits de configuration. Aucun portail n'est activé par ce module."""

from __future__ import annotations

_RIGHTS: dict[str, frozenset[str]] = {
    "private_client": frozenset(),
    "guest": frozenset(),
    "institution": frozenset({"designate_payer"}),
    "company": frozenset({"declare_capabilities", "configure_capabilities"}),
    "lirie_operator": frozenset({"prepare_payer_lock"}),
    "driver": frozenset({"consult_decision"}),
}


def actor_may(actor: str, action: str) -> bool:
    """Le chauffeur ne change ni le payeur ni les modes. L'activation reste refusée."""
    if action in {
        "activate_collection",
        "execute_collection",
        "choose_modality",
        "widen_modes",
        "change_payer",
        "sign_for_patient",
        "exceptional_correction",
    }:
        return False
    if action == "lock_payer_in_live_portal":
        return False
    return action in _RIGHTS.get(actor, frozenset())
