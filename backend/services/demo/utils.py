"""Utilitaires partagés pour le module demo."""

import os


def get_demo_default_password() -> str:
    """Retourne le mot de passe par défaut des comptes démo (depuis env)."""
    pwd = (os.getenv("DEMO_DEFAULT_PASSWORD") or "").strip()
    if not pwd:
        # Fallback pour prod (ALLOW_NON_DEMO_SEED) si DEMO_DEFAULT_PASSWORD non configuré
        if os.getenv("ALLOW_NON_DEMO_SEED", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return "LirieDemo2024!"
        raise RuntimeError(
            "DEMO_DEFAULT_PASSWORD doit être défini pour créer des comptes démo. "
            "Ajoutez-le dans les variables d'environnement ou GitHub Secrets."
        )
    return pwd
