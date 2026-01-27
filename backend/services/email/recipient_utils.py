"""Utilitaires de normalisation pour les emails."""

_RELATIONSHIP_MAP = {
    "pere": "père",
    "père": "père",
    "mere": "mère",
    "mère": "mère",
    "epouse": "épouse",
    "épouse": "épouse",
    "epoux": "époux",
    "époux": "époux",
    "fils": "fils",
    "fille": "fille",
}


def normalize_relationship_label(raw: str | None) -> str | None:
    """Normalise le lien de parenté pour les emails.

    - Mapping explicite pour éviter les fautes d'orthographe visibles
    - Retourne None si inconnu (ne pas dépendre du texte brut)
    """
    if not raw:
        return None

    cleaned = raw.strip().lower()
    if not cleaned:
        return None

    cleaned = cleaned.replace("_", " ").replace("-", " ")
    cleaned = " ".join(cleaned.split())

    return _RELATIONSHIP_MAP.get(cleaned)
