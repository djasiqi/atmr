"""✅ Utilitaire de sanitisation des inputs utilisateur pour prévenir XSS et injections.

Fournit des fonctions pour échapper HTML/JS et nettoyer les inputs utilisateur.
"""

import html
import re
from typing import Any
from urllib.parse import urlparse

# Constantes de sécurité
MAX_STRING_LENGTH = 10000  # Longueur maximale par défaut pour les strings
MAX_EMAIL_LENGTH = 254  # RFC 5321
MAX_URL_LENGTH = 2048  # Limite raisonnable pour les URLs
MIN_CONTROL_CHAR_CODE = 32  # Code ASCII minimum pour caractères non-contrôle (espace)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>", re.IGNORECASE)
SCRIPT_TAG_PATTERN = re.compile(r"<script[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL)


def escape_html(text: str | None) -> str | None:
    """Échappe les caractères HTML pour prévenir XSS.

    Args:
        text: Texte à échapper

    Returns:
        Texte échappé ou None si input None
    """
    if text is None:
        return None
    return html.escape(str(text), quote=False)


def escape_js(text: str | None) -> str | None:
    """Échappe les caractères JavaScript pour prévenir l'injection JS.

    Args:
        text: Texte à échapper

    Returns:
        Texte échappé ou None si input None
    """
    if text is None:
        return None
    # Échapper les caractères spéciaux JS
    text_str = str(text)
    text_str = text_str.replace("\\", "\\\\")
    text_str = text_str.replace("'", "\\'")
    text_str = text_str.replace('"', '\\"')
    text_str = text_str.replace("\n", "\\n")
    text_str = text_str.replace("\r", "\\r")
    # Dernière transformation avant return (RET504 : assignment avant return nécessaire ici
    # car on fait plusieurs transformations successives)
    return text_str.replace("\t", "\\t")


def sanitize_string(
    text: str | None,
    max_length: int | None = None,
    strip_html: bool = True,
    escape_html_chars: bool = True,
) -> str | None:
    """Sanitize une string en échappant HTML et limitant la longueur.

    Args:
        text: Texte à sanitizer
        max_length: Longueur maximale (par défaut MAX_STRING_LENGTH)
        strip_html: Si True, supprime les balises HTML
        escape_html_chars: Si True, échappe les caractères HTML

    Returns:
        Texte sanitizé ou None si input None
    """
    if text is None:
        return None

    text_str = str(text)

    # Limiter la longueur
    max_len = max_length or MAX_STRING_LENGTH
    if len(text_str) > max_len:
        text_str = text_str[:max_len]

    # Supprimer les balises HTML/script si demandé
    if strip_html:
        text_str = SCRIPT_TAG_PATTERN.sub("", text_str)
        text_str = HTML_TAG_PATTERN.sub("", text_str)

    # Échapper les caractères HTML si demandé
    if escape_html_chars:
        text_str = escape_html(text_str) or ""

    return text_str


def sanitize_email(email: str | None) -> str | None:
    """Valide et sanitize une adresse email.

    Args:
        email: Adresse email à valider

    Returns:
        Email sanitizé (lowercase, trimmed) ou None si invalide
    """
    if not email:
        return None

    email_str = str(email).strip().lower()

    # Vérifier la longueur
    if len(email_str) > MAX_EMAIL_LENGTH:
        return None

    # Pattern basique pour email (RFC 5322 simplifié)
    email_pattern = re.compile(
        r"^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$"
    )

    if not email_pattern.match(email_str):
        return None

    return email_str


def sanitize_url(
    url: str | None, allowed_schemes: list[str] | None = None
) -> str | None:
    """Valide et sanitize une URL.

    Args:
        url: URL à valider
        allowed_schemes: Schémas autorisés (par défaut: http, https)

    Returns:
        URL sanitizée ou None si invalide
    """
    if not url:
        return None

    url_str = str(url).strip()

    # Vérifier la longueur
    if len(url_str) > MAX_URL_LENGTH:
        return None

    # Vérifier le format
    try:
        parsed = urlparse(url_str)
    except Exception:
        return None

    # Vérifier le schéma
    schemes = allowed_schemes or ["http", "https"]
    if not parsed.scheme or parsed.scheme.lower() not in [s.lower() for s in schemes]:
        return None

    # Vérifier netloc (domaine)
    if not parsed.netloc:
        return None

    # Reconstruire l'URL en échappant les caractères dangereux dans le path
    # Note: urlparse gère déjà l'encodage, on ne fait que valider
    return url_str


def strip_control_characters(text: str | None) -> str | None:
    """Supprime les caractères de contrôle (sauf \n, \r, \t).

    Args:
        text: Texte à nettoyer

    Returns:
        Texte nettoyé ou None si input None
    """
    if text is None:
        return None

    text_str = str(text)
    # Garder \n, \r, \t, supprimer les autres caractères de contrôle
    return "".join(
        char
        for char in text_str
        if ord(char) >= MIN_CONTROL_CHAR_CODE or char in "\n\r\t"
    )


def sanitize_integer(
    value: Any, min_val: int | None = None, max_val: int | None = None
) -> int | None:
    """Valide et convertit une valeur en entier avec limites.

    Args:
        value: Valeur à convertir
        min_val: Valeur minimale (inclusive)
        max_val: Valeur maximale (inclusive)

    Returns:
        Entier validé ou None si invalide
    """
    if value is None:
        return None

    try:
        int_val = int(value)
    except (ValueError, TypeError):
        return None

    if min_val is not None and int_val < min_val:
        return None

    if max_val is not None and int_val > max_val:
        return None

    return int_val


def sanitize_float(
    value: Any, min_val: float | None = None, max_val: float | None = None
) -> float | None:
    """Valide et convertit une valeur en float avec limites.

    Args:
        value: Valeur à convertir
        min_val: Valeur minimale (inclusive)
        max_val: Valeur maximale (inclusive)

    Returns:
        Float validé ou None si invalide
    """
    if value is None:
        return None

    try:
        float_val = float(value)
    except (ValueError, TypeError):
        return None

    if min_val is not None and float_val < min_val:
        return None

    if max_val is not None and float_val > max_val:
        return None

    return float_val
