"""Utilitaires pour la validation des fichiers uploadés.

Fournit des fonctions réutilisables pour valider les uploads de fichiers
selon les bonnes pratiques de sécurité.
"""

import os
from typing import Any

# Magic bytes pour différents types de fichiers
MAGIC_BYTES = {
    "png": [b"\x89PNG\r\n\x1a\n"],
    "jpg": [b"\xff\xd8\xff"],
    "jpeg": [b"\xff\xd8\xff"],
    "gif": [b"GIF87a", b"GIF89a"],
    "webp": [b"RIFF", b"WEBP"],
    "pdf": [b"%PDF"],
    "svg": [b"<svg", b"<?xml"],  # SVG peut commencer par <svg ou <?xml
}

# Extensions autorisées par type
ALLOWED_IMAGE_EXT = {"png", "jpg", "jpeg", "gif", "webp", "svg"}
ALLOWED_PDF_EXT = {"pdf"}
ALLOWED_LOGO_EXT = {"png", "jpg", "jpeg", "svg"}


def validate_file_extension(
    filename: str, allowed_extensions: set[str]
) -> tuple[bool, str | None]:
    """Valide l'extension d'un fichier.

    Args:
        filename: Nom du fichier
        allowed_extensions: Ensemble des extensions autorisées (minuscules)

    Returns:
        Tuple (is_valid, error_message)
    """
    if not filename or "." not in filename:
        return False, "Nom de fichier invalide ou extension manquante"

    ext = filename.rsplit(".", 1)[1].lower()
    if ext not in allowed_extensions:
        return (
            False,
            f"Extension non autorisée: {ext}. Autorisées: {', '.join(sorted(allowed_extensions))}",
        )

    return True, None


def validate_file_size(file_size: int, max_size_bytes: int) -> tuple[bool, str | None]:
    """Valide la taille d'un fichier.

    Args:
        file_size: Taille du fichier en bytes
        max_size_bytes: Taille maximale autorisée en bytes

    Returns:
        Tuple (is_valid, error_message)
    """
    if file_size > max_size_bytes:
        max_size_mb = max_size_bytes / (1024 * 1024)
        return (
            False,
            f"Fichier trop volumineux: {file_size} bytes (max {max_size_mb:.1f} Mo)",
        )

    return True, None


def validate_file_content(
    file_bytes: bytes, expected_ext: str
) -> tuple[bool, str | None]:
    """Valide le contenu d'un fichier en vérifiant les magic bytes.

    Args:
        file_bytes: Contenu du fichier en bytes
        expected_ext: Extension attendue (minuscule)

    Returns:
        Tuple (is_valid, error_message)
    """
    if not file_bytes:
        return False, "Fichier vide"

    # Pour SVG, on accepte du texte XML/SVG
    if expected_ext == "svg":
        content_start = file_bytes[:100].decode("utf-8", errors="ignore").lower()
        if "svg" in content_start or "<?xml" in content_start:
            return True, None
        return False, "Fichier SVG invalide (contenu non reconnu)"

    # Pour les autres formats, vérifier les magic bytes
    if expected_ext not in MAGIC_BYTES:
        # Si pas de magic bytes définis pour cette extension, on accepte
        return True, None

    valid_magics = MAGIC_BYTES[expected_ext]
    file_start = file_bytes[: min(len(file_bytes), 20)]

    for magic in valid_magics:
        if file_start.startswith(magic):
            return True, None

    return (
        False,
        f"Type de fichier invalide: le contenu ne correspond pas à l'extension {expected_ext}",
    )


def _get_file_size(file: Any) -> tuple[int | None, str | None]:
    """Obtient la taille d'un fichier.

    Returns:
        Tuple (size, error_message)
    """
    try:
        if hasattr(file, "stream"):
            file.stream.seek(0, os.SEEK_END)
            size = file.stream.tell()
            file.stream.seek(0)
            return size, None
        if hasattr(file, "tell") and hasattr(file, "seek"):
            file.seek(0, os.SEEK_END)
            size = file.tell()
            file.seek(0)
            return size, None
        if isinstance(file, bytes):
            return len(file), None
        return None, "Impossible de déterminer la taille du fichier"
    except Exception as e:
        return None, f"Erreur lors de la lecture de la taille: {e!s}"


def _read_file_bytes(file: Any, size: int) -> tuple[bytes | None, str | None]:
    """Lit les premiers bytes d'un fichier pour validation.

    Returns:
        Tuple (file_bytes, error_message)
    """
    try:
        if hasattr(file, "read"):
            file.seek(0)
            file_bytes = file.read(min(1024, size))
            file.seek(0)
            return file_bytes, None
        if isinstance(file, bytes):
            return file[:1024], None
        return None, "Impossible de lire le contenu du fichier"
    except Exception as e:
        return None, f"Erreur lors de la lecture du contenu: {e!s}"


def validate_file_upload(  # noqa: PLR0911
    file: Any,
    filename: str | None,
    allowed_extensions: set[str],
    max_size_bytes: int,
    validate_content: bool = True,
) -> tuple[bool, str | None]:
    """Valide complètement un fichier uploadé.

    Args:
        file: Objet fichier (Flask FileStorage ou similaire)
        filename: Nom du fichier
        allowed_extensions: Extensions autorisées
        max_size_bytes: Taille maximale en bytes
        validate_content: Si True, valide aussi le contenu (magic bytes)

    Returns:
        Tuple (is_valid, error_message)
    """
    # Vérifications initiales
    if not file or not filename:
        return False, "Aucun fichier fourni" if not file else "Nom de fichier manquant"

    # Valider l'extension
    is_valid, error = validate_file_extension(filename, allowed_extensions)
    if not is_valid:
        return False, error

    # Obtenir l'extension et la taille
    ext = filename.rsplit(".", 1)[1].lower()
    size, error = _get_file_size(file)
    if error or size is None:
        return False, error or "Taille du fichier indéterminée"

    # Valider la taille et le contenu
    is_valid, error = validate_file_size(size, max_size_bytes)
    if not is_valid:
        return False, error

    # Valider le contenu si demandé
    if validate_content:
        file_bytes, error = _read_file_bytes(file, size)
        if error or file_bytes is None:
            return False, error or "Impossible de lire le contenu du fichier"
        is_valid, error = validate_file_content(file_bytes, ext)
        if not is_valid:
            return False, error

    return True, None
