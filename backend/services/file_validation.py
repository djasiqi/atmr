"""✅ Utilitaires pour validation stricte des uploads de fichiers.

Ce module fournit des fonctions pour valider les uploads de fichiers :
- Validation du type MIME via signature magique
- Validation de la taille
- Validation du contenu
"""

import base64
import logging
from typing import Any

from flask import request  # pyright: ignore[reportMissingImports]
from marshmallow import ValidationError  # pyright: ignore[reportMissingImports]

from schemas.validators import (
    MAX_FILE_SIZE_BYTES,
    MAX_FILE_SIZE_MB,
    validate_image_file,
)

logger = logging.getLogger(__name__)


def decode_base64_file(base64_string: str) -> bytes:
    """Décode une chaîne base64 en bytes.

    ✅ S1: Utilitaire pour décoder les fichiers uploadés en base64.

    Args:
        base64_string: Chaîne base64 (peut contenir le préfixe data:image/...;base64,)

    Returns:
        Contenu du fichier en bytes

    Raises:
        ValidationError: Si le décodage échoue
    """
    try:
        # Supprimer le préfixe data:image/...;base64, si présent
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]

        # Décoder base64
        return base64.b64decode(base64_string, validate=True)
    except Exception as e:
        logger.warning("[File Validation] Échec décodage base64: %s", e)
        raise ValidationError("Format base64 invalide") from e


def validate_uploaded_file(
    file_content: bytes | str,
    *,
    file_type: str = "image",
    max_size_mb: float | None = None,
    declared_mime_type: str | None = None,
) -> dict[str, Any]:
    """Valide un fichier uploadé (image ou document).

    ✅ S1: Validation serveur stricte pour uploads de fichiers (type MIME, signature magique).

    Args:
        file_content: Contenu du fichier (bytes ou base64 string)
        file_type: Type de fichier attendu ("image" ou "document")
        max_size_mb: Taille maximale en MB (défaut: MAX_FILE_SIZE_MB)
        declared_mime_type: Type MIME déclaré par le client (optionnel)

    Returns:
        Dict avec informations du fichier validé:
        {
            "content": bytes,
            "mime_type": str,
            "size_bytes": int,
            "size_mb": float
        }

    Raises:
        ValidationError: Si le fichier est invalide

    Exemple:
        >>> # Depuis une route Flask
        >>> data = request.get_json()
        >>> photo_base64 = data.get("photo")
        >>> validated = validate_uploaded_file(
        ...     photo_base64,
        ...     file_type="image",
        ...     declared_mime_type=data.get("mime_type")
        ... )
        >>> driver.driver_photo = validated["content"]
    """
    # Décoder base64 si nécessaire
    if isinstance(file_content, str):
        file_content_bytes = decode_base64_file(file_content)
    elif isinstance(file_content, bytes):  # pyright: ignore[reportUnnecessaryIsInstance]
        # Vérification nécessaire pour validation runtime (type checker ne peut pas garantir le type)
        file_content_bytes = file_content
    else:
        raise ValidationError("file_content doit être bytes ou base64 string")

    # Vérifier la taille
    max_size = (max_size_mb * 1024 * 1024) if max_size_mb else MAX_FILE_SIZE_BYTES
    if len(file_content_bytes) > max_size:
        size_mb = len(file_content_bytes) / (1024 * 1024)
        max_size_mb_actual = max_size_mb or MAX_FILE_SIZE_MB
        raise ValidationError(
            f"Fichier trop volumineux ({size_mb:.2f} MB). "
            + f"Taille maximale autorisée: {max_size_mb_actual} MB"
        )

    # Valider le type MIME via signature magique
    if file_type == "image":
        mime_type = validate_image_file(file_content_bytes, declared_mime_type)
    elif file_type == "document":
        from schemas.validators import validate_document_file

        mime_type = validate_document_file(file_content_bytes, declared_mime_type)
    else:
        raise ValidationError(f"Type de fichier non supporté: {file_type}")

    return {
        "content": file_content_bytes,
        "mime_type": mime_type,
        "size_bytes": len(file_content_bytes),
        "size_mb": len(file_content_bytes) / (1024 * 1024),
    }


def get_file_from_request(
    field_name: str = "file",
    *,
    file_type: str = "image",
    max_size_mb: float | None = None,
) -> dict[str, Any]:
    """Récupère et valide un fichier depuis une requête Flask.

    ✅ S1: Helper pour récupérer et valider un fichier depuis request.files ou request.json.

    Args:
        field_name: Nom du champ dans la requête
        file_type: Type de fichier attendu ("image" ou "document")
        max_size_mb: Taille maximale en MB

    Returns:
        Dict avec informations du fichier validé (voir validate_uploaded_file)

    Raises:
        ValidationError: Si le fichier est invalide ou manquant
    """
    # Essayer request.files d'abord (multipart/form-data)
    if field_name in request.files:
        file_obj = request.files[field_name]
        if file_obj.filename:
            file_content = file_obj.read()
            declared_mime_type = file_obj.content_type
            return validate_uploaded_file(
                file_content,
                file_type=file_type,
                max_size_mb=max_size_mb,
                declared_mime_type=declared_mime_type,
            )

    # Essayer request.json (base64)
    if request.is_json:
        data = request.get_json(silent=True)
        if data and field_name in data:
            file_content = data[field_name]
            declared_mime_type = data.get(f"{field_name}_mime_type") or data.get(
                "mime_type"
            )
            return validate_uploaded_file(
                file_content,
                file_type=file_type,
                max_size_mb=max_size_mb,
                declared_mime_type=declared_mime_type,
            )

    raise ValidationError(f"Fichier '{field_name}' manquant dans la requête")
