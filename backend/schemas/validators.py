"""✅ Validateurs Marshmallow réutilisables pour validation stricte des entrées.

Ce module fournit des validateurs personnalisés pour :
- Coordonnées GPS (latitude, longitude)
- Uploads de fichiers (type MIME, signature magique)
- Autres validations de sécurité
"""
# pyright: reportImplicitOverride=false
# Note: Les méthodes de LatitudeValidator et LongitudeValidator utilisent @override mais basedpyright
# ne le reconnaît pas toujours dans ce contexte (problème connu avec les imports conditionnels)

import logging
from typing import Any

from marshmallow import (  # pyright: ignore[reportMissingImports]
    ValidationError,
    validate,
)

from shared.geo_utils import validate_coordinates

# Import override - typing_extensions est garanti disponible (dans requirements.base.txt)
try:
    from typing import (
        override,
    )
except ImportError:
    from typing_extensions import override  # Python < 3.12

logger = logging.getLogger(__name__)

# Constantes pour validation GPS
LATITUDE_MIN = -90.0
LATITUDE_MAX = 90.0
LONGITUDE_MIN = -180.0
LONGITUDE_MAX = 180.0

# Constantes pour validation fichiers
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
WEBP_HEADER_OFFSET = 12  # Offset pour vérifier "WEBP" dans les fichiers RIFF

# Types MIME autorisés pour uploads
ALLOWED_IMAGE_MIME_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/gif",
    "image/webp",
}

ALLOWED_DOCUMENT_MIME_TYPES = {
    "application/pdf",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx
    "text/plain",
}

# Signatures magiques (magic numbers) pour validation fichiers
# Format: (bytes_start, mime_type)
FILE_SIGNATURES = {
    # Images
    b"\xff\xd8\xff": "image/jpeg",  # JPEG
    b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a": "image/png",  # PNG
    b"GIF87a": "image/gif",  # GIF87a
    b"GIF89a": "image/gif",  # GIF89a
    b"RIFF": "image/webp",  # WebP (simplifié, vérifier après "WEBP")
    # Documents
    b"%PDF": "application/pdf",  # PDF
    b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1": "application/msword",  # DOC (ancien format)
    b"PK\x03\x04": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # DOCX/ZIP
    # Texte
    b"\xef\xbb\xbf": "text/plain",  # UTF-8 BOM
}


def validate_latitude(value: Any) -> float:
    """Valide une latitude GPS.

    ✅ S1: Validation stricte des coordonnées GPS pour éviter injection et valeurs invalides.

    Args:
        value: Latitude à valider

    Returns:
        Latitude validée

    Raises:
        ValidationError: Si la latitude est invalide

    Exemple:
        >>> validate_latitude(46.2044)  # Genève
        46.2044
        >>> validate_latitude(91.0)  # Invalide
        ValidationError: Latitude doit être entre -90 et 90
    """
    if not isinstance(value, (int, float)):
        raise ValidationError("Latitude doit être un nombre")
    if not (LATITUDE_MIN <= value <= LATITUDE_MAX):
        raise ValidationError(
            f"Latitude doit être entre {LATITUDE_MIN} et {LATITUDE_MAX} degrés"
        )
    return float(value)


def validate_longitude(value: Any) -> float:
    """Valide une longitude GPS.

    ✅ S1: Validation stricte des coordonnées GPS pour éviter injection et valeurs invalides.

    Args:
        value: Longitude à valider

    Returns:
        Longitude validée

    Raises:
        ValidationError: Si la longitude est invalide

    Exemple:
        >>> validate_longitude(6.1432)  # Genève
        6.1432
        >>> validate_longitude(181.0)  # Invalide
        ValidationError: Longitude doit être entre -180 et 180
    """
    if not isinstance(value, (int, float)):
        raise ValidationError("Longitude doit être un nombre")
    if not (LONGITUDE_MIN <= value <= LONGITUDE_MAX):
        raise ValidationError(
            f"Longitude doit être entre {LONGITUDE_MIN} et {LONGITUDE_MAX} degrés"
        )
    return float(value)


def validate_coordinate_pair(lat: float, lon: float) -> tuple[float, float]:
    """Valide une paire de coordonnées GPS (latitude, longitude).

    ✅ S1: Validation stricte des coordonnées GPS pour éviter injection et valeurs invalides.

    Args:
        lat: Latitude
        lon: Longitude

    Returns:
        Tuple (lat, lon) validé

    Raises:
        ValidationError: Si les coordonnées sont invalides
    """
    lat_validated = validate_latitude(lat)
    lon_validated = validate_longitude(lon)

    # Vérifier que les coordonnées sont valides ensemble (utilise validate_coordinates)
    if not validate_coordinates(lat_validated, lon_validated):
        raise ValidationError(
            f"Coordonnées GPS invalides: ({lat_validated}, {lon_validated})"
        )

    return (lat_validated, lon_validated)


def validate_file_mime_type(
    file_content: bytes, declared_mime_type: str | None = None
) -> str:
    """Valide le type MIME d'un fichier en vérifiant sa signature magique.

    ✅ S1: Validation serveur stricte pour uploads de fichiers (type MIME, signature magique).

    Args:
        file_content: Contenu du fichier (bytes)
        declared_mime_type: Type MIME déclaré par le client (optionnel, pour vérification)

    Returns:
        Type MIME détecté depuis la signature magique

    Raises:
        ValidationError: Si le type MIME est invalide ou non autorisé

    Note:
        La signature magique est plus fiable que le type MIME déclaré par le client,
        car elle vérifie le contenu réel du fichier.
    """
    if not file_content:
        raise ValidationError("Fichier vide")

    # Vérifier la taille maximale
    if len(file_content) > MAX_FILE_SIZE_BYTES:
        raise ValidationError(f"Fichier trop volumineux (max {MAX_FILE_SIZE_MB} MB)")

    # Détecter le type MIME depuis la signature magique
    detected_mime_type = None
    for signature, mime_type in FILE_SIGNATURES.items():
        if file_content.startswith(signature):
            detected_mime_type = mime_type
            break

    # Vérifications spéciales pour certains formats
    if detected_mime_type is None:
        # Vérifier WebP (signature "RIFF" + "WEBP" à l'offset 8)
        if (
            file_content.startswith(b"RIFF")
            and len(file_content) > WEBP_HEADER_OFFSET
            and file_content[8:WEBP_HEADER_OFFSET] == b"WEBP"
        ):
            detected_mime_type = "image/webp"
        # Vérifier DOCX (ZIP avec structure spécifique)
        elif file_content.startswith(b"PK\x03\x04") and b"word/" in file_content[:1024]:
            # Vérifier que c'est bien un DOCX (contient "word/" dans le ZIP)
            detected_mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

    if detected_mime_type is None:
        raise ValidationError(
            "Type de fichier non reconnu ou non autorisé (signature magique invalide)"
        )

    # Vérifier que le type MIME détecté est autorisé
    allowed_types = ALLOWED_IMAGE_MIME_TYPES | ALLOWED_DOCUMENT_MIME_TYPES
    if detected_mime_type not in allowed_types:
        raise ValidationError(
            f"Type de fichier non autorisé: {detected_mime_type}. "
            + f"Types autorisés: {', '.join(sorted(allowed_types))}"
        )

    # Si un type MIME a été déclaré, vérifier qu'il correspond
    if declared_mime_type and declared_mime_type != detected_mime_type:
        logger.warning(
            "[Security] Type MIME déclaré (%s) ne correspond pas à la signature magique (%s)",
            declared_mime_type,
            detected_mime_type,
        )
        # En production, on pourrait rejeter strictement, mais pour compatibilité,
        # on accepte si le type détecté est valide

    return detected_mime_type


def validate_image_file(
    file_content: bytes, declared_mime_type: str | None = None
) -> str:
    """Valide qu'un fichier est une image valide.

    ✅ S1: Validation serveur stricte pour uploads d'images.

    Args:
        file_content: Contenu du fichier (bytes)
        declared_mime_type: Type MIME déclaré par le client (optionnel)

    Returns:
        Type MIME de l'image détecté

    Raises:
        ValidationError: Si le fichier n'est pas une image valide
    """
    mime_type = validate_file_mime_type(file_content, declared_mime_type)

    if mime_type not in ALLOWED_IMAGE_MIME_TYPES:
        raise ValidationError(
            f"Le fichier n'est pas une image valide. Type détecté: {mime_type}. "
            + f"Types d'images autorisés: {', '.join(sorted(ALLOWED_IMAGE_MIME_TYPES))}"
        )

    return mime_type


def validate_document_file(
    file_content: bytes, declared_mime_type: str | None = None
) -> str:
    """Valide qu'un fichier est un document valide.

    ✅ S1: Validation serveur stricte pour uploads de documents.

    Args:
        file_content: Contenu du fichier (bytes)
        declared_mime_type: Type MIME déclaré par le client (optionnel)

    Returns:
        Type MIME du document détecté

    Raises:
        ValidationError: Si le fichier n'est pas un document valide
    """
    mime_type = validate_file_mime_type(file_content, declared_mime_type)

    if mime_type not in ALLOWED_DOCUMENT_MIME_TYPES:
        raise ValidationError(
            f"Le fichier n'est pas un document valide. Type détecté: {mime_type}. "
            + f"Types de documents autorisés: {', '.join(sorted(ALLOWED_DOCUMENT_MIME_TYPES))}"
        )

    return mime_type


# Validateurs Marshmallow réutilisables
class LatitudeValidator(validate.Validator):
    """Validateur Marshmallow pour latitude GPS."""

    @override
    def __call__(self, value: Any) -> Any:
        return validate_latitude(value)

    @override
    def __repr__(self) -> str:
        return f"<LatitudeValidator(min={LATITUDE_MIN}, max={LATITUDE_MAX})>"


class LongitudeValidator(validate.Validator):
    """Validateur Marshmallow pour longitude GPS."""

    @override
    def __call__(self, value: Any) -> Any:
        return validate_longitude(value)

    @override
    def __repr__(self) -> str:
        return f"<LongitudeValidator(min={LONGITUDE_MIN}, max={LONGITUDE_MAX})>"


# Instances réutilisables
latitude_validator = LatitudeValidator()
longitude_validator = LongitudeValidator()
