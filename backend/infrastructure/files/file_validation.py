from __future__ import annotations

from typing import Any


def validate_uploaded_image(
    photo: Any, *, declared_mime_type: Any | None
) -> dict[str, Any]:
    """Adapter Infrastructure: validation upload image.

    Encapsule `services.file_validation.validate_uploaded_file` pour éviter que la couche
    Application dépende de `services.*`.
    """

    from services.file_validation import validate_uploaded_file

    return validate_uploaded_file(
        photo,
        file_type="image",
        declared_mime_type=declared_mime_type,
    )
