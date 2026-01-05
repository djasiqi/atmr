from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class UpdateDriverPhotoResult:
    response: dict[str, Any]
    status_code: int
    should_commit: bool = False


class UpdateDriverPhotoUseCase:
    """Use-case Application: mettre à jour la photo du chauffeur."""

    class _ValidateUploadedImageFn(Protocol):
        def __call__(
            self, photo: Any, *, declared_mime_type: Any | None
        ) -> dict[str, Any]: ...

    def __init__(
        self,
        *,
        validate_uploaded_image_fn: _ValidateUploadedImageFn,
    ) -> None:
        super().__init__()
        self._validate_uploaded_image = validate_uploaded_image_fn

    def execute(
        self, *, driver: Any, payload: dict[str, Any] | None
    ) -> UpdateDriverPhotoResult:
        if not payload or "photo" not in payload:
            return UpdateDriverPhotoResult(
                response={"error": "Donnée photo non fournie"},
                status_code=400,
                should_commit=False,
            )

        photo = payload.get("photo")
        if not photo:
            return UpdateDriverPhotoResult(
                response={"error": "Photo invalide"},
                status_code=400,
                should_commit=False,
            )

        declared_mime_type = payload.get("mime_type")
        try:
            validated = self._validate_uploaded_image(
                photo, declared_mime_type=declared_mime_type
            )
        except Exception as e:
            # Ne pas dépendre de marshmallow : on extrait un message "best-effort".
            # Certaines exceptions (comme ValidationError) ont un attribut messages
            msg: str | None = None
            messages_attr = getattr(e, "messages", None)
            if messages_attr is not None:
                msg = str(messages_attr)
            if not msg:
                msg = str(e)
            return UpdateDriverPhotoResult(
                response={"error": "validation_error", "message": msg},
                status_code=400,
                should_commit=False,
            )

        content = validated.get("content")
        if not isinstance(content, (bytes, bytearray)):
            return UpdateDriverPhotoResult(
                response={
                    "error": "validation_error",
                    "message": "Contenu image invalide",
                },
                status_code=400,
                should_commit=False,
            )

        driver.driver_photo = base64.b64encode(bytes(content)).decode("utf-8")
        return UpdateDriverPhotoResult(
            response={
                "profile": driver.serialize,
                "message": "Photo mise à jour avec succès",
            },
            status_code=200,
            should_commit=True,
        )
