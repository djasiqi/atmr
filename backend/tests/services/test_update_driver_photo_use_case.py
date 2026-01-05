from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Any

from application.drivers.update_driver_photo import UpdateDriverPhotoUseCase


@dataclass
class _Driver:
    driver_photo: str | None = None
    serialize: dict[str, object] = field(default_factory=lambda: {"id": 1})


def test_missing_payload_returns_400() -> None:
    driver = _Driver()
    uc = UpdateDriverPhotoUseCase(validate_uploaded_image_fn=lambda *_a, **_k: {})
    res = uc.execute(driver=driver, payload=None)
    assert res.status_code == 400


def test_valid_photo_sets_base64_and_requires_commit() -> None:
    driver = _Driver()

    def _validate(_photo: Any, *, declared_mime_type: Any | None):  # type: ignore[no-untyped-def]
        _ = declared_mime_type
        return {"content": b"abc"}

    uc = UpdateDriverPhotoUseCase(validate_uploaded_image_fn=_validate)
    res = uc.execute(driver=driver, payload={"photo": "x", "mime_type": "image/png"})
    assert res.status_code == 200
    assert res.should_commit is True
    assert driver.driver_photo == base64.b64encode(b"abc").decode("utf-8")


def test_validator_error_returns_400() -> None:
    driver = _Driver()

    def _validate(_photo: Any, *, declared_mime_type: Any | None):  # type: ignore[no-untyped-def]
        _ = declared_mime_type
        raise ValueError("bad")

    uc = UpdateDriverPhotoUseCase(validate_uploaded_image_fn=_validate)
    res = uc.execute(driver=driver, payload={"photo": "x"})
    assert res.status_code == 400
    assert res.should_commit is False
