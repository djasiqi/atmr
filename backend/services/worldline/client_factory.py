"""Client API Worldline Connect (SDK officiel, auth v1HMAC)."""

from __future__ import annotations

import logging
import os
import tempfile
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass


def worldline_configured() -> bool:
    return bool(
        (os.getenv("WORLDLINE_API_KEY_ID") or "").strip()
        and (os.getenv("WORLDLINE_API_SECRET") or "").strip()
        and (os.getenv("WORLDLINE_MERCHANT_ID") or "").strip()
    )


def get_worldline_api_client() -> Any:
    """Instancie le client SDK (fichier INI temporaire + clés env)."""
    if not worldline_configured():
        msg = "Worldline: WORLDLINE_API_KEY_ID, WORLDLINE_API_SECRET et WORLDLINE_MERCHANT_ID requis"
        raise RuntimeError(msg)

    from worldline.connect.sdk.factory import Factory

    key_id = os.environ["WORLDLINE_API_KEY_ID"].strip()
    secret = os.environ["WORLDLINE_API_SECRET"].strip()
    host = (
        os.getenv("WORLDLINE_API_ENDPOINT_HOST", "").strip()
        or "api.preprod.connect.worldline-solutions.com"
    )
    integrator = (os.getenv("WORLDLINE_INTEGRATOR", "Lirie") or "Lirie").strip()

    ini_body = (
        "[ConnectSDK]\n"
        f"connect.api.integrator={integrator}\n"
        f"connect.api.endpoint.host={host}\n"
        "connect.api.authorizationType=v1HMAC\n"
        "connect.api.connectTimeout=10\n"
        "connect.api.socketTimeout=300\n"
        "connect.api.maxConnections=10\n"
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix="_worldline.ini", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(ini_body)
        ini_path = tmp.name

    try:
        return Factory.create_client_from_file(ini_path, key_id, secret)
    finally:
        try:
            os.unlink(ini_path)
        except OSError as e:
            logger.debug("Could not remove temp Worldline ini: %s", e)


def get_worldline_merchant_id() -> str:
    return os.environ["WORLDLINE_MERCHANT_ID"].strip()
