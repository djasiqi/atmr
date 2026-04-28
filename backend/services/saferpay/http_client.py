"""Client HTTP JSON Saferpay (Basic Auth)."""

from __future__ import annotations

import logging
import os
from http import HTTPStatus
from typing import Any

import requests
from requests.auth import HTTPBasicAuth

from services.saferpay.config import saferpay_api_base_url

logger = logging.getLogger(__name__)


def saferpay_post_json(subpath: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any] | None, str]:
    """POST JSON vers l'API Saferpay. Retourne (status_code, json_dict_or_none, raw_text)."""
    base = saferpay_api_base_url()
    url = f"{base}/{subpath.lstrip('/')}"
    auth = HTTPBasicAuth(
        os.environ["SAFERPAY_API_USERNAME"].strip(),
        os.environ["SAFERPAY_API_PASSWORD"].strip(),
    )
    r = requests.post(
        url,
        json=payload,
        auth=auth,
        timeout=60,
        headers={"Accept": "application/json", "Content-Type": "application/json; charset=utf-8"},
    )
    text = r.text or ""
    try:
        data = r.json()
    except Exception:
        data = None
    if r.status_code >= HTTPStatus.BAD_REQUEST:
        logger.warning(
            "Saferpay HTTP %s %s: %s",
            r.status_code,
            subpath,
            text[:800],
        )
    return r.status_code, data if isinstance(data, dict) else None, text
