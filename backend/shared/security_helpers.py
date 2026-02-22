"""Helpers de securite reutilisables : masquage IP, parsing user-agent."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

IP_MASK_MODE = os.environ.get("IP_MASK_MODE", "first_octet")
_IPV4_PART_COUNT = 4


def mask_ip(ip: str | None) -> str:
    """Masque une adresse IP selon le mode configure.

    Modes (env IP_MASK_MODE):
        first_octet  - 178.***.***.*** / 2a02:****:...  (defaut)
        first_two    - 178.192.***.***
        full_mask    - ***.***.***.***
    """
    if not ip:
        return ""

    if IP_MASK_MODE == "full_mask":
        return "***.***.***.***" if "." in ip else "****:****:****:****:****:****:****:****"

    if ":" in ip:
        first_block = ip.split(":")[0]
        return f"{first_block}:****:****:****:****:****:****:****"

    parts = ip.split(".")
    if len(parts) == _IPV4_PART_COUNT:
        if IP_MASK_MODE == "first_two":
            return f"{parts[0]}.{parts[1]}.***.***"
        return f"{parts[0]}.***.***.***"

    return "***"


def parse_device(user_agent: str | None) -> str:
    """Parse un User-Agent en description lisible ('Chrome . Windows')."""
    if not user_agent:
        return "Appareil inconnu"
    try:
        from user_agents import parse  # type: ignore[import-untyped]

        ua = parse(user_agent)
        browser = ua.browser.family or "Navigateur inconnu"
        os_name = ua.os.family or ""
        result = f"{browser} · {os_name}".strip(" ·")
        return result or "Appareil inconnu"
    except Exception:
        logger.debug("Failed to parse user-agent, using fallback")
        return "Appareil inconnu"
