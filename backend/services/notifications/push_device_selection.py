"""Sélection des cibles push chauffeur (priorité FCM Android, dédup token)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ext import app_logger

PushDeviceDict = dict[str, Any]


def dedupe_push_devices_by_token(devices: list[PushDeviceDict]) -> list[PushDeviceDict]:
    """Garde la ligne la plus récente par valeur de token."""
    seen: dict[str, PushDeviceDict] = {}
    for candidate in devices:
        token = candidate.get("token")
        if not token:
            continue
        existing = seen.get(token)
        if existing is None:
            seen[token] = candidate
            continue
        cur_updated = existing.get("updated_at")
        new_updated = candidate.get("updated_at")
        if new_updated and (not cur_updated or new_updated > cur_updated):
            seen[token] = candidate
    return list(seen.values())


def prioritize_android_fcm_devices(
    devices: list[PushDeviceDict],
    *,
    driver_id: int | None = None,
) -> list[PushDeviceDict]:
    """Par device_id Android : FCM prioritaire ; Expo seulement en fallback explicite."""
    by_device: dict[str, list[PushDeviceDict]] = defaultdict(list)
    without_device: list[PushDeviceDict] = []

    for device in devices:
        device_id = device.get("device_id")
        if device_id:
            by_device[str(device_id)].append(device)
        else:
            without_device.append(device)

    selected: list[PushDeviceDict] = []

    for device_id, group in by_device.items():
        android = [d for d in group if (d.get("platform") or "").lower() == "android"]
        non_android = [
            d for d in group if (d.get("platform") or "").lower() != "android"
        ]

        if android:
            fcm_android = [d for d in android if (d.get("provider") or "expo") == "fcm"]
            expo_android = [
                d for d in android if (d.get("provider") or "expo") == "expo"
            ]
            if fcm_android:
                selected.extend(fcm_android)
            elif expo_android:
                app_logger.warning(
                    "[push] android_no_fcm_token_fallback_expo driver=%s device_id=%s",
                    driver_id,
                    device_id,
                )
                selected.extend(expo_android)
        selected.extend(non_android)

    selected.extend(without_device)
    return _drop_android_expo_when_driver_has_fcm(devices, selected)


def _drop_android_expo_when_driver_has_fcm(
    all_devices: list[PushDeviceDict],
    selected: list[PushDeviceDict],
) -> list[PushDeviceDict]:
    """Si le chauffeur a un token FCM Android, ne pas aussi pousser via Expo Android.

    Cas réel : migration Expo → FCM avec rotation de ``device_id`` (même téléphone,
    deux lignes actives) → double notification identique dans le tiroir.
    """
    has_android_fcm = any(
        (d.get("platform") or "").lower() == "android"
        and (d.get("provider") or "expo") == "fcm"
        for d in all_devices
    )
    if not has_android_fcm:
        return selected
    filtered = [
        d
        for d in selected
        if not (
            (d.get("platform") or "").lower() == "android"
            and (d.get("provider") or "expo") == "expo"
        )
    ]
    if len(filtered) < len(selected):
        app_logger.info(
            "[push] android_expo_skipped_driver_has_fcm before=%s after=%s",
            len(selected),
            len(filtered),
        )
    return filtered


def _keep_latest_android_fcm_only(
    devices: list[PushDeviceDict],
) -> list[PushDeviceDict]:
    """Un seul token FCM Android par chauffeur (device_id roté → plusieurs lignes actives)."""
    android_fcm = [
        d
        for d in devices
        if (d.get("platform") or "").lower() == "android"
        and (d.get("provider") or "expo") == "fcm"
    ]
    if len(android_fcm) <= 1:
        return devices

    def sort_key(row: PushDeviceDict) -> tuple[int, int]:
        updated = row.get("updated_at")
        ts = (
            updated.timestamp()
            if updated is not None and hasattr(updated, "timestamp")
            else 0
        )
        row_id = row.get("id")
        return (ts, int(row_id) if isinstance(row_id, int) else 0)

    keep = max(android_fcm, key=sort_key)
    drop_ids = {d.get("id") for d in android_fcm if d is not keep}
    filtered = [d for d in devices if d.get("id") not in drop_ids]
    app_logger.info(
        "[push] android_fcm_single_target kept_id=%s dropped=%s",
        keep.get("id"),
        len(drop_ids),
    )
    return filtered


def device_token_row_to_push_dict(row: Any) -> PushDeviceDict:
    return {
        "id": row.id,
        "token": row.token,
        "device_id": getattr(row, "device_id", None),
        "platform": getattr(row, "platform", None),
        "provider": getattr(row, "provider", "expo"),
        "updated_at": getattr(row, "updated_at", None),
    }


def prepare_driver_push_targets(
    device_tokens_raw: list[Any],
    *,
    driver_id: int | None = None,
) -> list[PushDeviceDict]:
    """Extrait, déduplique par token, priorise FCM sur Android."""
    extracted = [
        device_token_row_to_push_dict(row)
        for row in device_tokens_raw
        if getattr(row, "token", None)
    ]
    deduped = dedupe_push_devices_by_token(extracted)
    if len(deduped) < len(extracted):
        app_logger.info(
            "[push] dedup tokens driver=%s before=%s after=%s",
            driver_id,
            len(extracted),
            len(deduped),
        )
    prioritized = prioritize_android_fcm_devices(deduped, driver_id=driver_id)
    return _keep_latest_android_fcm_only(prioritized)


def android_has_fcm_token(active_tokens: list[Any]) -> bool:
    for token in active_tokens:
        if (getattr(token, "platform", None) or "").lower() != "android":
            continue
        if (getattr(token, "provider", None) or "expo") == "fcm":
            return True
    return False


def android_has_expo_only(active_tokens: list[Any]) -> bool:
    android_tokens = [
        t
        for t in active_tokens
        if (getattr(t, "platform", None) or "").lower() == "android"
    ]
    if not android_tokens:
        return False
    has_fcm = any(
        (getattr(t, "provider", None) or "expo") == "fcm" for t in android_tokens
    )
    has_expo = any(
        (getattr(t, "provider", None) or "expo") == "expo" for t in android_tokens
    )
    return has_expo and not has_fcm
