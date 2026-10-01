"""Détection de chevauchements. Les intervalles ne sont jamais fusionnés."""

from __future__ import annotations

from datetime import datetime


def mark_overlaps(entries: list[dict], *, threshold_minutes: int = 1) -> None:
    """Ajoute l'anomalie ``overlap`` sur les lignes qui se chevauchent."""
    by_driver: dict[int, list[dict]] = {}
    for entry in entries:
        start = entry.get("_start")
        end = entry.get("_end")
        driver_id = entry.get("driver_id")
        if not isinstance(start, datetime) or not isinstance(end, datetime):
            continue
        if end <= start or driver_id is None:
            continue
        by_driver.setdefault(int(driver_id), []).append(entry)

    threshold_seconds = max(0, threshold_minutes) * 60
    for group in by_driver.values():
        ordered = sorted(group, key=lambda item: item["_start"])
        for index, left in enumerate(ordered):
            for right in ordered[index + 1 :]:
                if right["_start"] >= left["_end"]:
                    break
                overlap = (
                    min(left["_end"], right["_end"])
                    - max(left["_start"], right["_start"])
                ).total_seconds()
                if overlap >= threshold_seconds and overlap > 0:
                    _add(left, "overlap")
                    _add(right, "overlap")


def _add(entry: dict, code: str) -> None:
    anomalies = entry.setdefault("anomalies", [])
    if code not in anomalies:
        anomalies.append(code)
