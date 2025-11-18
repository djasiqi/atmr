# backend/services/delay_tools.py
"""Delay tools for PMR transport (standalone).

Features:
- delay_minutes: ETA - scheduled (in minutes, float, negative if early)
- is_delayed: boolean with buffer tolerance
- severity: low / medium / high / critical (configurable thresholds)
- AntiFlapDelayChecker: confirm delay only after N consecutive hits within a time window

No external deps. Thread-safe. Can be used from sockets, cron, or workers.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Tuple

# -----------------------------
# Basic helpers (stateless)
# -----------------------------


def delay_minutes(current_eta: datetime, scheduled_time: datetime) -> float:
    """Returns ETA - scheduled in minutes (negative if early).
    Works with naive or tz-aware datetimes as long as they are comparable.
    """
    delta = current_eta - scheduled_time
    return float(delta.total_seconds()) / 60.0


def is_delayed(
    current_eta: datetime,
    scheduled_time: datetime,
    buffer_minutes: int = 5,
) -> bool:
    """True if ETA is strictly after scheduled + buffer."""
    return current_eta > (scheduled_time + timedelta(minutes=buffer_minutes))


@dataclass(frozen=True)
class SeverityThresholds:
    """Inclusive upper-bounds in minutes for each severity band.

    Example (defaults):
        low      : 0..5
        medium   : 5..10
        high     : 10..20
        critical : >20
    """

    low_max: float = 5.0
    med_max: float = 10.0
    high_max: float = 20.0


def severity_label(delay_min: float, thresholds: SeverityThresholds | None = None) -> str:
    """Map a delay in minutes to a severity bucket.
    Negative (early) returns 'low' by convention.
    """
    if thresholds is None:
        thresholds = SeverityThresholds()
    if delay_min <= thresholds.low_max:
        return "low"
    if delay_min <= thresholds.med_max:
        return "medium"
    if delay_min <= thresholds.high_max:
        return "high"
    return "critical"


# -----------------------------
# Anti-flap confirmation
# -----------------------------


@dataclass
class AntiFlapConfig:
    """Require 'confirm_hits' consecutive TRUE detections inside 'window_sec'
    to confirm a delay. Resets when a FALSE is observed or when the window is exceeded.
    """

    confirm_hits: int = 3  # how many consecutive hits required
    window_sec: int = 90  # time budget for those hits
    max_idle_sec: int = 600  # auto-gc entries idle longer than this


class AntiFlapDelayChecker:
    """Thread-safe anti-flap confirmation per (assignment_id, phase) key.

    Typical usage:
        checker = AntiFlapDelayChecker()
        delayed = is_delayed(eta, scheduled, buffer_minutes=5)
        confirmed = checker.observe("A123", "pickup", delayed)
        if confirmed:
            # emit one alert
    """

    def __init__(self, config: AntiFlapConfig | None = None) -> None:
        super().__init__()
        self.config = config or AntiFlapConfig()
        # store: key -> (count, first_ts, last_ts, last_state)
        self._store: Dict[Tuple[str, str], Tuple[int, float, float, bool]] = {}
        self._lock = threading.Lock()

    def _now(self) -> float:
        # float seconds since epoch
        return datetime.now().timestamp()

    def observe(self, assignment_id: str, phase: str, is_hit: bool) -> bool:
        """Observe one detection for a given (assignment_id, phase).

        Returns True only when the number of *consecutive* TRUE observations
        reaches 'confirm_hits' within 'window_sec'. Otherwise returns False.
        """
        key = (str(assignment_id), str(phase))
        now = self._now()
        with self._lock:
            self._gc(now)

            prev = self._store.get(key)
            if prev is None:
                # initialize
                count = 1 if is_hit else 0
                first_ts = now
                last_ts = now
                self._store[key] = (count, first_ts, last_ts, is_hit)
                return count >= self.config.confirm_hits and is_hit

            count, first_ts, last_ts, last_state = prev

            # If window expired, reset sequence
            if (now - first_ts) > self.config.window_sec:
                count = 0
                first_ts = now

            # If previous state was false, start a fresh positive streak
            # a single FALSE breaks the streak
            count = (count + 1 if last_state else 1) if is_hit else 0

            self._store[key] = (count, first_ts, now, is_hit)
            return is_hit and (count >= self.config.confirm_hits)

    def reset(self, assignment_id: str, phase: str) -> None:
        """Manually clear the streak for a given (assignment_id, phase)."""
        key = (str(assignment_id), str(phase))
        with self._lock:
            self._store.pop(key, None)

    def _gc(self, now: float) -> None:
        """Garbage collect idle entries."""
        idle = self.config.max_idle_sec
        if idle <= 0:
            return
        to_del = []
        for key, (_, _, last_ts, _) in self._store.items():
            if (now - last_ts) > idle:
                to_del.append(key)
        for key in to_del:
            self._store.pop(key, None)


# -----------------------------
# Convenience facade
# -----------------------------


class DelayDecider:
    """Small facade to combine the helpers:

    - compute delay_min
    - check delayed with buffer
    - apply anti-flap
    - return severity & confirmation state

    You pass the already computed ETA & scheduled time (no coupling to OSRM/data.py).
    """

    def __init__(
        self,
        pickup_buffer_min: int = 5,
        dropoff_buffer_min: int = 5,
        thresholds: SeverityThresholds | None = None,
        anti_flap: AntiFlapDelayChecker | None = None,
    ) -> None:
        super().__init__()
        self.pickup_buffer_min = pickup_buffer_min
        self.dropoff_buffer_min = dropoff_buffer_min
        self.thresholds = thresholds or SeverityThresholds()
        self.anti_flap = anti_flap or AntiFlapDelayChecker()

    def evaluate(
        self,
        *,
        assignment_id: str,
        phase: str,  # "pickup" | "dropoff"
        eta: datetime,
        scheduled: datetime,
    ) -> Dict[str, object]:
        """Returns a dict like:
        {
          "delayed": True/False,          # raw (with buffer)
          "confirmed": True/False,        # after anti-flap
          "delay_min": 7.3,
          "severity": "medium"
        }.
        """
        buffer_min = self.pickup_buffer_min if phase == "pickup" else self.dropoff_buffer_min
        dmin = delay_minutes(eta, scheduled)
        delayed_raw = dmin > buffer_min
        confirmed = self.anti_flap.observe(str(assignment_id), str(phase), delayed_raw)
        sev = severity_label(max(dmin, 0.0), self.thresholds)  # negative → low
        return {
            "delayed": bool(delayed_raw),
            "confirmed": bool(confirmed),
            "delay_min": float(dmin),
            "severity": str(sev),
        }
