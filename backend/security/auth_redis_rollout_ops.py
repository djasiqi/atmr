"""Helpers P0-6 rollout — SCAN / backfill / parité (hors chemin requête).

Utilisés par les scripts ops et les tests offline. Ne jamais utiliser KEYS.

Backfill : revalidation anti-résurrection entre lecture et écriture.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Iterator

from security.auth_redis_migration import (
    ACTIVE_PREFIX,
    PREVIOUS_PREFIX,
    REFRESH_KEY_PATTERNS,
    REVOKED_PREFIX,
    USER_ZSET_PREFIX,
)

logger = logging.getLogger(__name__)

# TTL : écart max (ms) considéré comme dérive d'horloge / temps écoulé pendant scan
TTL_DRIFT_MS_TOLERANCE = 2000


def _monotonic() -> float:
    """Horloge monotone (injectable en tests)."""
    return time.monotonic()


@dataclass
class KeySample:
    key: str
    key_type: str
    value: Any = None
    pttl_ms: int = -1  # -1 = no expire, -2 = missing
    memory_bytes: int | None = None


@dataclass
class SizeReport:
    active_count: int = 0
    previous_count: int = 0
    revoked_count: int = 0
    user_zset_count: int = 0
    sampled: list[KeySample] = field(default_factory=list)
    estimated_bytes: int = 0
    used_memory: int | None = None
    used_memory_peak: int | None = None
    maxmemory: int | None = None
    evicted_keys: int | None = None


@dataclass
class KeySnapshot:
    key: str
    key_type: str
    value: Any = None
    zset_members: list[tuple[Any, float]] | None = None
    pttl_ms: int = -1
    # Instant monotone de capture — pour préserver l'échéance absolue à l'écriture
    captured_at_mono: float | None = None


@dataclass
class ParityReport:
    legacy_active: int = 0
    auth_active: int = 0
    legacy_previous: int = 0
    auth_previous: int = 0
    legacy_revoked: int = 0
    auth_revoked: int = 0
    legacy_user_zset: int = 0
    auth_user_zset: int = 0
    missing_current_in_auth: list[str] = field(default_factory=list)
    extra_current_in_auth: list[str] = field(default_factory=list)
    mismatched_current: list[str] = field(default_factory=list)
    missing_previous_in_auth: list[str] = field(default_factory=list)
    extra_previous_in_auth: list[str] = field(default_factory=list)
    mismatched_previous: list[str] = field(default_factory=list)
    missing_revoked_in_auth: list[str] = field(default_factory=list)
    extra_revoked_in_auth: list[str] = field(default_factory=list)
    mismatched_revoked: list[str] = field(default_factory=list)
    missing_user_zset_in_auth: list[str] = field(default_factory=list)
    extra_user_zset_in_auth: list[str] = field(default_factory=list)
    mismatched_user_zset_members: list[str] = field(default_factory=list)
    mismatched_user_zset_scores: list[str] = field(default_factory=list)
    ttl_mismatched_current: list[str] = field(default_factory=list)
    ttl_mismatched_previous: list[str] = field(default_factory=list)
    ttl_mismatched_revoked: list[str] = field(default_factory=list)
    ttl_mismatched_user_zset: list[str] = field(default_factory=list)
    previous_expired_during_scan: list[str] = field(default_factory=list)

    # Alias rétrocompat (listes agrégées)
    @property
    def missing_in_auth(self) -> list[str]:
        return self.missing_current_in_auth + self.missing_previous_in_auth

    @property
    def extra_in_auth(self) -> list[str]:
        return self.extra_current_in_auth + self.extra_previous_in_auth

    @property
    def value_mismatches(self) -> list[str]:
        return self.mismatched_current + self.mismatched_previous

    @property
    def mismatched_user_zset(self) -> list[str]:
        """Union membres + scores (rétrocompat tests / scripts)."""
        return sorted(
            set(self.mismatched_user_zset_members)
            | set(self.mismatched_user_zset_scores)
        )

    @property
    def ttl_mismatches(self) -> list[str]:
        """Union de tous les TTL mismatch (rétrocompat)."""
        return (
            self.ttl_mismatched_current
            + self.ttl_mismatched_previous
            + self.ttl_mismatched_revoked
            + self.ttl_mismatched_user_zset
        )


def scan_keys(client: Any, match: str, *, count: int = 200) -> Iterator[str]:
    """Itère les clés via SCAN (jamais KEYS)."""
    cursor = 0
    while True:
        cursor, batch = client.scan(cursor=cursor, match=match, count=count)
        for key in batch:
            yield str(key)
        if int(cursor) == 0:
            break


def count_keys(client: Any, match: str, *, count: int = 200) -> int:
    return sum(1 for _ in scan_keys(client, match, count=count))


def measure_size(
    client: Any,
    *,
    sample_limit: int = 50,
    scan_count: int = 200,
) -> SizeReport:
    """Compte CURRENT/PREVIOUS + échantillon MEMORY USAGE."""
    report = SizeReport()
    try:
        mem = client.info(section="memory")
        stats = client.info(section="stats")
        report.used_memory = int(mem.get("used_memory") or 0)
        report.used_memory_peak = int(mem.get("used_memory_peak") or 0)
        report.maxmemory = int(mem.get("maxmemory") or 0)
        report.evicted_keys = int(stats.get("evicted_keys") or 0)
    except Exception:
        logger.debug("INFO memory/stats indisponible", exc_info=True)

    report.active_count = count_keys(client, f"{ACTIVE_PREFIX}*", count=scan_count)
    report.previous_count = count_keys(client, f"{PREVIOUS_PREFIX}*", count=scan_count)
    report.revoked_count = count_keys(client, f"{REVOKED_PREFIX}*", count=scan_count)
    report.user_zset_count = count_keys(
        client, f"{USER_ZSET_PREFIX}*", count=scan_count
    )

    sampled = 0
    for pattern in (f"{ACTIVE_PREFIX}*", f"{PREVIOUS_PREFIX}*"):
        for key in scan_keys(client, pattern, count=scan_count):
            if sampled >= sample_limit:
                break
            sample = KeySample(key=key, key_type=str(client.type(key)))
            try:
                sample.pttl_ms = int(client.pttl(key))
            except Exception:
                sample.pttl_ms = -1
            try:
                sample.memory_bytes = int(
                    client.execute_command("MEMORY", "USAGE", key)
                )
            except Exception:
                sample.memory_bytes = None
            if sample.key_type == "string":
                sample.value = client.get(key)
            report.sampled.append(sample)
            if sample.memory_bytes:
                report.estimated_bytes += sample.memory_bytes
            sampled += 1
        if sampled >= sample_limit:
            break

    if report.sampled:
        known = [s for s in report.sampled if s.memory_bytes]
        if known:
            avg = sum(s.memory_bytes or 0 for s in known) / len(known)
            total_keys = report.active_count + report.previous_count
            report.estimated_bytes = int(avg * total_keys)
    return report


def recommend_maxmemory_bytes(
    estimated_payload_bytes: int,
    *,
    headroom_factor: float = 3.0,
    floor_mb: int = 64,
) -> int:
    """Propose maxmemory (bytes) avec headroom."""
    floor = floor_mb * 1024 * 1024
    needed = int(estimated_payload_bytes * headroom_factor)
    return max(floor, needed)


def _read_snapshot(src: Any, key: str) -> KeySnapshot | None:
    if not src.exists(key):
        return None
    key_type = str(src.type(key))
    pttl = int(src.pttl(key))
    captured = _monotonic()
    if key_type == "string":
        value = src.get(key)
        if value is None:
            return None
        return KeySnapshot(
            key=key,
            key_type=key_type,
            value=value,
            pttl_ms=pttl,
            captured_at_mono=captured,
        )
    if key_type == "zset":
        members = src.zrange(key, 0, -1, withscores=True)
        if not members:
            return None
        return KeySnapshot(
            key=key,
            key_type=key_type,
            zset_members=[(m, float(s)) for m, s in members],
            pttl_ms=pttl,
            captured_at_mono=captured,
        )
    return None


def _remaining_pttl_ms(snap: KeySnapshot, *, now_mono: float | None = None) -> int:
    """PTTL restant à l'instant d'écriture (échéance absolue, jamais prolongée)."""
    if snap.pttl_ms < 0:
        return snap.pttl_ms
    if snap.captured_at_mono is None:
        return snap.pttl_ms
    now = _monotonic() if now_mono is None else now_mono
    elapsed_ms = max(int((now - snap.captured_at_mono) * 1000), 0)
    return snap.pttl_ms - elapsed_ms


def _snapshot_still_valid(src: Any, snap: KeySnapshot) -> bool:
    """True si la clé source est encore identique au snapshot (anti-résurrection)."""
    if not src.exists(snap.key):
        return False
    if str(src.type(snap.key)) != snap.key_type:
        return False
    if snap.key_type == "string":
        return src.get(snap.key) == snap.value
    if snap.key_type == "zset":
        current = src.zrange(snap.key, 0, -1, withscores=True) or []
        cur_map = {m: float(s) for m, s in current}
        snap_map = {m: float(s) for m, s in (snap.zset_members or [])}
        return cur_map == snap_map
    return False


def _write_snapshot(dst: Any, snap: KeySnapshot) -> None:
    remaining = _remaining_pttl_ms(snap)
    if snap.key_type == "string":
        if remaining > 0:
            dst.psetex(snap.key, remaining, snap.value)
        elif remaining == -1:
            dst.set(snap.key, snap.value)
        else:
            raise ValueError("pttl expired")
        return
    if snap.key_type == "zset":
        dst.delete(snap.key)
        mapping = dict(snap.zset_members or [])
        if mapping:
            dst.zadd(snap.key, mapping)
        if remaining > 0:
            dst.pexpire(snap.key, remaining)
        elif remaining == -1:
            pass
        else:
            # ZSET snapshot expiré : ne pas laisser un set sans TTL / prolongé
            dst.delete(snap.key)
            raise ValueError("pttl expired")
        return
    raise ValueError(f"unsupported type {snap.key_type}")


def _undo_stale_write(dst: Any, snap: KeySnapshot) -> None:
    """Si on a écrit un snapshot devenu caduc, retirer la résurrection sur dst."""
    try:
        if snap.key_type == "string" and dst.get(snap.key) == snap.value:
            dst.delete(snap.key)
        elif snap.key_type == "zset":
            # Ne supprimer que si le zset dst correspond encore au snapshot stale
            current = dst.zrange(snap.key, 0, -1, withscores=True) or []
            cur_map = {m: float(s) for m, s in current}
            snap_map = {m: float(s) for m, s in (snap.zset_members or [])}
            if cur_map == snap_map:
                dst.delete(snap.key)
    except Exception:
        logger.exception("undo_stale_write failed key=%s", snap.key)


def copy_key_preserve_ttl(
    src: Any,
    dst: Any,
    key: str,
    *,
    dry_run: bool = True,
    race_hook: Callable[[str], None] | None = None,
    post_write_race_hook: Callable[[str], None] | None = None,
) -> str:
    """Copie une clé avec revalidation anti-résurrection + TTL absolu.

    Returns:
      copied | skipped_missing | dry_run | unsupported_type |
      skipped_stale_race | rolled_back_stale | skipped_expired
    """
    snap = _read_snapshot(src, key)
    if snap is None:
        return "skipped_missing"
    if dry_run:
        return "dry_run"

    # Point d'injection tests : mutation concurrente entre GET et SET
    if race_hook is not None:
        race_hook(key)

    if not _snapshot_still_valid(src, snap):
        return "skipped_stale_race"

    # Échéance absolue déjà écoulée → ne pas prolonger la vie sur dst
    remaining = _remaining_pttl_ms(snap)
    if snap.pttl_ms >= 0 and remaining <= 0:
        return "skipped_expired"

    try:
        _write_snapshot(dst, snap)
    except ValueError:
        return "skipped_expired"

    # Course après écriture dst (tests) : rollback si source caduque
    if post_write_race_hook is not None:
        post_write_race_hook(key)

    # Course tardive après écriture dst : ne pas laisser un R0 ressuscité
    if not _snapshot_still_valid(src, snap):
        _undo_stale_write(dst, snap)
        return "rolled_back_stale"

    return "copied"


def backfill_refresh_keys(
    legacy: Any,
    auth: Any,
    *,
    dry_run: bool = True,
    scan_count: int = 200,
    patterns: tuple[str, ...] = REFRESH_KEY_PATTERNS,
    race_hook: Callable[[str], None] | None = None,
    post_write_race_hook: Callable[[str], None] | None = None,
) -> dict[str, int]:
    """Backfill SCAN incrémental legacy → auth (anti-résurrection)."""
    stats = {
        "scanned": 0,
        "copied": 0,
        "dry_run": 0,
        "skipped_missing": 0,
        "skipped_stale_race": 0,
        "rolled_back_stale": 0,
        "skipped_expired": 0,
        "unsupported_type": 0,
        "errors": 0,
    }
    for pattern in patterns:
        for key in scan_keys(legacy, pattern, count=scan_count):
            stats["scanned"] += 1
            try:
                status = copy_key_preserve_ttl(
                    legacy,
                    auth,
                    key,
                    dry_run=dry_run,
                    race_hook=race_hook,
                    post_write_race_hook=post_write_race_hook,
                )
                stats[status] = stats.get(status, 0) + 1
            except Exception:
                stats["errors"] += 1
                logger.exception("backfill_error key=%s", key)
    return stats


def _ttl_close(
    a_ms: int, b_ms: int, *, tolerance_ms: int = TTL_DRIFT_MS_TOLERANCE
) -> bool:
    if a_ms < 0 and b_ms < 0:
        return a_ms == b_ms
    if a_ms < 0 or b_ms < 0:
        return False
    return abs(a_ms - b_ms) <= tolerance_ms


def compare_parity(
    legacy: Any,
    auth: Any,
    *,
    scan_count: int = 200,
    ttl_tolerance_ms: int = TTL_DRIFT_MS_TOLERANCE,
) -> ParityReport:
    """Compare CURRENT/PREVIOUS/REVOKED/USER_ZSET (read-only), incl. EXTRA + TTL + scores."""
    report = ParityReport()
    report.legacy_active = count_keys(legacy, f"{ACTIVE_PREFIX}*", count=scan_count)
    report.auth_active = count_keys(auth, f"{ACTIVE_PREFIX}*", count=scan_count)
    report.legacy_previous = count_keys(legacy, f"{PREVIOUS_PREFIX}*", count=scan_count)
    report.auth_previous = count_keys(auth, f"{PREVIOUS_PREFIX}*", count=scan_count)
    report.legacy_revoked = count_keys(legacy, f"{REVOKED_PREFIX}*", count=scan_count)
    report.auth_revoked = count_keys(auth, f"{REVOKED_PREFIX}*", count=scan_count)
    report.legacy_user_zset = count_keys(
        legacy, f"{USER_ZSET_PREFIX}*", count=scan_count
    )
    report.auth_user_zset = count_keys(auth, f"{USER_ZSET_PREFIX}*", count=scan_count)

    legacy_active_keys = set(scan_keys(legacy, f"{ACTIVE_PREFIX}*", count=scan_count))
    auth_active_keys = set(scan_keys(auth, f"{ACTIVE_PREFIX}*", count=scan_count))
    legacy_prev_keys = set(scan_keys(legacy, f"{PREVIOUS_PREFIX}*", count=scan_count))
    auth_prev_keys = set(scan_keys(auth, f"{PREVIOUS_PREFIX}*", count=scan_count))
    legacy_rev_keys = set(scan_keys(legacy, f"{REVOKED_PREFIX}*", count=scan_count))
    auth_rev_keys = set(scan_keys(auth, f"{REVOKED_PREFIX}*", count=scan_count))
    legacy_zset_keys = set(scan_keys(legacy, f"{USER_ZSET_PREFIX}*", count=scan_count))
    auth_zset_keys = set(scan_keys(auth, f"{USER_ZSET_PREFIX}*", count=scan_count))

    report.missing_current_in_auth = sorted(legacy_active_keys - auth_active_keys)
    report.extra_current_in_auth = sorted(auth_active_keys - legacy_active_keys)

    for key in sorted(legacy_active_keys & auth_active_keys):
        if legacy.get(key) != auth.get(key):
            report.mismatched_current.append(key)
            continue
        if not _ttl_close(
            int(legacy.pttl(key)),
            int(auth.pttl(key)),
            tolerance_ms=ttl_tolerance_ms,
        ):
            report.ttl_mismatched_current.append(key)

    for key in sorted(legacy_prev_keys - auth_prev_keys):
        if not legacy.exists(key):
            report.previous_expired_during_scan.append(key)
        else:
            report.missing_previous_in_auth.append(key)

    for key in sorted(auth_prev_keys - legacy_prev_keys):
        if not auth.exists(key):
            report.previous_expired_during_scan.append(key)
        else:
            report.extra_previous_in_auth.append(key)

    for key in sorted(legacy_prev_keys & auth_prev_keys):
        if legacy.get(key) != auth.get(key):
            report.mismatched_previous.append(key)
            continue
        if not _ttl_close(
            int(legacy.pttl(key)),
            int(auth.pttl(key)),
            tolerance_ms=ttl_tolerance_ms,
        ):
            report.ttl_mismatched_previous.append(key)

    report.missing_revoked_in_auth = sorted(legacy_rev_keys - auth_rev_keys)
    report.extra_revoked_in_auth = sorted(auth_rev_keys - legacy_rev_keys)
    for key in sorted(legacy_rev_keys & auth_rev_keys):
        if legacy.get(key) != auth.get(key):
            report.mismatched_revoked.append(key)
            continue
        if not _ttl_close(
            int(legacy.pttl(key)),
            int(auth.pttl(key)),
            tolerance_ms=ttl_tolerance_ms,
        ):
            report.ttl_mismatched_revoked.append(key)

    report.missing_user_zset_in_auth = sorted(legacy_zset_keys - auth_zset_keys)
    report.extra_user_zset_in_auth = sorted(auth_zset_keys - legacy_zset_keys)
    for key in sorted(legacy_zset_keys & auth_zset_keys):
        leg_members = {
            str(m): float(s)
            for m, s in (legacy.zrange(key, 0, -1, withscores=True) or [])
        }
        auth_members = {
            str(m): float(s)
            for m, s in (auth.zrange(key, 0, -1, withscores=True) or [])
        }
        if set(leg_members.keys()) != set(auth_members.keys()):
            report.mismatched_user_zset_members.append(key)
            continue
        # Scores exacts : ordre FIFO (limit_active_tokens) doit être identique.
        if any(leg_members[m] != auth_members[m] for m in leg_members):
            report.mismatched_user_zset_scores.append(key)
            continue
        if not _ttl_close(
            int(legacy.pttl(key)),
            int(auth.pttl(key)),
            tolerance_ms=ttl_tolerance_ms,
        ):
            report.ttl_mismatched_user_zset.append(key)

    return report


def parity_gate_pass(report: ParityReport) -> bool:
    """Critère Phase E avant auth_primary — values + TTL + scores ZSET stricts."""
    unexplained_missing_prev = [
        k
        for k in report.missing_previous_in_auth
        if k not in report.previous_expired_during_scan
    ]
    unexplained_extra_prev = [
        k
        for k in report.extra_previous_in_auth
        if k not in report.previous_expired_during_scan
    ]
    return (
        len(report.missing_current_in_auth) == 0
        and len(report.extra_current_in_auth) == 0
        and len(report.mismatched_current) == 0
        and len(report.ttl_mismatched_current) == 0
        and len(unexplained_missing_prev) == 0
        and len(unexplained_extra_prev) == 0
        and len(report.mismatched_previous) == 0
        and len(report.ttl_mismatched_previous) == 0
        and len(report.missing_revoked_in_auth) == 0
        and len(report.extra_revoked_in_auth) == 0
        and len(report.mismatched_revoked) == 0
        and len(report.ttl_mismatched_revoked) == 0
        and len(report.missing_user_zset_in_auth) == 0
        and len(report.extra_user_zset_in_auth) == 0
        and len(report.mismatched_user_zset_members) == 0
        and len(report.mismatched_user_zset_scores) == 0
        and len(report.ttl_mismatched_user_zset) == 0
    )
