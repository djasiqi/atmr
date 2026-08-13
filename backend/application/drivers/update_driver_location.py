from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Callable

from services.geolocation.driver_location_dedup import should_skip_location_ingest
from services.monitoring.driver_location_metrics import inc_dedup_skipped
from services.tracking.mission_tracking_firewall import (
    evaluate_mission_live_admission,
    record_admission_metrics,
)
from services.tracking.time_contract import (
    TrackingInstantError,
    format_tracking_instant_utc_z,
    parse_tracking_instant_strict,
)
from services.tracking.tracking_ingress_contract import (
    TrackingIngressEnvelope,
    build_tracking_ingress_envelope,
    evaluate_event_contract,
)

LAT_THRESHOLD = 90.0
LON_THRESHOLD = 180.0


@dataclass(frozen=True, slots=True)
class UpdateDriverLocationCommand:
    driver_id: int
    latitude: float
    longitude: float
    speed: float | None = None
    heading: float | None = None
    accuracy: float | None = None
    ts: str | None = None
    location_mode: str | None = None
    recorded_at: str | None = None
    sent_at: str | None = None
    is_background: bool = False
    mission_id: int | None = None
    metrics_transport: str = "http"
    location_event_id: str | None = None
    emit_geofence: bool = True
    company_id: int | None = None
    # P0-A/D : enveloppe brute (présence avant defaults). Si None, reconstruite.
    ingress_envelope: TrackingIngressEnvelope | None = None
    tracking_session_id: str | None = None
    session_generation: int | None = None
    sequence_id: int | None = None
    capture_id: str | None = None
    defer_canonical_promotion: bool = False


@dataclass(frozen=True, slots=True)
class UpdateDriverLocationResult:
    snapped_lat: float
    snapped_lon: float
    source: str
    geofence_events: list[str]
    accept_status: str
    accept_reason: str
    received_at: str | None
    dedup_skipped: bool = False
    dedup_reason: str | None = None
    canonical_updated: bool = False
    db_persisted: bool | None = None
    live_eligible: bool = True
    canonical_eligible: bool = True
    admission_reason: str = ""


class UpdateDriverLocationUseCase:
    """Use-case Application: mise à jour de localisation chauffeur.

    Ce use-case encapsule la validation d'entrée et délègue l'exécution à une
    dépendance injectée (typiquement `LocationService.update_driver_location`).
    INV-P0-3 : EventContract + MissionFirewall avant dedup.
    """

    def __init__(
        self,
        *,
        update_location_fn: Callable[..., object],
        now_utc_fn: Callable[[], datetime] | None = None,
    ) -> None:
        super().__init__()
        self._update_location = update_location_fn
        self._now_utc = now_utc_fn or (lambda: datetime.now(UTC))

    def execute(self, cmd: UpdateDriverLocationCommand) -> UpdateDriverLocationResult:
        if not (-LAT_THRESHOLD <= cmd.latitude <= LAT_THRESHOLD) or not (
            -LON_THRESHOLD <= cmd.longitude <= LON_THRESHOLD
        ):
            raise ValueError("Coordinates out of valid range")

        envelope = cmd.ingress_envelope or build_tracking_ingress_envelope(
            {
                "latitude": cmd.latitude,
                "longitude": cmd.longitude,
                "location_event_id": cmd.location_event_id,
                "recorded_at": cmd.recorded_at,
                "mission_id": cmd.mission_id,
                "location_mode": cmd.location_mode,
                "tracking_session_id": cmd.tracking_session_id,
                "session_generation": cmd.session_generation,
                "sequence_id": cmd.sequence_id,
                "capture_id": cmd.capture_id,
            },
            transport=cmd.metrics_transport,
        )
        contract = evaluate_event_contract(envelope)
        admission = evaluate_mission_live_admission(
            driver_id=cmd.driver_id,
            envelope=envelope,
            contract=contract,
        )
        record_admission_metrics(admission, transport=cmd.metrics_transport)

        timestamp = self._parse_ts(cmd.ts)
        recorded_dt = self._parse_ts(cmd.recorded_at) if cmd.recorded_at else timestamp

        # INV-P0-3 : dedup APRÈS firewall (évite claim event_id sur stale_mission)
        skip, skip_reason = should_skip_location_ingest(
            cmd.driver_id,
            cmd.latitude,
            cmd.longitude,
            recorded_dt,
            cmd.location_mode or "mission_live",
            cmd.location_event_id,
        )
        if skip and skip_reason:
            inc_dedup_skipped(
                reason=skip_reason,
                location_mode=cmd.location_mode or "mission_live",
                transport=cmd.metrics_transport,
            )
            return UpdateDriverLocationResult(
                snapped_lat=cmd.latitude,
                snapped_lon=cmd.longitude,
                source="raw",
                geofence_events=[],
                accept_status="skipped",
                accept_reason=skip_reason,
                received_at=None,
                dedup_skipped=True,
                dedup_reason=skip_reason,
                canonical_updated=False,
                db_persisted=None,
                live_eligible=admission.live_eligible,
                canonical_eligible=admission.canonical_eligible,
                admission_reason=admission.reason,
            )

        # On garde la signature la plus permissive possible (typage runtime via attrs).
        res = self._update_location(
            driver_id=cmd.driver_id,
            latitude=cmd.latitude,
            longitude=cmd.longitude,
            speed=cmd.speed,
            heading=cmd.heading,
            accuracy=cmd.accuracy,
            source="gps",
            timestamp=timestamp,
            location_mode=cmd.location_mode or "mission_live",
            recorded_at=self._parse_ts(cmd.recorded_at)
            if cmd.recorded_at
            else timestamp,
            sent_at=self._parse_ts(cmd.sent_at) if cmd.sent_at else self._now_utc(),
            is_background=bool(cmd.is_background),
            mission_id=cmd.mission_id,
            transport=cmd.metrics_transport,
            live_eligible=admission.live_eligible,
            canonical_eligible=admission.canonical_eligible,
            admission_reason=admission.reason,
            capture_id=cmd.capture_id or envelope.capture_id,
            location_event_id=cmd.location_event_id,
            tracking_session_id=cmd.tracking_session_id,
            session_generation=cmd.session_generation,
            sequence_id=cmd.sequence_id,
            defer_canonical_promotion=cmd.defer_canonical_promotion,
        )

        snapped_lat = getattr(res, "snapped_lat", cmd.latitude)
        snapped_lon = getattr(res, "snapped_lon", cmd.longitude)
        source = getattr(res, "source", "raw")
        geofence_events = list(getattr(res, "geofence_events", []) or [])
        accept_status = str(
            getattr(res, "accept_status", "accepted_observability_only")
        )
        accept_reason = str(getattr(res, "accept_reason", ""))
        received_at = getattr(res, "received_at", None)
        if received_at is None:
            received_at_str = None
        elif isinstance(received_at, datetime):
            received_at_str = format_tracking_instant_utc_z(received_at)
        else:
            # LocationService renvoie déjà une chaîne …Z après TIME-1
            received_at_str = str(received_at)

        if cmd.emit_geofence and geofence_events:
            from services.tracking.geofence_emit import emit_driver_geofence_events

            emit_driver_geofence_events(
                driver_id=cmd.driver_id,
                company_id=cmd.company_id,
                geofence_events=geofence_events,
            )

        canonical_updated = bool(getattr(res, "canonical_updated", False))
        db_persisted_raw = getattr(res, "db_persisted", None)
        db_persisted: bool | None
        db_persisted = None if db_persisted_raw is None else bool(db_persisted_raw)

        live_eligible = bool(getattr(res, "live_eligible", admission.live_eligible))
        # INV-P0-2 : jamais remonter live_eligible à true
        if not admission.live_eligible:
            live_eligible = False
        canonical_eligible = bool(
            getattr(res, "canonical_eligible", admission.canonical_eligible)
        )
        if not admission.canonical_eligible:
            canonical_eligible = False

        return UpdateDriverLocationResult(
            snapped_lat=float(snapped_lat),
            snapped_lon=float(snapped_lon),
            source=str(source),
            geofence_events=geofence_events,
            accept_status=accept_status,
            accept_reason=accept_reason,
            received_at=received_at_str,
            dedup_skipped=False,
            dedup_reason=None,
            canonical_updated=canonical_updated and canonical_eligible,
            db_persisted=db_persisted,
            live_eligible=live_eligible,
            canonical_eligible=canonical_eligible,
            admission_reason=admission.reason,
        )

    def _parse_ts(self, ts: str | None) -> datetime:
        """Parse un instant tracking. Absent → now UTC ; naïf/invalide → REJET."""
        if ts is None or (isinstance(ts, str) and not ts.strip()):
            return self._now_utc()
        try:
            return parse_tracking_instant_strict(ts)
        except TrackingInstantError as exc:
            raise ValueError(str(exc)) from exc
