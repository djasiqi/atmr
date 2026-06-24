"""Historique heartbeat santé device chauffeur (tracking readiness / diagnostics)."""

from __future__ import annotations

from datetime import datetime

from ext import db


class DriverDeviceHealthEvent(db.Model):
    """Snapshot périodique device-health remonté par l'app mobile."""

    __tablename__ = "driver_device_health_events"

    id = db.Column(db.Integer, primary_key=True)
    driver_id = db.Column(
        db.Integer,
        db.ForeignKey("driver.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    recorded_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.utcnow(),
        index=True,
    )
    manufacturer = db.Column(db.String(64), nullable=True)
    model = db.Column(db.String(128), nullable=True)
    platform = db.Column(db.String(16), nullable=True)
    battery_optimized = db.Column(db.Boolean, nullable=True)
    location_permission = db.Column(db.String(32), nullable=True)
    notifications_enabled = db.Column(db.Boolean, nullable=True)
    tracking_active = db.Column(db.Boolean, nullable=True)
    app_state = db.Column(db.String(32), nullable=True)
    last_fix_age_seconds = db.Column(db.Integer, nullable=True)
    constraint_reason = db.Column(db.String(64), nullable=True)
    fgs_running = db.Column(db.Boolean, nullable=True)
    trigger_reason = db.Column(db.String(128), nullable=True)
    native_start_phase = db.Column(db.String(64), nullable=True)
    native_start_error = db.Column(db.String(512), nullable=True)
    native_task_defined = db.Column(db.Boolean, nullable=True)
    native_started_before = db.Column(db.Boolean, nullable=True)
    native_started_after = db.Column(db.Boolean, nullable=True)
    # Diagnostic Lot 1 (versions + signaux background iOS)
    app_version = db.Column(db.String(32), nullable=True)
    os_version = db.Column(db.String(32), nullable=True)
    native_last_fix_age_seconds = db.Column(db.Integer, nullable=True)
    native_task_running = db.Column(db.Boolean, nullable=True)
    ios_accuracy_authorization = db.Column(db.String(16), nullable=True)
    ios_low_power_mode = db.Column(db.Boolean, nullable=True)
    ios_background_refresh_status = db.Column(db.String(16), nullable=True)

    driver = db.relationship("Driver", backref=db.backref("device_health_events", lazy="dynamic"))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "driver_id": self.driver_id,
            "recorded_at": self.recorded_at.isoformat() if self.recorded_at else None,
            "manufacturer": self.manufacturer,
            "model": self.model,
            "platform": self.platform,
            "battery_optimized": self.battery_optimized,
            "location_permission": self.location_permission,
            "notifications_enabled": self.notifications_enabled,
            "tracking_active": self.tracking_active,
            "app_state": self.app_state,
            "last_fix_age_seconds": self.last_fix_age_seconds,
            "constraint_reason": self.constraint_reason,
            "fgs_running": self.fgs_running,
            "trigger_reason": self.trigger_reason,
            "native_start_phase": self.native_start_phase,
            "native_start_error": self.native_start_error,
            "native_task_defined": self.native_task_defined,
            "native_started_before": self.native_started_before,
            "native_started_after": self.native_started_after,
            "app_version": self.app_version,
            "os_version": self.os_version,
            "native_last_fix_age_seconds": self.native_last_fix_age_seconds,
            "native_task_running": self.native_task_running,
            "ios_accuracy_authorization": self.ios_accuracy_authorization,
            "ios_low_power_mode": self.ios_low_power_mode,
            "ios_background_refresh_status": self.ios_background_refresh_status,
        }
