export type DriverTelemetryEventName =
  | "auth.refresh.failure"
  | "auth.refresh.endpoint_used"
  | "auth.bootstrap.failure"
  | "realtime.socket.disconnect"
  | "realtime.socket.reconnect"
  | "realtime.auth.retry"
  | "realtime.auth.exhausted"
  | "realtime.transport.state"
  | "realtime.transport.authority"
  | "realtime.degraded.entered"
  | "realtime.degraded.exited"
  | "realtime.reconnect.cap_reached"
  | "realtime.transport.mismatch"
  | "tracking.permission.denied"
  | "tracking.queue.enqueued"
  | "tracking.queue.flush"
  | "tracking.queue.dropped"
  | "tracking.queue.expired"
  | "tracking.ingest.ack"
  | "tracking.queue.compacted"
  | "tracking.socket.emit_without_backend_ack"
  | "tracking.bridge.health"
  | "tracking.stale_fallback.timeout"
  | "tracking.batch.ack"
  | "tracking.send.backoff"
  | "tracking.send.failure"
  | "tracking.send.recovered"
  | "tracking.watch.started"
  | "tracking.watch.unavailable"
  | "tracking.watch.restarted"
  | "tracking.watch.restart.exhausted"
  | "tracking.remote_kick.received"
  | "tracking.recovery.step"
  | "tracking.background.task.tick"
  | "tracking.background.task.registered"
  | "tracking.background.task.unavailable"
  | "tracking.background.task.error"
  | "tracking.background.task.skipped"
  | "tracking.background.task.flush"
  | "tracking.background.task.started"
  | "tracking.background.task.stopped"
  | "tracking.background.task.registration_status"
  | "tracking.background.task_invoked"
  | "tracking.background.start_requested"
  | "tracking.background.start_success"
  | "tracking.background.start_failed"
  | "tracking.background.start_deferred"
  | "realtime.event.ignored"
  | "realtime.event.sequence_gap"
  | "realtime.polling.failure"
  | "realtime.polling.full_refetch"
  | "realtime.polling.skipped"
  | "realtime.reconcile.since"
  | "realtime.reconcile.full_refetch_guarded"
  | "realtime.mission.freshness"
  | "driver.network.tick"
  | "driver.network.tick.skipped"
  | "driver.network.wake"
  | "driver.network.profile"
  | "driver.http.timeout"
  | "driver.http.circuit_breaker"
  | "driver.sync_engine.flush.skipped"
  | "driver.foreground.resume.resync"
  | "driver.foreground.resume.resync.coalesced"
  | "realtime.drift.detected"
  | "transition.queue.retry"
  | "transition.queue.flush"
  | "transition.queue.failure"
  | "push.token.registered"
  | "push.token.refresh"
  | "push.notification.route_failed"
  | "push.notification.route_timeout"
  | "push.notification.received"
  | "push.notification.opened"
  | "push.notification.ignored"
  | "push.notification.filtered"
  | "push.notification.suppressed"
  | "push.notification.dedup_hit"
  | "push.notification.navigation"
  | "push.notification.sound_played"
  | "push.notification.resync_started"
  | "push.notification.resync_completed"
  | "push.notification.silent_sync"
  | "push.fcm.background_handler_no_callback"
  | "push.quick_action.dispatch"
  | "push.quick_action.success"
  | "push.quick_action.failure"
  | "push.channels.setup_failed"
  | "push.actions.setup_failed"
  | "push.grouping.setup_failed"
  | "push.mission_bar.setup_failed"
  | "push.token.permission_denied"
  | "push.permission.request_failed"
  | "push.display.schedule_failed"
  | "push.company.register_gate"
  | "driver.runtime.resume.start"
  | "driver.runtime.resume.success"
  | "driver.runtime.resume.failure"
  | "driver.runtime.heartbeat"
  | "driver.runtime.resync"
  | "driver.availability.updated"
  | "driver.gate.unified_evaluated"
  | "driver.runtime.reconcile"
  | "driver.runtime.reconcile.failure"
  | "driver.sync_engine.heartbeat"
  | "driver.battery_optimization.unavailable"
  | "tracking.battery_optimization.detected"
  | "tracking.battery_optimization.user_action"
  | "tracking.battery_optimization.exempted"
  | "tracking.battery_optimization.check_failed"
  | "tracking.device_health.sent"
  | "tracking.device_health.send_failed"
  | "tracking.device_health.send_skipped"
  | "tracking.health_monitor.constraint_changed"
  | "tracking.transition_blocked_permission"
  | "tracking.permission_revoked_during_mission"
  | "tracking.fgs_stopped_during_mission"
  | "tracking.stale_fix_during_mission"
  | "tracking.mission_live_guard.disclosure_shown"
  | "tracking.mission_live_guard.permission_requested"
  | "driver.biometric.unavailable"
  | "driver.push.fcm.token"
  | "driver.push.fcm.get_token_start"
  | "driver.push.fcm.unavailable"
  | "push.token.expo_fallback_unreliable"
  | "driver_push.bridge_mounted"
  | "driver_push.disclosure_blocked"
  | "driver_push.permission_blocked"
  | "driver_push.get_token_failed"
  | "driver_push.token_acquired"
  | "driver_push.register_success"
  | "ota.auto_reload.pending_detected"
  | "ota.auto_reload.deferred"
  | "ota.auto_reload.start"
  | "ota.auto_reload.failed"
  | "ota.auto_reload.applied"
  | "driver.mission_bar.android.unavailable"
  | "driver.mission_bar.background_event"
  | "driver.mission_bar.background.unavailable"
  | "driver.mission_bar.ios.unavailable"
  | "driver.mission_bar.ios.live_activity_unavailable"
  | "company.fleet.directions.failed"
  | "company.fleet.directions.exception";

export type DriverTelemetryPayload = {
  source: string;
  context_id?: string | null;
  driver_id?: string | null;
  mission_id?: number | null;
  app_state?: string | null;
  network_state?: string | null;
  reason?: string | null;
  retry_count?: number;
  [key: string]: unknown;
};

type DriverTelemetrySink = (
  event: DriverTelemetryEventName,
  payload: DriverTelemetryPayload
) => void;

const defaultSink: DriverTelemetrySink = (event, payload) => {
  console.info(`[driver-telemetry] ${event}`, payload);
};

let sink: DriverTelemetrySink = defaultSink;

export function emitDriverTelemetry(
  event: DriverTelemetryEventName,
  payload: DriverTelemetryPayload
) {
  sink(event, payload);
}

export function setDriverTelemetrySink(customSink: DriverTelemetrySink | null) {
  sink = customSink ?? defaultSink;
}

export function setDriverTelemetrySinkForTests(customSink: DriverTelemetrySink | null) {
  setDriverTelemetrySink(customSink);
}
