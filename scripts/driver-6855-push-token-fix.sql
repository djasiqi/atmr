-- Runbook P1 — driver 6855 push tokens (Postgres prod, après P0 stable)
UPDATE device_tokens SET platform = 'android', updated_at = NOW() WHERE id = 23 AND provider = 'fcm';
SELECT id, provider, platform, is_active, device_id, last_push_success_at, last_push_attempt_at, updated_at, created_at FROM device_tokens WHERE driver_id = 6855 AND is_active = true ORDER BY updated_at DESC;
