# Phase 2 Runtime Observability

## Required KPIs

| KPI | Source |
|---|---|
| `notification_open_rate` | mobile telemetry |
| `silent_sync_trigger_rate` | runtime logs |
| `deep_link_success_rate` | navigation events |
| `LiveActivity_active_sessions` | iOS runtime |
| `transfer_conflict_rate` | backend API |
| `attachment_upload_fail_rate` | chat API |

## Alerting Baseline

- Alert on sustained increase in `transfer_conflict_rate`.
- Alert when `deep_link_success_rate` drops below 99%.
- Alert on `attachment_upload_fail_rate` above accepted threshold.

## Committee Dashboard

Dashboard must include:

- 24h trend
- 7d trend
- rollout cohort filter
- platform split (android/ios)
