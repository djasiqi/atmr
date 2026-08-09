# Phase 2 Runtime Rollback

## Rollback Objective

Disable high-risk Phase 2 runtime features in less than 15 minutes.

## Feature Rollback Table

| Feature | Rollback Action |
|---|---|
| notification channels | fallback to default expo channels |
| silent sync | disable silent handler |
| deep links | fallback to legacy parser only |
| version gate | disable blocking scope |
| mission bar Android | disable foreground service notification |
| Live Activity iOS | disable activity manager bridge |
| transfer flow | feature flag off |

## Verification Steps

1. Confirm flags propagated on staging.
2. Validate login still accessible in all contexts.
3. Validate mission and chat routes still open with legacy parser.
4. Capture telemetry snapshot and attach to committee evidence.
