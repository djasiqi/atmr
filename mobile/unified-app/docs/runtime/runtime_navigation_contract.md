# Runtime Navigation Contract

## Entry Points

- Cold start (normal app launch)
- Push open
- Quick action open
- Deep link open
- OTA reopen after update gate decision

## Routing Guarantees

- Deep link parser resolves to a single deterministic route.
- Quick actions route to `quick-action` screen with stable params.
- Silent notifications trigger data resync without forced navigation.
- Company context deep links route to company surfaces only.

## Validation Scope

- Android and iOS
- Foreground, background, and killed-app scenarios
- Driver and company contexts
