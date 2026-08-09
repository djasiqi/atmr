# Quick Actions Contract

## Supported Actions

- `accept`
- `reject`
- `start`
- `complete`

## Payload Contract

Required fields:

- `missionId` or `bookingId`
- `action`

Optional fields:

- `deep_link`
- `event_id`
- `type`

## Runtime Behavior

- Normalize payload and route to `/(quick-action)` when action exists.
- Reject unknown actions and report telemetry.
- Keep backward compatibility with legacy booking payload names.
