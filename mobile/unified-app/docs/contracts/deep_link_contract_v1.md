# Deep Link Contract v1

## Canonical Routes

- `atmr://mission/{mission_id}`
- `atmr://chat/{thread_id}`
- `atmr://transfer/{ride_id}`
- `atmr://dashboard`
- `atmr://rides?filter=urgent`
- `atmr://quick-action?missionId={id}&action={accept|reject|start|complete}`

## Compatibility Routes

- `atmr://booking/{mission_id}`
- `atmr://booking/{mission_id}/{accept|reject|start|complete}`
- `atmr://bookings`

## Parsing Rules

- Accept both `missionId` and `bookingId` query keys for quick actions.
- Route names are case-insensitive, but emitted canonical route is lowercase.
- Invalid identifiers must be rejected without navigation side effects.

## Ownership

- Product: route semantics
- Backend: payload schema and migration window
- Mobile Platform: parser/runtime routing implementation
