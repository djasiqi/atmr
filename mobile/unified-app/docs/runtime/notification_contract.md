# Notification Contract (Phase 2)

## Objective

Provide a single source of truth for `event -> channel -> priority -> behavior`.

## Android Channels

- `mission_updates`
- `chat`
- `urgent`
- `silent`
- `lock-screen`

## Event Mapping

| Event | Channel | Priority | Silent | Action |
|---|---|---|---|---|
| `mission_assigned` | `mission_updates` | `HIGH` | no | open mission |
| `mission_updated` | `mission_updates` | `HIGH` | no | open mission |
| `mission_cancelled` | `urgent` | `MAX` | no | open mission |
| `mission_reassigned` | `mission_updates` | `HIGH` | no | open mission |
| `chat_message` | `chat` | `DEFAULT` | no | open chat |
| `mission_refresh` | `silent` | `LOW` | yes | resync only |
| `lockscreen_hint` | `lock-screen` | `HIGH` | no | open mission |

## Runtime Rules

- Every incoming push must be normalized into a known event.
- Unknown events fallback to `mission_updates`.
- Silent payloads must never force foreground UI navigation.
- Quick action payloads must include `missionId` and `action`.
