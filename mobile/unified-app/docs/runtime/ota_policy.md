# OTA Policy (Phase 2)

## Runtime Parameters

| Key | Example | Meaning |
|---|---|---|
| `minimum_supported_version` | `1.4.2` | Block when app version is lower |
| `recommended_version` | `1.6.0` | Show non-blocking update prompt |
| `kill_switch` | `false` | Global emergency disable for update gate logic |
| `blocking_scope` | `global` | `driver`, `company`, or `global` |

## Decision Rules

- Block user access only when version is under minimum and scope matches active context.
- Recommended gate is informational and never blocks navigation.
- `kill_switch=true` bypasses blocking logic and falls back to recommendation only.

## Release Governance

- Policy changes must be approved by Release Owner + Mobile Platform.
- Every policy update requires a dry-run on staging with mocked app versions.
