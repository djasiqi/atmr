# Auth multi-appareils — contrat P0/P1

## Capabilities

Réponse login / device-sessions :

```json
{
  "auth_contract_version": "mobile-device-session-v1",
  "capabilities": {
    "durable_device_session": true,
    "device_session_management": true,
    "device_session_replace": true,
    "provisional_session_confirmation": true
  }
}
```

Le mobile n'affiche le remplacement d'appareil que si `device_session_replace` est **strictement** `true` **et** qu'un `resolution_token` est présent dans le 409.

## Flags runtime (defaults `false`)

| Variable | Défaut | Effet |
|----------|--------|--------|
| `MOBILE_DEVICE_SESSION_REPLACE_ENABLED` | `false` | Replace + `resolution_token` / capability |
| `MOBILE_DEVICE_PROVISIONAL_CONFIRMATION_ENABLED` | `false` | Sessions provisional + confirm |

Rollout recommandé : P0 (tous false) → canary replace → P1 complet.

## Headers appareil

| Header | Contenu |
|--------|---------|
| `X-Device-ID` | installation_id stable |
| `X-Device-Name` | nom humain OS (jamais le nom d'app) |
| `X-Device-Model` | modèle matériel |
| `X-Device-Manufacturer` | fabricant |
| `X-Device-Type` | phone / tablet / … |
| `X-Client-Platform` | ios / android (+ `X-Platform` compat N-1) |
| `X-OS-Version` | version OS |
| `X-App-Version` / `X-App-Build` | version / build native |

## Flux 409 / replace (P1)

1. Login valide + quota atteint → `409` + `sessions` + `resolution_token` (snapshot `allowed_session_ids`).
2. `POST /auth/device-sessions/replace` : claim Redis → transaction PG unique (`commit=False` tokens) → COMMIT → `publish_session_revoked` (cible + reaped) + consume challenge.
3. Nouvelle session provisional → SecureStore OK → `pending_session_confirmation` → `POST .../confirm` (idempotent). Refresh / session-resume confirment aussi implicitement.

## Ops

```bash
docker compose exec atmr_api python scripts/report_mobile_device_sessions.py --email user@example.com
```

Lecture seule — aucune révocation automatique.
