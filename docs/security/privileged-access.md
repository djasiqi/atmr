# Accès privilégié LIRIE

Inventaire des comptes pour lesquels l’enrollment MFA est **obligatoire**.
Le helper unique est [`backend/security/privileged_access.py`](../../backend/security/privileged_access.py)
(`requires_mfa_enrollment`).

## Règle

Les privilèges se jugent sur les **capacités réelles**, pas sur le seul nom
`UserRole.COMPANY` ou `UserRole.INSTITUTION`.

```text
password OK
│
├─ totp_enabled = true   (tout rôle)
│     → temp_token purpose=2fa_challenge
│     → aucun access/refresh métier
│     → /totp/challenge
│     → JWT métier
│
├─ totp_enabled = false + privilegié
│     → temp_token purpose=mfa_enroll
│     → /totp/setup + /totp/verify
│     → challenge
│     → JWT métier
│
└─ totp_enabled = false + non privilegié
      → login normal
```

Un JWT avec `purpose` ∈ {`2fa_challenge`, `mfa_enroll`} est rejeté par
`jwt_required` (blocklist). Il ne peut pas appeler l’API métier.

## Qui est privilegié

| Capacité | Enrollment forcé | Notes |
|----------|------------------|--------|
| `UserRole.ADMIN` (super-admin LIRIE) | Oui | Cross-tenant autorisé **et audité** (P0-02) |
| `InstitutionRole.ADMIN` | Oui | Admin d’institution, pas tout `UserRole.INSTITUTION` |
| Owner entreprise (`Company.user_id`) | Oui | Admin tenant, pas tout `UserRole.COMPANY` |
| `InstitutionRole.BILLING` | Non | Pas d’admin utilisateurs / sécurité |
| `REQUESTER` / `READER` / `RECEPTION` / `CURATOR` | Non | Opérateurs ordinaires |
| `DRIVER` / `CLIENT` | Non | Jamais d’enrollment forcé |

## Sessions

Les comptes privilegiés ont un access token plafonné à **15 minutes**, y compris
sur mobile.

## Désactivation TOTP

- Compte privilegié : `/totp/disable` refuse (403 `mfa_disable_forbidden`).
- Break-glass : `/totp/admin-disable` réservé au super-admin LIRIE **déjà
  protégé par TOTP**, session fresh, code TOTP/recovery + motif obligatoire,
  audit `totp_admin_disabled`. Un JWT mot de passe seul est refusé.
- Un `temp_token` (`2fa_challenge` / `mfa_enroll`) est rejeté par
  `jwt_required`, le refresh, et l’auth websocket. TTL 5 min (challenge)
  / 15 min (enroll). Seul `/totp/challenge` (resp. setup/verify) le
  consomme.
- Pas d’usage quotidien du super-admin pour les opérations courantes.

## Feature flag

`SECURITY_2FA_ENABLED=true` en production. Le challenge TOTP et l’enrollment
privilegié restent disponibles même si le flag est off (pas de lock-out).
L’enrollment volontaire (compte non privilegié déjà connecté) reste derrière
le flag.
