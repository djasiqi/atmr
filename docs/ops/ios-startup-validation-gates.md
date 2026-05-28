# iOS startup — checklist validation avant reprise realtime

Gate minimale à valider sur **build 57+** après **clean install**.

## Prérequis appareil

- [ ] App **Lirie** (`ch.liri.operations`), pas variante legacy
- [ ] Build **57+** confirmé dans TestFlight / écran À propos
- [ ] Clean install effectuée (désinstaller → réinstaller)
- [ ] Identifiants retapés manuellement (pas d'autofill erroné)

## Gate auth (obligatoire)

| # | Vérification | Attendu | OK |
| - | ------------ | ------- | -- |
| 1 | `GET /api/v1/auth/csrf-token` | 200 | [ ] |
| 2 | `POST /api/v1/app/version-check` | 200, `status: OK` | [ ] |
| 3 | `POST /api/v1/auth/login` | 200 | [ ] |
| 4 | `GET /api/v1/auth/bootstrap` | 200 | [ ] |
| 5 | Aucun crash startup (splash → login ou dashboard) | pass | [ ] |

## Vérification kill-switch backend (ops)

```bash
curl -s https://api.lirie.ch/api/feature-flags/runtime-status
```

Attendu dans la réponse :

```json
{
  "mobile_startup": {
    "ios_startup_fatal_recovery_disabled": true
  }
}
```

> Activer avec `IOS_STARTUP_FATAL_RECOVERY_DISABLED=true` **après** déploiement du hotfix mobile qui consomme le flag.

## Autorisation reprise Test #1 realtime

**Autorisé uniquement si** toutes les cases gate auth sont cochées.

Ensuite seulement :

- recovery staging dogfood
- reconnect
- canary ws-service
- tests D3

## En cas d'échec

| Symptôme | Action |
| -------- | ------ |
| csrf/version OK, pas de login | Classer startup crash — voir [ios-startup-hotfix-qa.md](./ios-startup-hotfix-qa.md) |
| login 401 | Vérifier identifiants / mauvaise app |
| bootstrap absent | Vérifier token / session corrompue → clean install |
| build 49 | **Interdit** — mettre à jour vers 57+ |

## Références

- [ios-startup-hotfix-qa.md](./ios-startup-hotfix-qa.md) — règles QA et triage 30 min
- [ios-startup-hotfix-mobile-ticket.md](./ios-startup-hotfix-mobile-ticket.md) — ticket hotfix mobile
