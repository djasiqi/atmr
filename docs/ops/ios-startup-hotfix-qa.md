# iOS startup crash — règles QA et triage

> **Contexte incident:** crash volontaire au démarrage (`ErrorRecovery.crash()` → `StartupProcedure.throwException` → `SIGABRT`) sur certains builds iOS, indépendant d'une panne backend.

## Positionnement

| Composant | Rôle |
| --------- | ---- |
| Hotfix mobile | **Obligatoire** — supprime le crash fatal (`NSException.raise`/abort) |
| Kill-switch backend (`IOS_STARTUP_FATAL_RECOVERY_DISABLED`) | **Protection builds futurs** — ne corrige pas un build qui ne lit pas le flag |

## Règle QA prioritaire (immédiate)

1. **Build 49 interdit** pour tous les tests (crash startup confirmé).
2. **Build 57+ uniquement** — build actuel autorisé : **57**.
3. **Clean install obligatoire** sur appareil impacté (purge Keychain / état local recovery).
4. Vérifier dans TestFlight que le testeur a bien installé la **dernière build**.
5. Vérifier le nom de l'app : **Lirie** (`ch.liri.operations`), pas une variante legacy.

## Runbook triage (30 min)

### 0–5 min — Gel des tests

- Stopper les tests realtime (D3, recovery, canary) sur tout appareil en build 49.
- Communiquer à QA : « build 57+ only ».

### 5–10 min — État appareil

Collecter sur l'iPhone de test :

- Nom exact de l'app (icône)
- Build affiché (49, 57, autre)
- Version app (ex. 1.0.5)
- Modèle / OS (ex. iPhone 12, iOS 26.x)
- Heure exacte du test (UTC+2)

Procédure : **désinstaller → réinstaller** depuis TestFlight.

### 10–20 min — Scénario auth minimal

Exécuter dans l'ordre et noter le statut HTTP :

| Étape | Endpoint | Attendu |
| ----- | -------- | ------- |
| 1 | `GET /api/v1/auth/csrf-token` | 200 |
| 2 | `POST /api/v1/app/version-check` | 200, `status: OK` |
| 3 | `POST /api/v1/auth/login` | 200 |
| 4 | `GET /api/v1/auth/bootstrap` | 200 |

**Classification :**

- Si étapes 1–2 OK mais **pas de login** → crash startup client (avant flux auth).
- Si login 401 → vérifier identifiants / mauvaise app.
- Si login + bootstrap OK → appareil prêt pour tests realtime.

### 20–30 min — Ticket incident consolidé

Créer un ticket unique avec :

```text
Incident: iOS startup crash
Build: <49|57|autre>
Device/OS: <modèle + iOS>
Heure test: <UTC+2>
Écran bloqué: <splash infini | login | erreur | crash>
csrf-token: <200|absent>
version-check: <200|absent>
login: <200|absent|401>
bootstrap: <200|absent>
Verdict: <startup_crash | auth_ok | autre>
```

## Gate minimale avant reprise Test #1 realtime

**Ne pas relancer** recovery / reconnect / canary / D3 tant que :

- [ ] `csrf-token` → 200
- [ ] `version-check` → 200
- [ ] `login` → 200
- [ ] `bootstrap` → 200
- [ ] **Aucun crash startup** observé sur la session de test

## Gouvernance de périmètre

- Ne **pas mélanger** l'incident startup iOS avec les tests realtime D3.
- Clore le sujet startup (gate auth OK) **avant** reprise des tests realtime.
- Kafka et PR D restent des chantiers séparés.

## Kill-switch backend (ops)

Variable d'environnement :

```bash
IOS_STARTUP_FATAL_RECOVERY_DISABLED=true
```

Vérification runtime :

```bash
curl -s https://api.lirie.ch/api/feature-flags/runtime-status | jq
```

**Important :** ce switch ne modifie **pas** le comportement des builds anciens qui n'implémentent pas la lecture du flag. Il protège les builds futurs après hotfix mobile.

## Métriques observabilité (mobile, post-hotfix)

| Métrique | Sens |
| -------- | ---- |
| `startup_recovery_fallback` | Recovery réussi via fallback |
| `fatal_startup_blocked_by_switch` | Crash fatal évité par kill-switch |
| `startup_recovery_unrecoverable` | Échec recovery → écran d'erreur contrôlé (sans abort) |
