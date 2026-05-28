# Ticket hotfix mobile — iOS startup recovery

## Problème

Le build 49 (et potentiellement d'autres builds) peut déclencher un **SIGABRT volontaire** au démarrage via :

```text
ErrorRecovery.notify(newRemoteLoadStatus:)
  → ErrorRecovery.runNextTask()
  → ErrorRecovery.crash()
  → StartupProcedure.throwException(_:)
  → NSException.raise
```

**Impact :** crash avant `login`/`bootstrap`, alors que le backend répond correctement (`csrf-token`, `version-check` en 200).

**Preuve :** crash TestFlight build 57 (1.0.5) — incident `2811E668-8452-499B-BA22-69642250CD61`, thread 22, `SIGABRT`.

## Correctif requis (Release/TestFlight)

### Règle absolue

```text
Aucun NSException.raise / abort process pour un état recovery récupérable.
```

Même si le recovery échoue : **écran contrôlé**, pas de kill process.

### Implémentation

1. **Remplacer `ErrorRecovery.crash()`** pour les états récupérables :
   - Reset état local recovery (Keychain / UserDefaults / cache OTA selon stack Expo).
   - Reprise flux auth (écran login → bootstrap).

2. **Logger la cause métier** avant tout fallback :
   - Code erreur / enum recovery failure
   - `remoteLoadStatus` reçu
   - Build number + version app
   - Envoyer à Sentry avec tag `startup_recovery`.

3. **Garde-fou build ancien + état invalide** :
   - Si état recovery local incohérent → purge + reprise auth (pas d'abort).

4. **Garde-fou unrecoverable** :
   - Écran d'erreur contrôlé avec `incident_id`
   - Bouton « Réessayer » / « Se reconnecter »
   - Incrémenter métrique `startup_recovery_unrecoverable`.

5. **Consommer le kill-switch backend** (builds futurs) :
   - Lire `ios_startup_fatal_recovery_disabled` depuis :
     - `POST /api/v1/app/version-check` → `startup_runtime.ios_startup_fatal_recovery_disabled`
     - `GET /api/v1/auth/bootstrap` → `feature_flags.ios_startup_fatal_recovery_disabled`
   - Si `true` : ne jamais appeler `throwException`/`crash()` — fallback auth à la place.
   - Incrémenter `fatal_startup_blocked_by_switch` quand le switch évite un abort.

6. **Métriques Sentry/analytics** :
   - `startup_recovery_fallback`
   - `fatal_startup_blocked_by_switch`
   - `startup_recovery_unrecoverable`

## Fichiers cibles (repo mobile natif / Expo)

> Le code Swift (`StartupProcedure.swift`, `ErrorRecovery.swift`) n'est pas dans le monorepo ATMR actuel. Localiser dans le projet Expo/EAS iOS natif.

- `StartupProcedure.swift` — supprimer `throwException` en Release
- `ErrorRecovery.swift` — remplacer `crash()` par fallback
- Client bootstrap/version-check — lire `ios_startup_fatal_recovery_disabled`

## Référence backend (monorepo ATMR)

Flag ops : `IOS_STARTUP_FATAL_RECOVERY_DISABLED=true`

Exposé via :

- [backend/services/infrastructure/runtime_flags.py](../../backend/services/infrastructure/runtime_flags.py)
- Bootstrap `feature_flags.ios_startup_fatal_recovery_disabled`
- Version-check `startup_runtime.ios_startup_fatal_recovery_disabled`
- Ops : `GET /api/feature-flags/runtime-status`

## Validation

- [ ] Build 57+ démarre après clean install (aucun crash startup)
- [ ] `csrf-token` + `version-check` + `login` + `bootstrap` → tous 200
- [ ] État recovery corrompu simulé → fallback auth (pas de crash)
- [ ] Kill-switch `true` + hotfix déployé → pas d'abort, métrique `fatal_startup_blocked_by_switch`
- [ ] Recovery impossible → écran contrôlé + `startup_recovery_unrecoverable`

## Priorité

**P0** — bloque les tests realtime et fausse les diagnostics infra.

## Non-objectifs

- Ne pas traiter Kafka / PR D dans ce ticket.
- Le kill-switch backend seul ne suffit pas sans hotfix mobile.
